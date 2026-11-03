"""Read selected saved OTLP files and explicit bundles, without discovery."""

from __future__ import annotations

import hashlib
import codecs
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ...errors import LoadError
from .otlp import MAX_RECORDS, OtlpCapture, decode_otlp_bytes, deduplicate_capture, looks_like_otlp_payload

MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 256
MAX_PATH_DEPTH = 16
_MANIFEST_NAMES = ("manifest.json", "otlp-manifest.json")
_BUNDLE_TYPE = "agent-trajviz-otlp-bundle"


def _error(message: str, path: str, code: str = "invalid_otlp_bundle") -> LoadError:
    return LoadError(message, code=code, path=path)


def _relative_name(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise _error(f"Unsafe OTLP bundle path: {value!r}", label, "unsafe_bundle")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise _error("OTLP bundle path contains invalid Unicode", label, "unsafe_bundle") from exc
    name = PurePosixPath(value)
    if name.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise _error(f"Unsafe OTLP bundle path: {value!r}", label, "unsafe_bundle")
    if len(name.parts) > MAX_PATH_DEPTH:
        raise _error("OTLP bundle path depth limit exceeded", label, "size_limit")
    return name


def _manifest(data: bytes, label: str) -> dict:
    if len(data) > 1024 * 1024:
        raise _error("OTLP manifest exceeds the size limit", label, "size_limit")
    try:
        def unique(pairs: list[tuple[str, Any]]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result
        manifest = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise _error("OTLP bundle manifest must be valid UTF-8 JSON", label) from exc
    if not isinstance(manifest, dict) or manifest.get("type") != _BUNDLE_TYPE:
        raise _error(f"Expected an explicit {_BUNDLE_TYPE} manifest", label)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise _error("Unsupported OTLP manifest schema_version; expected 1", label, "unsupported_schema")
    if manifest.get("profile") != "claude_code":
        raise _error("Unsupported OTLP manifest profile; expected claude_code", label, "unsupported_otlp")
    seen = set()
    for key in ("traces", "logs"):
        values = manifest.get(key, [] if key == "logs" else None)
        if not isinstance(values, list) or len(values) > MAX_MEMBERS:
            raise _error(f"OTLP manifest {key} must be a bounded array of relative file names", label)
        manifest[key] = values
        for value in values:
            name = _relative_name(value, label)
            if name in seen:
                raise _error(f"Duplicate OTLP manifest member: {name}", label)
            seen.add(name)
    if not manifest["traces"]:
        raise _error("OTLP bundles must explicitly list at least one trace file", label)
    if len(seen) > MAX_MEMBERS:
        raise _error("OTLP bundle member count limit exceeded", label, "size_limit")
    return manifest


def _consume_file(handle: Any, path: Path, budget: list[int]) -> bytes:
    opened = os.fstat(handle.fileno())
    if not stat.S_ISREG(opened.st_mode):
        raise _error(f"OTLP source is not a regular file: {path}", str(path), "load_failed")
    if opened.st_size > MAX_MEMBER_BYTES or budget[0] + opened.st_size > MAX_TOTAL_BYTES:
        raise _error("OTLP capture size limit exceeded", str(path), "size_limit")
    data = handle.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES or budget[0] + len(data) > MAX_TOTAL_BYTES:
        raise _error("OTLP capture size limit exceeded", str(path), "size_limit")
    budget[0] += len(data)
    return data


def _open_file(path: Path) -> Any:
    try:
        if path.is_symlink():
            raise _error(f"Symlinked OTLP source is not allowed: {path}", str(path), "unsafe_bundle")
        # O_NOFOLLOW closes the check/open race for the selected file on
        # platforms that provide it. Compare the opened inode as a fallback.
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise _error(f"OTLP source is not a regular file: {path}", str(path), "load_failed")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise _error("OTLP source changed while opening it", str(path), "unsafe_bundle")
            if not stat.S_ISREG(opened.st_mode):
                raise _error(f"OTLP source is not a regular file: {path}", str(path), "load_failed")
        except BaseException:
            os.close(descriptor)
            raise
        return os.fdopen(descriptor, "rb")
    except OSError as exc:
        raise _error(f"Could not read OTLP source {path}: {exc}", str(path), "load_failed") from exc


def _read_file(path: Path, budget: list[int]) -> bytes:
    with _open_file(path) as handle:
        return _consume_file(handle, path, budget)


def _check_local_member(base: Path, name: PurePosixPath, label: str) -> Path:
    path = base.joinpath(*name.parts)
    if any(base.joinpath(*name.parts[:index]).is_symlink() for index in range(1, len(name.parts) + 1)):
        raise _error(f"Symlinked OTLP bundle member is not allowed: {name}", label, "unsafe_bundle")
    if not path.resolve().is_relative_to(base.resolve()):
        raise _error(f"OTLP bundle member escapes the bundle: {name}", label, "unsafe_bundle")
    return path


def _read_local_member(base: Path, name: PurePosixPath, budget: list[int], label: str) -> bytes:
    path = _check_local_member(base, name, label)
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        return _read_file(path, budget)
    # Open every bundle-relative component under pinned directory descriptors.
    # Replacing an ancestor with a symlink after validation cannot escape base.
    descriptors = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current = os.open(base, directory_flags)
        descriptors.append(current)
        for part in name.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        descriptor = os.open(name.name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
                             | getattr(os, "O_BINARY", 0), dir_fd=current)
        with os.fdopen(descriptor, "rb") as handle:
            return _consume_file(handle, path, budget)
    except OSError as exc:
        raise _error(f"Could not safely read OTLP member {name}: {exc}", label, "unsafe_bundle") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _bundle_capture(manifest: dict, manifest_data: bytes, files: list[tuple[str, str, bytes]]) -> OtlpCapture:
    capture = OtlpCapture(raw={"manifest": manifest, "profile": manifest["profile"], "requests": [], "source_files": []})
    digest = hashlib.sha256(b"agent-trajviz-otlp-bundle-v1\n")
    consumed = [("<manifest>", manifest_data)]
    for logical_name, label, data in files:
        member = decode_otlp_bytes(data, label)
        # The manifest specifies each file's signal. Mixing signals into either
        # file is harmless, but the mandatory traces must actually contain spans.
        capture.spans.extend(member.spans)
        capture.logs.extend(member.logs)
        capture.warnings.extend(member.warnings)
        capture.raw["requests"].extend(member.raw["requests"])
        capture.raw["source_files"].append(label)
        consumed.append((logical_name, data))
        if len(capture.spans) + len(capture.logs) > MAX_RECORDS:
            raise _error("OTLP record limit exceeded", label, "size_limit")
    for name, data in sorted(consumed):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    capture.sha256 = digest.hexdigest()
    return deduplicate_capture(capture)


def _directory_capture(manifest_path: Path, initial: bytes | None = None) -> OtlpCapture:
    budget = [0]
    data = initial if initial is not None else _read_file(manifest_path, budget)
    if initial is not None:
        budget[0] += len(initial)
    manifest = _manifest(data, str(manifest_path))
    files = []
    for name in sorted(manifest["traces"] + manifest["logs"]):
        relative = _relative_name(name, str(manifest_path))
        path = _check_local_member(manifest_path.parent, relative, str(manifest_path))
        if path == manifest_path:
            raise _error("An OTLP manifest cannot list itself as telemetry", str(manifest_path))
        files.append((name, str(path), _read_local_member(manifest_path.parent, relative, budget, str(manifest_path))))
    capture = _bundle_capture(manifest, data, files)
    capture.raw["_input_files"] = [str(manifest_path.resolve()), *(str(Path(label).resolve()) for _, label, _ in files)]
    return capture


def _zip_capture(path: Path) -> OtlpCapture:
    try:
        if path.is_symlink():
            raise _error("Symlinked OTLP archives are not allowed", str(path), "unsafe_bundle")
        with _open_file(path) as archive_file:
            if os.fstat(archive_file.fileno()).st_size > MAX_TOTAL_BYTES:
                raise _error("OTLP archive size limit exceeded", str(path), "size_limit")
            return _zip_from_file(path, archive_file)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError, NotImplementedError) as exc:
        raise _error(f"Could not read OTLP ZIP {path}: {exc}", str(path), "load_failed") from exc


def _zip_member_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int, label: str) -> bytes:
    """Normalize failures from ZIP's optional decompression backends."""
    try:
        with archive.open(info) as handle:
            return handle.read(limit)
    except Exception as exc:
        # zlib.error and lzma.LZMAError do not share the OSError/BadZipFile
        # hierarchy. Keep corrupt compressed members within the public error
        # contract rather than exposing a backend exception to the CLI/UI.
        raise _error(f"Could not read OTLP ZIP member {info.filename}: {exc}", label, "load_failed") from exc


def _zip_from_file(path: Path, archive_file: Any) -> OtlpCapture:
    with zipfile.ZipFile(archive_file) as archive:
            if len(archive.infolist()) > MAX_MEMBERS * 2:
                raise _error("OTLP archive member count limit exceeded", str(path), "size_limit")
            members = {}
            for info in archive.infolist():
                name = info.filename.rstrip("/") if info.is_dir() else info.filename
                relative = _relative_name(name, str(path))
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise _error(f"Symlinked OTLP archive member: {name}", str(path), "unsafe_bundle")
                if info.is_dir():
                    continue
                if relative in members:
                    raise _error(f"Duplicate OTLP archive member: {name}", str(path), "unsafe_bundle")
                members[relative] = info
            manifests = [name for name in members if name.name in _MANIFEST_NAMES]
            if len(manifests) != 1:
                raise _error("OTLP ZIP requires exactly one manifest.json or otlp-manifest.json", str(path))
            manifest_name = manifests[0]
            budget = [0]

            def read(name: PurePosixPath) -> bytes:
                if name not in members:
                    raise _error(f"Missing OTLP archive member: {name}", str(path))
                info = members[name]
                if info.file_size > MAX_MEMBER_BYTES or budget[0] + info.file_size > MAX_TOTAL_BYTES:
                    raise _error("OTLP capture size limit exceeded", str(path), "size_limit")
                data = _zip_member_bytes(archive, info, MAX_MEMBER_BYTES + 1, str(path))
                if len(data) > MAX_MEMBER_BYTES or budget[0] + len(data) > MAX_TOTAL_BYTES:
                    raise _error("OTLP capture size limit exceeded", str(path), "size_limit")
                budget[0] += len(data)
                return data

            manifest_data = read(manifest_name)
            manifest = _manifest(manifest_data, f"{path}!/{manifest_name}")
            files = []
            for name in sorted(manifest["traces"] + manifest["logs"]):
                member = manifest_name.parent / _relative_name(name, str(path))
                if member == manifest_name:
                    raise _error("An OTLP manifest cannot list itself as telemetry", str(path))
                files.append((name, f"{path}!/{member}", read(member)))
            return _bundle_capture(manifest, manifest_data, files)


def read_otlp_capture(path: str | Path) -> OtlpCapture:
    """Read a selected file, explicit local manifest, bundle directory or ZIP."""
    source = Path(path)
    if source.is_symlink():
        raise _error("Symlinked OTLP sources are not allowed", str(source), "unsafe_bundle")
    if source.is_dir():
        candidates = [source / name for name in _MANIFEST_NAMES if (source / name).is_file()]
        if len(candidates) != 1:
            raise _error("OTLP directory requires exactly one manifest.json or otlp-manifest.json", str(source))
        return _directory_capture(candidates[0])
    if source.suffix.lower() == ".zip":
        capture = _zip_capture(source)
        capture.raw["_input_files"] = [str(source.resolve())]
        return capture
    data = _read_file(source, [0])
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeError, ValueError, RecursionError):
        payload = None
    if isinstance(payload, dict) and payload.get("type") == _BUNDLE_TYPE:
        return _directory_capture(source, data)
    capture = decode_otlp_bytes(data, str(source))
    capture.raw["_input_files"] = [str(source.resolve())]
    return capture


def _prefix_has_otlp_key(text: str) -> bool:
    """Find only top-level envelope keys in an incomplete, bounded JSON prefix."""
    text = text.lstrip()[:4096]
    if not text.startswith("{"):
        return False
    index, depth, expecting_key = 1, 1, True
    while index < len(text):
        char = text[index]
        if char == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                return False
            if depth == 1 and expecting_key:
                try:
                    key = json.loads(text[start:index])
                except ValueError:
                    return False
                if text[index:].lstrip().startswith(":") and key in (
                    "resourceSpans", "resourceLogs", "resource_spans", "resource_logs",
                ):
                    return True
                expecting_key = False
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth <= 0:
                return False
        elif char == "," and depth == 1:
            expecting_key = True
        index += 1
    return False


def looks_like_otlp_source(path: str | Path) -> bool:
    """Bounded format sniffing only inside the explicitly selected source."""
    source = Path(path)
    try:
        if source.is_symlink():
            return False
        if source.is_dir():
            candidates = [source / name for name in _MANIFEST_NAMES if (source / name).is_file()]
            return any(looks_like_otlp_source(candidate) for candidate in candidates)
        if source.suffix.lower() == ".zip":
            if source.stat().st_size > MAX_TOTAL_BYTES:
                return False
            with _open_file(source) as handle, zipfile.ZipFile(handle) as archive:
                if len(archive.infolist()) > MAX_MEMBERS * 2:
                    return False
                candidates = [info for info in archive.infolist()
                              if PurePosixPath(info.filename).name in _MANIFEST_NAMES and not info.is_dir()]
                if len(candidates) != 1 or candidates[0].file_size > 1024 * 1024:
                    return False
                data = _zip_member_bytes(archive, candidates[0], 1024 * 1024 + 1, str(path))
                return looks_like_otlp_payload(json.loads(data.decode("utf-8-sig")))
        # Sniff a small first request; full validation and bounded reads happen
        # only after the caller selects this format.
        with _open_file(source) as handle:
            data = handle.read(min(MAX_MEMBER_BYTES, 1024 * 1024) + 1)
        # A bounded prefix can stop between UTF-8 continuation bytes. Keep
        # incomplete trailing code points buffered without hiding bad bytes
        # inside the prefix; the full decoder subsequently validates the file.
        text = codecs.getincrementaldecoder("utf-8-sig")().decode(data, final=False)
        try:
            return looks_like_otlp_payload(json.loads(text))
        except ValueError:
            first = next((line for line in re.split(r"\r\n|[\r\n]", text) if line.strip()), "")
            try:
                return looks_like_otlp_payload(json.loads(first))
            except ValueError:
                # A pretty-printed large request can extend beyond the sniff
                # budget. Recognition of a top-level transport key is enough;
                # the decoder subsequently validates the full envelope.
                return _prefix_has_otlp_key(text)
    except (LoadError, OSError, UnicodeError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError, RecursionError):
        return False
