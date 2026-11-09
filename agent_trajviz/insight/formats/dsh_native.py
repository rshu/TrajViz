"""Bounded, read-only access to released DeepSeek native session generations."""

from __future__ import annotations

import hashlib
import importlib
import io
from pathlib import Path
import re
from typing import Any

from ...errors import LoadError
from .common import decode_session_text
from .parse import _normalize_payload, _parse_trajectory_text

_GENERATION = re.compile(r"session(?:\.v([1-9][0-9]*))?\.jsonl(?:\.zstd)?\Z")
MAX_VERSION = 4
MAX_NATIVE_BYTES = 64 * 1024 * 1024
MAX_TREE_BYTES = 256 * 1024 * 1024
MAX_CHILDREN = 256
MAX_DEPTH = 16
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def looks_like_native_dsh_source(path: str, data: bytes) -> bool:
    match = _GENERATION.fullmatch(Path(path).name)
    return path.endswith(".zstd") or data.startswith(_ZSTD_MAGIC) or bool(match and match[1])


def resolve_native_dsh_path(path: str) -> str | None:
    """Select the highest canonical generation, without migrating or rewriting it."""
    directory = Path(path)
    if not directory.is_dir():
        return None
    if directory.is_symlink():
        raise LoadError("A DeepSeek session directory cannot be a symlink", code="unsafe_path", path=path)
    candidates: list[tuple[int, Path]] = []
    for candidate in directory.iterdir():
        match = _GENERATION.fullmatch(candidate.name)
        if match and candidate.is_file() and not candidate.is_symlink():
            candidates.append((int(match[1] or 0), candidate))
    if not candidates:
        return None
    newest = max(version for version, _ in candidates)
    if newest > MAX_VERSION:
        raise LoadError(f"DeepSeek session format v{newest} is newer than the supported v{MAX_VERSION}",
                        code="unsupported_version", path=path)
    selected = [candidate for version, candidate in candidates if version == newest]
    if len(selected) != 1:
        raise LoadError("DeepSeek session has ambiguous compressed and uncompressed generations; select a file",
                        code="ambiguous_session", path=path)
    return str(selected[0])


def decode_dsh_native_bytes(data: bytes, path: str) -> bytes:
    """Decode concatenated Zstandard frames with a strict expanded-byte budget."""
    if len(data) > MAX_NATIVE_BYTES:
        raise LoadError("DeepSeek session exceeds the source size limit", code="size_limit", path=path)
    match = _GENERATION.fullmatch(Path(path).name)
    if match and int(match[1] or 0) > MAX_VERSION:
        raise LoadError("DeepSeek session generation is newer than this reader supports",
                        code="unsupported_version", path=path)
    compressed = path.endswith(".zstd") or data.startswith(_ZSTD_MAGIC)
    if not compressed:
        return data
    if not data.startswith(_ZSTD_MAGIC):
        raise LoadError("DeepSeek .zstd session has no valid Zstandard frame header", code="invalid_compression", path=path)
    try:
        try:
            zstd = importlib.import_module("compression.zstd")
        except ImportError:
            try:
                zstd = importlib.import_module("zstandard")
            except ImportError:
                raise LoadError(
                    "Compressed DeepSeek sessions require Python 3.14+ or the optional compression extra: "
                    "pip install 'agent-trajviz[zstd]'", code="missing_extra", path=path,
                ) from None
            stream = zstd.ZstdDecompressor(max_window_size=max(1024, MAX_NATIVE_BYTES)).stream_reader(
                io.BytesIO(data), read_across_frames=True,
            )
        else:
            stream = zstd.ZstdFile(io.BytesIO(data), "rb")
        with stream:
            decoded = stream.read(MAX_NATIVE_BYTES + 1)
        if len(decoded) > MAX_NATIVE_BYTES:
            raise LoadError("DeepSeek session exceeds the expanded size limit", code="size_limit", path=path)
        return decoded
    except LoadError:
        raise
    except Exception as error:
        raise LoadError(f"Could not decode DeepSeek Zstandard session ({type(error).__name__})",
                        code="invalid_compression", path=path) from error


def native_events(data: bytes, path: str) -> list[dict[str, Any]]:
    decoded = decode_dsh_native_bytes(data, path)
    try:
        payload, error = _parse_trajectory_text(decode_session_text(decoded), path.removesuffix(".zstd"))
    except UnicodeDecodeError:
        raise LoadError("DeepSeek session is not UTF-8 or BOM-marked UTF-16 JSONL",
                        code="invalid_json", path=path) from None
    if error:
        raise LoadError(error, code="invalid_json", path=path)
    events = _normalize_payload(payload, path.removesuffix(".zstd"))
    if not isinstance(events, list) or not events or not isinstance(events[0], dict) or events[0].get("type") != "session":
        raise LoadError("DeepSeek native session has no session header", code="unknown_format", path=path)
    header = events[0]
    version = header.get("version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version > MAX_VERSION or version < 0:
        raise LoadError("Unsupported DeepSeek session header version", code="unsupported_version", path=path)
    match = _GENERATION.fullmatch(Path(path).name)
    if match and int(match[1] or 0) != version:
        raise LoadError("DeepSeek session filename and header versions disagree", code="invalid_json", path=path)
    return events


def _encode_segment(value: str) -> str:
    if value in {".", ".."}:
        return "~002E" * len(value)
    encoded = []
    units = value.encode("utf-16-be", errors="surrogatepass")
    for index in range(0, len(units), 2):
        code = int.from_bytes(units[index:index + 2], "big")
        character = chr(code)
        encoded.append(character if character.isascii() and (character.isalnum() or character in "._-") else f"~{code:04X}")
    return "".join(encoded)


def discover_native_dsh_children(
    parent_events: list[dict], source_path: str,
) -> tuple[list[tuple[str, list[dict]]], dict[str, str], list[str]]:
    """Follow explicit v4 child catalog identities in the same project only.

    This never scans other sessions' contents to guess lineage. Catalog ids
    select exact sibling directories; each decoded header must confirm its
    parent before its own descendants are considered.
    """
    header = parent_events[0] if parent_events else {}
    if header.get("version", 0) < 4 or not _GENERATION.fullmatch(Path(source_path).name):
        return [], {}, []
    project = Path(source_path).absolute().parent.parent
    result: list[tuple[str, list[dict]]] = []
    hashes: dict[str, str] = {}
    warnings: list[str] = []
    seen = {header.get("id")}
    total_bytes = 0
    expanded_bytes = 0
    pending = [(parent_events, header.get("id"), 0)]
    while pending:
        events, parent_id, depth = pending.pop(0)
        for event in events:
            if event.get("type") != "subagent/catalog":
                continue
            data = event.get("data")
            child_id = data.get("childId") if isinstance(data, dict) else None
            if not isinstance(child_id, str) or not child_id or child_id in seen:
                continue
            seen.add(child_id)
            if depth >= MAX_DEPTH or len(seen) > MAX_CHILDREN + 1:
                warnings.append("DeepSeek child discovery limit reached; additional descendants were omitted")
                continue
            try:
                if len(child_id) > 255:
                    raise LoadError("Child identity exceeds the path limit", code="unsafe_path")
                directory = project / _encode_segment(child_id)
                if directory.is_symlink() or directory.resolve().parent != project.resolve():
                    raise LoadError("Unsafe child session path", code="unsafe_path")
                selected = resolve_native_dsh_path(str(directory))
                if selected is None:
                    raise LoadError("Referenced child session is missing", code="missing_file")
                path = Path(selected)
                if path.stat().st_size > MAX_NATIVE_BYTES:
                    raise LoadError("Child source size limit exceeded", code="size_limit")
                data = path.read_bytes()
                total_bytes += len(data)
                if total_bytes > MAX_TREE_BYTES:
                    raise LoadError("Child tree size limit exceeded", code="size_limit")
                decoded = decode_dsh_native_bytes(data, str(path))
                expanded_bytes += len(decoded)
                if expanded_bytes > MAX_TREE_BYTES:
                    raise LoadError("Expanded child tree size limit exceeded", code="size_limit")
                child_events = native_events(decoded, str(path).removesuffix(".zstd"))
                child_header = child_events[0]
                if child_header.get("id") != child_id or child_header.get("parentSession") != parent_id:
                    raise LoadError("Child session header does not match catalog lineage", code="invalid_lineage")
                result.append((str(path), child_events))
                hashes[child_id] = hashlib.sha256(data).hexdigest()
                pending.append((child_events, child_id, depth + 1))
            except (LoadError, OSError) as error:
                warnings.append(f"DeepSeek child {child_id!r} was omitted: {error}")
    return result, hashes, warnings
