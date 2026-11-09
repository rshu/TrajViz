"""Read native Claude session trees and ZIP bundles without extracting files."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .common import decode_session_text
from .parse import _normalize_payload, _parse_trajectory_text
from .sniff import _looks_like_claude_code_events

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_CHILDREN = 256
_MAX_DEPTH = 16


@dataclass
class ClaudeSessionFiles:
    """Parsed parent and children, with hashes of the exact child bytes read."""

    events: list
    children: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    child_hashes: dict[str, str] = field(default_factory=dict)
    parent_member: str = ""


def resolve_session_path(file_path: str) -> tuple[str, str | None]:
    """Resolve a DSH directory or an unambiguous native Claude session directory."""
    directory = Path(file_path)
    if not directory.is_dir():
        return file_path, None
    from .dsh_native import resolve_native_dsh_path

    dsh = resolve_native_dsh_path(file_path)
    if dsh is not None:
        return dsh, None
    # The directory beside <id>.jsonl contains that session's subagents.
    sibling = directory.parent / (directory.name + ".jsonl")
    if sibling.is_file() and not directory.is_symlink():
        return str(sibling), None
    candidates = sorted(p for p in directory.glob("*.jsonl")
                        if p.is_file() and not p.is_symlink() and not p.name.startswith("agent-"))
    if len(candidates) == 1:
        return str(candidates[0]), None
    if len(candidates) > 1:
        return file_path, "Directory contains multiple sessions; select the parent .jsonl file."
    return file_path, f"Directory has no session.jsonl or Claude Code parent .jsonl: {file_path}"


def _session_id(events: list) -> str:
    for event in events:
        if isinstance(event, dict):
            value = event.get("sessionId")
            if isinstance(value, str) and value:
                return value
    return ""


def _events_from_bytes(data: bytes, label: str) -> tuple[list, str | None]:
    try:
        text = decode_session_text(data)
        payload, error = _parse_trajectory_text(text, label)
        if error:
            return [], error
        payload = _normalize_payload(payload, label)
        if not isinstance(payload, list) or not _looks_like_claude_code_events(payload):
            return [], "not a native Claude Code session"
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as exc:
        return [], type(exc).__name__
    return payload, None


def _agent_id(path: PurePosixPath | Path) -> str:
    return path.stem.removeprefix("agent-")


def _parent_agent(path: PurePosixPath | Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("agent-"):
            return parent.name.removeprefix("agent-")
    return ""


def _add_child(bundle: ClaudeSessionFiles, label: str, data: bytes, *, parent_agent: str = "",
               require_session: bool = False, source_path: str = "") -> None:
    events, error = _events_from_bytes(data, label)
    if error:
        bundle.warnings.append(f"Sub-agent log {label} could not be read ({error}); skipped.")
        return
    root_id = _session_id(bundle.events)
    child_session = _session_id(events)
    if (require_session and (not root_id or child_session != root_id)) or (
        root_id and child_session and child_session != root_id
    ):
        # Old versions placed every session's children in the project folder.
        # A different session is not a missing/damaged child of this one.
        return
    aid = _agent_id(PurePosixPath(label))
    if aid in bundle.children:
        bundle.warnings.append(f"Duplicate sub-agent log for {aid}; only the first copy was loaded.")
        return
    bundle.children[aid] = {"events": events, "source_path": source_path,
                            "parent_agent_id": parent_agent}
    # The topology affects attribution/depth even when the file bytes do not.
    # Exclude the relocatable outer folder, but include the owning agent.
    bundle.child_hashes[f"{parent_agent}/{aid}"] = hashlib.sha256(data).hexdigest()


def discover_claude_children(events: list, source_path: str) -> ClaudeSessionFiles:
    """Read only the loaded session's modern tree or session-matched old siblings.

    No lookup into a user's global Claude storage is performed. Symlinked
    children/directories are excluded, and discovery never follows paths from
    transcript contents outside the selected file's directory.
    """
    bundle = ClaudeSessionFiles(events)
    source = Path(source_path).resolve()
    base = source.parent
    roots = {source.with_suffix("") / "subagents"}
    sid = _session_id(events)
    standalone_child = source.name.startswith("agent-")
    if not standalone_child and _SAFE_ID.fullmatch(sid):
        roots.add(base / sid / "subagents")
    paths: dict[Path, bool] = {}
    for root in sorted(roots):
        # Check every component, rather than only the final directory.
        if not root.is_relative_to(base) or any(p.is_symlink() for p in (root, *root.parents) if p != base):
            continue
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            directory = Path(current)
            dirs[:] = sorted(d for d in dirs if not (directory / d).is_symlink())
            if len(directory.relative_to(root).parts) >= _MAX_DEPTH:
                dirs[:] = []
            for name in sorted(files):
                path = directory / name
                if name.startswith("agent-") and name.endswith(".jsonl") and not path.is_symlink():
                    paths[path] = False
    if not standalone_child and sid:
        for path in sorted(base.glob("agent-*.jsonl")):
            if not path.is_symlink() and path.is_file():
                paths.setdefault(path, True)
    total = 0
    loaded = 0
    for path, legacy in sorted(paths.items(), key=lambda item: (item[1], str(item[0]))):
        label = path.relative_to(base).as_posix()
        try:
            size = path.stat().st_size
            if size > _MAX_MEMBER_BYTES or total + size > _MAX_TOTAL_BYTES:
                bundle.warnings.append(f"Sub-agent log {label} exceeds the session size limit; skipped.")
                continue
            with path.open("rb") as handle:
                data = handle.read(_MAX_MEMBER_BYTES + 1)
            if len(data) > _MAX_MEMBER_BYTES:
                bundle.warnings.append(f"Sub-agent log {label} exceeds the session size limit; skipped.")
                continue
        except OSError as exc:
            bundle.warnings.append(f"Sub-agent log {label} could not be read ({type(exc).__name__}); skipped.")
            continue
        total += len(data)
        before = len(bundle.children)
        _add_child(bundle, label, data, parent_agent=_parent_agent(path.relative_to(base)),
                   require_session=legacy, source_path=str(path))
        loaded += len(bundle.children) - before
        if loaded >= _MAX_CHILDREN:
            bundle.warnings.append(f"Only the first {_MAX_CHILDREN} sub-agent logs were loaded.")
            break
    return bundle


def claude_tree_sha256(parent_sha: str, child_hashes: dict[str, str]) -> str:
    """Relocation-independent identity covering the exact bytes of all merged logs."""
    digest = hashlib.sha256(b"trajviz-claude-code-tree-v1\n")
    digest.update(b"parent\x00" + parent_sha.encode() + b"\n")
    for aid, sha in sorted(child_hashes.items()):
        digest.update(aid.encode() + b"\x00" + sha.encode() + b"\n")
    return digest.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath | None:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or ":" in name or name.startswith("/")
            or stat.S_ISLNK(info.external_attr >> 16)):
        return None
    return path


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: list[int]) -> bytes:
    if info.file_size > _MAX_MEMBER_BYTES or budget[0] + info.file_size > _MAX_TOTAL_BYTES:
        raise ValueError("session size limit exceeded")
    with archive.open(info) as handle:
        data = handle.read(_MAX_MEMBER_BYTES + 1)
    if len(data) > _MAX_MEMBER_BYTES:
        raise ValueError("session size limit exceeded")
    budget[0] += len(data)
    return data


def parse_claude_zip(file_path: str, *, data: bytes | None = None) -> ClaudeSessionFiles | dict | None:
    """Parse one native Claude session and children; None leaves DSH ZIPs to DSH."""
    try:
        with zipfile.ZipFile(io.BytesIO(data) if data is not None else file_path) as archive:
            members: dict[PurePosixPath, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                path = _safe_member(info)
                if path is None or info.is_dir() or path.suffix != ".jsonl" or "__MACOSX" in path.parts:
                    continue
                if path in members:
                    return {"_error": f"Duplicate JSONL member in zip: {path}"}
                members[path] = info
            parents = [p for p in members if "subagents" not in p.parts and not p.name.startswith("agent-")]
            if not parents:
                parents = [p for p in members if "subagents" not in p.parts]
            if not parents:
                return None
            budget = [0]
            native_parents = []
            for candidate in sorted(parents, key=str):
                parent_data = _read_member(archive, members[candidate], budget)
                events, error = _events_from_bytes(parent_data, str(candidate))
                if not error:
                    native_parents.append((candidate, events))
                if len(native_parents) > 1:
                    return {"_error": "Zip contains multiple parent sessions; bundle one session at a time."}
            if not native_parents:
                return None
            parent, events = native_parents[0]
            bundle = ClaudeSessionFiles(events, parent_member=str(parent))
            sid = _session_id(events)
            roots = {parent.parent / parent.stem / "subagents"}
            if not parent.name.startswith("agent-") and _SAFE_ID.fullmatch(sid):
                roots.add(parent.parent / sid / "subagents")
            for path, info in sorted(members.items(), key=lambda item: str(item[0])):
                if path == parent or not path.name.startswith("agent-"):
                    continue
                modern = any(path.is_relative_to(root) for root in roots)
                legacy = path.parent == parent.parent and not parent.name.startswith("agent-")
                if not modern and not legacy:
                    continue
                if len(bundle.children) >= _MAX_CHILDREN:
                    bundle.warnings.append(f"Only the first {_MAX_CHILDREN} sub-agent logs were loaded.")
                    break
                try:
                    data = _read_member(archive, info, budget)
                    _add_child(bundle, str(path), data,
                               parent_agent=_parent_agent(path.relative_to(parent.parent)),
                               require_session=legacy, source_path=f"{file_path}!/{path}")
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
                    bundle.warnings.append(f"Sub-agent log {path} could not be read ({exc}); skipped.")
            return bundle
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
        return {"_error": f"Could not read session zip: {exc}"}
