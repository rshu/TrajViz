"""Trajectory format detection and conversion (Claude Code, Cursor, OpenCode, CodeArts, ICode, Codex, Pi, DSH).

Converters live in ``agent_trajviz.insight.formats``. This module is the public
facade: ``detect_format``, ``load_trajectory``, and the names callers already
import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .formats.claude_code import convert_claude_code_events
from .formats.claude_files import (
    ClaudeSessionFiles,
    claude_tree_sha256,
    discover_claude_children,
    parse_claude_zip,
    resolve_session_path,
)
from .formats.codeagent import _convert_codeagent_to_internal
from .formats.codearts import _convert_codearts_metadata
from .formats.codex import _convert_codex_to_internal
from .formats.cursor import _convert_cursor_to_internal
from .formats.icode import _convert_icode_to_internal
from .formats.common import (  # noqa: F401  (re-exported public / test API)
    _classify_tool_error,
    _iso_to_epoch_ms,
    decode_session_text,
    safe_get,
)
from .formats.dsh import (
    DshZipContents,
    _convert_dsh_to_internal,
    _dsh_drop_seed_prefix,  # noqa: F401
    _zip_dsh_members,  # noqa: F401
    parse_dsh_zip,
)
from .formats.opencode import _convert_opencode_metadata
from .formats.parse import _normalize_payload, _parse_trajectory_text, _path_ext
from .formats.pi import _convert_pi_to_internal
from .formats.sniff import (
    _EVENT_FORMATS,
    _FORMAT_STAMPS,
    _OBJECT_FORMATS,
    _detect_event_stream_format,
    _detect_object_format,
)

logger = logging.getLogger(__name__)

# Canonical display names for the detected trajectory formats.  Defined here,
# next to detect_format, as the single source of truth — UI/report modules
# import this mapping instead of maintaining their own drifting copies.
FORMAT_LABELS = {
    "otlp": "OpenTelemetry",
    "claude_code": "Claude Code",
    "cursor": "Cursor",
    "codeagent": "CodeAgent",
    "codearts": "CodeArts",
    "icode": "ICode",
    "opencode": "OpenCode",
    "codex": "Codex CLI",
    "pi": "Pi",
    "dsh": "DeepSeek Harness",
}

# Dropdown choices for the Insight UI. Auto-detect is first so it is the
# default; an explicit format is still available as an override / mismatch gate.
FORMAT_DROPDOWN_CHOICES: list[tuple[str, str]] = [
    ("Auto-detect", ""),
    *((label, key) for key, label in FORMAT_LABELS.items()),
]


def _convert_codex_events(payload: list, *, source_path: str | None = None) -> dict:
    from .formats.codex_files import CodexSessionFiles, discover_codex_children

    bundle = discover_codex_children(payload, source_path) if source_path else CodexSessionFiles(payload)
    result = _convert_codex_to_internal(
        payload, source_path=source_path or "", child_event_lists=bundle.children, child_warnings=bundle.warnings,
    )
    result["_codex_child_hashes"] = bundle.child_hashes
    result["_input_files"] = [source_path, *(child["source_path"] for child in bundle.children.values()
                                            if isinstance(child, dict) and child.get("source_path"))]
    return result


def _convert_cursor_events(payload: list, *, source_path: str | None = None) -> dict:
    from .formats.cursor_stream import convert_cursor_stream

    return convert_cursor_stream(payload, source_path=source_path)


def _convert_dsh_events(payload: list, *, source_path: str | None = None) -> dict:
    return _convert_dsh_to_internal(payload, source_path=source_path)


def _convert_pi_events(payload: list, *, source_path: str | None = None) -> dict:
    del source_path
    return _convert_pi_to_internal(payload)


def _convert_claude_events(payload: list, *, source_path: str | None = None) -> dict:
    bundle = discover_claude_children(payload, source_path) if source_path else ClaudeSessionFiles(payload)
    result = convert_claude_code_events(
        payload, source_path=source_path, child_event_lists=bundle.children, child_warnings=bundle.warnings,
    )
    result["_claude_code_child_hashes"] = bundle.child_hashes
    result["_input_files"] = [source_path, *(child["source_path"] for child in bundle.children.values()
                                            if isinstance(child, dict) and child.get("source_path"))]
    return result


_EVENT_CONVERTERS = {
    "claude_code": _convert_claude_events,
    "cursor": _convert_cursor_events,
    "codex": _convert_codex_events,
    "dsh": _convert_dsh_events,
    "pi": _convert_pi_events,
}
if set(_EVENT_CONVERTERS) != _EVENT_FORMATS:
    raise RuntimeError(
        "Event converters must match sniff._EVENT_FORMATS: "
        f"converters={sorted(_EVENT_CONVERTERS)} sniff={sorted(_EVENT_FORMATS)}"
    )
if set(FORMAT_LABELS) != _OBJECT_FORMATS | _EVENT_FORMATS:
    raise RuntimeError(
        "FORMAT_LABELS must match sniff object+event formats: "
        f"labels={sorted(FORMAT_LABELS)} sniff={sorted(_OBJECT_FORMATS | _EVENT_FORMATS)}"
    )


def detect_format(raw: Any) -> str:
    """Detect trajectory format from parsed content.

    Accepts a JSON object or an event array. Returns ``claude_code``,
    ``opencode``, ``codeagent``, ``codearts``, ``cursor``, ``icode``, ``codex``,
    ``pi``, ``dsh``, or ``unknown``.
    """
    if isinstance(raw, list):
        return _detect_event_stream_format(raw)
    if isinstance(raw, dict):
        resolved = raw.get("_source_format")
        if isinstance(resolved, str) and resolved in FORMAT_LABELS:
            return resolved
        return _detect_object_format(raw)
    return "unknown"


def _resolve_format(detected: str, hint: str | None) -> tuple[str, str | None]:
    """Apply ``format_hint`` to a detected format.

    Returns ``(fmt, reason)`` where ``reason`` is ``None`` (proceed),
    ``"unknown"`` (auto-detect found nothing), or ``"mismatch"``.

    Empty hint is auto-detect. A recognized hint on ``unknown`` *forces*
    that converter. A recognized hint on a different detected format is
    a mismatch (including Codex/Pi).
    """
    selected = hint if hint in FORMAT_LABELS else ""
    if not selected:
        if detected in ("", "unknown"):
            return "unknown", "unknown"
        return detected, None
    if detected in ("", "unknown"):
        return selected, None
    if detected != selected:
        return detected, "mismatch"
    return detected, None


def check_format_selection(detected: str, selected: str | None) -> str | None:
    """Return an error reason if the dropdown selection cannot load this file.

    ``None`` means proceed. Empty ``selected`` is auto-detect: any recognized
    format is accepted, and ``unknown`` is rejected so we do not render an
    empty dashboard. An explicit selection forces conversion when detection
    is ``unknown``, and rejects a different recognized format (including
    Codex/Pi).
    """
    _, reason = _resolve_format(detected, selected or None)
    return reason


_UNSUPPORTED_EVENT_STREAM = (
    "Unsupported event-array input; expected native Claude Code JSONL "
    "(user/assistant messages with sessionId), Codex JSONL "
    "(leading session_meta event), Cursor CLI stream-json, Pi JSONL "
    "(leading session event), or DeepSeek Harness JSONL "
    "(leading session event with createdAt / slash-typed body events)."
)


def _already_converted(raw: dict, fmt: str) -> bool:
    stamp = _FORMAT_STAMPS.get(fmt)
    return bool(stamp and raw.get(stamp) is True)


def _format_mismatch_error(selected: str, detected: str) -> dict:
    return {
        "_error": (
            f"Format mismatch: selected {FORMAT_LABELS.get(selected, selected)} "
            f"but file detected as {FORMAT_LABELS.get(detected, detected)}."
        ),
        "_error_code": "mismatch",
        "_selected": selected,
        "_detected": detected,
    }


def _kind_mismatch_error(fmt: str) -> dict:
    label = FORMAT_LABELS.get(fmt, fmt)
    if fmt in _EVENT_FORMATS:
        shape = "native sessionId message records" if fmt == "claude_code" else (
            "leading session_meta event" if fmt == "codex" else "leading session event")
        return {"_error": f"{label} expects a JSONL event array ({shape})."}
    return {"_error": f"{label} expects a JSON object, not an event array."}


def _apply_format(payload: Any, fmt: str, *, source_path: str | None = None) -> dict:
    """Convert parsed content to the internal step-model dict."""
    if fmt == "unknown":
        if isinstance(payload, dict):
            return payload
        return {"_error": _UNSUPPORTED_EVENT_STREAM, "_error_code": "unknown_format"}

    if isinstance(payload, dict) and _already_converted(payload, fmt):
        return payload

    if fmt == "otlp":
        if source_path is None:
            return {"_error": "Saved OTLP input requires a source file or bundle.", "_error_code": "invalid_otlp"}
        from .formats.otlp_claude import convert_otlp_claude
        from .formats.otlp_files import read_otlp_capture

        result = convert_otlp_claude(read_otlp_capture(source_path))
        _stamp_otlp_identity(result)
        return result

    if fmt in _EVENT_FORMATS and (fmt != "cursor" or isinstance(payload, list)):
        if not isinstance(payload, list):
            return _kind_mismatch_error(fmt)
        convert = _EVENT_CONVERTERS.get(fmt)
        if convert is None:
            return {"_error": f"Unknown event format: {fmt}"}
        return convert(payload, source_path=source_path)

    if not isinstance(payload, dict):
        return _kind_mismatch_error(fmt)
    if fmt == "codeagent":
        return _convert_codeagent_to_internal(payload)
    if fmt == "codearts":
        # Legacy-JSON exports are detected as CodeArts (so the UI labels them
        # correctly) but have no parser yet: their 'sender'/'content' messages
        # are not OpenCode-shaped and would silently produce empty steps.
        if safe_get(payload, "export_metadata", "source_format") == "codearts_legacy_json":
            return {
                "_error": "CodeArts legacy-JSON export detected "
                "(source_format=codearts_legacy_json): this legacy "
                "message schema is not yet supported. Re-export the "
                "session from the CodeArts SQLite database instead."
            }
        return _convert_codearts_metadata(payload)
    if fmt == "icode":
        return _convert_icode_to_internal(payload)
    if fmt == "cursor":
        return _convert_cursor_to_internal(payload)
    if fmt == "opencode":
        return _convert_opencode_metadata(payload)
    return {"_error": f"Unknown trajectory format: {fmt}"}


def _load_zip_trajectory(file_path: str, format_hint: str | None, source_sha: str,
                         source_data: bytes | None = None) -> dict:
    """Load a native Claude Code or DSH session bundle."""
    # Preserve DSH's established selection rules: unrelated debug JSONL files
    # are not additional parent sessions in a DSH export.
    parsed = parse_dsh_zip(file_path, data=source_data)
    is_dsh = isinstance(parsed, DshZipContents) and detect_format(parsed.parent_events) == "dsh"
    claude = None if is_dsh else parse_claude_zip(file_path, data=source_data)
    if isinstance(claude, dict):
        return claude
    if isinstance(claude, ClaudeSessionFiles):
        if format_hint in FORMAT_LABELS and format_hint != "claude_code":
            return _format_mismatch_error(format_hint, "claude_code")
        result = convert_claude_code_events(
            claude.events, source_path=f"{file_path}!/{claude.parent_member}",
            child_event_lists=claude.children, child_warnings=claude.warnings,
        )
        result["_source_path"] = file_path
        result["_input_files"] = [str(Path(file_path).resolve())]
        result["_source_sha256"] = source_sha
        return result
    if not isinstance(parsed, DshZipContents):
        return parsed
    events = parsed.parent_events
    detected = detect_format(events)
    hint = format_hint if format_hint in FORMAT_LABELS else None
    fmt, reason = _resolve_format(detected, hint)
    if reason == "mismatch":
        return _format_mismatch_error(hint or "", detected)
    if fmt != "dsh":
        result = _apply_format(events, fmt, source_path=file_path)
    else:
        result = _convert_dsh_to_internal(
            events,
            source_path=file_path,
            child_event_lists=parsed.child_event_lists,
            child_warnings=list(parsed.child_warnings),
        )
    # Zip members carry no path, so this is always empty here; drop it so the
    # internal-only key never reaches the raw dict. The archive's own bytes
    # already cover every merged member.
    result.pop("_dsh_merged_sources", None)
    if "_error" in result:
        return result
    result["_source_path"] = file_path
    result["_input_files"] = [str(Path(file_path).resolve())]
    result["_source_sha256"] = source_sha
    return result


def tokens_are_estimates(raw: dict | None) -> bool:
    """True when the export recorded no runtime token usage, so token counts are
    the converter's estimates (Cursor derives them from logged text)."""
    caps = raw.get("_capabilities") if isinstance(raw, dict) else None
    return isinstance(caps, dict) and caps.get("has_runtime_token_usage") is False


def read_payload(file_path: str) -> tuple[Any, str | None, str | None]:
    """``(parsed JSON, sha256 of the bytes, error)`` for a file, without detection or conversion.

    The parsed JSON is the document, or for JSON Lines the list of records:
    what a format registered by an application (``agent_trajviz.Format``)
    parses. On failure the first element is ``None`` and the error is set.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        return None, None, str(exc)
    sha = _sha256_bytes(data)
    try:
        text = decode_session_text(data)
    except UnicodeDecodeError as exc:
        return None, sha, str(exc)
    payload, error = _parse_trajectory_text(text, file_path)
    return payload, sha, error


def load_trajectory(file_path: str, format_hint: str | None = None) -> dict:
    """Load a trajectory file; for any input, a parsed trajectory or ``{"_error": ...}``.

    The never-raise promise is enforced HERE, at the boundary, rather than at
    each site that might fail: converters and normalisers copy values from the
    export as found, and a value of an unexpected type, or JSON nested deeply
    enough to exhaust the recursion limit, used to raise out of whichever line
    touched it first. The traceback goes to the log; the caller
    gets a message. See :func:`_load_trajectory_unguarded` for the dispatch.
    """
    from ..errors import LoadError

    try:
        return _load_trajectory_unguarded(file_path, format_hint)
    except LoadError as exc:
        return {"_error": str(exc), "_error_code": exc.code or "load_failed", "_detected": exc.detected}
    except RecursionError:
        return {"_error": "This file nests JSON too deeply to be read safely."}
    except Exception as exc:  # noqa: BLE001 — the contract is "never raise"
        logger.exception("could not load %s", file_path)
        detail = f"{type(exc).__name__}: {exc}"
        return {"_error": f"Could not read this file as a trajectory ({detail[:200]})."}


def _load_trajectory_unguarded(file_path: str, format_hint: str | None = None) -> dict:
    """Load a trajectory file via the content dispatcher.

    Reads the file once (sha256 of those exact bytes), sniffs JSON object vs
    event array vs JSONL, detects format, then converts. ``format_hint``
    forces a converter when detection is ``unknown``
    and *requires* that format when detection already succeeded.

    Native Claude Code and DeepSeek Harness sessions may include sub-agent
    trees beside the parent file, in a session directory, or in a ZIP bundle.
    """
    from .formats.otlp_files import looks_like_otlp_source

    otlp_source = looks_like_otlp_source(file_path)
    if format_hint == "otlp" and not otlp_source:
        detected = _bounded_non_otlp_format(file_path)
        if detected in FORMAT_LABELS and detected != "otlp":
            return _format_mismatch_error("otlp", detected)
    if format_hint == "otlp" or otlp_source:
        if format_hint in FORMAT_LABELS and format_hint != "otlp":
            return _format_mismatch_error(format_hint, "otlp")
        result = _apply_format(None, "otlp", source_path=file_path)
        result["_source_path"] = file_path
        return result

    if _path_ext(file_path) == ".log":
        return {"_error": "Unsupported file type: .log files are no longer supported."}

    file_path, path_error = resolve_session_path(file_path)
    if path_error:
        return {"_error": path_error}

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        source_sha = _sha256_bytes(data)
    except OSError as exc:
        return {"_error": str(exc)}

    ext = _path_ext(file_path)
    if ext == ".zip" or (data.startswith(b"PK") and ext not in {".json", ".jsonl"}):
        return _load_zip_trajectory(file_path, format_hint, source_sha, data)

    from .formats.dsh_native import looks_like_native_dsh_source, native_events

    if looks_like_native_dsh_source(file_path, data):
        payload, parse_error = native_events(data, file_path), None
    else:
        try:
            text = decode_session_text(data)
        except UnicodeDecodeError as exc:
            return {"_error": str(exc)}
        payload, parse_error = _parse_trajectory_text(text, file_path)
    if parse_error:
        return {"_error": parse_error}

    payload = _normalize_payload(payload, file_path)
    detected = detect_format(payload)
    hint = format_hint if format_hint in FORMAT_LABELS else None
    fmt, reason = _resolve_format(detected, hint)
    if reason == "mismatch":
        return _format_mismatch_error(hint or "", detected)

    result = _apply_format(payload, fmt, source_path=file_path)
    merged_sources = result.pop("_dsh_merged_sources", None)
    dsh_child_hashes = result.pop("_dsh_child_hashes", None)
    claude_children = result.pop("_claude_code_child_hashes", None)
    codex_children = result.pop("_codex_child_hashes", None)
    if "_error" in result:
        return result
    # The resolved selection is authoritative even when the input lacked the
    # normal identifying markers and the caller selected its converter.
    result["_source_format"] = fmt
    result["_source_path"] = file_path
    if not (isinstance(payload, list) and fmt in {"claude_code", "codex"}) and fmt != "otlp":
        result["_input_files"] = [str(Path(file_path).resolve())]
    if merged_sources:
        result["_input_files"].extend(path for _, path in merged_sources)
    result["_input_files"] = [str(Path(path).resolve()) for path in result.get("_input_files", []) if path]
    if fmt == "otlp":
        # The OTLP reader hashes the exact bytes it consumed and stamps the
        # selected session/profile identity; retain it on fallback detection.
        return result
    if claude_children:
        result["_source_file_sha256"] = source_sha
        result["_source_merged_count"] = len(claude_children)
        source_sha = claude_tree_sha256(source_sha, claude_children)
    if codex_children:
        result["_source_file_sha256"] = source_sha
        result["_source_merged_count"] = len(codex_children)
        source_sha = _codex_tree_sha256(source_sha, codex_children)
    if merged_sources:
        # A DSH export is a TREE: the converter merged sibling
        # `subagents/<id>/session.jsonl` logs that are not in `source_sha`, so
        # the parent file's bytes do not identify what is displayed — two trees
        # with identical parents and different children would hash alike and
        # attribution would certify either as the other. Keep the plain sha for
        # display/debug and make the identity cover the whole tree.
        result["_source_file_sha256"] = source_sha
        result["_source_merged_count"] = len(merged_sources)
        source_sha = _dsh_tree_sha256(source_sha, merged_sources, dsh_child_hashes)
    # The displayed content's immutable identity — the sha256 of the EXACT
    # bytes parsed above (one read, one buffer): attribution requires the
    # canonical real export to still have these bytes at diagnosis time
    # (TOCTOU guard — never diagnose bytes the UI isn't showing).
    result["_source_sha256"] = source_sha
    return result


def _bounded_non_otlp_format(file_path: str) -> str:
    """Recognize an explicitly mismatched JSON source without converting it.

    Forcing OTLP must not fall back to the native loader's unbounded read or
    sibling-session discovery before the bounded capture reader can refuse it.
    """
    from ..errors import LoadError
    from .formats.otlp_files import MAX_MEMBER_BYTES, _open_file
    from pathlib import Path

    try:
        with _open_file(Path(file_path)) as handle:
            data = handle.read(MAX_MEMBER_BYTES + 1)
        if len(data) > MAX_MEMBER_BYTES:
            raise LoadError("OTLP capture size limit exceeded", code="size_limit", path=file_path)
        payload, error = _parse_trajectory_text(data.decode("utf-8-sig"), file_path)
        return detect_format(_normalize_payload(payload, file_path)) if error is None else "unknown"
    except (OSError, UnicodeError, ValueError, RecursionError):
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _stamp_otlp_identity(raw: dict) -> None:
    """Identify the selected run separately from its source artifact bytes."""
    import json

    metadata = raw["_otel"]
    capture_sha = raw.get("_source_capture_sha256") or raw["_source_sha256"]
    selection = {"capture_sha256": capture_sha, "session_id": metadata.get("session_id"),
                 "profile": metadata.get("profile"), "profile_version": metadata.get("profile_version")}
    identity = _sha256_bytes(b"trajviz-otlp-run-v1\x00" + json.dumps(
        selection, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    raw["_source_capture_sha256"] = capture_sha
    raw["_source_sha256"] = identity
    metadata.update(capture_sha256=capture_sha, run_sha256=identity)


def _dsh_tree_sha256(
    parent_sha: str, merged_sources: list[tuple[str, str]], child_hashes: dict[str, str] | None = None,
) -> str:
    """Content identity of a DSH export tree: parent log + merged child logs.

    Domain-separated so it can never collide with a plain file sha256, and keyed
    on each child's SESSION ID (see ``_dsh_merged_source_key``) so relocating or
    re-downloading the export does not change the identity of the same run.
    Computed ONLY when at least one child was merged, which is what keeps every
    single-file load — every other DSH load and every other format — on the
    plain sha256 it has always had.

    The bundled evaluator uses this loader for its canonical composite identity;
    child changes invalidate both the displayed run and evaluation caches.
    """
    import hashlib

    h = hashlib.sha256(b"trajviz-dsh-tree-v1\n")
    h.update(b"parent\x00" + parent_sha.encode() + b"\n")
    for key, path in sorted(merged_sources):
        if child_hashes and key in child_hashes:
            child_sha = child_hashes[key]
            h.update(key.encode() + b"\x00" + child_sha.encode() + b"\n")
            continue
        try:
            with open(path, "rb") as f:
                child_sha = _sha256_bytes(f.read())
        except OSError:
            # It parsed moments ago; if it has since vanished the identity must
            # still differ from any tree whose children are all readable.
            child_sha = "unreadable"
        h.update(key.encode() + b"\x00" + child_sha.encode() + b"\n")
    return h.hexdigest()


def _codex_tree_sha256(parent_sha: str, child_hashes: dict[str, str]) -> str:
    """Identify the exact verified parent and child bytes merged into a run."""
    import hashlib

    digest = hashlib.sha256(b"trajviz-codex-tree-v1\n")
    digest.update(b"parent\x00" + parent_sha.encode() + b"\n")
    for key, child_sha in sorted(child_hashes.items()):
        digest.update(key.encode() + b"\x00" + child_sha.encode() + b"\n")
    return digest.hexdigest()
