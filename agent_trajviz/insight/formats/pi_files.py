"""Pi active-history selection and bounded native-session archives."""

from __future__ import annotations

import io
import math
import zipfile
from typing import Any

from .claude_files import _read_member, _safe_member
from .common import decode_session_text
from .parse import _normalize_payload, _parse_trajectory_text
from .pi import _pi_extract_usage
from .sniff import _detect_event_stream_format


def pi_session_usage(events: list) -> dict[str, Any]:
    """Recorded usage over every branch, matching Pi's session stats scope."""
    tokens = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    cost = 0.0
    has_cost = False
    for event in events:
        if not isinstance(event, dict):
            continue
        usage = event.get("usage") if event.get("type") in {"usage", "compaction", "branch_summary"} else None
        message = event.get("message")
        if event.get("type") == "message" and isinstance(message, dict) and message.get("role") in {"assistant", "toolResult"}:
            usage = message.get("usage")
        normalized = _pi_extract_usage(usage)
        if not isinstance(usage, dict):
            continue
        if normalized:
            tokens["input"] += normalized["input"]
            tokens["output"] += normalized["output"]
            tokens["cacheRead"] += normalized["cache"]["read"]
            tokens["cacheWrite"] += normalized["cache"]["write"]
        recorded_cost = usage.get("cost")
        total_cost = recorded_cost.get("total") if isinstance(recorded_cost, dict) else None
        if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool) and math.isfinite(total_cost) and total_cost >= 0:
            cost += total_cost
            has_cost = True
    return {"tokens": {**tokens, "total": sum(tokens.values())}, "cost": cost if has_cost else None}

def normalize_pi_stream(events: list) -> tuple[list, bool, list[str]]:
    """Keep authoritative final messages from Pi's JSON progress protocol.

    Partial updates, turn summaries and agent summaries repeat the same model
    usage. They are progress records and must never become additional steps.
    """
    streamed = any(isinstance(event, dict) and event.get("type") in {"message_start", "message_update", "message_end"}
                   for event in events)
    if not streamed:
        return events, False, []
    finalized = []
    started = ended = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "session":
            finalized.append(event)
        elif kind == "message_start":
            started += 1
        elif kind == "message_end" and isinstance(event.get("message"), dict):
            ended += 1
            finalized.append({"type": "message", "message": event["message"]})
        elif kind == "entry_appended" and isinstance(event.get("entry"), dict):
            finalized.append(event["entry"])
        elif kind == "compaction_end" and isinstance(event.get("result"), dict) and not event.get("aborted"):
            finalized.append({**event["result"], "type": "compaction"})
        elif kind == "session_info_changed":
            finalized.append({"type": "session_info", "name": event.get("name") or ""})
    warnings = []
    if started > ended:
        warnings.append("Pi JSON stream has unfinished messages; only completed message_end records were imported.")
    return finalized, True, warnings


def select_pi_branch(events: list) -> tuple[list, dict[str, Any], list[str]]:
    """Select the final recorded leaf without replaying edits over raw history.

    Version 1 and older ad-hoc exports omit parent links and remain linear.
    A Pi fork's ``parentSession`` is provenance, never a file to read or merge.
    """
    entries = [event for event in events if isinstance(event, dict) and event.get("type") != "session"]
    context: dict[str, Any] = {"history_scope": "linear", "pi_session_usage": pi_session_usage(events)}
    if not any("parentId" in event for event in entries):
        return events, context, []
    if any("parentId" not in event for event in entries):
        return events, context, ["Pi parent links are incomplete; loaded the available linear history."]

    indexed = {event["id"]: event for event in entries if isinstance(event.get("id"), str) and event["id"]}
    if not indexed:
        return events, context, ["Pi parent links have no usable entry IDs; loaded the available linear history."]
    leaf = next(event for event in reversed(entries) if isinstance(event.get("id"), str) and event["id"])
    warnings = []
    if len(indexed) != sum(isinstance(event.get("id"), str) and bool(event["id"]) for event in entries):
        warnings.append("Pi session has duplicate entry IDs; the latest recorded occurrence was used for each ID.")
    selected = []
    visited: set[str] = set()
    current = leaf
    while current:
        entry_id = current["id"]
        if entry_id in visited:
            warnings.append("Pi active history contains a parent-link cycle; traversal stopped at the repeated entry.")
            break
        visited.add(entry_id)
        selected.append(current)
        parent = current.get("parentId")
        if parent is None:
            break
        if not isinstance(parent, str) or not parent or parent not in indexed:
            warnings.append("Pi active history references an unavailable parent entry; only the reachable branch was loaded.")
            break
        current = indexed[parent]
    selected.reverse()
    excluded = len(indexed) - len(selected)
    context.update(history_scope="active_branch", pi_active_leaf_id=leaf["id"], pi_excluded_entry_count=excluded)
    if excluded:
        warnings.append(
            f"Pi active branch selected; {excluded} entries from other branches were excluded from charts and token totals. "
            "The full session history and usage remain in raw metadata."
        )
    return [event for event in events if isinstance(event, dict) and event.get("type") == "session"] + selected, context, warnings


def parse_pi_zip(file_path: str, *, data: bytes | None = None) -> list | dict | None:
    """Read one Pi session without extracting ZIP members or following forks."""
    try:
        with zipfile.ZipFile(io.BytesIO(data) if data is not None else file_path) as archive:
            budget = [0]
            sessions = []
            names = set()
            for info in archive.infolist():
                path = _safe_member(info)
                if path is None or info.is_dir() or path.suffix != ".jsonl" or "__MACOSX" in path.parts:
                    continue
                if path in names:
                    return {"_error": f"Duplicate JSONL member in zip: {path}"}
                names.add(path)
                raw = _read_member(archive, info, budget)
                payload, error = _parse_trajectory_text(decode_session_text(raw), str(path))
                if error:
                    continue
                payload = _normalize_payload(payload, str(path))
                if isinstance(payload, list) and _detect_event_stream_format(payload) == "pi":
                    sessions.append(payload)
                    if len(sessions) > 1:
                        return {"_error": "Zip contains multiple Pi sessions; upload one session at a time."}
            return sessions[0] if sessions else None
    except (OSError, UnicodeError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
        return {"_error": f"Could not read Pi session zip: {exc}"}
