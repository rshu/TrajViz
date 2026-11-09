"""CodeAgent session-export adapter on the OpenCode message schema.

CodeAgent SQLite/session exports wrap each message and part in a ``data``
object and keep child agents under ``child_sessions``.  Normalize that envelope
into OpenCode's ``info`` + ``messages`` shape, then reuse the shared OpenCode
metadata/timing conversion while restoring CodeAgent product identity.
"""

from __future__ import annotations

from typing import Any

from ..tool_failure import FAILURE_STATUSES
from .common import as_text
from .opencode import _convert_opencode_metadata, _opencode_token_value

_ROLES = frozenset({"user", "assistant", "system"})


def _messages_have_data_wrapped_typed_part(messages: object) -> bool:
    """True when any message carries ``data.role`` and a part with ``data.type``."""
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        data = message.get("data")
        parts = message.get("parts")
        if not isinstance(data, dict) or as_text(data.get("role")) not in _ROLES:
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            payload = part.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("type"), str) and payload["type"]:
                return True
    return False


def looks_like_codeagent(raw: dict) -> bool:
    """True when a JSON object is a CodeAgent ``session`` + ``child_sessions`` export.

    Requires the CodeAgent envelope (``session`` plus a ``child_sessions`` list)
    and at least one data-wrapped typed part in the parent or a child.  A lone
    ``exported_at`` / ``data.time`` is not enough — that falsely claimed unrelated
    session JSON.
    """
    session = raw.get("session")
    children = raw.get("child_sessions")
    if not isinstance(session, dict) or not isinstance(children, list):
        return False
    if _messages_have_data_wrapped_typed_part(session.get("messages")):
        return True
    return any(
        isinstance(child, dict) and _messages_have_data_wrapped_typed_part(child.get("messages"))
        for child in children
    )


def _session_info(session: dict) -> dict[str, Any]:
    time_info: dict[str, Any] = {}
    created = session.get("time_created")
    updated = session.get("time_updated")
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        time_info["created"] = created
    if isinstance(updated, (int, float)) and not isinstance(updated, bool):
        time_info["updated"] = updated
    info: dict[str, Any] = {
        "id": as_text(session.get("id")),
        "slug": as_text(session.get("slug")),
        "directory": as_text(session.get("directory")),
        "title": as_text(session.get("title")),
        "version": as_text(session.get("version")),
        "time": time_info,
    }
    parent_id = as_text(session.get("parent_id"))
    if parent_id:
        info["parentID"] = parent_id
    share_url = as_text(session.get("share_url"))
    if share_url:
        info["shareURL"] = share_url
    return info


def _unwrap_part(part: dict, *, session_id: str, message_id: str) -> dict[str, Any]:
    payload = part.get("data") if isinstance(part.get("data"), dict) else {}
    unwrapped = dict(payload)
    part_id = as_text(part.get("id")) or as_text(unwrapped.get("id"))
    if part_id:
        unwrapped["id"] = part_id
    unwrapped.setdefault("sessionID", session_id)
    if message_id:
        unwrapped.setdefault("messageID", message_id)
    return unwrapped


def _unwrap_message(
    message: dict,
    *,
    session_id: str,
    is_sub_agent: bool = False,
    parent_session_id: str = "",
    session_title: str = "",
    session_depth: int = 0,
) -> dict[str, Any] | None:
    data = message.get("data")
    if not isinstance(data, dict):
        return None
    role = as_text(data.get("role"))
    if role not in _ROLES:
        return None
    message_id = as_text(message.get("id")) or as_text(data.get("id"))
    info: dict[str, Any] = {
        "id": message_id,
        "sessionID": session_id,
        "role": role,
    }
    for key in (
        "time",
        "agent",
        "modelID",
        "providerID",
        "mode",
        "path",
        "cost",
        "tokens",
        "finish",
        "parentID",
        "model",
        "variant",
        "summary",
    ):
        if key in data:
            info[key] = data[key]
    if is_sub_agent:
        info["isSubAgent"] = True
        if parent_session_id:
            info["parentSessionID"] = parent_session_id
        if session_depth:
            info["sessionDepth"] = session_depth
        if session_title:
            info["sessionTitle"] = session_title
    raw_parts = message.get("parts") if isinstance(message.get("parts"), list) else []
    parts = [
        _unwrap_part(part, session_id=session_id, message_id=message_id)
        for part in raw_parts
        if isinstance(part, dict)
    ]
    return {"info": info, "parts": parts}


def _manifest_entry(session: dict, *, depth: int, discovered_by: str) -> dict[str, Any]:
    messages = session.get("messages") if isinstance(session.get("messages"), list) else []
    parent_id = session.get("parent_id")
    return {
        "depth": depth,
        "discovered_by": discovered_by,
        "traversal_parent_id": parent_id,
        "message_count": len(messages),
        "info": {
            "id": as_text(session.get("id")),
            "parentID": as_text(parent_id) or None,
            "title": as_text(session.get("title")),
        },
    }


def _session_depth(child_id: str, parent_id: str, root_id: str, by_id: dict[str, dict]) -> int:
    """Depth of *child_id* below the root (1 for a direct child)."""
    depth = 1
    seen = {child_id}
    current = parent_id
    while current and current != root_id and current in by_id and current not in seen:
        seen.add(current)
        depth += 1
        current = as_text(by_id[current].get("parent_id"))
    return depth


def _first_message_model_id(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        info = message.get("info") if isinstance(message.get("info"), dict) else {}
        candidate = as_text(info.get("modelID"))
        if not candidate:
            nested = info.get("model") if isinstance(info.get("model"), dict) else {}
            candidate = as_text(nested.get("modelID"))
        if candidate:
            return candidate
    return ""


def _token_usage_from_messages(messages: list[dict[str, Any]]) -> dict[str, int | float]:
    totals = {
        "total": 0,
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    for message in messages:
        info = message.get("info") if isinstance(message.get("info"), dict) else {}
        tokens = info.get("tokens") if isinstance(info.get("tokens"), dict) else {}
        totals["total"] += _opencode_token_value(tokens.get("total"))
        totals["input"] += _opencode_token_value(tokens.get("input"))
        totals["output"] += _opencode_token_value(tokens.get("output"))
        totals["reasoning"] += _opencode_token_value(tokens.get("reasoning"))
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        totals["cache_read"] += _opencode_token_value(cache.get("read"))
        totals["cache_write"] += _opencode_token_value(cache.get("write"))
    return {
        "total_tokens": totals["total"],
        "prompt_tokens": totals["input"],
        "completion_tokens": totals["output"],
        "reasoning_tokens": totals["reasoning"],
        "cache_read_tokens": totals["cache_read"],
        "cache_write_tokens": totals["cache_write"],
    }


def _count_failed_tool_parts(messages: list[dict[str, Any]]) -> int:
    failed = 0
    for message in messages:
        for part in message.get("parts") or []:
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            status = str(state.get("status") or part.get("status") or "").lower()
            if status in FAILURE_STATUSES:
                failed += 1
    return failed


def _convert_codeagent_to_internal(raw: dict) -> dict:
    """Normalize a CodeAgent session export into the shared step-model dict."""
    session = raw.get("session")
    if not isinstance(session, dict):
        return {"_error": "CodeAgent export is missing a session object."}
    children = [
        child
        for child in (raw.get("child_sessions") if isinstance(raw.get("child_sessions"), list) else [])
        if isinstance(child, dict) and as_text(child.get("id"))
    ]
    by_id = {as_text(child.get("id")): child for child in children}

    root_id = as_text(session.get("id"))
    messages: list[dict[str, Any]] = []
    tool_parts = 0
    reasoning_parts = 0

    def _append(message: dict, **kwargs: Any) -> None:
        nonlocal tool_parts, reasoning_parts
        unwrapped = _unwrap_message(message, **kwargs)
        if unwrapped is None:
            return
        for part in unwrapped["parts"]:
            kind = part.get("type")
            if kind == "tool":
                tool_parts += 1
            elif kind == "reasoning":
                reasoning_parts += 1
        messages.append(unwrapped)

    for message in session.get("messages") if isinstance(session.get("messages"), list) else []:
        if isinstance(message, dict):
            _append(message, session_id=root_id)

    for child in children:
        child_id = as_text(child.get("id"))
        parent_id = as_text(child.get("parent_id")) or root_id
        title = as_text(child.get("title"))
        depth = _session_depth(child_id, parent_id, root_id, by_id)
        for message in child.get("messages") if isinstance(child.get("messages"), list) else []:
            if isinstance(message, dict):
                _append(
                    message,
                    session_id=child_id,
                    is_sub_agent=True,
                    parent_session_id=parent_id,
                    session_title=title,
                    session_depth=depth,
                )

    manifest = [
        _manifest_entry(session, depth=0, discovered_by="root"),
        *[
            _manifest_entry(
                child,
                depth=_session_depth(
                    as_text(child.get("id")),
                    as_text(child.get("parent_id")) or root_id,
                    root_id,
                    by_id,
                ),
                discovered_by="child_sessions",
            )
            for child in children
        ],
    ]
    export = {
        "schema_version": 1,
        "source_format": "codeagent_session_export",
        "source_database": as_text(raw.get("source_db")),
        "root_session_id": root_id,
        "generated_at": as_text(raw.get("exported_at")),
        "export_note": as_text(raw.get("export_note")),
        "complete": True,
        "warnings": [],
    }
    # OpenCode only reads subagent_sessions / event_rows from statistics;
    # message/tool counts come from the shared message walk.
    converted: dict[str, Any] = {
        "info": _session_info(session),
        "messages": messages,
        "session_manifest": manifest,
        "statistics": {
            "subagent_sessions": len(children),
            "event_rows": 0,
        },
        "export_metadata": export,
    }

    _convert_opencode_metadata(converted)

    metadata = converted.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        converted["metadata"] = metadata

    model_id = _first_message_model_id(messages) or as_text(metadata.get("model"))
    sub_agent_count = metadata.get("sub_agent_count", 0)
    if not isinstance(sub_agent_count, int):
        sub_agent_count = len(children)

    metadata.update({
        "agent": "codeagent",
        "model": model_id,
        "generator_name": "codeagent",
        "generator_version": as_text(session.get("version")),
        "format_version": str(export["schema_version"]),
        "sub_agent_count": sub_agent_count,
        "session_count": len(manifest) or 1,
        "export_generated_at": export["generated_at"],
        "export_complete": export["complete"],
        "export_warnings": export["warnings"],
    })

    token_usage = _token_usage_from_messages(messages)
    converted["token_usage"] = token_usage
    failed_tools = _count_failed_tool_parts(messages)
    stats = converted.get("stats")
    if not isinstance(stats, dict):
        stats = {}
        converted["stats"] = stats
    stats["failed_tool_calls"] = failed_tools

    has_runtime_tokens = any(
        value > 0
        for key, value in token_usage.items()
        if key != "total_tokens" and isinstance(value, (int, float))
    ) or bool(token_usage.get("total_tokens"))

    converted["_codeagent_format"] = True
    converted["_source_format"] = "codeagent"
    converted["_capabilities"] = {
        "has_timing": True,
        "has_tool_calls": bool(stats.get("total_tool_calls", tool_parts)),
        "has_runtime_token_usage": has_runtime_tokens,
        "has_reasoning_content": bool(reasoning_parts),
        "has_session_hierarchy": bool(sub_agent_count),
    }
    return converted
