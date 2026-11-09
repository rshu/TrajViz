"""OpenCode info+messages export → internal trajectory."""

import math
from datetime import datetime, UTC

from .common import as_text
from .opencode_v2 import normalize_opencode_v2


def _opencode_ms_to_iso(ms: object) -> str:
    """Epoch milliseconds → ISO 8601 UTC, or ``""`` when that is not an instant.

    Same shape as ``dsh._dsh_ms_to_iso`` but WITHOUT its ``ms <= 0`` rule:
    OpenCode exports legitimately record ``created: 0``, which must keep mapping
    to the epoch rather than being erased.  A recorder that wrote microseconds
    instead of milliseconds puts ``fromtimestamp`` out of its platform range,
    and that used to raise ``ValueError: year 56664 is out of range`` straight
    out of ``load_trajectory`` — which ``run_group`` does not catch, so one such
    file aborted a whole batch.  An unrepresentable instant is reported as
    missing, not clamped to a plausible-looking date.
    """
    if ms is None or isinstance(ms, bool) or not isinstance(ms, (int, float)):
        return ""
    if not math.isfinite(ms):
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


# Desktop / Windows session exports store each turn flat (`type` + `text` or
# `content`). The SQLite consolidator stores the same turns as `info` + `parts`.
# `idle` rows are session heartbeats with no transcript.
_FLAT_OPENCODE_TYPES = frozenset({"user", "assistant", "system", "idle", "compaction"})


def _is_flat_opencode_message(message: dict) -> bool:
    """True for a desktop session turn that is not already `info` + `parts`."""
    info = message.get("info")
    if isinstance(info, dict) and (
        isinstance(info.get("role"), str) or isinstance(message.get("parts"), list)
    ):
        return False
    kind = message.get("type")
    return isinstance(kind, str) and kind in _FLAT_OPENCODE_TYPES


def _flat_model_ids(message: dict, metadata: dict) -> tuple[str, str]:
    """Desktop turns name the model as ``modelID`` (user metadata) or ``id`` (assistant)."""
    model = message.get("model")
    if not isinstance(model, dict):
        model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    return as_text(model.get("modelID") or model.get("id")), as_text(model.get("providerID"))


def _with_opencode_token_total(tokens: dict) -> dict:
    """Copy token counts, filling `total` when the export only split the fields.

    OpenCode's own total is input + output + cache read + cache write.
    Reasoning is recorded beside that total, not inside it.
    """
    copied = dict(tokens)
    if "total" in copied:
        return copied
    cache = copied.get("cache") if isinstance(copied.get("cache"), dict) else {}
    pieces = (
        _opencode_token_value(copied.get("input")),
        _opencode_token_value(copied.get("output")),
        _opencode_token_value(cache.get("read")),
        _opencode_token_value(cache.get("write")),
    )
    if any(pieces):
        copied["total"] = sum(pieces)
    return copied


def _parts_from_flat_message(message: dict) -> list:
    """Body of a desktop turn. Content parts are already OpenCode part objects."""
    content = message.get("content")
    if isinstance(content, list):
        parts = [item for item in content if isinstance(item, dict)]
        if parts:
            return parts
    elif isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    text = message.get("text")
    if isinstance(text, str) and text:
        return [{"type": "text", "text": text}]
    return []


def _flat_message_to_internal(message: dict, session_id: str) -> dict:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    kind = message["type"]
    model_id, provider_id = _flat_model_ids(message, metadata)
    raw_time = message.get("time") if isinstance(message.get("time"), dict) else {}
    time_info = {
        key: raw_time[key] for key in ("created", "completed") if key in raw_time
    }

    info: dict = {
        "id": as_text(message.get("id")),
        "sessionID": session_id,
        "role": kind,
        "agent": as_text(message.get("agent") or metadata.get("agent")),
        "time": time_info,
    }
    if model_id:
        info["modelID"] = model_id
    if provider_id:
        info["providerID"] = provider_id
    if model_id or provider_id:
        info["model"] = {"modelID": model_id, "providerID": provider_id}
    tokens = message.get("tokens")
    if isinstance(tokens, dict):
        info["tokens"] = _with_opencode_token_total(tokens)
    finish = message.get("finish")
    if isinstance(finish, str) and finish:
        info["finish"] = finish
    cost = message.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        info["cost"] = cost
    return {"info": info, "parts": _parts_from_flat_message(message)}


def _normalize_flat_opencode_messages(raw: dict) -> None:
    """Rewrite a desktop session export into the consolidator's message shape.

    No-op for exports that already use ``info`` + ``parts``. ``idle`` rows are
    dropped: they record that the session went idle, not a turn.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return
    if not any(_is_flat_opencode_message(message) for message in messages if isinstance(message, dict)):
        return
    session = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    session_id = as_text(session.get("id"))
    normalized: list = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if not _is_flat_opencode_message(message):
            normalized.append(message)
            continue
        if message.get("type") == "idle":
            continue
        normalized.append(_flat_message_to_internal(message, session_id))
    raw["messages"] = normalized


def _session_directory(info: dict) -> str:
    directory = as_text(info.get("directory"))
    if directory:
        return directory
    location = info.get("location")
    if isinstance(location, dict):
        return as_text(location.get("directory"))
    return ""


def _session_model_name(info: dict, model_info: dict) -> str:
    """Message ``modelID`` when a turn recorded one, else the session model.

    Desktop session info stores ``model.id``. Flat turns are normalized to
    ``modelID`` before this runs, as are consolidator messages.
    """
    name = as_text(model_info.get("modelID"))
    if name:
        return name
    session_model = info.get("model")
    if not isinstance(session_model, dict):
        return ""
    return as_text(session_model.get("modelID") or session_model.get("id"))


def _opencode_token_value(value: object) -> int | float:
    """A reported token count, or 0 when the value is not a usable number.

    Same reading as ``parser._finite_token``, applied to the session-level
    accumulators: a string/bool/None/non-finite count carries no information and
    must be *rejected* rather than coerced, and a NEGATIVE count is PRESERVED —
    OpenCode really does report one (it subtracts the cache read from the prompt
    size) and the metrics layer counts those steps instead of hiding them.
    Duplicated rather than imported because the dependency runs
    parser → loaders → formats, so a converter cannot import ``parser``.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value):
        return 0
    return value


def _convert_opencode_metadata(raw: dict) -> dict:
    """Populate metadata, timing, and output keys from OpenCode info structure.

    Mutates *raw* in place so that compute_metrics() and the session-detail
    renderer can read the standard fields.  Returns the same dict for convenience.
    """
    normalize_opencode_v2(raw)
    info = raw.get("info", {})
    if not isinstance(info, dict):
        return raw

    _normalize_flat_opencode_messages(raw)

    time_info = info.get("time", {}) if isinstance(info.get("time"), dict) else {}
    created_ms = time_info.get("created")
    updated_ms = time_info.get("updated")

    duration_seconds = 0.0
    started_at = _opencode_ms_to_iso(created_ms)
    finished_at = _opencode_ms_to_iso(updated_ms)
    # A duration needs BOTH endpoints to be real instants.  Subtracting a raw
    # microsecond ``updated`` from a converted ``created`` reported a
    # 1.7-billion-second session instead of admitting the end is unknown.
    if started_at and finished_at:
        duration_seconds = round((updated_ms - created_ms) / 1000.0, 3)

    summary = info.get("summary", {}) if isinstance(info.get("summary"), dict) else {}
    model_info = {}
    # Prefer session model, then the first assistant request model.
    session_model = info.get("model") if isinstance(info.get("model"), dict) else {}
    model_info = {"modelID": session_model.get("modelID") or session_model.get("id")}
    # Try to get model from first message's info
    messages = raw.get("messages", [])
    if isinstance(messages, list) and messages:
        first_msg_info = next((m["info"] for m in messages if isinstance(m, dict)
                               and isinstance(m.get("info"), dict)
                               and m["info"].get("role") == "assistant"), {})
        if isinstance(first_msg_info, dict):
            request_model = first_msg_info.get("model", {})
            if isinstance(request_model, dict) and request_model.get("modelID"):
                model_info = request_model

    # Consolidated OpenCode/CodeArts exports flatten child-session messages into
    # the main messages array.  Count distinct child session IDs instead of
    # hard-coding zero.  A session_manifest is also accepted so a child with
    # no persisted messages is still represented accurately.
    sub_agent_session_ids: set[str] = set()
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        message_info = message.get("info", {})
        if not isinstance(message_info, dict):
            continue
        agent_name = message_info.get("agent", "")
        is_sub_agent = bool(message_info.get("isSubAgent")) or (
            isinstance(agent_name, str) and agent_name.endswith("(subagent)")
        )
        if not is_sub_agent:
            continue
        child_session_id = message_info.get("sessionID")
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    manifest = raw.get("session_manifest", [])
    for entry in manifest if isinstance(manifest, list) else []:
        if not isinstance(entry, dict) or not entry.get("depth"):
            continue
        manifest_info = entry.get("info", {})
        child_session_id = manifest_info.get("id") if isinstance(manifest_info, dict) else None
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    exported_stats = raw.get("statistics", {})
    exported_sub_agent_count = (
        exported_stats.get("subagent_sessions", 0)
        if isinstance(exported_stats, dict)
        else 0
    )
    if not isinstance(exported_sub_agent_count, int):
        exported_sub_agent_count = 0
    sub_agent_count = max(len(sub_agent_session_ids), exported_sub_agent_count)
    exported_event_count = (
        exported_stats.get("event_rows", 0) if isinstance(exported_stats, dict) else 0
    )
    if not isinstance(exported_event_count, int):
        exported_event_count = 0
    manifest_entries = manifest if isinstance(manifest, list) else []
    manifest_event_count = sum(
        len(entry.get("events", []))
        for entry in manifest_entries if isinstance(entry, dict)
        if isinstance(entry.get("events", []), list)
    )
    event_count = max(exported_event_count, manifest_event_count)

    directory = _session_directory(info)
    raw["metadata"] = {
        "session_id": as_text(info.get("id")),
        "slug": as_text(info.get("slug")),
        "title": as_text(info.get("title")),
        "directory": directory,
        "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1],
        "agent": "opencode",
        "model": _session_model_name(info, model_info),
        "hostname": "",
        "platform": "",
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": "",
        "ground_truth_patch": "",
        "baseline_commit": "",
        "sanitized": False,
        "server_url": "",
        "server_version": as_text(info.get("version")),
        "generator_name": "opencode",
        "generator_version": as_text(info.get("version")),
        "format_version": "",
        "sub_agent_count": sub_agent_count,
        "event_count": event_count + raw.get("_opencode_v2_control_count", 0),
    }
    raw["timing"] = {
        "total_duration": duration_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    raw["output"] = {
        "patch": "", "patch_length": 0, "patch_lines": 0,
        "has_patch": bool(summary.get("additions") or summary.get("deletions")),
        "error": "\n".join(raw.get("_opencode_v2_errors", [])) or None,
        "additions": summary.get("additions", 0),
        "deletions": summary.get("deletions", 0),
        "files_changed": summary.get("files", 0),
    }
    raw.setdefault("input", {"prompt": "", "prompt_length": 0})

    # Compute raw stats from messages array
    messages = raw.get("messages", [])
    user_count = 0
    asst_count = 0
    total_tool_calls = 0
    tool_breakdown: dict[str, int] = {}
    total_input = 0
    total_output = 0
    total_tokens = 0
    last_assistant_by_session: dict[str, dict] = {}
    for msg in (messages if isinstance(messages, list) else []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {}) if isinstance(msg.get("info"), dict) else {}
        role = msg_info.get("role", "")
        if role == "user":
            user_count += 1
        elif role == "assistant":
            asst_count += 1
            session_id = as_text(msg_info.get("sessionID")) or as_text(info.get("id"))
            last_assistant_by_session[session_id] = msg
        # Count tokens
        tok = msg_info.get("tokens", {}) if isinstance(msg_info.get("tokens"), dict) else {}
        total_input += _opencode_token_value(tok.get("input"))
        total_output += _opencode_token_value(tok.get("output"))
        reported_total = tok.get("total")
        if (isinstance(reported_total, (int, float)) and not isinstance(reported_total, bool)
                and math.isfinite(reported_total)):
            # Current exports already include cache and reasoning in total.
            # Rebuilding it from input+output discarded most cached usage.
            total_tokens += reported_total
        else:
            total_tokens += _opencode_token_value(tok.get("input")) + _opencode_token_value(tok.get("output"))
        # Count tool calls from parts
        for part in (msg.get("parts", []) if isinstance(msg.get("parts"), list) else []):
            if isinstance(part, dict) and part.get("type") == "tool":
                state = part.get("state")
                if isinstance(state, dict) and state.get("status") in {"pending", "running", "streaming"}:
                    metadata = state.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                        state["metadata"] = metadata
                    metadata["_outcome_observed"] = False
                total_tool_calls += 1
                tname = as_text(part.get("tool")) or "?"
                tool_breakdown[tname] = tool_breakdown.get(tname, 0) + 1

    # Consolidated exports interleave multiple sessions.  A finished child must
    # not conceal a parent whose capture stopped during its final response.
    for session_id, message in last_assistant_by_session.items():
        message_info = message["info"]
        parts = message.get("parts", [])
        part_types = {
            part.get("type") for part in parts if isinstance(part, dict)
            and isinstance(part.get("type"), str)
        } if isinstance(parts, list) else set()
        if (not message_info.get("finish")
                and part_types.intersection({"step-start", "step_start"})
                and not part_types.intersection({"step-finish", "step_finish"})):
            indices = raw.setdefault("_opencode_incomplete_indices", [])
            position = next(index for index, candidate in enumerate(messages) if candidate is message)
            if isinstance(indices, list) and position not in indices:
                indices.append(position)
            warning = (
                f"OpenCode session {session_id or '(unknown)'}: capture ends before "
                "the last assistant response finishes; it may be incomplete."
            )
            warnings = raw.setdefault("_load_warnings", [])
            if isinstance(warnings, list) and warning not in warnings:
                warnings.append(warning)

    raw.setdefault("token_usage", {
        "total_tokens": total_tokens,
        "prompt_tokens": total_input,
        "completion_tokens": total_output,
    })
    raw.setdefault("stats", {
        "total_messages": user_count + asst_count,
        "user_messages": user_count,
        "assistant_messages": asst_count,
        "total_tool_calls": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "failed_tool_calls": 0,
        "reasoning_steps": asst_count,
    })
    return raw
