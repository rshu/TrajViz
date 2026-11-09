"""Translate OpenCode 2's typed session records into the shared message model."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _error(value: object) -> str:
    return _text(_dict(value).get("message")) or _text(value)


def _is_step_interruption(value: object) -> bool:
    """The native runner's exact control-flow marker, not a provider abort.

    OpenCode's SessionStep uses this pair for runtime interruption or declined
    interactions. Extra status/response evidence, generic AbortError names,
    and other messages may describe real failures and must remain warnings.
    """
    return isinstance(value, dict) and value == {"type": "aborted", "message": "Step interrupted"}


def _tokens(value: dict) -> dict:
    tokens = deepcopy(value)
    cache = _dict(tokens.get("cache"))
    values = [tokens.get("input"), tokens.get("output"), tokens.get("reasoning"),
              cache.get("read"), cache.get("write")]
    # OpenCode 2's TokenUsage.total explicitly sums all five components.
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        tokens["total"] = sum(values)
    return tokens


def _content(part: dict, message_id: str, position: int) -> dict:
    result = deepcopy(part)
    kind = part.get("type")
    if kind == "reasoning":
        time = _dict(part.get("time"))
        result["time"] = {"start": time.get("created"), "end": time.get("completed")}
    if kind != "tool":
        return result
    result["id"] = f"{message_id}:tool:{position}"
    result["callID"] = _text(part.get("id")) or _text(part.get("callID"))
    # Native v2 records name the tool as ``name``; desktop flat exports use ``tool``.
    result["tool"] = _text(part.get("name")) or _text(part.get("tool")) or "?"
    state = deepcopy(_dict(part.get("state")))
    time = _dict(part.get("time"))
    state["time"] = {"start": time.get("ran", time.get("created")), "end": time.get("completed")}
    content = state.get("content")
    if isinstance(content, list):
        state["output"] = "\n".join(_text(item.get("text")) for item in content
                                      if isinstance(item, dict) and item.get("type") == "text")
    if "error" in state:
        state["error"] = _error(state["error"])
    metadata = state.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        state["metadata"] = metadata
    # A native tool invocation can complete while its result reports failure.
    # Only literal booleans carry this meaning; preserve the original evidence.
    if metadata.get("error") is True or metadata.get("timeout") is True:
        state["status"] = "error"
        if not state.get("error"):
            state["error"] = "Tool timed out" if metadata.get("timeout") is True else "Tool reported an error"
    if state.get("status") in {"streaming", "running"}:
        metadata["_outcome_observed"] = False
    metadata["opencode_v2_tool"] = deepcopy(part)
    result["state"] = state
    return result


def _has_opencode_v2_markers(records: list, info: dict | None = None) -> bool:
    """True when typed records carry native OpenCode 2 shape, not desktop flat.

    Desktop / Windows exports also use top-level ``type`` without ``info``, but
    their tools are named ``tool`` (not ``name``), they lack ``time.streamed``,
    and they never emit shell/synthetic/skill control records. Claiming those
    as v2 overwrites flat token totals and tool names.
    """
    version = _text(_dict(info).get("version"))
    if version.startswith("2."):
        return True
    for message in records:
        if not isinstance(message, dict) or isinstance(message.get("info"), dict):
            continue
        kind = message.get("type")
        if kind in {"shell", "synthetic", "skill"}:
            return True
        if any(key in message for key in ("isSubAgent", "parentSessionID", "sessionDepth")):
            return True
        time = message.get("time") if isinstance(message.get("time"), dict) else {}
        if "streamed" in time:
            return True
        content = message.get("content") if isinstance(message.get("content"), list) else []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool" and "name" in part:
                return True
    return False


def normalize_opencode_v2(raw: dict) -> dict:
    """Normalize only native v2 records; retain the original capture as evidence.

    Lifecycle markers are preserved in raw data, rather than invented as
    conversation turns. Usage is kept per request, never copied from the
    cumulative session token counters.
    """
    records = raw.get("messages")
    if not isinstance(records, list) or not any(isinstance(m, dict) and "info" not in m
                                               and isinstance(m.get("type"), str) for m in records):
        return raw
    root_info = _dict(raw.get("info"))
    if not _has_opencode_v2_markers(records, root_info):
        return raw
    raw["_opencode_v2_messages"] = deepcopy(records)
    root_id = _text(root_info.get("id"))
    root_info.setdefault("directory", _dict(root_info.get("location")).get("directory", ""))
    messages: list[dict[str, Any]] = []
    errors: list[str] = []
    warning_errors: list[str] = []
    terminal_error: str | None = None
    control_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("info"), dict):
            messages.append(record)
            continue
        kind = record.get("type")
        session_id = _text(record.get("sessionID")) or root_id
        if session_id == root_id and kind in {"user", "assistant", "shell", "compaction"}:
            # A resumed turn supersedes the preceding terminal outcome. Keep
            # that earlier failure in its native record and history warnings.
            terminal_error = _error(record.get("error")) or None
        if kind not in {"user", "assistant", "synthetic", "system", "skill", "shell", "compaction"}:
            control_count += 1
            if kind == "idle":
                outcome_error = None
                if record.get("outcome") in {"failed", "interrupted"}:
                    outcome_error = _error(record.get("error")) or f"Session turn {record['outcome']}."
                    errors.append(outcome_error)
                    normal_interruption = record["outcome"] == "interrupted" and (
                        not record.get("error") or _is_step_interruption(record["error"])
                    )
                    if not normal_interruption:
                        warning_errors.append(outcome_error)
                if session_id == root_id:
                    terminal_error = outcome_error
            continue
        role = kind if kind in {"user", "assistant"} else "system"
        if kind == "compaction" and record.get("status") == "completed":
            role = "compaction"
        info = {key: deepcopy(record[key]) for key in ("id", "time", "agent", "finish", "cost") if key in record}
        info.update({"role": role, "sessionID": _text(record.get("sessionID")) or root_id,
                     "opencode_event_type": kind})
        if kind == "compaction":
            status = _text(record.get("status"))
            info.update({"type": "compaction" if status == "completed" else f"compaction-{status or 'unknown'}",
                         "reason": record.get("reason", ""), "compaction_status": status})
        for key in ("isSubAgent", "parentSessionID", "sessionDepth", "sessionTitle"):
            if key in record:
                info[key] = record[key]
        if kind == "assistant":
            time = _dict(info.get("time"))
            created, streamed, completed = (time.get(key) for key in ("created", "streamed", "completed"))
            if (isinstance(streamed, (int, float)) and not isinstance(streamed, bool)
                    and math.isfinite(streamed) and isinstance(created, (int, float))
                    and not isinstance(created, bool) and math.isfinite(created) and streamed >= created
                    and (completed is None or (isinstance(completed, (int, float))
                         and not isinstance(completed, bool) and math.isfinite(completed) and streamed <= completed))):
                time["completed"] = streamed
                info["timing_source"] = "model_request"
        model = _dict(record.get("model"))
        if model:
            info["modelID"] = _text(model.get("id"))
            info["providerID"] = _text(model.get("providerID"))
            info["model"] = {"modelID": info["modelID"], "providerID": info["providerID"]}
        if isinstance(record.get("tokens"), dict):
            info["tokens"] = _tokens(record["tokens"])
        parts = []
        if kind == "assistant":
            parts = [_content(part, _text(record.get("id")), i)
                     for i, part in enumerate(record.get("content", []) if isinstance(record.get("content"), list) else [])
                     if isinstance(part, dict)]
        elif kind == "shell":
            status, exit_code = record.get("status"), record.get("exit")
            has_exit = isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and math.isfinite(exit_code)
            observed = status in {"timeout", "killed"} or (status == "exited" and has_exit)
            failed = status in {"timeout", "killed"} or (status == "exited" and has_exit and exit_code != 0)
            parts = [{"type": "tool", "tool": "bash", "callID": record.get("shellID", record.get("id", "")),
                      "state": {"status": "error" if failed else "completed" if observed else "running",
                                "input": {"command": record.get("command", "")},
                                "output": _dict(record.get("output")).get("output", ""),
                                "error": (f"Shell {status}" if status in {"timeout", "killed"} else f"exit {exit_code}") if failed else None,
                                "metadata": {"_outcome_observed": observed, "opencode_v2_shell": deepcopy(record)},
                                "time": {"start": _dict(record.get("time")).get("created"),
                                         "end": _dict(record.get("time")).get("completed")}}}]
        else:
            text = record.get("summary") if kind == "compaction" else record.get("text")
            if isinstance(text, str):
                parts = [{"type": "text", "text": text}]
        if kind == "user" and isinstance(record.get("files"), list):
            parts.extend({"type": "file", "data": deepcopy(file)} for file in record["files"])
        if kind == "assistant" and isinstance(record.get("snapshot"), dict):
            parts.append({"type": "snapshot", "data": deepcopy(record["snapshot"])})
        if record.get("error"):
            info["error"] = deepcopy(record["error"])
            error = _error(record["error"])
            if error:
                errors.append(error)
                if not _is_step_interruption(record["error"]):
                    warning_errors.append(error)
            if kind == "assistant":
                info.setdefault("finish", "error")
        if kind == "assistant" and not info.get("finish") and not _dict(record.get("time")).get("completed"):
            parts.insert(0, {"type": "step-start"})
        messages.append({"info": info, "parts": parts})
    raw["messages"] = messages
    raw["_opencode_v2"] = True
    raw["_opencode_v2_control_count"] = control_count
    raw["_opencode_v2_errors"] = [terminal_error] if terminal_error else []
    if errors:
        raw["_opencode_v2_history_errors"] = list(dict.fromkeys(errors))
    if warning_errors:
        warnings = raw.setdefault("_load_warnings", [])
        for error in dict.fromkeys(warning_errors):
            warning = f"OpenCode model/session error: {error}"
            if warning not in warnings:
                warnings.append(warning)
    return raw
