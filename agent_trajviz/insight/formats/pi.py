"""Pi coding-agent JSONL → internal trajectory."""

import json
import math
from typing import Any

from .common import _iso_to_epoch_ms

# Pi coding-agent tools use lowercase names and a `path` argument. Map them to
# the Claude Code / OpenCode vocabulary so file-interaction charts and write
# detection work without a second set of aliases.
_PI_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "find": "Glob",
    "ls": "Glob",
}


def _pi_event_ts_ms(event: dict) -> int | None:
    """Prefer the event ISO timestamp; fall back to nested message epoch-ms."""
    ts = _iso_to_epoch_ms(event.get("timestamp") if isinstance(event.get("timestamp"), str) else None)
    if ts is not None:
        return ts
    msg = event.get("message")
    if isinstance(msg, dict):
        nested = msg.get("timestamp")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool) and math.isfinite(nested):
            value = int(nested)
            # Pi stores nested timestamps as epoch milliseconds.
            return value if value > 10**12 else value * 1000
        if isinstance(nested, str):
            return _iso_to_epoch_ms(nested)
    return None


def _pi_int(value: Any) -> int:
    """Coerce a Pi usage field to int; bools and non-numerics become 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return int(value)


def _pi_extract_usage(usage: Any) -> dict | None:
    """Map Pi usage {input, output, cacheRead, cacheWrite, reasoning, totalTokens}.

    Reasoning is a subset of output in native Pi usage. Older sessions may
    omit cache counts from ``totalTokens``; include those disjoint components
    without counting reasoning a second time.
    """
    if not isinstance(usage, dict):
        return None
    inp = _pi_int(usage.get("input", 0))
    out = _pi_int(usage.get("output", 0))
    reasoning = _pi_int(usage.get("reasoning", 0))
    cache_read = _pi_int(usage.get("cacheRead", 0))
    cache_write = _pi_int(usage.get("cacheWrite", 0))
    component_total = inp + out + cache_read + cache_write
    vendor_total = _pi_int(usage.get("totalTokens"))
    total = max(component_total, vendor_total)
    if not any((total, inp, out, reasoning, cache_read, cache_write)):
        return None
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache": {
            "read": cache_read,
            "write": cache_write,
        },
    }


def _pi_iter_content(content: Any):
    """Yield content dicts without treating a string as a character sequence."""
    if isinstance(content, str):
        if content:
            yield {"type": "text", "text": content}
        return
    if isinstance(content, dict):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, str):
            if item:
                yield {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield item


def _pi_content_text(content: Any) -> str:
    chunks: list[str] = []
    for item in _pi_iter_content(content):
        if item.get("type") == "text":
            chunks.append(str(item.get("text") or ""))
        elif item.get("type") == "image":
            chunks.append(f"[Image: {item.get('mimeType') or 'unknown type'}]")
    return "\n".join(chunks)


def _pi_text_parts(content: Any, *, synthetic: bool = False) -> list[dict]:
    parts = []
    for item in _pi_iter_content(content):
        if item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text", ""), "synthetic": synthetic})
        elif item.get("type") == "image":
            parts.append({"type": "text", "text": f"[Image: {item.get('mimeType') or 'unknown type'}]",
                          "synthetic": synthetic, "metadata": {"pi_image": {"mimeType": item.get("mimeType")}}})
    return parts


def _pi_message_created_ms(message: dict) -> int | None:
    value = message.get("timestamp")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return int(value) if value > 10**12 else int(value * 1000)
    if isinstance(value, str):
        return _iso_to_epoch_ms(value)
    return None


def _pi_normalize_tool(name: str, arguments: Any) -> tuple[str, dict]:
    raw_name = name or "?"
    canonical = _PI_TOOL_NAMES.get(str(raw_name).lower(), raw_name)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    args = dict(arguments) if isinstance(arguments, dict) else ({} if arguments is None else {"raw": arguments})
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args and "path" in args:
        args["file_path"] = args["path"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _pi_nested_tool_parts(nested: Any, parent_id: str) -> list[dict]:
    """Expose recorded nested calls without inventing result text or windows."""
    if not isinstance(nested, dict) or not isinstance(nested.get("calls"), list):
        return []
    parts = []
    for call in nested["calls"]:
        if not isinstance(call, dict):
            continue
        name, arguments = _pi_normalize_tool(call.get("name") or "?", call.get("arguments"))
        recorded_status = call.get("status")
        status = {"ok": "success", "error": "error"}.get(recorded_status, "unknown")
        metadata = {"pi_nested_parent_id": parent_id, "pi_nested_call": call,
                    "_outcome_observed": recorded_status in {"ok", "error"},
                    "_input_observed": isinstance(call.get("arguments"), (dict, str)),
                    "_output_observed": False}
        duration = call.get("durationMs")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
            metadata["totalDurationMs"] = duration
        parts.append({"type": "tool_call", "tool_name": name, "tool_id": call.get("id") or "",
                      "state": {"status": status, "input": arguments, "output": "",
                                "error": call.get("error"), "metadata": metadata}})
    return parts


def _convert_pi_to_internal(events: list[dict]) -> dict:
    """Convert Pi coding-agent JSONL events into the TrajViz internal format.

    Pi emits newline-delimited JSON with event types:
    - session: session ID, cwd, version
    - model_change / thinking_level_change: metadata (not steps)
    - message: nested {role: user|assistant|toolResult, content, usage, ...}

    Assistant ``toolCall`` parts are paired with later ``toolResult`` messages
    by ``toolCallId``.  The result matches the OpenCode internal message shape
    used by parse_steps().
    """
    session: dict = {}
    messages: list[dict] = []
    pending_tools: dict[str, dict] = {}
    tools_by_entry: dict[str, dict[str, dict]] = {}
    tool_owner_parts: dict[int, list[dict]] = {}
    entries_by_id = {event["id"]: event for event in events
                     if isinstance(event, dict) and isinstance(event.get("id"), str)}
    current_model = ""
    current_provider = ""
    title = ""

    def _append(role: str, ts: int | None, parts: list, tokens: dict | None = None, extra: dict | None = None) -> None:
        info: dict[str, Any] = {
            "role": role,
            "time": {"created": ts or 0},
            "sessionID": session.get("id", ""),
            "path": {"cwd": session.get("cwd", "")},
        }
        if tokens:
            info["tokens"] = tokens
        if extra:
            info.update(extra)
        messages.append({"info": info, "parts": parts})

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        ts = _pi_event_ts_ms(event)

        if etype == "session":
            session = event
            continue
        if etype == "model_change":
            if event.get("provider"):
                current_provider = str(event.get("provider"))
            if event.get("modelId"):
                current_model = str(event.get("modelId"))
            continue
        if etype in ("thinking_level_change",):
            continue
        if etype == "session_info":
            if isinstance(event.get("name"), str):
                title = event["name"]
            continue
        if etype == "custom_message":
            parts = _pi_text_parts(event.get("content"), synthetic=True)
            if parts:
                parts[0]["metadata"] = {**parts[0].get("metadata", {}),
                                        "pi_custom": {"customType": event.get("customType"),
                                                      "display": event.get("display"), "details": event.get("details")}}
                _append("user", ts, parts, extra={"id": event.get("id") or "",
                                               "parentID": event.get("parentId") or "", "type": etype})
            continue
        if etype in {"usage", "compaction", "branch_summary"}:
            tokens = _pi_extract_usage(event.get("usage"))
            extra = {
                "id": event.get("id") or "", "parentID": event.get("parentId") or "",
                "modelID": event.get("model") or current_model,
                "providerID": event.get("provider") or current_provider,
                "type": etype,
            }
            if etype == "compaction":
                before = event.get("tokensBefore")
                if isinstance(before, (int, float)) and not isinstance(before, bool) and math.isfinite(before) and before >= 0:
                    extra["compaction_metadata"] = {"preTokens": int(before)}
                _append("compaction", ts, [{"type": "compaction", "summary": event.get("summary") or ""}],
                        tokens=tokens, extra=extra)
            elif etype == "branch_summary":
                _append("assistant", ts, [{"type": "text", "text": event.get("summary") or "", "synthetic": True}],
                        tokens=tokens, extra=extra)
            elif tokens:
                _append("assistant", ts, [{"type": "step_finish", "name": "usage", "reason": event.get("kind") or ""}],
                        tokens=tokens, extra=extra)
            continue
        if etype != "message":
            continue

        msg = event.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        created = _pi_message_created_ms(msg)

        if role in {"user", "custom", "hookMessage", "system"}:
            synthetic = role in {"custom", "hookMessage"}
            parts = _pi_text_parts(msg.get("content"), synthetic=synthetic)
            if role == "system":
                sections = msg.get("sections")
                if isinstance(sections, dict):
                    for name, content in sections.items():
                        text = f"[Prompt section removed: {name}]" if content is None else f"[{name}]\n{content}"
                        parts.append({"type": "text", "text": text})
                for field, label in (("toolsAdded", "Tools added"), ("toolsRemoved", "Tools removed")):
                    tools = msg.get(field)
                    if isinstance(tools, list) and tools:
                        names = ", ".join(str(tool.get("name") or "?") for tool in tools if isinstance(tool, dict))
                        parts.append({"type": "text", "text": f"[{label}: {names}]",
                                      "metadata": {f"pi_{field}": tools}})
            elif synthetic and parts:
                parts[0]["metadata"] = {**parts[0].get("metadata", {}),
                                        "pi_custom": {"customType": msg.get("customType"),
                                                      "display": msg.get("display"), "details": msg.get("details")}}
            if parts:
                _append("system" if role == "system" else "user", created if created is not None else ts, parts,
                        extra={"id": event.get("id") or "", "parentID": event.get("parentId") or "",
                               "type": "custom_message" if synthetic else role})

        elif role == "bashExecution":
            exit_code = msg.get("exitCode")
            has_exit = isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and math.isfinite(exit_code)
            cancelled = msg.get("cancelled") is True
            failed = cancelled or (has_exit and exit_code != 0)
            error = "Command cancelled" if cancelled else f"exit {exit_code}" if failed else None
            metadata = {"exit": exit_code if has_exit else None, "cancelled": cancelled,
                        "truncated": msg.get("truncated") is True, "fullOutputPath": msg.get("fullOutputPath"),
                        "excludeFromContext": msg.get("excludeFromContext") is True,
                        "_outcome_observed": cancelled or has_exit, "pi_user_command": True}
            _append("system", created if created is not None else ts,
                    [{"type": "tool", "tool_name": "Bash", "tool_id": event.get("id") or "",
                      "state": {"status": "error" if failed else "success" if has_exit else "unknown",
                                "input": {"command": msg.get("command") or ""}, "output": msg.get("output") or "",
                                "error": error, "metadata": metadata}}],
                    extra={"id": event.get("id") or "", "parentID": event.get("parentId") or "",
                           "type": "bash_execution"})

        elif role == "assistant":
            if msg.get("model"):
                current_model = str(msg.get("model"))
            if msg.get("provider"):
                current_provider = str(msg.get("provider"))
            parts = []
            for item in _pi_iter_content(msg.get("content")):
                ctype = item.get("type")
                if ctype == "thinking":
                    text = item.get("thinking") or item.get("text") or ""
                    parts.append({"type": "reasoning", "text": text})
                elif ctype == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif ctype == "toolCall":
                    tool_name, tool_input = _pi_normalize_tool(
                        item.get("name", "?"),
                        item.get("arguments"),
                    )
                    call_id = item.get("id") or ""
                    part = {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_id": call_id,
                        "status": "pending",
                        "input": tool_input,
                        "output": "",
                    }
                    parts.append(part)
                    tool_owner_parts[id(part)] = parts
                    if call_id:
                        pending_tools[call_id] = part
                        entry_id = event.get("id")
                        if isinstance(entry_id, str):
                            tools_by_entry.setdefault(entry_id, {})[call_id] = part
            error_message = msg.get("errorMessage")
            if error_message:
                parts.append({"type": "text", "text": f"[error] {error_message}"})
            tokens = _pi_extract_usage(msg.get("usage"))
            extra = {
                "id": event.get("id") or "",
                "modelID": current_model,
                "providerID": current_provider,
                "parentID": event.get("parentId") or "",
                "timing_source": "model_request",
            }
            stop = msg.get("stopReason")
            if stop:
                extra["finish"] = stop
            if not parts and stop in {"error", "aborted", "deferred", "pending"}:
                parts = [{"type": "text", "text": f"[Response {stop}]"}]
            if parts or tokens:
                _append("assistant", created if created is not None else ts, parts, tokens=tokens, extra=extra)
                # The outer timestamp is stamped by SessionManager when the
                # settled message is persisted; nested timestamp starts it.
                # A later message may be another branch or follow human idle.
                duration = msg.get("durationMs")
                if (created is not None and isinstance(duration, (int, float)) and not isinstance(duration, bool)
                        and math.isfinite(duration) and duration >= 0):
                    messages[-1]["info"]["time"]["completed"] = created + duration
                else:
                    persisted = _iso_to_epoch_ms(event.get("timestamp") if isinstance(event.get("timestamp"), str) else None)
                    if created is not None and persisted is not None and persisted >= created:
                        messages[-1]["info"]["time"]["completed"] = persisted

        elif role == "toolResult":
            call_id = msg.get("toolCallId") or ""
            output = _pi_content_text(msg.get("content"))
            is_error = bool(msg.get("isError"))
            status = "error" if is_error else "success"
            result_metadata = {"pi_result": {key: msg[key] for key in ("details", "usage", "nestedCalls") if key in msg}}
            result_state = {"status": status, "output": output, "metadata": result_metadata}
            duration = msg.get("durationMs")
            if (isinstance(duration, (int, float)) and not isinstance(duration, bool)
                    and math.isfinite(duration) and duration >= 0):
                result_metadata["totalDurationMs"] = duration
                end = created if created is not None else ts
                if end is not None:
                    result_state["time"] = {"start": end - duration, "end": end}
            part = None
            parent_id = event.get("parentId")
            seen = set()
            while call_id and isinstance(parent_id, str) and parent_id and parent_id not in seen:
                seen.add(parent_id)
                part = tools_by_entry.get(parent_id, {}).get(call_id)
                if part is not None:
                    break
                parent_id = (entries_by_id.get(parent_id) or {}).get("parentId")
            if part is None and ("parentId" not in event or not event.get("id")):
                part = pending_tools.get(call_id)
            if part is not None and pending_tools.get(call_id) is part:
                pending_tools.pop(call_id, None)
            if part is not None:
                part["output"] = output
                part["status"] = status
                if is_error:
                    part["error"] = output
                state = part.setdefault("state", {})
                state.update(result_state, input=part["input"])
                tool_owner_parts[id(part)].extend(_pi_nested_tool_parts(msg.get("nestedCalls"), call_id))
            else:
                tool_name, _ = _pi_normalize_tool(msg.get("toolName") or "?", {})
                _append(
                    "assistant",
                    ts,
                    [
                        {
                            "type": "tool",
                            "tool_name": tool_name,
                            "tool_id": call_id,
                            "status": status,
                            "input": {},
                            "output": output,
                            **({"error": output} if is_error else {}),
                            "state": result_state,
                        }
                    ],
                    extra={"id": event.get("id") or "", "parentID": event.get("parentId") or "",
                           "type": "tool_result"},
                )
                messages[-1]["parts"].extend(_pi_nested_tool_parts(msg.get("nestedCalls"), call_id))
            tokens = _pi_extract_usage(msg.get("usage"))
            if tokens:
                _append("tool", created if created is not None else ts,
                        [{"type": "step_finish", "name": "tool_usage", "reason": msg.get("toolName") or ""}],
                        tokens=tokens, extra={"id": f"{event.get('id') or call_id}:usage",
                                              "parentID": event.get("id") or "", "type": "usage"})

    for message in messages:
        for part in message["parts"]:
            if part.get("type") == "tool_call" and part.get("status") == "pending":
                part["status"] = "unknown"
                state = part.setdefault("state", {})
                state.update(status="unknown", input=part.get("input", {}), output=part.get("output", ""))
                state.setdefault("metadata", {}).update(_outcome_observed=False, _output_observed=False)

    first_ts_iso = None
    last_ts_iso = None
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            first_ts_iso = event["timestamp"]
            break
    for event in reversed(events):
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            last_ts_iso = event["timestamp"]
            break
    directory = session.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3) if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum((m["info"].get("tokens") or {}).get("total", 0) for m in messages)
    last_model = current_model
    last_provider = current_provider

    info = {
        "id": session.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": title,
        "version": str(session.get("version", "") or ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session.get("id", ""),
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
            "agent": "pi",
            "model": last_model,
            "source": "pi",
            "model_provider": last_provider,
            "originator": "Pi",
            "server_version": str(session.get("version", "") or ""),
            "timestamp_utc": first_ts_iso or "",
            "title": title,
            "parent_session_path": session.get("parentSession") or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": first_ts_iso or "",
            "finished_at": last_ts_iso or "",
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_pi_format": True,
        # Parent links, context edits, prompt checkpoints and extension data
        # remain inspectable without rewriting historical branches.
        "_pi_entries": events,
    }
