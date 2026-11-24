"""Claude Code's native session JSONL records → internal trajectory.

Session files contain transcript messages alongside bookkeeping records. Child
transcripts are supplied by the loader; inline ``agent_progress`` messages are a
fallback when the corresponding child file is unavailable. No session-export
package or nonstandard envelope is required.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

from .common import _iso_to_epoch_ms, as_text
from .sniff import _looks_like_claude_code_events


def _cc_token_value(value: object) -> int | float:
    """Accept finite reported token counts without coercing malformed values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value if math.isfinite(value) else 0


def _cc_extract_usage(usage: dict | None) -> dict:
    """Anthropic input, cache reads and cache creation are disjoint counts.

    Thinking tokens are included in output, but their count is not separately
    reported. Omit ``reasoning`` instead of inventing a measured zero.
    """
    usage = usage if isinstance(usage, dict) else {}
    inp = _cc_token_value(usage.get("input_tokens"))
    out = _cc_token_value(usage.get("output_tokens"))
    read = _cc_token_value(usage.get("cache_read_input_tokens"))
    write = _cc_token_value(usage.get("cache_creation_input_tokens"))
    return {"total": inp + out + read + write, "input": inp, "output": out,
            "cache": {"read": read, "write": write}}


def _cc_build_step(parts: list, *, role: str, usage: dict | None = None,
                   timestamp_ms: int | None = None, model: str = "",
                   finish: str = "", agent: str = "", message_id: str = "",
                   step_id: str = "", parent_id: str = "", cwd: str = "",
                   tool_calls: list | None = None,
                   error_count: int | None = None,
                   has_reasoning: bool | None = None,
                   text_preview: str = "", session_id: str = "",
                   source_path: str = "") -> dict:
    """Build a normalized step; explicit flags distinguish main and child."""
    calls = tool_calls if tool_calls is not None else [
        p for p in parts if p.get("type") == "tool_call"]
    usage = usage if usage is not None else _cc_extract_usage(None)
    return {
        "role": role,
        "tokens": {"total": usage["total"], "input": usage["input"],
                   "output": usage["output"], "cache_read": usage["cache"]["read"],
                   "cache_write": usage["cache"]["write"]},
        "duration": None, "parts": parts, "tool_calls": calls,
        "tool_call_count": len(calls),
        "error_count": (sum(c.get("status") == "error" for c in calls)
                        if error_count is None else error_count),
        "has_reasoning": (any(p.get("type") == "reasoning" for p in parts)
                          if has_reasoning is None else has_reasoning),
        "text_preview": text_preview, "finish": finish or "", "model_id": model,
        "provider_id": "anthropic", "time_created_ms": timestamp_ms,
        "time_completed_ms": None, "agent": agent, "is_sub_agent": bool(agent),
        "mode": "", "message_id": message_id, "id": step_id,
        "parent_id": parent_id, "session_id": session_id, "cwd": cwd, "root": "",
        "source_path": source_path,
        "parent_session_id": "", "parent_agent_id": "", "session_depth": 0,
        "session_title": "", "message_type": "", "is_compaction_checkpoint": False,
        "compaction_reason": "", "summary": False, "_metrics_unavailable_fields": [],
    }


def _content_list(content: object) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []


def _signature(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _message(record: dict) -> dict:
    message = record.get("message")
    return message if isinstance(message, dict) else {}


def _is_interruption_marker(record: dict) -> bool:
    """Claude's interruption control message is not a new user request."""
    if record.get("type") != "user":
        return False
    content = _content_list(_message(record).get("content"))
    return (len(content) == 1 and content[0].get("type") == "text"
            and content[0].get("text") in {
                "[Request interrupted by user]", "[Request interrupted by user for tool use]",
            })


def _output_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            as_text(item.get("text")) if isinstance(item, dict) and item.get("type") == "text"
            else _signature(item) for item in content)
    return "" if content is None else _signature(content)


def _timestamp(record: dict) -> int | None:
    return _iso_to_epoch_ms(record.get("timestamp"))


def _ordered(records: list[dict]) -> list[dict]:
    """Sort timestamped records stably; untimed records retain nearby order."""
    carried = None
    ordered = []
    for index, record in enumerate(records):
        ts = _timestamp(record)
        if ts is not None:
            carried = ts
        ordered.append((carried if carried is not None else float("-inf"), index, record))
    return [record for _, _, record in sorted(ordered, key=lambda item: item[:2])]


def _merge_messages(records: list[dict]) -> list[dict]:
    """Merge split assistant blocks and discard repeated transcript records.

    The usage on each block describes the same response, often cumulatively.
    Only the last usage snapshot is counted, and repeated blocks are not added
    twice when a transcript contains snapshots of the same message.
    """
    groups: dict[tuple, dict] = {}
    uuid_groups: dict[str, tuple] = {}
    seen_records: set[str] = set()
    merged = []
    for raw_index, record in enumerate(_ordered(records)):
        kind = record.get("type")
        if kind == "system" and record.get("subtype") == "compact_boundary":
            key = as_text(record.get("uuid")) or _signature(record)
            if key not in seen_records:
                seen_records.add(key)
                merged.append({"record": record, "content": [],
                               "raw_index": record.get("_source_index", raw_index),
                               "first_ms": _timestamp(record), "last_ms": _timestamp(record),
                               "block_times": {}, "compaction": True})
            continue
        if kind not in ("user", "assistant") or not isinstance(record.get("message"), dict):
            continue
        msg = _message(record)
        role = as_text(msg.get("role")) or kind
        if role not in ("user", "assistant"):
            continue
        # UUIDs identify transcript entries; message.id identifies a whole API
        # response that Claude Code may persist one content block at a time.
        mid = as_text(msg.get("id"))
        uuid = as_text(record.get("uuid"))
        key = (role, mid or uuid or ("line", raw_index))
        if uuid:
            # Replayed UUIDs identify the same native entry, even if a damaged
            # replay changes its message ID. Do not turn it into a second step.
            key = uuid_groups.setdefault(uuid, key)
        if key not in groups:
            entry = {"record": dict(record), "content": [],
                     "raw_index": record.get("_source_index", raw_index),
                     "first_ms": _timestamp(record), "last_ms": _timestamp(record),
                     "block_times": {}, "pieces": {}, "role": role,
                     "usage": None, "finish": "", "model": ""}
            groups[key] = entry
            merged.append(entry)
        entry = groups[key]
        ts = _timestamp(record)
        if ts is not None:
            entry["first_ms"] = min(entry["first_ms"], ts) if entry["first_ms"] is not None else ts
            entry["last_ms"] = max(entry["last_ms"], ts) if entry["last_ms"] is not None else ts
        content = _content_list(msg.get("content"))
        piece_id = uuid or (("inline", mid, _signature(content))
                           if record.get("_inline_progress") and mid else ("line", raw_index))
        entry["pieces"][piece_id] = content
        for block in entry["pieces"][piece_id]:
            tid = as_text(block.get("id"))
            if tid and tid not in entry["block_times"]:
                entry["block_times"][tid] = ts
        if isinstance(msg.get("usage"), dict):
            entry["usage"] = msg["usage"]
        entry["finish"] = as_text(msg.get("stop_reason")) or entry["finish"]
        entry["model"] = as_text(msg.get("model")) or entry["model"]
    for entry in merged:
        if entry.get("compaction"):
            continue
        content = []
        tool_positions = {}
        for blocks in entry.pop("pieces").values():
            # A full-message snapshot can follow earlier single-block records.
            # A matching prefix is a replay, while identical text blocks from
            # distinct UUIDs remain distinct legitimate content blocks.
            if (len(blocks) > 1 and content and len(blocks) >= len(content)
                    and blocks[:len(content)] == content):
                content = []
                tool_positions = {}
            for block in blocks:
                tool_id = as_text(block.get("id")) if block.get("type") == "tool_use" else ""
                if tool_id and tool_id in tool_positions:
                    content[tool_positions[tool_id]] = block
                else:
                    if tool_id:
                        tool_positions[tool_id] = len(content)
                    content.append(block)
        entry["content"] = content
    return merged


def _tool_results(records: list[dict]) -> dict[str, dict]:
    results = {}
    for record in _ordered(records):
        if record.get("type") != "user":
            continue
        metadata = record.get("toolUseResult")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        for item in _content_list(_message(record).get("content")):
            if item.get("type") != "tool_result":
                continue
            tool_id = as_text(item.get("tool_use_id"))
            if not tool_id:
                continue
            output = item.get("content", "")
            results[tool_id] = {"output": output,
                                "status": "error" if item.get("is_error") else "success",
                                "error": _output_text(output) if item.get("is_error") else None,
                                "time_end": _timestamp(record), "metadata": metadata}
    return results


def _parts(entry: dict, results: dict[str, dict]) -> list[dict]:
    parts = []
    for item in entry["content"]:
        kind = item.get("type")
        if kind == "tool_result":
            continue
        if kind == "text":
            parts.append({"type": "text", "text": as_text(item.get("text"))})
        elif kind == "thinking":
            parts.append({"type": "reasoning", "text": as_text(item.get("thinking"))})
        elif kind == "redacted_thinking":
            parts.append({"type": "reasoning", "text": "", "redacted": True, "raw": item})
        elif kind == "tool_use":
            tool_id = as_text(item.get("id"))
            result = results.get(tool_id, {})
            metadata = dict(result.get("metadata", {}))
            start = entry["block_times"].get(tool_id)
            end = result.get("time_end")
            valid_interval = start is not None and end is not None and end >= start
            reported_duration = next((value for key in ("totalDurationMs", "durationMs")
                                      if isinstance(value := metadata.get(key), (int, float))
                                      and not isinstance(value, bool) and math.isfinite(value) and value >= 0), None)
            if reported_duration is not None:
                metadata["timing_source"] = "tool_result"
            elif valid_interval:
                metadata["timing_source"] = "transcript"
            parts.append({
                "type": "tool_call", "tool_name": as_text(item.get("name")) or "?",
                "tool_id": tool_id, "status": result.get("status", "unknown"),
                "title": "", "input": item.get("input", {}),
                "output": result.get("output", ""), "error": result.get("error"),
                "time_start": start, "time_end": end,
                "duration_ms": (reported_duration if reported_duration is not None
                                else end - start if valid_interval else None),
                "metadata": metadata,
            })
        else:
            parts.append({"type": as_text(kind), "raw": item})
    return parts


def convert_claude_code_events(events: list[dict], *, source_path: str | None = None,
                               child_event_lists: dict | None = None,
                               child_warnings: list[str] | None = None) -> dict:
    """Convert a native parent transcript and its optional child transcripts.

    ``child_event_lists`` maps an agent ID to records, or to a bundle containing
    ``events``, ``source_path``, ``parent_agent_id`` and ``spawned_by``. The loader
    owns filesystem discovery; conversion never opens paths or scans sessions.
    """
    if not isinstance(events, list) or not _looks_like_claude_code_events(events):
        return {"_error": "Claude Code expects native session JSONL records with nested message objects.",
                "_error_code": "unknown_format"}

    main_records = [record for record in events if isinstance(record, dict)]
    root_sid = next((as_text(r.get("sessionId")) for r in main_records if r.get("sessionId")), "")
    source_name = os.path.basename(source_path or "")
    root_agent = source_name[6:-6] if source_name.startswith("agent-") and source_name.endswith(".jsonl") else ""
    actual: dict[str, list[dict]] = {"": []}
    inline: dict[str, list[dict]] = {}
    info: dict[str, dict] = {"": {"source_path": source_path or ""}}
    control_events: list[dict] = []
    prompt_records: list[dict] = []

    def note_agent(agent_id: str, *, parent: str = "", spawned_by: str = "", **fields):
        if not agent_id:
            return
        agent_info = info.setdefault(agent_id, {})
        if parent or "parent_agent_id" not in agent_info:
            agent_info["parent_agent_id"] = parent
        if spawned_by:
            agent_info.setdefault("spawned_by", spawned_by)
        for name, value in fields.items():
            if value not in (None, ""):
                agent_info.setdefault(name, value)

    def scan(records: list, owner: str, target: dict[str, list[dict]], depth: int = 0,
             stream_path: str = "", source_index: int | None = None):
        if depth > 32:
            return
        for record_index, original in enumerate(records):
            if not isinstance(original, dict):
                continue
            record = dict(original)
            record["_source_index"] = record_index if source_index is None else source_index
            record["_source_path"] = stream_path
            own_agent = as_text(record.get("agentId")) or owner
            if own_agent:
                note_agent(own_agent, parent=owner if own_agent != owner else "")
            if _is_interruption_marker(record):
                control_events.append({"kind": "request_interrupted", "agent_id": own_agent,
                                       "timestamp": record.get("timestamp"),
                                       "uuid": record.get("uuid"), "source_path": stream_path,
                                       "source_index": record["_source_index"]})
                continue
            kind = record.get("type")
            if kind in ("api-request-shape", "api-request-blob", "api-request"):
                # These describe requests, not additional conversation turns.
                # Keep their native ordering and owner, independently of the
                # timestamp-sorted transcript used for charts and accounting.
                from ..prompt_claude import preserve_prompt_record

                prompt_records.append(preserve_prompt_record(
                    original, source_path=stream_path, raw_index=record["_source_index"],
                    session_id=as_text(record.get("sessionId")) or root_sid, agent_id=own_agent,
                ))
            if kind in ("user", "assistant", "system"):
                target.setdefault(own_agent, []).append(record)
            # Parent tool-result records often carry the child's native ID even
            # when no progress events were retained.
            result = record.get("toolUseResult")
            if isinstance(result, dict) and result.get("agentId"):
                child = as_text(result.get("agentId"))
                result_blocks = [b for b in _content_list(_message(record).get("content"))
                                 if b.get("type") == "tool_result"]
                spawned = as_text(result_blocks[0].get("tool_use_id")) if len(result_blocks) == 1 else ""
                note_agent(child, parent=own_agent, spawned_by=spawned,
                           description=result.get("description"), native_summary=result)
            if kind != "progress":
                continue
            data = record.get("data")
            if not isinstance(data, dict):
                continue
            child = as_text(data.get("agentId"))
            nested = data.get("message")
            if not child:
                continue
            note_agent(child, parent=own_agent,
                       spawned_by=as_text(record.get("parentToolUseID")) or as_text(data.get("parentToolUseID")),
                       description=data.get("description"))
            if not isinstance(nested, dict):
                continue
            if nested.get("role") in ("user", "assistant") and "message" not in nested:
                nested = {"type": nested["role"], "message": nested}
            nested = dict(nested)
            nested.setdefault("timestamp", record.get("timestamp"))
            nested.setdefault("sessionId", record.get("sessionId") or root_sid)
            nested.setdefault("cwd", record.get("cwd"))
            nested.setdefault("agentId", child)
            nested["_inline_progress"] = True
            scan([nested], child, inline, depth + 1, stream_path, record["_source_index"])

    scan(events, root_agent, actual, stream_path=source_path or "")
    for child, bundle in (child_event_lists or {}).items():
        child = as_text(child)
        if not child:
            continue
        fields = bundle if isinstance(bundle, dict) else {}
        records = fields.get("events", []) if fields else bundle
        if not isinstance(records, list):
            continue
        note_agent(child, parent=as_text(fields.get("parent_agent_id")),
                   spawned_by=as_text(fields.get("spawned_by")),
                   source_path=fields.get("source_path"),
                   parent_session_id=fields.get("parent_session_id"))
        scan(records, child, actual, stream_path=as_text(fields.get("source_path")))

    streams = dict(actual)
    warnings = list(child_warnings or [])
    for control in control_events:
        owner = f"sub-agent {control['agent_id']}" if control["agent_id"] else "the main agent"
        warnings.append(f"Claude Code recorded a request interruption for {owner}; its transcript may be incomplete.")
    for child, records in inline.items():
        # An actual transcript is authoritative. The same response can appear
        # there and in several progress snapshots with different wrapper UUIDs.
        if not any(r.get("type") in ("user", "assistant") for r in actual.get(child, [])):
            streams[child] = records
            warnings.append(f"Sub-agent {child} has no complete child log; loaded inline progress snapshots only.")
    for child in info:
        if child and not any(r.get("type") in ("user", "assistant") for r in streams.get(child, [])):
            warnings.append(f"Sub-agent {child} was referenced but its transcript is unavailable.")

    steps = []
    turn_durations = []
    for agent, records in streams.items():
        results = _tool_results(records)
        for record in records:
            if record.get("type") == "system" and record.get("subtype") == "turn_duration":
                duration = record.get("durationMs")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
                    turn_durations.append({"agent_id": agent, "duration_ms": duration,
                                           "timestamp": record.get("timestamp"),
                                           "parent_uuid": record.get("parentUuid")})
        for entry in _merge_messages(records):
            record = entry["record"]
            if entry.get("compaction"):
                compact = record.get("compactMetadata")
                compact = compact if isinstance(compact, dict) else {}
                text = as_text(record.get("content"))
                step = _cc_build_step([{"type": "text", "text": text}] if text else [],
                                      role="compaction", timestamp_ms=entry["first_ms"],
                                      agent=agent, step_id=as_text(record.get("uuid")),
                                      parent_id=as_text(record.get("parentUuid")),
                                      session_id=as_text(record.get("sessionId")) or root_sid,
                                      cwd=as_text(record.get("cwd")), text_preview=text or "Context compacted",
                                      source_path=as_text(record.get("_source_path")))
                step.update(message_type="compaction", is_compaction_checkpoint=True,
                            compaction_reason=as_text(compact.get("trigger")),
                            compaction_metadata=compact)
            else:
                parts = _parts(entry, results)
                if entry["role"] == "user" and not parts:
                    continue  # Tool results are attached to their invoking call.
                if not parts and not isinstance(entry.get("usage"), dict):
                    continue
                preview = next((p.get("text", "") for p in parts
                                if p.get("type") == "text" and p.get("text")), "")
                if not preview:
                    preview = next((p.get("text", "") for p in parts
                                    if p.get("type") == "reasoning" and p.get("text")), "")
                if not preview:
                    preview = next((f"[Tool: {p['tool_name']}]" for p in parts
                                    if p.get("type") == "tool_call"), "")
                step = _cc_build_step(parts, role=entry["role"], usage=_cc_extract_usage(entry["usage"]),
                                      timestamp_ms=entry["first_ms"], model=entry["model"],
                                      finish=entry["finish"], agent=agent,
                                      message_id=as_text(_message(record).get("id")) or as_text(record.get("uuid")),
                                      step_id=as_text(record.get("uuid")),
                                      parent_id=as_text(record.get("parentUuid")),
                                      cwd=as_text(record.get("cwd")), text_preview=preview,
                                      session_id=as_text(record.get("sessionId")) or root_sid,
                                      source_path=as_text(record.get("_source_path")))
                boundaries = [entry["last_ms"]] + [p.get("time_end") for p in step["tool_calls"]]
                finish_ms = max((ts for ts in boundaries if ts is not None), default=None)
                if finish_ms is not None and entry["first_ms"] is not None and finish_ms > entry["first_ms"]:
                    step["time_completed_ms"] = finish_ms
                    step["duration"] = (finish_ms - entry["first_ms"]) / 1000
                    step["timing_source"] = "transcript"
            step["raw_index"] = entry["raw_index"]
            if agent:
                agent_info = info.get(agent, {})
                step["parent_agent_id"] = as_text(agent_info.get("parent_agent_id"))
                step["parent_session_id"] = as_text(agent_info.get("parent_session_id")) or root_sid
                step["session_title"] = as_text(agent_info.get("description"))
            steps.append(step)

    def agent_depth(agent: str) -> int:
        seen = set()
        while agent and agent not in seen:
            seen.add(agent)
            agent = as_text(info.get(agent, {}).get("parent_agent_id"))
        return len(seen)

    # Agent timelines were converted independently. Stable global ordering is
    # for presentation only and never changes a step's measured interval.
    carried = {}
    for step in steps:
        agent = step["agent"]
        if step["time_created_ms"] is not None:
            carried[agent] = step["time_created_ms"]
        step["_sort_timestamp"] = carried.get(agent, float("-inf"))
        step["session_depth"] = agent_depth(agent)
    steps.sort(key=lambda step: step["_sort_timestamp"])
    for index, step in enumerate(steps):
        step.pop("_sort_timestamp", None)
        step["index"] = index

    all_records = [r for records in streams.values() for r in records]
    timed_records = [(ts, r) for r in all_records if (ts := _timestamp(r)) is not None]
    started = min(timed_records, key=lambda pair: pair[0]) if timed_records else None
    ended = max(timed_records, key=lambda pair: pair[0]) if timed_records else None
    cwd = next((as_text(r.get("cwd")) for r in main_records if r.get("cwd")), "")
    model = next((s["model_id"] for s in steps if s["model_id"]), None)
    main_prompt = next((s["text_preview"] for s in steps if s["role"] == "user" and not s["agent"]), "")
    assistant = [s for s in steps if s["role"] == "assistant"]
    tool_counts = Counter(c["tool_name"] for s in steps for c in s["tool_calls"])
    sub_agents = []
    for agent in dict.fromkeys(s["agent"] for s in steps if s["agent"]):
        own = [s for s in steps if s["agent"] == agent]
        own_records = streams.get(agent, [])
        timed = [(ts, r) for r in own_records if (ts := _timestamp(r)) is not None]
        start = min(timed, key=lambda pair: pair[0]) if timed else None
        end = max(timed, key=lambda pair: pair[0]) if timed else None
        sub_agents.append({
            **info.get(agent, {}), "agent_id": agent,
            "session_id": next((s["session_id"] for s in own if s["session_id"]), root_sid),
            "parent_session_id": as_text(info.get(agent, {}).get("parent_session_id")) or root_sid,
            "session_depth": agent_depth(agent),
            "started_at": start[1].get("timestamp", "") if start else "",
            "ended_at": end[1].get("timestamp", "") if end else "",
            "duration_seconds": (end[0] - start[0]) / 1000 if start and end else None,
            "spawned_by": as_text(info.get(agent, {}).get("spawned_by")),
            "turns": len(own), "tool_calls": sum(s["tool_call_count"] for s in own),
            "tool_calls_by_name": dict(Counter(c["tool_name"] for s in own for c in s["tool_calls"])),
            "tokens": {out: sum(s["tokens"][field] for s in own if s["role"] == "assistant")
                       for out, field in (("input", "input"), ("output", "output"),
                                          ("cache_read", "cache_read"), ("cache_creation", "cache_write"))},
            "source": "file" if any(r.get("type") in ("user", "assistant") for r in actual.get(agent, [])) else "inline",
        })
    return {
        "metadata": {
            "session_id": root_sid, "slug": next((as_text(r.get("slug")) for r in main_records if r.get("slug")), ""),
            "directory": cwd, "directory_name": os.path.basename(cwd),
            "agent": "claude-code", "model": model, "hostname": "", "platform": "",
            "python_version": "", "timestamp_utc": started[1].get("timestamp", "") if started else "",
            "branch": next((as_text(r.get("gitBranch")) for r in main_records if r.get("gitBranch")), ""),
            "ground_truth_patch": "", "baseline_commit": "", "sanitized": False,
            "server_url": "", "server_version": next((as_text(r.get("version")) for r in main_records if r.get("version")), ""),
            "sub_agent_count": len(sub_agents), "event_count": len(all_records),
        },
        "input": {"prompt": main_prompt, "prompt_length": len(main_prompt)},
        "output": {"patch": "", "patch_length": 0, "patch_lines": 0, "has_patch": False, "error": None},
        "timing": {"total_duration": (ended[0] - started[0]) / 1000 if started and ended else 0,
                   "started_at": started[1].get("timestamp", "") if started else "",
                   "finished_at": ended[1].get("timestamp", "") if ended else ""},
        "token_usage": {"total_tokens": sum(s["tokens"]["total"] for s in assistant),
                        "prompt_tokens": sum(s["tokens"]["input"] for s in assistant),
                        "completion_tokens": sum(s["tokens"]["output"] for s in assistant)},
        "stats": {"total_messages": len(steps), "user_messages": sum(s["role"] == "user" for s in steps),
                  "assistant_messages": len(assistant), "total_tool_calls": sum(tool_counts.values()),
                  "tool_call_breakdown": dict(tool_counts), "failed_tool_calls": sum(s["error_count"] for s in steps),
                  "reasoning_steps": sum(s["has_reasoning"] for s in steps)},
        "trajectory": [], "_claude_code_format": True, "_claude_code_parsed_steps": steps,
        "_claude_code_sub_agents": sub_agents, "_claude_code_turn_durations": turn_durations,
        "_claude_code_control_events": control_events,
        "_claude_prompt_records": prompt_records,
        "_load_warnings": list(dict.fromkeys(warnings)),
    }
