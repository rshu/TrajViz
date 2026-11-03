"""Small native Claude Code event builders shared by regression tests.

The flat entries keep individual assertions readable; only native records are
written to disk. The reference ATTRIBUTION files remain byte-for-byte unchanged.
"""

from __future__ import annotations

import copy


def native_events(entries: list[dict], *, session_id: str = "test-session", cwd: str = "/p") -> list[dict]:
    events = []
    for i, entry in enumerate(entries):
        entry = copy.deepcopy(entry)
        role = entry.get("role", entry.get("type", "user"))
        event = {"type": role, "sessionId": session_id, "uuid": entry.get("uuid", f"event-{i}"),
                 "parentUuid": entry.get("parent_uuid"), "cwd": entry.get("cwd", cwd),
                 "isSidechain": entry.get("is_sidechain", False)}
        if "timestamp" in entry:
            event["timestamp"] = entry["timestamp"]
        if role in ("user", "assistant"):
            content = entry.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_call":
                        block["type"] = "tool_use"
                    elif block.get("type") == "thinking" and "text" in block:
                        block["thinking"] = block.pop("text")
                    elif block.get("type") == "tool_result":
                        if "tool_call_id" in block:
                            block["tool_use_id"] = block.pop("tool_call_id")
                        if "output" in block:
                            block["content"] = block.pop("output")
            event["message"] = {"role": role, "content": content}
            for old, new in (("message_id", "id"), ("model", "model"), ("usage", "usage"),
                             ("stop_reason", "stop_reason")):
                if old in entry:
                    event["message"][new] = entry[old]
            if "tool_execution" in entry:
                event["toolUseResult"] = entry["tool_execution"]
        else:
            event["type"] = entry.get("event_type", role)
            for key in ("subtype", "data"):
                if key in entry:
                    event[key] = entry[key]
        events.append(event)
    return events


def empty_session() -> dict:
    return {"type": "system", "subtype": "init", "sessionId": "test-session"}
