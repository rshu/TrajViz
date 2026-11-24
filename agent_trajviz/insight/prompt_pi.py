"""Replay Pi's recorded prompt state without claiming a provider request.

The v3 session format records prompt/tool patches and branch-relative context
edits. A generation's context ends *before* its assistant entry. Extensions and
provider adapters can still transform this state after persistence.
"""

from __future__ import annotations

import math
from typing import Any

from .formats.pi import _pi_event_ts_ms

_LIMITATION = "Pi extensions and provider conversion can change the final request after recorded prompt state."
_MAX_ENTRIES = 100_000
_MAX_REQUESTS = 10_000
_EVENT_LABELS = {
    "system_section_added": "System section added",
    "system_section_updated": "System section updated",
    "system_section_removed": "System section removed",
    "tool_added": "Tool declaration added",
    "tool_updated": "Tool declaration updated",
    "tool_removed": "Tool declaration removed",
    "context_edit": "Context edit recorded",
    "compaction": "Compaction recorded",
    "branch_summary": "Branch summary recorded",
    "user_messages": "User messages recorded",
    "tool_results": "Tool results recorded",
    "context_message": "Injected context recorded",
}


def _safe_copy(value: Any, limitations: list[str], depth: int = 0) -> Any:
    """Keep native JSON data while bounding malformed/deep source values."""
    if depth > 64:
        limitations.append("Pi content exceeds supported nesting; deeper values are unavailable.")
        return None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_safe_copy(item, limitations, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _safe_copy(item, limitations, depth + 1) for key, item in value.items()}
    limitations.append("Non-finite or unsupported Pi content was replaced with null for safe JSON export.")
    return None


def _ref(index: int, entry: dict, source_path: str) -> dict:
    return {"source_path": source_path, "raw_index": index,
            "record_id": str(entry.get("id") or ""), "type": str(entry.get("type") or "")}


def _event(kind: str, index: int, entry: dict, source_path: str, target_id: str = "", **extra) -> dict:
    return {"kind": kind, "label": _EVENT_LABELS[kind], "source": _ref(index, entry, source_path),
            "target_id": target_id, "count": 1, **extra}


def _state_events(index: int, entry: dict, source_path: str, sections: dict, tools: dict) -> list[dict]:
    """Record compact patch annotations without making request snapshots."""
    kind = entry.get("type")
    message = entry.get("message") if kind == "message" else None
    events = []
    if kind == "compaction":
        events.append(_event("compaction", index, entry, source_path,
                             str(entry.get("firstKeptEntryId") or "")))
        checkpoint = entry.get("systemMessage")
        # A checkpoint is complete state, not a patch over pre-compaction
        # sections. Do not infer specific removals when it was not captured.
        if not isinstance(checkpoint, dict) or checkpoint.get("role") != "system":
            sections.clear()
            tools.clear()
            return events
        checkpoint_sections: dict[str, Any] = {}
        if isinstance(checkpoint.get("content"), str) and checkpoint["content"]:
            checkpoint_sections["content"] = checkpoint["content"]
        if isinstance(checkpoint.get("sections"), dict):
            checkpoint_sections.update({key: value for key, value in checkpoint["sections"].items()
                                        if isinstance(value, str)})
        for key in sections:
            if key not in checkpoint_sections:
                events.append(_event("system_section_removed", index, entry, source_path, str(key)))
        for key, value in checkpoint_sections.items():
            if key not in sections or sections[key] != value:
                change = "system_section_updated" if key in sections else "system_section_added"
                events.append(_event(change, index, entry, source_path, str(key)))
        sections.clear()
        sections.update(checkpoint_sections)
        declarations = checkpoint.get("toolsAdded")
        if isinstance(declarations, list):
            checkpoint_tools = {d["name"]: _safe_copy(d, []) for d in declarations
                                if isinstance(d, dict) and isinstance(d.get("name"), str) and d["name"]}
            for name in tools:
                if name not in checkpoint_tools:
                    events.append(_event("tool_removed", index, entry, source_path, name))
            for name, declaration in checkpoint_tools.items():
                if name not in tools or tools[name] != declaration:
                    change = "tool_updated" if name in tools else "tool_added"
                    events.append(_event(change, index, entry, source_path, name))
            tools.clear()
            tools.update(checkpoint_tools)
        else:
            tools.clear()
        return events
    if kind == "context_edit":
        target = entry.get("targetId")
        return [_event("context_edit", index, entry, source_path,
                       target if isinstance(target, str) else "",
                       action="omit" if entry.get("replacement") is None else "replace")]
    if kind == "branch_summary":
        return [_event("branch_summary", index, entry, source_path, str(entry.get("fromId") or ""))]
    if kind == "custom_message":
        return [_event("context_message", index, entry, source_path, str(entry.get("id") or ""))]
    if not isinstance(message, dict):
        return []
    role = message.get("role")
    if role == "system":
        updates = dict(message["sections"]) if isinstance(message.get("sections"), dict) else {}
        if isinstance(message.get("content"), str) and message["content"]:
            updates = {"content": message["content"], **updates}
        for name, value in updates.items():
            if value is None:
                sections.pop(name, None)
                events.append(_event("system_section_removed", index, entry, source_path, str(name)))
            elif isinstance(value, str) and (name not in sections or sections[name] != value):
                change = "system_section_updated" if name in sections else "system_section_added"
                sections[name] = value
                events.append(_event(change, index, entry, source_path, str(name)))
        removed = message.get("toolsRemoved")
        for tool in removed if isinstance(removed, list) else []:
            name = tool.get("name") if isinstance(tool, dict) else tool
            if isinstance(name, str) and name:
                tools.pop(name, None)
                events.append(_event("tool_removed", index, entry, source_path, name))
        added = message.get("toolsAdded")
        for declaration in added if isinstance(added, list) else []:
            name = declaration.get("name") if isinstance(declaration, dict) else None
            recorded = _safe_copy(declaration, []) if isinstance(declaration, dict) else None
            if isinstance(name, str) and name and (name not in tools or tools[name] != recorded):
                change = "tool_updated" if name in tools else "tool_added"
                tools[name] = recorded
                events.append(_event(change, index, entry, source_path, name))
    elif role == "user":
        events.append(_event("user_messages", index, entry, source_path))
    elif role == "toolResult":
        events.append(_event("tool_results", index, entry, source_path))
    elif role in {"custom", "hookMessage"} or role == "bashExecution" and message.get("excludeFromContext") is not True:
        events.append(_event("context_message", index, entry, source_path, str(entry.get("id") or "")))
    return events


def _events(entries: list) -> tuple[list[tuple[int, dict]], list[str]]:
    """Prefer persisted entries to duplicate progress notifications."""
    streamed = any(isinstance(e, dict) and e.get("type") in {"message_end", "message_start", "entry_appended"}
                   for e in entries)
    appended = any(isinstance(e, dict) and e.get("type") == "entry_appended" for e in entries)
    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        if kind == "entry_appended" and isinstance(entry.get("entry"), dict):
            result.append((index, entry["entry"]))
        elif kind == "session" or not streamed:
            result.append((index, entry))
        elif not appended and kind == "message_end" and isinstance(entry.get("message"), dict):
            result.append((index, {"type": "message", "message": entry["message"]}))
        elif not appended and kind == "compaction_end" and isinstance(entry.get("result"), dict) and not entry.get("aborted"):
            result.append((index, {**entry["result"], "type": "compaction"}))
    return result, ["Pi progress streams may omit persisted prompt state."] if streamed else []


def _active_path(entries: list[tuple[int, dict]]) -> tuple[list[tuple[int, dict]], list[str], list[str]]:
    """Never fill a broken parent chain with messages from other branches."""
    if not any("parentId" in e for _, e in entries):
        return entries, [], []
    indexed: dict[str, tuple[int, dict]] = {}
    duplicated = set()
    for index, entry in entries:
        eid = entry.get("id")
        if isinstance(eid, str) and eid:
            if eid in indexed:
                duplicated.add(eid)
            indexed[eid] = (index, entry)
    limitations: list[str] = []
    missing: list[str] = []
    if not indexed:
        return [], ["Pi parent links contain no usable entry IDs."], ["active_leaf"]
    leaf = entries[-1]
    if not isinstance(leaf[1].get("id"), str) or not leaf[1].get("id"):
        return [], ["Pi active leaf is not identifiable."], ["active_leaf"]
    selected = []
    visited = set()
    current: tuple[int, dict] | None = leaf
    while current is not None:
        index, entry = current
        eid = entry.get("id")
        if eid in duplicated or eid in visited:
            missing.append(str(eid))
            limitations.append("Pi parent links are ambiguous or cyclic; only unambiguous reachable context is shown.")
            break
        visited.add(eid)
        selected.append(current)
        if "parentId" not in entry:
            missing.append(f"{eid}:parentId")
            limitations.append("Pi parent links are incomplete; earlier context is unavailable.")
            break
        parent = entry["parentId"]
        if parent is None:
            break
        previous = indexed.get(parent) if isinstance(parent, str) else None
        if previous is None or previous[0] >= index:
            missing.append(str(parent))
            limitations.append("Pi active history references an unavailable or later parent entry.")
            break
        current = previous
    selected.reverse()
    return selected, limitations, missing


def _step_map(steps: list) -> dict[str, int | None]:
    mapped: dict[str, int | None] = {}
    for position, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("role") != "assistant":
            continue
        eid = step.get("id")
        if isinstance(eid, str) and eid:
            value = step.get("index", position)
            mapped[eid] = None if eid in mapped or not isinstance(value, int) or isinstance(value, bool) else value
    return mapped


def _request(row: tuple[int, dict], session_id: str, model: str, step_map: dict,
             branch_id: str, previous_id: str | None, missing: list[str]) -> dict:
    index, entry = row
    eid = str(entry.get("id") or index)
    message = entry["message"]
    return {"id": f"pi:{session_id}:{eid}:{index}", "label": f"Pi generation · {eid}",
            "evidence": "reconstructed", "scope": "native_prompt_state", "model": model,
            "session_id": session_id, "agent_id": "",
            "step_index": step_map.get(entry.get("id")) if isinstance(entry.get("id"), str) else None,
            "timestamp_ms": _pi_event_ts_ms(entry), "missing_references": list(missing),
            "branch_id": branch_id, "previous_request_id": previous_id,
            "provider": str(message.get("provider") or "")}


def _prompt_state(prefix: list[tuple[int, dict]], source_path: str) -> tuple:
    limitations: list[str] = []
    missing: list[str] = []
    sources = []
    selected = prefix
    compact_positions = [i for i, (_, entry) in enumerate(prefix) if entry.get("type") == "compaction"]
    if compact_positions:
        position = compact_positions[-1]
        compact = prefix[position]
        first = compact[1].get("firstKeptEntryId")
        if first == compact[1].get("id") and first is not None:
            retained = []
        else:
            start = (next((i for i, (_, e) in enumerate(prefix[:position]) if e.get("id") == first), None)
                     if isinstance(first, str) and first else None)
            if start is None:
                retained = []
                missing.append(str(first) if first is not None else "firstKeptEntryId")
                limitations.append("Pi compaction retention is incomplete; unknown earlier messages are not included.")
            else:
                retained = [row for row in prefix[start:position]
                            if not (row[1].get("type") == "message"
                                    and isinstance(row[1].get("message"), dict)
                                    and row[1]["message"].get("role") == "system")]
        selected = [compact, *retained, *prefix[position + 1:]]

    edits: dict[str, tuple[int, dict]] = {}
    prior: dict[str, dict] = {}
    for index, entry in prefix:
        if entry.get("type") == "context_edit":
            target = entry.get("targetId")
            target_entry = prior.get(target) if isinstance(target, str) else None
            target_message = target_entry.get("message") if target_entry else None
            context_target = (target_entry is not None and
                              (target_entry.get("type") == "custom_message"
                               or (target_entry.get("type") == "message" and isinstance(target_message, dict)
                                   and target_message.get("role") in {"user", "assistant", "toolResult", "custom", "hookMessage"})))
            if isinstance(target, str) and context_target:
                edits[target] = (index, entry)
            else:
                missing.append(str(target))
                limitations.append("A Pi context edit targets an unavailable earlier entry.")
        eid = entry.get("id")
        if isinstance(eid, str):
            prior[eid] = entry

    sections: dict[str, str] = {}
    tools: dict[str, dict] = {}
    section_refs: dict[str, dict] = {}
    tool_refs: dict[str, dict] = {}
    system_seen = False
    tools_seen = False
    messages = []
    message_refs = []

    def patch(message: dict, index: int, entry: dict) -> None:
        nonlocal system_seen, tools_seen
        system_seen = True
        content = message.get("content")
        if isinstance(content, str) and content:
            sections["content"] = content
            section_refs["content"] = _ref(index, entry, source_path)
        updates = message.get("sections")
        if isinstance(updates, dict):
            for key, value in updates.items():
                if value is None:
                    sections.pop(str(key), None)
                    section_refs.pop(str(key), None)
                elif isinstance(value, str):
                    sections[str(key)] = value
                    section_refs[str(key)] = _ref(index, entry, source_path)
                else:
                    limitations.append("A Pi prompt section has an unsupported value and was not replayed.")
                    missing.append(f"{entry.get('id', index)}:sections:{key}")
        elif updates is not None:
            limitations.append("Pi prompt sections have an unsupported representation.")
        for field, removing in (("toolsRemoved", True), ("toolsAdded", False)):
            declarations = message.get(field)
            if declarations is None:
                continue
            if not isinstance(declarations, list):
                limitations.append("Pi tool declarations have an unsupported representation.")
                continue
            if field == "toolsAdded":
                tools_seen = True
            for declaration in declarations:
                name = declaration.get("name") if isinstance(declaration, dict) else declaration
                if not isinstance(name, str) or not name:
                    limitations.append("A Pi tool declaration has no usable name.")
                elif removing:
                    tools.pop(name, None)
                    tool_refs.pop(name, None)
                elif isinstance(declaration, dict):
                    tools[name] = _safe_copy(declaration, limitations)
                    tool_refs[name] = _ref(index, entry, source_path)
                    if not isinstance(declaration.get("parameters"), dict):
                        limitations.append("A Pi tool declaration is missing its schema.")
                else:
                    limitations.append("A Pi tool declaration is missing its schema.")
        sources.append(_ref(index, entry, source_path))

    for index, entry in selected:
        kind = entry.get("type")
        message: dict | None = None
        if kind == "compaction":
            checkpoint = entry.get("systemMessage")
            if isinstance(checkpoint, dict) and checkpoint.get("role") == "system":
                patch(checkpoint, index, entry)
            else:
                limitations.append("Pi compaction has no recorded system and tool checkpoint.")
            message = {"role": "compactionSummary", "content": _safe_copy(entry.get("summary") or "", limitations)}
        elif kind == "branch_summary":
            message = {"role": "branchSummary", "content": _safe_copy(entry.get("summary") or "", limitations)}
        elif kind == "custom_message":
            message = {"role": "user", "content": _safe_copy(entry.get("content", ""), limitations)}
        elif kind == "message" and isinstance(entry.get("message"), dict):
            native = entry["message"]
            role = native.get("role")
            if role == "system":
                patch(native, index, entry)
                continue
            if role == "bashExecution" and native.get("excludeFromContext") is True:
                continue
            if role not in {"user", "assistant", "toolResult", "custom", "hookMessage", "bashExecution"}:
                limitations.append("A Pi message role is unsupported for prompt replay.")
                continue
            keep = ("role", "content", "toolCallId", "toolName", "isError", "command", "output", "exitCode",
                    "cancelled", "truncated", "fullOutputPath")
            message = {key: _safe_copy(native[key], limitations) for key in keep if key in native}
            if role in {"custom", "hookMessage"}:
                message["role"] = "user"
            if role == "bashExecution":
                message["content"] = _safe_copy(native.get("output") or "", limitations)
        if message is None:
            continue
        edit = edits.get(entry.get("id")) if isinstance(entry.get("id"), str) else None
        if edit is not None:
            edit_index, edit_entry = edit
            sources.append(_ref(edit_index, edit_entry, source_path))
            if "replacement" not in edit_entry:
                missing.append(f"{edit_entry.get('id', edit_index)}:replacement")
                limitations.append("A Pi context edit has no recorded replacement; its target is omitted.")
                continue
            replacement = edit_entry["replacement"]
            if replacement is None:
                continue
            if not isinstance(replacement, (str, list)):
                limitations.append("A Pi context edit has an unsupported replacement; its target is omitted.")
                continue
            message["content"] = ([{"type": "text", "text": replacement}]
                                  if isinstance(replacement, str) and message["role"] in {"assistant", "toolResult"}
                                  else _safe_copy(replacement, limitations))
        messages.append(message)
        reference = {**_ref(index, entry, source_path),
                     "entry_id": entry.get("id") if isinstance(entry.get("id"), str) else "",
                     "origin": str(kind)}
        if edit is not None:
            edit_id = edit[1].get("id")
            reference["context_edit_id"] = edit_id if isinstance(edit_id, str) else ""
        message_refs.append(reference)
        sources.append(_ref(index, entry, source_path))
    if not system_seen:
        limitations.append("The Pi system prompt is not recorded before this generation.")
    if not tools_seen:
        limitations.append("The complete Pi tool declarations are not recorded before this generation.")
    system = [{"type": "text", "section": name, "text": text} for name, text in sections.items()]
    return (system, list(tools.values()), messages, sources, list(dict.fromkeys(limitations)),
            list(dict.fromkeys(missing)), message_refs,
            [section_refs.get(name, {}) for name in sections], [tool_refs.get(name, {}) for name in tools])


def inspect_pi(raw: dict, steps: list, request_id: str | None = None) -> dict[str, Any]:
    """Inspect a Pi generation's branch-local, recorded pre-response context.

    Only the selected generation is materialized; catalog construction and
    replay are linear in the native entry count. Neither history nor metrics
    are changed. ``request_id=None`` selects the latest generation.
    """
    result: dict[str, Any] = {"available": False, "requests": [], "selected_request": None,
                              "limitations": [_LIMITATION]}
    original = raw.get("_pi_entries")
    if not isinstance(original, list) or not original:
        result["limitations"].append("Native Pi prompt evidence is unavailable in this export.")
        return result
    if len(original) > _MAX_ENTRIES:
        result["limitations"].append("Pi prompt replay exceeds the supported entry limit.")
        return result
    events, stream_limits = _events(original)
    headers = [(index, e) for index, e in events if e.get("type") == "session"]
    if len(headers) != 1:
        result["limitations"].append("Pi prompt replay requires one identifiable native session header.")
        return result
    header = headers[0][1]
    version = header.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version not in (1, 2, 3):
        result["limitations"].append("This Pi session version is unsupported for prompt replay.")
        return result
    session_id = str(header.get("id") or "pi-session")
    path, branch_limits, missing = _active_path([(i, e) for i, e in events if e.get("type") != "session"])
    result["limitations"].extend(stream_limits + branch_limits)
    if header.get("parentSession"):
        result["limitations"].append("Parent-session files are not loaded; only recorded Pi entries are inspected.")
    source_path = raw.get("_pi_prompt_source")
    source_path = source_path if isinstance(source_path, str) else ""
    branch_id = str(path[-1][1].get("id") or f"linear:{session_id}") if path else f"linear:{session_id}"
    step_map = _step_map(steps)
    model = ""
    thinking = ""
    requests = []
    selected_position: dict[str, int] = {}
    previous_id = None
    settings: dict[str, dict] = {}
    pending_events: list[dict] = []
    pending_counts: dict[str, dict] = {}
    recorded_sections: dict[str, Any] = {}
    recorded_tools: dict[str, Any] = {}
    for position, row in enumerate(path):
        index, entry = row
        for event in _state_events(index, entry, source_path, recorded_sections, recorded_tools):
            kind = event["kind"]
            if kind in {"user_messages", "tool_results"}:
                if kind not in pending_counts:
                    pending_counts[kind] = event
                else:
                    pending_counts[kind]["count"] += 1
                pending_counts[kind]["end_raw_index"] = index
            else:
                pending_events.append(event)
        if entry.get("type") == "model_change":
            model = str(entry.get("modelId") or "")
        elif entry.get("type") == "thinking_level_change":
            thinking = str(entry.get("thinkingLevel") or "")
        message = entry.get("message")
        if entry.get("type") != "message" or not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        model = str(message.get("model") or model)
        request = _request(row, session_id, model, step_map, branch_id, previous_id, missing)
        request["prompt_events"] = sorted([*pending_events, *pending_counts.values()],
                                          key=lambda event: event["source"]["raw_index"])
        pending_events = []
        pending_counts = {}
        previous_id = request["id"]
        requests.append(request)
        selected_position[request["id"]] = position
        level = message.get("thinkingLevel") or thinking
        settings[request["id"]] = {"thinking_level": level if isinstance(level, str) else ""}
    if len(requests) > _MAX_REQUESTS:
        requests = requests[-_MAX_REQUESTS:]
        result["limitations"].append("Only the latest Pi generations are listed because the request catalog limit was reached.")
    result["requests"] = requests
    result["available"] = bool(requests)
    if not requests:
        result["limitations"].append("No recorded Pi assistant generation is available for prompt inspection.")
        return result
    selected = requests[-1] if request_id is None else next((r for r in requests if r["id"] == request_id), None)
    if selected is None:
        raise ValueError(f"Unknown prompt request id: {request_id}")
    system, tools, messages, sources, limits, detail_missing, message_refs, system_refs, tools_refs = _prompt_state(
        path[:selected_position[selected["id"]]], source_path)
    detail = {**selected, "system": system, "tools": tools, "messages": messages,
              "message_refs": message_refs, "system_refs": system_refs, "tools_refs": tools_refs,
              "parameters": {"model": selected["model"], **settings[selected["id"]]}, "sources": sources,
              "limitations": list(dict.fromkeys([*result["limitations"], *limits])),
              "missing_references": list(dict.fromkeys([*missing, *detail_missing]))}
    result["selected_request"] = detail
    return result
