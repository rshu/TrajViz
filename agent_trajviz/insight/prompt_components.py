"""Semantic, source-backed prompt components and bounded composition summaries.

Categories describe the purpose of recorded content independently of its message
role. They never add text or assert that transcript candidates reached a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from agent_trajviz.tool_vocab import SPAWN_TOOL_NAMES, is_mcp_tool, parse_skill_name

from .token_explain import _est_tokens

CATEGORIES = (
    ("system", "System prompt", "#6b7280"),
    ("tools", "Tool definitions", "#7c3aed"),
    ("rules", "Rules", "#e11d48"),
    ("skills", "Skills", "#ca8a04"),
    ("mcp", "MCP", "#0d9488"),
    ("subagents", "Subagent definitions", "#2563eb"),
    ("summarized", "Summarized conversation", "#8b5cf6"),
    ("conversation", "Conversation", "#ea580c"),
    ("tool_outputs", "Tool outputs", "#059669"),
    ("unattributed", "Unattributed", "#94a3b8"),
)
_CATEGORY_KEYS = {item[0] for item in CATEGORIES}
_SPAWN_NAMES = {name.lower() for name in SPAWN_TOOL_NAMES}
_MAX_CONTENT_SEGMENTS = 128
_SECTIONS = {
    "rules": "rules", "user_rules": "rules", "always_apply_rules": "rules", "project_rules": "rules",
    "skills": "skills", "skill": "skills", "skill_definitions": "skills",
    "mcp": "mcp", "mcp_tools": "mcp", "mcp_servers": "mcp",
    "subagents": "subagents", "subagent_definitions": "subagents", "agent_definitions": "subagents",
    "tools": "tools", "tool_definitions": "tools", "tool_schemas": "tools",
    "summary": "summarized", "compaction_summary": "summarized", "summarized": "summarized",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _pointer(path: str, key: Any) -> str:
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _refs(value: Any) -> list[dict]:
    def source(ref: dict) -> dict:
        # Alignment tables live once on selected_request; each segment carries
        # its own exact source rather than duplicating the whole message table.
        return copy.deepcopy({key: item for key, item in ref.items() if key != "content_refs"})

    if isinstance(value, dict):
        return [source(value)] if value else []
    return [source(ref) for ref in value if isinstance(ref, dict) and ref] if isinstance(value, list) else []


def _context(detail: dict, provenance: list[dict]) -> str:
    """Stable owner scope; a request ID is deliberately not a context identity."""
    return str(detail.get("context_id") or _hash([
        str(detail.get("id", "")).split(":", 1)[0], detail.get("session_id", ""),
        detail.get("agent_id", ""), detail.get("source_path") or next((ref.get("source_path") for ref in provenance if ref.get("source_path")), ""),
    ]))


def _logical(component: dict, detail: dict, provenance: list[dict], tool_counts: dict) -> tuple[str | None, str]:
    context = _context(detail, provenance)
    native_identity = component.get("identity")
    if native_identity:
        return "component:" + _hash([context, native_identity]), "native_identity"
    if component.get("kind") == "tool" and isinstance(component.get("content"), dict):
        name = component["content"].get("name")
        if isinstance(name, str) and name and tool_counts.get(name) == 1:
            return "component:" + _hash([context, "tool", name]), "native_tool_name"
    for source in provenance:
        if source.get("record") == "normalized_step":
            if source.get("message_id"):
                key = ["message_id", source["message_id"]]
            elif isinstance(source.get("raw_index"), int) and not isinstance(source["raw_index"], bool):
                key = ["raw_index", source["raw_index"]]
            elif source.get("step_index") is not None:
                key = ["step_index", source["step_index"]]
            else:
                continue
            if key:
                return "component:" + _hash([context, source.get("source_path", ""), key,
                                              source.get("field"), source.get("index")]), "normalized_source"
        if source.get("kind") == "blob" and source.get("reference"):
            return "component:" + _hash([context, "blob", source["reference"]]), "native_blob_reference"
    return None, "unavailable"


def _tool_category(name: str, arguments: Any = None) -> tuple[str | None, str]:
    if parse_skill_name(name, arguments):
        return "skills", "inferred_skill_tool_name"
    if name.lower() in _SPAWN_NAMES:
        return "subagents", "inferred_subagent_tool_name"
    if is_mcp_tool(name):
        return "mcp", "inferred_mcp_tool_name"
    return None, ""


def record_tool_associations(message: Any, names: dict) -> None:
    """Advance one snapshot-local index after classifying the current message."""
    if not isinstance(message, dict):
        return
    content = message.get("content")
    for block in content if isinstance(content, list) else []:
        if (not isinstance(block, dict) or not isinstance(block.get("type"), str)
                or block["type"] not in {"tool_use", "toolCall", "function_call"}):
            continue
        identifier = block.get("id") or block.get("call_id")
        name = block.get("name")
        if isinstance(identifier, str) and identifier and isinstance(name, str) and name:
            call = {"name": name, "arguments": block.get("input", block.get("arguments"))}
            names[identifier] = call if identifier not in names or names[identifier] == call else None


def _instruction_marker(value: Any) -> tuple[str | None, str]:
    text = value if isinstance(value, str) else value.get("text") if isinstance(value, dict) else None
    if not isinstance(text, str):
        return None, ""
    text = text.lstrip()
    if text.startswith("# AGENTS.md instructions"):
        return "rules", "inferred_recorded_instruction_marker"
    if text.startswith("<skill>") or text.startswith("<skill "):
        return "skills", "inferred_recorded_skill_marker"
    return None, ""


def _category(value: Any, kind: str, role: str, names: dict) -> tuple[str, str]:
    if isinstance(value, dict):
        section = value.get("section")
        if kind == "system" and isinstance(section, str):
            category = _SECTIONS.get(section.lower().replace("-", " ").replace(" ", "_"))
            if category:
                return category, "recorded_section_name"
        if value.get("_prompt_source") == "compaction_summary" or role in {"compactionSummary", "branchSummary"}:
            return "summarized", "recorded_summary"
        block_type = value.get("type")
        is_result = (isinstance(block_type, str) and block_type in {"tool_result", "function_call_output"}
                     or role in {"tool", "toolResult"})
        is_call = isinstance(block_type, str) and block_type in {"tool_use", "toolCall", "function_call"}
        raw_name = value.get("name") or value.get("toolName") or value.get("tool_name")
        name = raw_name if isinstance(raw_name, str) else ""
        associated = False
        call = None
        if is_result:
            identifier = value.get("tool_use_id") or value.get("toolCallId") or value.get("call_id")
            call = names.get(identifier) if isinstance(identifier, str) else None
            if isinstance(call, dict):
                recorded_name = call.get("name")
                if not name:
                    name = recorded_name
                    associated = bool(name)
                elif str(name).casefold() != str(recorded_name).casefold():
                    # A conflicting result name cannot borrow another call's
                    # arguments to identify a skill or other semantic source.
                    call = None
        if kind == "tool" or is_call or is_result:
            category, basis = _tool_category(str(name or ""), value.get("input", value.get("arguments")))
            if category and (category != "subagents" or kind == "tool"):
                return category, basis + ("_via_recorded_call" if associated else "")
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if is_result and str(name or "").lower() == "read" and isinstance(arguments, dict):
                path = arguments.get("file_path") or arguments.get("path") or arguments.get("filePath")
                if isinstance(path, str) and path.replace("\\", "/").rsplit("/", 1)[-1] == "SKILL.md":
                    return "skills", "inferred_recorded_skill_file_read"
        if is_result:
            return "tool_outputs", "recorded_tool_result"
    marker_category, marker_basis = _instruction_marker(value)
    if marker_category and kind != "tool":
        return marker_category, marker_basis
    if kind == "tool":
        return "tools", "recorded_tool_definition"
    if kind == "system" or role in {"system", "developer"}:
        return "system", "recorded_role"
    if role in {"tool", "toolResult"}:
        return "tool_outputs", "recorded_role"
    if role in {"compactionSummary", "branchSummary"}:
        return "summarized", "recorded_summary"
    if role in {"user", "assistant", "custom", "hookMessage", "bashExecution"}:
        return "conversation", "recorded_role"
    return "unattributed", "unavailable"


def _layout(original: Any, measurable: Any, measure: Callable, targets: set[str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Map original JSON pointers onto the exact measured canonical text."""
    pieces: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    length = 0

    def add(text: str) -> None:
        nonlocal length
        pieces.append(text)
        length += len(text)

    def walk(raw: Any, visible: Any, path: str) -> None:
        start = length
        if path in targets or not any(target.startswith(path + "/") for target in targets):
            add(visible if path == "" and isinstance(visible, str) else _json(visible))
        elif isinstance(visible, dict):
            add("{")
            for position, key in enumerate(sorted(visible)):
                if position:
                    add(",")
                add(_json(key) + ":")
                walk(raw.get(key) if isinstance(raw, dict) else visible[key], visible[key], _pointer(path, key))
            add("}")
        elif isinstance(visible, list):
            add("[")
            kept = [(index, item) for index, item in enumerate(raw) if measure(item)[0] is not None] if isinstance(raw, list) else list(enumerate(visible))
            for position, child in enumerate(visible):
                if position:
                    add(",")
                index, item = kept[position] if position < len(kept) else (position, child)
                walk(item, child, _pointer(path, index))
            add("]")
        else:
            add(visible if path == "" and isinstance(visible, str) else _json(visible))
        if path in targets:
            spans[path] = (start, length)

    walk(original, measurable, "")
    return "".join(pieces), spans


def enrich_component(component: dict, detail: dict, *, field: str, index: int,
                     measurable: Any, measured_text: str, measure: Callable,
                     associations: dict | None = None, tool_counts: dict | None = None) -> dict:
    """Attach independent semantics, identity, source locations and token partitions."""
    value, role, kind = component["content"], component["role"], component["kind"]
    field_refs = detail.get({"system": "system_refs", "tools": "tools_refs", "messages": "message_refs"}[field], [])
    source = field_refs[index] if isinstance(field_refs, list) and index < len(field_refs) else None
    provenance = _refs(source) or _refs(component.get("source_refs"))
    logical_id, identity_basis = _logical(component, detail, provenance, tool_counts or {})
    location = {"field": field, "index": index,
                "json_pointer": f"/{field}/{index}" if isinstance(detail.get(field), list) else f"/{field}"}
    occurrence_id = "occurrence:" + _hash([detail["id"], location])
    names = associations if associations is not None else {}
    category, basis = _category(value, kind, role, names)
    component.update(category=category, category_basis=basis, logical_id=logical_id,
                     identity_basis=identity_basis, content_hash=_hash(value), occurrence_id=occurrence_id,
                     location=location, provenance=provenance)
    targets = []
    if isinstance(value, dict) and kind == "message" and isinstance(value.get("content"), list):
        targets = [(_pointer("/content", pos), child) for pos, child in enumerate(value["content"][:_MAX_CONTENT_SEGMENTS])]
        if len(value["content"]) > _MAX_CONTENT_SEGMENTS:
            component["segmentation_partial"] = True
    elif isinstance(value, dict) and kind == "message" and "content" in value:
        targets = [("/content", value["content"])]
    elif isinstance(value, dict) and kind == "system" and "text" in value:
        targets = [("/text", value["text"])]
    else:
        targets = [("", value)]
    text, spans = _layout(value, measurable, measure, {pointer for pointer, _raw in targets})
    # Empty/opaque component measurements intentionally contain no token-bearing spans.
    if not measured_text:
        text = ""
        spans = {"": (0, 0)}
    elif text != measured_text:
        # A malformed shape must not publish offsets into a different serialization.
        spans = {"": (0, len(measured_text))}
        text = measured_text
    regions = []
    cursor = 0
    for pointer, raw in sorted((item for item in targets if item[0] in spans), key=lambda item: spans[item[0]][0]):
        start, end = spans[pointer]
        if start > cursor:
            regions.append((cursor, start, "", value, "framing"))
        if end > start:
            regions.append((start, end, pointer, raw, "content"))
        cursor = end
    if cursor < len(text):
        regions.append((cursor, len(text), "", value, "framing"))
    if not regions:
        regions = [(0, len(text), "", value, "framing")]
    weights = [_est_tokens(text[start:end]) for start, end, *_ in regions]
    weight_total, total = sum(weights), component["estimated_tokens"]
    allocations = [total * weight // weight_total if weight_total else 0 for weight in weights]
    if weight_total:
        remainder = total - sum(allocations)
        for position in sorted(range(len(weights)), key=lambda pos: (total * weights[pos]) % weight_total, reverse=True)[:remainder]:
            allocations[position] += 1
    segments = []
    for ordinal, ((start, end, pointer, raw, segment_kind), tokens) in enumerate(zip(regions, allocations, strict=True)):
        segment_category, segment_basis = _category(raw, kind, role, names) if segment_kind == "content" else (category, basis)
        # Native sections, summaries and result envelopes apply to their text blocks.
        if (basis in {"recorded_section_name", "recorded_summary"}
                or role in {"tool", "toolResult"} and pointer.startswith("/content")):
            segment_category, segment_basis = category, basis
        segment_refs = copy.deepcopy(provenance)
        if isinstance(source, dict) and pointer.startswith("/content/"):
            refs = source.get("content_refs")
            pos = int(pointer.split("/")[2])
            if isinstance(refs, list) and pos < len(refs):
                segment_refs = _refs(refs[pos]) or segment_refs
        # A native message identity does not identify its mutable array slots.
        # Insertion/reordering can move unrelated blocks onto the same pointer;
        # keep those segments request-local until a child identity is recorded.
        segment_logical_id = (logical_id + ":" + pointer
                              if logical_id and segment_kind == "content" and not pointer.startswith("/content/")
                              else None)
        segments.append({"category": segment_category, "category_basis": segment_basis,
                         "logical_id": segment_logical_id,
                         "identity_basis": identity_basis if segment_logical_id else "unavailable",
                         "content_hash": _hash(raw) if segment_kind == "content" else _hash(text[start:end]),
                         "occurrence_id": "segment:" + _hash([occurrence_id, ordinal, pointer]),
                         "json_pointer": pointer, "kind": segment_kind, "role": role,
                         "estimated_tokens": tokens, "provenance": segment_refs,
                         "serialized_span": {"start": start, "end": end, "basis": "measured_component_json"}})
    component["segments"] = segments
    return component


def composition_summary(components: list[dict], token_estimate: dict) -> dict:
    """Compact totals only; native content and source bodies remain in components."""
    totals: Counter = Counter()
    counts: Counter = Counter()
    for component in components:
        present = set()
        segments = component.get("segments") or [component]
        for segment in segments:
            category = segment.get("category", "unattributed")
            category = category if isinstance(category, str) and category in _CATEGORY_KEYS else "unattributed"
            totals[category] += int(segment.get("estimated_tokens") or 0)
            present.add(category)
        counts.update(present)
    return {"evidence": "estimated", "basis": "Recorded component text/JSON partitioned by semantic category; category inference is disclosed per segment.",
            "total": sum(totals.values()), "partial": bool(token_estimate.get("partial") or any(component.get("segmentation_partial") for component in components)),
            "component_count": len(components),
            "categories": [{"category": key, "label": label, "estimated_tokens": totals[key],
                            "component_count": counts[key]} for key, label, _color in CATEGORIES]}
