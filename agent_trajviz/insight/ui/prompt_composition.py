"""Bounded component explorer for estimated, source-backed prompt composition."""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from typing import Any

from ...i18n import text
from ..prompt_components import CATEGORIES
from . import prompt_text

CSS = """
.pc-explorer {margin-top:14px;}.pc-explorer>h3 {font-size:17px;margin:8px 0;}
.pc-composition-key {display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;}
.pc-composition-key>span {border:1px solid #dce3ef;border-left:4px solid var(--pc-category);border-radius:6px;padding:6px 9px;font-size:12px;}
.pc-category {border:1px solid #dce3ef;border-left:4px solid var(--pc-category);border-radius:9px;padding:12px;margin:10px 0;min-width:0;}
.pc-category>summary {cursor:pointer;font-weight:600;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;}
.pc-category>summary small {font-weight:400;color:#475569;margin-left:auto;}
.pc-component-meta {display:grid;grid-template-columns:minmax(100px,150px) minmax(0,1fr);gap:6px 12px;font-size:12px;line-height:1.5;margin:12px 0;}
.pc-component-meta dt {color:#475569;}.pc-component-meta dd {margin:0;overflow-wrap:anywhere;white-space:pre-wrap;}
.pc-explorer .pc-parts>details {margin-top:10px;}
.pc-component-snippet {font-weight:400;color:#475569;}.pc-component-origin {flex-basis:100%;font-size:11px;font-weight:400;color:#64748b;}
.pc-source-list {display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:#475569;margin:10px 0;}
.pc-quality {font-size:12px;line-height:1.55;}.pc-quality p {margin:4px 0;}
@media(max-width:640px){.pc-component-meta{grid-template-columns:1fr;gap:3px;}.pc-component-meta dd{margin-bottom:7px;}}
"""
_MAX_COMPONENTS = 40
_MAX_ENTRIES = 80
_MAX_SEGMENTS = 12
_MAX_TEXT = 6000
_MAX_TOTAL_TEXT = 180_000


def _literal(value: Any, limit: int = 1200) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return html.escape(str(value)[:limit])


def _pointer(value: Any, pointer: Any) -> Any:
    """Resolve recorded JSON pointers without interpreting source content."""
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    parts = pointer[1:].split("/")
    if len(parts) > 64:
        return None
    try:
        for part in parts:
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) and part.isdecimal() else value[part]
        return value
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def compact_count(value: Any) -> str:
    """Keep dense summaries readable; exact values remain available in titles."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
        if abs(value) >= 10_000:
            return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
        return f"{value:,.0f}"
    return str(value)


def _component_count(count: int) -> str:
    return text("{count} component" if count == 1 else "{count} components", count=compact_count(count))


def quality_note(detail: dict) -> str:
    """Explain content recovery independently of settings and token estimation."""
    quality = detail.get("evidence_quality") or {}
    missing = quality.get("missing_references", detail.get("missing_references", []))
    settings = quality.get("unsupported_settings", detail.get("unsupported_fields", []))
    pieces = []
    if missing:
        def catalog_marker(item):
            return item.get("kind") == "catalog" if isinstance(item, dict) else str(item).startswith("references:")

        def prefix_limit(item):
            return (item.get("kind") == "prefix" and item.get("reason") == "inspection_limit"
                    if isinstance(item, dict) else item == "prefix:inspection-limit")

        truncated = any(catalog_marker(item) for item in missing)
        prefix_unknown = any(prefix_limit(item) for item in missing)
        known = [item for item in missing if not catalog_marker(item) and not prefix_limit(item)]
        lower_bound = truncated or prefix_unknown
        if known:
            if lower_bound:
                template = ("At least {count} prompt reference could not be resolved; totals cover recovered content." if len(known) == 1
                            else "At least {count} prompt references could not be resolved; totals cover recovered content.")
            else:
                template = ("{count} prompt reference could not be resolved; totals cover recovered content." if len(known) == 1
                            else "{count} prompt references could not be resolved; totals cover recovered content.")
            pieces.append(text(template, count=f"{len(known):,}"))
            reasons = Counter(item.get("reason") for item in known if isinstance(item, dict))
            labels = {"not_recorded": "Unrecorded references: {count}.", "invalid": "Invalid references: {count}.",
                      "conflicting": "Conflicting references: {count}.", "inspection_limit": "Reference inspection limit reached: {count}."}
            known_labels = {"not_recorded": "Known unrecorded references: {count}.", "invalid": "Known invalid references: {count}.",
                            "conflicting": "Known conflicting references: {count}.", "inspection_limit": "Known references at inspection limit: {count}."}
            selected = known_labels if lower_bound else labels
            pieces.extend(text(selected[reason], count=f"{count:,}") for reason, count in reasons.items() if reason in selected)
        if truncated:
            pieces.append(text("Additional references were omitted after the inspection limit."))
        if prefix_unknown:
            pieces.append(text("The retained prefix could not be fully inspected; its unresolved reference count is unknown."))
    elif quality.get("content_complete") is True:
        pieces.append(text("Recorded components recovered."))
    if settings:
        pieces.append(text("Unsupported request settings") + ": <code>" + _literal(settings, 600) + "</code>.")
    if any(component.get("segmentation_partial") for component in detail.get("components", [])):
        pieces.append(text("Some components exceed the segment inspection limit; their remaining content is included in the parent category."))
    scope = detail.get("scope")
    if scope == "trajectory_estimate":
        pieces.append(text("Transcript candidates have unverified context membership."))
    elif scope == "native_prompt_state":
        pieces.append(text("Recorded prompt state precedes provider conversion."))
    if (detail.get("composition") or {}).get("partial") and not pieces:
        pieces.append(text("Category totals estimate the visible content available at this boundary."))
    return "<div class='pc-quality'>" + "".join("<p>" + piece + "</p>" for piece in pieces) + "</div>" if pieces else ""


def _references(component: dict, segment: dict | None) -> list[dict]:
    values = []
    for raw in ((segment or {}).get("provenance"), component.get("provenance"),
                component.get("sources"), component.get("source_refs")):
        if isinstance(raw, dict):
            raw = raw.get("source_refs", [raw])
        if isinstance(raw, list):
            values.extend(ref for ref in raw[:8] if isinstance(ref, dict))
    unique = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique[:8]


def _source_label(ref: dict) -> str:
    path = ref.get("source_path")
    filename = str(path).replace("\\", "/").rsplit("/", 1)[-1] if path else ""
    parts = ["<code>" + _literal(filename, 120) + "</code>"] if filename else []
    line = ref.get("line_number")
    raw_index = ref.get("raw_index")
    if type(line) is int and line > 0:
        parts.append(text("line {line}", line=f"{line:,}"))
    elif type(raw_index) is int and raw_index >= 0:
        parts.append(text("record {record}", record=f"{raw_index + 1:,}"))
    elif type(ref.get("step_index")) is int:
        parts.append(text("Workflow step #{step}", step=ref["step_index"]))
    if not parts and ref.get("record_id"):
        parts.append("<code>" + _literal(ref["record_id"], 100) + "</code>")
    return " · ".join(parts)


def _source_labels(component: dict, segment: dict | None) -> list[str]:
    labels = []
    refs = sorted(_references(component, segment), key=lambda ref: not bool(ref.get("source_path")))
    for ref in refs:
        label = _source_label(ref)
        if label and label not in labels:
            labels.append(label)
    return labels[:3]


def _semantic_title(component: dict, segment: dict | None, index: int) -> str:
    value = component.get("content") if segment is None else _pointer(component.get("content"), segment.get("json_pointer"))
    record = value if isinstance(value, dict) else {}
    parent = component.get("content") if isinstance(component.get("content"), dict) else {}
    role = (segment or {}).get("role") or component.get("role") or record.get("role")
    kind, block_type = component.get("kind"), record.get("type")
    name = record.get("name") or record.get("tool_name") or record.get("toolName")
    snippet_value = value
    if kind == "tool":
        title = text("Tool definition")
        name = name or parent.get("name")
        snippet_value = record.get("description", "")
    elif isinstance(block_type, str) and block_type in {"tool_use", "toolCall", "function_call"}:
        title = text("Tool call")
        arguments = record.get("input", record.get("arguments"))
        snippet_value = next((arguments[key] for key in ("command", "content", "file_path", "path")
                              if isinstance(arguments.get(key), str)), "") if isinstance(arguments, dict) else arguments
    elif (isinstance(block_type, str) and block_type in {"tool_result", "function_call_output"}
          or role in {"tool", "toolResult"}):
        title = text("Tool result")
        name = name or record.get("tool_use_id") or record.get("toolCallId") or record.get("call_id")
    elif kind == "system" or role in {"system", "developer"}:
        title = text("Developer" if role == "developer" else "System instructions")
        name = record.get("section") or parent.get("section")
    elif role in {"compactionSummary", "branchSummary"} or kind == "summary":
        title = text("Context summary")
    else:
        title = text({"user": "User message", "assistant": "Assistant message"}.get(role, "Message"))
    if isinstance(name, str) and name:
        title += " · <code>" + _literal(name, 100) + "</code>"
    from ..prompt_text import readable_component_blocks
    blocks = readable_component_blocks({**component, "content": snippet_value, "kind": "message"}, max_chars=180, max_blocks=1)
    snippet = " ".join((blocks[0].get("text", "") if blocks else "")[:180].split())
    if snippet:
        title += " <span class='pc-component-snippet'>— " + _literal(snippet, 120) + ("…" if len(snippet) > 120 else "") + "</span>"
    elif not name:
        title += " #" + str(index + 1)
    return title


def _metadata(component: dict, segment: dict | None, detail: dict) -> str:
    item = segment or component
    location = component.get("location") or {}
    source = item.get("provenance") or component.get("provenance") or {}
    refs = component.get("sources", component.get("source_refs", []))
    evidence = detail.get("evidence", "unavailable")
    labels = {"captured": "Captured", "reconstructed": "Reconstructed", "estimated": "Estimated", "unavailable": "Unavailable"}
    role = item.get("role") or component.get("role")
    role_names = {"user": "User", "assistant": "Assistant", "system": "System", "developer": "Developer",
                  "tool": "Tool result", "toolResult": "Tool result"}
    role_html = text(role_names[role]) if role in role_names else "<code>" + _literal(role) + "</code>"
    fields = [
        ("Logical identity", item.get("logical_id")),
        ("Identity basis", item.get("identity_basis")),
        ("Content version", item.get("content_hash")),
        ("Occurrence", item.get("occurrence_id")),
        ("Field location", location),
    ]
    if segment is not None:
        fields = [("Parent component identity", component.get("logical_id")),
                  ("Parent identity basis", component.get("identity_basis")),
                  ("Parent content version", component.get("content_hash")),
                  ("Parent occurrence", component.get("occurrence_id"))] + fields
        fields.append(("Segment JSON pointer", segment.get("json_pointer")))
    fields += [("Category basis", item.get("category_basis")), ("Provenance", source)]
    if refs:
        fields.append(("Source", refs[:8] if isinstance(refs, list) else refs))
    return ("<dl class='pc-component-meta'><dt>" + text("Prompt evidence") + "</dt><dd>"
            + text(labels.get(evidence, "Unavailable")) + "</dd><dt>" + text("Role") + "</dt><dd>" + role_html + "</dd>"
            + "".join("<dt>" + text(label) + "</dt><dd><code>" + _literal(value)
                      + "</code></dd>" for label, value in fields) + "</dl>"
            + ("<p class='pc-text-note'>" + text("This segment is identified by its recorded position within the parent component.") + "</p>"
               if segment is not None and not segment.get("logical_id") else ""))


def _preview(component: dict, segment: dict | None, detail: dict, budget: int) -> tuple[str, int]:
    if segment is not None and segment.get("kind") == "framing":
        return ("<p class='pc-text-note'>" + text(
            "Serialization framing contributes to the estimate and has no separate source text.") + "</p>", 0)
    pointer = "" if segment is None else segment.get("json_pointer")
    value = _pointer(component.get("content"), pointer)
    if value is None:
        return "<p class='pc-text-note'>" + text("No readable text was recorded for this component.") + "</p>", 0
    if budget >= _MAX_TEXT:
        scoped_detail = detail
        if pointer:
            evolution = detail.get("text_evolution") or {}
            # Change paths use the full component root; trim only this recorded
            # segment prefix so existing addition/repetition markers still align.
            single_segment = sum(item.get("kind") == "content" for item in component.get("segments", [])) == 1
            rewritten = any(item.get("after_id") == component.get("id") for item in evolution.get("rewrites", []))
            scoped_detail = {**detail, "text_evolution": {**evolution,
                "comparison_available": evolution.get("comparison_available") if single_segment or not rewritten else False,
                "rewrites": evolution.get("rewrites", []) if single_segment else [], "blocks": [
                {**block, "path": block["path"][len(pointer):]}
                for block in evolution.get("blocks", []) if isinstance(block, dict)
                and block.get("component_id") == component.get("id")
                and isinstance(block.get("path"), str)
                and (block["path"] == pointer or block["path"].startswith(pointer + "/"))]}}
        return prompt_text.render_component_text({**component, "content": value}, scoped_detail), _MAX_TEXT
    from ..prompt_text import readable_component_blocks
    blocks = readable_component_blocks({**component, "content": value}, max_chars=min(_MAX_TEXT, budget), max_blocks=12)
    content = "\n\n".join(block.get("text", "") for block in blocks)
    shortened = len(content) > min(_MAX_TEXT, budget) or any(block.get("text_truncated") for block in blocks)
    content = content[:min(_MAX_TEXT, budget)]
    if not content:
        return "<p class='pc-text-note'>" + text("No readable text was recorded for this component.") + "</p>", 0
    return ("<pre class='pc-text-content'>" + html.escape(content) + "</pre>"
            + ("<p class='pc-text-note'>" + text("Preview shortened. Evidence JSON keeps the full recorded data.") + "</p>"
               if shortened else ""), len(content))


def render_component_explorer(detail: dict, history: dict | None = None) -> str:
    """Group semantic segments by category while retaining their source identity."""
    parts = detail.get("components", [])
    if not parts:
        note = "<div class='pc-note'>" + text("No prompt components could be recovered.") + "</div>"
        return note + quality_note(detail)
    composition = detail.get("composition") or {}
    category_data = {item["category"]: item for item in composition.get("categories", [])}
    pieces = ["<div class='pc-explorer'><h3>" + text("Component explorer") + "</h3>",
              "<p class='pc-text-note'>" + text(
                  "Category sizes estimate serialized visible content. They do not measure billed usage or hidden provider input.") + "</p>"]
    if composition.get("basis"):
        pieces.append("<p class='pc-text-note'>" + text("Allocation basis") + ": <code>"
                      + text(composition["basis"]) + "</code></p>")
    pieces.append(quality_note(detail))
    if history and not history.get("complete", True):
        pieces.append("<div class='pc-note'>" + text(
            "Composition history includes {computed} of {total} boundaries.",
            computed=history.get("computed_requests", 0), total=history.get("total_requests", 0)) + "</div>")
    pieces.append("<div class='pc-composition-key'>")
    for key, label, color in CATEGORIES:
        row = category_data.get(key)
        if row and (row.get("estimated_tokens") or row.get("component_count")):
            pieces.append("<span style='--pc-category:" + color + "'>" + text(label) + " · ≈ "
                          + compact_count(row.get("estimated_tokens", 0)) + " tok · " + _component_count(row.get("component_count", 0)) + "</span>")
    pieces.append("</div>")
    visible = list(enumerate(parts))
    if len(visible) > _MAX_COMPONENTS:
        visible = visible[:20] + visible[-20:]
        pieces.append("<div class='pc-note'>" + text(
            "Showing {shown} of {total} components. Request JSON contains the full recorded data.",
            shown=len(visible), total=len(parts)) + "</div>")
    grouped = defaultdict(list)
    framing = defaultdict(lambda: {"count": 0, "tokens": 0})
    shown_entries = 0
    limited = False
    for index, component in visible:
        segments = component.get("segments")
        segments = segments if isinstance(segments, list) and segments else [None]
        semantic = [segment for segment in segments if segment is None or segment.get("kind") != "framing"]
        for segment in segments:
            if segment is not None and segment.get("kind") == "framing":
                category = segment.get("category", "unattributed")
                framing[category]["count"] += 1
                framing[category]["tokens"] += segment.get("estimated_tokens", 0) or 0
        limited |= len(semantic) > _MAX_SEGMENTS
        for segment in semantic[:_MAX_SEGMENTS]:
            if shown_entries >= _MAX_ENTRIES:
                limited = True
                break
            category = (segment or component).get("category", "unattributed")
            if category not in {row[0] for row in CATEGORIES}:
                category = "unattributed"
            grouped[category].append((index, component, segment))
            shown_entries += 1
    budget = _MAX_TOTAL_TEXT
    for key, label, color in CATEGORIES:
        entries = grouped.get(key, [])
        if not entries and key not in framing:
            continue
        amount = category_data.get(key, {}).get("estimated_tokens")
        size = "≈ " + compact_count(amount) + " tok" if amount is not None else "—"
        component_count = category_data.get(key, {}).get("component_count", len({index for index, _component, _segment in entries}))
        size += " · " + _component_count(component_count)
        pieces.append("<details class='pc-category' style='--pc-category:" + color + "' open><summary>"
                      + text(label) + "<small>" + size + "</small></summary><div class='pc-parts'>")
        for index, component, segment in entries:
            item = segment or component
            title = _semantic_title(component, segment, index)
            amount = item.get("estimated_tokens")
            size = "≈ " + compact_count(amount) + " tok" if amount is not None else "—"
            sources = _source_labels(component, segment)
            pieces.append("<details><summary><b>" + title + "</b><small title='" + _literal(amount) + "'>" + size + "</small>"
                          + ("<span class='pc-component-origin'>" + sources[0] + "</span>" if sources else "") + "</summary>")
            if budget > 0:
                preview, used = _preview(component, segment, detail, budget)
                pieces.append(preview)
                budget -= used
            else:
                limited = True
            if sources:
                pieces.append("<div class='pc-source-list'>" + "".join("<span>" + source + "</span>" for source in sources) + "</div>")
            pieces.append("<details class='pc-provenance'><summary>" + text("Identity and provenance")
                          + "</summary>" + _metadata(component, segment, detail) + "</details>")
            pieces.append("</details>")
        if key in framing:
            pieces.append("<details class='pc-text-note'><summary>" + text("Token estimate details") + "</summary><p>" + text(
                "Serialization framing contributes to the estimate and has no separate source text.") + " " + text(
                    "Framing in shown components: approximately {tokens} tokens across {count} segments.",
                    tokens=framing[key]["tokens"], count=framing[key]["count"]) + "</p></details>")
        pieces.append("</div></details>")
    if limited:
        pieces.append("<p class='pc-text-note'>" + text("Explorer preview shortened. Evidence JSON keeps the full recorded data.") + "</p>")
    return "".join(pieces) + "</div>"
