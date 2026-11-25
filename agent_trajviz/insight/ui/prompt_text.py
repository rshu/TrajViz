"""Bounded, escaped readable text previews for prompt evolution.

The public JSON retains the recorded data. These helpers only render previews;
they never infer rewrite pairings, translate source content, or modify evidence.
"""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from typing import Any

from ...i18n import text

CSS = """
.pc-text-legend {display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;}
.pc-text-status,.pc-text-repeat-badge {display:inline-flex;align-items:center;gap:5px;border:1px solid;border-radius:5px;padding:2px 7px;font-size:12px;font-weight:600;}
.pc-text-status.pc-text-added {background:#dcfce7!important;color:#14532d!important;border-color:#16a34a;}
.pc-text-status.pc-text-removed {background:#fee2e2!important;color:#7f1d1d!important;border-color:#dc2626;}
.pc-text-status.pc-text-retained,.pc-text-status.pc-text-uncompared {background:#f1f5f9!important;color:#334155!important;border-color:#94a3b8;}
.pc-text-status.pc-text-repeated,.pc-text-repeat-badge {background:#fef3c7!important;color:#78350f!important;border-color:#d97706;}
.pc-text-list {display:flex;flex-direction:column;gap:10px;margin-top:12px;}
.pc-text-card {border:1px solid #cbd5e1;border-left:4px solid #94a3b8;border-radius:8px;padding:10px 12px;min-width:0;}
.pc-text-card.pc-text-added {border-left-color:#16a34a;}.pc-text-card.pc-text-removed {border-left-color:#dc2626;}
.pc-text-card.pc-text-repeated {border-left-color:#d97706;}
.pc-text-card>summary {cursor:pointer;display:flex;align-items:center;gap:8px;flex-wrap:wrap;overflow-wrap:anywhere;}
.pc-text-card summary code {font-size:12px;overflow-wrap:anywhere;}
.pc-text-block {margin:8px 0;min-width:0;}.pc-text-labels {display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;}
pre.pc-text-content {max-height:360px;overflow:auto;white-space:pre-wrap!important;overflow-wrap:anywhere;line-height:1.55;font-size:12px;background:#f1f5f9!important;color:#334155!important;padding:12px!important;border-radius:6px;margin:8px 0!important;}
.pc-text-added>pre.pc-text-content {background:#f0fdf4!important;color:#14532d!important;}
.pc-text-removed>pre.pc-text-content {background:#fff1f2!important;color:#7f1d1d!important;}
.pc-text-repeated>pre.pc-text-content {background:#fffbeb!important;color:#78350f!important;}
.pc-text-source,.pc-text-note,.pc-text-repeat-note {font-size:12px;color:#475569;overflow-wrap:anywhere;line-height:1.5;margin:6px 0;}
.pc-text-repeat-note {color:#78350f;}.pc-text-source code {white-space:pre-wrap;overflow-wrap:anywhere;}
.pc-text-rewrite-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
.pc-text-rewrite-grid section {min-width:0;}.pc-text-rewrite-grid h4 {font-size:13px;margin:10px 0 0;}
ins.pc-text-added {background:#dcfce7!important;color:#14532d!important;text-decoration:none;border-bottom:1px solid #16a34a;}
del.pc-text-removed {background:#fee2e2!important;color:#7f1d1d!important;text-decoration:line-through;}
span.pc-text-retained,span.pc-text-uncompared {color:#475569!important;background:transparent!important;}
@media(max-width:640px){.pc-text-rewrite-grid{grid-template-columns:1fr;}.pc-text-card{padding:9px;}}
"""

_MAX_COMPONENT_CHARS = 6000
_MAX_COMPONENT_BLOCKS = 12
_MAX_CHANGE_CARDS = 16
_MAX_CHANGE_CHARS = 32_000
_MAX_BLOCK_CHARS = 4000
_MAX_REWRITES = 4
_MAX_DIFF_CHARS = 3000
_MAX_DIFF_TOKENS = 1000
_WORDS = re.compile(r"\s+|[\u3400-\u9fff\uf900-\ufaff]|[A-Za-z_][A-Za-z_0-9]*|\d+|[^\W_]+|[^\w\s]", re.UNICODE)
_STATUS = {
    "added": ("+", "Added text"), "removed": ("−", "Removed text"),
    "retained": ("=", "Retained text"), "uncompared": ("·", "Uncompared text"),
    "repeated": ("↻", "Repeated content"),
}
_KINDS = {"text": "Message", "tool_arguments": "Tool arguments", "tool_result": "Tool result"}


def _literal(value: Any, limit: int = 160) -> str:
    return value[:limit] if isinstance(value, str) else str(value)[:limit] if type(value) in (int, float) else ""


def _status(status: str) -> str:
    status = status if status in _STATUS else "uncompared"
    symbol, label = _STATUS[status]
    return (f"<span class='pc-text-status pc-text-{status}'><span aria-hidden='true'>{symbol}</span>"
            + text(label) + "</span>")


def legend() -> str:
    """Accessible color key; colors are always accompanied by words/symbols."""
    return ("<div class='pc-text-legend'>" + "".join(_status(status) for status in
            ("added", "removed", "retained", "uncompared", "repeated")) + "</div>")


def _preview(value: Any, limit: int) -> tuple[str, bool]:
    value = value if isinstance(value, str) else ""
    return value[:max(0, limit)], len(value) > max(0, limit)


def _shortened() -> str:
    return "<p class='pc-text-note'>" + text("Preview shortened. Evidence JSON keeps the full recorded data.") + "</p>"


def _readable(component: dict[str, Any]) -> list[dict[str, Any]]:
    from ..prompt_text import readable_component_blocks

    return readable_component_blocks(component, max_chars=_MAX_COMPONENT_CHARS, max_blocks=_MAX_COMPONENT_BLOCKS + 1)


def _block_key(block: dict[str, Any]) -> tuple[str, str]:
    return _literal(block.get("component_id"), 1000), _literal(block.get("path"), 1000)


def _repeat_html(block: dict[str, Any]) -> str:
    if block.get("repeated") is not True:
        return ""
    source_ids = block.get("repeat_source_component_ids")
    source_ids = source_ids if isinstance(source_ids, list) else []
    count = block.get("repeat_count")
    count = count if type(count) is int and count >= 0 else len(source_ids)
    result = ("<p class='pc-text-repeat-note'><span class='pc-text-repeat-badge'>↻ "
              + text("Matches {count} earlier recorded blocks.", count=count) + "</span></p>")
    ids = [_literal(identifier, 96) for identifier in source_ids[:5] if isinstance(identifier, str)]
    if ids:
        result += ("<p class='pc-text-repeat-note'>" + text("Repeats content from") + ": <code>"
                   + html.escape(", ".join(ids)) + (" …" if len(source_ids) > len(ids) else "") + "</code></p>")
    return result


def _source_html(block: dict[str, Any]) -> str:
    refs = block.get("source_refs")
    if not isinstance(refs, list):
        return ""
    labels = []
    for source in refs[:3]:
        if not isinstance(source, dict):
            continue
        label = _literal(source.get("source_path") or source.get("reference") or source.get("id"), 240)
        index = source.get("raw_index")
        if type(index) is int:
            label += f" : {index}"
        field_name = _literal(source.get("field"), 80)
        field_index = source.get("index")
        if field_name:
            if type(field_index) is int:
                field_name += f"[{field_index}]"
            label += (" · " if label else "") + field_name
        step = source.get("step_index")
        if type(step) is int:
            label += (" · " if label else "") + f"#{step}"
        if not label:
            label = _literal(source.get("request_id"), 160)
        if label:
            labels.append(label)
    return ("<p class='pc-text-source'>" + text("Source") + ": <code>"
            + html.escape(" · ".join(labels)) + "</code></p>") if labels else ""


def render_component_text(component: dict[str, Any], detail: dict[str, Any]) -> str:
    """Render one snapshot component inside its existing outer disclosure."""
    if not isinstance(component, dict):
        return ""
    evolution = detail.get("text_evolution")
    evolution = evolution if isinstance(evolution, dict) else {}
    identity = _literal(component.get("id"), 1000)
    rewrite = next((item for item in _proved_rewrites(detail, evolution)
                    if identity and item.get("after_id") == identity), None)
    if rewrite is not None:
        before, before_short = _preview(rewrite["before_text"], _MAX_DIFF_CHARS)
        after, after_short = _preview(rewrite["after_text"], _MAX_DIFF_CHARS)
        _, after_html, limited = _diff(before, after)
        return ("<div class='pc-text-block pc-text-rewrite'><div class='pc-text-labels'>"
                + text("Recorded rewrite") + "</div><pre class='pc-text-content'>" + after_html + "</pre>"
                + ("<p class='pc-text-note'>" + text(
                    "Inline diff exceeds the preview budget; before and after previews are shown.") + "</p>" if limited else "")
                + (_shortened() if before_short or after_short or rewrite.get("before_truncated")
                   or rewrite.get("after_truncated") else "") + "</div>")
    changed = evolution.get("blocks")
    changed = changed if isinstance(changed, list) else []
    overlay = {_block_key(block): block for block in changed
               if isinstance(block, dict) and block.get("change") == "added"}
    default = "retained" if evolution.get("comparison_available") is True and not evolution.get("blocks_truncated") else "uncompared"
    pieces = []
    remaining = _MAX_COMPONENT_CHARS
    blocks = _readable(component)
    shortened_any = len(blocks) > _MAX_COMPONENT_BLOCKS
    rendered = 0
    for block in blocks[:_MAX_COMPONENT_BLOCKS]:
        content, shortened = _preview(block.get("text"), remaining)
        remaining -= len(content)
        if not content and remaining <= 0:
            break
        rendered += 1
        shortened_any |= shortened or bool(block.get("text_truncated"))
        changed_block = overlay.get(_block_key(block), {})
        repeated = changed_block.get("repeated") is True
        status = "repeated" if repeated else "added" if changed_block else default
        pieces.append("<div class='pc-text-block pc-text-" + status + "'><div class='pc-text-labels'>"
                      + _status(status) + "</div><pre class='pc-text-content'>" + html.escape(content)
                      + "</pre>" + _repeat_html(changed_block) + "</div>")
        if remaining <= 0:
            break
    if not pieces:
        return "<p class='pc-text-note'>" + text("No readable text was recorded for this component.") + "</p>"
    return "".join(pieces) + (_shortened() if shortened_any or rendered < len(blocks) else "")


def _tokens(value: str) -> tuple[list[str], bool]:
    tokens = []
    for match in _WORDS.finditer(value):
        if len(tokens) >= _MAX_DIFF_TOKENS:
            return tokens, True
        tokens.append(match.group(0))
    return tokens, False


def _diff(before: str, after: str) -> tuple[str, str, bool]:
    old, old_limit = _tokens(before)
    new, new_limit = _tokens(after)
    if old_limit or new_limit or len(old) + len(new) > _MAX_DIFF_TOKENS:
        # An uncomputed alignment cannot establish which words changed.
        return ("<span class='pc-text-uncompared'>" + html.escape(before) + "</span>",
                "<span class='pc-text-uncompared'>" + html.escape(after) + "</span>", True)
    left, right = [], []
    for operation, a, b, c, d in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        old_text, new_text = html.escape("".join(old[a:b])), html.escape("".join(new[c:d]))
        if operation == "equal":
            left.append("<span class='pc-text-retained'>" + old_text + "</span>")
            right.append("<span class='pc-text-retained'>" + new_text + "</span>")
        if operation in ("delete", "replace"):
            left.append("<del class='pc-text-removed'>" + old_text + "</del>")
        if operation in ("insert", "replace"):
            right.append("<ins class='pc-text-added'>" + new_text + "</ins>")
    return "".join(left), "".join(right), False


def _legacy_blocks(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Readable fallback for saved older details; never invent rewrite pairs."""
    changes = detail.get("changes")
    if not isinstance(changes, dict) or changes.get("available") is not True:
        return []
    blocks = []
    for status in ("added", "removed"):
        components = changes.get(f"{status}_components")
        if isinstance(components, list):
            for component in components[:_MAX_CHANGE_CARDS]:
                if isinstance(component, dict):
                    blocks.extend({**block, "change": status} for block in _readable(component))
    return blocks


def _proved_rewrites(detail: dict[str, Any], evolution: dict[str, Any]) -> list[dict[str, Any]]:
    rewrites = evolution.get("rewrites")
    selected = [rewrite for rewrite in rewrites if isinstance(rewrite, dict)
                and isinstance(rewrite.get("before_text"), str) and isinstance(rewrite.get("after_text"), str)
                and isinstance(rewrite.get("before_id"), str) and isinstance(rewrite.get("after_id"), str)] \
        if (isinstance(rewrites, list) and evolution.get("comparison_available") is True
            and evolution.get("evidence") != "estimated" and detail.get("scope") != "trajectory_estimate") else []
    if evolution:
        return selected
    # Older exported details already provided explicitly identified rewrites.
    # Show their literal before/after values without inferring a new pairing.
    changes = detail.get("changes")
    if (not isinstance(changes, dict) or changes.get("available") is not True
            or changes.get("evidence") == "estimated" or detail.get("scope") == "trajectory_estimate"):
        return []
    old = changes.get("rewritten_components")
    if not isinstance(old, list):
        return []
    return [{**item, "before_text": item["before"], "after_text": item["after"], "legacy_preview": True}
            for item in old if isinstance(item, dict) and isinstance(item.get("identity"), str)
            and item["identity"] and isinstance(item.get("before"), str) and isinstance(item.get("after"), str)]


def render_text_evolution(detail: dict[str, Any]) -> str:
    """Render changed readable blocks and only explicitly supplied rewrites."""
    if not isinstance(detail, dict):
        return ""
    evolution = detail.get("text_evolution")
    evolution = evolution if isinstance(evolution, dict) else {}
    values = evolution.get("blocks")
    blocks = [block for block in values if isinstance(block, dict) and block.get("change") in ("added", "removed")] \
        if isinstance(values, list) else _legacy_blocks(detail)
    rewrites = _proved_rewrites(detail, evolution)
    if not blocks and not rewrites:
        return "<p class='pc-text-note'>" + text("No changed prompt text is available.") + "</p>"
    pieces = ["<div class='pc-text-list'>"]
    budget = _MAX_CHANGE_CHARS
    shown = 0
    for block in blocks[:_MAX_CHANGE_CARDS]:
        if budget <= 0:
            break
        content, shortened = _preview(block.get("text"), min(_MAX_BLOCK_CHARS, budget))
        budget -= len(content)
        status = "repeated" if block.get("repeated") is True else block["change"]
        kind = _KINDS.get(_literal(block.get("kind")), "Message")
        name = _literal(block.get("tool_name"), 120)
        if block.get("kind") == "text":
            if block.get("role") == "system":
                kind = "System instructions"
            elif block.get("role") == "developer":
                kind = "Developer"
            elif name:
                kind = "Tool definition"
        title = text(kind) + (" <code>" + html.escape(name) + "</code>" if name else "")
        pieces.append("<details class='pc-text-card pc-text-" + status + "'" + (" open" if shown < 3 else "")
                      + "><summary>" + _status(block["change"]) + (_status("repeated") if status == "repeated" else "")
                      + title + "</summary><pre class='pc-text-content'>" + html.escape(content) + "</pre>"
                      + _repeat_html(block) + _source_html(block)
                      + (_shortened() if shortened or block.get("text_truncated") else "") + "</details>")
        shown += 1
    for index, rewrite in enumerate(rewrites[:_MAX_REWRITES]):
        before, before_short = _preview(rewrite["before_text"], _MAX_DIFF_CHARS)
        after, after_short = _preview(rewrite["after_text"], _MAX_DIFF_CHARS)
        old_html, new_html, diff_limit = (html.escape(before), html.escape(after), False) \
            if rewrite.get("legacy_preview") else _diff(before, after)
        label = _literal(rewrite.get("label"), 180)
        pieces.append("<details class='pc-text-card pc-text-rewrite'" + (" open" if index < 2 else "")
                      + "><summary>" + text("Recorded rewrite") + (" <code>" + html.escape(label) + "</code>" if label else "")
                      + "</summary><div class='pc-text-rewrite-grid'><section><h4>" + text("Before rewrite")
                      + "</h4><pre class='pc-text-content'>" + old_html + "</pre></section><section><h4>"
                      + text("After rewrite") + "</h4><pre class='pc-text-content'>" + new_html
                      + "</pre></section></div>" + _source_html(rewrite)
                      + ("<p class='pc-text-note'>" + text(
                          "Inline diff exceeds the preview budget; before and after previews are shown.") + "</p>" if diff_limit else "")
                      + (_shortened() if before_short or after_short or rewrite.get("before_truncated")
                         or rewrite.get("after_truncated") else "") + "</details>")
        shown += 1
    pieces.append("</div>")
    total = len(blocks) + len(rewrites)
    if shown < total or evolution.get("blocks_truncated"):
        pieces.append("<p class='pc-text-note'>" + text(
            "Showing {shown} of {total} text changes. Evidence JSON keeps the full recorded data.",
            shown=shown, total=total) + "</p>")
    if any(block.get("repeated") is True for block in blocks[:_MAX_CHANGE_CARDS]):
        pieces.append("<p class='pc-text-note'>" + text(
            "Repeat markers describe matching recorded text; they do not establish unnecessary work.") + "</p>")
    return "".join(pieces)
