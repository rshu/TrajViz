"""Inspect prompt evidence separately from conversation turns and billed usage."""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from typing import Any

import gradio as gr

from ...i18n import text, translate
from ..session import MAX_STEPS
from .prompt_evolution import event_label, evolution_chart
from .shared import SharedState

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Follow how prompt context evolves across the workflow. Select a boundary to see "
    "what was added, removed or rewritten, then inspect the supporting snapshot."
)
CSS = """
body:has(#prompt-construction-tab-button[aria-selected="true"]) .overview-kpi-strip {display:none!important;}
.pc-summary {border:1px solid #dce3ef;border-radius:12px;padding:18px;background:#f8fafc;color:#0f172a;}
.pc-heading {display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
.pc-heading strong {font-size:17px;}
.pc-badge {display:inline-block;border-radius:999px;padding:3px 9px;font-size:12px;background:#dbeafe;color:#1e40af;}
.pc-badge.pc-estimated {background:#fef3c7;color:#92400e;}
.pc-badge.pc-partial {background:#fef3c7;color:#92400e;}
.pc-stats {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;}
.pc-stats>div {min-width:0;}.pc-stats small {display:block;color:#475569;font-size:12px;}
.pc-stats b {display:block;font-size:20px;margin-top:4px;overflow-wrap:anywhere;}
.pc-meta {display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;color:#475569;font-size:12px;overflow-wrap:anywhere;}
.pc-note {padding:12px 14px;border:1px solid #dce3ef;border-radius:8px;background:#f8fafc;color:#334155;}
.pc-note ul {margin:0;padding-left:20px;}.pc-note li+li {margin-top:6px;}
.pc-parts {display:flex;flex-direction:column;gap:8px;}
.pc-parts details {border:1px solid #dce3ef;border-radius:8px;padding:10px 12px;min-width:0;}
.pc-parts summary {cursor:pointer;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;color:#0f172a;}
.pc-parts summary small {margin-left:auto;color:#475569;}
.pc-parts pre {max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc!important;color:#0f172a!important;padding:12px!important;border-radius:6px;line-height:1.5;font-size:12px;}
.pc-parts .pc-source {font-size:11px;color:#475569;overflow-wrap:anywhere;margin-top:8px;}
.pc-change-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;}
.pc-change-grid section {border:1px solid #dce3ef;border-radius:8px;padding:14px;overflow-wrap:anywhere;}
.pc-change-grid section:has(details[open]) {grid-column:1 / -1;}
.pc-change-grid pre {max-height:250px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;color:#0f172a;padding:12px;}
.pc-rewrite {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
.pc-rewrite pre {max-height:250px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;color:#0f172a;}
.pc-bridge {display:none!important;}
.pc-navigation {align-items:flex-end!important;}.pc-navigation>button {margin-bottom:10px;}
@media(max-width:640px){.pc-stats,.pc-change-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.pc-summary{padding:12px;}}
@media(max-width:520px){.pc-rewrite{grid-template-columns:1fr;}}
"""

_EVIDENCE = {"captured": "Captured", "reconstructed": "Reconstructed", "estimated": "Estimated",
             "unavailable": "Unavailable"}
_SCOPES = {"recorded_request": "Recorded request", "native_prompt_state": "Recorded prompt state",
           "trajectory_estimate": "Transcript estimate"}


@dataclass(frozen=True)
class PromptRefs:
    request: gr.Dropdown
    chart: gr.Plot
    summary: gr.HTML
    components: gr.HTML
    changes: gr.HTML
    detail: gr.JSON
    previous: gr.Button
    next: gr.Button
    bridge: gr.Textbox
    select: gr.Button


def layout() -> PromptRefs:
    """Build a bounded, local-only prompt inspector."""
    with gr.Tab("Prompt Evolution", elem_id="prompt-construction-tab"):
        gr.Markdown(DESCRIPTION)
        chart = gr.Plot(show_label=False, visible=False, elem_id="prompt-evolution-chart")
        gr.Markdown("Click a timeline point or use Previous / Next. Only recorded sizes are plotted; unknown sizes stay blank.")
        with gr.Row(elem_classes=["pc-navigation"]):
            previous = gr.Button("Previous", scale=0, min_width=90)
            request = gr.Dropdown(label="Model request or response boundary", choices=[], value=None,
                                  interactive=False, elem_id="prompt-request-selector", scale=4)
            next_button = gr.Button("Next", scale=0, min_width=90)
        bridge = gr.Textbox(value="", elem_id="pc-request-bridge", elem_classes=["pc-bridge"])
        select = gr.Button("Select prompt request", elem_id="pc-select-request", elem_classes=["pc-bridge"])
        summary = gr.HTML(_empty(), elem_id="prompt-summary")
        with gr.Tabs(selected="changes"):
            with gr.Tab("Changes", id="changes"):
                changes = gr.HTML("", elem_id="prompt-changes")
            with gr.Tab("Prompt snapshot", id="snapshot"):
                components = gr.HTML("", elem_id="prompt-components")
            with gr.Tab("Evidence JSON", id="evidence"):
                gr.Markdown("Original prompt content is shown as recorded and is never translated.")
                detail = gr.JSON(value=None, show_label=False, elem_id="prompt-request-json")
    return PromptRefs(request, chart, summary, components, changes, detail,
                      previous, next_button, bridge, select)


def _empty() -> str:
    return "<div class='pc-note' role='status'>" + text("Load a trajectory to inspect prompt evidence.") + "</div>"


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _summary(detail: dict, run: Any) -> str:
    evidence = detail.get("evidence", "unavailable")
    missing = detail.get("missing_references", [])
    incomplete = bool(missing or detail.get("unsupported_fields"))
    badge = "Partial reconstruction" if incomplete and evidence == "reconstructed" else _EVIDENCE.get(evidence, "Unavailable")
    badge_class = "partial" if incomplete else str(evidence)
    scope = _SCOPES.get(detail.get("scope"), "Unavailable")
    estimate = detail.get("token_estimate", {})
    if isinstance(estimate, dict):
        tokens = estimate.get("total", estimate.get("estimated_tokens"))
    else:
        tokens = estimate
    if tokens is None and detail.get("components"):
        tokens = sum(c.get("estimated_tokens", 0) or 0 for c in detail.get("components", []))
    token_label = f"≈ {tokens:,}" if isinstance(tokens, (float, int)) and not isinstance(tokens, bool) else "—"
    if incomplete and not detail.get("components"):
        token_label = "—"
    message_count = len(detail.get("messages", []))
    tool_count = len(detail.get("tools", []))
    if not message_count and any(str(ref).startswith(("blob:", "messages:", "base:", "prefix:")) for ref in missing):
        message_count = "—"
    if not tool_count and any(str(ref).startswith("shape:") for ref in missing):
        tool_count = "—"
    counts = [("Messages", message_count), ("Tools", tool_count),
              ("Components", len(detail.get("components", []))), ("Estimated visible tokens", token_label)]
    stats = "".join("<div><small>" + text(label) + "</small><b>" + html.escape(str(value)) + "</b></div>"
                    for label, value in counts)
    step = detail.get("step_index")
    shown = {s.get("index") for s in run.steps[:MAX_STEPS]}
    link = (f"<a href='#tv-step-{step}'>" + text("Open step #{step} in Workflow", step=step) + "</a>"
            if type(step) is int and step in shown else "")
    metadata = [detail.get("model") or "", detail.get("agent_id") or detail.get("session_id") or ""]
    meta = "".join("<span>" + html.escape(str(value)) + "</span>" for value in metadata if value)
    return ("<div class='pc-summary'><div class='pc-heading'><strong>" + text(scope)
            + "</strong><span class='pc-badge pc-" + html.escape(badge_class, quote=True) + "'>"
            + text(badge) + "</span></div><div class='pc-stats'>"
            + stats + "</div><div class='pc-meta'>" + meta + link + "</div></div>")


def _components(detail: dict) -> str:
    parts = detail.get("components", [])
    if not parts:
        return "<div class='pc-note'>" + text("No prompt components could be recovered.") + "</div>"
    visible = list(enumerate(parts))
    note = ""
    if len(visible) > 40:
        visible = visible[:20] + visible[-20:]
        note = "<div class='pc-note'>" + text(
            "Showing {shown} of {total} components. Request JSON contains the full recorded data.",
            shown=len(visible), total=len(parts)) + "</div>"
    body = []
    for index, part in visible:
        content = _json(part.get("content", ""))
        shortened = len(content) > 6000
        content = content[:6000] if shortened else content
        kind = str(part.get("kind") or "message")
        name = {"system": "System instructions", "tool": "Tool definition", "message": "Message",
                "summary": "Context summary"}.get(kind, "Component")
        role = part.get("role")
        role_html = (" · " + text({"user": "User", "assistant": "Assistant", "system": "System",
                                   "developer": "Developer", "toolResult": "Tool result"}.get(role, str(role)))) if role else ""
        tokens = part.get("estimated_tokens")
        token_text = "≈ " + f"{tokens:,}" if isinstance(tokens, (int, float)) else "—"
        source = part.get("sources", part.get("source_refs", []))
        body.append("<details><summary><b>" + text(name) + " " + str(index + 1) + "</b>" + role_html
                    + "<small>" + html.escape(token_text) + " tok</small></summary><pre>"
                    + html.escape(content) + "</pre>" + ("<p>" + text(
                        "Preview shortened. Request JSON contains the full recorded data.") + "</p>" if shortened else "")
                    + ("<div class='pc-source'>" + text("Source") + ": <code>"
                       + html.escape(_json(source)) + "</code></div>" if source else "") + "</details>")
    return note + "<div class='pc-parts'>" + "".join(body) + "</div>"


def _recorded_events(detail: dict) -> str:
    events = detail.get("prompt_events", [])
    if not events:
        return "<p>" + text("No prompt assembly events were recorded for this boundary.") + "</p>"
    items = []
    for event in events[:30]:
        label = text(event_label(event))
        target = event.get("tool_name") or event.get("target_id") or event.get("section") or event.get("name")
        if target:
            label += " <code>" + html.escape(str(target)) + "</code>"
        count = event.get("count")
        if type(count) is int and count > 1:
            label += f" ×{count}"
        items.append("<li>" + label + "</li>")
    heading = ("Observed transcript events before this boundary"
               if any(event.get("evidence") == "observed_transcript" for event in events)
               else "Recorded changes before this boundary")
    return "<div class='pc-note'><strong>" + text(heading) + "</strong><ul>" + "".join(items) + "</ul></div>"


def _component_label_html(component: dict) -> str:
    """Translate generated message roles while keeping source names literal."""
    label = str(component.get("label") or component.get("identity") or "")
    after = component.get("after")
    role = component.get("role") or (after.get("role") if isinstance(after, dict) else None)
    role_name = {"user": "User", "assistant": "Assistant", "system": "System",
                 "developer": "Developer", "toolResult": "Tool result", "tool": "Tool result",
                 "compactionSummary": "Context summary", "branchSummary": "Branch summary"}.get(role)
    if component.get("kind") == "message" and role_name and label.startswith(str(role) + " "):
        suffix = label[len(str(role)):]
        if suffix.strip().isdigit():
            return text(role_name) + html.escape(suffix)
    return "<code>" + html.escape(label) + "</code>"


def _changes(detail: dict) -> str:
    events = _recorded_events(detail)
    changes = detail.get("changes") or {}
    if not changes.get("available"):
        return events + "<div class='pc-note'>" + text(str(changes.get("reason") or
            "No previous request with verified lineage is available for comparison.")) + "</div>"
    previous = changes.get("previous_request_id") or detail.get("previous_request_id") or ""
    comparison = ("Compared with the previous estimated transcript boundary for this agent."
                  if changes.get("evidence") == "estimated" else "Compared with the previous request in the same recorded lineage.")
    pieces = [events]
    if changes.get("evidence") == "estimated":
        pieces.append("<p><span class='pc-badge pc-estimated'>" + text("Estimated candidate changes") + "</span> "
                      + text("These changes compare transcript candidates, not verified model requests.") + "</p>")
    pieces += ["<p>" + text(comparison)
              + " <code>" + html.escape(str(previous)) + "</code></p>"]
    delta = changes.get("estimated_token_delta")
    if type(delta) is int:
        pieces.append("<p>" + text("Estimated token change: {delta}", delta=f"{delta:+,}") + "</p>")
    if changes.get("order_changed"):
        pieces.append("<p>" + text("The order of retained components changed.") + "</p>")
    pieces.append("<div class='pc-change-grid'>")
    for key, label in (("added", "Added components"), ("removed", "Removed components"),
                       ("unchanged", "Unchanged components"), ("rewritten", "Rewritten components")):
        values = (changes.get("retained", []) if key == "unchanged" else changes.get("rewritten_components", [])
                  if key == "rewritten" else changes.get(key, []))
        count = len(values) if isinstance(values, list) else values if type(values) is int else 0
        summaries = changes.get(f"{key}_components", [])
        listed = []
        for component in summaries[:25]:
            kind = component.get("kind")
            component_label = "Tool definition" if kind == "tool" else "System instructions" if kind == "system" else "Message"
            preview = component.get("preview")
            listed.append("<li>" + text(component_label) + " " + _component_label_html(component)
                          + ("<details><summary>" + text("Content preview") + "</summary><pre>"
                             + html.escape(str(preview)) + "</pre></details>" if preview else "") + "</li>")
        pieces.append("<section><strong>" + text(label) + "</strong><p>" + str(count) + "</p>"
                      + ("<ul>" + "".join(listed) + "</ul>" if listed else "") + "</section>")
    pieces.append("</div><p>" + text(
        "Changes compare recorded content; they do not establish what influenced the model.") + "</p>")
    rewritten = changes.get("rewritten_components", [])
    if rewritten:
        pieces.append("<p>" + text("Rewrites also appear in the added and removed content counts.") + "</p>")
    for component in rewritten[:12]:
        pieces.append("<details class='pc-parts'><summary>" + text("Rewritten component") + " "
                      + _component_label_html(component) + "</summary><div class='pc-rewrite'>")
        for key, label in (("before", "Before rewrite"), ("after", "After rewrite")):
            content = _json(component.get(key))
            pieces.append("<div><strong>" + text(label) + "</strong><pre>" + html.escape(content[:6000])
                          + "</pre>" + ("<p>" + text("Preview shortened. Request JSON contains the full recorded data.")
                                        + "</p>" if len(content) > 6000 else "") + "</div>")
        pieces.append("</div></details>")
    return "".join(pieces)


def outputs(run: Any, request_id: str | None = None, language: str = "en") -> tuple:
    """Render one selection, clearing every previous result on unload or failure."""
    empty = (gr.update(choices=[], value=None, interactive=False), gr.update(value=None, visible=False), _empty(), "", "", None)
    if run is None:
        return empty
    try:
        data = run.view("prompt_construction", **({"request_id": request_id} if request_id else {})).data
        rows = data.get("requests", [])
        detail = data.get("selected_request")
        if not detail:
            return (empty[0], empty[1], "<div class='pc-note'>" + text("Prompt evidence is unavailable for this trajectory.")
                    + "</div>", "", "", None)
        choices = []
        for index, row in enumerate(rows):
            scope = translate(_SCOPES.get(row.get("scope"), "Unavailable"), language)
            agent = row.get("agent_id") or row.get("session_id") or "main"
            choices.append((f"#{index + 1} · {agent} · {scope}", row["id"]))
        figure = evolution_chart(data, detail["id"], language)
        return (gr.update(choices=choices, value=detail["id"], interactive=True),
                gr.update(value=figure, visible=figure is not None), _summary(detail, run),
                _components(detail), _changes(detail), detail)
    except Exception:  # noqa: BLE001 -- never leave stale evidence from another loaded run
        logger.exception("Prompt construction inspection failed")
        return (empty[0], empty[1], "<div class='pc-note' role='status'>" + text(
            "Prompt inspection could not be completed. Reload the trajectory; details are in the server log.")
                + "</div>", "", "", None)


def bind(refs: PromptRefs, shared: SharedState, load_events, *, language: Any) -> None:
    """Refresh after loads, explicit request selections and language changes."""
    targets = [refs.request, refs.chart, refs.summary, refs.components, refs.changes, refs.detail]

    def on_prompt_load(run, locale="en"):
        return outputs(run, language=locale)

    def on_prompt_select(run, request_id, locale="en"):
        return outputs(run, request_id, locale)

    def move(run, request_id, locale, offset):
        if run is None:
            return outputs(None)
        data = run.view("prompt_construction").data
        rows = data.get("requests", [])
        if not rows:
            return outputs(run, language=locale)
        index = next((i for i, row in enumerate(rows) if row["id"] == request_id), len(rows) - 1)
        wanted = rows[min(max(index + offset, 0), len(rows) - 1)]["id"]
        return outputs(run, wanted, locale)

    def on_prompt_previous(run, request_id, locale="en"):
        return move(run, request_id, locale, -1)

    def on_prompt_next(run, request_id, locale="en"):
        return move(run, request_id, locale, 1)

    for event in load_events:
        event.then(fn=on_prompt_load, inputs=[shared.state_run, language], outputs=targets,
                   concurrency_id="prompt-construction", trigger_mode="always_last", api_name=False)
    refs.request.input(fn=on_prompt_select, inputs=[shared.state_run, refs.request, language], outputs=targets,
                       concurrency_id="prompt-construction", trigger_mode="always_last", api_name=False)
    language.change(fn=on_prompt_select, inputs=[shared.state_run, refs.request, language], outputs=targets,
                    concurrency_id="prompt-construction", trigger_mode="always_last", api_name=False)
    for button, callback in ((refs.previous, on_prompt_previous), (refs.next, on_prompt_next)):
        button.click(fn=callback, inputs=[shared.state_run, refs.request, language], outputs=targets,
                     concurrency_id="prompt-construction", trigger_mode="always_last", api_name=False)
    refs.select.click(fn=on_prompt_select, inputs=[shared.state_run, refs.bridge, language], outputs=targets,
                      concurrency_id="prompt-construction", trigger_mode="always_last", api_name=False)
