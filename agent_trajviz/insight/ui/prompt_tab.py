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
from . import prompt_composition, prompt_text
from .prompt_evolution import event_label, evolution_chart
from .shared import SharedState, is_latest_run

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
.pc-bridge {display:none!important;}
.pc-navigation {align-items:flex-end!important;}.pc-navigation>button {margin-bottom:10px;}
@media(max-width:640px){.pc-stats,.pc-change-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.pc-summary{padding:12px;}}
""" + prompt_text.CSS + prompt_composition.CSS

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
        gr.Markdown("Click a timeline point or use Previous / Next. Recorded occupancy and estimated category sizes are shown separately; unknown sizes stay blank.")
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
    quality = detail.get("evidence_quality") or {}
    missing = quality.get("missing_references", detail.get("missing_references", []))
    incomplete = quality.get("content_complete") is False or bool(missing)
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
    token_label = "≈ " + prompt_composition.compact_count(tokens) if isinstance(tokens, (float, int)) and not isinstance(tokens, bool) else "—"
    if incomplete and not detail.get("components"):
        token_label = "—"
    message_count = len(detail.get("messages", []))
    tool_count = len(detail.get("tools", []))
    if not message_count and any((ref.get("kind") in {"message", "base", "prefix", "catalog"} if isinstance(ref, dict)
                                  else str(ref).startswith(("blob:", "messages:", "base:", "prefix:", "catalog:"))) for ref in missing):
        message_count = "—"
    if not tool_count and any((ref.get("kind") == "shape" if isinstance(ref, dict)
                              else str(ref).startswith("shape:")) for ref in missing):
        tool_count = "—"
    counts = [("Messages", message_count), ("Tools", tool_count),
              ("Components", len(detail.get("components", []))), ("Estimated visible tokens", token_label)]
    stats = "".join("<div><small>" + text(label) + "</small><b title='"
                    + html.escape(str(tokens if label == "Estimated visible tokens" and type(tokens) in (int, float) else value), quote=True)
                    + "'>" + html.escape(prompt_composition.compact_count(value)) + "</b></div>" for label, value in counts)
    step = detail.get("step_index")
    shown = {s.get("index") for s in run.steps[:MAX_STEPS]}
    link = (f"<a href='#tv-step-{step}'>" + text("Open step #{step} in Workflow", step=step) + "</a>"
            if type(step) is int and step in shown else "")
    metadata = [detail.get("model") or "", detail.get("agent_id") or detail.get("session_id") or ""]
    meta = "".join("<span>" + html.escape(str(value)) + "</span>" for value in metadata if value)
    return ("<div class='pc-summary'><div class='pc-heading'><strong>" + text(scope)
            + "</strong><span class='pc-badge pc-" + html.escape(badge_class, quote=True) + "'>"
            + text(badge) + "</span></div><div class='pc-stats'>"
            + stats + "</div><div class='pc-meta'>" + meta + link + "</div>"
            + prompt_composition.quality_note(detail) + "</div>")


def _components(detail: dict, history: dict | None = None) -> str:
    return prompt_text.legend() + prompt_composition.render_component_explorer(detail, history)


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
            listed.append("<li>" + text(component_label) + " " + _component_label_html(component) + "</li>")
        pieces.append("<section><strong>" + text(label) + "</strong><p>" + str(count) + "</p>"
                      + ("<ul>" + "".join(listed) + "</ul>" if listed else "") + "</section>")
    pieces.append("</div><p>" + text(
        "Changes compare recorded content; they do not establish what influenced the model.") + "</p>")
    rewritten = changes.get("rewritten_components", [])
    if rewritten:
        pieces.append("<p>" + text("Rewrites also appear in the added and removed content counts.") + "</p>")
    return "".join(pieces) + prompt_text.legend() + prompt_text.render_text_evolution(detail)


def outputs(run: Any, request_id: str | None = None, language: str = "en") -> tuple:
    """Render one selection, clearing every previous result on unload or failure."""
    empty = (gr.update(choices=[], value=None, interactive=False), gr.update(value=None, visible=False), _empty(), "", "", None)
    if run is None:
        return empty
    try:
        try:
            data = run.view("prompt_construction", **({"request_id": request_id} if request_id else {})).data
        except ValueError as exc:
            # A queued selector/language event can still carry the old run's id.
            # Recover against the current run rather than emptying valid choices.
            unknown = (str(exc).startswith("Unknown prompt request id:")
                       or str(exc) == "Unknown prompt construction request_id")
            if not request_id or not unknown:
                raise
            data = run.view("prompt_construction").data
        rows = data.get("requests", [])
        detail = data.get("selected_request")
        if not detail:
            return (empty[0], empty[1], "<div class='pc-note'>" + text("Prompt evidence is unavailable for this trajectory.")
                    + "</div>", "", "", None)
        choices = []
        actors = {row["id"]: row for row in data.get("evolution", [])}
        for index, row in enumerate(rows):
            scope = translate(_SCOPES.get(row.get("scope"), "Unavailable"), language)
            actor = actors.get(row["id"], row)
            name = actor.get("actor_name") or actor.get("agent_id")
            if actor.get("actor_kind") == "unknown_child":
                agent = translate("Unidentified subagent", language)
            elif actor.get("actor_kind") == "main" or not name or name == "main":
                agent = translate("Main agent", language)
            else:
                agent = translate("Subagent {agent}", language, agent=str(name)[:48])
            choices.append((f"#{index + 1} · {agent} · {scope}", row["id"]))
        figure = evolution_chart(data, detail["id"], language)
        return (gr.update(choices=choices, value=detail["id"], interactive=True),
                gr.update(value=figure, visible=figure is not None), _summary(detail, run),
                _components(detail, data.get("composition_history")), _changes(detail), detail)
    except Exception:  # noqa: BLE001 -- never leave stale evidence from another loaded run
        logger.exception("Prompt construction inspection failed")
        return (empty[0], empty[1], "<div class='pc-note' role='status'>" + text(
            "Prompt inspection could not be completed. Reload the trajectory; details are in the server log.")
                + "</div>", "", "", None)


def bind(refs: PromptRefs, shared: SharedState, load_events, *, language: Any) -> None:
    """Refresh after loads, explicit request selections and language changes."""
    targets = [refs.request, refs.chart, refs.summary, refs.components, refs.changes, refs.detail]

    def current(run, request):
        return is_latest_run(getattr(request, "session_hash", None), run)

    def show(run, request_id, locale, request):
        if not current(run, request):
            return tuple(gr.skip() for _ in targets)
        result = outputs(run, request_id, locale)
        return result if current(run, request) else tuple(gr.skip() for _ in targets)

    def on_prompt_load(run, locale="en", request: gr.Request | None = None):
        return show(run, None, locale, request)

    def on_prompt_select(run, request_id, locale="en", request: gr.Request | None = None):
        return show(run, request_id, locale, request)

    def move(run, request_id, locale, offset, request):
        if not current(run, request):
            return tuple(gr.skip() for _ in targets)
        if run is None:
            return show(None, None, locale, request)
        data = run.view("prompt_construction").data
        rows = data.get("requests", [])
        if not rows:
            return show(run, None, locale, request)
        index = next((i for i, row in enumerate(rows) if row["id"] == request_id), len(rows) - 1)
        wanted = rows[min(max(index + offset, 0), len(rows) - 1)]["id"]
        return show(run, wanted, locale, request)

    def on_prompt_previous(run, request_id, locale="en", request: gr.Request | None = None):
        return move(run, request_id, locale, -1, request)

    def on_prompt_next(run, request_id, locale="en", request: gr.Request | None = None):
        return move(run, request_id, locale, 1, request)

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
