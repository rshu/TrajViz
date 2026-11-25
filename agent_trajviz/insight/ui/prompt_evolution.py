"""Workflow-aligned prompt events and measured sizes for the dashboard."""

from __future__ import annotations

import hashlib
import html
import json
from collections import defaultdict

from ...i18n import translate
from ..prompt_components import CATEGORIES

EVENT_LABELS = {
    "system_section_added": "Instruction section added",
    "system_section_updated": "Instruction section rewritten",
    "system_section_removed": "Instruction section removed",
    "system_changed": "System instructions changed",
    "tool_added": "Tool added", "tool_updated": "Tool definition rewritten", "tool_removed": "Tool removed",
    "tools_changed": "Tool definitions changed",
    "context_edit": "Context edited", "compaction": "Context compacted", "branch_summary": "Branch summarized",
    "user_messages": "User messages added", "tool_results": "Tool results added",
    "context_message": "Context message added", "request_snapshot": "Request snapshot recorded",
    "baseline_unavailable": "Previous prompt unavailable", "prompt_shape_recorded": "Prompt shape recorded",
    "context_prefix_unavailable": "Retained prefix unavailable", "message_tail_recorded": "Message tail recorded",
    "context_reset": "Context reset", "context_prefix_retained": "Context prefix retained",
    "message_tail_replaced": "Message tail replaced",
    "transcript_user_message": "User prompt recorded (transcript)",
    "transcript_system_message": "System/developer message recorded (transcript)",
    "transcript_tool_result": "Tool result recorded (transcript)",
    "transcript_compaction": "Compaction recorded (transcript)",
}


def event_label(event: dict) -> str:
    """Return authored prose separately from native identifiers."""
    return EVENT_LABELS.get(event.get("kind"), "Recorded context event")


def _lane(event: dict) -> str:
    kind = event.get("kind", "")
    if kind.startswith("tool_") or kind in {"tools_changed", "transcript_tool_result"}:
        return "Tools"
    if kind.startswith("system_") or kind in {"system_changed", "prompt_shape_recorded",
                                             "transcript_user_message", "transcript_system_message"}:
        return "Instructions"
    if kind in {"context_edit", "compaction", "branch_summary", "context_reset", "transcript_compaction"}:
        return "Context edits / compaction"
    return "Request sequence"


def _actor_label(row: dict, language: str, *, compact: bool = False) -> str:
    if row.get("actor_kind") == "unknown_child":
        return translate("Unidentified subagent", language)
    name = row.get("actor_name") or row.get("agent_id")
    if not name:
        return translate("Main agent", language)
    if compact and row.get("actor_kind") == "subagent":
        return translate("Subagent", language) + "<br>" + html.escape(str(name)[:8])
    return html.escape(str(name)[:28] if compact else str(name))


def evolution_chart(data: dict, selected_id: str, language: str = "en"):
    """Keep recorded occupancy, estimated composition and source events distinct."""
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    rows = data.get("evolution") or []
    if not rows:
        return None
    contexts = {row.get("context_id") or (row.get("session_id"), row.get("agent_id"), row.get("branch_id")) for row in rows}
    ordering = data.get("request_order") or {}
    measured = any(row.get("observed_prompt_tokens") is not None for row in rows)
    composed = any(isinstance(row.get("composition"), dict) for row in rows)
    # Preserve navigation zoom only within this request catalog and panel shape.
    # A different session or axis meaning must reset Plotly's interaction state.
    catalog = [[row.get(key) for key in ("id", "session_id", "agent_id", "branch_id", "context_id")] for row in rows]
    revision = hashlib.sha256(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:20]
    ui_revision = f"prompt-evolution:{int(measured)}:{int(composed)}:{revision}"
    panel_count = 1 + int(measured) + int(composed)
    composition_row = 1 + int(measured)
    events_row = panel_count
    heights = ([0.38, 0.38, 0.24] if panel_count == 3 else [0.6, 0.4])
    figure = (make_subplots(rows=panel_count, cols=1, shared_xaxes=True,
                            row_heights=heights, vertical_spacing=0.12)
              if panel_count > 1 else go.Figure())
    if measured:
        groups = defaultdict(list)
        for row in rows:
            key = row.get("context_id") or (row.get("session_id"), row.get("agent_id"), row.get("branch_id"))
            groups[key].append(row)
        used_names = defaultdict(int)
        for group in groups.values():
            owner = group[0]
            actor = owner.get("actor_name") or owner.get("agent_id")
            name = (translate("Unidentified subagent", language) if owner.get("actor_kind") == "unknown_child"
                    else html.escape(str(actor)) if actor else translate("Main agent", language))
            branch = owner.get("branch_id")
            if branch:
                name += " · " + html.escape(str(branch))
            used_names[name] += 1
            if used_names[name] > 1:
                name += f" · {used_names[name]}"
            trace = go.Scatter(
                x=[row["order"] for row in group], y=[row.get("observed_prompt_tokens") for row in group],
                mode="lines+markers", connectgaps=False, name=name,
                customdata=[[row["id"], html.escape(str(row.get("model") or ""))] for row in group],
                line={"width": 2}, marker={"size": 6},
                hovertemplate=translate("Boundary", language) + " #%{x}<br>"
                + translate("Observed prompt tokens", language) + ": %{y:,}<br>%{customdata[1]}<extra></extra>",
            )
            figure.add_trace(trace, row=1, col=1)
        figure.update_yaxes(title_text=translate("Observed prompt tokens", language), row=1, col=1,
                            rangemode="tozero", gridcolor="#e5eaf2")
    if composed:
        for key, label, color in CATEGORIES:
            amounts, details = [], []
            for row in rows:
                composition = row.get("composition")
                if isinstance(composition, dict):
                    amount = next((item.get("estimated_tokens") for item in composition.get("categories", [])
                                   if item.get("category") == key), 0)
                    if composition.get("partial") and composition.get("component_count") == 0:
                        amount = None
                    basis = composition.get("basis") or ""
                    status = translate("Partial composition" if composition.get("partial") else "Estimated", language)
                else:
                    amount, basis, status = None, "", translate("Unavailable", language)
                amounts.append(amount)
                actor = row.get("actor_name") or row.get("agent_id")
                owner = (translate("Unidentified subagent", language) if row.get("actor_kind") == "unknown_child"
                         else html.escape(str(actor)) if actor else translate("Main agent", language))
                if row.get("branch_id"):
                    owner += " · " + html.escape(str(row["branch_id"]))
                details.append([row["id"], owner, html.escape(str(basis)), status])
            if not any(amount for amount in amounts if amount is not None):
                continue
            figure.add_trace(go.Bar(
                x=[row["order"] for row in rows], y=amounts, name=translate(label, language),
                marker={"color": color},
                legendgroup="composition:" + key, customdata=details,
                hovertemplate=translate("Boundary", language) + " #%{x}<br>%{customdata[1]}<br>"
                + translate(label, language) + ": ≈ %{y:,}<br>%{customdata[3]}<br>%{customdata[2]}<extra></extra>",
            ), row=composition_row, col=1)
        figure.update_yaxes(title_text=translate("Estimated category tokens", language),
                            rangemode="tozero", gridcolor="#e5eaf2", row=composition_row, col=1)
        axis = figure.layout["yaxis" + (str(composition_row) if composition_row > 1 else "")]
        figure.add_annotation(x=0, y=axis.domain[1], xref="paper", yref="paper", xanchor="left",
                              yshift=18, showarrow=False,
                              text=translate("Estimated visible-content composition", language),
                              font={"size": 11, "color": "#475569"})
        history = data.get("composition_history") or {}
        partial = any((row.get("composition") or {}).get("partial") for row in rows)
        note = (translate("Composition history includes {computed} of {total} boundaries.", language,
                          computed=history.get("computed_requests", 0), total=history.get("total_requests", 0))
                if history and not history.get("complete", True) else "")
        if partial:
            note += (" · " if note else "") + translate("Partial composition", language)
        if note:
            figure.add_annotation(x=1, y=axis.domain[1], xref="paper", yref="paper", xanchor="right",
                                  yshift=18, showarrow=False, text=note,
                                  font={"size": 9, "color": "#64748b"})
    lanes = {name: [] for name in ("Request sequence", "Instructions", "Tools", "Context edits / compaction")}
    for row in rows:
        grouped: dict[str, list[str]] = defaultdict(list)
        grouped["Request sequence"] = []
        for event in row.get("prompt_events", []):
            name = translate(event_label(event), language)
            target = event.get("tool_name") or event.get("target_id") or event.get("name") or event.get("section")
            count = event.get("count")
            if target:
                name += ": " + html.escape(str(target))
            if type(count) is int and count > 1:
                name += f" ×{count}"
            source_steps = list(dict.fromkeys(ref["step_index"] for ref in event.get("source_refs", [])
                                              if isinstance(ref, dict) and type(ref.get("step_index")) is int))
            if source_steps:
                name += "<br>" + translate("Source steps: {steps}", language,
                                           steps=", ".join(f"#{step}" for step in source_steps[:3]))
            grouped[_lane(event)].append(name)
        for lane, labels in grouped.items():
            boundary = ("Estimated response boundary" if row.get("scope") == "trajectory_estimate" else "Response boundary")
            lanes[lane].append((row, "<br>".join(labels) or translate(boundary, language)))
    for lane, points in lanes.items():
        if not points:
            continue
        color = {"Request sequence": "#64748b", "Instructions": "#7c3aed", "Tools": "#0891b2",
                 "Context edits / compaction": "#d97706"}[lane]
        trace = go.Scatter(
            x=[row["order"] for row, _ in points], y=[translate(lane, language)] * len(points),
            mode="markers", name=translate(lane, language), showlegend=False,
            marker={"color": color, "symbol": [
                ("diamond" if row.get("actor_kind") == "subagent" else "x" if row.get("actor_kind") == "unknown_child"
                 else "circle") if lane == "Request sequence" else "diamond" for row, _ in points], "size": 9,
                    "line": {"width": [2 if row["id"] == selected_id else 0 for row, _ in points], "color": "#0f172a"}},
            customdata=[[row["id"], label] for row, label in points],
            hovertemplate=translate("Boundary", language) + " #%{x}<br>%{customdata[1]}<extra></extra>",
        )
        if panel_count > 1:
            figure.add_trace(trace, row=events_row, col=1)
        else:
            figure.add_trace(trace)
    height = 720 if panel_count == 3 else 520 if composed else 380 if measured else 230
    top = 85 if ordering else 50 if composed else 24
    bottom = 170 if composed else 46
    figure.update_layout(
        height=height,
        margin={"l": 145, "r": 18, "t": top, "b": bottom},
        barmode="stack",
        paper_bgcolor="white", plot_bgcolor="white", font={"color": "#334155", "size": 11},
        # Reserve a fixed gap below multi-line agent ticks and the axis title.
        # A fixed paper fraction overlaps them when Plotly changes panel height.
        legend={"orientation": "h", "y": -100 / (height - top - bottom) if composed else 1.12,
                "yanchor": "top" if composed else "auto", "x": 0},
        clickmode="event", uirevision=ui_revision, meta={"tv_prompt_evolution": True},
    )
    if any(event.get("evidence") == "observed_transcript" for row in rows for event in row.get("prompt_events", [])):
        figure.add_annotation(x=1, y=1.13, xref="paper", yref="paper", xanchor="right", showarrow=False,
                              text=translate("Transcript events; prompt membership unverified", language),
                              font={"size": 9, "color": "#64748b"})
    if ordering:
        note = translate("Boundaries ordered by recorded time" if ordering.get("chronological")
                         else "Source order; chronology unavailable", language)
        if len(contexts) > 1:
            note += "<br>" + translate("Each bar belongs to one agent context.", language)
        figure.add_annotation(x=0, y=1.24 if composed else 1.15, xref="paper", yref="paper",
                              xanchor="left", align="left", showarrow=False, text=note,
                              font={"size": 10, "color": "#475569"})
    for lane, points in lanes.items():
        if not points:
            figure.add_annotation(x=1, y=translate(lane, language),
                                  xref=f"x{events_row} domain" if panel_count > 1 else "x domain",
                                  yref=f"y{events_row}" if panel_count > 1 else "y",
                                  xanchor="right", showarrow=False, text=translate("Not recorded", language),
                                  font={"size": 10, "color": "#94a3b8"})
    selected = next((row for row in rows if row["id"] == selected_id), None)
    if selected:
        figure.add_vline(x=selected["order"], line_width=1, line_dash="dot", line_color="#2563eb")
    figure.update_xaxes(dtick=1)
    figure.update_xaxes(title_text=translate("Workflow response boundary", language), tickformat="d",
                        gridcolor="#e5eaf2", **({"row": events_row, "col": 1} if panel_count > 1 else {}))
    if composed and len(contexts) > 1:
        stride = max(1, (len(rows) + 9) // 10)
        ticks = rows[::stride]
        if ticks[-1]["id"] != rows[-1]["id"]:
            ticks = [*ticks, rows[-1]]
        figure.update_xaxes(tickmode="array", tickvals=[row["order"] for row in ticks], ticktext=[
            "#" + str(row["order"]) + "<br>" + _actor_label(row, language, compact=True)
            + (" · " + html.escape(str(row["branch_id"])[:16]) if row.get("branch_id") else "") for row in ticks],
            row=events_row, col=1)
    figure.update_yaxes(type="category", categoryorder="array",
                        categoryarray=[translate(lane, language) for lane in reversed(lanes)],
                        tickmode="array", tickvals=[translate(lane, language) for lane in lanes],
                        range=[-0.5, len(lanes) - 0.5],
                        **({"row": events_row, "col": 1} if panel_count > 1 else {}))
    return figure
