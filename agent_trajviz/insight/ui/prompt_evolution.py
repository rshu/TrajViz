"""Workflow-aligned prompt events and measured sizes for the dashboard."""

from __future__ import annotations

import html
from collections import defaultdict

from ...i18n import translate

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
}


def event_label(event: dict) -> str:
    """Return authored prose separately from native identifiers."""
    return EVENT_LABELS.get(event.get("kind"), "Recorded context event")


def _lane(event: dict) -> str:
    kind = event.get("kind", "")
    if kind.startswith("tool_") or kind == "tools_changed":
        return "Tools"
    if kind.startswith("system_") or kind == "system_changed" or kind == "prompt_shape_recorded":
        return "Instructions"
    if kind in {"context_edit", "compaction", "branch_summary", "context_reset"}:
        return "Context edits / compaction"
    return "Request sequence"


def evolution_chart(data: dict, selected_id: str, language: str = "en"):
    """Plot measured sizes only; unknown sizes still have event markers."""
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    rows = data.get("evolution") or []
    if not rows:
        return None
    measured = any(row.get("observed_prompt_tokens") is not None for row in rows)
    figure = (make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
              if measured else go.Figure())
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
            grouped[_lane(event)].append(name)
        for lane, labels in grouped.items():
            lanes[lane].append((row, "<br>".join(labels) or translate("Response boundary", language)))
    for lane, points in lanes.items():
        if not points:
            continue
        color = {"Request sequence": "#64748b", "Instructions": "#7c3aed", "Tools": "#0891b2",
                 "Context edits / compaction": "#d97706"}[lane]
        trace = go.Scatter(
            x=[row["order"] for row, _ in points], y=[translate(lane, language)] * len(points),
            mode="markers", name=translate(lane, language), showlegend=False,
            marker={"color": color, "symbol": "circle" if lane == "Request sequence" else "diamond", "size": 9,
                    "line": {"width": [2 if row["id"] == selected_id else 0 for row, _ in points], "color": "#0f172a"}},
            customdata=[[row["id"], label] for row, label in points],
            hovertemplate=translate("Boundary", language) + " #%{x}<br>%{customdata[1]}<extra></extra>",
        )
        if measured:
            figure.add_trace(trace, row=2, col=1)
        else:
            figure.add_trace(trace)
    figure.update_layout(
        height=380 if measured else 230, margin={"l": 145, "r": 18, "t": 24, "b": 46},
        paper_bgcolor="white", plot_bgcolor="white", font={"color": "#334155", "size": 11},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        clickmode="event", uirevision="prompt-evolution", meta={"tv_prompt_evolution": True},
    )
    selected = next((row for row in rows if row["id"] == selected_id), None)
    if selected:
        figure.add_vline(x=selected["order"], line_width=1, line_dash="dot", line_color="#2563eb")
    figure.update_xaxes(title_text=translate("Workflow response boundary", language), tickformat="d",
                        gridcolor="#e5eaf2", **({"row": 2, "col": 1} if measured else {}))
    figure.update_yaxes(type="category", categoryorder="array",
                        categoryarray=[translate(lane, language) for lane in reversed(lanes) if lanes[lane]],
                        **({"row": 2, "col": 1} if measured else {}))
    return figure
