"""Phase and action label charts."""

from __future__ import annotations

from ._layout import _apply_chart_layout, _apply_dark, _empty_figure
import plotly.graph_objects as go

from ...i18n import localize_figure
from ...i18n.label_names import label_spec
from ..palette import LABEL_PHASE_COLORS

_LABEL_FONT = dict(size=13)  # consistent font size across label charts
_CANONICAL_PHASES = ["understand", "plan", "implement", "debug", "validate", "report"]


def _add_phase_legend(fig: go.Figure, phases_used: set[str]) -> dict[int, dict]:
    """Add invisible trace per phase to create a color legend."""
    canonical = ["understand", "plan", "implement", "debug", "validate", "report"]
    trace_labels = {}
    for phase in canonical:
        if phase in phases_used:
            trace_labels[len(fig.data)] = {"name": label_spec("phase", phase)}
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    marker_color=LABEL_PHASE_COLORS.get(phase, "#6b7280"),
                    name=phase,
                    showlegend=True,
                )
            )
    return trace_labels


def _build_label_bar_chart(
    data: dict[str, int | float],
    color_fn,
    title: str,
    *,
    orientation: str = "v",
    label_kind: str = "phase",
    value_format: str = "",
    xaxis: str | None = None,
    yaxis: str | None = None,
    empty_message: str = "No data",
    action_to_phase: dict[str, str] | None = None,
    canonical_order: list[str] | None = None,
) -> go.Figure:
    """Parameterized builder for label bar charts.

    Args:
        data: mapping of label -> numeric value
        color_fn: callable(label) -> hex color string
        title: chart title
        orientation: "v" for vertical, "h" for horizontal
        label_kind: builtin taxonomy namespace used only for display labels
        value_format: format spec for text labels (e.g. ".1f" for durations)
        xaxis/yaxis: axis titles
        empty_message: message for empty chart
        action_to_phase: if provided, adds phase legend and hover customdata
        canonical_order: if provided, orders labels in this canonical order first
    """
    if not data:
        return _empty_figure(380, empty_message)

    # Determine label ordering
    if canonical_order:
        labels = [p for p in canonical_order if p in data]
        labels += [p for p in sorted(data) if p not in canonical_order]
    else:
        labels = sorted(data.keys(), key=lambda k: data[k], reverse=True)

    values = [round(data[k], 1) if isinstance(data[k], float) else data[k] for k in labels]
    colors = [color_fn(k) for k in labels]

    # Format text labels
    if value_format:
        texts = [f"{v:{value_format}}" for v in values]
    else:
        texts = [str(v) for v in values]

    # Build bar trace kwargs
    bar_kwargs: dict = dict(
        marker_color=colors,
        text=texts,
        textposition="outside",
        cliponaxis=False,
        textfont=_LABEL_FONT,
        showlegend=False,
    )

    if orientation == "h":
        bar_kwargs.update(y=labels, x=values, orientation="h")
        if action_to_phase:
            phases = [action_to_phase.get(a, "") for a in labels]
            bar_kwargs["customdata"] = phases
            hover_labels = [{"template": "{action} ({phase})", "labels": {
                "action": label_spec("action", action), "phase": label_spec("phase", phase),
            }} for action, phase in zip(labels, phases, strict=True)]
        else:
            hover_labels = [label_spec(label_kind, label) for label in labels]
        bar_kwargs["hovertemplate"] = (
            "%{hovertext}: %{x" + (f":{value_format}" if value_format else "") + "}<extra></extra>"
        )
    else:
        bar_kwargs.update(x=labels, y=values)
        hover_labels = [label_spec(label_kind, label) for label in labels]
        hover_fmt = f":{value_format}" if value_format else ""
        bar_kwargs["hovertemplate"] = "%{hovertext}: %{y" + hover_fmt + "}<extra></extra>"

    fig = go.Figure(go.Bar(**bar_kwargs))
    trace_labels = {0: {"hovertext": hover_labels}}

    # Add phase legend for action charts
    if action_to_phase:
        phases_used = set(action_to_phase.get(a, "") for a in labels)
        trace_labels.update(_add_phase_legend(fig, phases_used))

    # Layout
    layout_kwargs: dict = dict(margin=dict(t=50, b=40, l=60, r=40))
    if orientation == "h":
        max_label = max((len(a) for a in labels), default=10)
        # When the action chart shows a phase legend above the plot, give the
        # margin enough room for both title and a 2-row wrapped legend without
        # leaving a dead band between them.
        top_margin = 75 if action_to_phase else 50
        layout_kwargs.update(
            height=max(380, 32 * len(labels)),
            margin=dict(l=max(180, max_label * 7 + 30), r=80, t=top_margin, b=40),
        )
        if action_to_phase:
            layout_kwargs.update(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
    else:
        layout_kwargs["height"] = 380

    _apply_chart_layout(fig, title, xaxis=xaxis, yaxis=yaxis, **layout_kwargs)
    axis = "yaxis" if orientation == "h" else "xaxis"
    fig.update_layout(**{axis: {"tickmode": "array", "tickvals": labels}})
    return localize_figure(fig, trace_labels=trace_labels, layout_labels={
        f"{axis}.ticktext": [label_spec(label_kind, label) for label in labels],
    })


def build_label_phase_count_chart(phase_counts: dict[str, int], dark: bool = False) -> go.Figure:
    """Bar chart of step counts per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_counts,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Step Count by Phase",
        xaxis="Phase",
        yaxis="Steps",
        empty_message="No phase data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_count_chart(
    action_counts: dict[str, int], action_to_phase: dict[str, str], dark: bool = False
) -> go.Figure:
    """Bar chart of step counts per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_counts,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Step Count by Action",
        orientation="h",
        label_kind="action",
        xaxis="Steps",
        empty_message="No action data",
        action_to_phase=action_to_phase,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_phase_duration_chart(phase_durations: dict[str, float], dark: bool = False) -> go.Figure:
    """Bar chart of summed duration per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_durations,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Duration by Phase",
        value_format=".1f",
        xaxis="Phase",
        yaxis="Duration (s)",
        empty_message="No phase duration data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_duration_chart(
    action_durations: dict[str, float], action_to_phase: dict[str, str], dark: bool = False
) -> go.Figure:
    """Bar chart of summed duration per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_durations,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Duration by Action",
        orientation="h",
        label_kind="action",
        value_format=".1f",
        xaxis="Duration (s)",
        empty_message="No action duration data",
        action_to_phase=action_to_phase,
    )
    _apply_dark(fig, dark)
    return fig


def _build_phase_comparison_chart(
    ref_phase_values: dict[str, float],
    cmp_phase_values: dict[str, float],
    ref_label: str,
    cmp_label: str,
    title: str,
    y_title: str,
    value_format: str = ",",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart comparing two trajectories' per-phase metrics.

    Phases are plotted in canonical order, including any extras from either side
    appended at the end. Bars are grouped side-by-side per phase.
    """
    phases = list(_CANONICAL_PHASES)
    for p in list(ref_phase_values.keys()) + list(cmp_phase_values.keys()):
        if p not in phases:
            phases.append(p)
    # Filter phases that appear in at least one side
    phases = [p for p in phases if ref_phase_values.get(p, 0) or cmp_phase_values.get(p, 0)]
    if not phases:
        fig = _empty_figure(380, "Upload labeled JSONs for both trajectories to see phase comparison")
        _apply_dark(fig, dark)
        return fig

    ref_vals = [ref_phase_values.get(p, 0) for p in phases]
    cmp_vals = [cmp_phase_values.get(p, 0) for p in phases]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=phases,
            y=ref_vals,
            name=ref_label,
            marker_color="#1d4ed8",
            text=[format(v, value_format) if v else "" for v in ref_vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=phases,
            y=cmp_vals,
            name=cmp_label,
            marker_color="#dc2626",
            text=[format(v, value_format) if v else "" for v in cmp_vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    # Pad y-axis headroom so outside labels on the tallest bars don't collide
    # with the legend sitting just above the plot area.
    max_val = max([*ref_vals, *cmp_vals, 0])
    _apply_chart_layout(
        fig,
        title,
        xaxis="Phase",
        yaxis=y_title,
        height=380,
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    if max_val > 0:
        fig.update_yaxes(range=[0, max_val * 1.12])
    _apply_dark(fig, dark)
    fig.update_xaxes(tickmode="array", tickvals=phases)
    return localize_figure(fig, layout_labels={
        "xaxis.ticktext": [label_spec("phase", phase) for phase in phases],
    }, trace_labels={index: {
        "hovertext": [label_spec("phase", phase) for phase in phases],
        "hovertemplate": "%{hovertext}: %{y:" + value_format + "}<extra>%{fullData.name}</extra>",
    } for index in range(len(fig.data))})


def build_phase_count_comparison_chart(
    ref_phase_counts: dict[str, int],
    cmp_phase_counts: dict[str, int],
    ref_label: str = "reference",
    cmp_label: str = "compared",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart: step count per phase for reference vs compared."""
    return _build_phase_comparison_chart(
        ref_phase_counts,
        cmp_phase_counts,
        ref_label,
        cmp_label,
        title="Step Count by Phase — Reference vs Compared",
        y_title="Steps",
        value_format=",",
        dark=dark,
    )


def build_phase_duration_comparison_chart(
    ref_phase_durations: dict[str, float],
    cmp_phase_durations: dict[str, float],
    ref_label: str = "reference",
    cmp_label: str = "compared",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart: total duration (s) per phase for reference vs compared."""
    return _build_phase_comparison_chart(
        ref_phase_durations,
        cmp_phase_durations,
        ref_label,
        cmp_label,
        title="Duration by Phase — Reference vs Compared",
        y_title="Duration (s)",
        value_format=".1f",
        dark=dark,
    )


def build_label_timeline_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar timeline — one segment per step, colored by phase, action on hover."""
    if not steps:
        fig = _empty_figure(380, "No step data")
        _apply_dark(fig, dark)
        return fig

    _USER_COLOR = "#d1d5db"  # grey fill for user prompt bar
    _USER_BORDER = "#9ca3af"  # darker border

    step_indices = [s.get("index", i) for i, s in enumerate(steps)]
    step_labels = [str(idx) for idx in step_indices]
    durations = [s.get("duration_s") or 0 for s in steps]
    roles = [s.get("role", "assistant") for s in steps]
    phases = [s.get("phase", "") for s in steps]
    actions = [s.get("action", "") for s in steps]
    max_dur = max(durations) if durations else 1
    y_pos = list(range(len(steps)))

    # --- Assistant bars (only non-user rows; user rows get 0) ---
    asst_durations = [d if r != "user" else 0 for d, r in zip(durations, roles, strict=False)]
    asst_colors = [LABEL_PHASE_COLORS.get(p, "#6b7280") for p in phases]
    asst_text = [{"template": "{action}  ({duration:.1f}s)", "params": {"duration": duration},
                  "labels": {"action": label_spec("action", action)}} if role != "user" else {"literal": ""}
                 for role, action, duration in zip(roles, actions, durations, strict=False)]
    asst_hover = [
        {"template": "<b>Step {index}</b><br>Phase: {phase}<br>Action: {action}<br>"
                     "Duration: {duration:.1f}s<br>Tokens: {tokens:,}",
         "params": {"index": step_indices[i], "duration": durations[i], "tokens": s.get("tokens_total", 0)},
         "labels": {"phase": label_spec("phase", phases[i]), "action": label_spec("action", actions[i])}}
        if roles[i] != "user" else {"literal": ""}
        for i, s in enumerate(steps)
    ]

    fig = go.Figure()
    trace_labels = {0: {"text": asst_text, "hovertext": asst_hover}}
    layout_labels = {}
    fig.add_trace(
        go.Bar(
            x=asst_durations,
            y=y_pos,
            orientation="h",
            marker_color=asst_colors,
            textposition="outside",
            outsidetextfont=dict(size=11),
            textangle=0,
            cliponaxis=False,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # --- User prompt bars (small fixed-width marker + label) ---
    user_bar_width = max_dur * 0.04
    user_indices = [i for i, r in enumerate(roles) if r == "user"]
    if user_indices:
        user_hover = [
            {"template": "<b>Step {index} — user prompt</b><br>{preview}",
             "params": {"index": step_indices[i], "preview": steps[i].get("text_preview", "")[:120]}}
            for i in user_indices
        ]
        trace_labels[len(fig.data)] = {"hovertext": user_hover}
        fig.add_trace(
            go.Bar(
                x=[user_bar_width] * len(user_indices),
                y=user_indices,
                orientation="h",
                width=0.8,  # match agent bar height
                marker_color=_USER_COLOR,
                marker_line=dict(color=_USER_BORDER, width=1),
                hoverinfo="text",
                showlegend=False,
            )
        )
        for yi in user_indices:
            layout_labels[f"annotations[{len(fig.layout.annotations)}].text"] = "<i>user prompt</i>"
            fig.add_annotation(
                x=user_bar_width,
                y=yi,
                text="<i>user prompt</i>",
                showarrow=False,
                xanchor="left",
                xshift=6,
                font=dict(size=10, color="#6b7280"),
            )

    # --- Legend entries (one per phase + user) ---
    seen: set[str] = set()
    for role, phase in zip(roles, phases, strict=False):
        key = "user" if role == "user" else phase
        if key and key not in seen:
            seen.add(key)
            trace_labels[len(fig.data)] = {"name": "user prompt" if key == "user" else label_spec("phase", key)}
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker_color=_USER_COLOR if key == "user" else LABEL_PHASE_COLORS.get(key, "#6b7280"),
                    name="user prompt" if key == "user" else key,
                    showlegend=True,
                )
            )

    chart_height = max(450, 28 * len(steps))
    top_margin = 90  # title (~25 px) + small gap + legend (~25 px) + breathing room
    # Position both title and legend in container (figure-relative) coords with
    # constant pixel offsets so the title sits on top with the legend just below
    # it, regardless of how tall the chart is. yref="container" keeps both in
    # the same coordinate system so they don't collide.
    title_offset_px = 8  # title top, ~8 px below figure top
    legend_offset_px = 45  # legend top, ~45 px below figure top (just under title)
    title_y = 1 - title_offset_px / chart_height
    legend_y = 1 - legend_offset_px / chart_height
    _apply_chart_layout(
        fig,
        "Step Timeline (colored by phase, labeled by action)",
        xaxis="Duration (s)",
        yaxis="Step",
        height=chart_height,
        margin=dict(l=70, r=200, t=top_margin, b=40),
        showlegend=True,
        barmode="overlay",
        legend=dict(orientation="h", yref="container", yanchor="top", y=legend_y, xanchor="center", x=0.5),
    )
    # Pin the title above the legend in container coords.
    fig.update_layout(
        title=dict(
            text="Step Timeline (colored by phase, labeled by action)",
            y=title_y,
            yref="container",
            yanchor="top",
            x=0.5,
            xanchor="center",
        )
    )
    fig.update_yaxes(
        tickvals=y_pos,
        ticktext=step_labels,
        # Keep half a row at each edge instead of allowing numeric-axis
        # autorange padding to grow with long sessions. Recorded step indices
        # are labels; they must not introduce gaps between displayed rows.
        range=[len(steps) - 0.5, -0.5],
        autorange=False,
    )
    _apply_dark(fig, dark)
    return localize_figure(fig, trace_labels=trace_labels, layout_labels=layout_labels)
