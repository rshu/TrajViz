"""Step and agent usage charts: tokens, duration, and tools."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from ...i18n import localize_figure

from ._layout import (
    _add_legend_hint,
    _apply_chart_layout,
    _apply_dark,
    _empty_figure,
    _fmt_duration,
    _labels_with_call_counts,
    _truncate_chart_label,
    _truncate_chart_labels,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import chart_data
from ..palette import (
    AGENT_COLORS,
    CHART_ACCENT,
    DURATION_ERROR_COLORS,
    SESSION_COLORS,
    TOKEN_COLORS,
)


def _subplot_rc(row: int | None, col: int | None) -> dict[str, int]:
    """Plotly subplot kwargs, or ``{}`` for a standalone figure."""
    if row is None or col is None:
        return {}
    return {"row": row, "col": col}


# Fixed in axis units so sparse multi-agent panels (few steps on a shared
# Workflow x-range) do not auto-widen bars to fill the gap between points.
_STEP_BAR_WIDTH = 0.8


def _add_token_bar_traces(
    fig: go.Figure,
    x_values: list,
    fresh_input: list,
    cache_read: list,
    output: list,
    reasoning: list,
    *,
    cache_write: list | None = None,
    include_empty: bool = False,
    x_label: str = "Step",
    row: int | None = None,
    col: int | None = None,
    legend_shown: dict[str, bool] | None = None,
    bar_width: float | None = _STEP_BAR_WIDTH,
) -> None:
    """Add token category bar traces to a figure.

    By default only adds traces that have non-zero data to avoid misleading legends
    (e.g., trajectories that only report total tokens, no breakdown).

    Pass ``include_empty=True`` to force every tracked-but-zero trace into the
    legend. Cache Write is the one exception: OpenAI-style exports (OpenCode,
    CodeArts) structurally report zero for it, so an all-zero Cache Write never
    enters the legend — it would only add a dead entry every viewer asks about.

    Pass ``cache_write`` to add a 5th stacked trace (shown only when non-zero,
    or when ``include_empty`` would otherwise force it — still subject to the
    Cache Write opt-out above).

    Stack order (bottom → top) mirrors context-window occupancy: cache read
    forms the base, then fresh input, then generated reasoning / output;
    cache write (when reported) goes on top.

    When *legend_shown* is set, each series is claimed once across panels so a
    category that first appears on a later agent still gets a legend entry.

    *bar_width* defaults to the fixed step-axis width used by stacked Token
    Usage panels. Pass ``None`` for grouped charts (Token Breakdown by Agent)
    so Plotly can size side-by-side bars; an explicit width there makes every
    series claim most of the category and overlap.
    """
    traces = [
        ("Cache Read", cache_read, TOKEN_COLORS["cache_read"]),
        ("Fresh Input", fresh_input, TOKEN_COLORS["fresh_input"]),
        ("Reasoning", reasoning, TOKEN_COLORS["reasoning"]),
        ("Output", output, TOKEN_COLORS["output"]),
    ]
    if cache_write is not None:
        traces.append(("Cache Write", cache_write, TOKEN_COLORS["cache_write"]))
    rc = _subplot_rc(row, col)
    step_prefix = f"{x_label} " if x_label else ""
    for name, values, color in traces:
        forced = include_empty and name != "Cache Write"
        if forced or any(v is not None and v > 0 for v in values):
            show = _claim_legend(legend_shown, name) if legend_shown is not None else True
            bar_kwargs: dict = dict(
                x=x_values,
                y=values,
                name=name,
                legendgroup=name,
                showlegend=show,
                marker_color=color,
                hovertemplate=f"{step_prefix}%{{x}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            )
            if bar_width is not None:
                bar_kwargs["width"] = bar_width
            fig.add_trace(go.Bar(**bar_kwargs), **rc)


def _describe_impossible(fields: Sequence[Sequence[object]]) -> str:
    if not fields:
        return ""
    reported = ", ".join(
        f"{field} = {value:,}" if isinstance(value, (int, float)) else f"{field} = {value!r}"
        for field, value in fields
    )
    return f"Breakdown not drawn: the export reports {reported}"


def _workflow_x_range(rows: list[dict]) -> list[float]:
    xs = [int(r["step"]) for r in rows]
    return [-0.5, max(xs) + 0.5] if xs else [-0.5, 0.5]


def _agent_panel_height(n_panels: int, base: int) -> int:
    return min(2400, max(base, 200 * n_panels + 100))


def _panel_legend_y(height: int, top_margin: int, bottom_margin: int = 40, clear_px: int = 28) -> float:
    """Legend y (paper coords) that clears the first subplot's title band.

    Subplot titles sit at paper y=1.0 (yanchor="bottom") and extend upward
    by their font height, so the flat ``y=1.02`` the single-panel charts use
    collides with them. Paper units scale with the plot-area height, so the
    offset is derived from pixels instead.
    """
    plot_h = max(height - top_margin - bottom_margin, 1)
    return 1 + round(clear_px / plot_h, 4)


def _make_agent_subplots(agents: list[dict]) -> go.Figure:
    """One shared-x row per timeline agent; room for per-panel Step axes."""
    n = len(agents)
    # Plotly requires vertical_spacing <= 1/(rows-1). 0.12 keeps room for the
    # upper panel's tick labels plus the lower panel's subplot title.
    return make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[a["label"] for a in agents],
        vertical_spacing=min(0.12, 0.95 / max(n - 1, 1)),
    )


def _apply_shared_step_axes(fig: go.Figure, *, n_panels: int, x_range: list[float], y_title: str) -> None:
    # shared_xaxes hides tick labels on upper panels by default. The "Step"
    # axis title goes on the bottom panel only — repeating it between panels
    # collides with the lower panel's subplot title.
    for row in range(1, n_panels + 1):
        fig.update_yaxes(title_text=y_title, row=row, col=1)
        fig.update_xaxes(
            title_text="Step" if row == n_panels else None,
            range=x_range,
            showticklabels=True,
            row=row,
            col=1,
        )


def _finalize_step_chart(
    fig: go.Figure,
    *,
    title: str,
    height: int,
    barmode: str,
    legend: dict,
    dark: bool,
    xaxis: str | None = None,
    yaxis: str | None = None,
    extra_layout: dict | None = None,
) -> go.Figure:
    _apply_chart_layout(
        fig,
        title,
        xaxis=xaxis,
        yaxis=yaxis,
        height=height,
        barmode=barmode,
        legend=legend,
    )
    if extra_layout:
        fig.update_layout(**extra_layout)
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    localize_figure(fig, trace_names=True)
    return fig


def _claim_legend(legend_shown: dict[str, bool], name: str) -> bool:
    """True the first time *name* is drawn; keeps multi-panel legends unique."""
    if legend_shown.get(name):
        return False
    legend_shown[name] = True
    return True


def _group_rows_by_agent(data: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    """Agents that have at least one row, plus rows grouped by agent id."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in data["steps"]:
        groups[row.get("agent", "")].append(row)
    agents = [a for a in data.get("agents", []) if groups.get(a["id"])]
    return agents, groups


def build_token_chart(steps: list[dict], dark: bool = False, *, format: str | None = None) -> go.Figure:
    """Stacked bar of token breakdown over steps (non-overlapping segments).

    Drawn from :func:`chart_data.token_usage`, so the figure and the exported
    data cannot disagree; see :func:`draw_token_usage` for the drawing rules.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig
    return draw_token_usage(chart_data.token_usage(steps, format), dark=dark)


def _draw_token_panel(
    fig: go.Figure,
    rows: list[dict],
    *,
    has_breakdown: bool,
    row: int | None = None,
    col: int | None = None,
    legend_shown: dict[str, bool] | None = None,
    include_empty: bool = False,
    x_label: str = "Step",
) -> int:
    """Draw one stacked (or total-only) token panel onto *fig*. Returns flagged count."""
    rc = _subplot_rc(row, col)
    legend_shown = legend_shown if legend_shown is not None else {}
    x_values = [r["step"] for r in rows]
    impossible = [r["invalid"] for r in rows]
    fresh_input = [r["fresh_input"] for r in rows]
    cache_r = [r["cache_read"] for r in rows]
    output_t = [r["output"] for r in rows]
    reasoning_t = [r["reasoning"] for r in rows]
    cache_w = [r["cache_write"] for r in rows]

    if has_breakdown:
        _add_token_bar_traces(
            fig,
            x_values,
            fresh_input,
            cache_r,
            output_t,
            reasoning_t,
            cache_write=cache_w,
            include_empty=include_empty,
            x_label=x_label,
            row=row,
            col=col,
            legend_shown=legend_shown,
        )
    else:
        totals = [0 if r["invalid"] else r["total"] for r in rows]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=totals,
                name="Total",
                legendgroup="Total",
                showlegend=_claim_legend(legend_shown, "Total"),
                marker_color=CHART_ACCENT,
                width=_STEP_BAR_WIDTH,
                hovertemplate=f"{x_label + ' ' if x_label else ''}%{{x}}<br>Total: %{{y:,.0f}}<extra></extra>",
            ),
            **rc,
        )

    flagged = sum(1 for bad in impossible if bad)
    if flagged:
        name = "Total only (invalid breakdown)"
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=[(r["usable_total"] or 0) if r["invalid"] else 0 for r in rows],
                name=name,
                legendgroup=name,
                showlegend=_claim_legend(legend_shown, name),
                marker=dict(
                    color=TOKEN_COLORS["invalid_breakdown"],
                    pattern=dict(shape="/", fgcolor="#4b5563"),
                ),
                width=_STEP_BAR_WIDTH,
                customdata=[_describe_impossible(bad) for bad in impossible],
                hovertemplate="Step %{x}<br>Reported total: %{y:,.0f}<br>%{customdata}<extra></extra>",
            ),
            **rc,
        )
    return flagged


def _token_title(flagged: int) -> str:
    title = "Token Usage by Step"
    if flagged:
        title += f" — {flagged} step{'s' if flagged != 1 else ''} with an invalid breakdown"
    return title


def draw_token_usage(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.token_usage` output.

    Multi-agent trajectories get one subplot panel per timeline agent, sharing
    Workflow step indices on the x-axis.
    """
    rows = data["steps"]
    if not rows:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    agents, groups = _group_rows_by_agent(data)
    fmt = data.get("format")
    has_breakdown = data["has_breakdown"]
    opencode_schema = fmt in ("opencode", "codearts", "codeagent")
    legend = dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5)
    x_range = _workflow_x_range(rows)
    legend_shown: dict[str, bool] = {}

    if len(agents) <= 1:
        panel_rows = groups[agents[0]["id"]] if agents else rows
        fig = go.Figure()
        flagged = _draw_token_panel(
            fig,
            panel_rows,
            has_breakdown=has_breakdown,
            legend_shown=legend_shown,
            include_empty=opencode_schema,
        )
        fig.update_xaxes(title_text="Step", range=x_range, showticklabels=True)
        return _finalize_step_chart(
            fig,
            title=_token_title(flagged),
            height=380,
            barmode="stack",
            legend=legend,
            dark=dark,
            xaxis="Step",
            yaxis="Tokens",
            extra_layout=dict(clickmode="event"),
        )

    n = len(agents)
    fig = _make_agent_subplots(agents)
    flagged_total = 0
    for row_i, agent in enumerate(agents, start=1):
        flagged_total += _draw_token_panel(
            fig,
            groups[agent["id"]],
            has_breakdown=has_breakdown,
            row=row_i,
            col=1,
            legend_shown=legend_shown,
            include_empty=opencode_schema and row_i == 1,
        )
    _apply_shared_step_axes(fig, n_panels=n, x_range=x_range, y_title="Tokens")
    height = _agent_panel_height(n, 380)
    return _finalize_step_chart(
        fig,
        title=_token_title(flagged_total),
        height=height,
        barmode="stack",
        legend={**legend, "y": _panel_legend_y(height, 110)},
        dark=dark,
        extra_layout=dict(clickmode="event", margin=dict(t=110)),
    )


def build_duration_chart(
    steps: list[dict],
    phases: list[dict] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Bar chart of step durations with average line.

    Drawn from :func:`chart_data.step_durations`; see :func:`draw_step_durations`.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig
    return draw_step_durations(chart_data.step_durations(steps), dark=dark)


def _draw_duration_panel(
    fig: go.Figure,
    rows: list[dict],
    *,
    row: int | None = None,
    col: int | None = None,
    legend_shown: dict[str, bool] | None = None,
) -> None:
    """Draw one step-duration panel (bars, spikes, avg) onto *fig*."""
    rc = _subplot_rc(row, col)
    legend_shown = legend_shown if legend_shown is not None else {}
    # Keep missing durations as None so Plotly draws gaps rather than fake zeros.
    durations = [r["duration_s"] for r in rows]
    spawn_adjusted = [r["spawn_adjusted"] for r in rows]
    kinds = [r["error_kind"] for r in rows]
    step_ids = [r["step"] for r in rows]
    outlier_steps = {step_ids[i] for i, r in enumerate(rows) if r["outlier"]}
    real = [r["duration_s"] for r in rows if r["duration_s"] is not None]
    avg_d = sum(real) / len(real) if real else None

    series = [
        ("Normal", None, "#3b82f6", ""),
        ("System Error", "system", DURATION_ERROR_COLORS["system"], " (system error)"),
        ("Tool Error", "tool", DURATION_ERROR_COLORS["tool"], " (tool error)"),
    ]
    for name, kind, color, suffix in series:
        positions = [i for i, k in enumerate(kinds) if k == kind]
        if name != "Normal" and not positions:
            continue
        xs = [step_ids[i] for i in positions]
        ys = [durations[i] for i in positions]
        custom = [
            [step_ids[i], " (excl. task wait)" if spawn_adjusted[i] else ""]
            for i in positions
        ]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                customdata=custom,
                name=name,
                legendgroup=name,
                showlegend=_claim_legend(legend_shown, name),
                marker_color=color,
                width=_STEP_BAR_WIDTH,
                hovertemplate=(
                    "Step %{customdata[0]}<br>%{y:.1f}s%{customdata[1]}"
                    f"{suffix}<extra></extra>"
                ),
            ),
            **rc,
        )
        spike_x = [x for x, y in zip(xs, ys, strict=True)
                   if x in outlier_steps and y is not None]
        spike_y = [y for x, y in zip(xs, ys, strict=True)
                   if x in outlier_steps and y is not None]
        if spike_x:
            fig.add_trace(
                go.Scatter(
                    x=spike_x,
                    y=spike_y,
                    mode="text",
                    text=[f"{v:.1f}s" for v in spike_y],
                    textposition="top center",
                    textfont=dict(size=9, color="#dc2626"),
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,
                ),
                **rc,
            )

    if avg_d is not None:
        fig.add_hline(
            y=avg_d,
            line_dash="dash",
            line_color="#94a3b8",
            line_width=1,
            annotation_text=f"Avg: {avg_d:.1f}s",
            annotation_position="top left",
            annotation_font=dict(color="#64748b", size=11),
            **rc,
        )


def draw_step_durations(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.step_durations` output.

    System errors (scaffold tools) and tool errors (agentic/custom) are
    separate legend series. Compactions are drawn on the context-pressure
    chart, which detects them; this chart has no second detector.

    Spawn/delegation wait (``task``, ``Agent``, …) is subtracted from each
    bar so parent steps blocked on a child agent do not create false spikes.

    Multi-agent trajectories get one panel per timeline agent, sharing the
    Workflow step index on the x-axis.
    """
    rows = data["steps"]
    if not rows:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    agents, groups = _group_rows_by_agent(data)
    x_range = _workflow_x_range(rows)
    legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="center",
        x=0.5,
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    )
    duration_layout = dict(margin=dict(t=70), clickmode="event")

    if len(agents) <= 1:
        panel_rows = groups[agents[0]["id"]] if agents else rows
        fig = go.Figure()
        _draw_duration_panel(fig, panel_rows)
        fig.update_xaxes(title_text="Step", range=x_range, showticklabels=True)
        fig.update_yaxes(title_text="Duration (s)")
        return _finalize_step_chart(
            fig,
            title="Step Duration",
            height=400,
            barmode="overlay",
            legend=legend,
            dark=dark,
            xaxis="Step",
            yaxis="Duration (s)",
            extra_layout=duration_layout,
        )

    n = len(agents)
    fig = _make_agent_subplots(agents)
    legend_shown: dict[str, bool] = {}
    for row_i, agent in enumerate(agents, start=1):
        _draw_duration_panel(
            fig,
            groups[agent["id"]],
            row=row_i,
            col=1,
            legend_shown=legend_shown,
        )
    _apply_shared_step_axes(fig, n_panels=n, x_range=x_range, y_title="Duration (s)")
    height = _agent_panel_height(n, 400)
    return _finalize_step_chart(
        fig,
        title="Step Duration",
        height=height,
        barmode="overlay",
        legend={**legend, "y": _panel_legend_y(height, 110)},
        dark=dark,
        extra_layout={**duration_layout, "margin": dict(t=110)},
    )


def build_tool_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of tool call frequency by name, stacked by agent.

    Drawn from :func:`chart_data.tool_usage`; see :func:`draw_tool_usage`.
    """
    return draw_tool_usage(chart_data.tool_usage(steps), dark=dark)


def draw_tool_usage(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.tool_usage` output: calls per tool, stacked by agent."""
    agents = data["agents"]
    has_agents = len(agents) > 1
    all_tools = {t["tool"]: t["calls"] for t in data["tools"]}

    if not all_tools:
        fig = _empty_figure(300, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(all_tools.keys(), key=lambda t: all_tools[t])
    # Totals live at the bar end (text / annotations), not on the y-axis.
    display_names = _truncate_chart_labels(sorted_tools)
    by_tool = {t["tool"]: t["by_agent"] for t in data["tools"]}

    fig = go.Figure()
    if has_agents:
        # Stacked bars per agent, matching swimlane / workflow labeling
        for position, agent in enumerate(agents):
            label = agent["label"]
            counts = [by_tool[t].get(agent["id"], 0) for t in sorted_tools]
            if not any(counts):
                continue
            color = SESSION_COLORS[position % len(SESSION_COLORS)]
            fig.add_trace(
                go.Bar(
                    y=display_names,
                    x=counts,
                    orientation="h",
                    name=label,
                    marker_color=color,
                    hovertemplate=f"{label}: " + "%{x} call(s)<extra></extra>",
                )
            )
        _annotate_hbar_ends(
            fig,
            [
                (y_label, all_tools[tool], f" {all_tools[tool]}")
                for tool, y_label in zip(sorted_tools, display_names, strict=True)
            ],
        )
    else:
        counts = [all_tools[t] for t in sorted_tools]
        fig.add_trace(
            go.Bar(
                y=display_names,
                x=counts,
                orientation="h",
                marker_color=CHART_ACCENT,
                text=[str(c) for c in counts],
                textposition="outside",
                cliponaxis=False,
                customdata=sorted_tools,
                hovertemplate="%{customdata}: %{x} call(s)<extra></extra>",
            )
        )

    max_label = max(len(n) for n in display_names)
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Frequency" + (" by Agent" if has_agents else ""),
        xaxis="Count",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="stack" if has_agents else "relative",
        margin_t=150 if has_agents else 50,
        margin_r=70 if has_agents else 60,
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0))
    _apply_dark(fig, dark)
    return fig


def build_tool_duration_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bars of tool duration, stacked by individual timed calls.

    Drawn from :func:`chart_data.tool_durations`; see :func:`draw_tool_durations`.
    """
    return draw_tool_durations(chart_data.tool_durations(steps), dark=dark)


def _tool_duration_axis_max(totals: list[float]) -> float:
    """X-axis upper bound; soft-caps when one tool would crush the others."""
    positive = sorted(value for value in totals if value > 0)
    if not positive:
        return 0.0
    if len(positive) < 2:
        return positive[-1]
    peak, runner = positive[-1], positive[-2]
    if peak <= runner * 3 or peak - runner < 30:
        return peak
    return max(runner * 2.0, 1.0)


def _duration_axis_ticks(real_max: float) -> tuple[list[float], list[str]]:
    """Evenly spaced ticks labeled with human durations."""
    if real_max <= 0:
        return [0.0], ["0s"]
    raw_step = real_max / 5.0
    mag = 10.0 ** math.floor(math.log10(raw_step))
    step = next((s * mag for s in (1, 2, 5, 10) if s * mag >= raw_step), mag)
    ticks = [0.0]
    value = step
    while value < real_max * 0.98:
        ticks.append(value)
        value += step
    if ticks[-1] < real_max:
        ticks.append(real_max)
    return ticks, [_fmt_duration(tick) for tick in ticks]


def _annotate_hbar_ends(fig: go.Figure, rows: list[tuple[str, float, str]]) -> None:
    """Left-anchored end labels for horizontal bars: ``(y_label, x, text)``."""
    for y_label, x, text in rows:
        fig.add_annotation(
            x=x,
            y=y_label,
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color="#64748b"),
            xref="x",
            yref="y",
        )


def draw_tool_durations(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.tool_durations` output.

    Each segment is one timed call (clickable via step ``customdata``). Spawn
    tools are omitted so child wall-clock is not double-counted. A dominant
    tool soft-caps the x-axis; end labels keep the true human duration.
    """
    agents = data["agents"]
    has_agents = len(agents) > 1
    color_index = {a["id"]: i for i, a in enumerate(agents)}
    labels = {a["id"]: a["label"] for a in agents}
    total_secs = {t["tool"]: t["total_s"] for t in data["tools"]}
    calls_by_tool = {t["tool"]: t["calls"] for t in data["tools"]}

    if not total_secs:
        fig = _empty_figure(300, "No tool-call timing recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(total_secs.keys(), key=lambda t: total_secs[t])
    display_names = _labels_with_call_counts(
        sorted_tools, [len(calls_by_tool[tool]) for tool in sorted_tools]
    )
    axis_max = _tool_duration_axis_max([total_secs[t] for t in sorted_tools])

    fig = go.Figure()
    legend_seen: set[str] = set()
    # One trace per timed call, deliberately: each segment needs its own base,
    # color and step index in customdata, which is what makes a bar clickable
    # back to its Workflow step. Trace count therefore equals timed-call count.
    # Real runs need at most a few hundred traces (about 0.1s to build), so
    # the cost only bites far outside any real trajectory (10,000 synthetic
    # calls: 1.4s, 3.1 MB). Batching into one array-valued Bar per
    # agent would have to keep the per-point step index resolvable.
    for tool_name, y_label in zip(sorted_tools, display_names, strict=True):
        base = 0.0
        for call in calls_by_tool[tool_name]:
            step_idx, secs, agent_id = call["step"], call["seconds"], call["agent"]
            if has_agents:
                color = SESSION_COLORS[color_index.get(agent_id, 0) % len(SESSION_COLORS)]
                agent_label = labels.get(agent_id, agent_id or "main")
                show_legend = agent_id not in legend_seen
                if show_legend:
                    legend_seen.add(agent_id)
                hover_extra = f"<br>{agent_label}"
            else:
                color = CHART_ACCENT
                agent_label = ""
                show_legend = False
                hover_extra = ""

            fig.add_trace(
                go.Bar(
                    y=[y_label],
                    x=[secs],
                    base=[base],
                    orientation="h",
                    name=agent_label,
                    legendgroup=agent_id if has_agents else None,
                    showlegend=show_legend,
                    marker=dict(
                        color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.9)"),
                    ),
                    customdata=[step_idx],
                    hovertemplate=(
                        f"{tool_name}<br>Step %{{customdata}}"
                        f"<br>{_fmt_duration(secs)}{hover_extra}<extra></extra>"
                    ),
                )
            )
            base += secs

    tickvals, ticktext = _duration_axis_ticks(axis_max)
    max_label = max(len(n) for n in display_names)
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Duration" + (" by Agent" if has_agents else ""),
        # Each bar stacks per-call durations, so calls that ran in parallel are
        # counted once per call here, while the "Tool time" chip is wall-clock
        # (the union of the call windows). Say which one this axis is.
        xaxis="Summed call time — parallel calls each count",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="overlay",
        margin_t=200 if has_agents else 50,
        margin_r=90,
    )
    fig.update_layout(clickmode="event")
    fig.update_yaxes(categoryorder="array", categoryarray=display_names)
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        range=[0, axis_max * 1.02] if axis_max > 0 else None,
    )
    _annotate_hbar_ends(
        fig,
        [
            (
                y_label,
                axis_max if total_secs[tool_name] > axis_max else total_secs[tool_name],
                f" {_fmt_duration(total_secs[tool_name])}"
                + (" →" if total_secs[tool_name] > axis_max else ""),
            )
            for tool_name, y_label in zip(sorted_tools, display_names, strict=True)
        ],
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1.0))
    _apply_dark(fig, dark)
    return fig


def _apply_tool_hbar_layout(
    fig: go.Figure,
    title: str,
    *,
    xaxis: str,
    n_tools: int,
    max_label: int,
    barmode: str,
    margin_t: int = 50,
    margin_r: int = 60,
) -> None:
    """Shared layout for horizontal tool frequency / duration bars."""
    _apply_chart_layout(
        fig,
        title,
        xaxis=xaxis,
        height=max(250, 50 * n_tools),
        barmode=barmode,
        margin=dict(l=max(140, max_label * 7 + 20), r=margin_r, t=margin_t, b=40),
    )


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba()`` string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _spread_sankey_y(n: int) -> list[float]:
    """Even y positions in (0, 1), inset so first/last nodes are not clipped."""
    if n <= 1:
        return [0.5]
    return [0.08 + 0.84 * i / (n - 1) for i in range(n)]


def build_skill_agent_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Sankey diagram of Skill-tool invocations: which agent called which skill.

    Drawn from :func:`chart_data.skill_calls`; see :func:`draw_skill_calls`.
    """
    return draw_skill_calls(chart_data.skill_calls(steps), dark=dark)


def draw_skill_calls(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.skill_calls` output as an agent → skill Sankey diagram."""
    links = data["links"]
    if not links:
        fig = _empty_figure(300, "No Skill-tool invocations recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig
    color_index = {a["id"]: i for i, a in enumerate(data["agents"])}
    labels = {a["id"]: a["label"] for a in data["agents"]}

    def _label(aid: str) -> str:
        return labels.get(aid, aid or "main")

    skill_totals: Counter[str] = Counter()
    agent_totals: Counter[str] = Counter()
    for link in links:
        skill_totals[link["skill"]] += link["calls"]
        agent_totals[link["agent"]] += link["calls"]

    agent_ids = [a["id"] for a in data["agents"] if agent_totals[a["id"]]]
    skills = [name for name, _ in skill_totals.most_common()]

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []

    agent_ys = _spread_sankey_y(len(agent_ids))
    skill_ys = _spread_sankey_y(len(skills))

    agent_index: dict[str, int] = {}
    for i, aid in enumerate(agent_ids):
        agent_index[aid] = len(node_labels)
        node_labels.append(f"{_label(aid)} ({agent_totals[aid]})")
        node_colors.append(SESSION_COLORS[color_index[aid] % len(SESSION_COLORS)])
        node_x.append(0.02)
        node_y.append(agent_ys[i])

    skill_index: dict[str, int] = {}
    skill_color = "#64748b"
    for i, skill in enumerate(skills):
        skill_index[skill] = len(node_labels)
        node_labels.append(f"{_truncate_chart_label(skill, 28)} ({skill_totals[skill]})")
        node_colors.append(skill_color)
        node_x.append(0.98)
        node_y.append(skill_ys[i])

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    hovers: list[str] = []
    for link in links:
        agent, skill, n = link["agent"], link["skill"], link["calls"]
        sources.append(agent_index[agent])
        targets.append(skill_index[skill])
        values.append(n)
        agent_color = SESSION_COLORS[color_index[agent] % len(SESSION_COLORS)]
        link_colors.append(_hex_rgba(agent_color, 0.38))
        noun = "call" if n == 1 else "calls"
        hovers.append(f"{_label(agent)} → {skill}<br>{n} {noun}")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=14,
                    thickness=16,
                    line=dict(width=0.5, color="rgba(148,163,184,0.45)"),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )
    n_nodes = max(len(agent_ids), len(skills))
    _apply_chart_layout(
        fig,
        "Skill Calls by Agent",
        height=max(300, 48 * n_nodes + 80),
        margin=dict(l=16, r=16, t=90, b=24),
    )
    _apply_dark(fig, dark)
    return fig


def build_agent_token_chart(agent_summaries: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart showing token breakdown per agent.

    Drawn from :func:`chart_data.agent_tokens`; see :func:`draw_agent_tokens`.
    """
    return draw_agent_tokens(chart_data.agent_tokens(agent_summaries), dark=dark)


def draw_agent_tokens(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.agent_tokens` output.

    Uses the same vertical grouped-bar layout for single- and multi-agent
    sessions so the single-agent case still surfaces the four token
    categories (Fresh / Cache Read / Output / Reasoning) — just with one
    cluster instead of many.
    """
    agents = data["agents"]
    if not agents:
        fig = _empty_figure(240, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    labels = [a["label"] for a in agents]
    has_breakdown = any(a["input"] > 0 or a["output"] > 0 for a in agents)

    fig = go.Figure()
    if has_breakdown:
        # Schema-tolerant fresh-input: handles both Claude Code (input includes
        # cache) and OpenCode (input is already cache-excluded).
        segs = [a["segments"] for a in agents]
        fresh_input = [g["fresh"] for g in segs]
        cache_read = [g["cache_read"] for g in segs]
        output = [g["output"] for g in segs]
        reasoning = [g["reasoning"] for g in segs]
        cache_write = [g["cache_write"] for g in segs]
        _add_token_bar_traces(
            fig,
            labels,
            fresh_input,
            cache_read,
            output,
            reasoning,
            cache_write=cache_write,
            x_label="",
            bar_width=None,
        )
    else:
        # No token breakdown available — show total tokens per agent
        session_palette = AGENT_COLORS  # shared palette so agent colors match across views
        totals = [a["total"] for a in agents]
        colors = [session_palette[i % len(session_palette)] for i in range(len(labels))]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=totals,
                name="Total Tokens",
                marker_color=colors,
                text=[f"{t:,}" for t in totals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.0f} tokens<extra></extra>",
            )
        )

    # Say so when a bar is a partial sum. The agent rows drop any step whose
    # token counts are unusable (OpenCode can report a negative input), so
    # without this the chart presents an incomplete total as a definitive one —
    # and `infer_non_cache_input` clamps at 0, which reads as a confident "no
    # fresh input" rather than "not all of it could be counted". The Overview
    # token chip already carries the same caveat.
    dropped = sum(a["input_unusable_steps"] for a in agents)
    title = "Token Breakdown by Agent" if has_breakdown else "Total Tokens by Agent"
    if dropped:
        title += f" — partial, excludes {dropped} step{'s' if dropped != 1 else ''}"

    _apply_chart_layout(
        fig,
        title,
        xaxis="Agent",
        yaxis="Tokens (count)",
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_layout(margin=dict(t=70))
    _apply_dark(fig, dark)
    localize_figure(fig, trace_names=True)
    return fig
