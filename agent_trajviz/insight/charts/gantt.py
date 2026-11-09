"""Wall-clock execution Gantt: one real-time lane per timeline agent."""

from __future__ import annotations

import html
import math

import plotly.graph_objects as go

from ..palette import CHART_ACCENT, TOKEN_COLORS
from ..timeline_agents import _legend_label, bind_timeline_agents
from ..workflow_role import workflow_role
from ._layout import _apply_chart_layout, _apply_dark, _fmt_duration

# Two-color scheme: the step body is the agent's own thinking time; tool
# execution windows are overlaid in a contrasting color on the same lane.
STEP_BAR_COLOR = CHART_ACCENT  # indigo — thinking / step body
TOOL_BAR_COLOR = TOKEN_COLORS["output"]  # amber — tool execution
USER_MARKER_COLOR = "#10b981"  # green — user prompt markers
WAIT_BAND_COLOR = "#e11d48"  # rose — idle / waiting-for-user band

_STEP_LEGEND = "Step (thinking)"
_TOOL_LEGEND = "Tool execution"
_USER_LEGEND = "User message"
_USER_LANE_LABEL = "user"
_WAIT_LEGEND = "User wait (idle)"
_WAIT_LANE_LABEL = "user wait"
_WAIT_BAND_OPACITY = 0.45

# Axis-mode radio values (Overview tab control).
GANTT_AXIS_SKIP_IDLE = "skip_idle"
GANTT_AXIS_WALLCLOCK = "wallclock"

# An idle gap only counts as "the session stalled" (rather than normal
# per-step latency) once it passes this threshold.
_MIN_IDLE_GAP_S = 10.0
_IDLE_GAP_SPAN_FRACTION = 0.05
# When skipping idle gaps, each long gap collapses to at most this share of
# the busy time (with a floor), so a weekend away cannot flatten the chart.
_COMPRESSED_GAP_MIN_S = 15.0
_COMPRESSED_GAP_FRACTION = 0.08


def _epoch_ms(value) -> float | None:
    """A usable millisecond epoch, or None for missing / zero / non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _step_end_ms(step: dict, start_ms: float) -> float | None:
    """Completion epoch of a step, falling back to ``duration`` seconds."""
    end_ms = _epoch_ms(step.get("time_completed_ms"))
    if end_ms is None:
        duration = step.get("duration")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            end_ms = start_ms + duration * 1000.0
    return end_ms


class _IdleGapAxis:
    """Piecewise-linear x mapping that collapses long idle gaps.

    Real seconds in, plot seconds out. Every gap longer than the threshold
    shrinks to a fixed visible band; everything else keeps true scale. Hover
    text keeps real values, so no information is lost — only the axis is
    compressed. With no qualifying gaps the mapping is the identity.
    """

    def __init__(self, spans: list[tuple[float, float]], compress: bool):
        merged: list[list[float]] = []
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        busy = sum(e - s for s, e in merged)
        if not merged:
            # Degenerate input (e.g. every span was dropped): identity mapping.
            self.gaps = []
            self._shifts = []
            return
        threshold = max(_MIN_IDLE_GAP_S, (merged[-1][1] - merged[0][0]) * _IDLE_GAP_SPAN_FRACTION)
        # (real_start, real_end, shown_width). Real gap width = end - start.
        self.gaps: list[tuple[float, float, float]] = []
        for i in range(len(merged) - 1):
            gap_start, gap_end = merged[i][1], merged[i + 1][0]
            gap = gap_end - gap_start
            if gap < threshold:
                continue
            shown = min(gap, max(_COMPRESSED_GAP_MIN_S, busy * _COMPRESSED_GAP_FRACTION)) if compress else gap
            self.gaps.append((gap_start, gap_end, shown))
        self._shifts = [
            (gap_start, gap_end, gap_end - gap_start - shown)
            for gap_start, gap_end, shown in self.gaps
        ]

    @property
    def compressed(self) -> bool:
        return any(
            (gap_end - gap_start) - shown > 0.5
            for gap_start, gap_end, shown in self.gaps
        )

    def map(self, t: float) -> float:
        out = t
        for gap_start, gap_end, shift in self._shifts:
            if t >= gap_end:
                out -= shift
            elif t > gap_start:
                # Inside the gap: scale to the shown band width.
                real = gap_end - gap_start
                shown = real - shift
                out = gap_start + shown * (t - gap_start) / real
                break
        return out


def _axis_ticks(axis: _IdleGapAxis, real_max: float) -> tuple[list[float], list[str]]:
    """Tick positions (plot coords) labeled with the REAL time there.

    On a compressed axis the default numeric ticks would lie — a plot
    position of "100" is not a real 100s on the far side of a compressed
    band. Instead each tick is labeled with the real time at its position:
    the uneven spacing (two adjacent ticks a whole idle apart) is the
    visible axis break, and the numbers never lie.
    """
    span = max(real_max, 1.0)
    raw_step = span / 8.0
    mag = 10.0 ** math.floor(math.log10(raw_step))
    step = next(s * mag for s in (1, 2, 5, 10) if s * mag >= raw_step)
    times: list[float] = [float(t) for t in range(0, int(real_max) + 1, int(step))]
    for gap_start, gap_end, _shown in axis.gaps:
        times += [gap_start, gap_end]

    def inside_gap(t: float) -> bool:
        return any(gs < t < ge for gs, ge, _ in axis.gaps)

    ticks: list[float] = []
    seen: set[float] = set()
    for t in sorted(times):
        if inside_gap(t):
            continue
        key = round(t, 3)
        if key in seen:
            continue
        seen.add(key)
        ticks.append(t)
    return [axis.map(t) for t in ticks], [_fmt_duration(t) for t in ticks]


def build_execution_gantt_chart(
    steps: list[dict],
    dark: bool = False,
    axis_mode: str = GANTT_AXIS_SKIP_IDLE,
) -> go.Figure | None:
    """Horizontal Gantt of step execution windows on a wall-clock x-axis.

    One lane per timeline agent (same identity as the agent swimlane); each
    step is a bar from ``time_created_ms`` to ``time_completed_ms`` in seconds
    relative to the earliest step. Timed tool calls inside a step are overlaid
    in a second color, so bars overlapping across lanes expose parallel
    sub-agent execution. ``axis_mode`` collapses long idle gaps (a weekend
    away would otherwise flatten every work bar into invisibility); hover
    text always carries real times. Returns None when no step carries a
    creation timestamp — the caller hides the chart.
    """
    valid = [s for s in steps if isinstance(s, dict)]
    timed = [s for s in valid if _epoch_ms(s.get("time_created_ms")) is not None]
    if not timed:
        return None

    t0 = min(_epoch_ms(s["time_created_ms"]) for s in timed)
    color_map, label_map, agent_id_of = bind_timeline_agents(valid)

    # Lanes in palette order (main first); user prompts occupy no lane.
    lane_ids: list[str] = []
    for s in timed:
        if workflow_role(s) == "user":
            continue
        aid = agent_id_of(s)
        if aid not in lane_ids:
            lane_ids.append(aid)
    if not lane_ids:
        return None
    lane_ids.sort(key=lambda aid: color_map.get(aid, 10**9))
    y_labels = [_legend_label(aid, label_map) for aid in lane_ids]

    # One record per drawable window. Idle-gap compression needs every span
    # before any trace is mapped onto the axis, and each series is one Bar
    # trace rather than one trace per step or tool call.
    spans: list[tuple[float, float]] = []
    user_marks: list[tuple[float, str]] = []
    step_bars: list[tuple[str, float, float, str]] = []
    tool_bars: list[tuple[str, float, float, str]] = []
    for s in timed:
        start_ms = _epoch_ms(s["time_created_ms"])
        if workflow_role(s) == "user":
            preview = str(s.get("text_preview") or "").strip().replace("\n", " ")
            user_marks.append(((start_ms - t0) / 1000.0, preview[:80]))
            continue
        end_ms = _step_end_ms(s, start_ms)
        if end_ms is None or end_ms <= start_ms:
            # No positive time extent: the step cannot be placed on a
            # wall-clock axis, but its lane still exists for the others.
            continue
        label = _legend_label(agent_id_of(s), label_map)
        start_s = (start_ms - t0) / 1000.0
        end_s = (end_ms - t0) / 1000.0
        spans.append((start_s, end_s))
        idx = int(s.get("index") if s.get("index") is not None else 0)
        tok = int((s.get("tokens") or {}).get("total") or 0)
        tool_names = [
            str(tc.get("tool_name") or "tool")
            for tc in (s.get("tool_calls") or [])
            if isinstance(tc, dict)
        ]
        tools_line = ", ".join(tool_names) if tool_names else "—"
        step_bars.append((
            label,
            start_s,
            end_s,
            (
                f"<b>Step {idx}</b> · {label}<br>"
                f"{start_s:.1f}s → {end_s:.1f}s ({end_s - start_s:.1f}s)<br>"
                f"tools: {tools_line}<br>"
                f"{tok:,} tokens"
            ),
        ))
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ts = _epoch_ms(tc.get("time_start"))
            te = _epoch_ms(tc.get("time_end"))
            if ts is None or te is None or te <= ts:
                continue
            tool_start_s = (ts - t0) / 1000.0
            tool_end_s = (te - t0) / 1000.0
            spans.append((tool_start_s, tool_end_s))
            tool_name = str(tc.get("tool_name") or "tool")
            # A task call blocks the parent while the sub-agent works: the
            # wall-clock overlap with the child lane is waiting, not parallel
            # computation. Say so on hover instead of letting it read as both
            # agents thinking at once.
            wait_note = (
                "<br>(parent blocked, sub-agent working)"
                if tool_name == "task"
                else ""
            )
            tool_bars.append((
                label,
                tool_start_s,
                tool_end_s,
                (
                    f"<b>Step {idx}</b> · {label}<br>"
                    f"{tool_name}: "
                    f"{tool_start_s:.1f}s → {tool_end_s:.1f}s "
                    f"({tool_end_s - tool_start_s:.1f}s)"
                    f"{wait_note}"
                ),
            ))
    axis = _IdleGapAxis(spans, compress=axis_mode != GANTT_AXIS_WALLCLOCK)
    real_max = max((e for _, e in spans), default=0.0)

    fig = go.Figure()

    def _add_bars(bars: list[tuple[str, float, float, str]], *, color: str, name: str) -> None:
        if not bars:
            return
        fig.add_trace(
            go.Bar(
                y=[label for label, _, _, _ in bars],
                x=[axis.map(end) - axis.map(start) for _, start, end, _ in bars],
                base=[axis.map(start) for _, start, _, _ in bars],
                orientation="h",
                marker_color=color,
                name=name,
                legendgroup=name,
                hovertext=[hover for _, _, _, hover in bars],
                hoverinfo="text",
            )
        )

    _add_bars(step_bars, color=STEP_BAR_COLOR, name=_STEP_LEGEND)
    _add_bars(tool_bars, color=TOOL_BAR_COLOR, name=_TOOL_LEGEND)

    # User prompts become markers on a dedicated top lane: they explain the
    # idle gaps between steps (the human was away, not the model stalling).
    if user_marks:
        fig.add_trace(
            go.Scatter(
                y=[_USER_LANE_LABEL] * len(user_marks),
                x=[axis.map(t) for t, _ in user_marks],
                mode="markers",
                marker=dict(symbol="triangle-right", size=10, color=USER_MARKER_COLOR),
                name=_USER_LEGEND,
                hovertext=[
                    f"<b>User</b> at +{t:.1f}s<br>{html.escape(p)}"
                    for t, p in user_marks
                ],
                hoverinfo="text",
            )
        )
        # Bottom → top: main, other agents, user on top (swimlane convention).
        y_labels = y_labels + [_USER_LANE_LABEL]

    # Idle bands live on their own top lane ("user wait") as rose bars — a
    # full-height gray wash behind every lane was easy to miss and read as
    # wallpaper. The label carries the REAL duration; in skip-idle mode the
    # band is narrow — the axis break is visible, the truth is on the label.
    if axis.gaps:
        y_labels = y_labels + [_WAIT_LANE_LABEL]
    user_times = [t for t, _ in user_marks]
    wait_legend_shown = False
    for gap_start, gap_end, shown in axis.gaps:
        gap = gap_end - gap_start
        waiting = any(gap_end - 5.0 <= t <= gap_end + 5.0 for t in user_times)
        label = f"idle {_fmt_duration(gap)}" + (" — waiting for user" if waiting else "")
        if shown < gap - 0.5:
            label += f"<br>(axis compressed, real {gap_start:.0f}s → {gap_end:.0f}s)"
        x0, x1 = axis.map(gap_start), axis.map(gap_end)
        fig.add_trace(
            go.Bar(
                y=[_WAIT_LANE_LABEL],
                x=[x1 - x0],
                base=[x0],
                orientation="h",
                marker=dict(color=WAIT_BAND_COLOR, opacity=_WAIT_BAND_OPACITY),
                name=_WAIT_LEGEND,
                legendgroup=_WAIT_LEGEND,
                showlegend=not wait_legend_shown,
                hovertext=label,
                hoverinfo="text",
            )
        )
        wait_legend_shown = True
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=_WAIT_LANE_LABEL,
            text=label,
            showarrow=False,
            font_size=10,
            font_color="#fecdd3" if dark else "#9f1239",
            yanchor="middle",
        )

    x_title = "Time (s) — relative to the first message"
    if axis.compressed:
        x_title += " · idle gaps compressed"
    height = min(2400, max(260, 90 * len(y_labels) + 110))
    _apply_chart_layout(
        fig,
        "Execution Timeline (wall-clock)",
        xaxis=x_title,
        height=height,
        barmode="overlay",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=y_labels, automargin=True),
    )
    if axis.compressed:
        tickvals, ticktext = _axis_ticks(axis, real_max)
        fig.update_layout(xaxis=dict(tickvals=tickvals, ticktext=ticktext))
    _apply_dark(fig, dark)
    return fig
