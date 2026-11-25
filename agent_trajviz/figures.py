"""Plotly figures for views: the dashboard's own charts, drawn from view data.

Needs the ``charts`` extra (``pip install "agent-trajviz[charts]"``).

    >>> from agent_trajviz.figures import figure
    >>> fig = figure(run.view("token_usage"))      # a plotly.graph_objects.Figure
    >>> fig.to_json()                               # or the full Plotly JSON spec

Figures use the view's data, title and evidence-availability metadata, with
the same renderers as the dashboard.
:data:`FIGURE_VIEWS` lists the views that have one.
"""

from __future__ import annotations

from collections.abc import Callable, KeysView, Mapping
from typing import TYPE_CHECKING, Any

from ._extras import require

require("charts", "agent_trajviz.figures")

import plotly.graph_objects as go  # noqa: E402

from .converge import charts as _cvg  # noqa: E402
from .insight.charts import activity as _activity  # noqa: E402
from .insight.charts import label_charts as _labels  # noqa: E402
from .insight.charts import swimlanes as _swimlanes  # noqa: E402
from .insight.charts import usage as _usage  # noqa: E402
from .insight.charts._layout import _apply_chart_layout, _apply_dark, _empty_figure  # noqa: E402
from .insight.context_usage import usage_segments  # noqa: E402

if TYPE_CHECKING:
    from .views import View

__all__ = ["FIGURE_VIEWS", "figure", "figures", "register_figure"]


def _context_usage(data: dict, dark: bool) -> go.Figure:
    """One stacked bar: what fills the context window, by category."""
    rows = usage_segments(data)
    if not rows:
        fig = _empty_figure(160, "No context usage recorded.")
        _apply_dark(fig, dark)
        return fig
    window = data.get("window_limit")
    fig = go.Figure()
    for row in rows:
        fig.add_trace(go.Bar(
            x=[row["tokens"]], y=["context"], orientation="h", name=row["label"],
            marker_color=row["color"],
            hovertemplate=f"{row['label']}: %{{x:,}} tokens ({row['loaded_pct']}% of loaded)<extra></extra>",
        ))
    title = "Context composition"
    if isinstance(window, (int, float)) and window > 0:
        title += f" — {int(data.get('occupancy') or 0):,} of {int(window):,} tokens"
        fig.update_xaxes(range=[0, window])
    _apply_chart_layout(fig, title, xaxis="Tokens", height=200, barmode="stack",
                        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5))
    fig.update_yaxes(showticklabels=False)
    _apply_dark(fig, dark)
    return fig


def _plain(fig: go.Figure, dark: bool) -> go.Figure:
    # The comparison charts have no dark variant of their own.
    _apply_dark(fig, dark)
    return fig


_DRAW: dict[str, Callable[[Any, bool], go.Figure]] = {
    "token_usage": lambda d, dark: _usage.draw_token_usage(d, dark=dark),
    "step_durations": lambda d, dark: _usage.draw_step_durations(d, dark=dark),
    "tool_usage": lambda d, dark: _usage.draw_tool_usage(d, dark=dark),
    "tool_durations": lambda d, dark: _usage.draw_tool_durations(d, dark=dark),
    "skill_calls": lambda d, dark: _usage.draw_skill_calls(d, dark=dark),
    "agent_tokens": lambda d, dark: _usage.draw_agent_tokens(d, dark=dark),
    "agent_swimlane": lambda d, dark: _swimlanes.draw_agent_swimlane(d, dark=dark),
    "tool_outcomes": lambda d, dark: _swimlanes.draw_tool_outcomes(d, dark=dark),
    "file_interactions": lambda d, dark: _activity.draw_file_interactions(d, dark=dark),
    "plan": lambda d, dark: _activity.build_plan_timeline_chart(d["history"], d["metrics"], dark=dark),
    "context_pressure": lambda d, dark: _activity.build_context_pressure_chart(
        [], series=d["series"], dark=dark, highlight_step=d.get("highlight_step")),
    "context_usage": _context_usage,
    "label_phase_counts": lambda d, dark: _labels.build_label_phase_count_chart(d["counts"], dark=dark),
    "label_action_counts": lambda d, dark: _labels.build_label_action_count_chart(
        d["counts"], d["action_to_phase"], dark=dark),
    "label_phase_durations": lambda d, dark: _labels.build_label_phase_duration_chart(d["durations_s"], dark=dark),
    "label_action_durations": lambda d, dark: _labels.build_label_action_duration_chart(
        d["durations_s"], d["action_to_phase"], dark=dark),
    "label_timeline": lambda d, dark: _labels.build_label_timeline_chart(d["steps"], dark=dark),
    "label_phase_count_comparison": lambda d, dark: _labels.build_phase_count_comparison_chart(
        d["reference"]["counts"], d["compared"]["counts"],
        ref_label=d["reference"]["name"], cmp_label=d["compared"]["name"], dark=dark),
    "label_phase_duration_comparison": lambda d, dark: _labels.build_phase_duration_comparison_chart(
        d["reference"]["durations_s"], d["compared"]["durations_s"],
        ref_label=d["reference"]["name"], cmp_label=d["compared"]["name"], dark=dark),
    "run_timelines": lambda d, dark: _swimlanes.draw_run_timelines(d, dark=dark),
    "milestone_timeline": lambda d, dark: _plain(
        _cvg.build_milestone_timeline_chart(d["reference"] or {}, d["compared"] or {}), dark),
    "segment_costs": lambda d, dark: _plain(
        _cvg.build_segment_cost_chart(d or {}, bool((d or {}).get("milestone_order_matches"))), dark),
    "divergence": lambda d, dark: _plain(_cvg.build_divergence_waterfall_chart(d or []), dark),
    "anchor_classes": lambda d, dark: _plain(_cvg.build_anchor_class_chart(d), dark),
}

_BUILT_IN = frozenset(_DRAW)

#: Ids of the views :func:`figure` can draw: every built-in ``"chart"`` view,
#: and those given a figure with :func:`register_figure`.
FIGURE_VIEWS: KeysView[str] = _DRAW.keys()


def register_figure(view_id: str, draw: Callable[[Any, bool], go.Figure], *, replace: bool = False) -> None:
    """Draw a view of your own: ``draw(data, dark)`` returns the Plotly figure for the view's data.

    Typically for a view added with :func:`agent_trajviz.register_view`;
    :func:`figure` and the dashboards then draw it, regardless of its view
    kind. Built-in views without a renderer can also receive one. Raises
    :class:`ValueError` for an existing built-in renderer, or a registered
    custom renderer (pass ``replace=True`` to replace your own).
    """
    if view_id in _BUILT_IN:
        raise ValueError(f"{view_id!r} is a built-in view with its own figure")
    if view_id in _DRAW and not replace:
        raise ValueError(f"view {view_id!r} already has a figure; pass replace=True to replace it")
    if not callable(draw):
        raise TypeError("draw must be callable: draw(data, dark) -> plotly Figure")
    _DRAW[view_id] = draw


def figure(view: View, *, dark: bool = False) -> go.Figure:
    """The Plotly figure for a *view* with a registered renderer; ``dark`` selects the dark theme.

    Raises :class:`ValueError` when the view's id is not in :data:`FIGURE_VIEWS`.
    """
    try:
        draw = _DRAW[view.id]
    except KeyError:
        raise ValueError(f"view {view.id!r} is a {view.kind}, not a chart; see FIGURE_VIEWS") from None
    availability = view.availability or {}
    telemetry = availability.get("basis") == "captured_operations"
    status = availability.get("status")
    if telemetry and status == "unavailable":
        fig = _empty_figure(300, "Unavailable: required evidence was not recorded in this capture.")
        fig.update_layout(title_text=view.title)
        _apply_dark(fig, dark)
        return fig
    fig = draw(view.data, dark)
    if telemetry and status == "partial":
        # The observed points remain drawable; this annotation prevents a
        # partial sum or empty detector result from claiming full coverage.
        fig.add_annotation(
            text="Partial data: some required evidence is unavailable.", xref="paper", yref="paper",
            x=0, y=-0.24, xanchor="left", showarrow=False, font=dict(size=11, color="#64748b"),
        )
        margin = fig.layout.margin.b or 40
        fig.update_layout(margin_b=max(margin, 90))
    return fig


def figures(views: Mapping[str, View], *, dark: bool = False) -> dict[str, go.Figure]:
    """Render views whose ids are in :data:`FIGURE_VIEWS`, keyed by view id."""
    return {vid: figure(v, dark=dark) for vid, v in views.items() if vid in _DRAW}
