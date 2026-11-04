"""Plotly chart builders for Converge trajectory comparison visualization."""

from __future__ import annotations

# Pre-import pandas before plotly to avoid circular import error
# in plotly's basevalidators when running inside Gradio async threads.
try:
    import pandas  # noqa: F401
except ImportError:
    pass

import html
import textwrap

import plotly.graph_objects as go

from ..i18n import localize_figure
from .milestones import MILESTONE_LABELS, MILESTONE_NAMES


# -- Layout helpers -----------------------------------------------------------

_TPL = "plotly_white"

# Milestone names/labels live in milestones.py, next to extract_milestones, so
# the chart and the HTML table cannot drift apart. Aliased to the private
# names this module already used; the miss fallback stays `.get(name, name)`
# here and `.title()`-cased in rendering.py, so no rendered string changes.
_MILESTONE_NAMES = MILESTONE_NAMES
_MILESTONE_LABELS = MILESTONE_LABELS

_REF_COLOR = "#1d4ed8"   # blue
_CMP_COLOR = "#dc2626"   # red
_DELTA_COLOR = "#6b7280"  # gray

_PATTERN_COLORS = {
    "broad_exploration": "#3b82f6",
    "reverted_and_rewritten": "#ef4444",    # was write_retry
    "iterative_refinement": "#f87171",      # lighter red (less severe)
    "error_recovery_overhead": "#f59e0b",
    "premature_validation": "#8b5cf6",
    "redundant_search": "#06b6d4",
    "dead_end_branch": "#ec4899",
    "ordering_inefficiency": "#84cc16",
}


def _empty_figure(height: int = 380, message: str | None = None) -> go.Figure:
    """Return a blank Plotly figure, optionally with a centered message."""
    fig = go.Figure()
    if message:
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font_size=16,
        )
    fig.update_layout(template=_TPL, height=height)
    return localize_figure(fig, trace_names=True, annotations=True)


def _apply_chart_layout(
    fig: go.Figure, title: str,
    xaxis: str | None = None, yaxis: str | None = None,
    height: int = 380, **kwargs,
) -> None:
    """Apply standard chart layout."""
    layout = dict(
        title=dict(text="<br>".join(textwrap.wrap(title, width=26)),
                   x=0.02, xref="container", xanchor="left", y=0.94, yref="container", yanchor="top", pad=dict(t=2, b=6),
                   font=dict(size=15)),
        template=_TPL,
        height=height,
        autosize=True,
        margin=dict(t=110, b=60, l=60, r=20),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom", font=dict(size=11)),
    )
    if xaxis:
        layout["xaxis_title"] = xaxis
    if yaxis:
        layout["yaxis_title"] = yaxis
    layout.update(kwargs)
    fig.update_layout(**layout)


# -- Milestone timeline -------------------------------------------------------

def build_milestone_timeline_chart(
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
) -> go.Figure:
    """Compare recorded milestone steps in distinct, collision-free lanes.

    Each milestone has one named row. Reference and compared markers retain
    their exact step indices and sit slightly above/below that row; dotted
    connectors and signed deltas preserve the original comparison.
    """
    names = [name for name in _MILESTONE_NAMES
             if ref_milestones.get(name) is not None or cmp_milestones.get(name) is not None]
    if not names:
        return _empty_figure(message="No milestone data available")

    lanes = {name: len(names) - index - 1 for index, name in enumerate(names)}
    fig = go.Figure()
    for name in names:
        ref_val, cmp_val = ref_milestones.get(name), cmp_milestones.get(name)
        if ref_val is None or cmp_val is None:
            continue
        delta = cmp_val - ref_val
        fig.add_trace(go.Scatter(
            x=[ref_val, cmp_val], y=[lanes[name] + 0.14, lanes[name] - 0.14],
            mode="lines", line=dict(color=_DELTA_COLOR, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_annotation(
            x=(ref_val + cmp_val) / 2, y=lanes[name] + 0.42,
            text=f"+{delta}" if delta > 0 else str(delta),
            showarrow=False, font=dict(size=9, color=_DELTA_COLOR),
        )

    trace_labels = {}
    for milestones, other, label, color, offset in (
        (ref_milestones, cmp_milestones, "Reference", _REF_COLOR, 0.14),
        (cmp_milestones, ref_milestones, "Compared", _CMP_COLOR, -0.14),
    ):
        present = [name for name in names if milestones.get(name) is not None]
        if not present:
            continue
        comparisons = []
        comparison_labels = []
        for name in present:
            if other.get(name) is None:
                comparisons.append("Matching milestone not recorded in the other run")
                comparison_labels.append("Matching milestone not recorded in the other run")
            else:
                delta = cmp_milestones[name] - ref_milestones[name]
                comparisons.append(f"Compared − reference: {delta:+} steps")
                comparison_labels.append({"template": "Compared − reference: {delta} steps",
                                          "params": {"delta": f"{delta:+}"}})
        hovertemplate = "<b>%{text}</b><br>Step: %{x}<br>%{hovertext}<extra>" + label + "</extra>"
        labels = [_MILESTONE_LABELS.get(name, name) for name in present]
        trace_labels[len(fig.data)] = {
            "text": labels, "hovertext": comparison_labels, "hovertemplate": hovertemplate,
        }
        fig.add_trace(go.Scatter(
            x=[milestones[name] for name in present], y=[lanes[name] + offset for name in present],
            mode="markers", marker=dict(size=12, color=color, symbol="diamond"),
            text=labels, customdata=comparisons, hovertext=comparisons,
            name=label,
            hovertemplate=hovertemplate,
        ))

    _apply_chart_layout(
        fig, "Milestone Timeline", xaxis="Step Index", height=max(300, 160 + len(names) * 44),
        margin=dict(t=110, b=50, l=125, r=20),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom", font=dict(size=11)),
        yaxis_visible=True, yaxis_range=[-0.6, len(names) - 0.3],
        yaxis_tickvals=[lanes[name] for name in names],
        yaxis_ticktext=["<br>".join(textwrap.wrap(_MILESTONE_LABELS.get(name, name), width=18)) for name in names],
        yaxis_tickfont=dict(size=10), yaxis_showticklabels=True, yaxis_zeroline=False,
    )
    return localize_figure(fig, trace_names=True, annotations=True,
                           layout_labels={"yaxis.ticktext": list(fig.layout.yaxis.ticktext)},
                           trace_labels=trace_labels)


# -- Segment cost chart -------------------------------------------------------

def build_segment_cost_chart(
    segment_data: dict,
    milestone_order_matches: bool,
) -> go.Figure:
    """Grouped bar chart for segment costs.

    When milestone_order_matches is True, uses segment_comparison (paired bars).
    Otherwise uses reference_segments + compared_segments (separate bars).
    """
    fig = go.Figure()

    if milestone_order_matches:
        comparisons = segment_data.get("segment_comparison", [])
        if not comparisons:
            return _empty_figure(message="No segment comparison data")

        labels = [c["segment"] for c in comparisons]
        overheads = [c.get("overhead", 0) for c in comparisons]

        fig.add_trace(go.Bar(
            x=labels, y=overheads,
            name="Overhead Ratio",
            marker_color=_CMP_COLOR,
            text=[f"{v:.2f}x" for v in overheads],
            textposition="outside",
        ))
        _apply_chart_layout(
            fig, "Segment Cost Overhead (Compared / Reference)",
            xaxis="Segment", yaxis="Overhead Ratio",
        )
    else:
        ref_segs = segment_data.get("reference_segments", [])
        cmp_segs = segment_data.get("compared_segments", [])

        if not ref_segs and not cmp_segs:
            return _empty_figure(message="No segment data")

        if ref_segs:
            fig.add_trace(go.Bar(
                x=[s["segment"] for s in ref_segs],
                y=[s.get("tokens", 0) for s in ref_segs],
                name="Reference",
                marker_color=_REF_COLOR,
            ))
        if cmp_segs:
            fig.add_trace(go.Bar(
                x=[s["segment"] for s in cmp_segs],
                y=[s.get("tokens", 0) for s in cmp_segs],
                name="Compared",
                marker_color=_CMP_COLOR,
            ))
        _apply_chart_layout(
            fig, "Segment Token Costs",
            xaxis="Segment", yaxis="Tokens",
            barmode="group",
        )

    fig.update_layout(height=500, margin=dict(b=180),
                      xaxis=dict(title=None, tickangle=-45, automargin=True, tickfont=dict(size=10)))
    return localize_figure(fig, trace_names=True, annotations=True)


# -- Divergence waterfall chart ------------------------------------------------

def build_divergence_waterfall_chart(patterns: list[dict]) -> go.Figure:
    """Waterfall chart showing cost breakdown by divergence pattern category.

    Patterns are grouped by type and ordered by total tokens descending.
    """
    if not patterns:
        return _empty_figure(message="No divergence patterns detected")

    # Aggregate by pattern type
    by_type: dict[str, int] = {}
    for p in patterns:
        ptype = p.get("type", "unknown")
        tokens = p.get("estimated_extra_cost", {}).get("tokens", 0)
        by_type[ptype] = by_type.get(ptype, 0) + tokens

    # Sort by tokens descending
    sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)

    if not sorted_types or all(v == 0 for _, v in sorted_types):
        return _empty_figure(message="No measurable divergence cost")

    labels = [t for t, _ in sorted_types] + ["Total"]
    values = [v for _, v in sorted_types] + [sum(v for _, v in sorted_types)]
    measures = ["relative"] * len(sorted_types) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        text=[f"{v:,}" for v in values],
        textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color="#d1d5db", width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#6366f1")),
    ))

    # Outside labels need space beyond cumulative levels, including a final
    # total or negative adjustments. Keep every original token value unchanged.
    levels, cumulative = [0], 0
    for value in values[:-1]:
        cumulative += value
        levels.append(cumulative)
    low, high = min(levels), max(levels)
    padding = max((high - low) * 0.18, 1)
    _apply_chart_layout(
        fig, "Divergence Cost Breakdown by Pattern",
        xaxis="Pattern", yaxis="Extra Tokens",
        xaxis_tickmode="array", xaxis_tickvals=labels,
        yaxis_range=[low - padding if low < 0 else 0, high + padding],
    )
    return localize_figure(fig, trace_names=True, annotations=True, layout_labels={
        "xaxis.ticktext": [{"literal": html.escape(label)} for label in labels[:-1]] + ["Total"],
    })


# -- Anchor class recall chart ------------------------------------------------

def build_anchor_class_chart(anchor_analysis: dict | None) -> go.Figure:
    """Grouped bar chart showing per-class anchor-write recall for reference and compared.

    X-axis = file class, Y-axis = recall percentage.
    Reference bars are blue, compared bars are orange.
    """
    if not anchor_analysis:
        return _empty_figure(message="No anchor analysis available")

    # Per-class data is in reference/compared write_recall_by_class dicts
    ref_by_class = anchor_analysis.get("reference", {}).get("write_recall_by_class", {})
    cmp_by_class = anchor_analysis.get("compared", {}).get("write_recall_by_class", {})
    file_classes = anchor_analysis.get("file_classes", {})
    all_classes = sorted(set(ref_by_class.keys()) | set(cmp_by_class.keys()) | set(file_classes.keys()))

    if not all_classes:
        return _empty_figure(message="No per-class anchor data")

    classes = all_classes
    ref_recalls = [(ref_by_class.get(c) or 0) * 100 for c in classes]
    cmp_recalls = [(cmp_by_class.get(c) or 0) * 100 for c in classes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=classes,
        y=ref_recalls,
        name="Reference",
        marker_color=_REF_COLOR,
        text=[("" if v == 0 else f"{v:.1f}%" if v % 1 else f"{v:.0f}%") for v in ref_recalls],
        textposition=["inside" if v >= 10 else "outside" for v in ref_recalls], textangle=-90,
        textfont=dict(size=11), insidetextfont=dict(color="#ffffff"), outsidetextfont=dict(color="#334155"),
        constraintext="none", cliponaxis=False,
        hovertemplate="%{x}<br>Recall: %{y}%<extra>Reference</extra>",
    ))

    fig.add_trace(go.Bar(
        x=classes,
        y=cmp_recalls,
        name="Compared",
        marker_color="#f97316",  # orange
        text=[("" if v == 0 else f"{v:.1f}%" if v % 1 else f"{v:.0f}%") for v in cmp_recalls],
        textposition=["inside" if v >= 10 else "outside" for v in cmp_recalls], textangle=-90,
        textfont=dict(size=11), insidetextfont=dict(color="#111827"), outsidetextfont=dict(color="#334155"),
        constraintext="none", cliponaxis=False,
        hovertemplate="%{x}<br>Recall: %{y}%<extra>Compared</extra>",
    ))

    _apply_chart_layout(
        fig, "Anchor Write Recall by File Class",
        xaxis="File Class", yaxis="Recall (%)",
        barmode="group", yaxis_range=[0, max(100, *ref_recalls, *cmp_recalls) * 1.14],
        uniformtext=dict(minsize=11, mode="show"),
    )
    return localize_figure(fig, trace_names=True, annotations=True, trace_labels={
        index: {"hovertemplate": trace.hovertemplate} for index, trace in enumerate(fig.data)
    })
