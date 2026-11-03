"""Patterns tab HTML: tool sequences, failure clusters, and anti-patterns."""

from __future__ import annotations

import html

from ..issues import count_tool_errors  # noqa: F401  (re-exported for importers of the old path)
from ..rendering import _step_link_chip, build_antipattern_summary_html
from ..session import LoadedSession


def build_antipattern_html(session: LoadedSession, *, workflow_links: bool = True) -> str:
    """Anti-pattern summary HTML for Overview and Patterns."""
    from ..telemetry_quality import supports

    if session.raw.get("_otel") and not all(
        supports(session.steps, field) for field in ("tool_inputs", "tool_outputs", "tool_outcomes")
    ):
        return "<div style='padding:1em;color:var(--ov-muted);'>Anti-pattern checks unavailable: incomplete tool evidence.</div>"
    error_count, error_steps = count_tool_errors(session.steps)
    return build_antipattern_summary_html(
        session.fruitless_streaks,
        session.tool_selection,
        session.plan_metrics,
        error_count=error_count,
        error_steps=error_steps,
        workflow_links=workflow_links,
    )


def render_tool_sequences_html(sequences: list[dict], *, workflow_links: bool = True) -> str:
    """Render tool sequence patterns as an HTML table."""
    if not sequences:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No recurring tool sequences detected (minimum frequency: 3).</div>"
    rows = []
    for s in sequences[:20]:
        seq_str = " → ".join(html.escape(t) for t in s["sequence"])
        indices = "".join(_step_link_chip(i, workflow_links=workflow_links) for i in s["step_indices"][:10])
        if len(s["step_indices"]) > 10:
            indices += f" … (+{len(s['step_indices']) - 10} more)"
        rows.append(
            f"<tr style='border-bottom:1px solid var(--ov-border);'>"
            f"<td style='padding:6px 10px;font-family:monospace;font-size:12px;'>{seq_str}</td>"
            f"<td style='padding:6px 10px;text-align:center;font-weight:700;'>{html.escape(str(s['frequency']))}</td>"
            f"<td style='padding:6px 10px;font-size:11px;color:var(--ov-muted);'>{indices}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<tr style='background:var(--ov-table-header-bg);'>"
        "<th style='text-align:left;padding:6px 10px;'>Sequence</th>"
        "<th style='text-align:center;padding:6px 10px;'>Count</th>"
        "<th style='text-align:left;padding:6px 10px;'>Step Indices</th></tr>" + "".join(rows) + "</table>"
    )


def render_failure_patterns_html(
    patterns: list[dict], *, available: bool = True, workflow_links: bool = True,
) -> str:
    """Render failure pattern clusters as cards with Workflow jump chips."""
    if not available:
        return "<div style='padding:1em;color:var(--ov-muted);'>Failure-pattern checks unavailable: tool outcomes are not fully recorded.</div>"
    if not patterns:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No errors detected — no failure patterns to analyze.</div>"
    cards = []
    for p in patterns:
        label = html.escape(p.get("cluster_label", "Unknown"))
        count = p.get("count", 0)
        example = html.escape(str(p.get("example_error", ""))[:200])
        recovery = p.get("recovery_path")
        recovery_html = " → ".join(html.escape(t) for t in recovery) if recovery else "<em>No recovery path found</em>"
        step_ids = p.get("steps") or []
        steps_html = (
            "".join(_step_link_chip(int(idx), workflow_links=workflow_links) for idx in step_ids)
            if step_ids
            else "<em>—</em>"
        )
        cards.append(
            f"<div class='overview-card' style='margin-bottom:8px;'>"
            f"<div style='font-weight:700;font-size:13px;color:var(--ov-text);'>{label} "
            f"<span style='font-size:11px;color:var(--ov-muted);font-weight:400;'>({count} occurrences)</span></div>"
            f"<div style='font-size:12px;color:var(--ov-muted);margin:4px 0;'>{example}</div>"
            f"<div style='font-size:12px;margin-bottom:4px;'><strong>Steps:</strong> {steps_html}</div>"
            f"<div style='font-size:12px;'><strong>Recovery path:</strong> {recovery_html}</div>"
            f"</div>"
        )
    return "".join(cards)
