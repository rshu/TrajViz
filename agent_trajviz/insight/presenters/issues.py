"""Overview Issues panel: render the ranked issues as foldable HTML cards.

The issues themselves are computed by :mod:`agent_trajviz.insight.issues`.
"""

from __future__ import annotations

import html

from ..issues import (  # noqa: F401  (re-exported: UI and judge import them here)
    Confidence,
    IssueJudgment,
    IssueKind,
    OverviewIssue,
    collect_overview_issues,
    rank_issues,
)
from ..rendering import _step_link_chips
from ..session import LoadedSession

ISSUE_KIND_COLORS: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "antipattern": "var(--ov-warn)",
    "bottleneck": "var(--ov-accent)",
}


def _issue_card(issue: OverviewIssue, *, workflow_links: bool = True) -> str:
    """Compact scan row: title + steps + visible Fix; Why behind a disclosure."""
    title = html.escape(issue.title)
    detail = html.escape(issue.detail)
    border = ISSUE_KIND_COLORS[issue.kind]
    steps_html = _step_link_chips(list(issue.steps), workflow_links=workflow_links)

    judgment_html = ""
    if issue.judgment is not None:
        j = issue.judgment
        also_html = ""
        if j.also:
            also_html = (
                f"<div style='font-size:11px;color:var(--ov-muted);margin-top:6px;"
                f"line-height:1.35;'>Also: {html.escape(j.also)}</div>"
            )
        conf = html.escape(j.confidence)
        judgment_html = (
            "<div style='margin-top:8px;font-size:12px;line-height:1.4;'>"
            "<span style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "text-transform:uppercase;color:var(--ov-muted);'>Change</span>"
            f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"font-size:12px;color:var(--ov-text);margin-top:2px;'>"
            f"{html.escape(j.where)}</div></div>"
            "<div style='margin-top:8px;padding:8px 10px;background:var(--ov-bg);"
            "border-radius:4px;'>"
            "<div style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "text-transform:uppercase;color:var(--ov-muted);margin-bottom:4px;'>"
            f"Fix <span style='font-weight:500;letter-spacing:0;text-transform:none;"
            f"color:var(--ov-muted);'>({conf})</span></div>"
            f"<div style='font-size:13px;line-height:1.4;color:var(--ov-text);'>"
            f"{html.escape(j.fix)}</div>"
            f"{also_html}</div>"
        )

    why_html = ""
    if issue.why:
        why_html = (
            "<details class='overview-issue-more'>"
            "<summary class='overview-issue-more-summary'>Why it matters</summary>"
            f"<div class='overview-issue-more-body'>"
            f"<div style='font-size:11px;color:var(--ov-muted);font-style:italic;"
            f"margin-top:4px;'>{html.escape(issue.why)}</div>"
            f"</div></details>"
        )

    detail_html = (
        f"<span class='overview-issue-detail'>{detail}</span>" if issue.detail else ""
    )
    return (
        f"<div class='overview-issue-card' style='border-left-color:{border};'>"
        f"<div class='overview-issue-head'>"
        f"<span class='overview-issue-title'>{title}</span>"
        f"{detail_html}"
        f"</div>"
        f"{steps_html}"
        f"{judgment_html}"
        f"{why_html}"
        f"</div>"
    )


def _issue_cards_html(issues: list[OverviewIssue], *, workflow_links: bool = True) -> str:
    """Render all issue cards (no preview cap)."""
    return "".join(_issue_card(issue, workflow_links=workflow_links) for issue in issues)


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    banner: str = "",
    progress: str = "",
    limited_evidence: bool = False,
    workflow_links: bool = True,
) -> str:
    """Render a foldable Issues panel with an evidence-aware empty state.

    *progress* is a live status line shown inside the panel while the LLM judge runs.
    """
    banner_html = ""
    if banner:
        banner_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{html.escape(banner)}</div>"
        )

    progress_html = ""
    if progress:
        # SMIL (not CSS) so prefers-reduced-motion / Gradio HTML swaps don't freeze it.
        progress_html = (
            "<div class='overview-issues-progress' role='status' aria-live='polite'>"
            "<span class='overview-issues-progress-label'>Thinking…</span>"
            "<svg class='overview-issues-progress-spinner' width='14' height='14' "
            "viewBox='0 0 24 24' aria-hidden='true' "
            "style='flex-shrink:0;display:block'>"
            "<circle cx='12' cy='12' r='10' fill='none' "
            "stroke='currentColor' stroke-opacity='0.25' stroke-width='3'/>"
            "<path d='M12 2a10 10 0 0 1 10 10' fill='none' stroke='currentColor' "
            "stroke-width='3' stroke-linecap='round'>"
            "<animateTransform attributeName='transform' type='rotate' "
            "from='0 12 12' to='360 12 12' dur='0.7s' repeatCount='indefinite'/>"
            "</path></svg>"
            f"<span>{html.escape(progress)}</span>"
            "</div>"
        )

    if not issues:
        status = "limited evidence" if limited_evidence else "none detected"
        detail = (
            "Some workflow checks are unavailable for this capture. "
            "The available evidence does not establish that the run is free of issues."
            if limited_evidence else "No major workflow issues detected."
        )
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span>Issues</span>"
            f"<span class='overview-issues-summary-meta'>{status}</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            f"{progress_html}"
            f"{banner_html}"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            f"{detail}"
            "</div></div></details>"
        )

    count = len(issues)
    judged = sum(1 for i in issues if i.judgment is not None)
    count_label = f"{count} issue{'s' if count != 1 else ''}"
    if limited_evidence:
        count_label += " · limited evidence"
    if progress:
        count_label += " · suggesting fixes…"
    elif judged:
        count_label += f" · {judged} with LLM fix"

    hint = (
        "LLM Change/Fix shown below — click a step to open Workflow"
        if judged and not progress
        else "Ranked problems — click a step to open Workflow; expand Why for context"
    )
    if not workflow_links:
        hint = (
            "LLM Change/Fix shown below — step numbers refer to the recorded trajectory"
            if judged and not progress else
            "Ranked problems — step numbers refer to the recorded trajectory; expand Why for context"
        )
    return (
        "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        "<span>Issues</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        f"{progress_html}"
        f"{banner_html}"
        f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>{hint}</div>"
        + _issue_cards_html(issues, workflow_links=workflow_links)
        + "</div></details>"
    )


def build_overview_issues_html(
    session: LoadedSession,
    *,
    issues: list[OverviewIssue] | None = None,
    banner: str = "",
    progress: str = "",
    workflow_links: bool = True,
) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    if issues is None:
        ranked = rank_issues(collect_overview_issues(session))
    else:
        ranked = list(issues)
    return render_overview_issues_html(ranked, banner=banner, progress=progress, workflow_links=workflow_links,
                                       limited_evidence=(session.format == "otlp" or bool(
                                           getattr(session, "metrics", {}).get("tool_unknown_outcomes"))))
