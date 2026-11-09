"""Overview tab HTML and Plotly: summary banner, KPIs, charts, and diagnostics."""

from __future__ import annotations

import html
import os
from collections import Counter
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ...i18n import text as ui_text
from ..formats.common import as_text
from ..context_usage import PRESSURE_ALL_AGENTS, _explicit_compaction_events
from ..formatting import (
    format_banner_html,
    format_behavioral_md,
    format_context_pressure_html,
    format_performance_md,
    _build_hotspots_md,
    _build_per_message_md,
)
from ..help import HELP_TEXT, SECTION_GUIDES
from ..loaders import tokens_are_estimates
from ...plugins import format_label
from ..metrics import extract_agent_info, format_duration_seconds
from ..palette import AGENT_COLORS
from ..rendering import (
    build_root_cause_html,
    render_agent_summary_cards,
)
from ..charts import build_execution_gantt_chart
from .issues import (
    ISSUE_KIND_COLORS,
    OverviewIssue,
    _fmt_k,
    build_overview_issues_html,
    overview_issues_payload,
    prepare_overview_issues,
)

from ..session import MAX_STEPS, LoadedSession

if TYPE_CHECKING:
    from ...api import Trajectory


def _rate_text(rate: float | None) -> str:
    """A percentage for a KPI card, or "n/a" when there is nothing to rate."""
    return f"{rate}%" if isinstance(rate, (int, float)) else "n/a"


def trajectory_format_label(fmt: str | None) -> str:
    """Return a human-readable trajectory format label (registered formats included)."""
    return format_label(fmt) if fmt else "Unknown"


def render_section_guide(section: str) -> str:
    """Header for an Overview Contents section: summary line plus a hover/focus
    panel describing each chart and a debugging example."""
    guide = SECTION_GUIDES[section]
    charts = "".join(
        "<li class='section-guide-chart'>"
        f"<span class='section-guide-chart-name'>{html.escape(name)}</span>"
        f"：{html.escape(desc)}</li>"
        for name, desc in guide.charts
    )
    return (
        "<div class='section-guide'>"
        f"<span class='section-guide-summary'>{html.escape(guide.summary)}</span>"
        "<div class='section-guide-tip' tabindex='0'>"
        "<span class='section-guide-trigger'>ⓘ 图表说明与调试示例</span>"
        "<div class='section-guide-pop' role='tooltip'>"
        "<div class='section-guide-panel'>"
        "<div class='section-guide-heading'>本页图表</div>"
        f"<ul class='section-guide-charts'>{charts}</ul>"
        "<div class='section-guide-heading'>调试示例</div>"
        f"<p class='section-guide-example'>{html.escape(guide.example)}</p>"
        "</div></div></div></div>"
    )


def _build_sparkline_svg(values: list[float], width: int = 100, height: int = 20) -> str:
    """Generate a minimal inline SVG sparkline from a list of values."""
    if not values or len(values) < 2:
        return ""
    max_v = max(values) or 1
    min_v = min(values)
    range_v = max_v - min_v or 1
    points = []
    for i, v in enumerate(values):
        x = round(i / (len(values) - 1) * width, 1)
        y = round(height - (v - min_v) / range_v * (height - 2) - 1, 1)
        points.append(f"{x},{y}")
    polyline = " ".join(points)
    return (
        f"<div class='ov-kpi-sparkline'>"
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<polyline points='{polyline}' fill='none' stroke='#3b82f6' stroke-width='1.5' "
        f"stroke-linecap='round' stroke-linejoin='round'/>"
        f"</svg></div>"
    )


def _build_session_detail_html(
    timing: dict,
    metadata: dict,
    *,
    model_id: str = "",
    agent_id: str = "",
) -> str:
    """Build Session Details panel as a chip grid of session environment fields."""
    # Converters copy these from the export as found; a non-dict block or a
    # non-string value must degrade to "N/A", not break the Overview.
    md = metadata if isinstance(metadata, dict) else {}
    timing = timing if isinstance(timing, dict) else {}
    started = timing.get("started_at", "N/A")
    finished = timing.get("finished_at", "N/A")
    if isinstance(started, str) and len(started) > 19:
        started = started[:19].replace("T", " ")
    if isinstance(finished, str) and len(finished) > 19:
        finished = finished[:19].replace("T", " ")

    fields = [
        ("Model", as_text(model_id) or as_text(md.get("model")) or "N/A"),
        ("Agent", as_text(agent_id) or as_text(md.get("agent")) or "N/A"),
        ("Start", str(started)),
        ("End", str(finished)),
        ("Session", (as_text(md.get("session_id")) or "N/A")[:16]),
        ("Branch", as_text(md.get("branch")) or "N/A"),
        ("Directory", as_text(md.get("directory_name")) or "N/A"),
        ("Platform", (as_text(md.get("platform")) or "N/A")[:24]),
    ]
    if as_text(md.get("server_version")):
        fields.insert(2, ("Version", as_text(md["server_version"])))

    chips = "".join(
        f"<div style='display:inline-flex;flex-direction:column;"
        f"background:var(--ov-card);border:1px solid var(--ov-border);"
        f"border-radius:8px;padding:6px 10px;min-width:100px;'>"
        f"<span style='font-size:10px;color:var(--ov-muted);text-transform:uppercase;'>"
        f"{ui_text(label)}</span>"
        f"<span style='font-size:13px;font-weight:500;'>"
        f"{html.escape(str(val))}</span></div>"
        for label, val in fields
    )
    return (
        "<details class='session-detail' style='margin:0 0 10px;'>"
        "<summary style='cursor:pointer;font-size:12px;color:var(--ov-muted);"
        f"user-select:none;'>{ui_text('Session details')}</summary>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;'>{chips}</div>"
        "</details>"
    )


_ISSUE_KIND_LABELS = (
    ("error", "error"),
    ("risk", "risk"),
    ("antipattern", "pattern"),
    ("bottleneck", "bottleneck"),
)


def _kpi_breakdown_html(
    items: list[tuple[str, int, str]],
    *,
    title: str,
    localize_names: bool = False,
) -> str:
    """Color-coded swatch list used by Steps (agents) and Issues (kinds)."""
    if not items:
        return ""
    rows: list[str] = []
    for name, count, color in items:
        safe = html.escape(name)
        rows.append(
            "<div class='ov-kpi-breakdown-row'>"
            f"<span class='ov-kpi-breakdown-swatch' style='background:{color};'></span>"
            f"<span class='ov-kpi-breakdown-name' style='color:{color};' "
            f"title='{safe}'>{ui_text(name) if localize_names else safe}</span>"
            f"<span class='ov-kpi-breakdown-count'>{count}</span>"
            "</div>"
        )
    return (
        f"<div class='ov-kpi-breakdown' title='{html.escape(title)}'>"
        + "".join(rows)
        + "</div>"
    )


def _issues_kpi_parts(issues: list[OverviewIssue]) -> tuple[str, str, str, str, str]:
    """Return (value, sub, status, detail, breakdown_html) for the Issues KPI card."""
    issue_count = len(issues)
    value = f"{issue_count:,}"
    if issue_count <= 0:
        return (
            value,
            "none detected",
            "good",
            "No major workflow issues detected",
            "",
        )
    kinds = Counter(issue.kind for issue in issues)
    items = [
        (
            label if kinds[key] == 1 else f"{label}s",
            kinds[key],
            ISSUE_KIND_COLORS[key],
        )
        for key, label in _ISSUE_KIND_LABELS
        if kinds[key]
    ]
    if issue_count >= 3:
        status, detail = "bad", f"{issue_count} issues — review Overview Issues"
    else:
        status, detail = "warn", f"{issue_count} issue{'s' if issue_count != 1 else ''} — review Overview Issues"
    return value, "", status, detail, _kpi_breakdown_html(items, title="Issues by kind", localize_names=True)


def _agent_steps_breakdown(agent_summaries: list[dict] | None) -> str:
    """Color-coded agent step list for the Steps KPI card."""
    summaries = agent_summaries or []
    if not summaries:
        return ""
    items = [
        (
            str(agent.get("label") or agent.get("agent_id") or "agent").strip() or "agent",
            int(agent.get("step_count") or 0),
            AGENT_COLORS[idx % len(AGENT_COLORS)],
        )
        for idx, agent in enumerate(summaries)
    ]
    return _kpi_breakdown_html(items, title="Assistant steps by agent")


def build_overview_kpi_html(
    metrics: dict,
    wall_fmt: str,
    verdicts: list[dict] | None = None,
    message_rows: list[dict] | None = None,
    *,
    issues: list[OverviewIssue] | None = None,
    agent_summaries: list[dict] | None = None,
    tokens_estimated: bool = False,
) -> str:
    """Build at-a-glance KPI card strip (global strip above the main Tabs).

    When *verdicts* is provided, matching KPI cards get a colored left border
    and a tooltip with the verdict detail string. *issues* drives the Issues
    card after Tokens. Steps shows an agent/subagent breakdown instead of the
    Errors health verdict (that signal lives on Tool Success / Issues).
    """
    _verdict_map: dict[str, tuple[str, str]] = {}
    if verdicts:
        _metric_to_kpi = {
            "Tool Success": "Tool Success",
            "Throughput": "Tokens",
            "Token Efficiency": "Tokens",
        }
        for v in verdicts:
            kpi_label = _metric_to_kpi.get(v["metric"], "")
            if kpi_label:
                _verdict_map[kpi_label] = (v["status"], v["detail"])

    issue_value, issue_sub, issue_status, issue_detail, issue_breakdown = _issues_kpi_parts(
        issues or [],
    )
    if metrics.get("data_quality", {}).get("source") == "otlp" or metrics.get("tool_unknown_outcomes"):
        if not issues:
            issue_value, issue_sub, issue_status = "N/A", "Limited captured evidence", "unavailable"
            issue_detail = "Unavailable checks cannot establish that the run is free of issues."
        else:
            issue_sub = "observed in this capture"
    # Border/tooltip only — kind breakdown is rendered like the Steps agent list.
    _verdict_map["Issues"] = (issue_status, issue_detail)

    _status_colors = {
        "good": "#059669",
        "warn": "#d97706",
        "bad": "#dc2626",
    }

    sparkline_data: dict[str, list[float]] = {}
    if message_rows:
        sparkline_data["Tokens"] = [r["tokens_total"] for r in message_rows if r.get("tokens_total") is not None]
        sparkline_data["Wall-Clock"] = [r["duration"] for r in message_rows if r.get("duration") is not None]

    _label_to_help_key = {
        "Steps": "steps",
        "Wall-Clock": "wall_clock",
        "Tokens": "tokens",
        "Issues": "issues",
        "Tool Success": "tool_success",
    }

    output_rate = metrics.get("output_tokens_per_sec")
    if isinstance(output_rate, (int, float)) and not isinstance(output_rate, bool):
        throughput_sub = ui_text("{rate} gen tok/s", rate=f"{output_rate:,}")
        if (metrics.get("output_throughput_tool_wait_seconds") or 0) > 0:
            throughput_sub += " " + ui_text("excl. tools")
    else:
        throughput_sub = ui_text("Generation throughput: N/A")
    timed_steps = metrics.get("output_throughput_timed_steps")
    throughput_steps = metrics.get("output_throughput_total_steps")
    if (
        isinstance(timed_steps, int)
        and isinstance(throughput_steps, int)
        and throughput_steps > 0
    ):
        throughput_sub += " · " + ui_text("{timed}/{total} timed", timed=timed_steps, total=throughput_steps)

    p95 = metrics.get("p95_duration")
    p95_sub = (
        f"P95 {format_duration_seconds(p95)}"
        if isinstance(p95, (int, float)) and not isinstance(p95, bool)
        else "P95 N/A"
    )
    user_steps = metrics.get("user_steps", 0)
    agent_breakdown = _agent_steps_breakdown(agent_summaries)
    cards = [
        (
            "Steps",
            f"{metrics.get('total_steps', 0):,}",
            ui_text("{assistant} assistant, {user} user", assistant=metrics.get('assistant_steps', 0), user=user_steps),
        ),
        ("Wall-Clock", wall_fmt, p95_sub),
        (
            "Tokens",
            ("\u2248" if tokens_estimated else "")
            + ("N/A" if metrics.get('tokens', {}).get('total') is None
               else f"{metrics.get('tokens', {}).get('total', 0):,}"),
            (ui_text("estimated") + " · " if tokens_estimated else "") + throughput_sub,
        ),
        ("Issues", issue_value, ui_text(issue_sub) if issue_sub else ""),
        ("Tool Success", _rate_text(metrics.get("tool_success_rate")), ui_text("{count} calls", count=f"{metrics.get('tool_call_count', 0):,}")),
    ]
    card_html = []
    for label, value, sub in cards:
        verdict_info = _verdict_map.get(label)
        extra_style = ""
        title_attr = ""
        data_attr = ""
        jump_attr = ""
        extra_class = ""
        if verdict_info:
            status, detail = verdict_info
            border_color = _status_colors.get(status, "#6b7280")
            extra_style = f" style='border-left:4px solid {border_color};'"
            title_attr = f" title='{html.escape(detail)}'"
            data_attr = f" data-status='{html.escape(status)}'"
        if label == "Issues":
            extra_class = " ov-kpi-card--issues"
            jump_attr = (
                " role='link' tabindex='0'"
                " onclick=\"if(window.tvClickMainTab){window.tvClickMainTab('Overview');}"
                "var el=document.getElementById('overview-issues');"
                "if(el){el.scrollIntoView({behavior:'smooth',block:'start'});"
                "if(!el.open){el.open=true;}}\""
            )
        help_key = _label_to_help_key.get(label, "")
        help_attr = ""
        if help_key and help_key in HELP_TEXT:
            from ...i18n import attrs
            help_attr = " " + attrs(HELP_TEXT[help_key], attribute="data-help")
        sparkline = ""
        if label in sparkline_data:
            sparkline = _build_sparkline_svg(sparkline_data[label])
        sub_block = (
            f"<div class='ov-kpi-sub'>{sub}</div>" if sub else ""
        )
        extra_line = ""
        if label == "Steps" and agent_breakdown:
            extra_line = agent_breakdown
        elif label == "Issues" and issue_breakdown:
            extra_line = issue_breakdown
        elif label == "Tokens":
            # The rate and timing coverage stay visible. Keep the denominator
            # caveats accessible without making every KPI as tall as this one.
            detail = verdict_info[1] if verdict_info else ""
            methodology = (
                "Output tokens and generation time come from the same timed assistant steps. "
                "Timed tool and subagent waits are excluded from generation time; waits without "
                "timing remain in the denominator. Untimed output is excluded from both tokens "
                "and time, so missing timing cannot inflate the reported rate."
            )
            detail_block = f"<p>{html.escape(detail)}</p>" if detail else ""
            extra_line = (
                "<details class='ov-kpi-methodology' style='font-size:11px;margin-top:4px;'>"
                f"<summary style='cursor:pointer;color:var(--ov-muted);'>{ui_text('Throughput details')}</summary>"
                f"{detail_block}<p>{ui_text(methodology)}</p></details>"
            )
        elif verdict_info and label != "Issues":
            status, detail = verdict_info
            vcolor = _status_colors.get(status, "#6b7280")
            extra_line = (
                f"<div style='font-size:11px;color:{vcolor};margin-top:2px;'>"
                f"{html.escape(detail)}</div>"
            )
        card_html.append(
            f"<div class='ov-kpi-card{extra_class}'{extra_style}{title_attr}{data_attr}{jump_attr}>"
            f"<div class='ov-kpi-label'{help_attr}>{ui_text(str(label))}</div>"
            f"<div class='ov-kpi-value'>{html.escape(str(value))}</div>"
            f"{sub_block}"
            f"{extra_line}"
            f"{sparkline}"
            "</div>"
        )
    return "<div class='ov-kpi-grid'>" + "".join(card_html) + "</div>"


def empty_plotly_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def build_summary_outputs(session: LoadedSession) -> dict:
    """Legacy helper: stats one-liner (kept for tests; not shown in the live UI)."""
    return {
        "banner": format_banner_html(
            os.path.basename(session.path),
            session.metrics,
            session.wall_clock,
        ),
    }


def build_token_summary_html(session: LoadedSession) -> str:
    """One-line takeaway above the Token Usage chart: cache hit rate, tokens
    dropped by compaction, the biggest step, and the peak context — the
    numbers that used to need a bar click to see.

    Every value is generated (counts, percentages); no export text enters, so
    the line needs no escaping.
    """
    steps = session.steps
    if not steps:
        return ""
    cache_read = fresh = peak = 0
    max_total = 0
    max_idx = None
    for i, s in enumerate(steps):
        tokens = s.get("tokens") or {}
        cr = tokens.get("cache_read") or 0
        inp = tokens.get("input") or 0
        if cr or inp:
            cache_read += cr
            fresh += inp
            peak = max(peak, cr + inp)
        total = tokens.get("total") or 0
        if total > max_total:
            max_total = total
            max_idx = i

    parts = []
    if cache_read + fresh > 0:
        pct = round(cache_read / (cache_read + fresh) * 100)
        parts.append(f"缓存命中 {pct}%（命中 {_fmt_k(cache_read)} / 新算 {_fmt_k(fresh)}）")
    compactions = _explicit_compaction_events(steps)
    if compactions:
        dropped = sum(int(e.get("dropped") or 0) for e in compactions)
        if dropped > 0:
            parts.append(f"压缩丢弃 {_fmt_k(dropped)} token（{len(compactions)} 次）")
        else:
            parts.append(f"压缩 {len(compactions)} 次")
    if max_idx is not None and max_total > 0:
        parts.append(f"最大单步 {_fmt_k(max_total)}（第 {max_idx} 步）")
    if peak > 0:
        parts.append(f"上下文峰值 {_fmt_k(peak)}")
    if not parts:
        return ""
    return (
        "<div class='token-summary-line'>"
        + "<span class='token-summary-sep'> · </span>".join(
            f"<span>{p}</span>" for p in parts
        )
        + "</div>"
    )


def build_overview_outputs(session: LoadedSession, *, ai_pending: bool = False, workflow_links: bool = True) -> dict:
    """Compute overview-related outputs: banner, KPI, metadata, and metrics markdown."""
    steps = session.steps
    raw = session.raw
    metrics = session.metrics
    message_rows = session.message_rows
    verdicts = session.verdicts
    wfmt = session.wall_clock

    summary = build_summary_outputs(session)
    ranked_issues = prepare_overview_issues(session)
    kpi_html = build_overview_kpi_html(
        metrics,
        wfmt,
        verdicts=verdicts,
        message_rows=message_rows,
        issues=ranked_issues,
        agent_summaries=session.agent_summaries,
        tokens_estimated=tokens_are_estimates(session.raw),
    )
    issues_html = build_overview_issues_html(
        session, issues=ranked_issues, ai_pending=ai_pending, workflow_links=workflow_links,
    )
    issues_payload = overview_issues_payload(ranked_issues)
    metrics_text = format_performance_md(metrics, wfmt)

    behavior_text = format_behavioral_md(metrics, diag_metrics=session.diagnostic_metrics)
    hotspots_text = _build_hotspots_md(message_rows)
    per_message_text = _build_per_message_md(message_rows)

    def _d(k):
        return raw.get(k, {}) if isinstance(raw.get(k), dict) else {}

    _md, _timing = _d("metadata"), _d("timing")
    _model_id, _, _agent_id = extract_agent_info(steps)
    session_detail = _build_session_detail_html(
        _timing,
        _md,
        model_id=_model_id,
        agent_id=_agent_id,
    )

    return {
        **summary,
        "kpi_html": kpi_html,
        "issues_html": issues_html,
        "issues_payload": issues_payload,
        "session_detail": session_detail,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": per_message_text,
    }


def shown_run(session: LoadedSession) -> Trajectory:
    """The public trajectory a session draws (one is built for a session made field by field)."""
    if session.shown is not None:
        return session.shown
    from ...api import Trajectory

    return Trajectory(format=session.format, steps=session.steps, source=session.path or None,  # type: ignore[arg-type]
                      raw=session.raw)


def view_figure(session: LoadedSession, view_id: str, *, dark: bool = False, **options) -> go.Figure:
    """A chart of the session drawn from its public view, as any application would draw it."""
    from ...figures import figure

    return figure(shown_run(session).view(view_id, **options), dark=dark)


# Overview chart slot -> the view it draws.
CHART_VIEWS = {
    "tok_fig": "token_usage",
    "dur_fig": "step_durations",
    "tl_fig": "tool_usage",
    "tool_dur_fig": "tool_durations",
    "skill_fig": "skill_calls",
    "tool_outcome_fig": "tool_outcomes",
    "agent_tok_fig": "agent_tokens",
    "swimlane_fig": "agent_swimlane",
    "plan_timeline_fig": "plan",
}


def build_chart_outputs(session: LoadedSession, dark: bool = False, *, workflow_links: bool = True) -> dict:
    """Build Overview chart figures, each from the session's public view."""
    charts = {slot: view_figure(session, view_id, dark=dark) for slot, view_id in CHART_VIEWS.items()}
    return {
        **charts,
        "gantt_fig": build_execution_gantt_chart(session.steps, dark=dark) or empty_plotly_fig(),
        "agent_cards_html": render_agent_summary_cards(session.agent_summaries, workflow_links=workflow_links),
    }


def build_diagnostics_outputs(session: LoadedSession, dark: bool = False, *, workflow_links: bool = True) -> dict:
    """Build diagnostics outputs from precomputed session domain data."""
    empty_fig = empty_plotly_fig()
    interactions = session.file_interactions
    target_files = session.target_files
    file_chart = view_figure(session, "file_interactions", dark=dark) if interactions else empty_fig

    rootcause_html = build_root_cause_html(session.clusters, workflow_links=workflow_links) if session.show_root_cause else ""

    pressure_fig = view_figure(session, "context_pressure", dark=dark)
    pressure_html = format_context_pressure_html(
        session.pressure_series,
        steps=session.steps,
        raw=session.raw,
        agent_key=PRESSURE_ALL_AGENTS,
    )
    pressure_choices = session.pressure_choices
    pressure_dropdown = {
        "choices": pressure_choices or [("All agents", PRESSURE_ALL_AGENTS)],
        "value": PRESSURE_ALL_AGENTS,
        "visible": len(pressure_choices) > 2,
    }

    parts = []
    chain_metrics = session.chain_metrics
    if chain_metrics["total_chains"]:
        parts.append(ui_text("{count} failure chain(s)", count=chain_metrics['total_chains']))
    if session.clusters:
        parts.append(ui_text("{count} root cause(s)", count=len(session.clusters)))
    if session.performance_bottlenecks:
        parts.append(ui_text("{count} bottleneck(s)", count=len(session.performance_bottlenecks)))
    if interactions:
        unique_files = len({i["path"] for i in interactions})
        parts.append(ui_text("{files} file(s) touched · {edited} edited", files=unique_files, edited=len(target_files)))
    compaction_count = len(session.pressure_series.get("events") or [])
    if compaction_count:
        parts.append(ui_text("{count} compaction event(s)", count=compaction_count))
    summary = " &middot; ".join(parts) if parts else ui_text(
        "Some diagnostic checks are unavailable for this telemetry capture."
        if session.raw.get("_otel") else "No diagnostic issues detected."
    )
    summary_html = f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>{summary}</div>"

    return {
        "diag_summary_html": summary_html,
        "diag_pressure_html": pressure_html,
        "diag_pressure_dropdown": pressure_dropdown,
        "diag_pressure_chart": pressure_fig,
        "diag_file_chart": file_chart,
        "diag_rootcause_html": rootcause_html,
    }


def load_warnings_html(session: LoadedSession) -> str:
    """HTML warning strip: truncation, plus anything the loader had to skip."""
    style = "color:#d97706;font-size:13px;margin:0 0 4px;"
    lines: list[str] = []
    if session.truncated:
        extra = session.steps_total - MAX_STEPS
        lines.append(
            f"<p style='{style}'>&#9888; "
            + ui_text("Showing first {limit} of {total} steps ({truncated} truncated).",
                      limit=f"{MAX_STEPS:,}", total=f"{session.steps_total:,}", truncated=f"{extra:,}")
            + "</p>"
        )
    raw = session.raw if isinstance(session.raw, dict) else {}
    if raw.get("_otel"):
        quality = session.metrics.get("data_quality", {})
        missing = [key for key, value in quality.get("fields", {}).items()
                   if value.get("status") != "observed"]
        lines.append(
            f"<p style='{style}'>OpenTelemetry capture: totals describe captured operations; "
            "capture completeness is unknown. Native transcript detail may be unavailable.</p>"
        )
        if missing:
            lines.append(f"<p style='{style}'>Partial or unavailable evidence: {html.escape(', '.join(missing))}.</p>")
    if tokens_are_estimates(raw) and (session.metrics.get("tokens") or {}).get("total"):
        lines.append(
            f"<p style='{style}'>&#9888; "
            + ui_text("Token counts are estimates from the logged text (about 4 characters per token): "
                      "this export does not record the tokens the model was billed for.")
            + "</p>"
        )
    for warning in raw.get("_load_warnings") or []:
        if isinstance(warning, str) and warning:
            lines.append(f"<p style='{style}'>&#9888; {html.escape(warning)}</p>")
    return "".join(lines)
