"""HTML and markdown panels for the replica, built only from public view data.

Every value from the trajectory is escaped: an export can contain anything.
"""

from __future__ import annotations

import html
import json
from typing import Any

E = html.escape


def _fmt_int(value: Any) -> str:
    return f"{value:,}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def _fmt_secs(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{value:.0f}s" if value < 3600 else f"{value / 60:.1f}m"


def _fmt_decimal(value: Any, digits: int = 2, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def sparkline(values: list[float], width: int = 100, height: int = 20) -> str:
    """A small inline SVG polyline of *values* (empty when there are fewer than two)."""
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    points = " ".join(f"{i * step:.1f},{height - (v - lo) / span * height:.1f}" for i, v in enumerate(vals))
    return (f"<svg class='tv-spark' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
            f"<polyline points='{points}' fill='none' stroke='#3b82f6' stroke-width='1.2'/></svg>")


# -- header strip ---------------------------------------------------------------

def banner(name: str, summary: dict, warnings: list[str]) -> str:
    rate = summary.get("tool_success_rate")
    tokens = (summary.get("tokens") or {}).get("total")
    parts = [f"<b>{E(name)}</b> — {E(summary.get('format_label') or '')}",
             f"{summary.get('steps')} steps", f"{summary.get('tool_calls')} tool calls "
             f"({'no success rate' if rate is None else f'{rate}% success'})"]
    parts.append(f"{_fmt_int(tokens)} tokens")
    html_out = f"<div class='tv-banner'>{' · '.join(parts)}</div>"
    if warnings:
        html_out += "".join(f"<div class='tv-warnings'>⚠ {E(w)}</div>" for w in warnings)
    return html_out


def error_banner(message: str) -> str:
    return f"<div class='tv-banner tv-error'>{E(message)}</div>"


def kpi_strip(summary: dict, metrics: dict, health: list[dict], issues: list[dict], agents: list[dict],
              steps: list[dict], *, limited_evidence: bool = False) -> str:
    """Five cards: steps, wall clock, tokens, issues, tool success."""
    verdict = {v["metric"]: v for v in health}
    durations = [r.get("duration") for r in steps if r.get("role") == "assistant"]
    tokens_per_step = [r.get("tokens_total") for r in steps if r.get("role") == "assistant"]

    agent_rows = "".join(
        f"<div class='tv-kpi-row'><span>▪ {E(a['label'])}</span><span>{a['step_count']}</span></div>" for a in agents
    )
    steps_card = _card("Steps", _fmt_int(summary.get("steps")),
                       f"{metrics.get('assistant_steps', 0)} assistant, {metrics.get('user_steps', 0)} user",
                       agent_rows)

    wall_card = _card("Wall-clock", _fmt_secs(summary.get("wall_clock_s")),
                      f"P95 {_fmt_decimal(metrics.get('p95_duration'), suffix='s')}", sparkline(durations))

    throughput = verdict.get("Throughput", {})
    rate = metrics.get("output_tokens_per_sec")
    timed = metrics.get("output_throughput_timed_steps")
    total = metrics.get("output_throughput_total_steps")
    sub = (f"{rate} gen tok/s · {timed}/{total} timed" if rate is not None else "no generation timing")
    note = (f"<div class='tv-kpi-note {E(throughput.get('status', ''))}'>{E(throughput.get('detail', ''))}</div>"
            if throughput else "")
    tok_card = _card("Tokens", _fmt_int((summary.get("tokens") or {}).get("total")), sub,
                     note + sparkline(tokens_per_step), status=throughput.get("status", ""))

    kinds: dict[str, int] = {}
    for issue in issues:
        kinds[issue["kind"]] = kinds.get(issue["kind"], 0) + 1
    names = {"error": "error", "antipattern": "pattern", "bottleneck": "bottleneck"}
    kind_rows = "".join(f"<div class='tv-kpi-row'><span>▪ {E(names.get(k, k))}</span><span>{n}</span></div>"
                        for k, n in kinds.items())
    issue_status = "bad" if kinds.get("error") else ("warn" if issues or limited_evidence else "good")
    issue_note = "limited evidence" if limited_evidence else ("" if issues else "none found")
    issues_card = _card("Issues", str(len(issues)), issue_note, kind_rows, status=issue_status)

    tool = verdict.get("Tool Success", {})
    success = summary.get("tool_success_rate")
    tool_card = _card("Tool success", "n/a" if success is None else f"{success}%",
                      f"{summary.get('tool_calls', 0)} calls",
                      f"<div class='tv-kpi-note {E(tool.get('status', ''))}'>{E(tool.get('detail', ''))}</div>"
                      if tool else "", status=tool.get("status", ""))
    return f"<div class='tv-kpis'>{steps_card}{wall_card}{tok_card}{issues_card}{tool_card}</div>"


def _card(label: str, value: str, sub: str, extra: str = "", *, status: str = "") -> str:
    return (f"<div class='tv-kpi {E(status)}'><div class='tv-kpi-label'>{E(label)}</div>"
            f"<div class='tv-kpi-value'>{E(value)}</div><div class='tv-kpi-sub'>{E(sub)}</div>{extra}</div>")


# -- overview panels ------------------------------------------------------------------

def issues_panel(issues: list[dict], *, limited_evidence: bool = False) -> str:
    if not issues:
        status = "limited evidence — some workflow checks are unavailable" if limited_evidence else "none found"
        return f"<div class='tv-issues'><b>Issues</b> <span class='tv-muted'>{status}</span></div>"
    cards = []
    for issue in issues:
        chips = "".join(f"<span class='tv-chip'>#{int(s)}</span>" for s in issue.get("steps") or [])
        why = (f"<details class='tv-issue-why'><summary>Why it matters</summary>{E(issue['why'])}</details>"
               if issue.get("why") else "")
        j = issue.get("judgment")
        fix = (f"<div class='tv-issue-fix'><b>Where:</b> {E(j['where'])}<br><b>Fix:</b> {E(j['fix'])}"
               + (f"<br><b>Also:</b> {E(j['also'])}" if j.get("also") else "")
               + f" <span class='tv-muted'>({E(j['confidence'])} confidence)</span></div>") if j else ""
        cards.append(f"<div class='tv-issue {E(issue['kind'])}'><div class='tv-issue-title'>{E(issue['title'])}</div>"
                     f"<div class='tv-issue-detail'>{E(issue['detail'])}</div><div>{chips}</div>{why}{fix}</div>")
    count = f"{len(issues)} issue{'s' if len(issues) != 1 else ''}"
    if limited_evidence:
        count += " · limited evidence"
    return (f"<details class='tv-issues' open><summary>Issues <span class='tv-muted'>{count}</span></summary>"
            f"<div class='tv-issues-hint'>Ranked problems — errors, then wasted work, then latency.</div>"
            f"{''.join(cards)}</details>")


def session_details(session: dict) -> str:
    def short(v: Any, n: int = 24) -> str:
        return "N/A" if not v else str(v)[:n]
    started = (session.get("started_at") or "")[:19].replace("T", " ") or "N/A"
    finished = (session.get("finished_at") or "")[:19].replace("T", " ") or "N/A"
    fields = [("Model", short(session.get("model"), 40)), ("Agent", short(session.get("agent"))),
              ("Start", started), ("End", finished), ("Session", short(session.get("session_id"), 16)),
              ("Branch", short(session.get("branch"))), ("Directory", short(session.get("directory"))),
              ("Platform", short(session.get("platform")))]
    if session.get("version"):
        fields.insert(2, ("Version", short(session.get("version"))))
    chips = "".join(f"<div class='tv-detail-chip'><span>{E(k)}</span><span>{E(v)}</span></div>" for k, v in fields)
    return f"<details><summary class='tv-muted'>Session details</summary><div class='tv-details-chips'>{chips}</div></details>"


def agent_cards(agents: list[dict]) -> str:
    if not agents:
        return "<div class='tv-muted'>No agent activity recorded.</div>"
    cards = []
    for a in agents:
        cache = a.get("cache_efficiency_pct")
        gen = a.get("output_tokens_per_sec")
        n = a["step_count"]
        spawned = (f"<div class='tv-muted'>Spawned at step #{a['spawned_by_step']}</div>"
                   if a.get("spawned_by_step") is not None else "")
        grid = "".join(f"<div><b>{E(v)}</b><span>{E(k)}</span></div>" for k, v in (
            ("Tokens", _fmt_int(a["total_tokens"])), ("Duration", _fmt_decimal(a.get("total_duration_s"), 1, "s")),
            ("Tool calls", str(a["tool_call_count"])), ("Errors", str(a["error_count"])),
            ("Cache %", "N/A" if cache is None else f"{cache:.1f}%"),
            ("Gen tok/s", "n/a" if gen is None else f"{gen:,.1f}")))
        cards.append(f"<div class='tv-agent-card'><div class='tv-agent-card-head'>{E(a['label'])}"
                     f"<span>{n} assistant step{'s' if n != 1 else ''}</span></div>{spawned}"
                     f"<div class='tv-agent-grid'>{grid}</div></div>")
    return f"<div class='tv-agent-cards'>{''.join(cards)}</div>"


_USAGE_COLORS = {"system": "#6b7280", "tools": "#7c3aed", "rules": "#e11d48", "skills": "#ca8a04", "mcp": "#0d9488",
                 "subagents": "#2563eb", "summarized": "#8b5cf6", "conversation": "#ea580c",
                 "tool_outputs": "#059669", "unattributed": "#94a3b8"}


def context_strip(pressure: dict, usage: dict) -> str:
    stats = pressure.get("stats") or {}
    window = stats.get("window_limit")
    source = stats.get("window_source") or "unknown"
    peak_pct = stats.get("peak_pct")
    head = (f"<div class='tv-context'><div>Peak <b>{_fmt_int(stats.get('peak_occupancy'))}</b> tokens"
            + (f" ({peak_pct}% of {_fmt_int(window)})" if window else "") + "</div>"
            f"<div>Window <b>{_fmt_int(window)}</b> <span class='tv-muted'>({E(source)})</span></div>"
            f"<div>Compactions <b>{stats.get('compaction_count', 0)}</b></div></div>")
    if stats.get("window_note"):
        head += f"<div class='tv-muted'>{E(stats['window_note'])}</div>"
    buckets = usage.get("buckets") or {}
    occupancy = int(usage.get("occupancy") or 0)
    denom = int(usage.get("window_limit") or 0) or occupancy
    if denom <= 0 or not any(buckets.values()):
        return head
    segs = "".join(f"<span style='flex:{n};background:{_USAGE_COLORS.get(k, '#9ca3af')}' title='{E(k)}: {n:,}'></span>"
                   for k, n in buckets.items() if n)
    rest = max(denom - sum(buckets.values()), 0)
    bar = f"<div class='tv-usage-bar'>{segs}<span style='flex:{rest}'></span></div>"
    legend = "".join(f"<span><span class='tv-swatch' style='background:{_USAGE_COLORS.get(k, '#9ca3af')}'></span>"
                     f"{E(k.replace('_', ' '))} {n:,}</span>" for k, n in buckets.items() if n)
    pct = usage.get("loaded_pct")
    title = (f"Context composition — {pct}% full ({occupancy:,} tokens)" if pct is not None
             else f"Context composition — {occupancy:,} tokens")
    return head + f"<div class='tv-muted'>{E(title)}</div>{bar}<div class='tv-usage-legend'>{legend}</div>"


def diagnostics_summary(chains: dict, clusters: dict, bottlenecks: list, files: dict, pressure: dict,
                        *, limited_evidence: bool = False) -> str:
    parts = []
    if (chains.get("metrics") or {}).get("total_chains"):
        parts.append(f"{chains['metrics']['total_chains']} failure chain(s)")
    if clusters.get("clusters"):
        parts.append(f"{len(clusters['clusters'])} root cause(s)")
    if bottlenecks:
        parts.append(f"{len(bottlenecks)} bottleneck(s)")
    if files.get("interactions"):
        edited = sum(1 for p in files["files"] if any(i["target"] and i["path"] == p for i in files["interactions"]))
        parts.append(f"{len(files['files'])} file(s) touched · {edited} edited")
    events = (pressure.get("series") or {}).get("events") or []
    if events:
        parts.append(f"{len(events)} compaction event(s)")
    if limited_evidence:
        parts.append("Some diagnostic checks are unavailable")
    return f"<div class='tv-muted'>{' · '.join(parts) if parts else 'No diagnostic issues detected.'}</div>"


def root_causes(clusters: dict) -> str:
    if not clusters.get("reliable") or not clusters.get("clusters"):
        return ""
    rows = "".join(f"<tr><td>{E(c.get('tool') or '')}</td><td>{E(c.get('error_class') or '')}</td>"
                   f"<td>{E(c.get('pattern') or '')}</td><td>{c.get('count')}</td>"
                   f"<td>{', '.join(f'#{s}' for s in (c.get('steps') or [])[:8])}</td></tr>"
                   for c in clusters["clusters"])
    return ("<h4>Root causes</h4><table class='tv-table'><tr><th>Tool</th><th>Class</th><th>Error</th>"
            f"<th>Count</th><th>Steps</th></tr>{rows}</table>")


def md_table(rows: list[dict], columns: list[tuple[str, str]], limit: int = 80) -> str:
    """A markdown table of *rows*; *columns* are (key, header) pairs."""
    if not rows:
        return "*No data*"
    def cell(v: Any) -> str:
        if isinstance(v, float):
            v = f"{v:,.2f}"
        elif isinstance(v, int) and not isinstance(v, bool):
            v = f"{v:,}"
        return str("" if v is None else v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(h for _, h in columns) + " |\n|" + "---|" * len(columns)
    body = "\n".join("| " + " | ".join(cell(r.get(k)) for k, _ in columns) + " |" for r in rows[:limit])
    more = f"\n\n*Showing {limit} of {len(rows)}.*" if len(rows) > limit else ""
    return f"{head}\n{body}{more}"


def performance_md(metrics: dict) -> str:
    tok = metrics.get("tokens") or {}
    rows = [("Wall clock", _fmt_secs(metrics.get("wall_clock"))), ("Steps", _fmt_int(metrics.get("total_steps"))),
            ("Avg / median / P95 step (s)", f"{_fmt_decimal(metrics.get('avg_duration'))} / "
             f"{_fmt_decimal(metrics.get('median_duration'))} / {_fmt_decimal(metrics.get('p95_duration'))}"),
            ("Total tokens", _fmt_int(tok.get("total"))), ("Input", _fmt_int(tok.get("input"))),
            ("Output", _fmt_int(tok.get("output"))), ("Cache read", _fmt_int(tok.get("cache_read"))),
            ("Cache write", _fmt_int(tok.get("cache_write"))),
            ("Gen tok/s", "n/a" if metrics.get("output_tokens_per_sec") is None else str(metrics["output_tokens_per_sec"])),
            ("Total processed tok/s", "n/a" if metrics.get("tokens_per_second") is None else str(metrics["tokens_per_second"])),
            ("First response latency (s)", "n/a" if metrics.get("time_to_first_token") is None else str(metrics["time_to_first_token"]))]
    return "### Performance\n\n| Metric | Value |\n|---|---:|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def behavior_md(metrics: dict, diagnostics: dict) -> str:
    rate = metrics.get("tool_success_rate")
    rows = [("Tool calls", _fmt_int(metrics.get("tool_call_count"))),
            ("Tool success", "n/a" if rate is None else f"{rate}%"),
            ("Failed calls", _fmt_int(metrics.get("tool_fail"))),
            ("Command success", "n/a" if metrics.get("command_success_rate") is None else str(metrics["command_success_rate"])),
            ("Tool time share", "n/a" if metrics.get("tool_time_fraction") is None else str(metrics["tool_time_fraction"])),
            ("Edit precision", "n/a" if diagnostics.get("edit_precision") is None else str(diagnostics["edit_precision"])),
            ("Classified errors", _fmt_int(diagnostics.get("classified_error_count")))]
    breakdown = metrics.get("tool_breakdown") or {}
    tools = "\n".join(f"| {E(str(k))} | {v:,} |" for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1]))
    return ("### Tool use\n\n| Metric | Value |\n|---|---:|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
            + ("\n\n| Tool | Calls |\n|---|---:|\n" + tools if tools else ""))


def hotspots_md(hot: dict) -> str:
    cols = [("index", "Step"), ("role", "Role"), ("duration", "Duration (s)"), ("tokens_total", "Tokens"),
            ("tool_calls", "Tool calls")]
    out = ("### Hotspots\n\n**Slowest steps**\n\n" + md_table(hot.get("slowest") or [], cols)
           + "\n\n**Most tokens**\n\n" + md_table(hot.get("heaviest") or [], cols))
    if hot.get("lowest_cache_read"):
        out += "\n\n**Least cache reuse**\n\n" + md_table(hot["lowest_cache_read"], [
            ("index", "Step"), ("cache_ratio", "Cache read"), ("non_cache_tokens", "Fresh input"),
            ("tokens_total", "Tokens")])
    return out


def per_step_md(rows: list[dict]) -> str:
    return "### Per-step metrics\n\n" + md_table(rows, [
        ("index", "Step"), ("role", "Role"), ("agent", "Agent"), ("finish", "Finish"), ("duration", "Duration (s)"),
        ("tokens_total", "Tokens"), ("cache_ratio", "Cache read"), ("non_cache_tokens", "Fresh input"),
        ("tool_calls", "Tool calls")])


# -- patterns ----------------------------------------------------------------------------

def tool_sequences_html(seqs: list[dict]) -> str:
    if not seqs:
        return "<div class='tv-muted'>No repeated tool sequences.</div>"
    rows = "".join(f"<tr><td>{' → '.join(E(t) for t in s['sequence'])}</td><td>{s['frequency']}</td>"
                   f"<td>{', '.join(f'#{i}' for i in s['step_indices'][:10])}</td></tr>" for s in seqs)
    return f"<table class='tv-table'><tr><th>Sequence</th><th>Times</th><th>Starts at</th></tr>{rows}</table>"


def failure_patterns_html(patterns: list[dict], *, limited_evidence: bool = False) -> str:
    if not patterns:
        message = "Failure checks have limited evidence." if limited_evidence else "No failed tool calls."
        return f"<div class='tv-muted'>{message}</div>"
    rows = "".join(f"<tr><td>{E(p['cluster_label'])}</td><td>{p['count']}</td><td>{E(p.get('error_class') or '')}</td>"
                   f"<td>{' → '.join(E(t) for t in p.get('recovery_path') or [])}</td>"
                   f"<td>{', '.join(f'#{s}' for s in p.get('steps') or [])}</td></tr>" for p in patterns)
    return ("<table class='tv-table'><tr><th>Failure</th><th>Count</th><th>Class</th><th>Then</th>"
            f"<th>Steps</th></tr>{rows}</table>")


def antipatterns_html(views: dict, *, limited_evidence: bool = False) -> str:
    sections = []
    for vid, title, fmt in (
        ("fruitless_streaks", "Fruitless search streaks",
         lambda r: f"steps {r.get('start_step')}–{r.get('end_step')} ({r.get('length')} searches)"),
        ("tool_selection", "Tool selection", lambda r: f"#{r.get('step')}: {r.get('command') or r.get('detail') or ''}"),
        ("edit_thrash", "Edit thrash", lambda r: f"{r.get('path')}: {r.get('count')} edits, steps "
                                                  f"{r.get('start_step')}–{r.get('end_step')}"),
        ("repeated_searches", "Repeated searches", lambda r: f"{r.get('pattern') or r.get('query') or ''} "
                                                           f"×{r.get('count', '')}"),
        ("phase_regressions", "Phase regressions", lambda r: r.get("explanation") or ""),
        ("premature_compactions", "Premature compactions", lambda r: json.dumps(r, ensure_ascii=False)[:160]),
    ):
        rows = views.get(vid) or []
        items = "".join(f"<li>{E(str(fmt(r)))}</li>" for r in rows[:20])
        empty = "Not established from available evidence" if limited_evidence else "none"
        sections.append(f"<h4>{E(title)} <span class='tv-muted'>({len(rows)})</span></h4>"
                        + (f"<ul>{items}</ul>" if items else f"<div class='tv-muted'>{empty}</div>"))
    return "".join(sections)


# -- workflow ---------------------------------------------------------------------------

def _step_text(step: dict, limit: int = 240) -> str:
    texts = [p.get("text", "") for p in step.get("parts") or [] if isinstance(p, dict) and p.get("type") == "text"]
    text = " ".join(t for t in texts if isinstance(t, str)).strip() or step.get("text_preview") or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


_PART_KINDS = {"text": "text", "reasoning": "thought", "tool": "tool", "tool_call": "tool"}


def _part_kinds(step: dict) -> str:
    present = {_PART_KINDS.get(part.get("type", "")) for part in step.get("parts") or [] if isinstance(part, dict)}
    if step.get("tool_calls"):
        present.add("tool")
    return " · ".join(k for k in ("text", "thought", "tool") if k in present)


def _badge(text: str, cls: str) -> str:
    return f"<span class='tv-badge-sm {E(cls)}'>{E(text)}</span>"


def workflow_toc(workflow: dict, visible: list[int]) -> str:
    """The step list beside the cards: number and role of every visible step."""
    rows = "".join(f"<div class='tv-toc-row'><b>#{workflow['steps'][p]['step']}</b>"
                   f"{_badge(workflow['steps'][p]['role'].upper(), workflow['steps'][p]['role'])}</div>"
                   for p in visible)
    return f"<div class='tv-toc'><div class='tv-kpi-label'>Steps</div>{rows}</div>"


def workflow_cards(workflow: dict, steps: list[dict], visible: list[int]) -> str:
    labels = {a["id"]: a["label"] for a in workflow.get("agents") or []}
    cards = []
    for pos in visible:
        row, step = workflow["steps"][pos], steps[pos]
        badges = _badge(f"#{row['step']}", "num") + _badge(row["role"].upper(), row["role"])
        if row["category"] not in (row["role"], "final"):
            badges += _badge(row["category"].replace("_", " ").upper(), "category " + row["category"])
        if labels:
            badges += _badge(labels.get(row["agent"], row["agent"] or "main"), "agent")
        size = " ".join(x for x in (
            f"{row['duration_s']:.2f}s" if isinstance(row.get("duration_s"), (int, float)) else "",
            f"{row['tokens']:,} tok" if isinstance(row.get("tokens"), int) else "",
            f"{row['tool_calls']} tool(s)" if row.get("tool_calls") else "") if x)
        body = E(_step_text(step))
        cards.append(f"<div class='tv-step-card {E(row['category'])} {E(row['role'])}'>"
                     f"<div class='tv-step-head'>{badges}<span>{E(_part_kinds(step))}</span></div>"
                     f"<div class='tv-step-size'>{E(size)}</div>"
                     + (f"<div class='tv-step-text'>{body}</div>" if body else "") + "</div>")
    if not cards:
        return "<div class='tv-muted'>No steps match the filters.</div>"
    return f"<div class='tv-card-list'>{''.join(cards)}</div>"


def step_detail(step: dict | None) -> str:
    if not step:
        return "<div class='tv-muted'>Select a step to inspect.</div>"
    out = [f"<div class='tv-detail'><h3>Step #{step.get('index')} · {E(str(step.get('role')))}</h3>",
           f"<div class='tv-muted'>agent {E(step.get('agent') or 'main')} · model {E(step.get('model_id') or 'n/a')} · "
           f"finish {E(step.get('finish') or 'n/a')} · duration {step.get('duration')} s · "
           f"{_fmt_int((step.get('tokens') or {}).get('total'))} tokens</div>"]
    for part in step.get("parts") or []:
        if not isinstance(part, dict) or part.get("type") not in ("text", "reasoning"):
            continue
        out.append(f"<h4>{E(part['type'].title())}</h4><pre>{E(str(part.get('text') or ''))}</pre>")
    for tc in step.get("tool_calls") or []:
        status = tc.get("status") or ""
        out.append(f"<h4>Tool: {E(tc.get('tool_name') or '')} <span class='tv-muted'>{E(status)}</span></h4>")
        out.append(f"<pre>{E(json.dumps(tc.get('input') or {}, ensure_ascii=False, indent=2)[:4000])}</pre>")
        if tc.get("output"):
            out.append(f"<div class='tv-muted'>output</div><pre>{E(str(tc['output'])[:4000])}</pre>")
        if tc.get("error"):
            out.append(f"<div class='tv-muted'>error</div><pre>{E(str(tc['error'])[:2000])}</pre>")
    out.append("</div>")
    return "".join(out)


# -- comparison and attribution ---------------------------------------------------------------

def scorecard_html(rows: list[dict]) -> str:
    cols = [("label", "Run"), ("format", "Format"), ("steps", "Steps"), ("tokens", "Tokens"),
            ("wall_clock_fmt", "Wall clock"), ("tool_calls", "Tool calls"), ("tool_success_pct", "Tool success %"),
            ("peak_pct", "Context peak %"), ("compactions", "Compactions"), ("finished", "Finished"), ("error", "Error")]
    head = "".join(f"<th>{E(h)}</th>" for _, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{E('' if r.get(k) is None else str(r.get(k)))}</td>" for k, _ in cols)
                   + "</tr>" for r in rows)
    return f"<table class='tv-table'><tr>{head}</tr>{body}</table>"


def behavior_html(behavior: dict | None) -> str:
    if not behavior:
        return ""
    sims = behavior.get("similarity") or []
    sim_rows = "".join(f"<li>{E(str(s.get('a')))} ↔ {E(str(s.get('b')))}: F1 {s.get('f1')}</li>"
                       for s in sims if isinstance(s, dict))
    patterns = behavior.get("patterns_vs_baseline") or []
    return ("<h4>Behaviour</h4>" + (f"<ul>{sim_rows}</ul>" if sim_rows else "")
            + f"<div class='tv-muted'>{len(patterns)} pattern(s) against the baseline; "
              f"{behavior.get('action_matrix_total', 0)} distinct actions, "
              f"{behavior.get('file_matrix_total', 0)} files, {behavior.get('tool_matrix_total', 0)} tools.</div>")


def comparison_html(summary: dict, layers: dict | None) -> str:
    out = summary.get("outcome") or {}
    align = summary.get("alignment") or {}
    conf = summary.get("confidence") or {}
    rows = [("Reference", out.get("reference_filename"), out.get("reference_steps"), out.get("reference_tokens"),
             "stopped normally" if out.get("reference_success") else "did not stop normally"),
            ("Compared", out.get("compared_filename"), out.get("compared_steps"), out.get("compared_tokens"),
             "stopped normally" if out.get("compared_success") else "did not stop normally")]
    table = "".join(f"<tr><td>{E(str(a))}</td><td>{E(str(b))}</td><td>{E(str(c))}</td><td>{_fmt_int(d)}</td>"
                    f"<td>{E(e)}</td></tr>" for a, b, c, d, e in rows)
    metrics = "".join(f"<tr><td>{E(k.replace('_', ' '))}</td><td>{E(str(v))}</td></tr>" for k, v in align.items())
    layer_rows = "".join(f"<tr><td>{E(k.replace('_', ' '))}</td><td>{E(str(v.get('label') or v.get('score') or ''))}"
                         f"</td><td>{E(str(v.get('description') or ''))}</td></tr>"
                         for k, v in (layers or {}).items() if isinstance(v, dict))
    return ("<h4>Outcome</h4><table class='tv-table'><tr><th></th><th>File</th><th>Steps</th><th>Tokens</th>"
            f"<th>Finish</th></tr>{table}</table><h4>Alignment</h4><table class='tv-table'>{metrics}</table>"
            + (f"<h4>Evaluation layers</h4><table class='tv-table'>{layer_rows}</table>" if layer_rows else "")
            + f"<div class='tv-muted'>Confidence: {E(json.dumps(conf))} · evidence: "
              f"{E(str(summary.get('evidence_level')))}</div>")


def attribution_html(result: dict) -> str:
    if not result.get("available"):
        return f"<div class='tv-muted'>No gold-grounded attribution — {E(result.get('reason') or 'unavailable')}</div>"
    primary = result.get("primary") or {}
    rows = "".join(f"<tr><td>{E(str(c.get('capability') or c.get('name') or ''))}</td>"
                   f"<td>{E(str(c.get('status') or c.get('verdict') or ''))}</td>"
                   f"<td>{E(str(c.get('evidence') or c.get('detail') or ''))[:200]}</td></tr>"
                   for c in result.get("scorecard") or [] if isinstance(c, dict))
    faults = "".join(f"<li>{E(json.dumps(f, ensure_ascii=False)[:240])}</li>" for f in result.get("faults") or [])
    return (f"<h4>Primary failure</h4><div>{E(json.dumps(primary, ensure_ascii=False)[:600])}</div>"
            f"<h4>Capability scorecard</h4><table class='tv-table'>{rows}</table>"
            + (f"<h4>Faults</h4><ul>{faults}</ul>" if faults else ""))


def labels_badge(summary: dict) -> str:
    return (f"<span class='tv-badge'>Labels loaded — {summary.get('classified')}/{summary.get('total')} assistant "
            f"steps classified, {summary.get('phases')} phases</span> <span class='tv-muted'>taxonomy "
            f"{E(str(summary.get('taxonomy_version')))} · model {E(str(summary.get('model')))}</span>")
