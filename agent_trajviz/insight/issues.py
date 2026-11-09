"""Issue triage: turn a run's diagnostics into a ranked list of issues.

Each issue says what went wrong, why it matters and which steps show it. It
is built only from results the analysis already computed -- no detector runs
twice -- and carries plain text; the dashboard renders it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from .context_usage import format_token_count
from .tool_failure import tool_call_failed

if TYPE_CHECKING:
    from .analysis import Analysis
    from .session import LoadedSession

IssueKind = Literal["error", "risk", "antipattern", "bottleneck"]
Confidence = Literal["high", "medium", "low"]

# Lower = more urgent. Errors before correctness risks before waste before
# latency.
_SEVERITY: dict[IssueKind, int] = {
    "error": 0,
    "risk": 1,
    "antipattern": 2,
    "bottleneck": 3,
}

@dataclass(frozen=True)
class IssueJudgment:
    """LLM-suggested Change/Fix for one OverviewIssue (on-demand judge)."""

    where: str
    fix: str
    also: str = ""
    confidence: Confidence = "medium"


@dataclass(frozen=True)
class OverviewIssue:
    """One Overview triage item (session diagnostics; optional LLM judgment)."""

    kind: IssueKind
    title: str
    detail: str
    why: str = ""
    steps: tuple[int, ...] = ()
    source_id: str = ""
    judgment: IssueJudgment | None = None

    def with_judgment(self, judgment: IssueJudgment) -> OverviewIssue:
        return replace(self, judgment=judgment)


def rank_issues(issues: list[OverviewIssue]) -> list[OverviewIssue]:
    """Sort by severity, then earliest step, then title. Show all (panel is foldable)."""
    return sorted(
        issues,
        key=lambda i: (
            _SEVERITY[i.kind],
            min(i.steps) if i.steps else 10**9,
            i.title,
        ),
    )


_CASCADE_COVER_FRAC = 0.8


def dedupe_overlapping_issues(issues: list[OverviewIssue]) -> list[OverviewIssue]:
    """Drop issues that restate another issue's failures.

    Failure cascades that are mostly covered by an error-cluster issue add little
    beyond consecutive-thrash chrome, so they are removed when ≥80% of their
    steps already appear on a ``fail:`` card (includes ``fail:system:``).
    """
    fail_steps = {
        step
        for issue in issues
        if issue.source_id.startswith("fail:")
        for step in issue.steps
    }
    if not fail_steps:
        return list(issues)

    kept: list[OverviewIssue] = []
    for issue in issues:
        if issue.source_id.startswith("cascade:") and issue.steps:
            covered = sum(1 for s in issue.steps if s in fail_steps)
            if covered / len(issue.steps) >= _CASCADE_COVER_FRAC:
                continue
        kept.append(issue)
    return kept


def prepare_overview_issues(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Collect, drop overlaps, and rank — the dashboard Issues leaf list."""
    return rank_issues(dedupe_overlapping_issues(collect_overview_issues(session)))


def collect_overview_issues(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Map the analysis results of *session* into issues (no re-detection)."""
    issues: list[OverviewIssue] = []
    issues.extend(_from_failure_patterns(session))
    issues.extend(_from_failure_chains(session))
    issues.extend(_from_antipatterns(session))
    issues.extend(_from_premature_compactions(session))
    issues.extend(_from_compaction_aftermath(session))
    issues.extend(_from_bottlenecks(session))
    return issues


_SYSTEM_ERROR_HIGH = 5


def _count_label(label: str, count: int) -> str:
    return f"{label} ({count}×)" if count else label


def _short_display(text: str, limit: int = 48) -> str:
    """Truncate *text* for an issue detail/title, keeping *limit* chars."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "…"


def _from_failure_patterns(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []
    for i, pat in enumerate(session.failure_patterns or []):
        label = str(pat.get("cluster_label") or "Unknown error")
        count = int(pat.get("count") or 0)
        example = str(pat.get("example_error") or "")[:200]
        recovery = pat.get("recovery_path")
        steps = tuple(int(s) for s in (pat.get("steps") or []) if s is not None)
        error_class = str(pat.get("error_class") or "tool").lower()

        if error_class == "system":
            if count >= _SYSTEM_ERROR_HIGH:
                title = f"Frequent system errors: {_count_label(label, count)}"
                why = (
                    "Many Read/Grep/Edit-style failures in one run — each is usually "
                    "low risk, but volume suggests path, permission, or harness setup "
                    "problems worth checking."
                )
            else:
                title = f"System error: {_count_label(label, count)}"
                why = (
                    "Scaffold/tooling miss (search, read, or write) — usually low risk; "
                    "the agent can often recover without an author change."
                )
            out.append(
                OverviewIssue(
                    kind="antipattern",
                    title=title,
                    detail=example or "system tool failure",
                    why=why,
                    steps=steps,
                    source_id=f"fail:system:{i}:{label}",
                )
            )
            continue

        if recovery and isinstance(recovery, (list, tuple)):
            why = "Typical recovery: " + " → ".join(str(t) for t in recovery)
        else:
            why = "No recovery path found after this error cluster."
        out.append(
            OverviewIssue(
                kind="error",
                title=_count_label(label, count),
                detail=example,
                why=why,
                steps=steps,
                source_id=f"fail:{i}:{label}",
            )
        )
    return out


def _from_failure_chains(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Consecutive assistant error runs (cascades). Skip length-1 (covered by clusters)."""
    out: list[OverviewIssue] = []
    chains = session.failure_chains or []
    for i, chain in enumerate(chains):
        steps = tuple(int(s) for s in (chain.get("steps") or []) if s is not None)
        if len(steps) < 2:
            continue
        start = chain.get("start", steps[0])
        end = chain.get("end", steps[-1])
        out.append(
            OverviewIssue(
                kind="error",
                title=f"Failure cascade ({len(steps)} steps)",
                detail=f"steps {start}–{end}",
                why=(
                    "Consecutive assistant steps failed without a clean recovery in between; "
                    "cascades often amplify one root error into thrash."
                ),
                steps=steps,
                source_id=f"cascade:{i}:{start}-{end}",
            )
        )
    return out


def _from_antipatterns(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []

    # Skip tool-error rollup when failure_patterns already cover the same failures.
    if not session.failure_patterns:
        error_count, error_steps = count_tool_errors(session.steps)
        if error_count > 0:
            out.append(
                OverviewIssue(
                    kind="error",
                    title=_count_label("Tool errors", error_count),
                    detail="detected from tool output (platform, permission, missing file)",
                    why=(
                        "Failed tool calls cost tokens and turns to recover from, and often "
                        "indicate environment problems rather than agent mistakes."
                    ),
                    steps=tuple(error_steps),
                    source_id="antipattern:tool_errors",
                )
            )

    streaks = session.fruitless_streaks or []
    if streaks:
        total_wasted = sum(int(s.get("length") or 0) for s in streaks)
        shown = streaks[:3]
        streak_desc = ", ".join(
            f"steps {s.get('start_step')}-{s.get('end_step')} ({s.get('length')})"
            for s in shown
        )
        remaining = len(streaks) - len(shown)
        if remaining > 0:
            remaining_len = sum(int(s.get("length") or 0) for s in streaks[len(shown):])
            streak_desc += f", +{remaining} more ({remaining_len})"
        streak_indices: list[int] = []
        for s in streaks:
            streak_indices.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Fruitless search streaks", len(streaks)),
                detail=f"{total_wasted} wasted steps — {streak_desc}",
                why=(
                    "Three or more consecutive searches that returned no matches; "
                    "sustained streaks suggest looking in the wrong place."
                ),
                steps=tuple(streak_indices),
                source_id="antipattern:fruitless",
            )
        )

    tool_selection = session.tool_selection or []
    if tool_selection:
        bash_steps = [
            int(f["step"]) for f in tool_selection if f.get("step") is not None
        ]
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Bash-for-reading", len(tool_selection)),
                detail="steps used sed/cat/head instead of Read tool",
                why=(
                    "Shell reads bypass structured Read tooling — no line numbers, "
                    "weaker cache, larger context."
                ),
                steps=tuple(bash_steps),
                source_id="antipattern:bash_read",
            )
        )

    stalled = (session.plan_metrics or {}).get("stalled") or []
    if stalled:
        items_desc = ", ".join(f"'{s.get('content', '')[:30]}'" for s in stalled[:2])
        stall_steps: list[int] = []
        for s in stalled:
            stall_steps.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Stalled plan items", len(stalled)),
                detail=items_desc,
                why=(
                    "Todo items stayed in_progress without completion — often a "
                    "context switch that never closed the loop."
                ),
                steps=tuple(stall_steps),
                source_id="antipattern:stalled_plan",
            )
        )

    plan_resets = int((session.plan_metrics or {}).get("plan_resets") or 0)
    if plan_resets > 0:
        plan_steps = sorted({
            int(snap["step"])
            for snap in (session.plan_history or [])
            if snap.get("step") is not None
        })
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Plan resets", plan_resets),
                detail="todo list content replaced with no overlapping items",
                why=(
                    "A full plan rewrite mid-run usually means the agent abandoned context "
                    "instead of completing or revising items in place."
                ),
                steps=tuple(plan_steps[:12]),
                source_id="antipattern:plan_resets",
            )
        )

    for i, thrash in enumerate(session.edit_thrash or []):
        path = str(thrash.get("path") or "")
        count = int(thrash.get("count") or 0)
        fail_count = int(thrash.get("fail_count") or 0)
        steps = tuple(int(s) for s in (thrash.get("steps") or []) if s is not None)
        short = path if len(path) <= 48 else ("…" + path[-47:])
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Failed edit retries on {short} ({count}×)",
                detail=(
                    f"{fail_count} failed write(s) in steps "
                    f"{thrash.get('start_step')}–{thrash.get('end_step')}"
                ),
                why=(
                    "Same file rewritten after write failures — usually a bad path, "
                    "patch mismatch, or approach that needs a precondition check."
                ),
                steps=steps,
                source_id=f"antipattern:edit_thrash:{i}:{path}",
            )
        )

    for i, rep in enumerate(session.repeated_searches or []):
        display = str(rep.get("display") or rep.get("signature") or "")
        count = int(rep.get("count") or 0)
        steps = tuple(int(s) for s in (rep.get("steps") or []) if s is not None)
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Repeated empty search ({count}×)",
                detail=_short_display(display),
                why=(
                    "The same search ran multiple times with empty results (not only as a "
                    "consecutive streak) — the query or path is likely wrong."
                ),
                steps=steps,
                source_id=f"antipattern:repeated_search:{i}",
            )
        )

    oversized = session.oversized_outputs or []
    if oversized:
        shown = oversized[:3]
        desc = ", ".join(
            f"step {o['step']} {o['tool']} ≈{int(o['tokens']):,} tokens"
            + (f" ({int(o['share'])}% of context)" if o.get("share") is not None else "")
            for o in shown
        )
        remaining = len(oversized) - len(shown)
        if remaining > 0:
            desc += f", +{remaining} more"
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Oversized tool outputs", len(oversized)),
                detail=desc,
                why=(
                    "A tool result this large stays in the conversation history, so every "
                    "later step pays for it again until a compaction drops it. Typical "
                    "sources: whole-file cat, unbounded find, verbose test output."
                ),
                steps=tuple(sorted({int(o["step"]) for o in oversized})),
                source_id="antipattern:oversized_output",
            )
        )

    for i, retry in enumerate(session.blind_retries or []):
        tool = str(retry.get("tool") or "tool")
        attempts = int(retry.get("attempts") or 0)
        failed = int(retry.get("failed") or 0)
        steps = tuple(int(s) for s in (retry.get("steps") or []) if s is not None)
        display = str(retry.get("display") or "")
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Blind retry of failed {tool} call ({attempts}×)",
                detail=(
                    f"{failed} of {attempts} identical attempts failed — "
                    f"{_short_display(display)}"
                ),
                why=(
                    "A failed call was resubmitted verbatim and failed again — unchanged "
                    "input in an unchanged environment yields the same result, so the "
                    "retries only burned tokens and turns."
                ),
                steps=steps,
                source_id=f"antipattern:blind_retry:{i}",
            )
        )

    for i, rep in enumerate(session.user_repeats or []):
        message = str(rep.get("message") or "")
        count = int(rep.get("count") or 0)
        steps = tuple(int(s) for s in (rep.get("steps") or []) if s is not None)
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"User repeated a message ({count}×)",
                detail=_short_display(message, 60),
                why=(
                    "The user sent the exact same message again — the most direct evidence "
                    "available that the previous response did not satisfy them; no "
                    "inference required."
                ),
                steps=steps,
                source_id=f"antipattern:user_repeat:{i}",
            )
        )

    unvalidated = session.unvalidated_writes or []
    if unvalidated:
        u = unvalidated[0]
        write_steps = tuple(int(s) for s in (u.get("steps") or []) if s is not None)
        first, last = write_steps[0], write_steps[-1]
        out.append(
            OverviewIssue(
                kind="antipattern",
                title="Code changes were never validated",
                detail=(
                    f"{int(u.get('files') or 0)} code file(s) written in steps "
                    f"{first}–{last}; no test/build/lint command in the whole run"
                ),
                why=(
                    "Every write succeeded but nothing checked the result — correctness "
                    "rests on the agent's confidence alone, and regressions it cannot "
                    "see have no chance of being caught."
                ),
                steps=write_steps[:12],
                source_id="antipattern:unvalidated_writes",
            )
        )

    for i, reread in enumerate(session.redundant_re_reads or []):
        path = str(reread.get("path") or "")
        count = int(reread.get("count") or 0)
        steps = tuple(int(s) for s in (reread.get("steps") or []) if s is not None)
        step_list = ", ".join(str(s) for s in steps)
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Re-read an unchanged file ({count}×)",
                detail=(
                    f"{path} re-read {count}× with no write or compaction in "
                    f"between (steps {step_list})"
                ),
                why=(
                    "Nothing wrote to the file and no compaction dropped it — the "
                    "content was already in context, so each re-read spends fresh "
                    "tokens for nothing."
                ),
                steps=steps,
                source_id=f"antipattern:redundant_reread:{i}",
            )
        )

    return out


def _from_premature_compactions(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Compactions that ran before the window was full, or grew it."""
    out: list[OverviewIssue] = []
    flagged = session.premature_compactions or []
    if not flagged:
        return out

    def _fmt(n: object) -> str:
        return format_token_count(int(n)) if isinstance(n, (int, float)) else "?"

    grew = [c for c in flagged if c.get("grew")]
    if grew:
        worst = max(grew, key=lambda c: int(c.get("occupancy_after") or 0))
        detail = ", ".join(
            f"step {c.get('step')}: {_fmt(c.get('occupancy_before'))} → {_fmt(c.get('occupancy_after'))}"
            for c in grew[:4]
        )
        if len(grew) > 4:
            detail += f", +{len(grew) - 4} more"
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Compactions that grew the context window", len(grew)),
                detail=detail,
                why=(
                    f"Compacting left the window at least as large as before (worst: "
                    f"{_fmt(worst.get('occupancy_before'))} → {_fmt(worst.get('occupancy_after'))}) — "
                    "the stored summary cost more tokens than the compacted messages freed, "
                    "so the compaction was wasted work at that point."
                ),
                steps=tuple(int(c.get("step", 0)) for c in grew),
                source_id="antipattern:compaction_grew",
            )
        )

    premature = [c for c in flagged if not c.get("grew")]
    if premature:
        pcts = [
            float(c["before_pct"])
            for c in premature
            if isinstance(c.get("before_pct"), (int, float))
        ]
        limit = _fmt(premature[0].get("window_limit"))
        pct_desc = (
            f"{min(pcts):.0f}–{max(pcts):.0f}% of the assumed {limit}-token window"
            if pcts
            else "well below the window limit"
        )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("Premature compactions", len(premature)),
                detail=f"window occupancy at compaction time: {pct_desc}",
                why=(
                    "The context window was compacted well before it was full — usually "
                    "an agent-initiated compress/compact call rather than a harness-forced "
                    "one. Context was lost early; check the stored summaries kept what "
                    "mattered."
                ),
                steps=tuple(int(c.get("step", 0)) for c in premature),
                source_id="antipattern:compaction_premature",
            )
        )
    return out


def _from_compaction_aftermath(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Correctness-risk card: a compaction between two consecutive writes
    thinned the reads feeding the second write (see
    ``detect_compaction_aftermath``)."""
    out: list[OverviewIssue] = []
    for i, ev in enumerate(session.compaction_aftermath or []):
        if not isinstance(ev, dict) or not isinstance(ev.get("reads_before_tokens"), int):
            continue
        c_step = int(ev.get("step") or 0)
        before = int(ev.get("reads_before_tokens") or 0)
        after = int(ev.get("reads_after_tokens") or 0)
        pct = ev.get("shrink_pct")
        w_prev = ev.get("prev_write_step")
        w_next = ev.get("next_write_step")
        dropped = ev.get("dropped")
        segs = []
        if isinstance(dropped, int):
            segs.append(f"compaction dropped {format_token_count(dropped)} tokens")
        segs.append(
            f"consecutive writes at steps {w_prev} and {w_next} span it: "
            f"~{format_token_count(before)} tokens of file reads before vs "
            f"~{format_token_count(after)} after ({pct}% less)"
        )
        out.append(
            OverviewIssue(
                kind="risk",
                title=f"Post-compaction write may rest on summarized context (#{c_step})",
                detail="; ".join(segs),
                why=(
                    "Two consecutive writes span this compaction, and the file reads "
                    "between them shrank after it: the information behind the second "
                    "write survives mostly as the compaction summary rather than fresh "
                    "reads. How much a write needs to read varies with the task — "
                    "follow the step links to review the second write."
                ),
                steps=(c_step, int(w_next or c_step)),
                source_id=f"antipattern:compaction_thin_read:{i}:{c_step}",
            )
        )
    return out


def _from_bottlenecks(session: LoadedSession | Analysis) -> list[OverviewIssue]:
    """Map detected performance bottlenecks (outlier + clear cause), not top-N slow steps."""
    out: list[OverviewIssue] = []
    for i, bn in enumerate(session.performance_bottlenecks or []):
        step_idx = bn.get("step_idx")
        if step_idx is None:
            continue
        idx = int(step_idx)
        title = str(bn.get("title") or f"Performance bottleneck at #{idx}")
        detail = str(bn.get("detail") or "")[:200]
        why = str(bn.get("why") or (
            "Session-relative duration outlier with a dominant tool, idle/queue, "
            "or context/inference cause."
        ))
        out.append(
            OverviewIssue(
                kind="bottleneck",
                title=title,
                detail=detail,
                why=why,
                steps=(idx,),
                source_id=f"bottleneck:{bn.get('cause', 'unknown')}:{i}:{idx}",
            )
        )
    return out


def count_tool_errors(steps: list[dict]) -> tuple[int, list[int]]:
    """Return ``(failed_call_count, step_indices)`` using :func:`tool_call_failed`."""
    error_steps: list[int] = []
    error_count = 0
    for step in steps:
        failed = [tc for tc in (step.get("tool_calls") or []) if tool_call_failed(tc)]
        if not failed:
            continue
        error_count += len(failed)
        error_steps.append(int(step.get("index", 0)))
    return error_count, error_steps


def _indices_for_step_range(start: int | None, end: int | None, *, max_span: int = 5) -> list[int]:
    """Expand a streak range to step indices; long ranges keep only endpoints."""
    if start is None:
        return []
    if end is None or end == start:
        return [int(start)]
    lo, hi = int(start), int(end)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo + 1 <= max_span:
        return list(range(lo, hi + 1))
    return [lo, hi]
