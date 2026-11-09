"""Every analysis of one trajectory, computed on first use and cached.

This is the single orchestration of the analysis functions: which detector
runs on which steps, and which results feed which. The public
:class:`agent_trajviz.Trajectory`, the JSON views and the dashboard's
``LoadedSession`` all read from an :class:`Analysis`, so none of them can
compute a figure the others do not.

Nothing here builds HTML, figures or UI state.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from .analytics import compute_step_analytics
from .context_usage import (
    PRESSURE_ALL_AGENTS,
    context_pressure_series,
    detect_compaction_aftermath,
    detect_premature_compactions,
    pressure_agent_choices,
    step_context_occupancy,
)
from .diagnostics import (
    annotate_clusters_with_agents,
    cluster_errors,
    compute_failure_chain_metrics,
    detect_failure_chains,
    detect_performance_bottlenecks,
    extract_file_interactions,
    identify_target_files,
    link_chains_to_agents,
)
from .metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_diagnostic_metrics,
    compute_health_verdict,
    compute_metrics,
    wall_clock_fmt,
)
from .parser import step_token_segments, usable_token_count
from .patterns import (
    build_structural_phase_segments,
    compute_plan_metrics,
    detect_blind_retries,
    detect_edit_thrash,
    detect_failure_patterns,
    detect_fruitless_streaks,
    detect_oversized_outputs,
    detect_phase_anomalies,
    detect_repeated_searches,
    detect_tool_selection_antipatterns,
    detect_tool_sequences,
    detect_user_repeats,
    detect_redundant_re_reads,
    detect_unvalidated_writes,
    extract_plan_history,
)
from .telemetry_quality import data_quality, field_observed, is_otel_step, supports

# Formats whose error text is too uniform for clustering to name a root cause.
NOISY_ROOT_CAUSE_FORMATS = frozenset({"opencode", "codearts", "codex"})


def _top_growth_rows(message_rows: list[dict], limit: int, steps: list[dict] | None) -> list[dict]:
    """Steps that grew the live context the most before the agent's next call.

    The growth of an assistant step is how much larger the next prompt of
    the same agent is: the step's own completion (output + reasoning) plus
    the new content it triggered (tool results, user messages). Rows from
    other agents interleaved between the two calls are skipped, so
    sub-agent traffic never counts as the parent's growth. The last call of
    an agent has no successor and thus no growth to report.
    """
    # One reverse pass remembers each agent's next positive-prompt assistant,
    # so a long single-agent trajectory stays linear. A stable sort then keeps
    # equal-growth ties in forward encounter order.
    successor: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for position in range(len(message_rows) - 1, -1, -1):
        r = message_rows[position]
        if r.get("role") != "assistant":
            continue
        step = steps[position] if steps is not None else None
        agent = ((step or {}).get("session_id") or "", r.get("agent") or "")
        total = usable_token_count(r["tokens_total"])
        components_observed = step is None or all(
            field_observed(step, f"tokens.{field}") for field in ("input", "output", "cache_read", "cache_write")
        )
        if total is None or not components_observed:
            # An unobserved model call is a gap, not an empty prompt. Do not
            # assign growth across it to an earlier call.
            successor.pop(agent, None)
            continue
        if step is None:
            fresh = r.get("non_cache_tokens", r.get("tokens_input"))
            prompt = (fresh or 0) + (r.get("cache_read") or 0)
        else:
            tokens = step.get("tokens") or {}
            segments = step_token_segments(tokens)
            if segments["reading"] == "total_only" and "context_window" not in tokens and "prompt_peak" not in tokens:
                successor.pop(agent, None)
                continue
            prompt = step_context_occupancy(step)["occupancy"]
        if prompt <= 0:
            continue
        nxt = successor.get(agent)
        successor[agent] = prompt
        if nxt is None:
            continue
        growth = nxt - prompt
        if growth <= 0:
            continue
        if step is None:
            own = max(total - prompt, 0)
        else:
            # A step can aggregate several requests, whereas its occupancy
            # represents one window. Use the disjoint completion segments.
            own = segments["output"] + segments["reasoning"]
        row = dict(r)
        row["growth"] = growth
        row["own_completion"] = own
        row["new_content"] = max(growth - own, 0)
        out.append(row)
    out.reverse()
    out.sort(key=lambda r: r["growth"], reverse=True)
    return out[:limit]


def hotspots(message_rows: list[dict], limit: int = 5, *, steps: list[dict] | None = None) -> dict[str, list[dict]]:
    """The steps worth looking at first, from the per-step metrics rows.

    ``slowest`` and ``heaviest`` are the *limit* longest and most token-heavy
    steps. ``lowest_cache_read`` is the *limit* assistant steps with the
    smallest cache-read share -- empty when no step reports any cache read,
    because a ranking of zeros points at nothing. Rows whose cache share is
    unknown (a self-inconsistent record) are never ranked by it.
    ``top_growth`` is the *limit* assistant steps that enlarged the next
    prompt of the same agent the most.
    """
    with_duration = [r for r in message_rows if r.get("duration") is not None]
    with_cache = [
        r for r in message_rows
        if r.get("role") == "assistant" and (r["tokens_total"] or 0) > 0 and r["cache_ratio"] is not None
    ]
    return {
        "slowest": sorted(with_duration, key=lambda r: r["duration"], reverse=True)[:limit],
        "heaviest": sorted(
            (row for row in message_rows if row["tokens_total"] is not None),
            key=lambda r: r["tokens_total"], reverse=True,
        )[:limit],
        "lowest_cache_read": (
            sorted(with_cache, key=lambda r: r["cache_ratio"])[:limit]
            if with_cache and any(r["cache_ratio"] > 0 for r in with_cache) else []
        ),
        "top_growth": _top_growth_rows(message_rows, limit, steps),
    }


class Analysis:
    """Lazily computed analyses of *steps* (parsed) from the export *raw*."""

    def __init__(self, steps: list[dict], raw: dict, fmt: str) -> None:
        self.steps = steps
        self.raw = raw
        self.format = fmt

    # -- per-step and session metrics ------------------------------------
    @cached_property
    def message_rows(self) -> list[dict]:
        return build_message_metrics(self.steps)

    @cached_property
    def data_quality(self) -> dict:
        return data_quality(self.steps, self.raw)

    @cached_property
    def metrics(self) -> dict:
        return compute_metrics(self.steps, self.raw, message_rows=self.message_rows)

    @cached_property
    def wall_clock(self) -> str:
        return wall_clock_fmt(self.metrics)[1]

    @cached_property
    def step_analytics(self) -> list[dict]:
        rows = compute_step_analytics(self.steps)
        for step, row in zip(self.steps, rows, strict=True):
            if step.get("timing_source") == "model_request":
                row["tool_time_share"] = None
            if not is_otel_step(step):
                continue
            for key, fields in {
                "tok_total": ("tokens.total",), "tok_per_s": ("tokens.total", "duration"),
                "cache_ratio": ("tokens.total", "tokens.cache_read"),
                "non_cache_tok": ("tokens.input",),
                "out_in_ratio": ("tokens.input", "tokens.output"), "duration_s": ("duration",),
            }.items():
                if not all(field_observed(step, field) for field in fields):
                    row[key] = None
            if field_observed(step, "tokens.input"):
                row["non_cache_tok"] = (step.get("tokens") or {}).get("input", 0)
            # Cross-operation order is not a model's sequential idle time;
            # concurrent tools and child agents routinely overlap.
            row["idle_before_s"] = None
            row["tool_time_share"] = None
            calls = step.get("tool_calls") or []
            if calls:
                from .metrics import tool_call_stats_duration_ms
                if not any(tool_call_stats_duration_ms(call) is not None for call in calls):
                    row["tool_time_ms"] = None
        return rows

    @cached_property
    def verdicts(self) -> list[dict]:
        return compute_health_verdict(self.metrics, self.step_analytics if self.steps else [])

    @cached_property
    def agent_summaries(self) -> list[dict]:
        return compute_agent_summary(self.steps, self.raw)

    @cached_property
    def hotspots(self) -> dict[str, list[dict]]:
        return hotspots(self.message_rows, steps=self.steps)

    # -- diagnostics ------------------------------------------------------
    @cached_property
    def file_interactions(self) -> list[dict]:
        return extract_file_interactions(self.steps) if supports(self.steps, "tool_inputs") else []

    @cached_property
    def target_files(self) -> set[str]:
        return identify_target_files(self.steps) if supports(self.steps, "tool_inputs") else set()

    @cached_property
    def failure_chains(self) -> list:
        if not supports(self.steps, "tool_outcomes"):
            return []
        return link_chains_to_agents(detect_failure_chains(self.steps), self.steps, self.agent_summaries)

    @cached_property
    def chain_metrics(self) -> dict:
        assistant_steps = sum(1 for s in self.steps if s.get("role") == "assistant")
        result = compute_failure_chain_metrics(self.failure_chains, assistant_steps)
        return result if supports(self.steps, "tool_outcomes") else dict.fromkeys(result)

    @cached_property
    def clusters(self) -> list:
        if not supports(self.steps, "tool_outcomes") or not supports(self.steps, "tool_outputs"):
            return []
        return annotate_clusters_with_agents(cluster_errors(self.steps), self.steps, self.agent_summaries)

    @property
    def show_root_cause(self) -> bool:
        return (self.format not in NOISY_ROOT_CAUSE_FORMATS
                and supports(self.steps, "tool_outcomes") and supports(self.steps, "tool_outputs"))

    @cached_property
    def performance_bottlenecks(self) -> list:
        if self.data_quality["source"] == "otlp":
            # Existing diagnoses infer sequential tool waits and idle gaps;
            # these assumptions do not describe an operation graph.
            return []
        return detect_performance_bottlenecks(self.steps, self.step_analytics)

    @cached_property
    def diagnostic_metrics(self) -> dict:
        traj = self.raw.get("trajectory") or self.raw.get("messages") or []
        result = compute_diagnostic_metrics(
            self.steps,
            traj if isinstance(traj, list) else [],
            tool_fail=self.metrics.get("tool_fail"),
        )
        if not supports(self.steps, "tool_outcomes"):
            for key in ("classified_error_count", "edit_success", "edit_precision"):
                result[key] = None
        if not supports(self.steps, "tool_inputs"):
            for key in ("edit_total", "edit_success", "edit_precision"):
                result[key] = None
        if self.data_quality["source"] == "otlp":
            # These legacy counters derive child sessions from transcript
            # records. OTel's explicit agent graph is exposed in agent rows.
            for key in ("subagent_session_count", "subagent_total_steps", "subagent_total_tokens"):
                result[key] = None
        return result

    # -- context window ---------------------------------------------------
    @cached_property
    def pressure_series(self) -> dict:
        steps = self.steps if supports(self.steps, "context") else []
        return context_pressure_series(steps, agent_key=PRESSURE_ALL_AGENTS, raw=self.raw)

    @cached_property
    def pressure_choices(self) -> list:
        return pressure_agent_choices(self.steps) if supports(self.steps, "context") else []

    @cached_property
    def premature_compactions(self) -> list:
        return detect_premature_compactions(self.steps, self.raw) if supports(self.steps, "context") else []

    @cached_property
    def compaction_aftermath(self) -> list:
        return detect_compaction_aftermath(self.steps)

    # -- plan and phases --------------------------------------------------
    @cached_property
    def plan_history(self) -> list:
        return (extract_plan_history(self.steps)
                if supports(self.steps, "tool_inputs") and supports(self.steps, "tool_outputs") else [])

    @cached_property
    def plan_metrics(self) -> dict:
        result = compute_plan_metrics(self.plan_history)
        if not supports(self.steps, "tool_inputs") or not supports(self.steps, "tool_outputs"):
            return {key: None if isinstance(value, (int, float)) else value for key, value in result.items()}
        return result

    @cached_property
    def structural_phases(self) -> list[dict]:
        return build_structural_phase_segments(self.steps) if supports(self.steps, "tool_inputs") else []

    @cached_property
    def phase_regressions(self) -> list:
        if not supports(self.steps, "tool_inputs"):
            return []
        return [
            a for a in detect_phase_anomalies(self.steps, self.structural_phases)
            if a.get("category") == "unintentional_drift"
        ]

    # -- behavioural patterns ---------------------------------------------
    @cached_property
    def tool_sequences(self) -> list[dict]:
        return detect_tool_sequences(self.steps)

    @cached_property
    def failure_patterns(self) -> list[dict]:
        return detect_failure_patterns(self.steps) if supports(self.steps, "tool_outcomes") else []

    @cached_property
    def fruitless_streaks(self) -> list:
        return (detect_fruitless_streaks(self.steps) if all(
            supports(self.steps, field) for field in ("tool_inputs", "tool_outputs", "tool_outcomes")
        ) else [])

    @cached_property
    def tool_selection(self) -> list:
        return detect_tool_selection_antipatterns(self.steps) if supports(self.steps, "tool_inputs") else []

    @cached_property
    def edit_thrash(self) -> list:
        return detect_edit_thrash(self.steps) if supports(self.steps, "tool_inputs") else []

    @cached_property
    def repeated_searches(self) -> list:
        return detect_repeated_searches(self.steps) if supports(self.steps, "tool_inputs") else []

    @cached_property
    def oversized_outputs(self) -> list:
        return detect_oversized_outputs(self.steps)

    @cached_property
    def blind_retries(self) -> list:
        return detect_blind_retries(self.steps)

    @cached_property
    def user_repeats(self) -> list:
        return detect_user_repeats(self.steps)

    @cached_property
    def unvalidated_writes(self) -> list:
        return detect_unvalidated_writes(self.steps)

    @cached_property
    def redundant_re_reads(self) -> list:
        return detect_redundant_re_reads(self.steps)

    # -- triage -------------------------------------------------------------
    @cached_property
    def issues(self) -> list:
        """Ranked issues (see :mod:`.issues`): errors, then waste, then latency."""
        from .issues import prepare_overview_issues

        return prepare_overview_issues(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        computed: list[Any] = [k for k in vars(self) if k not in ("steps", "raw", "format")]
        return f"Analysis(format={self.format!r}, steps={len(self.steps)}, computed={computed})"
