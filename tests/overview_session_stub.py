"""Minimal session stand-in for Overview Issues / pattern detector tests."""

from types import SimpleNamespace


def overview_session(**kwargs) -> SimpleNamespace:
    """Return a SimpleNamespace with every field ``collect_overview_issues`` reads."""
    base = dict(
        steps=[],
        failure_patterns=[],
        failure_chains=[],
        fruitless_streaks=[],
        tool_selection=[],
        plan_metrics={},
        plan_history=[],
        edit_thrash=[],
        repeated_searches=[],
        oversized_outputs=[],
        blind_retries=[],
        user_repeats=[],
        unvalidated_writes=[],
        redundant_re_reads=[],
        phase_regressions=[],
        performance_bottlenecks=[],
        premature_compactions=[],
        compaction_aftermath=[],
        file_interactions=[],
        format="",
        metrics={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)
