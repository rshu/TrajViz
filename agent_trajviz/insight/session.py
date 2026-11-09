"""Gradio-free load pipeline: ingest a trajectory into a LoadedSession DTO.

The dashboard reads files through the public :func:`agent_trajviz.load`, so it
opens whatever the library opens (formats registered by plugins, exported
``trajectory.json`` documents), and a session is a view of one public
:class:`~agent_trajviz.Trajectory`.
"""

from __future__ import annotations

import dataclasses
import os
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .analysis import NOISY_ROOT_CAUSE_FORMATS
from .loaders import detect_format, tokens_are_estimates
from .parser import parse_steps

if TYPE_CHECKING:
    from ..api import Trajectory

logger = logging.getLogger(__name__)

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
MAX_STEPS = 2000

LoadErrorCode = Literal["not_found", "mismatch", "unknown", "read_error", "analysis_error"]

_NOISY_ROOT_CAUSE_FORMATS = NOISY_ROOT_CAUSE_FORMATS  # kept for importers of the old name


@dataclass(frozen=True)
class LoadError:
    """Failed ingest. Messages are plain text — the UI packer wraps HTML."""

    code: LoadErrorCode
    message: str
    selected: str | None = None
    detected: str | None = None


@dataclass
class LoadedSession:
    """Domain snapshot of a successfully loaded trajectory (no HTML / Plotly / Gradio)."""

    path: str
    raw: dict
    steps: list[dict]
    steps_total: int
    format: str
    message_rows: list[dict]
    metrics: dict
    verdicts: list[dict]
    agent_summaries: list[dict]
    step_analytics: list[dict]
    wall_clock: str
    tool_sequences: list[dict]
    failure_patterns: list[dict]
    file_interactions: list[dict]
    target_files: set[str]
    failure_chains: list
    chain_metrics: dict
    clusters: list
    performance_bottlenecks: list
    diagnostic_metrics: dict
    pressure_series: dict
    pressure_choices: list
    show_root_cause: bool
    plan_history: list
    plan_metrics: dict
    fruitless_streaks: list
    tool_selection: list
    edit_thrash: list
    repeated_searches: list
    #: Single tool outputs big enough to tax every later step
    #: (see detect_oversized_outputs).
    oversized_outputs: list
    #: A failed tool call retried verbatim and failing again
    #: (see detect_blind_retries).
    blind_retries: list
    #: User messages sent verbatim two or more times
    #: (see detect_user_repeats).
    user_repeats: list
    #: Code files written while no validation command ran at all
    #: (see detect_unvalidated_writes).
    unvalidated_writes: list
    #: Same file re-read with no write or compaction in between
    #: (see detect_redundant_re_reads).
    redundant_re_reads: list
    # Computed on every load but read by no presenter, chart, report or UI
    # module — only by the two tests that construct a LoadedSession by keyword.
    # Kept rather than deleted: the field has no default, so dropping it is a
    # breaking change to every constructor site, and wiring it to Overview
    # Issues instead would add new antipattern issues to most runs, i.e. a
    # product decision about what the Overview claims, not a cleanup.
    phase_regressions: list
    premature_compactions: list
    #: A compaction between two consecutive writes whose file reads shrank
    #: afterward (see detect_compaction_aftermath).
    compaction_aftermath: list = dataclasses.field(default_factory=list)
    truncated: bool = False
    #: The trajectory as loaded (every step): comparisons and run groups use it.
    run: Trajectory | None = None
    #: What the dashboard draws, at most MAX_STEPS steps of ``run``: views and figures come from it.
    shown: Trajectory | None = None


def _path_exists(file_path: str) -> bool:
    return os.path.isfile(file_path) or os.path.isdir(file_path)


def load_session(path: str, format_hint: str = "", *, session_id: str | None = None) -> LoadedSession | LoadError:
    """Load and analyze a trajectory file. Does not build HTML, Plotly, or Gradio updates."""
    if not path or not _path_exists(path):
        return LoadError(code="not_found", message="No file selected or file not found.")

    from ..api import FORMATS, load
    from ..errors import LoadError as TrajectoryLoadError

    hint = format_hint if format_hint in FORMATS else None
    try:
        run = load(path, format=hint, session_id=session_id) if session_id else load(path, format=hint)
    except TrajectoryLoadError as exc:
        if exc.code == "mismatch":
            return LoadError(code="mismatch", message=str(exc), selected=hint, detected=exc.detected)
        if exc.code == "unknown_format":
            return LoadError(
                code="unknown",
                message=(
                    "Could not detect trajectory format. "
                    "Select a format from the dropdown and try again."
                ),
            )
        return LoadError(code="read_error", message=str(exc))

    # Same promise as the loader: a LoadedSession or a LoadError, never an
    # exception. The step model is type-normalised in parse_steps, so this
    # guard is the backstop for an analyser bug, not the routine path for
    # malformed input. The traceback goes to the log.
    try:
        return session_from_run(run, path)
    except Exception as exc:  # noqa: BLE001 — see above
        logger.exception("could not analyse %s", path)
        return LoadError(
            code="analysis_error",
            message=f"The file loaded but could not be analysed ({type(exc).__name__}: {str(exc)[:160]}).",
        )


def build_loaded_session(path: str, raw: dict, *, detected: str | None = None) -> LoadedSession:
    """Analyze already-loaded *raw* into a ``LoadedSession`` (no file I/O)."""
    from ..api import Trajectory

    detected = detected if detected is not None else detect_format(raw)
    run = Trajectory(
        format=detected,
        steps=parse_steps(raw),  # type: ignore[arg-type]
        source=path or None,
        sha256=raw.get("_source_sha256"),
        warnings=tuple(str(w) for w in raw.get("_load_warnings") or ()),
        tokens_estimated=tokens_are_estimates(raw),
        raw=raw,
    )
    return session_from_run(run, path)


def session_from_run(run: Trajectory, path: str | None = None) -> LoadedSession:
    """The dashboard's session for a public trajectory, showing at most ``MAX_STEPS`` steps."""
    steps_total = len(run.steps)
    truncated = steps_total > MAX_STEPS
    shown = dataclasses.replace(run, steps=run.steps[:MAX_STEPS]) if truncated else run
    # One orchestration for every consumer: the trajectory's own analysis.
    a = shown._analysis
    steps = a.steps

    return LoadedSession(
        path=path if path is not None else (run.source or ""),
        raw=run.raw,
        steps=steps,
        steps_total=steps_total,
        format=run.format,
        message_rows=a.message_rows,
        metrics=a.metrics,
        verdicts=a.verdicts,
        agent_summaries=a.agent_summaries,
        step_analytics=a.step_analytics,
        wall_clock=a.wall_clock,
        tool_sequences=a.tool_sequences,
        failure_patterns=a.failure_patterns,
        file_interactions=a.file_interactions,
        target_files=a.target_files,
        failure_chains=a.failure_chains,
        chain_metrics=a.chain_metrics,
        clusters=a.clusters,
        performance_bottlenecks=a.performance_bottlenecks,
        diagnostic_metrics=a.diagnostic_metrics,
        pressure_series=a.pressure_series,
        pressure_choices=a.pressure_choices,
        show_root_cause=a.show_root_cause,
        plan_history=a.plan_history,
        plan_metrics=a.plan_metrics,
        fruitless_streaks=a.fruitless_streaks,
        tool_selection=a.tool_selection,
        edit_thrash=a.edit_thrash,
        repeated_searches=a.repeated_searches,
        oversized_outputs=a.oversized_outputs,
        blind_retries=a.blind_retries,
        user_repeats=a.user_repeats,
        unvalidated_writes=a.unvalidated_writes,
        redundant_re_reads=a.redundant_re_reads,
        phase_regressions=a.phase_regressions,
        premature_compactions=a.premature_compactions,
        compaction_aftermath=a.compaction_aftermath,
        truncated=truncated,
        run=run,
        shown=shown,
    )
