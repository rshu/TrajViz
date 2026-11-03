"""Every chart and table TrajViz computes, as JSON-serializable views.

A :class:`View` is one chart's or table's data with an id, a title, a kind
(``"chart"``, ``"table"`` or ``"summary"``) and a description. Views carry
data, never figures or HTML: the dashboard draws its charts from exactly
these numbers, so an application can render them any way it likes, or write
them out with :func:`export_json`.

    >>> run = tv.load("trajectory.json")
    >>> run.view("tool_usage").data["tools"][0]
    >>> tv.export_json(run, "out/")          # one <id>.json per view + index.json

:data:`VIEW_CATALOG` lists the single-run views; :func:`comparison_views`
turns a :func:`compare` report into views the same way.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import json
import math
import os
import re
import tempfile
from importlib import resources
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from .insight import chart_data
from .insight.telemetry_quality import supports, view_availability
from .insight.context_usage import (
    PRESSURE_ALL_AGENTS,
    context_pressure_series,
    context_pressure_stats,
    context_usage_breakdown,
    usage_snapshot_choices,
)

if TYPE_CHECKING:
    from .api import Trajectory
    from .insight.analysis import Analysis

#: Version of the layout of :meth:`View.to_dict` and the files :func:`export_json` writes.
VIEWS_SCHEMA_VERSION = 2

ViewKind = Literal["chart", "table", "summary"]


@dataclass(frozen=True)
class View:
    """One chart's or table's data, ready to serialize."""

    #: Stable identifier, a key of :data:`VIEW_CATALOG` (or of :func:`comparison_views`).
    id: str
    title: str
    kind: ViewKind
    #: What the view shows and how it was computed, in plain words.
    description: str
    #: JSON-serializable: dicts, lists, strings, numbers, booleans and None.
    data: Any
    #: Evidence availability for this view; does not assert capture completeness.
    availability: Mapping[str, Any] | None = None

    # Export preflight only: never part of the public value or serialized schema.
    _input_sources: tuple[str, ...] = dataclasses.field(default=(), init=False, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """The view as one JSON object."""
        return {
            "schema_version": VIEWS_SCHEMA_VERSION,
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "description": self.description,
            "data": self.data,
            **({"availability": dict(self.availability)} if self.availability is not None else {}),
        }

    def to_json(self, **kwargs: Any) -> str:
        """:meth:`to_dict` as JSON text; keyword arguments go to :func:`json.dumps`."""
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)


def _jsonable(value: Any) -> Any:
    """*value* as plain JSON types. Sets become sorted lists, NaN/inf become None.

    Raises :class:`TypeError` for anything else: a view that is not data is a
    bug to fix, not a value to stringify.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, View):
        # Match the existing dataclass JSON shape without private export metadata.
        return _jsonable({field.name: getattr(value, field.name)
                          for field in dataclasses.fields(value) if field.init})
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable({field.name: getattr(value, field.name)
                          for field in dataclasses.fields(value)})
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"not JSON data: {type(value).__name__}")


def _steps(run: Trajectory) -> list[dict[str, Any]]:
    """The steps as the analysis modules type them (plain dicts)."""
    return cast("list[dict[str, Any]]", run.steps)


@dataclass(frozen=True)
class _Spec:
    kind: ViewKind
    title: str
    description: str
    build: Callable[..., Any]
    #: Keyword options ``build`` accepts (documented in the catalog).
    options: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: JSON Schema of a registered view's data (built-in views ship theirs as files).
    schema: Mapping[str, Any] | None = None


def _summary(run: Trajectory, a: Analysis) -> dict[str, Any]:
    m = a.metrics
    return {
        "format": run.format,
        "format_label": run.format_label,
        "source": run.source,
        "sha256": run.sha256,
        "warnings": list(run.warnings),
        "tokens_estimated": run.tokens_estimated,
        "steps": m.get("total_steps"),
        "assistant_steps": m.get("assistant_steps"),
        "agents": len(a.agent_summaries),
        "wall_clock_s": m.get("wall_clock"),
        "tokens": m.get("tokens"),
        "tool_calls": m.get("tool_call_count"),
        "tool_success_rate": m.get("tool_success_rate"),
        "output_tokens_per_sec": m.get("output_tokens_per_sec"),
        "health": {v["metric"]: v["status"] for v in a.verdicts},
        "issues": len(a.issues),
        "data_quality": a.data_quality,
    }


def _session(run: Trajectory, a: Analysis) -> dict[str, Any]:
    from .insight.formats.common import as_text
    from .insight.metrics import extract_agent_info

    md_raw, timing_raw = run.raw.get("metadata"), run.raw.get("timing")
    md: dict[str, Any] = md_raw if isinstance(md_raw, dict) else {}
    timing: dict[str, Any] = timing_raw if isinstance(timing_raw, dict) else {}
    model_id, provider_id, agent_id = extract_agent_info(_steps(run))
    return {
        "model": as_text(model_id) or as_text(md.get("model")) or None,
        "provider": as_text(provider_id) or None,
        "agent": as_text(agent_id) or as_text(md.get("agent")) or None,
        "started_at": as_text(timing.get("started_at")) or None,
        "finished_at": as_text(timing.get("finished_at")) or None,
        "session_id": as_text(md.get("session_id")) or None,
        "title": as_text(md.get("title")) or None,
        "branch": as_text(md.get("branch")) or None,
        "directory": as_text(md.get("directory_name")) or None,
        "platform": as_text(md.get("platform")) or None,
        "version": as_text(md.get("server_version")) or None,
    }


def _context_pressure(
    run: Trajectory,
    a: Analysis,
    *,
    agent: str | None = None,
    window_limit: int | float | None = None,
    snapshot_step: int | None = None,
) -> dict[str, Any]:
    key = agent or PRESSURE_ALL_AGENTS
    if key == PRESSURE_ALL_AGENTS and window_limit is None:
        series = a.pressure_series
    else:
        steps = _steps(run) if supports(_steps(run), "context") else []
        series = context_pressure_series(steps, agent_key=key, raw=run.raw, window_limit=window_limit)
    return {
        "agent": key,
        "agents": [{"label": label, "key": value} for label, value in a.pressure_choices],
        "highlight_step": snapshot_step,
        "stats": context_pressure_stats(series),
        "series": series,
    }


def _context_usage(
    run: Trajectory,
    a: Analysis,
    *,
    agent: str | None = None,
    snapshot_step: int | None = None,
    window_limit: int | float | None = None,
) -> dict[str, Any]:
    steps = _steps(run) if all(supports(_steps(run), field) for field in ("context", "text", "tool_inputs", "tool_outputs")) else []
    breakdown = context_usage_breakdown(
        steps, raw=run.raw, agent_key=agent, window_limit=window_limit, snapshot_step=snapshot_step,
    )
    snapshots = usage_snapshot_choices(steps, agent_key=agent)
    return {**breakdown, "snapshots": [{"label": label, "value": value} for label, value in snapshots]}


def _plan(run: Trajectory, a: Analysis) -> dict[str, Any]:
    return {"metrics": a.plan_metrics, "history": a.plan_history}


def _failure_chains(run: Trajectory, a: Analysis) -> dict[str, Any]:
    return {"metrics": a.chain_metrics, "chains": a.failure_chains}


def _error_clusters(run: Trajectory, a: Analysis) -> dict[str, Any]:
    # Clustering names root causes poorly on formats whose error text is
    # uniform; the dashboard hides the panel there, the data says so.
    return {"reliable": a.show_root_cause, "clusters": a.clusters}


_SPECS: dict[str, _Spec] = {
    "summary": _Spec(
        "summary", "Summary",
        "Provenance and headline numbers: format, steps, agents, tokens, tool use, health and issue count.",
        _summary),
    "data_quality": _Spec(
        "summary", "Data quality",
        "Evidence coverage and missing-data reasons over captured operations; capture completeness is unknown.",
        lambda run, a: a.data_quality),
    "session": _Spec(
        "summary", "Session details",
        "Where and how the run happened: model, provider, agent, start and end, session id, title, "
        "branch, directory, platform and client version, as the export records them (null if absent).",
        _session),
    "metrics": _Spec(
        "summary", "Session metrics",
        "Every session metric: tokens by field, timing and latency, throughput, tool use and failures, "
        "finish reasons. A rate is null when its denominator is empty.",
        lambda run, a: a.metrics),
    "health": _Spec(
        "table", "Health checks",
        "One verdict per check (good/warn/bad/unavailable) with the value it judged and why.",
        lambda run, a: a.verdicts),
    "issues": _Spec(
        "table", "Issues",
        "Ranked triage of what went wrong: errors, then wasted work, then latency; each issue names its steps.",
        lambda run, a: a.issues),
    "agents": _Spec(
        "table", "Agents",
        "One row per agent (main and sub-agents): steps, tokens, duration, tool calls, errors, cache share "
        "and generation rate.",
        lambda run, a: a.agent_summaries),
    "steps": _Spec(
        "table", "Per-step metrics",
        "One row per step: role, agent, finish, duration, tokens, cache share, fresh input and tool calls.",
        lambda run, a: a.message_rows),
    "step_analytics": _Spec(
        "table", "Per-step analytics",
        "One row per step: tool-time share, token rate, cache ratio, output/input ratio and idle gap.",
        lambda run, a: a.step_analytics),
    "hotspots": _Spec(
        "table", "Hotspots",
        "The slowest steps, the most token-heavy steps and the assistant steps with the least cache reuse.",
        lambda run, a: a.hotspots),
    "token_usage": _Spec(
        "chart", "Token usage by step",
        "Per-step tokens in disjoint segments (fresh input, cache read, cache write, output, reasoning) that "
        "add up to the step total; steps with an impossible count list it under `invalid`.",
        lambda run, a: chart_data.token_usage(run.steps, run.format)),
    "step_durations": _Spec(
        "chart", "Step durations",
        "Per-step duration without sub-agent spawn wait, its error kind, and whether it is a spike.",
        lambda run, a: chart_data.step_durations(run.steps)),
    "tool_usage": _Spec(
        "chart", "Tool calls",
        "Calls per tool, split by agent.",
        lambda run, a: chart_data.tool_usage(run.steps)),
    "tool_durations": _Spec(
        "chart", "Tool call durations",
        "Every timed tool call per tool; parallel calls each count. Spawn tools are left out.",
        lambda run, a: chart_data.tool_durations(run.steps)),
    "tool_outcomes": _Spec(
        "chart", "Tool outcomes",
        "Every tool call with its step, agent and success.",
        lambda run, a: chart_data.tool_outcomes(run.steps)),
    "skill_calls": _Spec(
        "chart", "Skill calls",
        "Which agent invoked which skill, and how often.",
        lambda run, a: chart_data.skill_calls(run.steps)),
    "agent_tokens": _Spec(
        "chart", "Tokens by agent",
        "Token totals per agent with the segments that add up to them.",
        lambda run, a: chart_data.agent_tokens(a.agent_summaries)),
    "agent_swimlane": _Spec(
        "chart", "Agent swimlane",
        "Each agent's contiguous step ranges with their tokens and tool calls; the user has a lane of its own.",
        lambda run, a: chart_data.agent_swimlane(run.steps)),
    "file_interactions": _Spec(
        "chart", "File interactions",
        "Every file read, write, search or skill load by step and agent; `target` marks files the run edited.",
        lambda run, a: chart_data.file_interactions(a.file_interactions, a.target_files, run.steps)),
    "plan": _Spec(
        "chart", "Plan progress",
        "The agent's todo items with the steps they started and finished on, stalled items, and every "
        "plan snapshot.",
        _plan),
    "context_pressure": _Spec(
        "chart", "Context window pressure",
        "Context-window occupancy per agent by step, compaction events, and the window it is measured "
        "against (with how that window was determined). `agents` lists the agent keys it can show.",
        _context_pressure,
        {"agent": "agent key from `agents` (default: all agents)",
         "window_limit": "context window in tokens, overriding the declared or inferred one",
         "snapshot_step": "step to highlight (a pre-compaction snapshot)"}),
    "context_usage": _Spec(
        "chart", "Context composition",
        "What fills one agent's context window, by category, at the latest occupancy point or a "
        "snapshot. `snapshots` lists the selectable pre-compaction points.",
        _context_usage,
        {"agent": "agent key (default: all agents)",
         "snapshot_step": "a step from `snapshots` (default: the current window)",
         "window_limit": "context window in tokens"}),
    "workflow": _Spec(
        "table", "Workflow",
        "One entry per step for a step browser: workflow role, what the step leads with, its agent, "
        "and the filter labels it matches. Step contents are in Trajectory.steps.",
        lambda run, a: chart_data.workflow(run.steps)),
    "phases": _Spec(
        "table", "Workflow phases",
        "Contiguous structural phases of the run (exploration, editing, validation, ...).",
        lambda run, a: a.structural_phases),
    "tool_sequences": _Spec(
        "table", "Tool sequences",
        "Recurring sequences of tool calls.",
        lambda run, a: a.tool_sequences),
    "failure_patterns": _Spec(
        "table", "Failure patterns",
        "Failed tool calls grouped by tool and error.",
        lambda run, a: a.failure_patterns),
    "failure_chains": _Spec(
        "table", "Failure chains",
        "Runs of consecutive failing assistant steps, with chain statistics.",
        _failure_chains),
    "error_clusters": _Spec(
        "table", "Error clusters",
        "Errors clustered by message into candidate root causes; `reliable` is false for formats whose "
        "error text is too uniform to cluster.",
        _error_clusters),
    "bottlenecks": _Spec(
        "table", "Performance bottlenecks",
        "Steps that are outliers in duration with a clear cause (tool wait, idle gap, generation).",
        lambda run, a: a.performance_bottlenecks),
    "fruitless_streaks": _Spec(
        "table", "Fruitless search streaks",
        "Consecutive searches that found nothing.",
        lambda run, a: a.fruitless_streaks),
    "tool_selection": _Spec(
        "table", "Tool selection anti-patterns",
        "Shell commands used where a dedicated tool exists (for example `cat` instead of Read).",
        lambda run, a: a.tool_selection),
    "edit_thrash": _Spec(
        "table", "Edit thrash",
        "Files edited repeatedly in a short span.",
        lambda run, a: a.edit_thrash),
    "repeated_searches": _Spec(
        "table", "Repeated searches",
        "The same search run more than once.",
        lambda run, a: a.repeated_searches),
    "phase_regressions": _Spec(
        "table", "Phase regressions",
        "Unintended returns to an earlier workflow phase.",
        lambda run, a: a.phase_regressions),
    "premature_compactions": _Spec(
        "table", "Premature compactions",
        "Context compactions that happened well before the window was full.",
        lambda run, a: a.premature_compactions),
    "diagnostics": _Spec(
        "summary", "Diagnostic metrics",
        "Derived diagnostics: error recovery, retries, exploration ratio and related measures.",
        lambda run, a: a.diagnostic_metrics),
}

_BUILT_IN_VIEWS = frozenset(_SPECS)
# View ids are file names in an export; these two are taken by export_json.
_VIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_VIEW_IDS = frozenset({"index", "trajectory"})
_VIEW_KINDS = frozenset(get_args(ViewKind))


def _catalog_entry(spec: _Spec) -> Mapping[str, Any]:
    return MappingProxyType({"title": spec.title, "kind": spec.kind, "description": spec.description,
                             "options": MappingProxyType(dict(spec.options))})


_CATALOG: dict[str, Mapping[str, Any]] = {vid: _catalog_entry(spec) for vid, spec in _SPECS.items()}

#: Single-run view id -> ``{"title", "kind", "description", "options"}``, in display order.
#: ``options`` maps each keyword :meth:`Trajectory.view` accepts for the view to its meaning.
#: Includes the views registered with :func:`register_view`, after the built-in ones.
VIEW_CATALOG: Mapping[str, Mapping[str, Any]] = MappingProxyType(_CATALOG)


def _validate_view_id(view_id: object) -> None:
    if not isinstance(view_id, str) or not _VIEW_ID.fullmatch(view_id) or view_id.casefold() in _RESERVED_VIEW_IDS:
        raise ValueError(f"invalid view id {view_id!r}: use letters, digits, '_', '.' and '-' "
                         f"(not {' or '.join(sorted(_RESERVED_VIEW_IDS))})")


def register_view(
    view_id: str,
    build: Callable[..., Any],
    *,
    title: str,
    kind: ViewKind = "table",
    description: str = "",
    options: Mapping[str, str] | None = None,
    schema: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> None:
    """Add a view of your own: ``build(run, **options)`` returns its JSON data.

    *run* is the :class:`Trajectory`; its other views, :attr:`~Trajectory.metrics`
    and steps are at hand. The view joins :data:`VIEW_CATALOG` after the built-in
    views, so :meth:`Trajectory.views`, :func:`export_json`, ``agent-trajviz
    --json`` and the dashboards' view lists include it. *options* maps each
    keyword *build* accepts to its meaning; :meth:`Trajectory.view` passes them.
    *schema* is the JSON Schema of the data, returned within :func:`view_schema`.

        >>> tv.register_view("edits", lambda run: {"count": sum(...)}, title="Edits")
        >>> run.view("edits").data

    Raises :class:`ValueError` for an id that is malformed (it names a file in
    an export), built in, or already registered (pass ``replace=True`` to
    replace your own), or for an unknown *kind*.
    """
    _validate_view_id(view_id)
    if view_id in _BUILT_IN_VIEWS:
        raise ValueError(f"{view_id!r} is a built-in view")
    if view_id in _SPECS and not replace:
        raise ValueError(f"view {view_id!r} is already registered; pass replace=True to replace it")
    if kind not in _VIEW_KINDS:
        raise ValueError(f"unknown view kind {kind!r}; expected one of: {', '.join(sorted(_VIEW_KINDS))}")
    if not callable(build):
        raise TypeError("build must be callable: build(run, **options) -> data")
    if schema is not None and not isinstance(schema, Mapping):
        raise TypeError("schema must be a JSON Schema object (a mapping)")

    def run_build(run: Trajectory, analysis: Analysis, **opts: Any) -> Any:
        return build(run, **opts)

    spec = _Spec(kind, title, description, run_build, dict(options or {}),
                 None if schema is None else copy.deepcopy(dict(schema)))
    _SPECS[view_id] = spec
    _CATALOG[view_id] = _catalog_entry(spec)


def _document_schema(view_id: str, title: str, kind: str, description: str, data: Any) -> dict[str, Any]:
    """The schema of a view document (:meth:`View.to_dict`) whose ``data`` has schema *data*."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "description": description,
        "type": "object",
        "required": ["schema_version", "id", "title", "kind", "description", "data"],
        "properties": {
            "schema_version": {"const": VIEWS_SCHEMA_VERSION},
            "id": {"const": view_id},
            "title": {"type": "string"},
            "kind": {"const": kind},
            "description": {"type": "string"},
            "data": data,
            "availability": {"type": "object"},
        },
    }


@functools.cache
def _shipped_schema_text(view_id: str) -> str | None:
    resource = resources.files(__package__) / "schemas" / f"{view_id}.schema.json"
    return resource.read_text(encoding="utf-8") if resource.is_file() else None


def _shipped_view_ids() -> list[str]:
    suffix = ".schema.json"
    names = (entry.name for entry in (resources.files(__package__) / "schemas").iterdir())
    return sorted(name[: -len(suffix)] for name in names if name.endswith(suffix))


def view_schema(view_id: str) -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of a view, as :meth:`View.to_dict` and :func:`export_json` write it.

    Every view has one: the single-run views of :data:`VIEW_CATALOG`, the
    comparison, label and run-group views, and views added with
    :func:`register_view` (described by the schema given there; without one,
    their ``data`` may be anything). The schema's ``data`` property describes
    the view's data. Records allow fields the schema does not list, so data
    from a newer release still validates. Returns a copy; raises
    :class:`KeyError` for an unknown id.

        >>> import jsonschema                         # any JSON Schema validator
        >>> jsonschema.validate(view.to_dict(), tv.view_schema(view.id))
    """
    spec = _SPECS.get(view_id)
    if spec is not None and view_id not in _BUILT_IN_VIEWS:
        data = copy.deepcopy(dict(spec.schema)) if spec.schema is not None else {}
        return _document_schema(view_id, spec.title, spec.kind, spec.description, data)
    text = _shipped_schema_text(view_id) if isinstance(view_id, str) and _VIEW_ID.fullmatch(view_id) else None
    if text is None:
        raise KeyError(f"no view {view_id!r}; see agent_trajviz.view_schemas()")
    return cast("dict[str, Any]", json.loads(text))


def view_schemas() -> dict[str, dict[str, Any]]:
    """:func:`view_schema` of every view, keyed by id: :data:`VIEW_CATALOG` order first, then the others."""
    ids = list(_CATALOG) + [vid for vid in _shipped_view_ids() if vid not in _CATALOG]
    return {vid: view_schema(vid) for vid in ids}


def build_view(run: Trajectory, analysis: Analysis, view_id: str, options: Mapping[str, Any] | None = None) -> View:
    """One view of *run* (see :meth:`Trajectory.view`)."""
    try:
        spec = _SPECS[view_id]
    except KeyError:
        raise KeyError(f"unknown view id: {view_id}; see agent_trajviz.VIEW_CATALOG") from None
    options = dict(options or {})
    bad = sorted(set(options) - set(spec.options))
    if bad:
        raise TypeError(f"view {view_id!r} takes no option(s) {', '.join(bad)}; "
                        f"it takes: {', '.join(spec.options) or 'none'}")
    return View(
        view_id, spec.title, spec.kind, spec.description, _jsonable(spec.build(run, analysis, **options)),
        view_availability(view_id, analysis.data_quality, _steps(run)),
    )


def build_views(run: Trajectory, analysis: Analysis, only: Iterable[str] | None = None) -> dict[str, View]:
    """Views of *run* with default options, computed from *analysis* (see :meth:`Trajectory.views`)."""
    ids = list(_SPECS) if only is None else list(only)
    unknown = [vid for vid in ids if vid not in _SPECS]
    if unknown:
        raise KeyError(f"unknown view id(s): {', '.join(unknown)}; see agent_trajviz.VIEW_CATALOG")
    return {vid: build_view(run, analysis, vid) for vid in ids}


_COMPARISON_SPECS: dict[str, tuple[ViewKind, str, str, Callable[[dict[str, Any]], Any]]] = {
    "comparison_summary": (
        "summary", "Comparison summary",
        "The two runs, their outcomes, alignment scores, confidence and evidence level.",
        lambda r: {k: r.get(k) for k in ("task_id", "reference_agent", "compared_agent", "outcome",
                                          "alignment", "confidence", "evidence_level", "anchor_mode", "notes")}),
    "milestone_timeline": (
        "chart", "Milestones",
        "The step each run reached each milestone (first relevant file, first edit, passing validation, "
        "final patch) and the difference.",
        lambda r: {"reference": r.get("ref_milestones"), "compared": r.get("cmp_milestones"),
                   "deltas": r.get("milestones")}),
    "segment_costs": (
        "chart", "Segment costs",
        "Cost of each stretch between milestones in both runs.",
        lambda r: r.get("segments")),
    "divergence": (
        "chart", "Divergence patterns",
        "Where and how the compared run diverged from the reference, with cost.",
        lambda r: r.get("patterns")),
    "anchor_classes": (
        "chart", "Anchor write recall",
        "Recall of the anchor patch's files by class; null without an anchor patch.",
        lambda r: r.get("anchor_analysis")),
    "evaluation_layers": (
        "table", "Evaluation layers",
        "File selection, edit stability and completion quality, each with what it measures.",
        lambda r: r.get("eval_layers")),
}


def comparison_views(report: Mapping[str, Any]) -> dict[str, View]:
    """Views of a :func:`agent_trajviz.compare` report, keyed by id."""
    return {
        vid: View(vid, title, kind, description, _jsonable(build(dict(report))))
        for vid, (kind, title, description, build) in _COMPARISON_SPECS.items()
    }


def _source_views(source: Trajectory | Mapping[str, View], only: Iterable[str] | None) -> dict[str, View]:
    from .api import Trajectory

    if isinstance(source, Trajectory):
        return source.views(only=only)
    views = dict(source)
    if only is not None:
        wanted = list(only)
        missing = [vid for vid in wanted if vid not in views]
        if missing:
            raise KeyError(f"no such view(s): {', '.join(missing)}")
        views = {vid: views[vid] for vid in wanted}
    return views


def export_json(
    source: Trajectory | Mapping[str, View],
    directory: str | os.PathLike[str],
    *,
    only: Iterable[str] | None = None,
    indent: int | None = 2,
) -> list[Path]:
    """Write each view to ``<directory>/<id>.json``, plus ``index.json``; return the paths.

    *source* is a loaded trajectory (all of its views, or those in *only*) or
    a mapping of views such as :func:`comparison_views` returns. For a
    trajectory, ``trajectory.json`` holds :meth:`Trajectory.to_dict` with the
    raw export, so the directory reloads with :func:`load`. The
    directory is created if needed; existing files of the same names are
    replaced atomically per file. View ids must be valid export filenames and
    cannot collide with ``index`` or ``trajectory``. Output cannot replace
    files consumed by the trajectory or label views, including either side of
    a label comparison. Copying the view mapping preserves these safeguards.
    """
    from . import __version__
    from .api import Trajectory
    from ._fileio import ensure_output_does_not_overwrite, input_sources

    views = _source_views(source, only)
    seen: set[str] = set()
    for view in views.values():
        if not isinstance(view, View):
            raise TypeError("export_json() expects View values")
        _validate_view_id(view.id)
        filename_key = view.id.casefold()
        if filename_key in seen:
            raise ValueError(f"duplicate view id {view.id!r} (case-insensitive filename)")
        seen.add(filename_key)
    out_dir = Path(directory)
    sources = input_sources(source.source, source.raw) if isinstance(source, Trajectory) else []
    source_values = views.values() if isinstance(source, Trajectory) else source.values()
    sources.extend(path for view in source_values if isinstance(view, View) for path in view._input_sources)
    sources = list(dict.fromkeys(sources))
    planned = [*(view.id + ".json" for view in views.values()), "index.json"]
    if isinstance(source, Trajectory):
        planned.append("trajectory.json")
    for name in planned:
        ensure_output_does_not_overwrite(out_dir / name, sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(name: str, payload: Any) -> None:
        path = out_dir / name
        text = json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_dir,
                                             prefix=".trajviz-", suffix=".tmp", delete=False) as handle:
                tmp = Path(handle.name)
                handle.write(text)
            os.replace(tmp, path)
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
        written.append(path)

    for view in views.values():
        _write(f"{view.id}.json", view.to_dict())
    index: dict[str, Any] = {
        "schema_version": VIEWS_SCHEMA_VERSION,
        "generator": f"agent-trajviz {__version__}",
        "views": [{"id": v.id, "title": v.title, "kind": v.kind, "file": f"{v.id}.json"} for v in views.values()],
    }
    if isinstance(source, Trajectory):
        _write("trajectory.json", source.to_dict(include_raw=True))
        index.update(source=source.source, sha256=source.sha256, format=source.format,
                     trajectory_file="trajectory.json")
    _write("index.json", index)
    return written


# -- step labels (agent-trajviz-label sidecars) --------------------------------

def _labels_data(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    from .errors import LoadError
    from .insight.labels import load_labeled_json

    if isinstance(source, Mapping):
        return dict(source)
    path = os.fspath(source)
    try:
        return load_labeled_json(path)
    except (OSError, ValueError) as exc:
        raise LoadError(f"not a step-labels file: {exc}", code="invalid_labels", path=path) from exc


def _aggregate_label_source(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    from .insight.labels import aggregate_labels

    inputs = [] if isinstance(source, Mapping) else [str(Path(source).resolve())]
    data = aggregate_labels(_labels_data(source), _input_files=inputs)
    return data, tuple(dict.fromkeys(inputs))


def _with_input_sources(views: dict[str, View], sources: Iterable[str]) -> dict[str, View]:
    paths = tuple(dict.fromkeys(str(Path(path).resolve()) for path in sources))
    for view in views.values():
        object.__setattr__(view, "_input_sources", paths)
    return views


def label_views(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, View]:
    """Views of a step-labels sidecar written by ``agent-trajviz-label``.

    *source* is the ``*_labeled.json`` path or its parsed content. Steps
    labelled ``unknown`` count in ``labels_summary`` but are left out of the
    distributions and the timeline. Raises :class:`LoadError` (code
    ``"invalid_labels"``) for a file that is not a labels sidecar.
    """
    agg, inputs = _aggregate_label_source(source)
    specs: list[tuple[str, ViewKind, str, str, Any]] = [
        ("labels_summary", "summary", "Labels summary",
         "Assistant steps labelled, how many were classified, and the taxonomy and model used.",
         {k: agg.get(k) for k in ("total", "classified", "unknown", "classification_rate",
                                  "taxonomy_version", "model")} | {"phases": len(agg["phase_counts"])}),
        ("label_phase_counts", "chart", "Steps by phase",
         "Classified assistant steps per workflow phase.",
         {"counts": agg["phase_counts"]}),
        ("label_action_counts", "chart", "Steps by action",
         "Classified assistant steps per action, with each action's phase.",
         {"counts": agg["action_counts"], "action_to_phase": agg["action_to_phase"]}),
        ("label_phase_durations", "chart", "Time by phase",
         "Seconds spent in each phase.",
         {"durations_s": agg["phase_durations"]}),
        ("label_action_durations", "chart", "Time by action",
         "Seconds spent in each action, with each action's phase.",
         {"durations_s": agg["action_durations"], "action_to_phase": agg["action_to_phase"]}),
        ("label_timeline", "chart", "Label timeline",
         "Every labelled step in order with its phase, action and duration (user turns included).",
         {"steps": agg["steps"]}),
    ]
    return _with_input_sources(
        {vid: View(vid, title, kind, desc, _jsonable(data)) for vid, kind, title, desc, data in specs}, inputs)


def label_comparison_views(
    reference: str | os.PathLike[str] | Mapping[str, Any],
    compared: str | os.PathLike[str] | Mapping[str, Any],
    *,
    reference_name: str = "reference",
    compared_name: str = "compared",
) -> dict[str, View]:
    """Phase counts and time of two labelled runs side by side."""
    ref, ref_inputs = _aggregate_label_source(reference)
    cmp_, cmp_inputs = _aggregate_label_source(compared)
    return _with_input_sources({
        "label_phase_count_comparison": View(
            "label_phase_count_comparison", "Steps by phase, compared", "chart",
            "Classified steps per phase in each run.",
            _jsonable({"reference": {"name": reference_name, "counts": ref["phase_counts"]},
                       "compared": {"name": compared_name, "counts": cmp_["phase_counts"]}})),
        "label_phase_duration_comparison": View(
            "label_phase_duration_comparison", "Time by phase, compared", "chart",
            "Seconds per phase in each run.",
            _jsonable({"reference": {"name": reference_name, "durations_s": ref["phase_durations"]},
                       "compared": {"name": compared_name, "durations_s": cmp_["phase_durations"]}})),
    }, (*ref_inputs, *cmp_inputs))


# -- run groups ------------------------------------------------------------------

def compare_runs(
    sources: Iterable[Trajectory | str | os.PathLike[str]],
    *,
    labels: Iterable[str] | None = None,
    format: str | None = None,
    fuzzy_commands: bool = False,
) -> dict[str, Any]:
    """Score several runs side by side; the first is the baseline.

    *sources* are trajectories (from :func:`load`, :meth:`Trajectory.from_steps`
    or :meth:`Trajectory.from_dict`) or files, read with :func:`load` and
    *format*. Returns ``{"rows", "behavior", "timelines"}``: one scorecard row
    per run (a file that cannot be read gets a row with its ``error``), the behavioural
    comparison (similarity, shared actions, files, tools and skills, patterns
    against the baseline) and :func:`run_group_views`' timeline data. *labels*
    names the runs (default: from their file names). Raises
    :class:`ValueError` for fewer than two runs.
    """
    from .api import FORMATS, Trajectory
    from .insight.run_group import LoadedRun, build_run_group_scorecard, load_run, loaded_run

    if format is not None and format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; expected one of: {', '.join(FORMATS)}")
    runs: list[str | LoadedRun] = [
        loaded_run(src) if isinstance(src, Trajectory) else load_run(os.fspath(src), format) for src in sources
    ]
    result = build_run_group_scorecard(
        runs, labels=list(labels) if labels is not None else None, fuzzy=fuzzy_commands,
    )
    if not result.get("ok"):
        raise ValueError(result.get("error") or "the runs could not be compared")
    return cast("dict[str, Any]", _jsonable({
        "rows": result["rows"],
        "behavior": result["behavior"],
        "timelines": chart_data.run_timelines(result["timeline_runs"]),
    }))


def run_group_views(result: Mapping[str, Any]) -> dict[str, View]:
    """Views of a :func:`compare_runs` result, keyed by id."""
    return {
        "run_scorecard": View(
            "run_scorecard", "Run scorecard", "table",
            "One row per run: steps, tokens, wall clock, tool calls and success, context peak, compactions.",
            _jsonable(result["rows"])),
        "run_behavior": View(
            "run_behavior", "Behavioural comparison", "summary",
            "Pairwise similarity, and which actions, files, tools and skills each run used, with "
            "patterns against the baseline (the first run).",
            _jsonable(result["behavior"])),
        "run_timelines": View(
            "run_timelines", "Agent timeline by run", "chart",
            "Each run's contiguous agent segments; the same kind of agent shares a colour across runs.",
            _jsonable(result["timelines"])),
    }
