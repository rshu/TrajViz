"""The public API: load a trajectory, analyze it, compare two, export the results.

Everything here needs only the standard library except :func:`write_report`,
which needs the ``charts`` extra. Names not exported from ``agent_trajviz``
(the ``insight`` and ``converge`` subpackages included) are internal and may
change between minor versions.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from . import plugins
from ._extras import require
from .errors import LoadError
from .insight import loaders
from .insight.analysis import Analysis
from .insight.context_usage import context_pressure_series, context_pressure_stats
from .insight.parser import complete_steps, parse_steps
from .model import Step
from .plugins import Format

if TYPE_CHECKING:
    from .views import View

#: Supported formats: format id (as in :attr:`Trajectory.format`) -> display name.
#: Includes the formats registered with :func:`register_format`.
FORMATS: Mapping[str, str] = MappingProxyType(plugins._LABELS)

#: Version of the layout :meth:`Trajectory.to_dict` produces.
SCHEMA_VERSION = 1

#: The ``type`` field that marks a :meth:`Trajectory.to_dict` document.
TRAJECTORY_TYPE = "agent-trajviz-trajectory"

PathLike = str | os.PathLike[str]

logger = logging.getLogger(__name__)


def _plain(steps: list[Step]) -> list[dict[str, Any]]:
    """The same list, typed as the internal modules see it (steps are plain dicts)."""
    return cast("list[dict[str, Any]]", steps)


def _exported_steps(value: list[Any]) -> list[Step]:
    """Validate normalized record containers without rewriting source evidence."""
    import math

    for position, step in enumerate(value):
        prefix = f"invalid trajectory step {position}"
        if not isinstance(step, dict):
            raise ValueError(f"{prefix}: expected an object")
        for key, kind in (("tokens", dict), ("parts", list), ("tool_calls", list)):
            if not isinstance(step.get(key), kind):
                raise ValueError(f"{prefix}: {key} must be a {kind.__name__}")
        for key in ("tool_call_count", "error_count"):
            if type(step.get(key)) is not int or step[key] < 0:
                raise ValueError(f"{prefix}: {key} must be a nonnegative integer")
        if type(step.get("index")) is not int:
            raise ValueError(f"{prefix}: index must be an integer")
        if "raw_index" in step and type(step["raw_index"]) is not int:
            raise ValueError(f"{prefix}: raw_index must be an integer")
        if not isinstance(step.get("role"), str):
            raise ValueError(f"{prefix}: role must be a string")
        for key in ("total", "input", "output", "cache_read", "cache_write"):
            number = step["tokens"].get(key)
            try:
                finite = not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{prefix}: tokens.{key} must be a finite number")
        for part in step["parts"]:
            if not isinstance(part, dict):
                raise ValueError(f"{prefix}: parts must contain objects")
            for key in ("type", "text"):
                if key in part and not isinstance(part[key], str):
                    raise ValueError(f"{prefix}: part {key} must be a string")
        for call in step["tool_calls"]:
            if not isinstance(call, dict) or not isinstance(call.get("tool_name"), str):
                raise ValueError(f"{prefix}: tool_calls must contain named tool objects")
            if not isinstance(call.get("status"), str):
                raise ValueError(f"{prefix}: tool status must be a string")
            if "metadata" in call and not isinstance(call["metadata"], dict):
                raise ValueError(f"{prefix}: tool metadata must be an object")
    return cast("list[Step]", value)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """One loaded agent run: its detected format and its normalized steps.

    Create one with :func:`load`. Treat ``steps`` as read-only: :attr:`metrics`,
    :attr:`health` and :attr:`agents` are computed from them once and cached.
    """

    #: Format id, a key of :data:`FORMATS` (for example ``"codex"``).
    format: str
    steps: list[Step]
    #: The file it was loaded from.
    source: str | None = None
    #: SHA-256 source identity (OTLP also includes session and mapping profile).
    sha256: str | None = None
    #: What the loader had to skip or could not read, in plain words.
    warnings: tuple[str, ...] = ()
    #: True when the export recorded no token usage and the counts are estimates.
    tokens_estimated: bool = False
    #: The converted export the steps came from (format-specific; for advanced use).
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def format_label(self) -> str:
        """Display name of :attr:`format`, for example ``"Codex CLI"``."""
        return FORMATS.get(self.format, self.format)

    @cached_property
    def _analysis(self) -> Analysis:
        # The one orchestration of every analysis; views and properties share it.
        return Analysis(_plain(self.steps), self.raw, self.format)

    @property
    def metrics(self) -> dict[str, Any]:
        """Session metrics: tokens, timing, tool use, finish reasons, throughput."""
        return self._analysis.metrics

    @property
    def health(self) -> list[dict[str, Any]]:
        """Health verdicts, one per check, each with a status and an explanation."""
        return self._analysis.verdicts

    @property
    def agents(self) -> list[dict[str, Any]]:
        """Per-agent summary rows (the main agent and each sub-agent)."""
        return self._analysis.agent_summaries

    def context_pressure(self, window_limit: int | None = None) -> dict[str, Any]:
        """Context-window occupancy per agent and its summary statistics.

        Returns ``{"stats": ..., "series": ...}``. *window_limit* overrides the
        window the export declares or the model implies; without either, the
        stats say the window is assumed or unknown rather than guess silently.
        """
        from .insight.telemetry_quality import supports

        if self.format == "otlp" and not supports(_plain(self.steps), "context"):
            series = context_pressure_series([], window_limit=window_limit)
            stats = context_pressure_stats(series)
            stats = {key: None if isinstance(value, (int, float)) else value for key, value in stats.items()}
            return {"stats": stats, "series": series,
                    "availability": {"status": "unavailable", "reason": "context occupancy is not recorded"}}
        series = context_pressure_series(_plain(self.steps), raw=self.raw, window_limit=window_limit)
        return {"stats": context_pressure_stats(series), "series": series}

    def views(self, *, only: Iterable[str] | None = None) -> dict[str, View]:
        """Every chart and table as a JSON-serializable :class:`View`, keyed by id.

        *only* restricts the result to those ids (see :data:`VIEW_CATALOG`);
        an unknown id raises :class:`KeyError`. Results are computed once per
        trajectory and shared between views.
        """
        from .views import build_views

        return build_views(self, self._analysis, only)

    def view(self, view_id: str, **options: Any) -> View:
        """One view by id (see :data:`VIEW_CATALOG`).

        Some views take options, listed under ``"options"`` in the catalog --
        for example ``run.view("context_pressure", agent=key, window_limit=200_000)``.
        """
        from .views import build_view

        return build_view(self, self._analysis, view_id, options)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """A JSON-serializable copy: format, provenance, warnings and steps.

        ``include_raw=True`` adds the converted export (:attr:`raw`), which
        some analyses read (session metadata, timing, sub-agent links); the
        result then reloads with :meth:`from_dict` or :func:`load` to
        identical views.
        """
        from .views import _jsonable

        data: dict[str, Any] = {
            "type": TRAJECTORY_TYPE,
            "schema_version": SCHEMA_VERSION,
            "format": self.format,
            "format_label": self.format_label,
            "source": self.source,
            "sha256": self.sha256,
            "warnings": list(self.warnings),
            "tokens_estimated": self.tokens_estimated,
            "steps": self.steps,
        }
        if include_raw:
            data["raw"] = {key: value for key, value in self.raw.items() if key != "_input_files"}
        if isinstance(self.raw.get("_otel"), dict):
            # Retain evidence and selection metadata even in lightweight
            # exports, without copying raw span/log bodies or other sessions.
            data["telemetry"] = {
                key: value for key, value in self.raw["_otel"].items()
                if key not in {"spans", "logs"}
            }
            data["telemetry"]["session_metadata"] = self.raw.get("metadata", {})
            data["telemetry"]["timing"] = self.raw.get("timing", {})
        return cast("dict[str, Any]", _jsonable(data))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Rebuild a trajectory from :meth:`to_dict` output (parsed JSON).

        Raises :class:`ValueError` for a document of another layout or
        version, or malformed normalized records. For partial custom records,
        use :meth:`from_steps`. Without ``raw`` (``to_dict()`` without ``include_raw``) the
        steps are restored; OTLP exports also retain their selection, timing
        and evidence metadata. Other formats may lose analyses that read raw
        export metadata.
        """
        if not isinstance(data, Mapping) or not isinstance(data.get("steps"), list):
            raise ValueError("not an agent-trajviz trajectory: no steps")
        version = data.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema_version {version!r} (expected {SCHEMA_VERSION})")
        fmt = data.get("format")
        if not isinstance(fmt, str) or not fmt:
            raise ValueError("not an agent-trajviz trajectory: no format")
        steps = _exported_steps(data["steps"])
        warnings = data.get("warnings")
        if warnings is not None and not isinstance(warnings, (list, tuple)):
            raise ValueError("not an agent-trajviz trajectory: warnings must be an array")
        raw = data.get("raw")
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        telemetry = data.get("telemetry")
        if fmt == "otlp" and isinstance(telemetry, Mapping):
            raw.setdefault("_otel", {key: value for key, value in telemetry.items()
                                     if key not in {"session_metadata", "timing"}})
            raw.setdefault("metadata", telemetry.get("session_metadata", {}))
            raw.setdefault("timing", telemetry.get("timing", {}))
            raw.setdefault("_otlp_sub_agents", telemetry.get("agents", []))
            raw.setdefault("_source_format", "otlp")
            raw.setdefault("_otlp_format", True)
            raw.setdefault("_source_sha256", data.get("sha256"))
            raw.setdefault("_source_capture_sha256", telemetry.get("capture_sha256"))
        if fmt == "ccsession":
            # Schema-v1 public exports already contain normalized steps. Keep
            # those saved documents readable after replacing the old source
            # converter, without accepting its raw session-export wrapper.
            fmt = "claude_code"
            for old, new in (("_cc_format", "_claude_code_format"),
                             ("_cc_parsed_steps", "_claude_code_parsed_steps"),
                             ("_cc_sub_agents", "_claude_code_sub_agents")):
                if old in raw:
                    raw.setdefault(new, raw.pop(old))
        return cls(
            format=fmt,
            steps=list(steps),
            source=data.get("source"),
            sha256=data.get("sha256"),
            warnings=tuple(str(w) for w in warnings or ()),
            tokens_estimated=bool(data.get("tokens_estimated")),
            raw=raw,
        )

    @classmethod
    def from_steps(
        cls,
        steps: Iterable[Mapping[str, Any]],
        *,
        format: str = "custom",
        source: str | None = None,
        sha256: str | None = None,
        warnings: Iterable[str] = (),
        tokens_estimated: bool = False,
    ) -> Trajectory:
        """A trajectory from steps you parsed yourself, for an agent TrajViz does not read.

        Each step is a mapping with keys of :class:`Step`; only ``role``
        (``"user"`` or ``"assistant"``) matters for most views. Tool calls are
        mappings with keys of :class:`ToolCall` (``tool_name``, ``input``,
        ``output``, and ``error`` or ``status`` for a failure); token counts are
        a :class:`TokenUsage`. A key a step lacks gets the value a format that
        does not record it gets, so every view works on the result; ``total``
        tokens default to the sum of the reported fields, and a step's
        ``duration`` to the span between ``time_created_ms`` and
        ``time_completed_ms`` (epoch milliseconds). A sub-agent's steps name it
        in ``agent``: leave ``agent`` empty on the main agent's steps, or set
        ``is_sub_agent`` on every step. The steps are copied, never modified.

        Raises :class:`ValueError` when there are no steps and
        :class:`TypeError` for a step that is not a mapping.
        """
        completed = complete_steps(steps)
        if not completed:
            raise ValueError("no steps")
        return cls(
            format=format,
            steps=cast("list[Step]", completed),
            source=source,
            sha256=sha256,
            warnings=tuple(str(w) for w in warnings),
            tokens_estimated=tokens_estimated,
        )


def register_format(fmt: Format, *, replace: bool = False) -> None:
    """Make :func:`load` read a format of your own (see :class:`Format`).

    The format joins :data:`FORMATS`. Raises :class:`ValueError` for a key
    that is malformed or built in, or already registered (pass
    ``replace=True`` to replace your own).
    """
    plugins.register_format(fmt, replace=replace)


def load_plugins() -> list[str]:
    """Run the hook of every installed plugin once; returns the names loaded so far.

    A plugin is an entry point in the ``agent_trajviz.plugins`` group that
    names a function taking no arguments, which typically calls
    :func:`register_format` or :func:`register_view`. The ``agent-trajviz``
    command runs this on start. Raises :class:`PluginError` naming a plugin
    whose hook fails.
    """
    return plugins.load_plugins()


def load(source: PathLike, *, format: str | None = None, session_id: str | None = None) -> Trajectory:
    """Load a trajectory source of any supported format (see :data:`FORMATS`).

    The format is detected from the content. *format* (a key of
    :data:`FORMATS`) requires that format instead: a file detected as another
    format raises :class:`LoadError` with code ``"mismatch"``, and a file
    whose format cannot be detected is read as *format*. Raises
    :class:`LoadError` when the file cannot be read, is not a supported
    format, or holds no steps (see its ``code``); :class:`ValueError` for an
    unknown *format*. Formats added with :func:`register_format` are detected
    after the built-in ones, and a :meth:`Trajectory.to_dict` document (such
    as the ``trajectory.json`` of :func:`export_json`) reloads as it was.

    Saved OTLP sources can be a JSON/JSONL file, an explicit bundle manifest,
    its directory, or a ZIP bundle. Use :func:`list_sessions` to discover
    selectable ids, then pass *session_id* to load one; a capture containing
    multiple sessions raises ``LoadError(code="ambiguous_session")`` without
    a selection. Native JSONL and its subagents keep their existing behavior.
    """
    if format is not None and format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; expected one of: {', '.join(FORMATS)}")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise ValueError("session_id must be a non-empty string")
    path = os.fspath(source)
    from .insight.formats.otlp_files import looks_like_otlp_source

    otlp_source = looks_like_otlp_source(path)
    directory = Path(path).is_dir()
    if otlp_source or format == "otlp" and directory:
        if format is not None and format != "otlp":
            raise LoadError(f"the file is otlp, not {format}", code="mismatch", path=path, detected="otlp")
        return _load_otlp(path, session_id=session_id)
    exported = _read_exported(path, bounded=format == "otlp" or session_id is not None)
    if exported is not None:
        run = _load_exported(exported, path, format)
        if session_id is not None and run.raw.get("_otel", {}).get("session_id") != session_id:
            raise LoadError(f"the exported trajectory does not contain session {session_id!r}",
                            code="session_not_found", path=path)
        return run
    if (format == "otlp" or session_id is not None) and not directory:
        native_format = loaders._bounded_non_otlp_format(path)
        if native_format == "otlp":
            # A large envelope can put its discriminator beyond the cheap
            # sniff prefix. Selection must still use the bounded OTLP reader,
            # rather than reject a capture that loads without a selector.
            if format is not None and format != "otlp":
                raise LoadError(f"the file is otlp, not {format}", code="mismatch", path=path, detected="otlp")
            return _load_otlp(path, session_id=session_id)
        if format == "otlp" and native_format in FORMATS:
            raise LoadError(f"the file is {native_format}, not otlp", code="mismatch", path=path,
                            detected=native_format)
    if format == "otlp" or otlp_source:
        if format is not None and format != "otlp":
            raise LoadError(f"the file is otlp, not {format}", code="mismatch", path=path, detected="otlp")
        return _load_otlp(path, session_id=session_id)
    if session_id is not None:
        raise LoadError("session_id selection applies to saved OTLP captures", code="unsupported_selection", path=path)
    registered = plugins.registered_format(format)
    if registered is not None:
        return _load_registered(registered, path)
    raw = loaders.load_trajectory(path, format_hint=format)
    if raw.get("_error"):
        if format is None and raw.get("_error_code") == "unknown_format":
            detected = _detect_registered(path)
            if detected is not None:
                return detected
        raise LoadError(str(raw["_error"]), code=raw.get("_error_code") or "load_failed", path=path,
                        detected=raw.get("_detected"))
    if raw.get("type") == TRAJECTORY_TYPE:
        # An export written with sorted keys: no format claims it, so the
        # loader hands the document back as parsed.
        return _load_exported(raw, path, format)
    if format is not None and loaders.check_format_selection(loaders.detect_format(raw), format) == "mismatch":
        detected_format = loaders.detect_format(raw)
        raise LoadError(f"the file is {detected_format}, not {format}", code="mismatch", path=path,
                        detected=detected_format)
    fmt = raw.get("_source_format") or loaders.detect_format(raw)
    if format is None and fmt == "unknown":
        detected = _detect_registered(path)
        if detected is not None:
            return detected
        raise LoadError(
            "could not detect the trajectory format; name the format to read it as (one of: "
            + ", ".join(FORMATS) + ")", code="unknown_format", path=path,
        )
    steps = parse_steps(raw)
    if not steps:
        raise LoadError(f"no trajectory steps found (detected format: {fmt})", code="no_steps", path=path)
    return Trajectory(
        format=fmt,
        steps=cast("list[Step]", steps),
        source=path,
        sha256=raw.get("_source_sha256"),
        warnings=tuple(str(w) for w in raw.get("_load_warnings") or ()),
        tokens_estimated=loaders.tokens_are_estimates(raw),
        raw=raw,
    )


def list_sessions(source: PathLike, *, format: str | None = None) -> list[dict[str, Any]]:
    """List selectable Claude Code sessions in a saved OTLP capture.

    Each row includes an ``id`` accepted by :func:`load`'s ``session_id``.
    Native session JSONL loading and subagent discovery keep their existing
    behavior; this discovery API currently supports only OTLP captures.
    """
    if format not in (None, "otlp"):
        raise ValueError("session discovery currently supports format='otlp'")
    from .insight.formats.otlp_claude import list_otlp_sessions
    from .insight.formats.otlp_files import read_otlp_capture

    return list_otlp_sessions(read_otlp_capture(os.fspath(source)))


def _load_otlp(path: str, *, session_id: str | None = None) -> Trajectory:
    from .insight.formats.otlp_claude import convert_otlp_claude
    from .insight.formats.otlp_files import read_otlp_capture

    capture = read_otlp_capture(path)
    raw = convert_otlp_claude(capture, session_id=session_id)
    loaders._stamp_otlp_identity(raw)
    raw["_source_path"] = path
    steps = parse_steps(raw)
    if not steps:
        raise LoadError("no supported Claude Code operations found in the OTLP capture", code="no_steps", path=path)
    return Trajectory(
        format="otlp", steps=cast("list[Step]", steps), source=path,
        sha256=raw["_source_sha256"], warnings=tuple(str(w) for w in raw.get("_load_warnings") or ()),
        tokens_estimated=False, raw=raw,
    )


def _load_registered(fmt: Format, path: str, payload: Any = None, sha256: str | None = None) -> Trajectory:
    """*path* parsed by the registered format *fmt* (*payload*: the file, already read)."""
    if payload is None:
        payload, sha256, error = loaders.read_payload(path)
        if error is not None:
            raise LoadError(error, code="load_failed", path=path)
    try:
        steps = list(fmt.parse(payload))
    except Exception as exc:
        raise LoadError(f"the {fmt.label} parser failed: {type(exc).__name__}: {exc}",
                        code="parse_failed", path=path) from exc
    if not steps:
        raise LoadError(f"no trajectory steps found (format: {fmt.key})", code="no_steps", path=path)
    try:
        run = Trajectory.from_steps(steps, format=fmt.key, source=path, sha256=sha256)
        run.raw["_input_files"] = [str(Path(path).resolve())]
        return run
    except TypeError as exc:
        raise LoadError(f"the {fmt.label} parser returned {exc}", code="parse_failed", path=path) from exc


def _detect_registered(path: str) -> Trajectory | None:
    """*path* read by the first registered format whose ``detect`` accepts it, else ``None``.

    Runs only for files no built-in format recognizes. A ``detect`` that
    raises is a non-match (logged): one plugin's bug must not stop a file of
    another format from loading.
    """
    candidates = [(f, f.detect) for f in plugins.registered_formats() if f.detect is not None]
    if not candidates:
        return None
    payload, sha256, error = loaders.read_payload(path)
    if error is not None:
        return None
    for fmt, detect in candidates:
        try:
            matched = bool(detect(payload))
        except Exception:
            logger.warning("format %r: detect() raised on %s; treated as no match", fmt.key, path, exc_info=True)
            continue
        if matched:
            return _load_registered(fmt, path, payload, sha256)
    return None


def _load_exported(data: Mapping[str, Any], path: str, format: str | None) -> Trajectory:
    """A :meth:`Trajectory.to_dict` document read back from *path*."""
    try:
        run = Trajectory.from_dict(data)
    except ValueError as exc:
        raise LoadError(str(exc), code="unsupported_schema", path=path) from exc
    if format is not None and run.format != format:
        raise LoadError(f"the file is {run.format}, not {format}", code="mismatch", path=path, detected=run.format)
    if not run.steps:
        raise LoadError("no trajectory steps found in the exported trajectory", code="no_steps", path=path)
    run.raw["_input_files"] = [str(Path(path).resolve())]
    return run


def _read_exported(path: str, *, bounded: bool = False) -> dict[str, Any] | None:
    """The parsed document when *path* is a :meth:`Trajectory.to_dict` export, else ``None``.

    Only a JSON object whose ``type`` is :data:`TRAJECTORY_TYPE` qualifies; a
    cheap prefix check keeps every other file on the format loaders' path.
    """
    import json
    from .insight.formats.otlp_files import _open_file

    try:
        handle = _open_file(Path(path))
    except LoadError:
        return None

    try:
        with handle as fh:
            if bounded:
                from .insight.formats.otlp_files import MAX_MEMBER_BYTES

                data_bytes = fh.read(MAX_MEMBER_BYTES + 1)
                if len(data_bytes) > MAX_MEMBER_BYTES:
                    raise LoadError("OTLP capture size limit exceeded", code="size_limit", path=path)
                data = json.loads(data_bytes)
                return data if isinstance(data, dict) and data.get("type") == TRAJECTORY_TYPE else None
            head = fh.read(4096)
            if TRAJECTORY_TYPE.encode() not in head:
                return None
            fh.seek(0)
            data = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) and data.get("type") == TRAJECTORY_TYPE else None


def compare(reference: Trajectory | PathLike, compared: Trajectory | PathLike, *,
            fuzzy_commands: bool = False, anchor_patch: PathLike | None = None) -> dict[str, Any]:
    """Align two runs of the same task and report where they diverge.

    Accepts loaded trajectories or file paths. The report includes the step
    alignment, divergence points, outcome and a confidence label; see the
    ``agent-trajviz-converge`` command for the same report from the shell.
    *anchor_patch* is a unified diff of the intended change (for example the
    task's reference patch): the comparison is then grounded on the files it
    touches, and the report gains an anchor analysis. A patch with no
    recognizable ``a/``/``b/`` headers leaves the comparison un-anchored.
    """
    from .converge.alignment import _parse_anchor_files, build_comparison_report_from_steps

    anchor_files = None
    if anchor_patch is not None:
        patch = os.fspath(anchor_patch)
        if not os.path.isfile(patch):
            raise FileNotFoundError(f"anchor patch not found: {patch}")
        anchor_files = _parse_anchor_files(patch)
    ref = reference if isinstance(reference, Trajectory) else load(reference)
    cmp = compared if isinstance(compared, Trajectory) else load(compared)
    return build_comparison_report_from_steps(
        ref.raw, cmp.raw, _plain(ref.steps), _plain(cmp.steps),
        fuzzy_commands=fuzzy_commands, anchor_files=anchor_files,
        ref_path=ref.source or "", cmp_path=cmp.source or "",
    )


def write_report(source: Trajectory | PathLike, destination: PathLike | None = None, *,
                 dark: bool = False, inline_plotly: bool = False) -> Path:
    """Write the standalone HTML report for a trajectory and return its path.

    *source* is a loaded trajectory (from a file or not, such as one from
    :meth:`Trajectory.from_steps`) or a file to load. *destination* is a file
    or a directory (default: a temporary file). ``inline_plotly`` embeds
    plotly.js so the report works offline. Needs the ``charts`` extra; raises
    :class:`ReportError` if the trajectory cannot be read or has no steps.
    """
    require("charts", "write_report()")
    from .insight.report import report_from_path, report_from_run

    dest = os.fspath(destination) if destination is not None else None
    if isinstance(source, Trajectory):
        return Path(report_from_run(source, dest, dark=dark, inline_plotly=inline_plotly))
    return Path(report_from_path(os.fspath(source), dest, dark=dark, inline_plotly=inline_plotly))
