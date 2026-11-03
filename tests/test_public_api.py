"""The public API: what ``import agent_trajviz`` promises, and that it keeps the promise.

``ApiSurfaceTests`` pins every public name and signature, so a change to the
API is a deliberate edit of this file rather than a side effect of a refactor.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from datetime import UTC, datetime
from unittest.mock import patch

import agent_trajviz as tv

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CC = FIXTURES / "native_claude_code"
DJANGO = CC / "django__django-11477.jsonl"
ASTROPY = CC / "astropy__astropy-13033.jsonl"


class ApiSurfaceTests(unittest.TestCase):
    def test_the_public_names_are_the_pinned_ones(self):
        self.assertEqual(sorted(tv.__all__), sorted([
            "FORMATS", "SCHEMA_VERSION", "LoadError", "MissingExtraError", "ReportError", "Step",
            "TokenUsage", "ToolCall", "Trajectory", "TrajVizError", "compare", "load", "write_report",
            "VIEWS_SCHEMA_VERSION", "VIEW_CATALOG", "View", "comparison_views", "export_json",
            "compare_runs", "label_comparison_views", "label_views", "run_group_views",
            "Format", "PluginError", "load_plugins", "register_format", "register_view",
            "view_schema", "view_schemas", "list_sessions",
        ]))
        for name in tv.__all__:
            self.assertTrue(hasattr(tv, name), name)

    def test_the_function_signatures_are_the_pinned_ones(self):
        pinned = {
            "load": "(source: 'PathLike', *, format: 'str | None' = None, session_id: 'str | None' = None) -> 'Trajectory'",
            "list_sessions": "(source: 'PathLike', *, format: 'str | None' = None) -> 'list[dict[str, Any]]'",
            "compare": "(reference: 'Trajectory | PathLike', compared: 'Trajectory | PathLike', *, "
                       "fuzzy_commands: 'bool' = False, anchor_patch: 'PathLike | None' = None) "
                       "-> 'dict[str, Any]'",
            "write_report": "(source: 'Trajectory | PathLike', destination: 'PathLike | None' = None, *, "
                            "dark: 'bool' = False, inline_plotly: 'bool' = False) -> 'Path'",
            "export_json": "(source: 'Trajectory | Mapping[str, View]', directory: 'str | os.PathLike[str]', *, "
                           "only: 'Iterable[str] | None' = None, indent: 'int | None' = 2) -> 'list[Path]'",
            "comparison_views": "(report: 'Mapping[str, Any]') -> 'dict[str, View]'",
            "register_format": "(fmt: 'Format', *, replace: 'bool' = False) -> 'None'",
            "register_view": "(view_id: 'str', build: 'Callable[..., Any]', *, title: 'str', "
                             "kind: 'ViewKind' = 'table', description: 'str' = '', "
                             "options: 'Mapping[str, str] | None' = None, "
                             "schema: 'Mapping[str, Any] | None' = None, replace: 'bool' = False) -> 'None'",
            "view_schema": "(view_id: 'str') -> 'dict[str, Any]'",
            "view_schemas": "() -> 'dict[str, dict[str, Any]]'",
            "load_plugins": "() -> 'list[str]'",
        }
        for name, signature in pinned.items():
            self.assertEqual(str(inspect.signature(getattr(tv, name))), signature, name)
        methods = {
            "to_dict": "(self, *, include_raw: 'bool' = False) -> 'dict[str, Any]'",
            "from_dict": "(data: 'Mapping[str, Any]') -> 'Trajectory'",
            "from_steps": "(steps: 'Iterable[Mapping[str, Any]]', *, format: 'str' = 'custom', "
                          "source: 'str | None' = None, sha256: 'str | None' = None, "
                          "warnings: 'Iterable[str]' = (), tokens_estimated: 'bool' = False) -> 'Trajectory'",
        }
        for name, signature in methods.items():
            self.assertEqual(str(inspect.signature(getattr(tv.Trajectory, name))), signature, name)
        self.assertEqual([f.name for f in dataclasses.fields(tv.Format)], ["key", "label", "parse", "detect"])

    def test_the_trajectory_fields_are_the_pinned_ones(self):
        self.assertEqual([f.name for f in dataclasses.fields(tv.Trajectory)],
                         ["format", "steps", "source", "sha256", "warnings", "tokens_estimated", "raw"])
        for member in ("format_label", "metrics", "health", "agents", "context_pressure", "views", "view",
                       "to_dict", "from_dict", "from_steps"):
            self.assertTrue(hasattr(tv.Trajectory, member), member)

    def test_every_error_is_a_trajviz_error(self):
        for error in (tv.LoadError, tv.ReportError, tv.MissingExtraError, tv.PluginError):
            self.assertTrue(issubclass(error, tv.TrajVizError), error)
        self.assertTrue(issubclass(tv.MissingExtraError, ImportError))
        self.assertTrue(issubclass(tv.ReportError, ValueError))

    def test_formats_is_a_read_only_mapping_of_every_supported_format(self):
        self.assertEqual(set(tv.FORMATS), {"claude_code", "cursor", "codeagent", "codearts", "icode", "opencode", "codex", "pi", "dsh", "otlp"})
        with self.assertRaises(TypeError):
            tv.FORMATS["new"] = "New"  # type: ignore[index]


class LoadTests(unittest.TestCase):
    def test_load_returns_a_trajectory_with_provenance(self):
        run = tv.load(DJANGO)
        self.assertEqual((run.format, run.format_label), ("claude_code", "Claude Code"))
        self.assertGreater(len(run.steps), 0)
        self.assertEqual(run.source, str(DJANGO))
        self.assertEqual(run.sha256, hashlib.sha256(DJANGO.read_bytes()).hexdigest())
        self.assertEqual(run.warnings, ())
        self.assertFalse(run.tokens_estimated)

    def test_estimated_token_counts_are_flagged(self):
        self.assertTrue(tv.load(FIXTURES / "cursor_minimal.json").tokens_estimated)

    def test_a_trajectory_is_immutable_and_its_analyses_are_cached(self):
        run = tv.load(DJANGO)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            run.format = "codex"  # type: ignore[misc]
        self.assertIs(run.metrics, run.metrics)
        self.assertGreater(run.metrics["tokens"]["total"], 0)
        self.assertTrue(run.health)
        self.assertTrue(run.agents)

    def test_context_pressure_returns_stats_and_series(self):
        pressure = tv.load(DJANGO).context_pressure()
        self.assertEqual(set(pressure), {"stats", "series"})
        self.assertGreater(pressure["stats"]["peak_occupancy"], 0)

    def test_to_dict_is_json_and_versioned(self):
        doc = json.loads(json.dumps(tv.load(DJANGO).to_dict()))
        self.assertEqual(doc["schema_version"], tv.SCHEMA_VERSION)
        self.assertEqual(doc["format"], "claude_code")
        self.assertTrue(doc["steps"])

    def test_unreadable_and_unknown_files_raise_load_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            not_json = Path(tmp, "a.json")
            not_json.write_text("not json", encoding="utf-8")
            unknown = Path(tmp, "b.json")
            unknown.write_text('{"hello": "world"}', encoding="utf-8")
            for path, code in ((not_json, None), (unknown, "unknown_format"), (Path(tmp, "missing.json"), None)):
                with self.subTest(path=path.name), self.assertRaises(tv.LoadError) as caught:
                    tv.load(path)
                self.assertEqual(caught.exception.path, str(path))
                self.assertTrue(caught.exception.code)
                if code:
                    self.assertEqual(caught.exception.code, code)


    def test_a_format_mismatch_names_the_detected_format(self):
        with self.assertRaises(tv.LoadError) as caught:
            tv.load(FIXTURES / "codearts_minimal.json", format="claude_code")
        self.assertEqual((caught.exception.code, caught.exception.detected), ("mismatch", "codearts"))

    def test_an_undetectable_file_with_steps_is_not_guessed_at(self):
        # OpenCode-shaped messages without the export's markers: readable as
        # OpenCode when asked, but auto-detection must not pass them as "unknown".
        doc = {"messages": [{"info": {"role": "user"}, "parts": [{"type": "text", "text": "hi"}]},
                            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "hello"}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "run.json")
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(tv.LoadError) as caught:
                tv.load(path)
            self.assertEqual(caught.exception.code, "unknown_format")
            self.assertEqual(len(tv.load(path, format="opencode").steps), 2)


class CompareAndReportTests(unittest.TestCase):
    def test_compare_accepts_trajectories_and_paths_and_leaves_steps_alone(self):
        run = tv.load(DJANGO)
        before = copy.deepcopy(run.steps)
        report = tv.compare(run, ASTROPY)
        self.assertIn("alignment", report)
        self.assertIn("confidence", report)
        self.assertEqual(run.steps, before)

    def test_write_report_writes_html_for_a_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = tv.write_report(tv.load(DJANGO), tmp)
            self.assertTrue(path.is_file())
            self.assertIn("<html", path.read_text(encoding="utf-8")[:2000].lower())

    def test_write_report_works_for_a_trajectory_that_was_never_a_file(self):
        made = tv.Trajectory.from_steps([
            {"role": "user", "text_preview": "Fix the test"},
            {"role": "assistant", "tokens": {"input": 10, "output": 5},
             "tool_calls": [{"tool_name": "bash", "input": {"command": "pytest"}, "error": "exit 1"}]},
        ], format="myagent")
        with tempfile.TemporaryDirectory() as tmp:
            path = tv.write_report(made, tmp)
            self.assertEqual(path.name, "trajectory-trajviz-report.html")
            self.assertIn("<html", path.read_text(encoding="utf-8")[:2000].lower())

    def test_write_report_of_a_loaded_run_matches_the_file_report(self):
        run = tv.load(ASTROPY)
        # Generating the two reports may cross a minute boundary on a slow
        # runner. Freeze the header clock so the comparison still checks the
        # complete HTML rather than trying to strip presentation-dependent dates.
        with tempfile.TemporaryDirectory() as tmp, patch("agent_trajviz.insight.report.datetime") as clock:
            clock.now.return_value = datetime(2026, 1, 1, tzinfo=UTC)
            from_run = tv.write_report(run, Path(tmp, "a.html")).read_text(encoding="utf-8")
            from_file = tv.write_report(ASTROPY, Path(tmp, "b.html")).read_text(encoding="utf-8")
        # Plotly gives every chart a random div id.
        strip = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        self.assertEqual(strip.sub("", from_run), strip.sub("", from_file))

    def test_write_report_refuses_an_empty_trajectory(self):
        run = tv.load(DJANGO)
        with self.assertRaises(tv.ReportError):
            tv.write_report(dataclasses.replace(run, steps=[]))


if __name__ == "__main__":
    unittest.main()
