"""The built-in dashboard reads and draws through the public API.

It loads files with `agent_trajviz.load` (so formats registered by an
application or a plugin open, and so do exported `trajectory.json` files),
keeps the loaded `Trajectory` in session state for the handlers that need the
run itself, draws its charts from the public views, and shows registered
views in a tab of their own.
"""

from __future__ import annotations

import contextlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_trajviz as tv
from agent_trajviz import plugins
from agent_trajviz import views as views_mod

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CC_RUN = FIXTURES / "native_claude_code" / "django__django-11477.jsonl"
T0 = 1_700_000_000_000


def _toy_steps(doc):
    steps = []
    for i, turn in enumerate(doc["turns"]):
        step = {"role": turn["who"], "text_preview": turn["say"], "time_created_ms": T0 + 1000 * i,
                "time_completed_ms": T0 + 1000 * i + 600, "tokens": {"input": 100 * (i + 1), "output": 20}}
        if turn.get("tool"):
            step["tool_calls"] = [{"tool_name": turn["tool"], "input": {"command": "pytest"},
                                   "output": "1 failed", "error": "exit 1" if turn.get("failed") else None}]
        steps.append(step)
    return steps


TOY = tv.Format(key="toy", label="Toy Agent", parse=_toy_steps,
                detect=lambda doc: isinstance(doc, dict) and doc.get("agent") == "toy")
TOY_DOC = {"agent": "toy", "turns": [
    {"who": "user", "say": "make the test pass"},
    {"who": "assistant", "say": "running tests", "tool": "bash", "failed": True},
    {"who": "assistant", "say": "fixed", "tool": "bash"},
]}


def _tool_count(run):
    return {"calls": sum(len(s["tool_calls"]) for s in run.steps)}


@contextlib.contextmanager
def _registered():
    """TOY and a view of our own registered for the duration; every registry restored after."""
    saved = (dict(plugins._CUSTOM), dict(plugins._LABELS), dict(views_mod._SPECS), dict(views_mod._CATALOG))
    try:
        tv.register_format(TOY)
        tv.register_view("tool_count", _tool_count, title="Tool calls", kind="summary",
                         description="Tool calls in the run.")
        yield
    finally:
        for live, old in zip((plugins._CUSTOM, plugins._LABELS, views_mod._SPECS, views_mod._CATALOG), saved,
                             strict=True):
            live.clear()
            live.update(old)


class LoadPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_a_registered_format_opens_in_the_dashboard(self):
        from agent_trajviz.insight.presenters.overview import build_chart_outputs
        from agent_trajviz.insight.session import LoadedSession, load_session
        from agent_trajviz.insight.ui.load import pack_load_outputs

        path = self.tmp / "run.json"
        path.write_text(json.dumps(TOY_DOC), encoding="utf-8")
        with _registered():
            session = load_session(str(path))
            self.assertIsInstance(session, LoadedSession)
            self.assertEqual((session.format, session.run.format_label), ("toy", "Toy Agent"))
            self.assertIs(session.shown, session.run)
            charts = build_chart_outputs(session)
            self.assertEqual(len(charts["tl_fig"].data[0].x), 1)  # one tool, drawn from the tool_usage view
            packed = pack_load_outputs(session)
        self.assertIs(packed["state_run"], session.run)
        self.assertIn("2 assistant, 1 user", packed["overview_kpi_html"]["value"])
        self.assertFalse(packed["plan_timeline_chart"]["visible"])

    def test_plan_chart_appears_for_recorded_todos_and_hides_on_clear(self):
        from agent_trajviz.insight.session import session_from_run
        from agent_trajviz.insight.ui.load import empty_load_outputs, pack_load_outputs

        run = tv.Trajectory.from_steps([{
            "role": "assistant",
            "tool_calls": [{
                "tool_name": "TodoWrite",
                "input": {"todos": [{"content": "Run tests", "status": "in_progress"}]},
            }],
        }])
        chart = pack_load_outputs(session_from_run(run))["plan_timeline_chart"]
        self.assertTrue(chart["visible"])
        self.assertTrue(chart["value"].data)
        self.assertFalse(empty_load_outputs()["plan_timeline_chart"]["visible"])

    def test_an_exported_trajectory_opens_with_the_same_numbers(self):
        from agent_trajviz.insight.session import load_session

        run = tv.load(CC_RUN)
        tv.export_json(run, self.tmp, only=["summary"])
        original, exported = load_session(str(CC_RUN)), load_session(str(self.tmp / "trajectory.json"))
        self.assertEqual(exported.format, original.format)
        self.assertEqual(exported.metrics, original.metrics)
        self.assertEqual(exported.verdicts, original.verdicts)

    def test_the_dashboard_draws_the_first_steps_and_keeps_the_whole_run(self):
        from agent_trajviz.insight.session import load_session

        with mock.patch("agent_trajviz.insight.session.MAX_STEPS", 5):
            session = load_session(str(CC_RUN))
        self.assertTrue(session.truncated)
        self.assertEqual(len(session.shown.steps), 5)
        self.assertEqual(len(session.run.steps), session.steps_total)
        self.assertEqual(len(session.shown.view("token_usage").data["steps"]), 5)

    def test_undetectable_and_mismatched_files_keep_their_messages(self):
        from agent_trajviz.insight.session import LoadError, load_session

        unknown = self.tmp / "unknown.json"
        unknown.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        result = load_session(str(unknown))
        self.assertIsInstance(result, LoadError)
        self.assertEqual(result.code, "unknown")
        self.assertIn("Select a format from the dropdown", result.message)
        result = load_session(str(FIXTURES / "codearts_minimal.json"), format_hint="codex")
        self.assertEqual((result.code, result.selected, result.detected), ("mismatch", "codex", "codearts"))


class HandlerTests(unittest.TestCase):
    """Handlers that need the run read the Trajectory from state, so any trajectory works."""

    def setUp(self):
        self.run = tv.Trajectory.from_steps(_toy_steps(TOY_DOC), format="toy")

    def test_the_html_export_takes_the_loaded_trajectory(self):
        from agent_trajviz.insight.ui.upload import prepare_html_export

        update = prepare_html_export(self.run, None, False)
        self.assertTrue(update["interactive"])
        self.assertTrue(Path(update["value"]).is_file())
        self.assertFalse(prepare_html_export(None, None, False)["interactive"])

    def test_the_run_group_takes_the_loaded_trajectory_as_baseline(self):
        from agent_trajviz.insight.run_group import build_run_group_scorecard, load_run, loaded_run

        with _registered(), tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "other.json"
            other.write_text(json.dumps(TOY_DOC), encoding="utf-8")
            result = build_run_group_scorecard([load_run(str(other))], baseline_raw=loaded_run(self.run))
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual([r["format"] for r in result["rows"]], ["toy", "toy"])
        self.assertEqual([bool(r["error"]) for r in result["rows"]], [False, False])

    def test_the_pairwise_comparison_takes_parsed_steps(self):
        from agent_trajviz.insight.comparison import run_comparison

        reference = tv.Trajectory.from_steps(_toy_steps(TOY_DOC)[:2], format="toy")
        result = run_comparison(ref_raw=reference.raw, cmp_raw=self.run.raw, ref_steps=reference.steps,
                                cmp_steps=self.run.steps, ref_path="reference.json", cmp_path="compared.json")
        self.assertTrue(result["ok"])
        self.assertIn("reference.json", result["report_html"])


class ViewsTabTests(unittest.TestCase):
    def test_no_tab_without_registered_views(self):
        from agent_trajviz.insight.insight import build_ui

        labels = {c.get("props", {}).get("label") for c in build_ui().config["components"]}
        self.assertNotIn("Views", labels)

    def test_registered_views_get_a_tab_and_their_data(self):
        import plotly.graph_objects as go

        from agent_trajviz import figures
        from agent_trajviz.insight.insight import build_ui
        from agent_trajviz.insight.ui.views_tab import fill_views

        saved_draw = dict(figures._DRAW)
        self.addCleanup(lambda: (figures._DRAW.clear(), figures._DRAW.update(saved_draw)))
        with _registered():
            tv.register_view("broken", lambda run: 1 / 0, title="Broken")
            tv.register_view("call_chart", _tool_count, title="Calls", kind="chart")
            figures.register_figure("call_chart", lambda data, dark: go.Figure(go.Bar(y=[data["calls"]])))
            labels = {c.get("props", {}).get("label") for c in build_ui().config["components"]}
            self.assertIn("Views", labels)
            run = tv.Trajectory.from_steps(_toy_steps(TOY_DOC), format="toy")
            count, broken, chart = fill_views(run)
            self.assertEqual(count, {"calls": 2})
            self.assertEqual(broken, {"error": "ZeroDivisionError (see the server log)"})
            self.assertEqual(list(chart.data[0].y), [2])
            self.assertEqual(fill_views(None), [None, None, None])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveDashboardTests(unittest.TestCase):
    """The real dashboard over HTTP: a registered format loads, a registered view is served."""

    def test_load_a_registered_format_and_read_its_view(self):
        try:
            from gradio_client import Client, handle_file
        except ImportError:  # pragma: no cover - gradio ships it
            self.skipTest("gradio_client unavailable")
        from agent_trajviz.dashboard import insight_app

        with _registered(), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(TOY_DOC), encoding="utf-8")
            app = insight_app()
            app.launch(server_name="127.0.0.1", server_port=_free_port(), prevent_thread_lock=True, quiet=True,
                       allowed_paths=[tmp])
            try:
                client = Client(app.local_url, verbose=False)
                outputs = client.predict(handle_file(str(path)), "", api_name="/_do_load")
                rendered = json.dumps(outputs, default=str)
                self.assertIn("2 assistant, 1 user", rendered)  # the KPI strip of the toy run
                self.assertNotIn("Error loading", rendered)
                self.assertEqual(client.predict(api_name="/on_load"), {"calls": 2})
            finally:
                app.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
