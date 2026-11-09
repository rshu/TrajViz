"""Native Cursor mirrors must not turn absent evidence into measurements."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema

import agent_trajviz as tv
from agent_trajviz.insight.formatting import format_performance_md
from agent_trajviz.insight.metrics import compute_metrics, wall_clock_fmt
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from agent_trajviz.insight.presenters.overview import build_overview_kpi_html
from agent_trajviz.insight.session import session_from_run

FIXTURE = Path(__file__).parent / "fixtures/cursor_minimal.json"


def capture(*, timed=False, status="unknown", output=""):
    raw = json.loads(FIXTURE.read_text())
    if not timed:
        raw["info"].pop("time", None)
    user = copy.deepcopy(raw["messages"][0])
    user["info"].pop("time", None)
    messages = [user]
    for index in range(3):
        message = copy.deepcopy(raw["messages"][1])
        message["info"].update(id=f"assistant-{index}")
        message["info"].pop("time", None)
        message["parts"] = [{
            "type": "tool", "tool": "Grep", "callID": f"search-{index}",
            "state": {"status": status, "input": {"pattern": "needle"}, "output": output},
        }]
        messages.append(message)
    raw["messages"] = messages
    return raw


def load(raw):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cursor.json"
        path.write_text(json.dumps(raw))
        return tv.load(path)


class CursorEvidenceTests(unittest.TestCase):
    def test_unrecorded_search_results_are_not_fruitless_or_repeated_misses(self):
        run = load(capture())
        self.assertFalse(run.raw["_capabilities"]["has_complete_tool_outcomes"])
        self.assertEqual(run.view("fruitless_streaks").data, [])
        self.assertEqual(run.view("repeated_searches").data, [])
        session = session_from_run(run)
        self.assertEqual(session.fruitless_streaks, [])
        html = build_overview_issues_html(session)
        self.assertIn("证据有限", html)
        self.assertNotIn("未检测到明显的工作流问题", html)
        self.assertNotIn("Fruitless search streak", html)

    def test_confirmed_empty_results_remain_an_observed_fruitless_streak(self):
        run = load(capture(status="completed", output="No matches found"))
        streaks = run.view("fruitless_streaks").data
        self.assertEqual(len(streaks), 1)
        self.assertEqual(streaks[0]["length"], 3)

    def test_explicitly_missing_native_clocks_are_nullable_everywhere(self):
        run = load(capture())
        for key in ("wall_clock", "total_duration", "avg_duration", "median_duration", "p95_duration",
                    "max_duration", "tokens_per_second", "median_tokens_per_second"):
            with self.subTest(key=key):
                self.assertIsNone(run.metrics[key])
        self.assertTrue(run.agents)
        self.assertTrue(all(agent["total_duration_s"] is None for agent in run.agents))
        session = session_from_run(run)
        self.assertEqual(session.wall_clock, "n/a")
        kpis = build_overview_kpi_html(run.metrics, session.wall_clock, message_rows=session.message_rows)
        self.assertIn("P95 N/A", kpis)
        self.assertNotIn("P95 0s", kpis)
        self.assertNotIn("P95 Nones", kpis)
        details = format_performance_md(run.metrics, session.wall_clock)
        self.assertNotIn("Nones", details)
        self.assertNotIn("ov-kpi-sparkline", kpis[kpis.index("Wall-Clock"):kpis.index("Tokens")])
        for view_id, view in run.views().items():
            with self.subTest(view=view_id):
                jsonschema.Draft202012Validator(tv.view_schema(view_id)).validate(json.loads(view.to_json()))
        restored = tv.Trajectory.from_dict(run.to_dict(include_raw=True))
        self.assertEqual(restored.metrics, run.metrics)
        self.assertEqual({name: view.to_json() for name, view in restored.views().items()},
                         {name: view.to_json() for name, view in run.views().items()})
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            tv.write_report(run, report)
            html = report.read_text()
        self.assertIn("P95 N/A", html)
        self.assertNotIn("Nones", html)
        self.assertNotIn("Fruitless search streaks (", html)

    def test_session_clock_survives_without_per_message_clocks(self):
        run = load(capture(timed=True))
        self.assertEqual(run.metrics["wall_clock"], 4)
        self.assertEqual(wall_clock_fmt(run.metrics), (4, "4s"))
        self.assertIsNone(run.metrics["p95_duration"])

    def test_wall_clock_card_uses_human_duration_for_long_sessions(self):
        from agent_trajviz.insight.metrics import format_duration_seconds

        self.assertEqual(format_duration_seconds(45), "45s")
        self.assertEqual(format_duration_seconds(125), "2m 5s")
        self.assertEqual(format_duration_seconds(3725), "1h 02m")
        self.assertEqual(wall_clock_fmt({"wall_clock": 5400, "total_duration": 5400}), (5400, "1h 30m"))
        kpis = build_overview_kpi_html(
            {"total_steps": 1, "assistant_steps": 1, "user_steps": 0, "tokens": {"total": 0},
             "p95_duration": 125, "tool_call_count": 0, "tool_fail": 0},
            "1h 30m",
        )
        self.assertIn("1h 30m", kpis)
        self.assertIn("P95 2m 5s", kpis)

    def test_observed_zero_duration_remains_zero(self):
        raw = capture()
        for message in raw["messages"]:
            message["info"]["time"] = {"created": 1000, "completed": 1000}
        run = load(raw)
        self.assertEqual(run.metrics["wall_clock"], 0)
        self.assertEqual(run.metrics["p95_duration"], 0)
        self.assertTrue(all(agent["total_duration_s"] == 0 for agent in run.agents))
        self.assertEqual(wall_clock_fmt(run.metrics), (0, "0s"))

    def test_legacy_custom_no_clock_defaults_remain_compatible(self):
        steps = [{"role": "assistant", "tokens": {}, "duration": None,
                  "tool_call_count": 0, "tool_calls": [], "parts": []}]
        metrics = compute_metrics(steps, {})
        self.assertEqual(metrics["wall_clock"], 0)
        self.assertEqual(metrics["p95_duration"], 0)


if __name__ == "__main__":
    unittest.main()
