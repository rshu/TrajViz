"""A replaced load must stop the still-streaming LLM Issues judge.

The requested judge paints the Issues panel across many yields (one LLM call
per issue). If the user clears the upload or
loads another trajectory while that stream is running, the old stream would
keep overwriting the new trace's panel with the old one's content. Every load
records the run its browser session displays (``shared.record_latest_run``);
the judge checks ``shared.is_latest_run`` on each resume and stops itself.
"""

from __future__ import annotations

import os
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_trajviz as tv
from agent_trajviz.insight.issue_judge import JudgeProgress
from agent_trajviz.insight.issues import OverviewIssue
from agent_trajviz.insight.ui.load import do_load
from agent_trajviz.insight.ui.overview_tab import on_suggest_fixes
from agent_trajviz.insight.ui.shared import (
    drop_latest_run,
    is_latest_run,
    record_latest_run,
)

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "tests" / "fixtures" / "native_claude_code"
DJANGO, ASTROPY = REF / "django__django-11477.jsonl", REF / "astropy__astropy-13033.jsonl"

_ANALYZE_ENV = ("ANALYZE_BASE_URL", "ANALYZE_API_KEY", "ANALYZE_MODEL", "ANALYZE_PROVIDER")


class LatestRunRegistryTests(unittest.TestCase):
    def tearDown(self):
        drop_latest_run("s1")
        drop_latest_run("s2")

    def test_untracked_browser_is_stale_but_direct_callers_work(self):
        run = tv.Trajectory("codeagent", [])
        self.assertFalse(is_latest_run("never-loaded", run))
        self.assertTrue(is_latest_run(None, run))

    def test_recorded_run_is_current_until_replaced(self):
        run_a, run_b = tv.Trajectory("codeagent", []), tv.Trajectory("codeagent", [])
        record_latest_run("s1", run_a)
        self.assertTrue(is_latest_run("s1", run_a))
        self.assertFalse(is_latest_run("s1", run_b))
        record_latest_run("s1", run_b)
        self.assertFalse(is_latest_run("s1", run_a))
        self.assertTrue(is_latest_run("s1", run_b))

    def test_cleared_upload_supersedes_every_run(self):
        run = tv.Trajectory("codeagent", [])
        record_latest_run("s1", run)
        record_latest_run("s1", None)
        self.assertFalse(is_latest_run("s1", run))

    def test_sessions_are_independent(self):
        run_a, run_b = tv.Trajectory("codeagent", []), tv.Trajectory("codeagent", [])
        record_latest_run("s1", run_a)
        record_latest_run("s2", run_b)
        self.assertTrue(is_latest_run("s1", run_a))

    def test_registry_does_not_keep_an_evicted_run_alive(self):
        run = tv.Trajectory("codeagent", [])
        reference = weakref.ref(run)
        record_latest_run("s1", run)
        del run
        self.assertIsNone(reference())

    def test_unloaded_session_stays_stale(self):
        run = tv.Trajectory("codeagent", [])
        record_latest_run("s1", run)
        drop_latest_run("s1")
        self.assertFalse(is_latest_run("s1", run))


class DoLoadRecordsTests(unittest.TestCase):
    def tearDown(self):
        drop_latest_run("s1")

    def test_each_load_replaces_the_sessions_latest_run(self):
        request = SimpleNamespace(session_hash="s1")
        run_a = do_load(str(DJANGO), request=request)["state_run"]
        self.assertTrue(is_latest_run("s1", run_a))
        run_b = do_load(str(ASTROPY), request=request)["state_run"]
        self.assertFalse(is_latest_run("s1", run_a))
        self.assertTrue(is_latest_run("s1", run_b))

    def test_failed_load_records_none(self):
        request = SimpleNamespace(session_hash="s1")
        do_load(str(DJANGO), request=request)
        do_load("/no/such/file.json", request=request)
        self.assertFalse(is_latest_run("s1", do_load(str(DJANGO), request=None)["state_run"]))


class SupersededJudgeStreamTests(unittest.TestCase):
    """Drive ``on_suggest_fixes`` directly: the first yield happens before any
    LLM call, so superseding right after it must end the stream there."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ANALYZE_ENV}
        # A ready-looking config pointed at a dead port: reaching it at all
        # after the supersede would mean the guard failed.
        os.environ.update(
            ANALYZE_BASE_URL="http://127.0.0.1:9",
            ANALYZE_API_KEY="k",
            ANALYZE_MODEL="m",
            ANALYZE_PROVIDER="openai",
        )

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        drop_latest_run("s1")

    def test_stream_stops_once_another_run_is_loaded(self):
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        record_latest_run("s1", run)
        stream = on_suggest_fixes(run, request=request)
        html_out, _ = next(stream)
        self.assertIn("正在等待 AI 总结任务与结局…", html_out)
        # The user clears the upload and loads another trajectory.
        record_latest_run("s1", tv.load(ASTROPY))
        with self.assertRaises(StopIteration):
            next(stream)

    def test_stream_never_yields_when_already_superseded(self):
        # Another load landed before the first resume — do not paint the old
        # run's pending spinner over the new panel.
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        replacement = tv.load(ASTROPY)
        record_latest_run("s1", replacement)
        stream = on_suggest_fixes(run, request=request)
        with self.assertRaises(StopIteration):
            next(stream)

    def test_stream_stops_once_the_upload_is_cleared(self):
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        record_latest_run("s1", run)
        stream = on_suggest_fixes(run, request=request)
        next(stream)
        record_latest_run("s1", None)
        with self.assertRaises(StopIteration):
            next(stream)

    def test_stream_stops_after_browser_unload(self):
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        record_latest_run("s1", run)
        stream = on_suggest_fixes(run, request=request)
        next(stream)
        drop_latest_run("s1")
        with self.assertRaises(StopIteration):
            next(stream)

    def test_superseded_progress_does_not_resume_the_provider_iterator(self):
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        record_latest_run("s1", run)
        issue = OverviewIssue("error", "Issue", "Detail", "Why", (1,))

        def progress(*_args, **_kwargs):
            yield JudgeProgress([issue], [], 1, 1, issue.title, False)
            self.fail("The obsolete issue stream tried to start another provider request")

        with (patch("agent_trajviz.insight.issue_judge.judge_run_summary", return_value=("task", "outcome")),
              patch("agent_trajviz.insight.ui.overview_tab.prepare_overview_issues", return_value=[issue]),
              patch("agent_trajviz.insight.ui.overview_tab.iter_judge_overview_issues", side_effect=progress)):
            stream = on_suggest_fixes(run, request=request)
            next(stream)
            html_out, _ = next(stream)
            self.assertIn("Suggesting fixes 1/1", html_out)
            drop_latest_run("s1")
            with self.assertRaises(StopIteration):
                next(stream)

    def test_stream_continues_while_its_run_is_still_current(self):
        run = tv.load(DJANGO)
        request = SimpleNamespace(session_hash="s1")
        record_latest_run("s1", run)
        stream = on_suggest_fixes(run, request=request)
        next(stream)  # pending spinner
        # Not superseded: the stream keeps going — the summary call fails fast
        # against the dead endpoint and the per-issue judging still yields —
        # instead of stopping silently right after the first yield.
        yields = sum(1 for _ in stream)
        self.assertGreaterEqual(yields, 1)


if __name__ == "__main__":
    unittest.main()
