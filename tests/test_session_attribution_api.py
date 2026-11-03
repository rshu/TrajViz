"""The Overview's session details and the Attribution tab, through the public API."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz import attribution
from agent_trajviz.insight import attribution as _attr

CC_RUN = (Path(__file__).resolve().parent / "fixtures" / "native_claude_code" / "django__django-11477.jsonl")
CC_ARCHIVE = (Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory"
              / "claude_code" / "django__django-11477.json")


class SessionViewTests(unittest.TestCase):
    def test_session_view_carries_the_session_detail_fields(self):
        data = json.loads(tv.load(CC_RUN).view("session").to_json())["data"]
        self.assertEqual(data["model"], "claude-opus-4-6")
        self.assertEqual(data["agent"], "claude-code")
        self.assertTrue(data["started_at"].startswith("2026-"))
        self.assertEqual(set(data), {"model", "provider", "agent", "started_at", "finished_at", "session_id",
                                     "title", "branch", "directory", "platform", "version"})


@unittest.skipUnless(_attr.ATTRIBUTION_AVAILABLE, "attribution not importable")
class AttributionApiTests(unittest.TestCase):
    def test_public_diagnose_is_the_tabs_diagnosis(self):
        # Test-only bridge to attribution's byte-stable archived reference. Load the
        # equivalent native records, then attach the original source identity
        # through the normalized API; production never parses the old envelope.
        run = tv.Trajectory.from_steps(tv.load(CC_RUN).steps, format="claude_code",
                                       source=str(CC_ARCHIVE),
                                       sha256=hashlib.sha256(CC_ARCHIVE.read_bytes()).hexdigest())
        public = attribution.diagnose(run)
        from dataclasses import asdict

        tab = asdict(_attr.diagnose(agent="claude_code", instance_id="django__django-11477",
                                    source_path=run.source, fmt=run.format, expected_sha=run.sha256,
                                    merged_sources=0, reference_root=None))
        self.assertEqual(public, json.loads(json.dumps(tab)))
        self.assertTrue(public["available"], public.get("reason"))

    def test_available_reports_attribution(self):
        self.assertEqual(attribution.available(), (True, None))


class AttributionDegradesTests(unittest.TestCase):
    def test_an_unknown_task_is_a_reason_not_an_exception(self):
        result = attribution.diagnose(tv.load(CC_RUN), agent="no_such_agent", instance_id="x")
        self.assertFalse(result["available"])
        self.assertTrue(result["reason"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
