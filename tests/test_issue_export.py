"""Agent-ready Overview Issues JSON export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_trajviz.insight.issue_export import (
    SCHEMA_ID,
    build_issues_agent_export,
    issue_suggested_action,
    issues_export_filename,
    write_issues_agent_export,
)
from agent_trajviz.insight.issues import IssueJudgment, OverviewIssue
from tests.overview_session_stub import overview_session as _session


def _step(index: int, *, role: str = "assistant", preview: str = "", tools=None, session_id: str = ""):
    row = {
        "index": index,
        "role": role,
        "text_preview": preview,
        "tool_calls": tools or [],
    }
    if session_id:
        row["session_id"] = session_id
    return row


class IssueExportTests(unittest.TestCase):
    def test_empty_issues_still_exports_schema(self):
        session = _session(steps=[_step(0, role="user", preview="Fix the flaky test")])
        payload = build_issues_agent_export(session, issues=[])
        self.assertEqual(payload["schema"], SCHEMA_ID)
        self.assertEqual(payload["issue_count"], 0)
        self.assertEqual(payload["issues"], [])
        self.assertIn("user_task", payload["session"])
        self.assertEqual(payload["session"]["user_task"], "Fix the flaky test")
        self.assertTrue(payload["purpose"])
        self.assertTrue(payload["instructions"])

    def test_issue_includes_action_judgment_and_step_evidence(self):
        tools = [{
            "tool_name": "Bash",
            "status": "error",
            "input": {"command": "npm test"},
            "output": "FAIL suite/spec.js",
            "error": "exit code 1",
            "error_type": "tool",
        }]
        session = _session(
            steps=[
                _step(0, role="user", preview="Make CI green"),
                _step(4, preview="retrying", tools=[]),
                _step(5, preview="running tests", tools=tools),
                _step(6, preview="checking logs"),
            ],
            path="/tmp/run.json",
            format="claude_code",
            steps_total=7,
            wall_clock="1m 2s",
        )
        issue = OverviewIssue(
            kind="error",
            title="npm: exit code 1 (2×)",
            detail="command failed",
            why="No recovery path found after this error cluster.",
            steps=(5,),
            source_id="fail:tool:npm: exit code 1",
            judgment=IssueJudgment(
                where="AGENTS.md",
                fix="After a test failure, read the failing file before editing.",
                also="Avoid blind npm test retries.",
                confidence="high",
            ),
        )
        payload = build_issues_agent_export(session, issues=[issue])
        self.assertEqual(payload["issue_count"], 1)
        card = payload["issues"][0]
        self.assertEqual(card["id"], 1)
        self.assertEqual(card["kind"], "error")
        self.assertEqual(card["related_steps"], [5])
        self.assertIn("did not recover", card["suggested_action"])
        self.assertEqual(card["judgment"]["where"], "AGENTS.md")
        self.assertEqual(card["judgment"]["confidence"], "high")

        evidence_idxs = [s["index"] for s in card["evidence"]["steps"]]
        self.assertIn(5, evidence_idxs)
        # Radius-1 neighbors included when present.
        self.assertIn(4, evidence_idxs)
        self.assertIn(6, evidence_idxs)

        step5 = next(s for s in card["evidence"]["steps"] if s["index"] == 5)
        self.assertEqual(step5["tools"][0]["tool"], "Bash")
        self.assertEqual(step5["tools"][0]["command"], "npm test")
        self.assertTrue(step5["tools"][0].get("failed"))
        self.assertIn("FAIL", step5["tools"][0]["output"])

        self.assertEqual(payload["session"]["path"], "/tmp/run.json")
        self.assertEqual(payload["session"]["format"], "claude_code")
        self.assertEqual(payload["session"]["wall_clock"], "1m 2s")

    def test_playbook_action_for_antipattern_prefix(self):
        issue = OverviewIssue(
            kind="antipattern",
            title="Blind retry",
            detail="retried same command",
            why="wasted turns",
            steps=(3,),
            source_id="antipattern:blind_retry:bash",
        )
        action = issue_suggested_action(issue)
        self.assertIn("read the error", action.lower())

    def test_export_filename_uses_session_id(self):
        session = _session(
            steps=[_step(0)],
            raw={"metadata": {"session_id": "ses_abc123"}},
        )
        self.assertEqual(issues_export_filename(session), "issues_ses_abc123.json")

    def test_write_issues_agent_export_roundtrip(self):
        session = _session(
            steps=[_step(1, session_id="ses_run1", tools=[{
                "tool_name": "Read",
                "input": {"file_path": "src/a.py"},
                "output": "def a():\n    pass\n",
            }])],
            failure_patterns=[{
                "cluster_label": "Read: missing",
                "count": 1,
                "example_error": "ENOENT",
                "recovery_path": [],
                "steps": [1],
                "error_class": "tool",
            }],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_issues_agent_export(session, tmp)
            self.assertTrue(path.endswith("issues_ses_run1.json"))
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], SCHEMA_ID)
        self.assertGreaterEqual(data["issue_count"], 1)
        self.assertEqual(data["issues"][0]["related_steps"], [1])
        self.assertEqual(data["session"]["session_id"], "ses_run1")


if __name__ == "__main__":
    unittest.main()
