import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from agent_trajviz.awe._engine.evidenceview import build_view, render_rows, retrieve_steps, retrieve_source
from agent_trajviz.awe._engine.trajectory import NormalizedStep as S, NormalizedTrajectory as T


class EvidenceViews(unittest.TestCase):
    def test_middle_focus_retains_result_and_budget(self):
        steps = [S(i, i, "agent", "message", text="ordinary " * 40) for i in range(60)]
        steps[30] = S(30, 30, "agent", "tool_call", canonical_action="read", call_id="c", files=["a.py"])
        steps[31] = S(31, 30, "tool", "tool_result", call_id="c", output_excerpt="CRITICAL feedback " * 30)
        view = build_view(
            "x" * 4000 + "\nMUST retain constraint", T("a", "i", steps), max_rows=6, max_text=80, focus_steps=[30]
        )
        self.assertLessEqual(len(view["rows"]), 6)
        self.assertEqual(len(view["problem_statement"]), 4023)
        self.assertTrue({30, 31} <= {r["step"] for r in view["rows"]})
        self.assertTrue(all(len(r["text"]) <= 80 for r in view["rows"]))
        self.assertEqual(view["omissions"]["retained_rows"] + view["omissions"]["omitted_rows"], 60)
        self.assertIn("VIEW COVERAGE", "\n".join(render_rows(view)))

    def test_unknown_hidden_result_not_displayed_as_observed(self):
        traj = T(
            "a",
            "i",
            [
                S(0, 0, "agent", "reasoning", text="secret", observability="hidden"),
                S(1, 1, "user", "message", text="Additional constraint", subagent_id="worker"),
            ],
        )
        view = build_view("task", traj)
        self.assertEqual(view["rows"][0]["kind"], "instruction")
        self.assertEqual(view["rows"][0]["thread"], "worker")
        self.assertTrue(view["hidden_reasoning"])
        self.assertEqual(view["omissions"]["hidden_or_unrenderable_steps"], 1)

    def test_canonical_retrieval_reports_missing_and_truncated(self):
        traj = T("a", "i", [S(0, 0, "tool", "tool_result", output_excerpt="x" * 100)])
        result = retrieve_steps(traj, [0, 999], max_chars=20)
        self.assertEqual(result["used_chars"], 20)
        self.assertTrue(result["rows"][0]["text_truncated"])
        self.assertEqual(result["unreturned_step_ids"], [999])

    def test_original_record_retrieval_checks_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            body = json.dumps({"messages": [{"native": "x" * 50}]}).encode()
            path.write_bytes(body)
            traj = T(
                "a",
                "i",
                [S(0, 0, "agent", "message", source_ref={"pointer": "/messages/0"})],
                source={"path": str(path), "sha256": hashlib.sha256(body).hexdigest()},
            )
            result = retrieve_source(traj, 0, max_chars=20)
            self.assertTrue(result["source_verified"])
            self.assertTrue(result["truncated"])
            path.write_text("{}")
            with self.assertRaises(ValueError):
                retrieve_source(traj, 0)
