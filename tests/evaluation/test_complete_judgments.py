import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_trajviz.awe._engine import arbiter, config, judge
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory


def complete_judge():
    return {
        cap: {
            "error_type": "none",
            "confidence": "high",
            "evidence_steps": [1],
            "rationale": "Visible process is adequate.",
        }
        for cap in judge.JUDGE_CAPS
    }


def response(obj, finish="stop"):
    return {"content": json.dumps(obj), "finish_reason": finish}


class TestCompleteJudgments(unittest.TestCase):
    def setUp(self):
        self.traj = NormalizedTrajectory(
            "a", "i", [NormalizedStep(1, 0, "agent", "message", text="Investigate the issue.")]
        )
        self.gold = SimpleNamespace(problem_statement="Fix the behavior.")

    def test_truncated_outer_object_does_not_become_inner_verdict(self):
        text = '{"requirement_understanding": ' + json.dumps(complete_judge()["requirement_understanding"]) + ","
        self.assertIsNone(judge._extract_json(text))
        self.assertIsNotNone(judge._extract_json("```json\n" + json.dumps(complete_judge()) + "\n```"))

    def run_judge(self, client, directory):
        with (
            patch.object(judge, "build_gold_reference", return_value=self.gold),
            patch.object(judge, "load_normalized", return_value=self.traj),
            patch.object(judge, "_verdict_key", return_value="key"),
            patch.object(config, "JUDGE_CACHE_DIR", Path(directory)),
        ):
            return judge.judge_one("a", "i", client=client, model="m")

    def test_missing_capability_retries_instead_of_clean_bill(self):
        partial = {"requirement_understanding": complete_judge()["requirement_understanding"]}
        client = Mock(side_effect=[response(partial), response(complete_judge())])
        with tempfile.TemporaryDirectory() as td:
            record = self.run_judge(client, td)
        self.assertEqual(client.call_count, 2)
        self.assertNotIn("parse_failed", record)

    def test_length_limited_complete_object_is_unavailable_and_resumable(self):
        client = Mock(return_value=response(complete_judge(), "length"))
        with tempfile.TemporaryDirectory() as td:
            record = self.run_judge(client, td)
            self.assertTrue(record["parse_failed"])
            with patch.object(config, "JUDGE_CACHE_DIR", Path(td)):
                self.assertEqual(judge.judge_candidates("a", "i", "m"), ([], False))
            replacement = Mock(return_value=response(complete_judge()))
            repaired = self.run_judge(replacement, td)
        self.assertEqual(replacement.call_count, 1)
        self.assertNotIn("parse_failed", repaired)

    def test_structurally_invalid_json_is_not_an_arbiter_verdict(self):
        client = Mock(return_value=response({"support": "confirm"}))
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(arbiter, "build_gold_reference", return_value=self.gold),
            patch.object(arbiter, "load_normalized", return_value=self.traj),
            patch.object(arbiter, "_verdict_key", return_value="key"),
            patch.object(arbiter, "ARBITER_CACHE_DIR", Path(td)),
        ):
            record = arbiter.verify_one("a", "i", "code_editing", "relevant_change_omitted", client=client, model="m")
            self.assertTrue(record["parse_failed"])
            self.assertIsNone(arbiter.verdict_cached("a", "i", "m", "code_editing", "relevant_change_omitted"))
        self.assertEqual(client.call_count, 2)

    def test_unknown_labels_and_truncated_arbiter_answers_are_rejected(self):
        invalid = complete_judge()
        invalid["task_planning"]["error_type"] = "invented_label"
        self.assertFalse(judge._valid_response(response(invalid)))
        verdict = {"support": "refute", "confidence": "high", "evidence_steps": [1], "rationale": "x"}
        self.assertTrue(arbiter._valid_response(response(verdict)))
        self.assertFalse(arbiter._valid_response(response(verdict, "length")))
