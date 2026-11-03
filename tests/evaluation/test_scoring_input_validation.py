import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_trajviz.awe._engine import config, outcomes, scoring


class TestScoringInputValidation(unittest.TestCase):
    def test_invalid_gold_cannot_silently_shrink_denominator(self):
        with (
            patch.object(scoring, "resolved", return_value=False),
            patch.object(scoring, "build_gold_reference", side_effect=ValueError("bad JSON")),
            self.assertRaisesRegex(ValueError, "a/i.*invalid gold reference"),
        ):
            scoring.agent_capability_scores("a", ["i"], use_judge=False)

    def test_explicit_empty_scope_does_not_fall_back_to_full_corpus(self):
        with patch.object(config, "list_instances", side_effect=AssertionError("full corpus")):
            r = scoring.agent_capability_scores("a", [], use_judge=False)
        self.assertEqual(r["n_tasks"], 0)
        self.assertEqual(r["n_requested"], 0)

    def test_excluded_or_missing_outcomes_are_listed(self):
        with patch.object(scoring, "resolved", return_value=None):
            r = scoring.agent_capability_scores("a", ["i"], use_judge=False)
        self.assertEqual(r["unscored_instances"], ["i"])
        self.assertEqual(r["n_requested"], 1)
        self.assertEqual(r["n_tasks"], 0)

    def test_duplicate_instances_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            scoring.agent_capability_scores("a", ["i", "i"], use_judge=False)

    def test_raw_outcomes_require_real_booleans(self):
        with tempfile.TemporaryDirectory() as td, patch.object(config, "DATA", Path(td)):
            p = Path(td) / "eval_a.json"
            for bad in [1, 0, "false", {}, []]:
                p.write_text(json.dumps({"i": bad}))
                with self.assertRaisesRegex(ValueError, "true/false/null"):
                    outcomes.load_outcomes("a")
            p.write_text(json.dumps({"i": False, "j": None}))
            self.assertIs(outcomes.load_outcomes("a")["i"], False)

    def test_override_rejects_integer_boolean_lookalikes(self):
        with tempfile.TemporaryDirectory() as td, patch.object(config, "PROJECT_DATA", Path(td)):
            p = Path(td) / "overrides/resolved_overrides.json"
            p.parent.mkdir()
            p.write_text(
                json.dumps(
                    {"overrides": [{"agent": "a", "instance_id": "i", "resolved": 1, "reason": "r", "evidence": "e"}]}
                )
            )
            with self.assertRaisesRegex(ValueError, "true/false/null"):
                outcomes.load_overrides()
