"""Controls must supply observed reachability and preserve attribution gaps."""

import unittest
from unittest.mock import patch

from agent_trajviz.awe._engine import config, validate
from agent_trajviz.awe._engine.detect import detect
from agent_trajviz.awe._engine.gold import GoldReference


class ValidationObservabilityTests(unittest.TestCase):
    def gold(self):
        return GoldReference(
            instance_id="case",
            gold_files=["src/core.py", "src/other.py"],
            gold_patch_loc=8,
            gold_symbols=["repair"],
            repo="fixture",
            base_commit="base",
            problem_statement="Repair `repair`.",
            test_patch_files=[],
            fail_to_pass=["test_repair"],
            pass_to_pass=[],
            is_multi_file=True,
            is_cross_module=False,
            modules=["src"],
        )

    def test_mutation_harness_supplies_observed_read_set(self):
        gold = self.gold()

        def injected(*args, **kwargs):
            return detect(*args, **kwargs, gold_override=gold)

        with (
            patch.object(config, "list_instances", return_value=["case"]),
            patch.object(validate, "build_gold_reference", return_value=gold),
            patch.object(validate, "detect", side_effect=injected),
        ):
            result = validate.patch_mutation()
        for name, row in result["per_mutation"].items():
            self.assertEqual(row["recall"], row["n"], name)
            self.assertEqual(row["cap_ok"], row["n"], name)

    def test_unknown_read_set_still_abstains(self):
        gold = self.gold()
        wrong = validate._patch("case", ["src/unrelated.py"], 12, ["repair"])
        result = detect(
            "case",
            "synthetic",
            use_trajectory=False,
            use_judge=False,
            gold_override=gold,
            patch_override=wrong,
            resolved_override=False,
        )
        self.assertFalse(result["signals"]["read_set_available"])
        self.assertNotIn("missed_relevant_file", [c["error_type"] for c in result["candidates"]])

    def test_arbiter_controls_cover_both_refutation_outcomes(self):
        result = validate.arbiter_wiring()
        self.assertTrue(result["checks"]["refuted_singleton_is_unattributed"])
        self.assertTrue(result["checks"]["refute_with_surviving_contribution_is_conjunctive"])
        self.assertEqual(result["passed"], result["n"])
