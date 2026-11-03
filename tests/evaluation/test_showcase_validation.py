import unittest

from agent_trajviz.awe._engine.showcase import arbiter_refutation_target, validation_summary


class ShowcaseValidationTests(unittest.TestCase):
    def summary(self, n=4):
        return validation_summary(
            {"failure_errors_emitted": 1 if n else 0, "n_resolved": n},
            {"per_mutation": {"wrong_file": {"n": n, "recall": 3 if n else 0}}},
            {"control": {"n": n, "ok": 2 if n else 0}},
            {"passed": 2, "n": 3},
            {"passed": 1, "n": 1},
            {"passed": 1, "n": 2},
            {"passed": 3, "n": 3},
            {"counts": {"re_derived": 2, "structural_ok": 1, "mismatch": 1}, "n": 4},
            655,
        )

    def test_partial_failures_are_visible(self):
        result = self.summary()
        self.assertEqual(result["fp"], "1/4")
        self.assertEqual(result["mutation"], "75.0%")
        self.assertEqual(result["synthetic"], "50.0%")
        self.assertEqual(result["judge_wiring"], "3/4")
        self.assertEqual(result["evidence_audit"], "3/4")
        self.assertEqual(result["evidence_mismatches"], 1)
        self.assertEqual(result["sample_counts"]["mutation"], {"hits": 3, "n": 4})

    def test_empty_controls_do_not_claim_perfect_validation(self):
        result = self.summary(0)
        for key in ("fp", "mutation", "synthetic"):
            self.assertEqual(result[key], "not measured")

    def test_applied_demotion_is_eligible_for_example(self):
        target = {"capability": "code_verification", "error_type": "relevant_test_not_run"}
        case = {"arbiter": {"support": "refute", "applied": "demoted_to_conjunctive", "target": target}}
        self.assertEqual(arbiter_refutation_target(case), target)

    def test_raw_refutation_and_missing_targets_are_ineligible(self):
        for applied in ("noted", "refuted_reassigned", "demoted"):
            case = {
                "arbiter": {
                    "support": "refute",
                    "applied": applied,
                    "target": {"capability": "code_verification", "error_type": "relevant_test_not_run"},
                }
            }
            self.assertIsNone(arbiter_refutation_target(case))
        self.assertIsNone(
            arbiter_refutation_target({"arbiter": {"support": "refute", "applied": "demoted_to_conjunctive"}})
        )
        self.assertIsNone(arbiter_refutation_target({}))
