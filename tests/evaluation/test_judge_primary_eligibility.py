import unittest
from agent_trajviz.awe._engine.rootcause import build_blame
from .test_review_fixes_blame import _det, _cand


class TestJudgePrimaryEligibility(unittest.TestCase):
    def test_lone_judge_fault_retains_weight_but_never_primary(self):
        for cap, et in [("task_planning", "flawed_plan"), ("requirement_understanding", "misread_requirement")]:
            b = build_blame(_det([_cand(cap, et)], loc_flag="clean"))
            self.assertIsNone(b["primary"])
            self.assertEqual(b["status"], "conjunctive")
            self.assertEqual(b["capability_blame"][cap], 1.0)

    def test_multiple_judge_labels_cannot_dilute_eligible_dominance(self):
        b = build_blame(
            _det(
                [
                    _cand("code_editing", "incorrect_patch"),
                    _cand("requirement_understanding", "misread_requirement"),
                    _cand("task_planning", "flawed_plan"),
                    _cand("task_planning", "ignored_plan"),
                ]
            )
        )
        self.assertEqual(b["primary"]["capability"], "code_editing")
        self.assertEqual(b["status"], "primary")
        self.assertAlmostEqual(sum(b["capability_blame"].values()), 1)
