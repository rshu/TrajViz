"""Remedies must follow supported attribution and case-specific signatures."""

import math
import unittest
from dataclasses import replace

from agent_trajviz.awe._engine.countermeasures import countermeasures_for
from agent_trajviz.awe._engine.remedies import select_remedies, refresh_case
from agent_trajviz.awe._engine.dossier import to_markdown


class RemedySelectionTests(unittest.TestCase):
    def fault(self, overlap=0.8, weight=1):
        return {
            "capability": "code_editing",
            "error_type": "incorrect_patch",
            "blame_weight": weight,
            "detector_evidence": {"region_overlap": overlap},
        }

    def ids(self, fault):
        return [r["id"] for r in select_remedies({"agent": "a"}, fault)["countermeasures"]]

    def test_content_and_region_variants_are_exclusive(self):
        self.assertEqual(self.ids(self.fault(0.8)), ["CM-EDIT-INCORRECT-1"])
        self.assertEqual(self.ids(self.fault(0)), ["CM-EDIT-INCORRECT-2"])
        self.assertEqual(self.ids(self.fault(0.01)), ["CM-EDIT-INCORRECT-1"])

    def test_unknown_and_invalid_region_abstain(self):
        for value in (None, math.nan, math.inf, True, "0", -1, 2):
            result = select_remedies({"agent": "a"}, self.fault(value))
            self.assertEqual(result["countermeasures"], [], value)
            self.assertTrue(all(d["decision"] == "uncertain" for d in result["countermeasure_decisions"]))

    def test_refuted_target_is_retained_without_active_remedy(self):
        fault = {
            "capability": "code_editing",
            "error_type": "relevant_change_omitted",
            "blame_weight": 0,
            "recommendation": ["old advice"],
        }
        case = {
            "agent": "a",
            "arbiter": {
                "applied": "demoted_to_conjunctive",
                "target": {"capability": "code_editing", "error_type": "relevant_change_omitted"},
            },
            "faults": [fault, self.fault()],
        }
        refresh_case(case)
        self.assertEqual(len(case["faults"]), 2)
        self.assertEqual(fault["recommendation_support"], "refuted")
        self.assertEqual(fault["recommendation"], [])
        self.assertEqual(case["faults"][1]["recommendation_support"], "supported")
        self.assertTrue(case["faults"][1]["countermeasures"])

    def test_zero_and_tiny_live_weights_are_different(self):
        self.assertEqual(self.ids(self.fault(weight=0)), [])
        f = self.fault(weight=0)
        f.update(select_remedies({"agent": "a"}, f, weight=0.001))
        self.assertTrue(select_remedies({"agent": "a"}, f)["countermeasures"])

    def test_refuted_recipe_and_mismatched_agent_are_rejected(self):
        e = countermeasures_for(error_type="incorrect_patch")[0]
        for entry in (replace(e, status="refuted"), replace(e, applicability={"agents": ["other"]})):
            r = select_remedies({"agent": "a"}, self.fault(), entries=[entry])
            self.assertEqual(r["countermeasures"], [])
            self.assertEqual(r["countermeasure_decisions"][0]["decision"], "rejected")

    def test_bad_audit_or_rules_defer_recommendation(self):
        f = self.fault()
        f["audit"] = {"verdict": "mismatch"}
        self.assertEqual(self.ids(f), [])
        e = replace(countermeasures_for(error_type="incorrect_patch")[0], selection_rules={"unsafe": True})
        self.assertEqual(
            select_remedies({"agent": "a"}, self.fault(), entries=[e])["countermeasure_decisions"][0]["decision"],
            "uncertain",
        )

    def test_metadata_retains_limits_and_negative_calibration(self):
        r = select_remedies({"agent": "a"}, self.fault())["countermeasures"][0]
        for field in (
            "contract",
            "calibration",
            "expected_effect",
            "hook_event",
            "verification",
            "diagnostic_signature",
            "deployment_status",
        ):
            self.assertIn(field, r)
        self.assertIn("0.83", r["calibration"]["adjacent_negative"])
        self.assertEqual(r["deployment_status"], "prerequisites_unverified")

    def test_refutation_renderer_explains_suppression(self):
        from .test_dossier_arbiter import _case

        case = _case(
            {
                "applied": "demoted_to_conjunctive",
                "target": {"capability": "code_editing", "error_type": "relevant_change_omitted"},
            }
        )
        case["faults"][0]["blame_weight"] = 0
        refresh_case(case)
        md = to_markdown([case])
        self.assertIn("fault_refuted", md)
        self.assertNotIn("Secondary-fault fix", md)


if __name__ == "__main__":
    unittest.main()
