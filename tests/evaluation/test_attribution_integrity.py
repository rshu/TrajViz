import unittest
from unittest.mock import patch
from agent_trajviz.awe._engine.rootcause import build_blame, _needs_adjudication
from agent_trajviz.awe._engine.scoring import agent_capability_scores
from agent_trajviz.awe._engine.evidence import build_chain
from agent_trajviz.awe._engine.gold import gold_from_requirement


def detection():
    return {
        "resolved": False,
        "loc_flag": "no_source_edit",
        "candidates": [{"capability": "code_editing", "error_type": "gave_up_early", "confidence": "high"}],
        "opportunities": {"code_localization": True, "code_editing": True},
    }


def refute(*args):
    return {"support": "refute", "confidence": "high", "evidence_steps": [1]}


class AttributionIntegrity(unittest.TestCase):
    def test_refuted_singleton_preserves_failure_mass(self):
        b = build_blame(detection(), adjudicate=refute)
        self.assertEqual(b["status"], "unattributed")
        self.assertEqual(b["capability_blame"], {"unattributed": 1.0})
        self.assertTrue(b["blame"][0]["arbiter_refuted"])
        self.assertEqual(b["arbiter"]["applied"], "unattributed_after_refute")

    def test_score_reports_uncertainty_instead_of_certified_perfection(self):
        with (
            patch("agent_trajviz.awe._engine.scoring.resolved", return_value=False),
            patch("agent_trajviz.awe._engine.scoring.build_gold_reference", return_value=object()),
            patch("agent_trajviz.awe._engine.scoring.detect", return_value=detection()),
            patch("agent_trajviz.awe._engine.scoring.auto_adjudicate", return_value=refute),
        ):
            s = agent_capability_scores("a", ["i"], use_trajectory=False, use_judge=False)
        self.assertEqual(s["unattributed"], 1)
        for c in s["capabilities"].values():
            self.assertEqual(c["score_identification_bounds"], (0.0, 100.0))
            self.assertEqual(c["fault_incidence"], 0)
            self.assertEqual(c["top_errors"], [])

    def test_associational_extra_edits_require_adjudication(self):
        self.assertTrue(_needs_adjudication({"capability": "code_editing", "error_type": "excessive_extra_edits"}))
        self.assertFalse(_needs_adjudication({"capability": "code_editing", "error_type": "incorrect_patch"}))

    def test_judge_explanation_matches_secondary_policy(self):
        c = {
            "capability": "task_planning",
            "error_type": "flawed_plan",
            "confidence": "high",
            "evidence": {"evidence_steps": [1]},
        }
        g = gold_from_requirement("i", {"FAIL_TO_PASS": [], "PASS_TO_PASS": []})
        e = build_chain(c, detection(), g, {}, {}, 0.2, False)
        text = " ".join(x["statement"] for x in e["links"])
        self.assertIn("not eligible for primary", text)
        self.assertNotIn("takes primary blame", text)

    def test_incidence_does_not_dilute_when_contributor_is_added(self):
        d = detection()
        d["candidates"][0]["error_type"] = "incorrect_patch"
        results = []
        for extra in (
            [],
            [{"capability": "code_verification", "error_type": "relevant_test_not_run", "confidence": "high"}],
        ):
            current = dict(d, candidates=d["candidates"] + extra)
            with (
                patch("agent_trajviz.awe._engine.scoring.resolved", return_value=False),
                patch("agent_trajviz.awe._engine.scoring.build_gold_reference", return_value=object()),
                patch("agent_trajviz.awe._engine.scoring.detect", return_value=current),
                patch("agent_trajviz.awe._engine.scoring.auto_adjudicate", return_value=None),
            ):
                results.append(agent_capability_scores("a", ["i"], use_trajectory=False, use_judge=False))
        self.assertEqual([r["capabilities"]["code_editing"]["fault_incidence"] for r in results], [1, 1])
        self.assertNotEqual(
            results[0]["capabilities"]["code_editing"]["rate"], results[1]["capabilities"]["code_editing"]["rate"]
        )
