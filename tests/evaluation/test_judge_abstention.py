"""Tests for the v0.36 selective-voting abstention bar on the judge panel
(Raj et al. arXiv:2607.28802 §6): a winning error label below the configured
agreement fraction keeps its record but is demoted to low confidence
(blame-ineligible) and marked ``abstained``.

Run:  python3 -m unittest tests.test_judge_abstention -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.judge import aggregate_verdicts, verdict_to_specs


def _v(req="none", conf="high", steps=(3,)):
    return {
        "requirement_understanding": {
            "error_type": req,
            "confidence": conf,
            "evidence_steps": list(steps),
            "rationale": "r",
        },
        "task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": [], "rationale": ""},
    }


PANEL_3_OF_5 = [_v("misread_requirement"), _v("misread_requirement"), _v("misread_requirement"), _v(), _v()]


class TestAbstention(unittest.TestCase):
    def test_default_bar_preserves_v032_behavior(self):
        out = aggregate_verdicts(PANEL_3_OF_5, min_agree=0)
        ru = out["requirement_understanding"]
        self.assertEqual(ru["error_type"], "misread_requirement")
        self.assertEqual(ru["agreement"], 0.6)
        self.assertNotIn("abstained", ru)
        self.assertNotEqual(ru["confidence"], "low")

    def test_below_bar_demotes_and_marks(self):
        out = aggregate_verdicts(PANEL_3_OF_5, min_agree=0.8)
        ru = out["requirement_understanding"]
        self.assertEqual(ru["error_type"], "misread_requirement")  # kept, not erased
        self.assertEqual(ru["confidence"], "low")  # blame-ineligible
        self.assertTrue(ru["abstained"])
        specs = verdict_to_specs({"verdict": out})
        self.assertFalse(any(s["capability"] == "requirement_understanding" for s in specs))

    def test_unanimity_passes_the_strictest_bar(self):
        panel = [_v("misread_requirement")] * 5
        out = aggregate_verdicts(panel, min_agree=1.0)
        ru = out["requirement_understanding"]
        self.assertNotIn("abstained", ru)
        self.assertNotEqual(ru["confidence"], "low")

    def test_none_verdicts_never_abstain(self):
        out = aggregate_verdicts([_v(), _v(), _v()], min_agree=1.0)
        self.assertNotIn("abstained", out["requirement_understanding"])
        self.assertEqual(out["requirement_understanding"]["error_type"], "none")


if __name__ == "__main__":
    unittest.main()
