"""Regression tests for the 2026-07-06 review fixes — audit hardening.

Pins the silent-pass routes (review 1.11):
  - atom deletion / unknown error type -> `unverifiable`, never `re_derived`
  - resolved=None -> `unverifiable`, not a tamper `mismatch`
  - judge fault: empty citations fail; cap/error-type cross-swap fails;
    relabel vs the cached verdict fails when the cache is readable

Run:  python3 -m unittest tests.test_review_fixes_audit -v
"""

from __future__ import annotations

import unittest


import agent_trajviz.awe._engine.judge as judge
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, UNVERIFIABLE, MISMATCH, STRUCTURAL_OK
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory

GOLD = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
AGENT = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,1 +1,1 @@\n-old\n+other\n"


def _fault(cap="code_editing", et="incorrect_patch", conf="high", ev=None):
    # true fraction for GOLD vs AGENT above: the removed line matches, the added
    # line differs -> 1 of 2 gold changed lines reproduced = 0.5
    return {
        "capability": cap,
        "error_type": et,
        "confidence": conf,
        "detector_evidence": ev if ev is not None else {"hit": ["pkg/a.py"], "reproduced_fraction": 0.5},
    }


def _audit(fault, resolved=False, traj=None):
    return audit_fault(fault, agent="a", inst="i", gold_text=GOLD, agent_text=AGENT, resolved=resolved, traj=traj)


class TestMinimumAtomRule(unittest.TestCase):
    def test_intact_fault_re_derives(self):
        self.assertEqual(_audit(_fault())["verdict"], RE_DERIVED)

    def test_atom_deletion_is_unverifiable_not_re_derived(self):
        self.assertEqual(_audit(_fault(ev={}))["verdict"], UNVERIFIABLE)

    def test_unknown_error_type_is_unverifiable(self):
        f = _fault(et="regression_introduced", ev={"hit": ["pkg/a.py"]})
        self.assertEqual(_audit(f)["verdict"], UNVERIFIABLE)

    def test_tampered_atom_still_mismatches(self):
        f = _fault(ev={"hit": ["pkg/a.py"], "reproduced_fraction": 0.9})
        self.assertEqual(_audit(f)["verdict"], MISMATCH)

    def test_missing_outcome_is_unverifiable_not_mismatch(self):
        self.assertEqual(_audit(_fault(), resolved=None)["verdict"], UNVERIFIABLE)


class TestJudgeFaultAudit(unittest.TestCase):
    def _traj(self):
        return NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(step_id=0, turn_id=0, actor="agent", type="message", text="plan"),
                NormalizedStep(step_id=1, turn_id=1, actor="agent", type="message", text="more"),
            ],
        )

    def _jf(self, cap="requirement_understanding", et="misread_requirement", conf="medium", steps=(0,)):
        return {
            "capability": cap,
            "error_type": et,
            "confidence": conf,
            "detector_evidence": {"evidence_steps": list(steps)},
        }

    def _audit_j(self, fault, cached=None):
        orig = judge.verdict_cached
        judge.verdict_cached = lambda *a, **k: cached
        try:
            return audit_fault(
                fault, agent="a", inst="i", gold_text=GOLD, agent_text=AGENT, resolved=False, traj=self._traj()
            )
        finally:
            judge.verdict_cached = orig

    def test_intact_judge_fault_structural_ok(self):
        self.assertEqual(self._audit_j(self._jf())["verdict"], STRUCTURAL_OK)

    def test_empty_citations_fail(self):
        self.assertEqual(self._audit_j(self._jf(steps=()))["verdict"], MISMATCH)

    def test_capability_error_type_cross_swap_fails(self):
        f = self._jf(cap="task_planning", et="misread_requirement")
        self.assertEqual(self._audit_j(f)["verdict"], MISMATCH)

    def test_relabel_vs_cached_verdict_fails(self):
        cached = {
            "verdict": {
                "requirement_understanding": {
                    "error_type": "missed_constraint",
                    "confidence": "medium",
                    "evidence_steps": [0],
                }
            }
        }
        self.assertEqual(self._audit_j(self._jf(), cached=cached)["verdict"], MISMATCH)

    def test_matching_cached_verdict_passes(self):
        cached = {
            "verdict": {
                "requirement_understanding": {
                    "error_type": "misread_requirement",
                    "confidence": "medium",
                    "evidence_steps": [0],
                }
            }
        }
        self.assertEqual(self._audit_j(self._jf(), cached=cached)["verdict"], STRUCTURAL_OK)


if __name__ == "__main__":
    unittest.main()
