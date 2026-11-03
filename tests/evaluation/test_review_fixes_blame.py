"""Regression tests for the 2026-07-06 review fixes — blame + statistics.

Pins:
  1.9  contributing caps weighted like verification; never primary;
       capability-level primary gate (same-capability splits are not conjunctions)
  1.16 bootstrap p never exactly 0; tiny-n comparisons are not significant

Run:  python3 -m unittest tests.test_review_fixes_blame -v
"""

from __future__ import annotations

import random
import unittest


from agent_trajviz.awe._engine.rootcause import build_blame, CONTRIBUTING, DEFAULT_ROLES
from agent_trajviz.awe._engine.significance import paired_bootstrap_diff, MIN_PAIRS


def _det(cands, loc_flag="right_place_wrong_change", resolved=False):
    return {"resolved": resolved, "candidates": cands, "loc_flag": loc_flag, "agent": "a", "instance_id": "i"}


def _cand(cap, et, conf="high"):
    return {"capability": cap, "error_type": et, "confidence": conf}


class TestContributingRoles(unittest.TestCase):
    def test_self_repair_and_tool_weighted_like_verification(self):
        # editing primary + one candidate from each contributing cap, equal conf:
        # the three contributing entries must carry the same (verification) role.
        b = build_blame(
            _det(
                [
                    _cand("code_editing", "incorrect_patch"),
                    _cand("code_verification", "relevant_test_not_run"),
                    _cand("self_repair_loop", "repair_not_reverified"),
                    _cand("tool_use", "tool_failure_not_retried"),
                ]
            )
        )
        w = {e["capability"]: e["weight"] for e in b["blame"]}
        self.assertAlmostEqual(w["code_verification"], w["self_repair_loop"])
        self.assertAlmostEqual(w["code_verification"], w["tool_use"])
        # and the primary (editing) outweighs them by the role ratio 1.0 : 0.5
        self.assertAlmostEqual(
            w["code_editing"] / w["code_verification"], DEFAULT_ROLES["primary"] / DEFAULT_ROLES["verification"]
        )

    def test_lone_contributing_candidate_is_never_primary(self):
        # A failed task whose only candidate is a verification fault used to
        # normalize to weight 1.0 and seize primary — "contributing, never causal"
        # is now enforced.
        for cap, et in (
            ("code_verification", "relevant_test_not_run"),
            ("self_repair_loop", "repair_not_reverified"),
            ("tool_use", "tool_failure_not_retried"),
        ):
            b = build_blame(_det([_cand(cap, et)], loc_flag="clean"))
            self.assertEqual(b["status"], "conjunctive", cap)
            self.assertIsNone(b["primary"], cap)
            self.assertIn(cap, CONTRIBUTING)


class TestCapabilityLevelPrimaryGate(unittest.TestCase):
    def test_same_capability_split_is_primary_not_conjunctive(self):
        # relevant_change_omitted + incorrect_patch are BOTH code_editing: 0.5/0.5
        # per entry, but 1.0 of capability mass -> editing is the primary cause.
        b = build_blame(
            _det(
                [
                    _cand("code_editing", "relevant_change_omitted"),
                    _cand("code_editing", "incorrect_patch"),
                ],
                loc_flag="loc_partial",
            )
        )
        self.assertEqual(b["status"], "primary")
        self.assertEqual(b["primary"]["capability"], "code_editing")
        self.assertAlmostEqual(b["capability_blame"]["code_editing"], 1.0)

    def test_cross_capability_near_tie_stays_conjunctive(self):
        # loc primary role vs editing secondary with high conf: 0.71/0.29 -> primary;
        # but two equal-role capabilities (both secondary to a judge primary that
        # is absent) split the mass -> conjunctive.
        b = build_blame(
            _det(
                [
                    _cand("code_localization", "missed_relevant_file"),
                    _cand("code_editing", "incorrect_patch"),
                ],
                loc_flag="loc_miss",
            )
        )
        # localization is upstream-primary (role 1.0) vs editing secondary (0.4):
        # 0.714 vs 0.286 -> a genuine capability-level primary.
        self.assertEqual(b["status"], "primary")
        self.assertEqual(b["primary"]["capability"], "code_localization")


class TestBootstrapP(unittest.TestCase):
    def test_tiny_n_is_never_significant(self):
        rng = random.Random(0)
        vec_a = {f"i{k}": 1.0 for k in range(MIN_PAIRS - 1)}
        vec_b = {f"i{k}": 0.0 for k in range(MIN_PAIRS - 1)}
        r = paired_bootstrap_diff(vec_a, vec_b, rng)
        self.assertEqual(r["p_two_sided"], 1.0)

    def test_p_never_exactly_zero(self):
        rng = random.Random(0)
        vec_a = {f"i{k}": 1.0 for k in range(50)}
        vec_b = {f"i{k}": 0.0 for k in range(50)}
        r = paired_bootstrap_diff(vec_a, vec_b, rng)
        self.assertGreater(r["p_two_sided"], 0.0)
        self.assertLess(r["p_two_sided"], 0.01)


if __name__ == "__main__":
    unittest.main()
