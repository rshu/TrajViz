"""Tests for cohort-contrast discovery (behaviors associated with failure).

Pure-function tests on the statistics + feature mapping — no ARGUS data needed.

Run:  python3 -m unittest tests.test_cohort -v
"""

from __future__ import annotations

import random
import unittest


from agent_trajviz.awe._engine import cohort
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals


def _sig(**over):
    base = dict(
        n_steps=10,
        read_set=["a.py", "b.py"],
        edit_set_traj=["a.py"],
        ran_any_test=True,
        fail_to_pass_referenced=True,
        claimed_success_final=False,
        claimed_success_any=False,
        repeated_edit_files=[],
        edits_per_file={},
        n_tool_errors=0,
        n_test_runs=1,
        hidden_reasoning=False,
    )
    base.update(over)
    return TrajectorySignals(**base)


class TestContrast(unittest.TestCase):
    def test_clear_separation_is_significant(self):
        rng = random.Random(0)
        c = cohort.contrast([1.0] * 12, [0.0] * 12, rng, B=500)
        self.assertAlmostEqual(c["diff"], 1.0)
        self.assertLess(c["p"], 0.05)

    def test_identical_cohorts_not_significant(self):
        rng = random.Random(0)
        c = cohort.contrast([1, 0, 1, 0] * 5, [1, 0, 1, 0] * 5, rng, B=500)
        self.assertAlmostEqual(c["diff"], 0.0)
        self.assertGreater(c["p"], 0.2)

    def test_empty_cohort_is_not_separable(self):
        c = cohort.contrast([], [1.0, 0.0], random.Random(0), B=100)
        self.assertEqual(c["p"], 1.0)


class TestBinaryAssociation(unittest.TestCase):
    def test_risk_and_odds_ratio(self):
        # failed: 3 present / 1 absent ; solved: 1 present / 3 absent
        a = cohort.binary_association([1, 1, 1, 0], [1, 0, 0, 0])
        self.assertAlmostEqual(a["p_fail_if_present"], 0.75)
        self.assertAlmostEqual(a["p_fail_if_absent"], 0.25)
        self.assertAlmostEqual(a["risk_ratio"], 3.0)
        self.assertAlmostEqual(a["odds_ratio"], 9.0)

    def test_perfect_separation_is_data_not_missing(self):
        # Behavior present in every failure, absent in every success: PERFECTLY
        # predictive. RR is inf (review fix 1.16: 0.0/inf are data; only an empty
        # stratum is undefined). Odds ratio stays undefined (b=0, c=0).
        a = cohort.binary_association([1, 1], [0, 0])
        self.assertEqual(a["risk_ratio"], float("inf"))
        self.assertIsNone(a["odds_ratio"])

    def test_perfectly_protective_rr_is_zero(self):
        # Behavior present only in successes: RR = 0.0, not None.
        a = cohort.binary_association([0, 0], [1, 1])
        self.assertEqual(a["risk_ratio"], 0.0)


class TestFeaturesOf(unittest.TestCase):
    def test_hallucinated_path_from_subtypes(self):
        f = cohort.features_of(
            _sig(
                tool_error_subtypes={"invalid_target": 2},
                n_tool_calls=10,
                n_tool_call_errors=2,
                encountered_tool_error=True,
            )
        )
        self.assertEqual(f["hallucinated_path"], 1.0)
        self.assertEqual(f["encountered_tool_error"], 1.0)
        self.assertAlmostEqual(f["tool_error_rate"], 0.2)
        self.assertEqual(f["ran_no_test"], 0.0)  # ran_any_test=True

    def test_ran_no_test_inverts(self):
        f = cohort.features_of(_sig(ran_any_test=False))
        self.assertEqual(f["ran_no_test"], 1.0)


class TestDiscover(unittest.TestCase):
    def _rows(self):
        # 'hallucinated_path' strongly failure-associated; 'n_steps' constant (noise).
        rows = []
        for i in range(20):
            rows.append((False, {"hallucinated_path": 1.0 if i < 18 else 0.0, "n_steps": 10.0}))
        for i in range(20):
            rows.append((True, {"hallucinated_path": 1.0 if i < 3 else 0.0, "n_steps": 10.0}))
        return rows

    def test_failure_associated_feature_ranks_top_and_significant(self):
        d = cohort.discover(self._rows(), seed=0, B=800)
        top = d["associations"][0]
        self.assertEqual(top["feature"], "hallucinated_path")
        self.assertLess(top["p_holm"], 0.05)
        self.assertGreater(top["diff"], 0)  # more common in failures
        self.assertGreater(top["risk_ratio"], 2.0)  # binary effect size attached

    def test_constant_feature_not_significant(self):
        d = cohort.discover(self._rows(), seed=0, B=800)
        noise = next(a for a in d["associations"] if a["feature"] == "n_steps")
        self.assertGreater(noise["p_holm"], 0.05)

    def test_report_md_lists_the_behavior(self):
        d = cohort.discover(self._rows(), seed=0, B=800)
        d["agent"] = "x"
        md = cohort.cohort_report_md(d)
        self.assertIn("hallucinated_path", md)
        self.assertIn("failures", md)


if __name__ == "__main__":
    unittest.main()
