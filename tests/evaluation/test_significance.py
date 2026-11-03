"""Paired significance layer — pure-function tests (no ARGUS data needed).

Run:  python3 -m unittest tests.test_significance -v
"""

import random
import unittest

from agent_trajviz.awe._engine import significance as sig


def vec(val, n=40, start=0):
    return {f"i{j}": val for j in range(start, start + n)}


class TestPairedBootstrapDiff(unittest.TestCase):
    def test_identical_vectors_not_separable(self):
        rng = random.Random(0)
        r = sig.paired_bootstrap_diff(vec(0.3), vec(0.3), rng, B=200)
        self.assertEqual(r["n_pairs"], 40)
        self.assertAlmostEqual(r["diff"], 0.0, places=9)
        self.assertEqual(r["p_two_sided"], 1.0)

    def test_clear_separation_is_significant(self):
        rng = random.Random(0)
        r = sig.paired_bootstrap_diff(vec(0.1), vec(0.8), rng, B=200)
        self.assertAlmostEqual(r["diff"], -0.7, places=9)
        self.assertLess(r["p_two_sided"], 0.05)

    def test_pairs_only_on_shared_keys(self):
        rng = random.Random(0)
        a = {"i1": 0.0, "i2": 0.0, "i3": 0.0}
        b = {"i2": 1.0, "i3": 1.0, "i4": 1.0}
        r = sig.paired_bootstrap_diff(a, b, rng, B=100)
        self.assertEqual(r["n_pairs"], 2)  # only i2, i3 overlap

    def test_empty_intersection_not_separable(self):
        rng = random.Random(0)
        r = sig.paired_bootstrap_diff({"a": 0.0}, {"b": 1.0}, rng, B=100)
        self.assertEqual(r["n_pairs"], 0)
        self.assertEqual(r["p_two_sided"], 1.0)

    def test_determinism_same_seed(self):
        r1 = sig.paired_bootstrap_diff(vec(0.2), vec(0.5), random.Random(7), B=200)
        r2 = sig.paired_bootstrap_diff(vec(0.2), vec(0.5), random.Random(7), B=200)
        self.assertEqual(r1, r2)


class TestHolm(unittest.TestCase):
    def test_known_values(self):
        adj = sig.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03, places=9)
        self.assertAlmostEqual(adj[1], 0.06, places=9)
        self.assertAlmostEqual(adj[2], 0.06, places=9)

    def test_empty(self):
        self.assertEqual(sig.holm([]), [])

    def test_clamped_to_one(self):
        self.assertTrue(all(p <= 1.0 for p in sig.holm([0.9, 0.9, 0.9])))


class TestRankWithSignificance(unittest.TestCase):
    def test_picks_best_and_worst_by_score(self):
        rng = random.Random(0)
        scores = {"a": 90.0, "b": 50.0, "c": 10.0}  # higher score = better
        vectors = {"a": vec(0.1), "b": vec(0.5), "c": vec(0.9)}
        r = sig.rank_with_significance(scores, vectors, rng, B=200)
        self.assertEqual(r["best"], "a")
        self.assertEqual(r["worst"], "c")
        self.assertLess(r["p_best"], 0.05)  # a clearly below b
        self.assertLess(r["p_worst"], 0.05)  # c clearly above b

    def test_single_agent_row(self):
        rng = random.Random(0)
        r = sig.rank_with_significance({"a": 90.0}, {"a": vec(0.1)}, rng, B=50)
        self.assertEqual(r["best"], "a")
        self.assertEqual(r["worst"], "a")
        self.assertEqual(r["p_best"], 1.0)


if __name__ == "__main__":
    unittest.main()
