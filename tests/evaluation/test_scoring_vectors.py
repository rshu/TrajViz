"""return_vectors invariant: the emitted per-instance vectors are EXACTLY the
values the score averages — score == 100*(1 - mean(vector)). Mocks the data layer
so no ARGUS tree is needed.

Run:  python3 -m unittest tests.test_scoring_vectors -v
"""

import unittest
from unittest import mock

from agent_trajviz.awe._engine import scoring


class TestReturnVectors(unittest.TestCase):
    def _run(self):
        insts = ["i1", "i2", "i3"]
        outcomes = {"i1": False, "i2": False, "i3": True}  # i3 resolved
        blame = {
            "i1": {
                "status": "primary",
                "blame": [],
                "primary": None,
                "capability_blame": {"code_localization": 1.0, "code_editing": 0.0},
            },
            "i2": {
                "status": "primary",
                "blame": [],
                "primary": None,
                "capability_blame": {"code_localization": 0.0, "code_editing": 1.0},
            },
            "i3": {"status": "resolved", "blame": [], "primary": None, "capability_blame": {}},
        }

        def fake_detect(i, a, **k):
            return {"opportunities": {"code_localization": True, "code_editing": True}, "candidates": [], "_inst": i}

        with (
            mock.patch.object(scoring, "build_gold_reference", lambda i: object()),
            mock.patch.object(scoring, "resolved", lambda a, i: outcomes[i]),
            mock.patch.object(scoring, "detect", fake_detect),
            mock.patch.object(scoring, "build_blame", lambda d, **k: blame[d["_inst"]]),
        ):
            return scoring.agent_capability_scores(
                "agent_x", instances=insts, use_trajectory=True, bootstrap=50, use_judge=False, return_vectors=True
            )

    def test_vectors_average_to_rate(self):
        res = self._run()
        self.assertIn("vectors", res)
        for cap, cv in res["capabilities"].items():
            v = res["vectors"][cap]
            mean = sum(v.values()) / len(v)
            self.assertAlmostEqual(cv["score"], 100 * (1 - mean), places=6)

    def test_vectors_keyed_by_instance(self):
        res = self._run()
        self.assertEqual(set(res["vectors"]["code_localization"]), {"i1", "i2", "i3"})
        self.assertEqual(res["vectors"]["code_localization"]["i1"], 1.0)
        self.assertEqual(res["vectors"]["code_editing"]["i2"], 1.0)

    def test_fail_vector_matches_failures(self):
        res = self._run()
        self.assertEqual(res["fail_vector"], {"i1": 1, "i2": 1, "i3": 0})
        self.assertEqual(sum(res["fail_vector"].values()), res["n_failed"])

    def test_default_omits_vectors(self):
        # backward-compat: without the flag, the output shape is unchanged
        insts = ["i1"]
        with (
            mock.patch.object(scoring, "build_gold_reference", lambda i: object()),
            mock.patch.object(scoring, "resolved", lambda a, i: False),
            mock.patch.object(
                scoring,
                "detect",
                lambda i, a, **k: {"opportunities": {"code_localization": True}, "candidates": [], "_inst": i},
            ),
            mock.patch.object(
                scoring,
                "build_blame",
                lambda d, **k: {
                    "status": "primary",
                    "blame": [],
                    "primary": None,
                    "capability_blame": {"code_localization": 1.0},
                },
            ),
        ):
            res = scoring.agent_capability_scores("a", instances=insts, use_judge=False, bootstrap=10)
        self.assertNotIn("vectors", res)
        self.assertNotIn("fail_vector", res)


if __name__ == "__main__":
    unittest.main()
