"""The INSIGHTS Headline block: three data-derived bullets, each independently
guarded — absent when its data is, never hand-written. Synthetic D (no ARGUS).

Run:  python3 -m unittest tests.test_digest_headline -v
"""

import unittest
from collections import Counter
from unittest import mock

from agent_trajviz.awe._engine import config, digest


def _D(with_fault_side=True, split_bottleneck=False):
    A = ["claude_code", "codex"]
    D = {}
    for i, a in enumerate(A):
        bn = "code_localization" if (split_bottleneck and i == 1) else "code_editing"
        D[a] = {
            "n": 500,
            "n_failed": 100,
            "bottleneck": bn,
            "scores": {"code_editing": 80.0 + i, "code_localization": 90.0},
            "primary": Counter({"incorrect_patch": 60, "missed_relevant_file": 20}),
            "allf": Counter({"incorrect_patch": 60}),
            "failure_rate": 0.2,
            # keys the full build_insights_md path reads (headline ignores them)
            "cap_mass": Counter({"code_editing": 60.0}),
            "strength": {"deductive": 1.0, "associational": 0.0, "model_inferred": 0.0},
            "vectors": {},
            "fail_vector": {},
            "examples": {},
            "conjunctive": 0,
            "adj_total": 0,
            "adj_refute": 0,
            "hidden": False,
            "tool_behavior": {},
        }
        if with_fault_side:
            D[a]["fault_side"] = Counter({"model": 95, "grader": 5})
    return A, D


class TestHeadline(unittest.TestCase):
    def test_all_three_bullets_with_computed_numbers(self):
        A, D = _D()
        with mock.patch.object(config, "AGENTS", A):
            out = "\n".join(digest._headline_section(D))
        self.assertIn("## Headline", out)
        self.assertIn("2 agents × 500 tasks — 200 failed cases", out)
        self.assertIn("**Bottleneck:** Code Editing is every agent's weakest capability (scores 80.0–81.0).", out)
        # 120 incorrect_patch primaries over 200 failures = 60%
        self.assertIn("`incorrect_patch` — 120 primary cases (60% of failures)", out)
        self.assertIn("next: `missed_relevant_file` (40)", out)
        self.assertIn("190 of 200 failures are model-side; 10 (5.0%) are grader suspects", out)

    def test_split_bottleneck_says_k_of_n(self):
        A, D = _D(split_bottleneck=True)
        with mock.patch.object(config, "AGENTS", A):
            out = "\n".join(digest._headline_section(D))
        self.assertIn("the weakest capability for 1 of 2 agents", out)
        self.assertNotIn("every agent's weakest", out)

    def test_missing_fault_side_drops_only_that_bullet(self):
        A, D = _D(with_fault_side=False)
        with mock.patch.object(config, "AGENTS", A):
            out = "\n".join(digest._headline_section(D))
        self.assertNotIn("Repair locus", out)
        self.assertIn("**Bottleneck:**", out)
        self.assertIn("**Dominant root cause:**", out)

    def test_judge_free_run_omits_judge_note(self):
        A, D = _D()
        with mock.patch.object(config, "AGENTS", A), mock.patch.object(digest, "_judge_present", return_value=False):
            out = "\n".join(digest._headline_section(D))
        self.assertNotIn("judge:", out)

    def test_headline_placed_before_coverage(self):
        A, D = _D()
        with (
            mock.patch.object(config, "AGENTS", A),
            mock.patch.object(digest, "_judge_present", return_value=False),
            mock.patch.object(
                digest, "_significance", return_value={"_overall": {"strong_sig": False, "weak_sig": False}}
            ),
        ):
            md = digest.build_insights_md(D)
        self.assertLess(md.index("## Headline"), md.index("## Coverage"))


if __name__ == "__main__":
    unittest.main()


class TestEvalPlusCoverageNote(unittest.TestCase):
    """v0.39: the eval+ coverage caveat is emitted only for MIXED per-test
    coverage — never for full, none, or a digest that predates the field."""

    def _note(self, cov):
        A, D = _D()
        for a, c in cov.items():
            if c is not None:
                D[a]["pertest_cov"] = c
        with mock.patch.object(config, "AGENTS", A):
            return digest._evalplus_coverage_note(D), "\n".join(digest._headline_section(D))

    def test_partial_coverage_is_named(self):
        note, out = self._note({"claude_code": 97, "codex": 0})
        self.assertIsNotNone(note)
        self.assertIn("`claude_code` (97/100 failed)", note)
        self.assertIn("`codex` carry none", note)
        for t in digest._EVALPLUS_TYPES:
            self.assertIn(f"`{t}`", note)
        self.assertIn("eval+ coverage is partial", out)

    def test_full_or_no_coverage_is_silent(self):
        self.assertIsNone(self._note({"claude_code": 100, "codex": 100})[0])
        self.assertIsNone(self._note({"claude_code": 0, "codex": 0})[0])

    def test_pre_v039_digest_is_silent(self):
        note, out = self._note({"claude_code": None, "codex": None})
        self.assertIsNone(note)
        self.assertNotIn("eval+ coverage", out)
