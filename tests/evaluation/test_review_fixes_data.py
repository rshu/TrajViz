"""Regression tests for the 2026-07-06 review fixes — objective data layer.

Pins:
  1.17 headerless unified diffs parse; header-lookalike body lines don't corrupt
  1.12 (companion) multiset/indent behavior lives in test_grounding
  M5   django/test/ is source, not tests
  M6/F1 no_source_edit three-way split + gold gating-test tamper signal
  2.2  test_result_misinterpreted requires an observed failing run

Run:  python3 -m unittest tests.test_review_fixes_data -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.diffutil import parse_unified_diff, is_test_file
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals


def _gold(files=("pkg/a.py",), test_files=("tests/test_a.py",)):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=list(files),
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=list(test_files),
        fail_to_pass=["tests/test_a.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
        gold_changed_lines={},
    )


def _patch(files=(), test_files=(), exists=True, unparseable=False):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=exists,
        edited_files=list(files) + list(test_files),
        edited_source_files=list(files),
        edited_test_files=list(test_files),
        unparseable=unparseable,
        patch_loc=5,
        symbols=[],
        changed_lines={},
    )


class TestDiffParser(unittest.TestCase):
    def test_headerless_plain_diff_parses(self):
        text = "--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n context\n"
        fds = parse_unified_diff(text)
        self.assertEqual([f.path for f in fds], ["pkg/a.py"])
        self.assertEqual((fds[0].added, fds[0].removed), (1, 1))

    def test_git_diff_still_parses(self):
        text = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,1 +1,2 @@\n old\n+new\n"
        fds = parse_unified_diff(text)
        self.assertEqual([f.path for f in fds], ["pkg/a.py"])
        self.assertEqual(fds[0].added, 1)

    def test_header_lookalike_body_lines_do_not_corrupt(self):
        # A removed SQL comment line '--- x' and an added '+++ x' inside the hunk
        # body must be counted as changed lines, not consumed as file headers.
        text = (
            "diff --git a/q.sql b/q.sql\n--- a/q.sql\n+++ b/q.sql\n"
            "@@ -1,2 +1,2 @@\n"
            "--- legacy comment\n"
            "+++ new comment\n"
        )
        fds = parse_unified_diff(text)
        self.assertEqual(len(fds), 1)
        self.assertEqual(fds[0].path, "q.sql")  # not reassigned
        self.assertEqual((fds[0].added, fds[0].removed), (1, 1))

    def test_multiple_headerless_files(self):
        text = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n--- a/y.py\n+++ b/y.py\n@@ -1,1 +1,1 @@\n-c\n+d\n"
        self.assertEqual([f.path for f in parse_unified_diff(text)], ["x.py", "y.py"])


class TestIsTestFile(unittest.TestCase):
    def test_django_test_package_is_source(self):
        self.assertFalse(is_test_file("django/test/testcases.py"))
        self.assertFalse(is_test_file("django/test/utils.py"))

    def test_real_test_files_still_detected(self):
        for p in (
            "tests/test_debug.py",
            "pkg/tests/utils.py",
            "test_foo.py",
            "pkg/sub/conftest.py",
            "testing/example_scripts/x.py",
            "pkg/foo_test.py",
        ):
            self.assertTrue(is_test_file(p), p)

    def test_source_lookalikes_stay_source(self):
        for p in ("pkg/latest.py", "numpy/testing/utils.py", "pandas/testing.py"):
            self.assertFalse(is_test_file(p), p)


class TestPatchStateSplit(unittest.TestCase):
    def test_missing_empty_unparseable_test_only(self):
        g = _gold()
        cases = {
            "missing": _patch(exists=False),
            "unparseable": _patch(files=(), unparseable=True),
            "empty": _patch(files=()),
            "test_only": _patch(files=(), test_files=("tests/test_a.py",)),
        }
        for expect, p in cases.items():
            loc = localization_signals(g, p, resolved=False)
            self.assertEqual(loc["flag"], "no_source_edit", expect)
            self.assertEqual(loc["patch_state"], expect)

    def test_gold_gating_test_edit_is_flagged(self):
        g = _gold(test_files=("tests/test_a.py",))
        p = _patch(files=(), test_files=("tests/test_a.py",))
        loc = localization_signals(g, p, resolved=False)
        self.assertEqual(loc["gold_tests_edited"], ["tests/test_a.py"])
        cands = objective_candidates(g, loc, None, False)
        np = next(c for c in cands if c.error_type == "no_patch_produced")
        self.assertEqual(np.evidence["patch_state"], "test_only")
        self.assertEqual(np.evidence["gold_tests_edited"], ["tests/test_a.py"])


class TestMisinterpretedNeedsObservedFailure(unittest.TestCase):
    def _tsig(self, last_test_failed):
        return TrajectorySignals(
            n_steps=10,
            read_set=[],
            edit_set_traj=[],
            ran_any_test=True,
            fail_to_pass_referenced=True,
            claimed_success_final=True,
            claimed_success_any=True,
            repeated_edit_files=[],
            edits_per_file={},
            n_tool_errors=0,
            n_test_runs=2,
            hidden_reasoning=False,
            encountered_failure=last_test_failed,
            last_test_failed=last_test_failed,
        )

    def _loc(self):
        g = _gold()
        return g, localization_signals(g, _patch(files=("pkg/a.py",)), resolved=False)

    def test_claim_after_last_failing_run_is_misinterpretation(self):
        g, loc = self._loc()
        ets = {c.error_type for c in objective_candidates(g, loc, self._tsig(True), False)}
        self.assertIn("test_result_misinterpreted", ets)

    def test_claim_after_local_pass_is_not_misinterpretation(self):
        # Gating test referenced, last observed run passed, eval still failed:
        # local-vs-eval divergence, not a misread of test output (review 2.2).
        g, loc = self._loc()
        ets = {c.error_type for c in objective_candidates(g, loc, self._tsig(False), False)}
        self.assertNotIn("test_result_misinterpreted", ets)
        self.assertNotIn("unsupported_success_claim", ets)


if __name__ == "__main__":
    unittest.main()
