"""Tests for the v0.39 `test_expectation_rewritten` signal chain — the calibrated
replacement for the demoted `gating_test_modified`:

  signals.localization_signals   assert_rewritten_files / _gold_tests / _other_tests
  trajsignals.compute            test_files_edited_after_failure, test_commands (schema 17)
  evidence.is_assertion_line     the shared classifier both layers use
  audit._audit_test_rewrite      independent re-derivation, multi-section aggregation

Run:  python3 -m unittest tests.test_test_rewrite -v
"""

from __future__ import annotations

import unittest
from collections import Counter


from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.evidence import is_assertion_line, norm_line
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, SCHEMA_VERSION
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals, compute


def _gold():
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_a.py"],
        fail_to_pass=["tests/test_a.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
        gold_changed_lines={},
    )


def _patch(changed: dict):
    tests = sorted(changed)
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=["pkg/a.py"] + tests,
        edited_source_files=["pkg/a.py"],
        edited_test_files=tests,
        patch_loc=5,
        symbols=[],
        changed_lines=changed,
    )


def _tsig(**kw):
    base = dict(
        n_steps=5,
        read_set=[],
        edit_set_traj=[],
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
    base.update(kw)
    return TrajectorySignals(**base)


REWRITE = {
    "added": Counter({norm_line("    assert f() == 3"): 1}),
    "removed": Counter({norm_line("    assert f() == 2"): 1}),
}
APPEND = {"added": Counter({norm_line("def test_new(): assert f() == 2"): 1}), "removed": Counter()}
DELETE_ONLY = {"added": Counter({norm_line("    x = 1"): 1}), "removed": Counter({norm_line("    assert f() == 2"): 1})}


class TestAssertionClassifier(unittest.TestCase):
    def test_idioms(self):
        for ln in (
            "assert x == 1",
            "    assert x",
            "self.assertEqual(a, b)",
            "assert_array_equal(a, b)",
            "np.testing.assert_allclose(a, b)",
            "with pytest.raises(ValueError):",
            "self.assertRaises(TypeError, f)",
            "expect(value).toBe(5)",
            norm_line("    assert x == 1"),
        ):
            self.assertTrue(is_assertion_line(ln), ln)
        for ln in (
            "x = 1",
            "def test_a():",
            "    return assert_helper",
            "# assert nothing",
            "assertion_count += 1",
            "",
        ):
            self.assertFalse(is_assertion_line(ln), ln)


class TestSignal(unittest.TestCase):
    def test_rewrite_in_gold_and_other_files(self):
        loc = localization_signals(_gold(), _patch({"tests/test_a.py": REWRITE, "tests/test_b.py": REWRITE}), False)
        self.assertEqual(loc["assert_rewritten_files"], ["tests/test_a.py", "tests/test_b.py"])
        self.assertEqual(loc["assert_rewritten_gold_tests"], ["tests/test_a.py"])
        self.assertEqual(loc["assert_rewritten_other_tests"], ["tests/test_b.py"])
        self.assertEqual(loc["assert_net_removed_files"], [])

    def test_append_and_delete_only_are_not_rewrites(self):
        loc = localization_signals(_gold(), _patch({"tests/test_a.py": APPEND}), False)
        self.assertEqual(loc["assert_rewritten_files"], [])
        loc = localization_signals(_gold(), _patch({"tests/test_a.py": DELETE_ONLY}), False)
        self.assertEqual(loc["assert_rewritten_files"], [])
        self.assertEqual(loc["assert_net_removed_files"], ["tests/test_a.py"])

    def test_legacy_set_fixtures_still_count(self):
        legacy = {"added": {norm_line("assert a == 2")}, "removed": {norm_line("assert a == 1")}}
        loc = localization_signals(_gold(), _patch({"tests/test_a.py": legacy}), False)
        self.assertEqual(loc["assert_rewritten_files"], ["tests/test_a.py"])

    def test_computed_on_solved_too(self):
        # the SIGNAL is outcome-blind (solve-side integrity audits consume it);
        # only the FAULT is outcome-conditioned.
        loc = localization_signals(_gold(), _patch({"tests/test_b.py": REWRITE}), True)
        self.assertEqual(loc["assert_rewritten_other_tests"], ["tests/test_b.py"])
        self.assertEqual(objective_candidates(_gold(), loc, _tsig(), True), [])


class TestDetectorAttributes(unittest.TestCase):
    def test_after_observed_failure_attribute(self):
        loc = localization_signals(_gold(), _patch({"tests/test_a.py": REWRITE}), False)
        t = _tsig(test_files_edited_after_failure=["tests/test_a.py"])
        c = next(
            c for c in objective_candidates(_gold(), loc, t, False) if c.error_type == "test_expectation_rewritten"
        )
        self.assertTrue(c.evidence["after_observed_failure"])
        self.assertEqual(c.evidence["rewritten_after_failure"], ["tests/test_a.py"])
        t = _tsig(test_files_edited_after_failure=["tests/test_zzz.py"])
        c = next(
            c for c in objective_candidates(_gold(), loc, t, False) if c.error_type == "test_expectation_rewritten"
        )
        self.assertFalse(c.evidence["after_observed_failure"])


def _step(i, **kw):
    base = dict(step_id=i, turn_id=i, actor="agent", type="tool_call")
    base.update(kw)
    return NormalizedStep(**base)


class TestTrajectorySignals(unittest.TestCase):
    def test_schema_bumped_for_new_fields(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 17)

    def test_test_files_edited_after_failure_and_commands(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(0, canonical_action="edit", files=["tests/test_a.py"]),  # before any failure
                _step(
                    1,
                    canonical_action="run_test",
                    test_run=True,
                    call_id="c1",
                    args={"command": "pytest tests/test_a.py::test_x"},
                ),
                _step(
                    2, actor="tool", type="tool_result", call_id="c1", args={"exit_code": 1}, output_excerpt="1 failed"
                ),
                _step(3, canonical_action="edit", files=["pkg/a.py"]),
                _step(4, canonical_action="edit", files=["tests/test_a.py"]),  # after the failure
                _step(
                    5,
                    canonical_action="run_test",
                    test_run=True,
                    call_id="c2",
                    args={"command": "pytest tests/test_a.py::test_x"},
                ),  # duplicate command
                _step(
                    6, actor="tool", type="tool_result", call_id="c2", args={"exit_code": 0}, output_excerpt="1 passed"
                ),
                _step(7, type="message", text="done"),
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.test_files_edited_after_failure, ["tests/test_a.py"])
        self.assertEqual(sig.test_commands, ["pytest tests/test_a.py::test_x"])  # deduped

    def test_no_failure_means_no_post_failure_edits(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(
                    0,
                    canonical_action="run_test",
                    test_run=True,
                    call_id="c1",
                    args={"command": "pytest tests/test_a.py"},
                ),
                _step(
                    1, actor="tool", type="tool_result", call_id="c1", args={"exit_code": 0}, output_excerpt="1 passed"
                ),
                _step(2, canonical_action="edit", files=["tests/test_a.py"]),
                _step(3, type="message", text="done"),
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.test_files_edited_after_failure, [])


class TestAudit(unittest.TestCase):
    ONE = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n"
        "+++ b/tests/test_a.py\n@@ -1,1 +1,1 @@\n-    assert f() == 2\n+    assert f() == 3\n"
    )
    # the removed and added assertion sit in DIFFERENT diff sections for the same
    # path (concatenated patches) — the auditor must aggregate per path
    SPLIT = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n"
        "+++ b/tests/test_a.py\n@@ -1,1 +1,0 @@\n-    assert f() == 2\n"
        "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n"
        "+++ b/tests/test_a.py\n@@ -5,0 +5,1 @@\n+    assert f() == 3\n"
    )
    SRC_ONLY = (
        "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n"
        "@@ -1,1 +1,1 @@\n-    assert x\n+    assert y\n"
    )

    def _audit(self, diff, files, test_files=("tests/test_a.py",)):
        fault = {
            "capability": "code_verification",
            "error_type": "test_expectation_rewritten",
            "confidence": "medium",
            "detector_evidence": {"assert_rewritten_files": files, "assert_rewritten_gold_tests": files},
        }
        return audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=diff,
            resolved=False,
            traj=None,
            test_files=list(test_files),
        )

    def test_re_derives_single_and_split_sections(self):
        self.assertEqual(self._audit(self.ONE, ["tests/test_a.py"])["verdict"], RE_DERIVED)
        self.assertEqual(self._audit(self.SPLIT, ["tests/test_a.py"])["verdict"], RE_DERIVED)

    def test_source_assertions_never_count(self):
        # assertion churn in SOURCE files is not a test-expectation rewrite
        a = self._audit(self.SRC_ONLY, ["pkg/a.py"])
        self.assertEqual(a["verdict"], MISMATCH)


if __name__ == "__main__":
    unittest.main()
