"""Tests for the v0.23 second-batch error types + the truncation guard:

  retry_without_change         (self_repair_loop) — blind identical re-run
  repeated_patch_apply_failure (tool_use)         — apply_patch loop
  authored_test_never_run      (code_verification) — wrote a test, never ran it
  truncated_run                 (signal)           — killed ≠ gave up

Run:  python3 -m unittest tests.test_new_error_types_batch2 -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, EDIT, RUN_COMMAND, RUN_TEST
from agent_trajviz.awe._engine.trajsignals import compute, TrajectorySignals


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


def _patch(files=("pkg/a.py",)):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=list(files),
        edited_source_files=list(files),
        edited_test_files=[],
        patch_loc=5,
        symbols=[],
        changed_lines={},
    )


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _test_call(i, cid, cmd="pytest tests/test_a.py"):
    return _step(
        i, actor="agent", type="tool_call", canonical_action=RUN_TEST, test_run=True, call_id=cid, args={"command": cmd}
    )


def _result(i, cid, out, error=False):
    return _step(i, actor="tool", type="tool_result", output_excerpt=out, tool_error=error, call_id=cid)


def _edit(i, f="pkg/a.py", cid=None):
    return _step(i, actor="agent", type="tool_call", canonical_action=EDIT, files=[f], call_id=cid)


class TestRetryWithoutChange(unittest.TestCase):
    def _traj(self, edit_between):
        steps = [_test_call(0, "t1"), _result(1, "t1", "1 failed")]
        if edit_between:
            steps += [_edit(2, cid="e1"), _result(3, "e1", "ok")]
        n = len(steps)
        steps += [
            _test_call(n, "t2"),
            _result(n + 1, "t2", "1 failed"),
            _step(n + 2, actor="agent", type="message", text="hmm"),
        ]
        return NormalizedTrajectory("a", "i", steps)

    def test_blind_retry_detected(self):
        sig = compute(self._traj(edit_between=False), _gold())
        self.assertTrue(sig.retry_without_change)

    def test_retry_after_edit_is_fine(self):
        sig = compute(self._traj(edit_between=True), _gold())
        self.assertFalse(sig.retry_without_change)

    def test_different_command_is_not_blind_retry(self):
        steps = [
            _test_call(0, "t1"),
            _result(1, "t1", "1 failed"),
            _test_call(2, "t2", cmd="pytest tests/test_a.py::test_x -vv"),
            _result(3, "t2", "1 failed"),
            _step(4, actor="agent", type="message", text="ok"),
        ]
        sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
        self.assertFalse(sig.retry_without_change)

    def test_audit_re_derives_and_tampers(self):
        fault = {
            "capability": "self_repair_loop",
            "error_type": "retry_without_change",
            "confidence": "medium",
            "detector_evidence": {"retry_without_change": True},
        }
        a = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(edit_between=False)
        )
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(edit_between=True)
        )
        self.assertEqual(a["verdict"], MISMATCH)


class TestRepeatedPatchApplyFailure(unittest.TestCase):
    def _traj(self, n_failures):
        steps, sid = [], 0
        for k in range(n_failures):
            steps += [_edit(sid, cid=f"e{k}"), _result(sid + 1, f"e{k}", "error: patch does not apply")]
            sid += 2
        steps.append(_step(sid, actor="agent", type="message", text="stuck"))
        return NormalizedTrajectory("a", "i", steps)

    def test_two_apply_failures_fire(self):
        self.assertTrue(compute(self._traj(2), _gold()).repeated_patch_apply_failure)

    def test_single_failure_does_not(self):
        self.assertFalse(compute(self._traj(1), _gold()).repeated_patch_apply_failure)

    def test_audit_re_derives_and_tampers(self):
        fault = {
            "capability": "tool_use",
            "error_type": "repeated_patch_apply_failure",
            "confidence": "medium",
            "detector_evidence": {"repeated_patch_apply_failure": True},
        }
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(2))
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(1))
        self.assertEqual(a["verdict"], MISMATCH)


class TestAuthoredTestNeverRun(unittest.TestCase):
    def _traj(self, ran_it):
        steps = [
            _edit(0, f="tests/test_repro.py", cid="e1"),
            _result(1, "e1", "ok"),
            _test_call(2, "t1", cmd="pytest tests/test_a.py"),
            _result(3, "t1", "1 passed"),
        ]
        if ran_it:
            steps += [_test_call(4, "t2", cmd="pytest tests/test_repro.py"), _result(5, "t2", "1 passed")]
        steps.append(_step(len(steps), actor="agent", type="message", text="done"))
        return NormalizedTrajectory("a", "i", steps)

    def test_authored_and_never_run_detected(self):
        sig = compute(self._traj(ran_it=False), _gold())
        self.assertEqual(sig.authored_test_files, ["tests/test_repro.py"])
        self.assertTrue(sig.authored_test_never_run)
        g = _gold()
        loc = localization_signals(g, _patch(), resolved=False)
        ets = {c.error_type for c in objective_candidates(g, loc, sig, False)}
        self.assertIn("authored_test_never_run", ets)

    def test_running_it_clears_the_signal(self):
        self.assertFalse(compute(self._traj(ran_it=True), _gold()).authored_test_never_run)

    def test_not_emitted_when_no_tests_ran_at_all(self):
        # relevant_test_not_run/high owns the no-tests case.
        sig = TrajectorySignals(
            n_steps=3,
            read_set=[],
            edit_set_traj=["tests/test_repro.py"],
            ran_any_test=False,
            fail_to_pass_referenced=False,
            claimed_success_final=False,
            claimed_success_any=False,
            repeated_edit_files=[],
            edits_per_file={},
            n_tool_errors=0,
            n_test_runs=0,
            hidden_reasoning=False,
            authored_test_files=["tests/test_repro.py"],
            authored_test_never_run=True,
        )
        g = _gold()
        loc = localization_signals(g, _patch(), resolved=False)
        ets = {c.error_type for c in objective_candidates(g, loc, sig, False)}
        self.assertNotIn("authored_test_never_run", ets)
        self.assertIn("relevant_test_not_run", ets)

    def test_audit_re_derives(self):
        fault = {
            "capability": "code_verification",
            "error_type": "authored_test_never_run",
            "confidence": "medium",
            "detector_evidence": {"authored_test_never_run": True},
        }
        a = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(ran_it=False)
        )
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(ran_it=True)
        )
        self.assertEqual(a["verdict"], MISMATCH)


class TestTruncationGuard(unittest.TestCase):
    def test_killed_run_is_not_flagged_not_retried(self):
        # error result, then the run is cut on a dangling call: no opportunity
        # to retry, so tool_failure_not_retried must NOT fire.
        steps = [
            _step(
                0,
                actor="agent",
                type="tool_call",
                canonical_action=RUN_COMMAND,
                call_id="c1",
                args={"command": "python repro.py"},
            ),
            _result(1, "c1", "bash: python: command not found"),
            _step(
                2,
                actor="agent",
                type="tool_call",
                canonical_action=RUN_COMMAND,
                call_id="c2",
                args={"command": "python3 repro.py"},
            ),
        ]
        sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
        self.assertTrue(sig.truncated_run)
        self.assertFalse(sig.tool_failure_not_retried)

    def test_killed_after_edit_is_not_unreverified_repair(self):
        steps = [_test_call(0, "t1"), _result(1, "t1", "1 failed"), _edit(2, cid="e1")]  # dangling edit: cut mid-repair
        sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
        self.assertTrue(sig.truncated_run)
        self.assertFalse(sig.repair_not_reverified)

    def test_completed_giveup_still_flagged(self):
        steps = [
            _test_call(0, "t1"),
            _result(1, "t1", "1 failed"),
            _edit(2, cid="e1"),
            _result(3, "e1", "ok"),
            _step(4, actor="agent", type="message", text="that should do it"),
        ]
        sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
        self.assertFalse(sig.truncated_run)
        self.assertTrue(sig.repair_not_reverified)


if __name__ == "__main__":
    unittest.main()
