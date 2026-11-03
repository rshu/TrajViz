"""Tests for the v0.35 gating-run-specific outcome signal (schema 14):
``gating_run_observed`` / ``last_gating_run_failed`` — the tight per-gating-run
outcome DESIGN.md §0ai reserved the environment fault_side arm on.

Run:  python3 -m unittest tests.test_gating_run_signal -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.faultside import assess
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, RUN_TEST
from agent_trajviz.awe._engine.trajsignals import compute, TrajectorySignals


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _test_call(i, cmd, sub=None):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=RUN_TEST,
        native_tool="Bash",
        test_run=True,
        call_id=f"c{i}",
        args={"command": cmd},
        subagent_id=sub,
    )


def _result(i, call_i, exit_code, sub=None):
    return _step(
        i,
        actor="tool",
        type="tool_result",
        call_id=f"c{call_i}",
        args={"exit_code": exit_code},
        output_excerpt="1 failed" if exit_code else "1 passed",
        subagent_id=sub,
    )


def _gold(**kw):
    d = dict(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_core.py"],
        fail_to_pass=["tests/test_core.py::test_gating"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
    )
    d.update(kw)
    return GoldReference(**d)


GATING_CMD = "pytest tests/test_core.py::test_gating"
OTHER_CMD = "pytest tests/test_other.py"


class TestGatingRunSignal(unittest.TestCase):
    def test_last_gating_run_failed(self):
        # gating run fails, then an UNRELATED test passes: last_test_failed is
        # False (any-kind looseness) while the gating-specific signal holds.
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, GATING_CMD), _result(1, 0, 1), _test_call(2, OTHER_CMD), _result(3, 2, 0)]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_run_observed)
        self.assertTrue(sig.last_gating_run_failed)
        self.assertFalse(sig.last_test_failed)

    def test_last_gating_run_green_with_unrelated_failure(self):
        # the §0ai divergence shape, tight form: gating run green, a later
        # unrelated test fails — last_test_failed True, gating signal green.
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, GATING_CMD), _result(1, 0, 0), _test_call(2, OTHER_CMD), _result(3, 2, 1)]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_run_observed)
        self.assertFalse(sig.last_gating_run_failed)
        self.assertTrue(sig.last_test_failed)

    def test_no_gating_named_run(self):
        traj = NormalizedTrajectory("a", "i", [_test_call(0, OTHER_CMD), _result(1, 0, 1)])
        sig = compute(traj, _gold())
        self.assertFalse(sig.gating_run_observed)
        self.assertFalse(sig.last_gating_run_failed)

    def test_subagent_gating_run_does_not_count(self):
        # ordering signals are main-thread only (the CC adapter appends
        # sub-agent steps after the main trajectory).
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, OTHER_CMD),
                _result(1, 0, 0),
                _test_call(2, GATING_CMD, sub="s1"),
                _result(3, 2, 1, sub="s1"),
            ],
        )
        sig = compute(traj, _gold())
        self.assertFalse(sig.gating_run_observed)

    def test_unobserved_result_is_not_observed(self):
        # dangling gating test call (no result): fail-closed, not observed.
        traj = NormalizedTrajectory("a", "i", [_test_call(0, GATING_CMD)])
        sig = compute(traj, _gold())
        self.assertFalse(sig.gating_run_observed)

    def test_last_gating_run_wins_over_earlier_gating_failure(self):
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, GATING_CMD), _result(1, 0, 1), _test_call(2, GATING_CMD), _result(3, 2, 0)]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_run_observed)
        self.assertFalse(sig.last_gating_run_failed)


class TestDetectorEvidence(unittest.TestCase):
    def _tsig(self, **kw):
        d = dict(
            n_steps=5,
            read_set=["pkg/a.py"],
            edit_set_traj=["pkg/a.py"],
            ran_any_test=True,
            fail_to_pass_referenced=True,
            claimed_success_final=True,
            claimed_success_any=True,
            repeated_edit_files=[],
            edits_per_file={},
            n_tool_errors=0,
            n_test_runs=2,
            hidden_reasoning=False,
            last_test_failed=True,
        )
        d.update(kw)
        return TrajectorySignals(**d)

    def _misinterpret_fault(self, tsig):
        gold = _gold()
        patch = AgentPatch(
            agent="a",
            instance_id="i",
            exists=True,
            edited_files=["pkg/a.py"],
            edited_source_files=["pkg/a.py"],
            edited_test_files=[],
            patch_loc=5,
            symbols=[],
            changed_lines={},
        )
        loc = localization_signals(gold, patch, resolved=False)
        cands = objective_candidates(gold, loc, tsig, False)
        return next((c for c in cands if c.error_type == "test_result_misinterpreted"), None)

    def test_evidence_carries_gating_specificity(self):
        c = self._misinterpret_fault(self._tsig(gating_run_observed=True, last_gating_run_failed=True))
        self.assertIsNotNone(c)
        self.assertTrue(c.evidence["gating_run_observed"])
        self.assertTrue(c.evidence["last_gating_run_failed"])

    def test_fire_condition_unchanged_without_gating_observation(self):
        # semantics changes are calibration-gated: the fault still fires on the
        # any-kind last_test_failed shape, with the specificity fields False.
        c = self._misinterpret_fault(self._tsig())
        self.assertIsNotNone(c)
        self.assertFalse(c.evidence["gating_run_observed"])
        self.assertFalse(c.evidence["last_gating_run_failed"])


class TestFaultSidePayload(unittest.TestCase):
    def test_payload_carries_gating_keys(self):
        out = assess(
            {
                "truncated_run": False,
                "ran_any_test": True,
                "last_test_failed": True,
                "gating_run_observed": True,
                "last_gating_run_failed": True,
            }
        )
        self.assertIn("gating_run_observed", out["signals"])
        self.assertIn("last_gating_run_failed", out["signals"])
        self.assertTrue(out["signals"]["last_gating_run_failed"])

    def test_missing_gating_keys_stay_none(self):
        out = assess({"truncated_run": False})
        self.assertIsNone(out["signals"]["gating_run_observed"])
        self.assertIsNone(out["signals"]["last_gating_run_failed"])


class TestAuditRederivation(unittest.TestCase):
    TEST_FILES = ["tests/test_core.py"]
    F2P = ["tests/test_core.py::test_gating"]

    def _fault(self, observed, failed):
        return {
            "capability": "code_verification",
            "error_type": "test_result_misinterpreted",
            "detector_evidence": {
                "claimed_success_final": True,
                "last_test_failed": True,
                "gating_run_observed": observed,
                "last_gating_run_failed": failed,
            },
        }

    def test_forged_gating_failure_is_caught(self):
        # trajectory: gating run PASSED (then an unrelated failure satisfies
        # last_test_failed) — a forged last_gating_run_failed must mismatch.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, GATING_CMD),
                _result(1, 0, 0),
                _test_call(2, OTHER_CMD),
                _result(3, 2, 1),
                _step(4, actor="agent", type="message", text="All tests pass — the fix is complete."),
            ],
        )
        r = audit.audit_fault(
            self._fault(True, True),
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=traj,
            test_files=self.TEST_FILES,
            fail_to_pass=self.F2P,
        )
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "last_gating_run_failed" for m in r["mismatches"]))

    def test_true_gating_failure_re_derives(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, GATING_CMD),
                _result(1, 0, 1),
                _step(2, actor="agent", type="message", text="All tests pass — the fix is complete."),
            ],
        )
        r = audit.audit_fault(
            self._fault(True, True),
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=traj,
            test_files=self.TEST_FILES,
            fail_to_pass=self.F2P,
        )
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_gating_atoms_without_inputs_unverifiable(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, GATING_CMD),
                _result(1, 0, 1),
                _step(2, actor="agent", type="message", text="All tests pass — the fix is complete."),
            ],
        )
        r = audit.audit_fault(
            self._fault(True, True), agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj
        )
        self.assertEqual(r["verdict"], "unverifiable")


if __name__ == "__main__":
    unittest.main()
