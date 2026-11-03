"""Tests for the v0.36 batch-8 signals (schema 15): suite-wide test detection +
the medium relevant_test_not_run guard, gating execution proof, delegation
accounting, the weak-oracle repro flag, and the _ENV_REPAIR_RE make fix.

Run:  python3 -m unittest tests.test_batch8_signals -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    DELEGATE,
    RUN_COMMAND,
    RUN_TEST,
    EDIT,
    is_suite_wide_test_command,
)
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


def _cmd_call(i, cmd):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=RUN_COMMAND,
        native_tool="Bash",
        call_id=f"c{i}",
        args={"command": cmd},
    )


def _result(i, call_i, exit_code, out=None, sub=None, err=False):
    return _step(
        i,
        actor="tool",
        type="tool_result",
        call_id=f"c{call_i}",
        args={"exit_code": exit_code},
        tool_error=err,
        output_excerpt=out if out is not None else ("1 failed" if exit_code else "1 passed"),
        subagent_id=sub,
    )


def _edit(i, path):
    return _step(i, actor="agent", type="tool_call", canonical_action=EDIT, files=[path], call_id=f"c{i}", args={})


def _delegate(i, call_id=None):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=DELEGATE,
        native_tool="Task",
        call_id=call_id or f"c{i}",
        args={"subagent_type": "explore", "description": "look around"},
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
        fail_to_pass=["tests/test_core.py::test_gating_behavior"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
    )
    d.update(kw)
    return GoldReference(**d)


class TestSuiteWideClassifier(unittest.TestCase):
    # NB: `tox -e <env>` moved to NOT_SUITE in v0.37 — an -e env can be
    # lint/format and run no tests at all, so only generic tox counts.
    SUITE = [
        "pytest",
        "pytest -x -q",
        "py.test",
        "tox",
        "make test",
        "make check",
        "python -m pytest",
        "python -m unittest",
        "python -m unittest discover",
        "python setup.py test",
        "cd repo && pytest -q",
    ]
    NOT_SUITE = [
        "pytest tests/test_core.py",
        "pytest -k gating",
        "pytest -m slow",
        "pytest tests/test_core.py::test_x",
        "python -m unittest tests.test_core",
        "make build",
        "pytest -n auto",
        "grep pytest file.py",
        "ls",
        # v0.37 review fixes: no-run / subset / exclusion invocations
        "pytest --collect-only",
        "pytest --co",
        "pytest --lf",
        "pytest --last-failed",
        "pytest --ff",
        "pytest --ignore=tests/test_core.py",
        "pytest --deselect=tests/test_core.py::test_x",
        "tox -e lint",
        "tox -epy39",
    ]

    def test_suite_wide(self):
        for c in self.SUITE:
            self.assertTrue(is_suite_wide_test_command(c), c)

    def test_not_suite_wide(self):
        for c in self.NOT_SUITE:
            self.assertFalse(is_suite_wide_test_command(c), c)


class TestGatingExecProof(unittest.TestCase):
    def test_output_naming_gating_function_failed(self):
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, "pytest"), _result(1, 0, 1, out="FAILED tests/test_core.py::test_gating_behavior")]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_exec_observed)
        self.assertTrue(sig.last_gating_exec_failed)
        # command intent did NOT name the test — the tiers are independent
        self.assertFalse(sig.gating_run_observed)

    def test_quiet_green_output_proves_nothing(self):
        traj = NormalizedTrajectory("a", "i", [_test_call(0, "pytest"), _result(1, 0, 0, out="42 passed in 1.2s")])
        sig = compute(traj, _gold())
        self.assertFalse(sig.gating_exec_observed)

    def test_verbose_green_naming_the_test_is_the_affirmative_shape(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [_test_call(0, "pytest -v"), _result(1, 0, 0, out="tests/test_core.py::test_gating_behavior PASSED")],
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_exec_observed)
        self.assertFalse(sig.last_gating_exec_failed)


class TestDelegationSignals(unittest.TestCase):
    def test_counts_and_error_result(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _delegate(0),
                _result(1, 0, 0, out="sub-agent report", err=True),
                _delegate(2),
                _result(3, 2, 0, out="ok"),
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.n_delegations, 2)
        self.assertTrue(sig.delegate_error_result)
        self.assertFalse(sig.delegate_no_result)

    def test_missing_result_mid_run(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _delegate(0),  # no result ever arrives
                _cmd_call(5, "ls"),
                _result(6, 5, 0, out=""),
            ],
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.delegate_no_result)

    def test_cut_edge_dangling_delegate_is_truncation_not_loss(self):
        traj = NormalizedTrajectory("a", "i", [_cmd_call(0, "ls"), _result(1, 0, 0, out=""), _delegate(2)])
        sig = compute(traj, _gold())
        self.assertTrue(sig.truncated_run)
        self.assertFalse(sig.delegate_no_result)

    def test_subagent_steps_do_not_count_as_delegations(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _cmd_call(0, "ls"),
                _result(1, 0, 0, out=""),
                _delegate(2),
                _result(3, 2, 0, out="report"),
                _cmd_call(4, "ls"),
            ]
            + [
                _step(
                    10,
                    actor="agent",
                    type="tool_call",
                    canonical_action=DELEGATE,
                    call_id="c10",
                    args={},
                    subagent_id="s1",
                )
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.n_delegations, 1)


class TestWeakOracleRepro(unittest.TestCase):
    def test_authored_test_ran_and_never_failed(self):
        traj = NormalizedTrajectory(
            "a", "i", [_edit(0, "tests/test_repro.py"), _test_call(1, "pytest tests/test_repro.py"), _result(2, 1, 0)]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.authored_test_ran_never_failed)
        self.assertFalse(sig.authored_test_never_run)

    def test_failing_repro_exercised_the_oracle(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _edit(0, "tests/test_repro.py"),
                _test_call(1, "pytest tests/test_repro.py"),
                _result(2, 1, 1),
                _test_call(3, "pytest tests/test_repro.py"),
                _result(4, 3, 0),
            ],
        )
        sig = compute(traj, _gold())
        self.assertFalse(sig.authored_test_ran_never_failed)

    def test_never_run_stays_the_other_flag(self):
        traj = NormalizedTrajectory(
            "a", "i", [_edit(0, "tests/test_repro.py"), _test_call(1, "pytest tests/test_other.py"), _result(2, 1, 0)]
        )
        sig = compute(traj, _gold())
        self.assertFalse(sig.authored_test_ran_never_failed)
        self.assertTrue(sig.authored_test_never_run)


class TestEnvRepairMakeFix(unittest.TestCase):
    def _retry(self, between_cmd):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, "pytest tests/test_core.py"),
                _result(1, 0, 1),
                _cmd_call(2, between_cmd),
                _result(3, 2, 0, out=""),
                _test_call(4, "pytest tests/test_core.py"),
                _result(5, 4, 1),
            ],
        )
        return compute(traj, _gold()).retry_without_change

    def test_bare_make_still_excuses_the_retry(self):
        self.assertFalse(self._retry("make"))

    def test_make_lint_no_longer_excuses(self):
        self.assertTrue(self._retry("make lint"))


class TestSuiteWideGuardOnDetect(unittest.TestCase):
    def _tsig(self, **kw):
        d = dict(
            n_steps=5,
            read_set=["pkg/a.py"],
            edit_set_traj=["pkg/a.py"],
            ran_any_test=True,
            fail_to_pass_referenced=False,
            claimed_success_final=False,
            claimed_success_any=False,
            repeated_edit_files=[],
            edits_per_file={},
            n_tool_errors=0,
            n_test_runs=1,
            hidden_reasoning=False,
        )
        d.update(kw)
        return TrajectorySignals(**d)

    def _cands(self, tsig):
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
        return objective_candidates(gold, loc, tsig, False)

    def test_medium_arm_suppressed_by_suite_wide_run(self):
        cands = self._cands(self._tsig(ran_suite_wide=True))
        self.assertFalse(any(c.error_type == "relevant_test_not_run" for c in cands))

    def test_medium_arm_fires_without_suite_wide_run(self):
        cands = self._cands(self._tsig(ran_suite_wide=False))
        c = next(c for c in cands if c.error_type == "relevant_test_not_run")
        self.assertEqual(c.confidence, "medium")
        self.assertFalse(c.evidence["ran_suite_wide"])

    def test_high_arm_unaffected(self):
        cands = self._cands(self._tsig(ran_any_test=False, ran_suite_wide=False))
        c = next(c for c in cands if c.error_type == "relevant_test_not_run")
        self.assertEqual(c.confidence, "high")


class TestSuiteWideAudit(unittest.TestCase):
    def test_forged_ran_suite_wide_false_is_caught(self):
        # fault asserts no suite-wide run happened; the trajectory ran bare
        # pytest — the defining atom must mismatch.
        traj = NormalizedTrajectory("a", "i", [_test_call(0, "pytest"), _result(1, 0, 0)])
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": True, "fail_to_pass_referenced": False, "ran_suite_wide": False},
        }
        r = audit.audit_fault(
            f,
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=traj,
            test_files=["tests/test_core.py"],
            fail_to_pass=["tests/test_core.py::test_gating_behavior"],
        )
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "ran_suite_wide" for m in r["mismatches"]))

    def test_honest_ran_suite_wide_false_re_derives(self):
        traj = NormalizedTrajectory("a", "i", [_test_call(0, "pytest tests/test_other.py"), _result(1, 0, 0)])
        f = {
            "capability": "code_verification",
            "error_type": "relevant_test_not_run",
            "detector_evidence": {"ran_any_test": True, "fail_to_pass_referenced": False, "ran_suite_wide": False},
        }
        r = audit.audit_fault(
            f,
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=traj,
            test_files=["tests/test_core.py"],
            fail_to_pass=["tests/test_core.py::test_gating_behavior"],
        )
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])


if __name__ == "__main__":
    unittest.main()
