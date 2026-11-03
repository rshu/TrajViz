"""Tests for the discovered-mode detectors (agent_trajviz.awe._engine.behaviors) + the env-error scan.

Pure-function tests + a compute() integration — no ARGUS data, no API key.

Run:  python3 -m unittest tests.test_behaviors -v
"""

from __future__ import annotations

import json
import unittest


from agent_trajviz.awe._engine import behaviors
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    looks_like_environment_error,
    RUN_TEST,
)
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals, compute
from agent_trajviz.awe._engine.gold import GoldReference


def _sig(**over):
    base = dict(
        n_steps=10,
        read_set=["a.py", "b.py", "c.py"],
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
    base.update(over)
    return TrajectorySignals(**base)


class TestEnvErrorDetector(unittest.TestCase):
    def test_positives(self):
        for s in [
            "ModuleNotFoundError: No module named 'numpy'",
            "ImportError: cannot import name 'X' from 'y'",
            "ERROR: Failed building wheel for scipy",
            "bash: pytest: command not found",
            "ERROR: Could not find a version that satisfies the requirement foo",
        ]:
            self.assertTrue(looks_like_environment_error(s), f"missed: {s!r}")

    def test_negatives(self):
        # a real test failure / code bug / clean output is NOT an environment failure
        for s in [
            "",
            "AssertionError: 1 != 2",
            "def foo():\n    return 1",
            "3 passed in 0.12s",
            "FAILED tests/test_x.py::test_y - ValueError",
        ]:
            self.assertFalse(looks_like_environment_error(s), f"false positive: {s!r}")


class TestAnalysisParalysis(unittest.TestCase):
    """Uses the AUTHORITATIVE patch flag, not the trajectory edit-set (some scaffolds
    don't surface edits in the trajectory)."""

    def test_investigated_via_reads_no_patch_fires(self):
        self.assertTrue(
            behaviors.analysis_paralysis(_sig(read_set=["a", "b", "c"], n_test_runs=0), produced_source_patch=False)
        )

    def test_investigated_via_test_no_patch_fires(self):
        self.assertTrue(behaviors.analysis_paralysis(_sig(read_set=["a"], n_test_runs=2), produced_source_patch=False))

    def test_produced_a_patch_does_not_fire(self):
        self.assertFalse(behaviors.analysis_paralysis(_sig(read_set=["a", "b", "c"]), produced_source_patch=True))

    def test_gave_up_early_does_not_fire(self):
        # minimal investigation (read < 3, no test) + no patch -> not paralysis
        self.assertFalse(behaviors.analysis_paralysis(_sig(read_set=["a"], n_test_runs=0), produced_source_patch=False))


class TestEnvironmentBlocked(unittest.TestCase):
    def test_reads_sig_field(self):
        self.assertTrue(behaviors.environment_error(_sig(environment_error=True)))
        self.assertFalse(behaviors.environment_error(_sig(environment_error=False)))

    def test_behavior_flags(self):
        f = behaviors.behavior_flags(
            _sig(read_set=["a", "b", "c"], environment_error=True), produced_source_patch=False
        )
        self.assertEqual(f, {"analysis_paralysis": True, "environment_error": True})


class TestComputeIntegration(unittest.TestCase):
    def _gold(self):
        return GoldReference(
            instance_id="i",
            repo="r",
            base_commit="c",
            problem_statement="p",
            gold_files=["pkg/a.py"],
            gold_symbols=[],
            gold_patch_loc=10,
            test_patch_files=[],
            fail_to_pass=["pkg/tests/test_x.py::test_x"],
            pass_to_pass=[],
            is_multi_file=False,
            is_cross_module=False,
            modules=["pkg"],
        )

    def test_env_error_in_test_output_sets_flag(self):
        # env errors surface DURING test runs — which the tool-error layer excludes — so
        # the env scan must cover all result outputs, including RUN_TEST results.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="t1",
                ),
                NormalizedStep(
                    step_id=1,
                    turn_id=1,
                    actor="tool",
                    type="tool_result",
                    output_excerpt="ModuleNotFoundError: No module named 'foo'",
                    call_id="t1",
                ),
            ],
        )
        sig = compute(traj, self._gold())
        self.assertTrue(sig.environment_error)
        # survives the cache round-trip (signals_cached reconstructs via TrajectorySignals(**json))
        again = TrajectorySignals(**json.loads(json.dumps(sig.to_dict())))
        self.assertTrue(again.environment_error)

    def test_clean_run_no_env_flag(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="t1",
                ),
                NormalizedStep(
                    step_id=1,
                    turn_id=1,
                    actor="tool",
                    type="tool_result",
                    output_excerpt="1 failed, 3 passed",
                    call_id="t1",
                ),
            ],
        )
        self.assertFalse(compute(traj, self._gold()).environment_error)


if __name__ == "__main__":
    unittest.main()
