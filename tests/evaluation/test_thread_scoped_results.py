import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, RUN_COMMAND, RUN_TEST, EDIT
from agent_trajviz.awe._engine.trajsignals import compute
from .test_review_fixes_trajectory import _call, _result, _gold


class TestThreadScopedResults(unittest.TestCase):
    def test_command_error_not_hidden_by_subagent_test_id(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "same", RUN_COMMAND),
                _result(1, "same", "command not found", error=True),
                _call(2, "same", RUN_TEST, test=True, sub="child"),
                _result(3, "same", "1 passed", sub="child"),
            ],
        )
        sig = compute(t, _gold())
        self.assertTrue(sig.tool_failure_not_retried)
        self.assertEqual(sig.n_tool_call_errors, 1)
        self.assertTrue(audit._redrv_tool_failure_not_retried(t))

    def test_subagent_command_cannot_change_main_read_scope(self):
        from agent_trajviz.awe._engine.trajectory import READ

        t = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "same", READ),
                _result(1, "same", "ImportError: source content"),
                _call(2, "same", RUN_COMMAND, sub="child"),
                _result(3, "same", "ok", sub="child"),
            ],
        )
        self.assertFalse(compute(t, _gold()).environment_error)

    def test_patch_error_audit_keeps_each_thread(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "x", EDIT),
                _result(1, "x", "Invalid patch: bad arguments", error=True),
                _call(2, "y", EDIT),
                _result(3, "y", "Invalid patch: bad arguments", error=True),
                _call(4, "x", RUN_TEST, test=True, sub="child"),
                _call(5, "y", RUN_TEST, test=True, sub="child"),
            ],
        )
        self.assertTrue(compute(t, _gold()).repeated_patch_apply_failure)
        self.assertTrue(audit._redrv_repeated_patch_apply_failure(t))
