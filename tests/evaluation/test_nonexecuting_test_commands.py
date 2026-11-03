"""Runner metadata and collection do not verify the submitted source state."""

import unittest

from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    EDIT,
    classify_command,
    is_test_command,
    is_suite_wide_test_command,
)
from agent_trajviz.awe._engine.trajsignals import compute
from .test_batch8_signals import _gold


class TestNonexecutingCommands(unittest.TestCase):
    def test_introspection_and_collection_are_not_tests(self):
        for cmd in [
            "pytest --version",
            "pytest -V",
            "python -m pytest --help",
            "pytest --collect-only tests/test_core.py",
            "pytest --co",
            "pytest --collect-only=true",
            "tox --help",
            "which pytest && pytest --version",
            "make --dry-run test",
        ]:
            with self.subTest(cmd=cmd):
                self.assertFalse(is_test_command(cmd))
                self.assertFalse(is_suite_wide_test_command(cmd))

    def test_real_execution_in_sibling_segment_is_kept(self):
        self.assertTrue(is_test_command("pytest -v"))
        self.assertTrue(is_test_command("pytest --version && pytest tests/test_core.py"))
        self.assertTrue(is_suite_wide_test_command("pytest --version && pytest"))

    def test_collection_after_edit_does_not_close_verification_gap(self):
        cmd = "pytest --collect-only tests/test_core.py"
        steps = [
            NormalizedStep(0, 0, "agent", "tool_call", canonical_action=EDIT, files=["pkg/a.py"], call_id="e"),
            NormalizedStep(1, 0, "tool", "tool_result", call_id="e"),
            NormalizedStep(
                2,
                1,
                "agent",
                "tool_call",
                canonical_action=classify_command(cmd),
                test_run=is_test_command(cmd),
                args={"command": cmd},
                call_id="t",
            ),
            NormalizedStep(
                3, 1, "tool", "tool_result", call_id="t", args={"exit_code": 0}, output_excerpt="12 tests collected"
            ),
        ]
        sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
        self.assertFalse(sig.ran_any_test)
        self.assertFalse(sig.fail_to_pass_referenced)
        self.assertFalse(sig.encountered_failure)
        self.assertTrue(sig.no_test_after_final_edit)
