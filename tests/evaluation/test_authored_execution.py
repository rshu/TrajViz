import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory
from agent_trajviz.awe._engine.trajsignals import compute
from .test_batch8_signals import _gold, _edit, _test_call, _result


class TestAuthoredExecution(unittest.TestCase):
    def check(self, steps, unrun):
        t = NormalizedTrajectory("a", "i", steps)
        s = compute(t, _gold())
        self.assertEqual(s.authored_test_never_run, unrun)
        self.assertEqual(audit._redrv_authored_test_never_run(t), unrun)
        return s

    def test_test_before_edit_does_not_verify_new_file(self):
        self.check(
            [_test_call(0, "pytest tests/test_repro.py"), _result(1, 0, 0), _edit(2, "tests/test_repro.py")], True
        )

    def test_dangling_selected_run_is_not_execution(self):
        self.check([_edit(0, "tests/test_repro.py"), _test_call(1, "pytest tests/test_repro.py")], True)

    def test_suite_output_can_prove_execution(self):
        s = self.check(
            [
                _edit(0, "tests/test_repro.py"),
                _test_call(1, "pytest"),
                _result(2, 1, 0, "tests/test_repro.py . [100%]\n1 passed"),
            ],
            False,
        )
        self.assertTrue(s.authored_test_ran_never_failed)

    def test_each_authored_file_needs_a_run(self):
        s = self.check(
            [
                _edit(0, "tests/test_first.py"),
                _edit(1, "tests/test_second.py"),
                _test_call(2, "pytest tests/test_first.py"),
                _result(3, 2, 0),
            ],
            True,
        )
        self.assertEqual(s.authored_test_unrun_files, ["tests/test_second.py"])
        self.assertFalse(s.authored_test_ran_never_failed)

    def test_collection_error_is_not_execution(self):
        self.check(
            [
                _edit(0, "tests/test_repro.py"),
                _test_call(1, "pytest tests/test_repro.py"),
                _result(2, 1, 2, "ERROR collecting tests/test_repro.py"),
            ],
            True,
        )
