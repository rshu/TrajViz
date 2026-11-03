"""Shell syntax must not turn quoted search patterns into executed tests."""

import unittest

from agent_trajviz.awe._engine.trajectory import command_read_targets, is_suite_wide_test_command, is_test_command


class TestShellCommandParsing(unittest.TestCase):
    def test_quoted_searches_are_not_tests(self):
        commands = [
            'python -m pip list | rg "numpy|pytest|wheel"',
            'rg -n "editable|python -m pytest|tests" README.md',
            "env | rg 'PYTHON|PYTEST|PATH'",
            'echo "make test"',
            "grep 'setup.py test' README.md",
            "python -c 'print(\"hello; pytest\")'",
            r"echo hello\;pytest",
        ]
        for cmd in commands:
            with self.subTest(cmd=cmd):
                self.assertFalse(is_test_command(cmd))
                self.assertFalse(is_suite_wide_test_command(cmd))

    def test_actual_shell_lists_still_execute_tests(self):
        for cmd in ["cd repo && pytest -q", "echo 'a|b'; python -m pytest", "printf done & pytest", "cd repo\npytest"]:
            with self.subTest(cmd=cmd):
                self.assertTrue(is_test_command(cmd))
                self.assertTrue(is_suite_wide_test_command(cmd))

    def test_framework_entry_points(self):
        for cmd in [
            "python manage.py test app",
            "python setup.py test",
            "django-admin test",
            "hatch run test",
            "bin/test",
        ]:
            self.assertTrue(is_test_command(cmd), cmd)

    def test_malformed_quote_is_not_a_test(self):
        self.assertFalse(is_test_command('echo "pytest | pytest'))

    def test_read_filename_can_contain_spaces_and_operators(self):
        self.assertEqual(command_read_targets('cat "pkg/a b.py"'), ["pkg/a b.py"])
        self.assertEqual(command_read_targets('cat "pkg/a|b.py"'), ["pkg/a|b.py"])
