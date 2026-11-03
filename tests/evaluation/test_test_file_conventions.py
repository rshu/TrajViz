"""ADDING_A_BENCHMARK.md §4.2: is_test_file recognizes JS/TS and Go test-file
conventions (SWE-bench Pro repositories) without loosening the Python rules;
the shipped SWE-bench Verified corpus replays with zero per-type change.

Run:  python3 -m unittest tests.test_test_file_conventions -v
"""

import unittest


from agent_trajviz.awe._engine.diffutil import is_test_file


class TestConventions(unittest.TestCase):
    def test_js_ts_go(self):
        for p in (
            "src/app.test.ts",
            "src/app.spec.js",
            "ui/Button.test.tsx",
            "lib/__tests__/x.js",
            "pkg/foo_test.go",
            "__tests__/setup.js",
        ):
            self.assertTrue(is_test_file(p), p)

    def test_python_rules_unchanged(self):
        for p in (
            "tests/test_a.py",
            "pkg/tests/x.py",
            "test/x.py",
            "testing/x.py",
            "conftest.py",
            "a/tests.py",
            "a/b_test.py",
        ):
            self.assertTrue(is_test_file(p), p)
        for p in (
            "django/test/testcases.py",
            "numpy/testing/utils.py",
            "latest.py",
            "src/testing.go",
            "src/contest.py",
            "pkg/utils_testing.py",
        ):
            self.assertFalse(is_test_file(p), p)


if __name__ == "__main__":
    unittest.main()
