"""Tests for the v0.37 fragile-solve audit re-derivation (the §0ag deferral):
oracle-touch records must be independently re-derivable from the raw agent
diff + test_patch files, and a forged record must be caught.

Run:  python3 -m unittest tests.test_fragile_audit -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.audit import audit_fragile_solve

DESTR_DIFF = (
    "diff --git a/tests/test_a.py b/tests/test_a.py\n"
    "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
    "@@ -1,1 +1,1 @@\n-assert f() == 2\n+assert f() == 3\n"
)
TEST_FILES = ["tests/test_a.py"]


def _record(severity="high", modified=("tests/test_a.py",)):
    return {"edited_test_files": ["tests/test_a.py"], "modified_existing": list(modified), "severity": severity}


class TestFragileAudit(unittest.TestCase):
    def test_true_record_re_derives(self):
        r = audit_fragile_solve(_record(), DESTR_DIFF, TEST_FILES, True)
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_forged_severity_is_caught(self):
        # destructive diff claimed as benign append-only
        r = audit_fragile_solve(_record(severity="medium", modified=()), DESTR_DIFF, TEST_FILES, True)
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] in ("severity", "modified_existing") for m in r["mismatches"]))

    def test_record_on_failed_case_is_caught(self):
        # fragile records are solve-side only (failed cases belong to
        # test_expectation_rewritten / the gold_tests_edited signal)
        r = audit_fragile_solve(_record(), DESTR_DIFF, TEST_FILES, False)
        self.assertEqual(r["verdict"], "mismatch")

    def test_missing_inputs_unverifiable(self):
        r = audit_fragile_solve(_record(), None, TEST_FILES, True)
        self.assertEqual(r["verdict"], "unverifiable")


if __name__ == "__main__":
    unittest.main()
