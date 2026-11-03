import difflib
import unittest
from agent_trajviz.awe._engine.oracles import oracle_candidates
from agent_trajviz.awe._engine.diffutil import parse_unified_diff


def patch(old, new, n=3):
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/tests/test_checks.py",
            tofile="b/tests/test_checks.py",
            n=n,
        )
    )


def fn(check):
    return "def test_check():\n    " + check + "\n"


class OracleCandidatesTests(unittest.TestCase):
    def candidates(self, old, new, n=3):
        return oracle_candidates(patch(old, new, n))["symptoms"]

    def test_literal_true_and_assertion_removal(self):
        for new, kind in [("assert True", "assertion_replaced_by_true"), ("pass", "assertion_removed_in_hunk")]:
            rows = self.candidates(fn("assert value == 2"), fn(new))
            self.assertEqual(rows[0]["evidence"]["change"], kind)
            self.assertEqual(rows[0]["scoring"], "not_scored")

    def test_numeric_relaxation_and_stricter_controls(self):
        for old, new in [("value < 5", "value < 10"), ("value > 5", "value > 1"), ("value < -5", "value <= -5")]:
            self.assertEqual(len(self.candidates(fn("assert " + old), fn("assert " + new))), 1)
            self.assertEqual(self.candidates(fn("assert " + new), fn("assert " + old)), [])
        self.assertEqual(self.candidates(fn("assert value == 2"), fn("assert value == 3")), [])
        self.assertEqual(self.candidates(fn("assert a < 5"), fn("assert b < 10")), [])

    def test_different_functions_and_assertion_relocation_abstain(self):
        old = fn("assert value == 2") + "\ndef test_other():\n    pass\n"
        new = fn("pass") + "\ndef test_other():\n    assert value == 2\n"
        self.assertEqual(self.candidates(old, new), [])
        self.assertEqual(self.candidates(fn("assert value == 2"), "def test_other():\n    assert True\n"), [])

    def test_multiple_changes_do_not_guess_pairing(self):
        self.assertEqual(
            self.candidates(fn("assert a < 5\n    assert b < 5"), fn("assert a < 10\n    assert b < 10")), []
        )

    def test_docstrings_and_nested_control_flow_are_not_assertions(self):
        self.assertEqual(self.candidates(fn('"""assert value < 5"""'), fn('"""assert value < 10"""')), [])
        self.assertEqual(
            self.candidates(fn("if flag:\n        assert value < 5"), fn("if flag:\n        assert value < 10")), []
        )

    def test_hunk_heading_is_not_a_function_definition(self):
        text = "--- a/tests/test_checks.py\n+++ b/tests/test_checks.py\n@@ -5 +5 @@ def test_check():\n-    assert value < 5\n+    assert value < 10\n"
        self.assertEqual(oracle_candidates(text)["symptoms"], [])
        self.assertGreater(oracle_candidates(text)["coverage"]["unknown_hunks_or_definitions"], 0)

    def test_truncated_hunk_and_non_python_abstain(self):
        p = patch(fn("assert value < 5"), fn("assert value < 10"))
        self.assertEqual(oracle_candidates(p.replace("-1,2 +1,2", "-1,3 +1,3"))["symptoms"], [])
        self.assertEqual(oracle_candidates(p.replace("test_checks.py", "test_checks.js"))["symptoms"], [])

    def test_parser_context_and_existing_counts_remain_consistent(self):
        p = patch(fn("assert value < 5"), fn("assert value < 10"))
        f = parse_unified_diff(p)[0]
        self.assertEqual((f.added, f.removed), (1, 1))
        self.assertTrue(f.hunk_contents[0]["complete"])
        self.assertEqual(f.hunk_contents[0]["old"], fn("assert value < 5").splitlines())
