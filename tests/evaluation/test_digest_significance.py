"""The cross-agent matrix bolds best / italics worst ONLY when the paired,
Holm-corrected comparison is significant; otherwise the row is tagged 'ns'.
Synthetic D (no ARGUS data).

Run:  python3 -m unittest tests.test_digest_significance -v
"""

import unittest
from collections import Counter
from unittest import mock

from agent_trajviz.awe._engine import config, digest


def vec(val, n=30):
    return {f"i{j}": val for j in range(n)}


def _D():
    # 3 agents; Editing is cleanly separated, Verification is a dead tie.
    A = ["claude_code", "codex", "opencode_glm"]
    edit = {"claude_code": 90.0, "codex": 50.0, "opencode_glm": 10.0}
    edit_v = {"claude_code": vec(0.1), "codex": vec(0.5), "opencode_glm": vec(0.9)}
    ver_v = {a: vec(0.3) for a in A}
    D = {}
    for a in A:
        D[a] = {
            "scores": {"code_editing": edit[a], "code_verification": 70.0},
            "vectors": {"code_editing": edit_v[a], "code_verification": ver_v[a]},
            "fail_vector": vec(0.0),
            "failure_rate": 0.0,
            "allf": Counter(),
            "n_failed": 10,
        }
    return A, D


class TestSignificanceGating(unittest.TestCase):
    def test_matrix_marks_only_significant(self):
        A, D = _D()
        with mock.patch.object(config, "AGENTS", A):
            out = "\n".join(digest._performance_by_process(D))
        # Editing: best bolded, worst italicised
        self.assertIn("**90.0**", out)
        self.assertIn("_10.0_", out)
        # Verification: dead tie -> no markers anywhere on 70.0
        self.assertNotIn("**70.0**", out)
        self.assertNotIn("_70.0_", out)
        # the row carries an explicit 'ns' tag
        self.assertIn("ns", out)

    def test_per_process_read_suppresses_on_tie(self):
        A, D = _D()
        with mock.patch.object(config, "AGENTS", A):
            out = "\n".join(digest._performance_by_process(D))
        # the Verification read must NOT crown a winner
        self.assertNotIn("Verification** — strongest", out)


class TestHonestNotes(unittest.TestCase):
    def test_judge_yield_note_states_real_count(self):
        from collections import Counter

        A = ["claude_code", "codex"]
        D = {
            a: {
                "primary": Counter({"incomplete_requirement_coverage": 2}),
                "allf": Counter({"incomplete_requirement_coverage": 5, "flawed_plan": 3}),
                "n_failed": 100,
            }
            for a in A
        }
        with mock.patch.object(config, "AGENTS", A):
            note = digest._judge_yield_note(D)
        # primary judge-cap count = 2 per agent x2 = 4; total failures = 200
        self.assertIn("4", note)
        self.assertIn("200", note)

    def test_arbiter_summary_line_reports_refute_rate(self):
        A = ["claude_code", "codex"]
        D = {a: {"adj_total": 50, "adj_refute": 35} for a in A}
        with mock.patch.object(config, "AGENTS", A):
            line = digest._arbiter_summary_line(D)
        self.assertIn("100", line)  # 50+50 adjudicated
        self.assertIn("70", line)  # 70 refuted -> 70%

    def test_arbiter_summary_empty_when_no_arbiter(self):
        A = ["claude_code"]
        D = {"claude_code": {"adj_total": 0, "adj_refute": 0}}
        with mock.patch.object(config, "AGENTS", A):
            self.assertEqual(digest._arbiter_summary_line(D), "")


if __name__ == "__main__":
    unittest.main()
