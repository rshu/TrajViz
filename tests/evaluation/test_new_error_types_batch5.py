"""Tests for v0.26 batch 5 — solved_unrelated_problem promotion (the calibrated
requirement-understanding anchor, RR=8.09) and the rejected anchors staying
signals-only.

Run:  python3 -m unittest tests.test_new_error_types_batch5 -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH, UNVERIFIABLE
from agent_trajviz.awe._engine.rootcause import build_blame, _needs_adjudication
from agent_trajviz.awe._engine.evidence import strength_of, issue_code_tokens, issue_has_repro
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory
from agent_trajviz.awe._engine.trajsignals import compute

STATEMENT = (
    "Calling `set_signing_key` on JWTAuth raises TypeError.\n"
    "```\n>>> client.refresh_tokens()\nTraceback (most recent call last):\n```"
)


def _gold(files=("pkg/auth.py",)):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement=STATEMENT,
        gold_files=list(files),
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_a.py"],
        fail_to_pass=["tests/test_a.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
        gold_changed_lines={},
    )


def _patch(files=("pkg/csv_export.py",), symbols=("render_csv_rows",)):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=list(files),
        edited_source_files=list(files),
        edited_test_files=[],
        patch_loc=5,
        symbols=list(symbols),
        changed_lines={},
    )


class TestPromotedAnchor(unittest.TestCase):
    def test_signal_and_blame_integration(self):
        loc = localization_signals(_gold(), _patch(), resolved=False)
        self.assertTrue(loc["solved_unrelated_candidate"])
        # medium confidence: contributes weight, arbiter-eligible, never primary
        # over a deductive fact (JUDGE_CAPS eligibility needs HIGH)
        b = build_blame(
            {
                "resolved": False,
                "agent": "a",
                "instance_id": "i",
                "loc_flag": "loc_miss",
                "candidates": [
                    {"capability": "code_localization", "error_type": "missed_relevant_file", "confidence": "high"},
                    {
                        "capability": "requirement_understanding",
                        "error_type": "solved_unrelated_problem",
                        "confidence": "medium",
                    },
                ],
            }
        )
        self.assertEqual(b["primary"]["capability"], "code_localization")
        req = next(x for x in b["blame"] if x["capability"] == "requirement_understanding")
        self.assertGreater(req["weight"], 0)
        self.assertTrue(_needs_adjudication(req))
        self.assertEqual(strength_of("requirement_understanding", "solved_unrelated_problem"), "associational")

    def test_audit_re_derives_and_tampers(self):
        agent_diff = (
            "diff --git a/pkg/csv_export.py b/pkg/csv_export.py\n"
            "--- a/pkg/csv_export.py\n+++ b/pkg/csv_export.py\n"
            "@@ -1,1 +1,2 @@\n old\n+def render_csv_rows(): pass\n"
        )
        gold_diff = (
            "diff --git a/pkg/auth.py b/pkg/auth.py\n--- a/pkg/auth.py\n+++ b/pkg/auth.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
        )
        fault = {
            "capability": "requirement_understanding",
            "error_type": "solved_unrelated_problem",
            "confidence": "medium",
            "detector_evidence": {"issue_symbol_overlap": [], "patch_symbols": ["render_csv_rows"]},
        }
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text=gold_diff,
            agent_text=agent_diff,
            resolved=False,
            traj=None,
            problem_statement=STATEMENT,
        )
        self.assertEqual(a["verdict"], RE_DERIVED)
        # tamper: the patch ACTUALLY shares an identifier with the issue
        related = agent_diff.replace("render_csv_rows", "set_signing_key")
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text=gold_diff,
            agent_text=related,
            resolved=False,
            traj=None,
            problem_statement=STATEMENT,
        )
        self.assertEqual(a["verdict"], MISMATCH)
        # no problem statement -> unverifiable, never a silent pass
        a = audit_fault(
            fault, agent="a", inst="i", gold_text=gold_diff, agent_text=agent_diff, resolved=False, traj=None
        )
        self.assertEqual(a["verdict"], UNVERIFIABLE)


class TestRejectedAnchorsStaySignals(unittest.TestCase):
    """Calibration rejected edited_before_any_run (RR 0.83) and repro_ignored
    (RR 0.95): they must exist as signals/features but never as candidates."""

    def test_edited_before_any_run_is_computed(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0, turn_id=0, actor="agent", type="tool_call", canonical_action="edit", files=["pkg/a.py"]
                ),
                NormalizedStep(
                    step_id=1,
                    turn_id=1,
                    actor="agent",
                    type="tool_call",
                    canonical_action="run_test",
                    test_run=True,
                    args={"command": "pytest tests/test_a.py"},
                ),
            ],
        )
        self.assertTrue(compute(traj, _gold()).edited_before_any_run)

    def test_issue_repro_detector(self):
        self.assertTrue(issue_has_repro(STATEMENT))
        self.assertFalse(issue_has_repro("plain prose description of a bug"))

    def test_never_emitted_as_error_types(self):
        from agent_trajviz.awe._engine.detect import produced_error_types

        produced = produced_error_types()
        self.assertIn("solved_unrelated_problem", produced)
        self.assertNotIn("no_reproduction_first", produced)
        self.assertNotIn("ignored_reproduction_steps", produced)

    def test_issue_tokens_shared_primitive(self):
        toks = issue_code_tokens(STATEMENT)
        self.assertIn("set_signing_key", toks)
        self.assertIn("refresh_tokens", toks)


if __name__ == "__main__":
    unittest.main()
