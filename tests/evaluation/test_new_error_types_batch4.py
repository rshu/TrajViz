"""Batch-4a tests, re-based on v0.39: the destructiveness split of the demoted
gating_test_modified became the assertion-REWRITE anchor of
test_expectation_rewritten — an existing assertion replaced by a new one fires
(medium); an append-only repro addition to the same gold gating file does not
(the flat fact stays the gold_tests_edited signal).

Run:  python3 -m unittest tests.test_new_error_types_batch4 -v
"""

from __future__ import annotations

import unittest
from collections import Counter


from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals


def _gold():
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
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


def _patch(test_changed):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=["pkg/a.py", "tests/test_a.py"],
        edited_source_files=["pkg/a.py"],
        edited_test_files=["tests/test_a.py"],
        patch_loc=5,
        symbols=[],
        changed_lines={"tests/test_a.py": test_changed},
    )


def _tsig():
    return TrajectorySignals(
        n_steps=5,
        read_set=[],
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


APPEND_ONLY = {"added": Counter({"0|def test_new(): assert f() == 2": 1}), "removed": Counter()}
DESTRUCTIVE = {"added": Counter({"0|assert f() == 3": 1}), "removed": Counter({"0|assert f() == 2": 1})}


class TestAssertionRewriteSplit(unittest.TestCase):
    def _cands(self, changed):
        loc = localization_signals(_gold(), _patch(changed), resolved=False)
        return loc, objective_candidates(_gold(), loc, _tsig(), False)

    def test_append_only_is_signal_only(self):
        loc, cands = self._cands(APPEND_ONLY)
        self.assertEqual(loc["gold_tests_edited"], ["tests/test_a.py"])
        self.assertEqual(loc["gold_tests_modified_existing"], [])
        self.assertEqual(loc["assert_rewritten_files"], [])
        self.assertFalse(any(c.error_type in ("test_expectation_rewritten", "gating_test_modified") for c in cands))

    def test_assertion_rewrite_fires_medium(self):
        loc, cands = self._cands(DESTRUCTIVE)
        c = next(c for c in cands if c.error_type == "test_expectation_rewritten")
        self.assertEqual(c.confidence, "medium")
        self.assertEqual(c.evidence["assert_rewritten_gold_tests"], ["tests/test_a.py"])
        self.assertEqual(c.evidence["assert_rewritten_other_tests"], [])
        self.assertEqual(loc["gold_tests_modified_existing"], ["tests/test_a.py"])


class TestAssertionRewriteAudit(unittest.TestCase):
    DESTR_DIFF = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n"
        "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
        "@@ -1,1 +1,1 @@\n-assert f() == 2\n+assert f() == 3\n"
    )
    ADD_DIFF = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n"
        "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
        "@@ -1,0 +1,1 @@\n+def test_new(): assert f() == 2\n"
    )

    def _audit(self, diff, claimed_rewritten):
        fault = {
            "capability": "code_verification",
            "error_type": "test_expectation_rewritten",
            "confidence": "medium",
            "detector_evidence": {
                "assert_rewritten_files": claimed_rewritten,
                "assert_rewritten_gold_tests": claimed_rewritten,
            },
        }
        return audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=diff,
            resolved=False,
            traj=None,
            test_files=["tests/test_a.py"],
        )

    def test_rewrite_re_derives(self):
        self.assertEqual(self._audit(self.DESTR_DIFF, ["tests/test_a.py"])["verdict"], RE_DERIVED)

    def test_append_only_claimed_as_rewrite_mismatches(self):
        # claiming an append-only patch rewrote an assertion must fail the audit
        self.assertEqual(self._audit(self.ADD_DIFF, ["tests/test_a.py"])["verdict"], MISMATCH)

    def test_legacy_destructiveness_fault_still_audits(self):
        fault = {
            "capability": "code_verification",
            "error_type": "gating_test_modified",
            "confidence": "high",
            "detector_evidence": {
                "gold_tests_edited": ["tests/test_a.py"],
                "gold_tests_modified_existing": ["tests/test_a.py"],
            },
        }
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=self.DESTR_DIFF,
            resolved=False,
            traj=None,
            test_files=["tests/test_a.py"],
        )
        self.assertEqual(a["verdict"], RE_DERIVED)


if __name__ == "__main__":
    unittest.main()


class TestScoutConsolidation(unittest.TestCase):
    """Batch 4b: LLM label-consolidation pass (deterministic mock client)."""

    def _novel(self):
        return [
            {
                "label": "analysis-loop",
                "count": 5,
                "examples": ["a/i1"],
                "description": "re-diagnoses, never patches",
                "alt_descriptions": [],
            },
            {
                "label": "analysis-without-action-loop",
                "count": 3,
                "examples": ["a/i2"],
                "description": "keeps analyzing",
                "alt_descriptions": [],
            },
            {
                "label": "environment-blocked",
                "count": 2,
                "examples": ["a/i3"],
                "description": "setup failure",
                "alt_descriptions": [],
            },
        ]

    def test_mock_consolidation_merges_semantic_duplicates(self):
        from agent_trajviz.awe._engine.scout import consolidate_novel, mock_consolidate_client

        out = consolidate_novel(self._novel(), client=mock_consolidate_client)
        self.assertEqual(len(out), 2)
        top = out[0]
        self.assertEqual(top["count"], 8)  # 5 + 3 merged
        self.assertEqual(sorted(top["members"]), ["analysis-loop", "analysis-without-action-loop"])
        self.assertEqual(out[1]["label"], "environment-blocked")

    def test_garbage_response_falls_back_to_unconsolidated(self):
        from agent_trajviz.awe._engine.scout import consolidate_novel

        def garbage(*a, **k):
            return {"content": "not json at all"}

        out = consolidate_novel(self._novel(), client=garbage)
        self.assertEqual(len(out), 3)  # untouched

    def test_invented_members_are_ignored_and_dropped_become_singletons(self):
        from agent_trajviz.awe._engine.scout import consolidate_novel

        def evil(*a, **k):
            return {"content": '{"groups": [{"canonical": "x", "members": ["analysis-loop", "made-up-label"]}]}'}

        out = consolidate_novel(self._novel(), client=evil)
        labels = {g["label"] for g in out}
        self.assertNotIn("made-up-label", str(out))
        # the two labels the model dropped come back as singletons
        self.assertIn("analysis-without-action-loop", labels)
        self.assertIn("environment-blocked", labels)
