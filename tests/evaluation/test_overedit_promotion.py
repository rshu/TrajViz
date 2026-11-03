"""Tests for the v0.37 `excessive_extra_edits` promotion — the v0.9 over-edit
deferral executed with the robust absolute metric (extra source LOC in
non-gold files; RR 2.38, Holm p=0.005).

Run:  python3 -m unittest tests.test_overedit_promotion -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.detect import objective_candidates, OBJECTIVE_ERROR_SIGNALS
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.signals import localization_signals, OVEREDIT_EXTRA_LOC
from agent_trajviz.awe._engine.patches import AgentPatch


def _gold():
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=[],
        fail_to_pass=[],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": False, "task_planning": False},
    )


def _patch(extra_loc):
    files = ["pkg/a.py", "pkg/other.py"]
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=files,
        edited_source_files=files,
        edited_test_files=[],
        patch_loc=5 + extra_loc,
        symbols=[],
        file_loc={"pkg/a.py": 5, "pkg/other.py": extra_loc},
    )


def _cands(extra_loc, resolved=False):
    loc = localization_signals(_gold(), _patch(extra_loc), resolved)
    return loc, objective_candidates(_gold(), loc, None, resolved)


class TestDetector(unittest.TestCase):
    def test_fires_at_threshold(self):
        loc, cands = _cands(OVEREDIT_EXTRA_LOC)
        c = next(c for c in cands if c.error_type == "excessive_extra_edits")
        self.assertEqual(c.capability, "code_editing")
        self.assertEqual(c.confidence, "medium")
        self.assertEqual(c.evidence["extra_src_loc"], OVEREDIT_EXTRA_LOC)
        self.assertEqual(loc["extra_src_loc"], OVEREDIT_EXTRA_LOC)

    def test_below_threshold_silent(self):
        _loc, cands = _cands(OVEREDIT_EXTRA_LOC - 1)
        self.assertFalse(any(c.error_type == "excessive_extra_edits" for c in cands))

    def test_never_fires_on_solved(self):
        _loc, cands = _cands(500, resolved=True)
        self.assertEqual(cands, [])

    def test_signal_binding(self):
        self.assertEqual(OBJECTIVE_ERROR_SIGNALS["excessive_extra_edits"], "extra_src_loc")


GOLD_DIFF = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"


def _agent_diff(n_extra_lines):
    extra = "".join(f"+line{i} = {i}\n" for i in range(n_extra_lines))
    return (
        GOLD_DIFF + f"diff --git a/pkg/other.py b/pkg/other.py\n--- a/pkg/other.py\n"
        f"+++ b/pkg/other.py\n@@ -1,0 +1,{n_extra_lines} @@\n{extra}"
    )


class TestAudit(unittest.TestCase):
    def _fault(self, claimed_loc):
        return {
            "capability": "code_editing",
            "error_type": "excessive_extra_edits",
            "detector_evidence": {"extra_src_loc": claimed_loc, "threshold": OVEREDIT_EXTRA_LOC},
        }

    def test_true_fault_re_derives(self):
        r = audit.audit_fault(
            self._fault(60),
            agent="a",
            inst="i",
            gold_text=GOLD_DIFF,
            agent_text=_agent_diff(60),
            resolved=False,
            traj=None,
        )
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])

    def test_forged_loc_is_caught(self):
        # evidence claims 60 extra LOC; the raw diff has only 10 (below the
        # threshold) — both the value and the condition must mismatch.
        r = audit.audit_fault(
            self._fault(60),
            agent="a",
            inst="i",
            gold_text=GOLD_DIFF,
            agent_text=_agent_diff(10),
            resolved=False,
            traj=None,
        )
        self.assertEqual(r["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "extra_src_loc" for m in r["mismatches"]))

    def test_fault_on_solved_case_is_caught(self):
        r = audit.audit_fault(
            self._fault(60),
            agent="a",
            inst="i",
            gold_text=GOLD_DIFF,
            agent_text=_agent_diff(60),
            resolved=True,
            traj=None,
        )
        self.assertEqual(r["verdict"], "mismatch")


if __name__ == "__main__":
    unittest.main()
