"""Regression tests for the PR-47 adversarial-review fixes (v0.37):
fn_tokens file-stem leak, make-lookahead flag bypass + audit mirror sync,
file_loc duplicate-section summing, strength-tier of excessive_extra_edits,
read-time judge abstention, main-thread result-index scoping.

Run:  python3 -m unittest tests.test_pr47_review_fixes -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine import audit, config
from agent_trajviz.awe._engine.evidence import strength_of, strength_basis_for, build_chain, ASSOCIATIONAL
from agent_trajviz.awe._engine.judge import verdict_to_specs
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import _summed_loc
from agent_trajviz.awe._engine.diffutil import parse_unified_diff
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, RUN_COMMAND
from agent_trajviz.awe._engine.trajsignals import compute

from .test_batch8_signals import _step, _test_call, _cmd_call, _result, _gold

GATING_CMD = "pytest tests/test_core.py::test_gating_behavior"


class TestFnTokensStemLeak(unittest.TestCase):
    def test_progress_line_naming_only_the_file_is_not_execution_proof(self):
        # default pytest progress output names the FILE, not the function —
        # must not count as gating execution (the stem 'test_core' leaked
        # into fn_tokens before the fix).
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, "pytest"), _result(1, 0, 0, out="tests/test_core.py ........   [100%]")]
        )
        sig = compute(traj, _gold())
        self.assertFalse(sig.gating_exec_observed)

    def test_function_named_in_output_still_counts(self):
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, "pytest"), _result(1, 0, 1, out="FAILED tests/test_core.py::test_gating_behavior")]
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.gating_exec_observed)

    def test_id_without_function_part_contributes_no_token(self):
        gold = _gold(fail_to_pass=["tests/test_core.py"])  # file-level id
        traj = NormalizedTrajectory(
            "a", "i", [_test_call(0, "pytest"), _result(1, 0, 1, out="FAILED in tests/test_core.py")]
        )
        self.assertFalse(compute(traj, gold).gating_exec_observed)


class TestMakeLookaheadFlags(unittest.TestCase):
    def _retry(self, between_cmd):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, "pytest tests/test_core.py"),
                _result(1, 0, 1),
                _cmd_call(2, between_cmd),
                _result(3, 2, 0, out=""),
                _test_call(4, "pytest tests/test_core.py"),
                _result(5, 4, 1),
            ],
        )
        return compute(traj, _gold()).retry_without_change

    def test_flag_carrying_make_lint_no_longer_excuses(self):
        for cmd in ("make -j4 lint", "make -C src lint", "make V=1 lint", "make --silent lint"):
            self.assertTrue(self._retry(cmd), cmd)

    def test_bare_make_build_still_excuses(self):
        self.assertFalse(self._retry("make"))
        self.assertFalse(self._retry("make -j4 build"))

    def test_audit_mirror_agrees_with_detector(self):
        # honest retry_without_change fault (make lint between identical
        # failing runs) must audit re_derived, not mismatch — the stale
        # `make\b` mirror flagged the detector's own faults as forged.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, "pytest tests/test_core.py"),
                _result(1, 0, 1),
                _cmd_call(2, "make lint"),
                _result(3, 2, 0, out=""),
                _test_call(4, "pytest tests/test_core.py"),
                _result(5, 4, 1),
            ],
        )
        f = {
            "capability": "self_repair_loop",
            "error_type": "retry_without_change",
            "detector_evidence": {"retry_without_change": True},
        }
        r = audit.audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(r["verdict"], "re_derived", r["mismatches"])


class TestFileLocSummed(unittest.TestCase):
    def test_duplicate_sections_sum(self):
        text = (
            "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n"
            "@@ -1,1 +1,1 @@\n-a\n+b\n"
            "diff --git a/pkg/x.py b/pkg/x.py\n--- a/pkg/x.py\n+++ b/pkg/x.py\n"
            "@@ -5,1 +5,2 @@\n-c\n+d\n+e\n"
        )
        fl = _summed_loc(parse_unified_diff(text))
        self.assertEqual(fl["pkg/x.py"], 5)  # 2 + 3, not last-wins 3


class TestStrengthTier(unittest.TestCase):
    def test_excessive_extra_edits_is_associational(self):
        self.assertEqual(strength_of("code_editing", "excessive_extra_edits"), ASSOCIATIONAL)
        self.assertIn("calibration", strength_basis_for(ASSOCIATIONAL, "excessive_extra_edits"))

    def test_chain_has_links(self):
        gold = GoldReference(
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
        )
        cand = {
            "capability": "code_editing",
            "error_type": "excessive_extra_edits",
            "confidence": "medium",
            "evidence": {"extra_src_loc": 120, "extra_files": ["pkg/b.py"], "threshold": 50},
        }
        chain = build_chain(cand, {"signals": {}}, gold, {}, {}, 0.4, False)
        self.assertTrue(chain["links"])
        self.assertEqual(chain["strength"], ASSOCIATIONAL)


class TestReadTimeAbstention(unittest.TestCase):
    RECORD = {
        "model": "m",
        "verdict": {
            "requirement_understanding": {
                "error_type": "misread_requirement",
                "confidence": "medium",
                "evidence_steps": [3],
                "rationale": "r",
                "agreement": 0.6,
                "n_votes": 5,
            },
            "task_planning": {
                "error_type": "none",
                "confidence": "high",
                "evidence_steps": [],
                "rationale": "",
                "agreement": 1.0,
                "n_votes": 5,
            },
        },
    }

    def test_bar_applies_to_cached_panel_records(self):
        old = config.JUDGE_MIN_AGREE
        try:
            config.JUDGE_MIN_AGREE = 0.8
            self.assertEqual(verdict_to_specs(self.RECORD), [])
            config.JUDGE_MIN_AGREE = 0.5
            specs = verdict_to_specs(self.RECORD)
            self.assertEqual(len(specs), 1)
        finally:
            config.JUDGE_MIN_AGREE = old

    def test_single_vote_records_unaffected(self):
        rec = {
            "model": "m",
            "verdict": {
                "requirement_understanding": {
                    "error_type": "misread_requirement",
                    "confidence": "medium",
                    "evidence_steps": [3],
                    "rationale": "r",
                },
                "task_planning": {"error_type": "none", "confidence": "high"},
            },
        }
        old = config.JUDGE_MIN_AGREE
        try:
            config.JUDGE_MIN_AGREE = 1.0
            self.assertEqual(len(verdict_to_specs(rec)), 1)
        finally:
            config.JUDGE_MIN_AGREE = old


class TestMainThreadResultIndex(unittest.TestCase):
    def test_colliding_subagent_result_does_not_shadow_main(self):
        # main-thread gating call c1 FAILED; an appended sub-agent thread
        # reuses call_id c1 with a PASSING result — main-thread ordering must
        # read the main-thread result (global last-wins read the sub-agent's).
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _test_call(0, GATING_CMD),
                _result(1, 0, 1),
                _step(
                    2,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_COMMAND,
                    test_run=True,
                    call_id="c0",
                    args={"command": GATING_CMD},
                    subagent_id="s1",
                ),
                _result(3, 0, 0, sub="s1"),
            ],
        )
        sig = compute(traj, _gold())
        self.assertTrue(sig.last_gating_run_failed)
        self.assertTrue(sig.last_test_failed)


if __name__ == "__main__":
    unittest.main()
