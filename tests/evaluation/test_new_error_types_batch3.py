"""Tests for the v0.24 third batch:

  hallucinated_path_unrecovered (tool_use error type) — phantom path reused
  region_overlap                (severity attribute)  — same file, disjoint region
  solved_unrelated_candidate    (signal)              — patch ∩ issue identifiers = ∅
  exploration_loop              (behavioral/cohort)   — re-read without edit

Run:  python3 -m unittest tests.test_new_error_types_batch3 -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.cohort import features_of, BINARY_FEATURES
from agent_trajviz.awe._engine.diffutil import parse_unified_diff
from agent_trajviz.awe._engine.evidence import region_overlap
from agent_trajviz.awe._engine.signals import localization_signals, _issue_code_tokens
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, READ
from agent_trajviz.awe._engine.trajsignals import compute


def _gold(files=("pkg/a.py",), statement="p", hunks=None):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement=statement,
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
        gold_hunks=hunks or {},
    )


def _patch(files=("pkg/b.py",), symbols=(), hunks=None):
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
        hunks=hunks or {},
    )


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


class TestRegionOverlap(unittest.TestCase):
    def test_hunk_ranges_parsed(self):
        fds = parse_unified_diff(
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            "@@ -10,3 +10,4 @@\n old\n-a\n+b\n+c\n old\n"
            "@@ -40,2 +41,2 @@\n-d\n+e\n old\n"
        )
        self.assertEqual(fds[0].hunks, [(10, 3), (40, 2)])

    def test_same_region_high_overlap(self):
        ov = region_overlap({"x.py": [(10, 4)]}, {"x.py": [(11, 2)]}, ["x.py"])
        self.assertEqual(ov["region_overlap"], 1.0)  # ±3 margin covers 10..13

    def test_disjoint_region_zero_overlap(self):
        ov = region_overlap({"x.py": [(10, 4)]}, {"x.py": [(400, 5)]}, ["x.py"])
        self.assertEqual(ov["region_overlap"], 0.0)

    def test_attribute_attached_to_editing_evidence(self):
        g = _gold(hunks={"pkg/a.py": [(10, 4)]})
        p = _patch(files=("pkg/a.py",), hunks={"pkg/a.py": [(400, 5)]})
        loc = localization_signals(g, p, resolved=False)
        self.assertEqual(loc["flag"], "right_place_wrong_change")
        cands = objective_candidates(g, loc, None, False)
        ip = next(c for c in cands if c.error_type == "incorrect_patch")
        self.assertEqual(ip.evidence["region_overlap"], 0.0)

    def test_audit_re_derives_region_overlap(self):
        gold_diff = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -10,2 +10,2 @@\n-x\n+y\n"
        agent_diff = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -400,2 +400,2 @@\n-p\n+q\n"
        fault = {
            "capability": "code_editing",
            "error_type": "incorrect_patch",
            "confidence": "high",
            "detector_evidence": {"hit": ["pkg/a.py"], "region_overlap": 0.0},
        }
        a = audit_fault(
            fault, agent="a", inst="i", gold_text=gold_diff, agent_text=agent_diff, resolved=False, traj=None
        )
        self.assertEqual(a["verdict"], RE_DERIVED)
        fault["detector_evidence"]["region_overlap"] = 0.9  # tamper
        a = audit_fault(
            fault, agent="a", inst="i", gold_text=gold_diff, agent_text=agent_diff, resolved=False, traj=None
        )
        self.assertEqual(a["verdict"], MISMATCH)


class TestSolvedUnrelatedSignal(unittest.TestCase):
    STATEMENT = (
        "Calling `set_signing_key` on JWTAuth raises TypeError. Steps: create AuthClient, call refresh_tokens()."
    )

    def test_issue_tokens_extracted(self):
        toks = _issue_code_tokens(self.STATEMENT)
        self.assertIn("set_signing_key", toks)
        self.assertIn("refresh_tokens", toks)
        self.assertIn("AuthClient", toks)
        self.assertNotIn("Calling", toks)  # plain English excluded

    def test_unrelated_patch_flagged(self):
        g = _gold(statement=self.STATEMENT)
        p = _patch(files=("pkg/csv_export.py",), symbols=["render_csv_rows"])
        loc = localization_signals(g, p, resolved=False, read_set=set())
        self.assertEqual(loc["issue_symbol_overlap"], [])
        self.assertTrue(loc["solved_unrelated_candidate"])
        cands = objective_candidates(g, loc, None, False)
        wf = next(c for c in cands if c.error_type == "wrong_file_localization")
        self.assertTrue(wf.evidence["solved_unrelated_candidate"])

    def test_related_patch_not_flagged(self):
        g = _gold(statement=self.STATEMENT)
        p = _patch(files=("pkg/jwt_auth.py",), symbols=["set_signing_key"])
        loc = localization_signals(g, p, resolved=False)
        self.assertIn("set_signing_key", loc["issue_symbol_overlap"])
        self.assertFalse(loc["solved_unrelated_candidate"])

    def test_hit_patch_never_flagged(self):
        # the flag targets total localization misses only
        g = _gold(statement=self.STATEMENT, files=("pkg/a.py",))
        p = _patch(files=("pkg/a.py",), symbols=["render_csv_rows"])
        loc = localization_signals(g, p, resolved=False)
        self.assertFalse(loc["solved_unrelated_candidate"])


class TestHallucinatedPathUnrecovered(unittest.TestCase):
    def _traj(self, second_path):
        # shell reads (RUN_COMMAND): unflagged errors are recovered from the
        # result text — an unflagged READ result is never text-scanned (v0.21
        # scoping), so scaffolds without error flags surface these via the shell.
        steps = []
        for k, path in enumerate(["pkg/ghost.py", second_path]):
            steps.append(
                _step(
                    2 * k,
                    actor="agent",
                    type="tool_call",
                    canonical_action="run_command",
                    files=[path],
                    args={"command": f"cat {path}"},
                    call_id=f"c{k}",
                )
            )
            steps.append(
                _step(
                    2 * k + 1,
                    actor="tool",
                    type="tool_result",
                    output_excerpt=f"cat: {path}: No such file or directory",
                    call_id=f"c{k}",
                )
            )
        steps.append(_step(4, actor="agent", type="message", text="hm"))
        return NormalizedTrajectory("a", "i", steps)

    def test_same_phantom_path_twice_fires(self):
        sig = compute(self._traj("pkg/ghost.py"), _gold())
        self.assertTrue(sig.hallucinated_path_unrecovered)

    def test_two_different_typos_do_not(self):
        sig = compute(self._traj("pkg/other_ghost.py"), _gold())
        self.assertFalse(sig.hallucinated_path_unrecovered)

    def test_audit_re_derives_and_tampers(self):
        fault = {
            "capability": "tool_use",
            "error_type": "hallucinated_path_unrecovered",
            "confidence": "medium",
            "detector_evidence": {"hallucinated_path_unrecovered": True},
        }
        a = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj("pkg/ghost.py")
        )
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=self._traj("pkg/other_ghost.py"),
        )
        self.assertEqual(a["verdict"], MISMATCH)


class TestExplorationLoop(unittest.TestCase):
    def _traj(self, n_reads, edited=False):
        steps = [
            _step(i, actor="agent", type="tool_call", canonical_action=READ, files=["pkg/a.py"]) for i in range(n_reads)
        ]
        if edited:
            steps.append(_step(n_reads, actor="agent", type="tool_call", canonical_action="edit", files=["pkg/a.py"]))
        return NormalizedTrajectory("a", "i", steps)

    def test_re_reads_without_edit_flag(self):
        sig = compute(self._traj(4), _gold())
        self.assertEqual(sig.max_reads_single_file, 4)
        self.assertTrue(sig.exploration_loop)

    def test_edited_file_does_not_count(self):
        sig = compute(self._traj(4, edited=True), _gold())
        self.assertFalse(sig.exploration_loop)

    def test_cohort_feature_registered(self):
        self.assertIn("exploration_loop", BINARY_FEATURES)
        f = features_of(compute(self._traj(4), _gold()))
        self.assertEqual(f["exploration_loop"], 1.0)


if __name__ == "__main__":
    unittest.main()
