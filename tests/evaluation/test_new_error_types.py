"""Tests for the v0.22 first-batch error types:

  gave_up_early           (code_editing)      — no patch AND no investigation
  no_test_after_final_edit (code_verification) — verified early, edited after
  test_expectation_rewritten (code_verification) — v0.39: replaced
  gating_test_modified (whose fact calibrated flat); existing assertions in a
  test file were replaced by new ones

Covers detection (incl. the boundaries with no_patch_produced,
relevant_test_not_run and repair_not_reverified), blame integration, evidence
strength, catalog grounding, and audit re-derivation + tamper.

Run:  python3 -m unittest tests.test_new_error_types -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.detect import objective_candidates, expected_evidence_signal
from agent_trajviz.awe._engine.rootcause import build_blame, _needs_adjudication
from agent_trajviz.awe._engine.evidence import strength_of, build_chain
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH, UNVERIFIABLE
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals, compute


def _gold(files=("pkg/a.py",), test_files=("tests/test_a.py",)):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=list(files),
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=list(test_files),
        fail_to_pass=["tests/test_a.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
        gold_changed_lines={},
    )


def _patch(files=(), test_files=(), exists=True):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=exists,
        edited_files=list(files) + list(test_files),
        edited_source_files=list(files),
        edited_test_files=list(test_files),
        patch_loc=5,
        symbols=[],
        changed_lines={},
    )


def _tsig(**kw):
    base = dict(
        n_steps=5,
        read_set=[],
        edit_set_traj=[],
        ran_any_test=False,
        fail_to_pass_referenced=False,
        claimed_success_final=False,
        claimed_success_any=False,
        repeated_edit_files=[],
        edits_per_file={},
        n_tool_errors=0,
        n_test_runs=0,
        hidden_reasoning=False,
    )
    base.update(kw)
    return TrajectorySignals(**base)


class TestGaveUpEarly(unittest.TestCase):
    def _ets(self, t):
        g = _gold()
        loc = localization_signals(g, _patch(exists=False), resolved=False, read_set=set(t.read_set))
        return {c.error_type for c in objective_candidates(g, loc, t, False)}

    def test_no_patch_no_investigation_is_gave_up_early(self):
        ets = self._ets(_tsig(read_set=["pkg/a.py"]))
        self.assertIn("gave_up_early", ets)
        self.assertNotIn("no_patch_produced", ets)

    def test_investigated_no_patch_stays_no_patch_produced(self):
        for t in (_tsig(ran_any_test=True, n_test_runs=1), _tsig(read_set=["a.py", "b.py", "c.py"])):
            ets = self._ets(t)
            self.assertIn("no_patch_produced", ets)
            self.assertNotIn("gave_up_early", ets)

    def test_no_trajectory_stays_no_patch_produced(self):
        g = _gold()
        loc = localization_signals(g, _patch(exists=False), resolved=False)
        ets = {c.error_type for c in objective_candidates(g, loc, None, False)}
        self.assertIn("no_patch_produced", ets)
        self.assertNotIn("gave_up_early", ets)

    def test_private_gating_ids_do_not_create_test_selection_fault(self):
        g = _gold()
        g.gating_tests_visible = False
        t = _tsig(ran_any_test=True, n_test_runs=2, fail_to_pass_referenced=False)
        loc = localization_signals(g, _patch(files=["pkg/a.py"]), resolved=False, read_set={"pkg/a.py"})
        ets = {c.error_type for c in objective_candidates(g, loc, t, False)}
        self.assertNotIn("relevant_test_not_run", ets)

    def test_blame_and_adjudication(self):
        b = build_blame(
            {
                "resolved": False,
                "agent": "a",
                "instance_id": "i",
                "loc_flag": "no_source_edit",
                "candidates": [{"capability": "code_editing", "error_type": "gave_up_early", "confidence": "high"}],
            }
        )
        self.assertEqual(b["status"], "primary")
        self.assertTrue(_needs_adjudication(b["primary"]))
        self.assertEqual(strength_of("code_editing", "gave_up_early"), "associational")

    def test_audit_re_derives_and_tampers(self):
        traj = NormalizedTrajectory(
            "a", "i", [NormalizedStep(step_id=0, turn_id=0, actor="agent", type="message", text="cannot do this")]
        )
        fault = {
            "capability": "code_editing",
            "error_type": "gave_up_early",
            "confidence": "high",
            "detector_evidence": {"not_investigated": True, "patch_state": "missing"},
        }
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(a["verdict"], RE_DERIVED)
        self.assertEqual(a["tier"], "associational")
        # tamper: plant the fault on a run that DID investigate
        busy = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=i, turn_id=i, actor="agent", type="tool_call", canonical_action="read", files=[f"f{i}.py"]
                )
                for i in range(4)
            ],
        )
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=busy)
        self.assertEqual(a["verdict"], MISMATCH)


class TestNoTestAfterFinalEdit(unittest.TestCase):
    def _ets(self, t):
        g = _gold()
        loc = localization_signals(g, _patch(files=("pkg/a.py",)), resolved=False)
        return {c.error_type for c in objective_candidates(g, loc, t, False)}

    def test_fires_when_verified_early_edited_after(self):
        t = _tsig(ran_any_test=True, n_test_runs=1, no_test_after_final_edit=True)
        self.assertIn("no_test_after_final_edit", self._ets(t))

    def test_deferred_to_repair_not_reverified(self):
        t = _tsig(
            ran_any_test=True,
            n_test_runs=1,
            no_test_after_final_edit=True,
            encountered_failure=True,
            repair_not_reverified=True,
        )
        self.assertNotIn("no_test_after_final_edit", self._ets(t))

    def test_not_emitted_when_no_tests_at_all(self):
        t = _tsig(ran_any_test=False, no_test_after_final_edit=True)
        ets = self._ets(t)
        self.assertNotIn("no_test_after_final_edit", ets)
        self.assertIn("relevant_test_not_run", ets)  # the no-tests case is covered

    def test_signal_computed_from_ordering(self):
        # test -> edit -> final message (a COMPLETED run: a dangling trailing
        # edit would be a truncated run, which the killed≠gave-up guard exempts):
        # signal True; test -> edit -> test: False
        def steps(with_final_test):
            s = [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action="run_test",
                    test_run=True,
                    call_id="t1",
                    args={"command": "pytest tests/test_a.py"},
                ),
                NormalizedStep(
                    step_id=1, turn_id=1, actor="tool", type="tool_result", output_excerpt="1 passed", call_id="t1"
                ),
                NormalizedStep(
                    step_id=2,
                    turn_id=2,
                    actor="agent",
                    type="tool_call",
                    canonical_action="edit",
                    files=["pkg/a.py"],
                    call_id="e1",
                ),
                NormalizedStep(
                    step_id=3, turn_id=3, actor="tool", type="tool_result", output_excerpt="ok", call_id="e1"
                ),
            ]
            if with_final_test:
                s.append(
                    NormalizedStep(
                        step_id=4,
                        turn_id=4,
                        actor="agent",
                        type="tool_call",
                        canonical_action="run_test",
                        test_run=True,
                        call_id="t2",
                        args={"command": "pytest tests/test_a.py"},
                    )
                )
                s.append(
                    NormalizedStep(
                        step_id=5, turn_id=5, actor="tool", type="tool_result", output_excerpt="1 passed", call_id="t2"
                    )
                )
            else:
                s.append(NormalizedStep(step_id=4, turn_id=4, actor="agent", type="message", text="done"))
            return NormalizedTrajectory("a", "i", s)

        self.assertTrue(compute(steps(False), _gold()).no_test_after_final_edit)
        self.assertFalse(compute(steps(True), _gold()).no_test_after_final_edit)

    def test_truncated_run_is_exempt(self):
        # killed right after the edit (dangling final tool_call): no fault
        s = [
            NormalizedStep(
                step_id=0,
                turn_id=0,
                actor="agent",
                type="tool_call",
                canonical_action="run_test",
                test_run=True,
                call_id="t1",
                args={"command": "pytest tests/test_a.py"},
            ),
            NormalizedStep(
                step_id=1, turn_id=1, actor="tool", type="tool_result", output_excerpt="1 passed", call_id="t1"
            ),
            NormalizedStep(
                step_id=2, turn_id=2, actor="agent", type="tool_call", canonical_action="edit", files=["pkg/a.py"]
            ),
        ]
        sig = compute(NormalizedTrajectory("a", "i", s), _gold())
        self.assertTrue(sig.truncated_run)
        self.assertFalse(sig.no_test_after_final_edit)

    def test_audit_re_derives(self):
        # COMPLETED run (final message follows the edit) — a dangling trailing
        # edit would be truncation, which both detector and auditor now exempt.
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0, turn_id=0, actor="agent", type="tool_call", canonical_action="run_test", test_run=True
                ),
                NormalizedStep(
                    step_id=1, turn_id=1, actor="agent", type="tool_call", canonical_action="edit", files=["pkg/a.py"]
                ),
                NormalizedStep(step_id=2, turn_id=2, actor="agent", type="message", text="submitting"),
            ],
        )
        fault = {
            "capability": "code_verification",
            "error_type": "no_test_after_final_edit",
            "confidence": "medium",
            "detector_evidence": {"ran_any_test": True, "no_test_after_final_edit": True},
        }
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(a["verdict"], RE_DERIVED)
        # tamper: the same fault planted on a TRUNCATED run must mismatch
        cut = NormalizedTrajectory("a", "i", traj.steps[:2])
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=cut)
        self.assertEqual(a["verdict"], MISMATCH)


_REWRITE = {"added": {"4|assert f() == 3": 1}, "removed": {"4|assert f() == 2": 1}}


def _rewrite_patch():
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=["pkg/a.py", "tests/test_a.py"],
        edited_source_files=["pkg/a.py"],
        edited_test_files=["tests/test_a.py"],
        patch_loc=5,
        symbols=[],
        changed_lines={"tests/test_a.py": _REWRITE},
    )


class TestTestExpectationRewritten(unittest.TestCase):
    AGENT_DIFF = (
        "diff --git a/tests/test_a.py b/tests/test_a.py\n"
        "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
        "@@ -1,1 +1,1 @@\n-    assert f() == 2\n+    assert f() == 3\n"
    )

    def test_detected_on_failed_case(self):
        g = _gold()
        loc = localization_signals(g, _rewrite_patch(), resolved=False)
        cands = objective_candidates(g, loc, _tsig(ran_any_test=True), False)
        gm = [c for c in cands if c.error_type == "test_expectation_rewritten"]
        self.assertEqual(len(gm), 1)
        self.assertEqual(gm[0].evidence["assert_rewritten_files"], ["tests/test_a.py"])
        self.assertEqual(gm[0].evidence["assert_rewritten_gold_tests"], ["tests/test_a.py"])
        self.assertEqual(gm[0].capability, "code_verification")
        self.assertEqual(gm[0].confidence, "medium")
        # the demoted fact is still surfaced as a signal
        self.assertEqual(loc["gold_tests_edited"], ["tests/test_a.py"])
        self.assertNotIn("gating_test_modified", {c.error_type for c in cands})

    def test_touching_a_gold_test_alone_is_not_a_fault(self):
        # v0.39: the pre-existing gating_test_modified fact (any edit to a gold
        # gating test) calibrated flat (RR 1.01) — it is a signal, never a fault.
        g = _gold()
        p = _patch(files=("pkg/a.py",), test_files=("tests/test_a.py",))
        loc = localization_signals(g, p, resolved=False)
        cands = objective_candidates(g, loc, _tsig(ran_any_test=True), False)
        ets = {c.error_type for c in cands}
        self.assertNotIn("gating_test_modified", ets)
        self.assertNotIn("test_expectation_rewritten", ets)
        self.assertEqual(loc["gold_tests_edited"], ["tests/test_a.py"])

    def test_not_emitted_in_trajectory_free_slice(self):
        # The fact needs only patch+gold, but without a trajectory the scoring
        # slice is loc/edit-only and the verification mass would silently vanish
        # from every vector (the judge-spec vanishing-mass failure).
        g = _gold()
        loc = localization_signals(g, _rewrite_patch(), resolved=False)
        ets = {c.error_type for c in objective_candidates(g, loc, None, False)}
        self.assertNotIn("test_expectation_rewritten", ets)
        self.assertEqual(loc["assert_rewritten_files"], ["tests/test_a.py"])  # signal stays

    def test_never_primary(self):
        b = build_blame(
            {
                "resolved": False,
                "agent": "a",
                "instance_id": "i",
                "loc_flag": "clean",
                "candidates": [
                    {
                        "capability": "code_verification",
                        "error_type": "test_expectation_rewritten",
                        "confidence": "medium",
                    }
                ],
            }
        )
        self.assertEqual(b["status"], "conjunctive")

    def test_audit_re_derives_and_tampers(self):
        fault = {
            "capability": "code_verification",
            "error_type": "test_expectation_rewritten",
            "confidence": "medium",
            "detector_evidence": {
                "assert_rewritten_files": ["tests/test_a.py"],
                "assert_rewritten_gold_tests": ["tests/test_a.py"],
            },
        }
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=self.AGENT_DIFF,
            resolved=False,
            traj=None,
            test_files=["tests/test_a.py"],
        )
        self.assertEqual(a["verdict"], RE_DERIVED, a["mismatches"])
        # tamper: fault claims a file the patch never touched
        fault["detector_evidence"]["assert_rewritten_files"] = ["tests/test_b.py"]
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=self.AGENT_DIFF,
            resolved=False,
            traj=None,
            test_files=["tests/test_b.py"],
        )
        self.assertEqual(a["verdict"], MISMATCH)
        # no test_patch available -> the gold-arm atom is unverifiable, never a silent pass
        fault["detector_evidence"]["assert_rewritten_files"] = ["tests/test_a.py"]
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text=self.AGENT_DIFF, resolved=False, traj=None)
        self.assertEqual(a["verdict"], UNVERIFIABLE)

    def test_legacy_gating_fault_still_audits(self):
        # persisted pre-v0.39 dossiers carry gating_test_modified faults; the
        # legacy auditor keeps re-deriving them (no silent unverifiable drift).
        fault = {
            "capability": "code_verification",
            "error_type": "gating_test_modified",
            "confidence": "medium",
            "detector_evidence": {"gold_tests_edited": ["tests/test_a.py"]},
        }
        a = audit_fault(
            fault,
            agent="a",
            inst="i",
            gold_text="",
            agent_text=self.AGENT_DIFF,
            resolved=False,
            traj=None,
            test_files=["tests/test_a.py"],
        )
        self.assertEqual(a["verdict"], RE_DERIVED)


class TestCatalogAndChains(unittest.TestCase):
    def test_grounding_signals_bound(self):
        self.assertEqual(expected_evidence_signal("gave_up_early"), "gave_up_early")
        self.assertEqual(expected_evidence_signal("no_test_after_final_edit"), "no_test_after_final_edit")
        self.assertEqual(expected_evidence_signal("test_expectation_rewritten"), "assert_rewritten_files")
        self.assertIsNone(expected_evidence_signal("gating_test_modified"))  # demoted v0.39

    def test_chains_render(self):
        g = _gold()
        det = {"signals": {}, "loc_flag": "no_source_edit"}
        for cap, et, ev in (
            (
                "code_editing",
                "gave_up_early",
                {"patch_state": "missing", "ran_any_test": False, "files_read": 0, "n_steps": 3},
            ),
            (
                "code_verification",
                "no_test_after_final_edit",
                {"ran_any_test": True, "no_test_after_final_edit": True, "n_test_runs": 2},
            ),
            (
                "code_verification",
                "test_expectation_rewritten",
                {
                    "assert_rewritten_files": ["tests/test_a.py"],
                    "assert_rewritten_gold_tests": ["tests/test_a.py"],
                    "assert_rewritten_other_tests": [],
                    "after_observed_failure": True,
                },
            ),
        ):
            chain = build_chain(
                {"capability": cap, "error_type": et, "confidence": "medium", "evidence": ev},
                det,
                g,
                {},
                {},
                0.5,
                False,
            )
            self.assertGreaterEqual(len(chain["links"]), 2, et)
            self.assertEqual(chain["strength"], "associational", et)


if __name__ == "__main__":
    unittest.main()
