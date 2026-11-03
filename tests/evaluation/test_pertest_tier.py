"""Tests for the v0.39 eval+ tier — per-test harness outcomes unlock
``partial_fix_only`` / ``regression_introduced`` (code_editing, deductive from
the grader's record) and ``regression_test_missing`` (code_verification,
contributing). Covers the report.json loader, the detector rules (incl. the
regression-only supersession of incorrect_patch), detect() wiring, evidence
chains, and audit re-derivation + tamper.

Run:  python3 -m unittest tests.test_pertest_tier -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.detect import (
    objective_candidates,
    detect,
    OBJECTIVE_ERROR_SIGNALS,
    expected_evidence_signal,
)
from agent_trajviz.awe._engine.outcomes import parse_pertest
from agent_trajviz.awe._engine.evidence import build_chain, strength_of, strength_basis_for
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH, UNVERIFIABLE
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory
from agent_trajviz.awe._engine.trajsignals import TrajectorySignals


def _gold(verif=True):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_a.py"],
        fail_to_pass=["tests/test_a.py::test_x", "tests/test_a.py::test_y"],
        pass_to_pass=["tests/test_other.py::test_keep"],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": verif, "task_planning": False},
        gold_changed_lines={},
    )


def _patch():
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=["pkg/a.py"],
        edited_source_files=["pkg/a.py"],
        edited_test_files=[],
        patch_loc=5,
        symbols=[],
        changed_lines={},
    )


def _tsig(**kw):
    base = dict(
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
    base.update(kw)
    return TrajectorySignals(**base)


def _report(f2p_ok, f2p_bad, p2p_ok=(), p2p_bad=(), applied=True, keyed=True):
    rec = {
        "patch_is_None": False,
        "patch_exists": True,
        "patch_successfully_applied": applied,
        "resolved": bool(applied and not f2p_bad and not p2p_bad),
        "tests_status": {
            "FAIL_TO_PASS": {"success": list(f2p_ok), "failure": list(f2p_bad)},
            "PASS_TO_PASS": {"success": list(p2p_ok), "failure": list(p2p_bad)},
        },
    }
    return {"i": rec} if keyed else rec


X, Y, KEEP = "tests/test_a.py::test_x", "tests/test_a.py::test_y", "tests/test_other.py::test_keep"


class TestLoader(unittest.TestCase):
    def test_keyed_and_bare_forms_agree(self):
        a = parse_pertest(_report([X], [Y]), "i")
        b = parse_pertest(_report([X], [Y], keyed=False))
        self.assertEqual(a, b)
        self.assertEqual(a["f2p_pass_fraction"], 0.5)
        self.assertEqual(a["shape"], "partial_fix")
        self.assertEqual(a["eval_status"], "graded")

    def test_shapes(self):
        self.assertEqual(parse_pertest(_report([X, Y], []), "i")["shape"], "resolved")
        self.assertEqual(parse_pertest(_report([X, Y], [], p2p_bad=[KEEP]), "i")["shape"], "regression_only")
        self.assertEqual(parse_pertest(_report([X], [Y], p2p_bad=[KEEP]), "i")["shape"], "partial_fix_regressive")
        self.assertEqual(parse_pertest(_report([], [X, Y]), "i")["shape"], "unfixed")
        self.assertEqual(parse_pertest(_report([], [X, Y], p2p_bad=[KEEP]), "i")["shape"], "unfixed_regressive")

    def test_patch_apply_failure_carries_no_test_information(self):
        pt = parse_pertest(_report([X], [Y], applied=False), "i")
        self.assertFalse(pt["patch_applied"])
        self.assertIsNone(pt["f2p_pass_fraction"])
        self.assertEqual(pt["shape"], "patch_apply_failed")
        self.assertEqual(pt["eval_status"], "patch_apply_failed")

    def test_signal_bindings(self):
        self.assertEqual(OBJECTIVE_ERROR_SIGNALS["partial_fix_only"], "f2p_pass_fraction")
        self.assertEqual(OBJECTIVE_ERROR_SIGNALS["regression_introduced"], "p2p_broken")
        self.assertEqual(expected_evidence_signal("regression_test_missing"), "p2p_broken")


def _cands(report, t=None, resolved=False, gold=None):
    g = gold or _gold()
    loc = localization_signals(g, _patch(), resolved)
    pt = parse_pertest(report, "i") if report is not None else None
    return objective_candidates(g, loc, t, resolved, pertest=pt)


class TestDetector(unittest.TestCase):
    def test_partial_fix_refines_incorrect_patch(self):
        cands = _cands(_report([X], [Y]))
        ets = [c.error_type for c in cands]
        self.assertIn("partial_fix_only", ets)
        # v0.40: the per-test fact REPLACES the coarse "content diverges"
        # deduction (severity = fraction) instead of adding rival editing mass
        self.assertNotIn("incorrect_patch", ets)
        c = next(c for c in cands if c.error_type == "partial_fix_only")
        self.assertEqual(c.capability, "code_editing")
        self.assertEqual(c.confidence, "high")
        self.assertEqual(c.evidence["f2p_pass_fraction"], 0.5)
        self.assertEqual(c.evidence["f2p_failed"], [Y])

    def test_regression_only_supersedes_incorrect_patch(self):
        cands = _cands(_report([X, Y], [], p2p_bad=[KEEP]))
        ets = [c.error_type for c in cands]
        self.assertIn("regression_introduced", ets)
        self.assertNotIn("incorrect_patch", ets)  # the fix worked; one fact, one blame
        c = next(c for c in cands if c.error_type == "regression_introduced")
        self.assertTrue(c.evidence["supersedes_incorrect_patch"])
        self.assertEqual(c.evidence["p2p_broken"], [KEEP])
        self.assertEqual(c.evidence["shape"], "regression_only")

    def test_unfixed_regressive_keeps_incorrect_patch(self):
        cands = _cands(_report([], [X, Y], p2p_bad=[KEEP]))
        ets = [c.error_type for c in cands]
        self.assertIn("regression_introduced", ets)
        self.assertIn("incorrect_patch", ets)
        self.assertNotIn("partial_fix_only", ets)

    def test_no_record_or_no_apply_changes_nothing(self):
        base = {c.error_type for c in _cands(None)}
        self.assertEqual({c.error_type for c in _cands(_report([X], [Y], applied=False))}, base)
        self.assertNotIn("partial_fix_only", base)

    def test_never_on_solved(self):
        self.assertEqual(_cands(_report([X], [Y]), resolved=True), [])

    def test_regression_test_missing_needs_unchecked_surface(self):
        rep = _report([X, Y], [], p2p_bad=[KEEP])
        # never named, no suite run -> fires (contributing, medium)
        t = _tsig(test_commands=["pytest tests/test_a.py::test_x"], ran_suite_wide=False)
        ets = {c.error_type for c in _cands(rep, t)}
        self.assertIn("regression_test_missing", ets)
        # the broken test's file was named -> not a verification gap
        t = _tsig(test_commands=["pytest tests/test_other.py -q"])
        self.assertNotIn("regression_test_missing", {c.error_type for c in _cands(rep, t)})
        # a selector-free suite run covers the regression surface
        t = _tsig(test_commands=["pytest"], ran_suite_wide=True)
        self.assertNotIn("regression_test_missing", {c.error_type for c in _cands(rep, t)})
        # no trajectory -> the verification slice is closed
        self.assertNotIn("regression_test_missing", {c.error_type for c in _cands(rep, None)})
        # no verification opportunity on the task -> closed too
        self.assertNotIn(
            "regression_test_missing", {c.error_type for c in _cands(rep, _tsig(), gold=_gold(verif=False))}
        )


class TestDetectWiring(unittest.TestCase):
    def test_pertest_override_surfaces_signals(self):
        pt = parse_pertest(_report([X], [Y], p2p_bad=[KEEP]), "i")
        d = detect(
            "i",
            "a",
            use_trajectory=False,
            gold_override=_gold(),
            patch_override=_patch(),
            resolved_override=False,
            pertest_override=pt,
        )
        self.assertTrue(d["signals"]["pertest_available"])
        self.assertEqual(d["signals"]["f2p_pass_fraction"], 0.5)
        self.assertEqual(d["signals"]["p2p_broken"], [KEEP])
        self.assertEqual(d["signals"]["pertest_shape"], "partial_fix_regressive")
        ets = {c["error_type"] for c in d["candidates"]}
        self.assertTrue({"partial_fix_only", "regression_introduced"} <= ets)

    def test_synthetic_inputs_never_read_real_pertest(self):
        d = detect(
            "i", "a", use_trajectory=False, gold_override=_gold(), patch_override=_patch(), resolved_override=False
        )
        self.assertFalse(d["signals"]["pertest_available"])
        self.assertNotIn("f2p_pass_fraction", d["signals"])


class TestChainsAndStrength(unittest.TestCase):
    def test_strength_tiers(self):
        self.assertEqual(strength_of("code_editing", "partial_fix_only"), "deductive")
        self.assertEqual(strength_of("code_editing", "regression_introduced"), "deductive")
        self.assertEqual(strength_of("code_verification", "regression_test_missing"), "associational")
        self.assertIn("harness", strength_basis_for("deductive", "partial_fix_only"))

    def test_chains_render(self):
        g = _gold()
        det = {"signals": {}, "loc_flag": "right_place_wrong_change"}
        for cap, et, ev in (
            (
                "code_editing",
                "partial_fix_only",
                {"f2p_pass_fraction": 0.5, "f2p_passed": [X], "f2p_failed": [Y], "shape": "partial_fix"},
            ),
            (
                "code_editing",
                "regression_introduced",
                {
                    "p2p_broken": [KEEP],
                    "n_p2p_broken": 1,
                    "f2p_pass_fraction": 1.0,
                    "shape": "regression_only",
                    "supersedes_incorrect_patch": True,
                },
            ),
            (
                "code_verification",
                "regression_test_missing",
                {"p2p_broken": [KEEP], "ran_suite_wide": False, "broken_tests_referenced": False},
            ),
        ):
            chain = build_chain(
                {"capability": cap, "error_type": et, "confidence": "high", "evidence": ev}, det, g, {}, {}, 0.5, True
            )
            self.assertGreaterEqual(len(chain["links"]), 3, et)
            self.assertTrue(any("eval_report" in evidence_link["source"] for evidence_link in chain["links"]), et)
            self.assertTrue(chain["counterfactual"]["what_would_confirm"], et)


class TestAudit(unittest.TestCase):
    def _fault(self, et, ev, cap="code_editing"):
        return {"capability": cap, "error_type": et, "confidence": "high", "detector_evidence": ev}

    def test_partial_re_derives_and_forged_fraction_mismatches(self):
        pt = parse_pertest(_report([X], [Y]), "i")
        f = self._fault("partial_fix_only", {"f2p_pass_fraction": 0.5})
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], RE_DERIVED, a["mismatches"])
        f["detector_evidence"]["f2p_pass_fraction"] = 0.9
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], MISMATCH)
        self.assertTrue(any(m["atom"] == "f2p_pass_fraction" for m in a["mismatches"]))

    def test_regression_re_derives_forged_list_and_solved_mismatch(self):
        pt = parse_pertest(_report([X, Y], [], p2p_bad=[KEEP]), "i")
        f = self._fault("regression_introduced", {"p2p_broken": [KEEP]})
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], RE_DERIVED, a["mismatches"])
        forged = self._fault("regression_introduced", {"p2p_broken": ["tests/test_zzz.py::t"]})
        a = audit_fault(forged, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], MISMATCH)
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=True, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], MISMATCH)  # outcome atom

    def test_missing_or_unapplied_record_is_unverifiable(self):
        f = self._fault("partial_fix_only", {"f2p_pass_fraction": 0.5})
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None)
        self.assertEqual(a["verdict"], UNVERIFIABLE)
        pt = parse_pertest(_report([X], [Y], applied=False), "i")
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], UNVERIFIABLE)

    def test_regression_test_missing_re_derives_from_raw_steps(self):
        pt = parse_pertest(_report([X, Y], [], p2p_bad=[KEEP]), "i")
        f = self._fault(
            "regression_test_missing",
            {"p2p_broken": [KEEP], "ran_suite_wide": False, "broken_tests_referenced": False},
            cap="code_verification",
        )
        unchecked = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action="run_test",
                    test_run=True,
                    args={"command": "pytest tests/test_a.py::test_x"},
                )
            ],
        )
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=unchecked, pertest=pt)
        self.assertEqual(a["verdict"], RE_DERIVED, a["mismatches"])
        # tamper: the run DID name the broken test's file
        named = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action="run_test",
                    test_run=True,
                    args={"command": "pytest tests/test_other.py"},
                )
            ],
        )
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=named, pertest=pt)
        self.assertEqual(a["verdict"], MISMATCH)
        self.assertTrue(any(m["atom"] == "broken_tests_referenced" for m in a["mismatches"]))
        # no trajectory -> unverifiable (never a silent pass)
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], UNVERIFIABLE)


if __name__ == "__main__":
    unittest.main()
