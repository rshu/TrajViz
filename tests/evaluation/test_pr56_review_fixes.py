"""Regression tests for the PR-56 adversarial-review fixes (v0.40):
Django/docstring-form test ids in the named-test primitive (the un-nameable
case is UNKNOWN, never "not named"), the f2p == 1.0 supersession set,
partial_fix_only REFINING (not rivalling) incorrect_patch, the ran_any_test
gate on regression_test_missing, untruncated test_commands, the keyed
report.json instance check, the audit's f2p_total gate, gradable-only eval+
coverage, and the widened assertion classifier.

Run:  python3 -m unittest tests.test_pr56_review_fixes -v
"""

from __future__ import annotations

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package


import os
import subprocess
import sys
import unittest
from dataclasses import asdict
from unittest import mock


from agent_trajviz.awe._engine import digest
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.evidence import is_assertion_line
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.outcomes import parse_pertest
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine.rootcause import build_blame
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    test_id_tokens as _test_id_tokens,
    names_any_test,
    RUN_TEST,
)
from agent_trajviz.awe._engine.trajsignals import compute

from .test_pertest_tier import _report, _tsig, X, Y, KEEP

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent

DJ = "test_extract_year_func (db_functions.datetime.test_extract_trunc.DateFunctionTests)"
DOC = "Extract year uses a BETWEEN filter to compare the year to allow indexes"


def _gold(files, verif=True):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=list(files),
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_a.py"],
        fail_to_pass=[X, Y],
        pass_to_pass=[KEEP],
        is_multi_file=len(files) > 1,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": verif, "task_planning": False},
        gold_changed_lines={},
    )


def _patch(files=("pkg/a.py",)):
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=True,
        edited_files=list(files),
        edited_source_files=list(files),
        edited_test_files=[],
        patch_loc=5,
        symbols=[],
        changed_lines={},
    )


def _cands(report, t=None, gold=None, patch=None):
    g = gold or _gold(["pkg/a.py"])
    loc = localization_signals(g, patch or _patch(), False, read_set=set())
    pt = parse_pertest(report, "i") if report is not None else None
    return objective_candidates(g, loc, t, False, pertest=pt)


def _blame(cands):
    return build_blame(
        {
            "resolved": False,
            "candidates": [asdict(c) for c in cands],
            "signals": {},
            "localization": {"flag": "loc_partial"},
        }
    )


class TestTestIdForms(unittest.TestCase):
    def test_django_form_yields_module_tokens(self):
        toks = _test_id_tokens(DJ)
        for want in (
            "test_extract_year_func",
            "db_functions",
            "db_functions.datetime.test_extract_trunc",
            "db_functions/datetime/test_extract_trunc",
            "test_extract_trunc",
        ):
            self.assertIn(want, toks)
        self.assertNotIn("DateFunctionTests", toks)  # the class is not a module

    def test_docstring_form_is_un_nameable(self):
        self.assertEqual(_test_id_tokens(DOC), set())

    def test_pytest_form_unchanged(self):
        self.assertEqual(_test_id_tokens(X), {"tests/test_a.py", "test_a.py", "test_a", "test_x"})

    def test_names_any_test_three_valued(self):
        runner = ["cd tests && python runtests.py db_functions.datetime.test_extract_trunc -v 1"]
        pytest_path = ["python -m pytest tests/db_functions/datetime/test_extract_trunc.py -q"]
        self.assertTrue(names_any_test([DJ], runner))
        self.assertTrue(names_any_test([DJ], pytest_path))
        self.assertFalse(names_any_test([DJ], ["pytest tests/lookup -q"]))
        # an un-nameable id makes the question unanswerable unless something
        # nameable WAS named
        self.assertIsNone(names_any_test([DOC], ["pytest tests/lookup -q"]))
        self.assertIsNone(names_any_test([DOC, DJ], ["pytest tests/lookup -q"]))
        self.assertTrue(names_any_test([DOC, DJ], runner))


class TestRegressionTestMissingGates(unittest.TestCase):
    def _ets(self, broken, t):
        rep = _report([X, Y], [], p2p_bad=broken)
        return {c.error_type for c in _cands(rep, t)}

    def test_django_module_named_via_runner_is_not_a_gap(self):
        t = _tsig(test_commands=["cd tests && python runtests.py db_functions.datetime.test_extract_trunc -v 1"])
        self.assertNotIn("regression_test_missing", self._ets([DJ], t))
        t = _tsig(test_commands=["pytest tests/lookup -q"])
        self.assertIn("regression_test_missing", self._ets([DJ], t))

    def test_docstring_id_never_blames(self):
        t = _tsig(test_commands=["pytest tests/lookup -q"])
        self.assertNotIn("regression_test_missing", self._ets([DOC], t))
        self.assertNotIn("regression_test_missing", self._ets([DOC, DJ], t))

    def test_no_test_run_defers_to_relevant_test_not_run(self):
        t = _tsig(ran_any_test=False, test_commands=[], n_test_runs=0)
        ets = self._ets([KEEP], t)
        self.assertNotIn("regression_test_missing", ets)
        self.assertIn("relevant_test_not_run", ets)


class TestSupersessionAndRivalMass(unittest.TestCase):
    """loc_partial fixture: gold a.py + b.py, agent edited a.py only."""

    def _loc(self, report, t=None):
        return _cands(report, t, gold=_gold(["pkg/a.py", "pkg/b.py"]))

    def test_regression_only_supersedes_localization_too(self):
        cands = self._loc(_report([X, Y], [], p2p_bad=[KEEP]))
        ets = {c.error_type for c in cands}
        self.assertIn("regression_introduced", ets)
        for gone in ("missed_relevant_file", "incorrect_patch", "relevant_change_omitted"):
            self.assertNotIn(gone, ets)
        c = next(c for c in cands if c.error_type == "regression_introduced")
        self.assertEqual(c.evidence["superseded_types"], ["missed_relevant_file"])
        self.assertTrue(c.evidence["supersedes_incorrect_patch"])

    def test_partial_fix_never_manufactures_rival_mass(self):
        without = _blame(self._loc(None))
        with_pt = _blame(self._loc(_report([X], [Y])))
        self.assertEqual(without["status"], "primary")
        self.assertEqual(with_pt["status"], "primary")
        self.assertEqual(with_pt["primary"]["capability"], "code_localization")
        # no incorrect_patch to refine -> the fraction is an attribute of the
        # licensed localization miss, not a new editing cause
        ets = {b["error_type"] for b in with_pt["blame"]}
        self.assertNotIn("partial_fix_only", ets)
        self.assertNotIn("incorrect_patch", ets)
        miss = next(c for c in self._loc(_report([X], [Y])) if c.error_type == "missed_relevant_file")
        self.assertEqual(miss.evidence["f2p_pass_fraction"], 0.5)
        self.assertEqual(miss.evidence["shape"], "partial_fix")

    def test_partial_fix_records_what_it_refined(self):
        # right_place_wrong_change: incorrect_patch present -> replaced, flagged
        cands = _cands(_report([X], [Y]))
        c = next(c for c in cands if c.error_type == "partial_fix_only")
        self.assertTrue(c.evidence["supersedes_incorrect_patch"])
        self.assertNotIn("incorrect_patch", {c.error_type for c in cands})


class TestTestCommandsNotTruncated(unittest.TestCase):
    def test_long_command_kept_whole_and_audit_agrees(self):
        cmd = "cd /very/long/prefix/" + "x" * 380 + " && pytest tests/test_other.py::test_keep"
        self.assertGreater(len(cmd), 400)
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    native_tool="Bash",
                    test_run=True,
                    call_id="c0",
                    args={"command": cmd},
                ),
                NormalizedStep(
                    step_id=1,
                    turn_id=0,
                    actor="tool",
                    type="tool_result",
                    call_id="c0",
                    args={"exit_code": 0},
                    output_excerpt="1 passed",
                ),
            ],
        )
        sig = compute(traj, _gold(["pkg/a.py"]))
        self.assertEqual(sig.test_commands, [cmd])
        # the detector sees the named test -> no fault; a persisted fault
        # claiming otherwise audits as a mismatch, never as re-derived
        rep = _report([X, Y], [], p2p_bad=[KEEP])
        self.assertNotIn("regression_test_missing", {c.error_type for c in _cands(rep, sig)})
        f = {
            "capability": "code_verification",
            "error_type": "regression_test_missing",
            "confidence": "medium",
            "detector_evidence": {"p2p_broken": [KEEP], "ran_suite_wide": False, "broken_tests_referenced": False},
        }
        a = audit_fault(
            f,
            agent="a",
            inst="i",
            gold_text="",
            agent_text="",
            resolved=False,
            traj=traj,
            pertest=parse_pertest(rep, "i"),
        )
        self.assertEqual(a["verdict"], MISMATCH)


class TestLoaderAndAuditGates(unittest.TestCase):
    def test_keyed_by_another_instance_is_not_a_record(self):
        rec = _report([X], [Y])["i"]
        self.assertIsNone(parse_pertest({"other": rec}, "mine"))
        self.assertIsNone(parse_pertest({"other": rec, "other2": rec}, "mine"))
        self.assertIsNone(parse_pertest({"other": rec, "other2": rec}, None))
        self.assertEqual(parse_pertest({"other": rec}, None)["shape"], "partial_fix")
        self.assertEqual(parse_pertest(rec, "mine")["shape"], "partial_fix")

    def test_p2p_only_regression_is_rederived(self):
        pt = parse_pertest(_report([], [], p2p_bad=[KEEP]), "i")
        self.assertEqual(pt["shape"], "regression_only")
        f = {
            "capability": "code_editing",
            "error_type": "regression_introduced",
            "confidence": "high",
            "detector_evidence": {"p2p_broken": [KEEP]},
        }
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=None, pertest=pt)
        self.assertEqual(a["verdict"], RE_DERIVED)
        self.assertIn("regression_introduced", {c.error_type for c in _cands(_report([], [], p2p_bad=[KEEP]))})

    def test_docstring_id_audit_never_re_derives_not_named(self):
        pt = parse_pertest(_report([X, Y], [], p2p_bad=[DOC]), "i")
        f = {
            "capability": "code_verification",
            "error_type": "regression_test_missing",
            "confidence": "medium",
            "detector_evidence": {"p2p_broken": [DOC], "ran_suite_wide": False, "broken_tests_referenced": False},
        }
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action=RUN_TEST,
                    test_run=True,
                    args={"command": "pytest tests/lookup -q"},
                )
            ],
        )
        a = audit_fault(f, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj, pertest=pt)
        self.assertNotEqual(a["verdict"], RE_DERIVED)


class TestGradableCoverage(unittest.TestCase):
    def test_pertest_cov_counts_gradable_records_only(self):
        recs = {
            "i1": parse_pertest(_report([X], [Y]), "i"),
            "i2": parse_pertest(_report([X], [Y], applied=False), "i"),
            "i3": parse_pertest(_report([], [], p2p_bad=[KEEP]), "i"),
            "i4": None,
        }
        with (
            mock.patch.object(digest, "load_pertest", side_effect=lambda a, i: recs[i]),
            mock.patch.object(digest, "resolved", return_value=False),
            mock.patch.object(digest, "detect", return_value={"signals": {}}),
            mock.patch.object(digest, "fault_side_assess", return_value={"side": "model"}),
            mock.patch.object(
                digest,
                "build_blame",
                return_value={"status": "unattributed", "blame": [], "capability_blame": {"unattributed": 1.0}},
            ),
            mock.patch.object(digest, "auto_adjudicate", return_value=None),
            mock.patch.object(
                digest, "agent_capability_scores", return_value={"capabilities": {"code_editing": {"score": 90.0}}}
            ),
            mock.patch.object(digest, "tool_behavior", return_value={}),
        ):
            d = digest.agent_digest("a", ["i1", "i2", "i3", "i4"])
        self.assertEqual(d["n_failed"], 4)
        self.assertEqual(d["pertest_cov"], 1)
        self.assertEqual(d["unattributed"], 4)
        self.assertEqual(d["conjunctive"], 0)


class TestAssertionClassifier(unittest.TestCase):
    def test_bare_idioms_count(self):
        for line in (
            "    raises(NotImplementedError, lambda: Contains(x, S.Reals).as_set())",
            "        assertEqual(a, b)",
            "    assertFieldType(field, 'CharField')",
            "4|raises(TypeError, lambda: x)",
        ):
            self.assertTrue(is_assertion_line(line), line)
        for line in ("    raise ValueError('x')", "    x = raises_flag", "    self.raise_it()"):
            self.assertFalse(is_assertion_line(line), line)


class TestScriptGuards(unittest.TestCase):
    def test_pertest_report_agent_flag_needs_a_value(self):
        r = subprocess.run(
            [sys.executable, "scripts/pertest_report.py", "--agent"],
            cwd=str(_ENGINE_ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "AWE_ARGUS_ROOT": "/nonexistent"},
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expected one argument", r.stderr)


if __name__ == "__main__":
    unittest.main()
