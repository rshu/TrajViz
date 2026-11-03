"""Tests for the v0.33 case-level fault_side assessment (agent_trajviz.awe._engine/faultside.py).

Covers the shared BA-HARNESS-1 predicate (annex-aligned grader arm), the
harness/OP-TRUNC-1 arm, the model default, truncation-first precedence, the
fail-closed gates, the rejected environment signature (must stay unreachable),
constant lockstep with the queue script and the detect/evidence near-gold
split, the payload contract (None = unmeasured, never asserted-False), and
behavioral case_record wiring.

Run:  python3 -m unittest tests.test_fault_side -v
"""

from __future__ import annotations

import unittest
from unittest import mock


from agent_trajviz.awe._engine.faultside import BA_HARNESS_COVERAGE, SIDES, assess, is_ba_harness_suspect


def _sig(**kw):
    base = {
        "ran_any_test": True,
        "last_test_failed": False,
        "truncated_run": False,
        "gold_added_coverage": 0.0,
        "claimed_success_final": False,
        "fail_to_pass_referenced": False,
        "no_test_after_final_edit": False,
    }
    base.update(kw)
    return base


class TestPredicate(unittest.TestCase):
    def test_qualifying_shape(self):
        self.assertTrue(is_ba_harness_suspect(True, False, 0.75, False))

    def test_each_gate_disqualifies(self):
        self.assertFalse(is_ba_harness_suspect(False, False, 0.9, False))  # no tests
        self.assertFalse(is_ba_harness_suspect(True, True, 0.9, False))  # red last test
        self.assertFalse(is_ba_harness_suspect(True, False, 0.9, True))  # truncated
        self.assertFalse(is_ba_harness_suspect(True, False, None, False))  # cov unmeasured
        self.assertFalse(is_ba_harness_suspect(True, False, 0.5, False))  # below threshold

    def test_threshold_boundary(self):
        self.assertTrue(is_ba_harness_suspect(True, False, BA_HARNESS_COVERAGE, False))
        self.assertFalse(is_ba_harness_suspect(True, False, BA_HARNESS_COVERAGE - 1e-9, False))

    def test_fail_closed_on_non_strict_bools(self):
        # 1 / 'true' / None must all disqualify — a red last test recorded as 1
        # must never count as green (the is-not-True trap fails OPEN; the
        # predicate must fail CLOSED).
        self.assertFalse(is_ba_harness_suspect(1, False, 0.9, False))
        self.assertFalse(is_ba_harness_suspect(True, 1, 0.9, False))
        self.assertFalse(is_ba_harness_suspect(True, "false", 0.9, False))
        self.assertFalse(is_ba_harness_suspect(True, None, 0.9, False))
        self.assertFalse(is_ba_harness_suspect(True, False, 0.9, None))


class TestConstantLockstep(unittest.TestCase):
    def test_queue_script_shares_predicate_and_metric(self):
        from agent_trajviz.awe._engine.scripts import harness_suspects
        from agent_trajviz.awe._engine import diffutil

        self.assertIs(harness_suspects.is_ba_harness_suspect, is_ba_harness_suspect)
        self.assertIs(harness_suspects.gold_added_coverage, diffutil.gold_added_coverage)
        self.assertIs(harness_suspects.BA_HARNESS_COVERAGE, BA_HARNESS_COVERAGE)

    def test_near_gold_split_single_home(self):
        # detect owns the v0.32 near-gold constant; evidence must import it,
        # not re-hardcode 0.8 (three drifting copies was a review finding).
        from agent_trajviz.awe._engine import detect, evidence

        self.assertEqual(detect.NEAR_GOLD_FRACTION, 0.8)
        self.assertIs(evidence.NEAR_GOLD_FRACTION, detect.NEAR_GOLD_FRACTION)


class TestGraderArm(unittest.TestCase):
    def test_annex_shape_is_grader_suspect(self):
        out = assess(_sig(gold_added_coverage=0.85))
        self.assertEqual(out["side"], "grader")
        self.assertEqual(out["certainty"], "suspect")
        self.assertIn("BA-HARNESS-1", out["basis"])
        # the vocabulary correction: route to grader review, never "harness
        # diagnosis" in the deprecated evaluation-apparatus sense
        self.assertIn("grader", out["basis"])
        self.assertNotIn("harness-diagnosis", out["basis"])

    def test_under_threshold_is_model(self):
        self.assertEqual(assess(_sig(gold_added_coverage=0.69))["side"], "model")

    def test_red_last_test_is_model(self):
        self.assertEqual(assess(_sig(gold_added_coverage=0.9, last_test_failed=True))["side"], "model")


class TestHarnessArm(unittest.TestCase):
    def test_truncation_is_harness_suspect(self):
        out = assess(_sig(truncated_run=True))
        self.assertEqual(out["side"], "harness")
        # v0.33: only the kill is deductive; the repair-locus attribution is
        # not (the patch may independently be wrong) — certainty is suspect.
        self.assertEqual(out["certainty"], "suspect")
        self.assertIn("OP-TRUNC-1", out["basis"])

    def test_truncation_first_precedence(self):
        # A truncated run's "green last test" may be a dangling, unobserved
        # call — it must NOT mint a grader suspect (reverses the draft order).
        out = assess(_sig(truncated_run=True, gold_added_coverage=0.95))
        self.assertEqual(out["side"], "harness")


class TestModelDefault(unittest.TestCase):
    def test_empty_and_none_signals(self):
        for s in ({}, None):
            out = assess(s)
            self.assertEqual(out["side"], "model")
            self.assertEqual(out["certainty"], "default")

    def test_no_trajectory_keys_is_model_even_with_coverage(self):
        self.assertEqual(assess({"gold_added_coverage": 0.95})["side"], "model")

    def test_divergence_shape_stays_model(self):
        # The rejected environment signature (22.8% of shipped failed cases —
        # under-verification, model-side; DESIGN.md §0ai).
        out = assess(_sig(claimed_success_final=True, fail_to_pass_referenced=True))
        self.assertEqual(out["side"], "model")

    def test_environment_and_owner_unreachable(self):
        for cov in (None, 0.9):
            for trunc in (True, False, None):
                for claimed in (True, False):
                    out = assess(
                        _sig(
                            gold_added_coverage=cov,
                            truncated_run=trunc,
                            claimed_success_final=claimed,
                            fail_to_pass_referenced=True,
                        )
                    )
                    self.assertIn(out["side"], SIDES)
                    self.assertNotIn(out["side"], ("environment", "owner"))


class TestPayload(unittest.TestCase):
    KEYS = {
        "truncated_run",
        "ran_any_test",
        "last_test_failed",
        "claimed_success_final",
        "fail_to_pass_referenced",
        "no_test_after_final_edit",
        "gold_added_coverage",
        "gating_run_observed",
        "last_gating_run_failed",
        "eval_status",
        "eval_attribution",
    }

    def test_keys_uniform_on_every_outcome(self):
        for s in (None, {}, _sig(truncated_run=True), _sig(gold_added_coverage=0.9)):
            self.assertEqual(set(assess(s)["signals"]), self.KEYS)

    def test_missing_keys_stay_none_not_false(self):
        # None = unmeasured. A no-trajectory case must not assert
        # truncated_run=False as an observation (review finding: .get(..., False)
        # published unknowns as facts).
        out = assess({})["signals"]
        for k in self.KEYS:
            self.assertIsNone(out[k], k)

    def test_divergence_facts_visible(self):
        out = assess(_sig(claimed_success_final=True, fail_to_pass_referenced=True, no_test_after_final_edit=True))
        self.assertIs(out["signals"]["claimed_success_final"], True)
        self.assertIs(out["signals"]["fail_to_pass_referenced"], True)
        self.assertIs(out["signals"]["no_test_after_final_edit"], True)


class TestDossierWiring(unittest.TestCase):
    def test_unattributed_case_record_carries_fault_side(self):
        # Behavioral: mock the data layer, drive the unattributed return path,
        # and assert the attribute — a truncated run with nothing to blame is
        # exactly the shape fault_side exists for.
        from agent_trajviz.awe._engine import dossier
        from agent_trajviz.awe._engine.gold import GoldReference

        g = GoldReference(
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
        with (
            mock.patch.object(
                dossier,
                "detect",
                return_value={"resolved": False, "candidates": [], "signals": {"truncated_run": True}},
            ),
            mock.patch.object(dossier, "build_gold_reference", return_value=g),
        ):
            rec = dossier.case_record("agentx", "i")
        self.assertEqual(rec["blame_status"], "unattributed")
        self.assertEqual(rec["fault_side"]["side"], "harness")
        self.assertEqual(rec["fault_side"]["certainty"], "suspect")

    def test_renderers_surface_fault_side(self):
        from agent_trajviz.awe._engine.dossier import to_csv_rows, to_markdown

        case = {
            "case_id": "a__i",
            "agent": "a",
            "instance_id": "i",
            "repo": "r",
            "blame_status": "primary",
            "fault_side": {
                "side": "grader",
                "certainty": "suspect",
                "basis": "BA-HARNESS-1: test basis",
                "signals": {},
            },
            "task": {"problem_statement": "p", "gold_files": ["f.py"], "multi_file": False, "cross_module": False},
            "reasoning_observability": "observed",
            "thinking_process": [],
            "faults": [
                {
                    "label": "Code Editing → incorrect_patch",
                    "capability": "code_editing",
                    "error_type": "incorrect_patch",
                    "confidence": "high",
                    "blame_weight": 1.0,
                    "is_primary": True,
                    "detector_evidence": {},
                    "code_evidence": {},
                    "evidence_chain": None,
                    "recommendation": [],
                    "countermeasures": [],
                }
            ],
        }
        md = to_markdown([case])
        self.assertIn("**Fault side:** grader (suspect) — BA-HARNESS-1: test basis", md)
        rows = to_csv_rows([case])
        self.assertEqual(rows[0]["fault_side"], "grader")
        self.assertEqual(rows[0]["fault_side_certainty"], "suspect")


if __name__ == "__main__":
    unittest.main()
