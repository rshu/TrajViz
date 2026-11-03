"""v0.42 (DESIGN §0ba): the grader-log audit record reaches the pipeline —
``outcomes.load_pertest_log`` reads it, ``detect`` refines ``eval_status``
from it, and ``faultside.assess`` gains the two log-affirmed grader arms
(GRADER-APPLY-1, GRADER-GOLDDEP-1) ahead of the trajectory-shaped
BA-HARNESS-1 guess. The environment arm stays reserved.

Run:  python3 -m unittest tests.test_grader_log_signals -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from agent_trajviz.awe._engine import config, outcomes, detect as detect_mod
from agent_trajviz.awe._engine.faultside import assess, GRADER_LOG_STATUSES, GRADER_LOG_ATTRIBUTIONS

from .test_pertest_tier import _report, _gold, _patch


def _record(**kw):
    base = {
        "instance_id": "i",
        "agent": "a",
        "eval_status": "graded",
        "attribution": None,
        "crash_kind": None,
        "test_patch_apply_error": None,
        "runner_summary": "OK",
        "gating": {},
        "error": {},
        "parsed_counts": {},
    }
    base.update(kw)
    return base


class TestLoader(unittest.TestCase):
    def test_reads_and_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "pertest" / "a"
            d.mkdir(parents=True)
            (d / "i.log.json").write_text(
                json.dumps(
                    _record(
                        eval_status="test_patch_apply_failed",
                        test_patch_apply_error="error: tests/x.py: already exists in working directory",
                    )
                )
            )
            (d / "other.log.json").write_text(json.dumps(_record(instance_id="someone_else")))
            with mock.patch.object(config, "PROJECT_DATA", Path(td)):
                outcomes.load_pertest_log.cache_clear()
                lg = outcomes.load_pertest_log("a", "i")
                self.assertEqual(lg["eval_status"], "test_patch_apply_failed")
                self.assertIn("already exists", lg["test_patch_apply_error"])
                self.assertIsNone(outcomes.load_pertest_log("a", "missing"))
                self.assertIsNone(outcomes.load_pertest_log("a", "other"))  # keyed to another case
            outcomes.load_pertest_log.cache_clear()


class TestFaultSideArms(unittest.TestCase):
    def _base(self, **kw):
        s = {"truncated_run": False, "ran_any_test": True, "last_test_failed": True, "gold_added_coverage": 0.0}
        s.update(kw)
        return s

    def test_apply_failure_is_a_log_affirmed_grader_suspect(self):
        r = assess(self._base(eval_status="test_patch_apply_failed"))
        self.assertEqual((r["side"], r["certainty"]), ("grader", "suspect"))
        self.assertIn("GRADER-APPLY-1", r["basis"])
        self.assertEqual(r["signals"]["eval_status"], "test_patch_apply_failed")

    def test_gold_dependency_crash_is_a_grader_suspect(self):
        r = assess(self._base(eval_status="crash_before_summary", eval_attribution="gold_dependency_unverified"))
        self.assertEqual(r["side"], "grader")
        self.assertIn("GRADER-GOLDDEP-1", r["basis"])
        # an agent-added crash is model-side
        r = assess(self._base(eval_status="crash_before_summary", eval_attribution="agent_added"))
        self.assertEqual(r["side"], "model")

    def test_graded_and_unknown_statuses_change_nothing(self):
        for st in ("graded", "patch_apply_failed", None, "timeout"):
            self.assertEqual(assess(self._base(eval_status=st))["side"], "model", st)

    def test_truncation_still_wins(self):
        r = assess(self._base(truncated_run=True, eval_status="test_patch_apply_failed"))
        self.assertEqual(r["side"], "harness")

    def test_vocabulary_is_closed(self):
        self.assertEqual(set(GRADER_LOG_STATUSES), {"test_patch_apply_failed"})
        self.assertEqual(set(GRADER_LOG_ATTRIBUTIONS), {"gold_dependency_unverified"})


class TestDetectWiring(unittest.TestCase):
    def test_synthetic_override_never_reads_a_log(self):
        # a pertest override is synthetic: the disk log must not be consulted
        with mock.patch.object(detect_mod, "load_pertest_log", side_effect=AssertionError("must not be called")):
            d = detect_mod.detect(
                "i",
                "a",
                use_trajectory=True,
                patch_override=_patch(),
                resolved_override=False,
                trajectory_override=None,
                gold_override=_gold(),
                pertest_override=outcomes.parse_pertest(_report(["x"], ["y"]), "i"),
            )
        self.assertEqual(d["signals"]["eval_status"], "graded")

    def test_log_override_refines_eval_status(self):
        lg = {
            "eval_status": "test_patch_apply_failed",
            "attribution": None,
            "crash_kind": None,
            "test_patch_apply_error": "error: x: already exists",
        }
        with mock.patch.object(detect_mod, "load_pertest_log", side_effect=AssertionError("must not be called")):
            d = detect_mod.detect(
                "i",
                "a",
                use_trajectory=True,
                patch_override=_patch(),
                resolved_override=False,
                trajectory_override=None,
                gold_override=_gold(),
                pertest_override=outcomes.parse_pertest(_report(["x"], ["y"]), "i"),
                pertest_log_override=lg,
            )
        self.assertEqual(d["signals"]["eval_status"], "test_patch_apply_failed")
        self.assertIn("already exists", d["signals"]["test_patch_apply_error"])
        self.assertEqual(assess(d["signals"])["side"], "grader")


if __name__ == "__main__":
    unittest.main()
