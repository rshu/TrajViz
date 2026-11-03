"""Printed control failures and unavailable evidence must fail the command."""

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import contextlib
import io
import runpy
import unittest
from unittest.mock import patch

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent


class ValidationExitTests(unittest.TestCase):
    def run_validation(self, arbiter_passed=1, resolved=1):
        main = runpy.run_path(str(_ENGINE_ROOT / "scripts/run_validation.py"))["main"]
        good = {"n": 1, "passed": 1, "checks": {"control": True}}
        functions = {
            "false_positive_control": lambda: {
                "n_resolved": resolved,
                "failure_errors_emitted": 0,
                "false_positive_rate": 0,
                "over_editing_on_success": 0,
            },
            "patch_mutation": lambda **kw: {
                "per_mutation": {"fixture": {"n": 1, "recall": 1, "cap_ok": 1}},
                "cap_confusion": {},
            },
            "synthetic_trajectory": lambda **kw: {"fixture": {"n": 1, "ok": 1}},
            "judge_parse": lambda: good,
            "judge_blame_upstream": lambda: good,
            "arbiter_wiring": lambda: {**good, "passed": arbiter_passed},
            "countermeasure_catalog": lambda: {**good, "uncovered": []},
        }
        audit = {"n": 1, "counts": {"re_derived": 1}, "checks": {"no_mismatches": True}}
        with (
            patch.dict(main.__globals__, functions),
            patch.object(main.__globals__["validate"], "evidence_chain_audit", return_value=audit),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()

    def test_failed_wiring_exits_unsuccessfully(self):
        with self.assertRaises(SystemExit):
            self.run_validation(arbiter_passed=0)

    def test_missing_corpus_is_not_a_pass(self):
        with self.assertRaises(SystemExit):
            self.run_validation(resolved=0)

    def test_measured_passing_controls_succeed(self):
        self.run_validation()
