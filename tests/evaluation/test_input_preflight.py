"""Malformed exports must fail before producing scored artifacts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import parse_pertest, load_pertest
from agent_trajviz.awe._engine.preflight import validate_inputs, require_valid_inputs


DIFF = "--- a/src/core.go\n+++ b/src/core.go\n@@ -1 +1 @@\n-old\n+fixed\n"


class TestInputPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.project = self.root / "project"
        for p in (self.data / "requirements", self.data / "patch/a", self.project / "pertest/a"):
            p.mkdir(parents=True)
        self.task = {
            "instance_id": "i",
            "repo": "r",
            "base_commit": "c",
            "problem_statement": "fix",
            "gold_patch": DIFF,
            "test_patch": "",
            "FAIL_TO_PASS": '["fix", "edge"]',
            "PASS_TO_PASS": ["keep"],
        }
        self.record = {
            "patch_successfully_applied": True,
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["fix"], "failure": ["edge"]},
                "PASS_TO_PASS": {"success": ["keep"], "failure": []},
            },
        }
        for name, val in (
            ("DATA", self.data),
            ("REQUIREMENTS_DIR", self.data / "requirements"),
            ("PATCH_DIR", self.data / "patch"),
            ("PROJECT_DATA", self.project),
            ("TRAJECTORY_DIR", self.data / "trajectory"),
        ):
            mock = patch.object(config, name, val)
            mock.start()
            self.addCleanup(mock.stop)
        (self.data / "eval_a.json").write_text('{"i": false}')
        (self.data / "patch/a/i.diff").write_text(DIFF)
        self.save()

    def save(self):
        (self.data / "requirements/i.json").write_text(json.dumps(self.task))
        (self.project / "pertest/a/i.json").write_text(json.dumps({"i": self.record}))

    def test_valid_nonpython_export_and_optional_coverage(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        report = require_valid_inputs(["a"], ["i"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["counts"]["pertest_records"], 1)
        self.assertEqual(report["warnings"][0]["code"], "missing_trajectory")
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_invalid_reference_and_task_metadata(self):
        original = copy.deepcopy(self.task)
        cases = [
            ("gold_patch", ""),
            ("gold_patch", "not a diff"),
            ("gold_patch", DIFF.replace("src/core.go", "tests/core.go")),
            ("gold_patch", DIFF.replace("@@ -1 +1 @@", "@@ -3,4 +3,4 @@")),
            ("gold_patch", DIFF.replace("src/core.go", "../core.go")),
            ("instance_id", "wrong"),
            ("base_commit", ""),
            ("gating_tests_visible", "false"),
            ("FAIL_TO_PASS", ["fix", "fix"]),
            ("FAIL_TO_PASS", "fix"),
            ("PASS_TO_PASS", ["fix"]),
        ]
        for key, val in cases:
            with self.subTest(key=key, value=val):
                self.task = {**original, key: val}
                self.save()
                self.assertTrue(validate_inputs(["a"], ["i"])["errors"])

    def test_partitions_reject_incomplete_or_fabricated_grades(self):
        original = copy.deepcopy(self.record)
        mutations = [
            lambda d: d.update(resolved="false"),
            lambda d: d.update(patch_successfully_applied=1),
            lambda d: d["tests_status"]["FAIL_TO_PASS"].update(success=["fix", "fix"]),
            lambda d: d["tests_status"]["FAIL_TO_PASS"].update(failure=["fix"]),
            lambda d: d["tests_status"]["FAIL_TO_PASS"].update(failure=[]),
            lambda d: d["tests_status"]["FAIL_TO_PASS"].update(failure=["invented"]),
            lambda d: d["tests_status"]["PASS_TO_PASS"].pop("failure"),
            lambda d: d.update(resolved=True),
            lambda d: d["tests_status"]["FAIL_TO_PASS"].update(skipped=["edge"]),
        ]
        for mutate in mutations:
            self.record = copy.deepcopy(original)
            mutate(self.record)
            self.save()
            with self.subTest(record=self.record):
                self.assertTrue(validate_inputs(["a"], ["i"])["errors"])
                with self.assertRaises(ValueError):
                    load_pertest("a", "i")

    def test_application_failure_discards_stale_test_results(self):
        self.record["patch_successfully_applied"] = False
        pt = parse_pertest(self.record, expected_f2p=["fix", "edge"], expected_p2p=["keep"])
        self.assertEqual(pt["f2p_total"], 0)
        self.assertEqual(pt["p2p_broken"], [])
        self.assertIsNone(pt["f2p_pass_fraction"])

    def test_unknown_outcomes_do_not_require_agent_artifacts(self):
        (self.data / "eval_a.json").write_text('{"i": null}')
        (self.data / "patch/a/i.diff").unlink()
        report = validate_inputs(["a"], ["i"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["counts"]["unknown_or_unattempted"], 1)

    def test_missing_patch_is_not_an_empty_patch(self):
        path = self.data / "patch/a/i.diff"
        path.unlink()
        self.assertTrue(validate_inputs(["a"], ["i"])["errors"])
        path.write_text("")
        self.assertFalse(validate_inputs(["a"], ["i"])["errors"])

    def test_wrong_join_and_unsafe_ids(self):
        (self.data / "eval_a.json").write_text('{"other": false}')
        self.assertTrue(validate_inputs(["a"], ["i"])["errors"])
        for name in ("../escape", "/abs", "a\\b", ".", "a:b"):
            self.assertTrue(validate_inputs([name], ["i"])["errors"])
            with self.assertRaises(ValueError):
                config.patch_path(name, "i")

    def test_effective_overrides_and_duplicates(self):
        self.record["resolved"] = True
        self.record["tests_status"]["FAIL_TO_PASS"] = {"success": ["fix", "edge"], "failure": []}
        self.save()
        self.assertTrue(validate_inputs(["a"], ["i"])["errors"])
        entry = {"agent": "a", "instance_id": "i", "resolved": True, "reason": "verified rerun", "evidence": "report"}
        target = self.project / "overrides/resolved_overrides.json"
        target.parent.mkdir()
        target.write_text(json.dumps({"overrides": [entry]}))
        self.assertFalse(validate_inputs(["a"], ["i"])["errors"])
        self.record["resolved"] = False
        self.record["tests_status"]["FAIL_TO_PASS"] = {"success": ["fix"], "failure": ["edge"]}
        self.save()
        report = validate_inputs(["a"], ["i"])
        self.assertFalse(report["errors"])
        self.assertEqual(report["counts"]["withheld_pertest"], 1)
        self.assertIsNone(load_pertest("a", "i"))
        target.write_text(json.dumps({"overrides": [entry, entry]}))
        self.assertTrue(validate_inputs(["a"], ["i"])["errors"])

    def test_partial_scope_accepts_other_valid_outcome_ids(self):
        (self.data / "requirements/other.json").write_text("{}")
        (self.data / "eval_a.json").write_text('{"i": false, "other": true}')
        self.assertFalse(validate_inputs(["a"], ["i"])["errors"])
