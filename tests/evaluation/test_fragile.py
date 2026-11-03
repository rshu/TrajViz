from __future__ import annotations

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import unittest
from agent_trajviz.awe._engine.fragile import suspect_solve, report_md

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent


class TestSuspectSolve(unittest.TestCase):
    def test_solved_with_destructive_test_edit_is_high(self):
        rec = suspect_solve(
            {"gold_tests_edited": ["pkg/tests/test_x.py"], "gold_tests_modified_existing": ["pkg/tests/test_x.py"]},
            True,
        )
        self.assertIsNotNone(rec)
        self.assertEqual(rec["severity"], "high")
        self.assertEqual(rec["edited_test_files"], ["pkg/tests/test_x.py"])

    def test_solved_append_only_is_medium(self):
        rec = suspect_solve({"gold_tests_edited": ["pkg/tests/test_x.py"], "gold_tests_modified_existing": []}, True)
        self.assertEqual(rec["severity"], "medium")

    def test_solved_no_test_edit_is_none(self):
        self.assertIsNone(suspect_solve({"gold_tests_edited": []}, True))

    def test_failed_case_is_none(self):
        self.assertIsNone(suspect_solve({"gold_tests_edited": ["pkg/tests/test_x.py"]}, False))

    def test_report_md_high_first_and_header(self):
        recs = [
            {
                "agent": "a",
                "instance_id": "i2",
                "repo": "r",
                "edited_test_files": ["t.py"],
                "modified_existing": [],
                "severity": "medium",
            },
            {
                "agent": "a",
                "instance_id": "i1",
                "repo": "r",
                "edited_test_files": ["t.py"],
                "modified_existing": ["t.py"],
                "severity": "high",
            },
        ]
        md = report_md(recs, n_resolved=100)
        self.assertIn("integrity", md.lower())  # honest integrity framing
        self.assertLess(md.index("i1"), md.index("i2"))  # HIGH row first


class TestIsolation(unittest.TestCase):
    def test_failure_pipeline_does_not_import_fragile(self):

        pkg = _ENGINE_ROOT
        for mod in ("detect.py", "rootcause.py", "scoring.py", "dossier.py"):
            for line in (pkg / mod).read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and "fragile" in stripped:
                    self.fail(f"{mod} imports agent_trajviz.awe._engine.fragile (isolation broken): {stripped!r}")


class TestReportShape(unittest.TestCase):
    def test_empty_queue_renders_header(self):
        md = report_md([], n_resolved=0)
        self.assertIn("Fragile solves", md)
        self.assertIn("gold gating-test file", md)
