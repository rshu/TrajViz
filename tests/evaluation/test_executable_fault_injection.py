"""Actual evaluator executions and operational-error counterexamples."""

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_trajviz.awe._engine.fault_injection import FIXTURES, _execute, _grade, run_fixture, run_pilot

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent


class TestExecutablePilot(unittest.TestCase):
    def check_fixture(self, language, runtime):
        folder = FIXTURES / language
        before = {str(p): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
        result = run_fixture(folder, runtime)
        self.assertTrue(result["passed"], json.dumps(result, indent=2))
        self.assertTrue(result["baseline_contract_passed"])
        self.assertTrue(result["reference_contract_passed"])
        self.assertEqual(len(result["cases"]), 9)
        rows = {row["name"]: row for row in result["cases"]}
        self.assertEqual(rows["missing_upper_bound"]["pertest"]["f2p_pass_fraction"], 0.5)
        self.assertEqual(rows["regression_in_add"]["pertest"]["p2p_broken"], ["addition"])
        self.assertEqual(rows["wrong_file"]["pertest"]["f2p_pass_fraction"], 0)
        source = json.loads((folder / "fixture.json").read_text())["source"]
        for row in result["cases"]:
            self.assertEqual(row["status"], "graded")
            self.assertEqual(row["returncode"], 0)
            self.assertEqual(len(json.loads(row["stdout"])["tests"]), 4)
            if row["kind"] == "restoration":
                self.assertEqual(row["source_sha256"], rows["reference"]["source_sha256"])
                self.assertEqual(list(row["source_sha256"]), [source])
                self.assertTrue(row["pertest"]["resolved"])
                self.assertEqual(row["observed_candidate_types"], [])
        control = rows["harmless_comment"]
        self.assertNotEqual(control["source_sha256"], rows["reference"]["source_sha256"])
        self.assertTrue(control["pertest"]["resolved"])
        self.assertFalse(control["observed_candidate_types"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in folder.rglob("*") if p.is_file()})

    def test_python_real_executions_and_controls(self):
        self.check_fixture("python", sys.executable)

    @unittest.skipUnless(shutil.which("node"), "optional Node.js runtime unavailable")
    def test_javascript_real_executions_and_controls(self):
        self.check_fixture("javascript", shutil.which("node"))

    def test_bad_evaluator_records_are_rejected(self):
        record = {"tests": [{"id": "fix", "passed": True}, {"id": "keep", "passed": True}], "resolved": True}
        mutations = [
            lambda d: d.update(resolved="true"),
            lambda d: d.update(resolved=False),
            lambda d: d["tests"].pop(),
            lambda d: d["tests"].append({"id": "invented", "passed": False}),
            lambda d: d["tests"].append(d["tests"][0]),
            lambda d: d["tests"][0].update(passed=1),
            lambda d: d["tests"][0].update(status="skipped"),
        ]
        for mutate in mutations:
            doc = copy.deepcopy(record)
            mutate(doc)
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                _grade(doc, ["fix"], ["keep"])

    def test_unavailable_runtime_and_timeout_are_ungraded(self):
        with tempfile.TemporaryDirectory() as td:
            row = _execute("/nonexistent/agent_trajviz.awe._engine-runtime", "evaluate.py", td, ["fix"], [], 1)
            self.assertEqual(row["status"], "ungraded")
            self.assertIsNone(row["pertest"])
            with patch(
                "agent_trajviz.awe._engine.fault_injection.subprocess.run",
                side_effect=subprocess.TimeoutExpired("evaluate", 1, output=b"partial output", stderr=b"timed out"),
            ):
                row = _execute(sys.executable, "evaluate.py", td, ["fix"], [], 1)
            self.assertEqual(row["status"], "ungraded")
            self.assertEqual(row["stdout"], "partial output")
            self.assertIsNone(row["pertest"])

    def modified_fixture(self, evaluator):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name) / "fixture"
        shutil.copytree(FIXTURES / "python", folder)
        (folder / "evaluate.py").write_text(evaluator)
        return run_fixture(folder, sys.executable)

    def test_real_evaluator_crash_is_not_a_program_fault(self):
        result = self.modified_fixture('raise RuntimeError("broken evaluator")\n')
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["cases"]), 2)
        for row in result["cases"]:
            self.assertEqual(row["status"], "ungraded")
            self.assertIsNone(row["detection"])
            self.assertIsNone(row["pertest"])
            self.assertNotEqual(row["returncode"], 0)

    def test_incomplete_real_evaluator_output_is_ungraded(self):
        result = self.modified_fixture('print(\'{"tests": [], "resolved": true}\')\n')
        self.assertFalse(result["passed"])
        self.assertTrue(all(row["status"] == "ungraded" for row in result["cases"]))
        self.assertTrue(all(row["detection"] is None for row in result["cases"]))

    def test_always_passing_grader_fails_the_baseline_control(self):
        doc = {
            "tests": [{"id": i, "passed": True} for i in ["lower_bound", "upper_bound", "inside_interval", "addition"]],
            "resolved": True,
        }
        result = self.modified_fixture("print(" + repr(json.dumps(doc)) + ")\n")
        self.assertFalse(result["baseline_contract_passed"])
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(any(r["kind"] == "mutant" for r in result["cases"]))

    def test_always_failing_grader_does_not_blame_the_correct_reference(self):
        doc = {
            "tests": [
                {"id": i, "passed": False} for i in ["lower_bound", "upper_bound", "inside_interval", "addition"]
            ],
            "resolved": False,
        }
        result = self.modified_fixture("print(" + repr(json.dumps(doc)) + ")\n")
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(all(row["detection"] is None for row in result["cases"]))
        self.assertTrue(all(row["pertest"] is None for row in result["cases"]))

    def unstable_evaluator(self, fail_at):
        prefix = (
            "from pathlib import Path\n"
            'marker = Path("evaluation_count")\n'
            "count = int(marker.read_text()) if marker.exists() else 0\n"
            "marker.write_text(str(count + 1))\n"
            f'if count == {fail_at}: raise RuntimeError("unstable evaluator")\n'
        )
        return prefix + (FIXTURES / "python/evaluate.py").read_text()

    def test_failed_harmless_control_blocks_mutants(self):
        result = self.modified_fixture(self.unstable_evaluator(2))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["cases"]), 3)
        self.assertIsNone(result["cases"][-1]["detection"])
        self.assertIn("harmless control", result["error"])

    def test_failed_restoration_withholds_the_mutant_diagnosis(self):
        result = self.modified_fixture(self.unstable_evaluator(4))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["cases"]), 5)
        for row in result["cases"][-2:]:
            self.assertEqual(row["status"], "ungraded")
            self.assertIsNone(row["pertest"])
            self.assertIsNone(row["detection"])
        self.assertIn("restoration control", result["error"])

    def test_source_changed_during_evaluation_withholds_the_grade(self):
        evaluator = (FIXTURES / "python/evaluate.py").read_text()
        result = self.modified_fixture(evaluator + '\nopen("core.py", "a").write("# changed\\n")\n')
        self.assertEqual(result["status"], "blocked")
        for row in result["cases"]:
            self.assertEqual(row["status"], "ungraded")
            self.assertIsNone(row["detection"])
            self.assertIn("changed during evaluation", row["error"])

    def test_source_deleted_during_evaluation_is_ungraded(self):
        evaluator = (FIXTURES / "python/evaluate.py").read_text()
        result = self.modified_fixture(evaluator + '\nimport os\nos.unlink("core.py")\n')
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(all(row["status"] == "ungraded" for row in result["cases"]))

    def test_missing_optional_runtime_is_explicit(self):
        with patch("agent_trajviz.awe._engine.fault_injection.shutil.which", return_value=None):
            report = run_pilot(["javascript"])
        self.assertEqual(report["summary"]["fixtures_skipped"], 1)
        self.assertEqual(report["summary"]["cases_attempted"], 0)
        self.assertFalse(report["summary"]["passed"])

    def test_timeout_must_be_positive(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                run_fixture(FIXTURES / "python", sys.executable, timeout)

    def test_cli_is_self_contained_and_records_provenance(self):
        root = _ENGINE_ROOT
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "pilot.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/run_fault_injection.py"),
                    "--language",
                    "python",
                    "--output",
                    str(target),
                ],
                env={**os.environ, "AWE_ARGUS_ROOT": "/nonexistent"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(target.read_text())
            self.assertEqual(report["summary"]["cases_graded"], 9)
            self.assertEqual(report["summary"]["cases_passed"], 9)
            self.assertTrue(report["provenance"]["awe_code_sha256"])
            self.assertTrue(report["fixtures"][0]["runtime"]["version"])


if __name__ == "__main__":
    unittest.main()
