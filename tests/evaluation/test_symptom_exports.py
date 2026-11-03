import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent_trajviz.awe._engine import symptom_exports as export
from agent_trajviz.awe._engine.symptoms import analyze
from .test_verification_observations import call, result, traj


class SymptomExportTests(unittest.TestCase):
    def test_long_command_is_fingerprinted_after_analysis(self):
        command = "pytest " + ("x" * 1100)
        analysis = analyze(traj(call(command=command), result(out="no tests ran in 0.2s")))
        clipped = export.compact(analysis)
        for row in clipped["verification"]["runs"] + clipped["symptoms"]:
            self.assertTrue(row["command_truncated"])
            self.assertEqual(row["command_sha256"], hashlib.sha256(command.encode()).hexdigest())
            self.assertEqual(len(row["command"]), 1000)
        self.assertEqual(analysis["verification"]["runs"][0]["command"], command)

    def test_all_outcomes_and_missing_inputs_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "absent.diff"
            with (
                patch.object(export.config, "require_instances", side_effect=lambda x: list(x)),
                patch.object(export.config, "patch_path", return_value=missing),
                patch.object(
                    export,
                    "load_normalized",
                    side_effect=lambda a, i: None if i == "unknown" else traj(call(), result(out="3 skipped in 0.2s")),
                ),
                patch.object(
                    export, "resolved", side_effect=lambda a, i: {"solved": True, "failed": False, "unknown": None}[i]
                ),
            ):
                cases = export.collect(["a"], ["solved", "failed", "unknown"], workers=2)
            self.assertEqual([c["instance_id"] for c in cases], ["solved", "failed", "unknown"])
            summary = export.write(cases, Path(d))
            self.assertEqual(summary["by_outcome"]["solved"]["cases_with_symptom"], {"all_tests_skipped": 1})
            self.assertEqual(summary["by_outcome"]["failed"]["cases"], 1)
            self.assertEqual(summary["by_outcome"]["excluded_or_unknown"]["trajectories_available"], 0)
            self.assertFalse(cases[2]["analysis"]["patch_available"])
            rows = export.to_csv_rows(cases)
            self.assertEqual(len(rows), 2)
            self.assertNotIn(b"\r\n", (Path(d) / "symptoms.csv").read_bytes())
            self.assertTrue(all(r["scoring"] == "not_scored" for r in rows))
            self.assertEqual(len(json.loads((Path(d) / "symptoms.json").read_text())["cases"]), 3)

    def test_input_errors_never_silently_drop_cases(self):
        with (
            patch.object(export.config, "require_instances", side_effect=lambda x: x),
            patch.object(export, "load_normalized", side_effect=ValueError("corrupt trace")),
            self.assertRaisesRegex(ValueError, "corrupt trace"),
        ):
            export.collect(["a"], ["i"], workers=2)

    def test_oracle_patch_can_be_reviewed_without_a_trajectory(self):
        from .test_oracle_candidates import patch as make_patch, fn

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.diff"
            p.write_text(make_patch(fn("assert value < 5"), fn("assert value < 10")))
            with (
                patch.object(export.config, "patch_path", return_value=p),
                patch.object(export, "load_normalized", return_value=None),
                patch.object(export, "resolved", return_value=True),
            ):
                case = export.case_record("a", "i")
            self.assertFalse(case["analysis"]["available"])
            self.assertTrue(case["analysis"]["patch_available"])
            self.assertEqual(case["analysis"]["symptoms"][0]["kind"], "oracle_weakening_candidate")
