"""Tests for the v0.34 per-run snapshot (agent_trajviz.awe._engine/runs.py).

Run:  python3 -m unittest tests.test_runs -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from agent_trajviz.awe._engine.runs import new_run_id, snapshot


def _fixture(root: Path):
    dossier = root / "dossier"
    reports = root / "reports"
    dossier.mkdir(parents=True)
    reports.mkdir()
    cases = [
        {
            "case_id": "a__i1",
            "blame_status": "primary",
            "fault_side": {"side": "grader", "certainty": "suspect"},
            "faults": [{"error_type": "incorrect_patch"}],
        },
        {
            "case_id": "a__i2",
            "blame_status": "unattributed",
            "fault_side": {"side": "model", "certainty": "default"},
            "faults": [],
        },
    ]
    (dossier / "faults.json").write_text(json.dumps(cases))
    (dossier / "faults.csv").write_text("case_id\na__i1\n")
    (dossier / "INSIGHTS.md").write_text("# Cross-Agent Insights Digest\n")
    (dossier / "claude_code.md").write_text("# dossier\n")
    (reports / "claude_code.md").write_text("# report\n")
    return dossier, reports


class TestRunId(unittest.TestCase):
    def test_sortable_and_filesystem_safe(self):
        from datetime import datetime

        rid = new_run_id(datetime(2026, 8, 13, 21, 45))
        self.assertEqual(rid, "2026-08-13_2145")
        self.assertNotIn("/", rid)
        self.assertNotIn(":", rid)


class TestSnapshot(unittest.TestCase):
    def test_full_snapshot_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dossier, reports = _fixture(root)
            dest = snapshot(run_id="2026-08-13_0001", dossier_dir=dossier, reports_dir=reports, runs_dir=root / "runs")
            self.assertEqual(dest.name, "2026-08-13_0001")
            for name in ("faults.json", "faults.csv", "INSIGHTS.md", "claude_code.md", "manifest.json"):
                self.assertTrue((dest / name).exists(), name)
            self.assertTrue((dest / "reports" / "claude_code.md").exists())
            m = json.loads((dest / "manifest.json").read_text())
            self.assertEqual(m["run_id"], "2026-08-13_0001")
            self.assertEqual(m["n_failed_cases"], 2)
            self.assertEqual(m["n_faults"], 1)
            self.assertEqual(m["blame_status"], {"primary": 1, "unattributed": 1})
            self.assertEqual(m["fault_side"], {"grader": 1, "model": 1})

    def test_same_minute_rerun_never_clobbers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dossier, reports = _fixture(root)
            kw = dict(run_id="2026-08-13_0002", dossier_dir=dossier, reports_dir=reports, runs_dir=root / "runs")
            first = snapshot(**kw)
            second = snapshot(**kw)
            self.assertNotEqual(first, second)
            self.assertEqual(second.name, "2026-08-13_0002.2")
            self.assertTrue((first / "manifest.json").exists())

    def test_empty_source_raises_and_leaves_no_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dossier").mkdir()
            with self.assertRaises(FileNotFoundError):
                snapshot(run_id="x", dossier_dir=root / "dossier", reports_dir=root / "reports", runs_dir=root / "runs")
            self.assertFalse((root / "runs" / "x").exists())

    def test_extra_merged_into_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dossier, reports = _fixture(root)
            dest = snapshot(
                run_id="r",
                dossier_dir=dossier,
                reports_dir=reports,
                runs_dir=root / "runs",
                extra={"args": ["--limit", "20"]},
            )
            m = json.loads((dest / "manifest.json").read_text())
            self.assertEqual(m["args"], ["--limit", "20"])


if __name__ == "__main__":
    unittest.main()
