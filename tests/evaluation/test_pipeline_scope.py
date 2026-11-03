"""Exercise the real subprocess pipeline with a small external benchmark."""

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent

ROOT = _ENGINE_ROOT


class TestPipelineScope(unittest.TestCase):
    def _run(self, flags, solved=False, invalid=False, external=False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data, output = root / "benchmark/data", root / "output"
            (data / "requirements").mkdir(parents=True)
            diff = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
            if external:
                diff = diff.replace("pkg/a.py", "src/core.rs").replace("x = ", "const X: i32 = ")
            for iid in ("i1", "i2"):
                (data / "requirements" / (iid + ".json")).write_text(
                    json.dumps(
                        {
                            "instance_id": iid,
                            "gold_patch": diff,
                            "repo": "r",
                            "base_commit": "baseline",
                            "problem_statement": "Fix x",
                            "FAIL_TO_PASS": ["fix::subcase[alpha]", "fix::subcase[beta]"] if external else ["fix"],
                            "PASS_TO_PASS": ["preserves::state[日本語]"] if external else [],
                        }
                    )
                )
            for agent in ("a", "b"):
                (data / "patch" / agent).mkdir(parents=True)
                (data / ("eval_" + agent + ".json")).write_text(json.dumps({"i1": solved, "i2": solved}))
                for iid in ("i1", "i2"):
                    (data / "patch" / agent / (iid + ".diff")).write_text(diff.replace("+x = 1", "+x = 2"))
            (output / "dossier").mkdir(parents=True)
            canonical = output / "dossier/faults.json"
            canonical.write_text("historical baseline")
            (output / "dossier/faults.csv").write_text("stale CSV")
            (output / "reports").mkdir()
            (output / "reports/orphan.md").write_text("historical report")
            if invalid:
                (data / "requirements/i1.json").write_text("bad JSON")
            env = {
                **os.environ,
                "AWE_ARGUS_ROOT": str(root / "benchmark"),
                "AWE_PROJECT_DATA": str(output),
                "AWE_OUTPUT_DATA": str(output),
                "AWE_AGENTS": "a,b",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            env.pop("AWE_BENCHMARK_NAME", None)
            env.pop("AWE_BENCHMARK_PROFILE", None)
            env.pop("AWE_EXECUTION_CONTEXT", None)
            if external:
                profile = root / "profile.json"
                profile.write_text(json.dumps({"name": "Rust regression benchmark", "gating_tests_visible": False}))
                env["AWE_BENCHMARK_PROFILE"] = str(profile)
                context = root / "execution-context.json"
                context.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "agents": {
                                "a": {
                                    "model": "fixture-model",
                                    "provider": "fixture-provider",
                                    "scaffold_revision": "fixture-revision",
                                    "languages": ["Rust"],
                                    "capabilities": [
                                        "hook.pre_submit",
                                        "tool.project_tests",
                                        "state.workspace_revision",
                                        "state.local_verification",
                                    ],
                                }
                            },
                        }
                    )
                )
                env["AWE_EXECUTION_CONTEXT"] = str(context)
                for agent in ("a", "b"):
                    path = output / "pertest" / agent
                    path.mkdir(parents=True)
                    for iid in ("i1", "i2"):
                        (path / (iid + ".json")).write_text(
                            json.dumps(
                                {
                                    "patch_successfully_applied": True,
                                    "resolved": False,
                                    "tests_status": {
                                        "FAIL_TO_PASS": {
                                            "success": ["fix::subcase[alpha]"],
                                            "failure": ["fix::subcase[beta]"],
                                        },
                                        "PASS_TO_PASS": {"success": [], "failure": ["preserves::state[日本語]"]},
                                    },
                                }
                            )
                        )
            env.pop("OPENROUTER_API_KEY", None)
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_awe.py"), *flags], env=env, capture_output=True, text=True
            )
            if invalid:
                self.assertNotEqual(r.returncode, 0)
                self.assertEqual(canonical.read_text(), "historical baseline")
                self.assertFalse((output / "runs").exists())
                return
            self.assertEqual(r.returncode, 0, r.stderr)
            manifests = (
                list((output / "partial").glob("*/runs/*/manifest.json"))
                if flags
                else list((output / "runs").glob("*/manifest.json"))
            )
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["agents"], ["a"] if flags else ["a", "b"])
            self.assertEqual(manifest["instance_ids"], ["i1"] if flags else ["i1", "i2"])
            self.assertEqual(manifest["n_instances"], 1 if flags else 2)
            self.assertEqual(manifest["partial"], bool(flags))
            faults = json.loads((manifests[0].parent / "faults.json").read_text())
            self.assertEqual([(c["agent"], c["instance_id"]) for c in faults], [] if solved else [("a", "i1")])
            digest = json.loads((manifests[0].parent / "insights.json").read_text())
            graphs = json.loads((manifests[0].parent / "attribution_graphs.json").read_text())
            self.assertEqual(graphs["scope"]["agents"], manifest["agents"])
            self.assertEqual(graphs["scope"]["instance_ids"], manifest["instance_ids"])
            self.assertEqual(len(graphs["graphs"]), len(faults))
            self.assertEqual(graphs["scope"]["benchmark"], "Rust regression benchmark" if external else "unspecified")
            if external:
                self.assertEqual(faults[0]["task"]["gold_files"], ["src/core.rs"])
                self.assertIn("Rust regression benchmark", (manifests[0].parent / "ATTRIBUTION.html").read_text())
                evidence = [n["details"]["detector_evidence"] for n in graphs["graphs"][0]["nodes"] if n["phase"] == 2]
                self.assertTrue(any(e.get("f2p_pass_fraction") == 0.5 for e in evidence))
                self.assertTrue(any("preserves::state[日本語]" in e.get("p2p_broken", []) for e in evidence))
                self.assertEqual(faults[0]["language"], "Rust")
                self.assertEqual(faults[0]["benchmark"], "Rust regression benchmark")
                rsi = json.loads((manifests[0].parent / "rsi_proposals.json").read_text())
                regression = [p for p in rsi["proposals"] if p["recipe_id"] == "CM-EDIT-REGRESSION-1"]
                self.assertEqual(regression[0]["compatibility"]["status"], "declared_compatible")
            self.assertTrue((manifests[0].parent / "ATTRIBUTION.html").exists())
            self.assertIn("ATTRIBUTION.html", manifest["artifacts"])
            self.assertIn("attribution_graphs.json", manifest["artifacts"])
            self.assertIn("VISUAL", r.stdout)
            self.assertIn("ATTRIBUTION.html", (manifests[0].parent / "INSIGHTS.md").read_text())
            plan = json.loads((manifests[0].parent / "recommendations.json").read_text())
            self.assertEqual(plan["scope"]["agents"], ["a"] if flags else ["a", "b"])
            self.assertEqual(plan["scope"]["instance_ids"], ["i1"] if flags else ["i1", "i2"])
            self.assertEqual(plan["scope"]["diagnosed_failed_cases"], len(faults))
            self.assertEqual(plan["scope"]["benchmark"], graphs["scope"]["benchmark"])
            self.assertTrue((manifests[0].parent / "RECOMMENDATIONS.md").exists())
            self.assertIn("RECOMMENDATIONS.md", manifest["artifacts"])
            self.assertIn("recommendations.json", manifest["artifacts"])
            self.assertIn("rsi_proposals.json", manifest["artifacts"])
            self.assertIn("rsi_feedback.json", manifest["artifacts"])
            self.assertFalse(
                json.loads((manifests[0].parent / "rsi_feedback.json").read_text())["runtime_input_allowed"]
            )
            self.assertIn("Recommended experiments", (manifests[0].parent / "INSIGHTS.md").read_text())
            self.assertEqual(list(digest), ["a"] if flags else ["a", "b"])
            self.assertEqual(digest["a"]["n"], 1 if flags else 2)
            if flags:
                self.assertEqual(canonical.read_text(), "historical baseline")
            else:
                import csv

                self.assertEqual(json.loads(canonical.read_text()), [])
                with (output / "dossier/faults.csv").open() as stream:
                    self.assertEqual(list(csv.DictReader(stream)), [])
                self.assertTrue((output / "dossier/a.md").exists())
                self.assertTrue((output / "last_run.json").exists())
                self.assertTrue((output / "reports/orphan.md").exists())
                self.assertFalse((manifests[0].parent / "reports/orphan.md").exists())
                self.assertNotIn("reports/orphan.md", manifest["artifacts"])
                md = (output / "dossier/INSIGHTS.md").read_text()
                self.assertIn("No primary root causes were diagnosed", md)
                self.assertNotIn("dominant root cause across all agents", md)

    def test_filtered_outputs_agree_and_preserve_canonical_artifacts(self):
        self._run(["--agent", "a", "--limit", "1"])

    def test_profile_named_rust_benchmark_generates_consistent_default_viewer(self):
        self._run(["--agent", "a", "--limit", "1"], external=True)

    def test_equals_syntax_has_identical_scope(self):
        self._run(["--agent=a", "--limit=1"])

    def test_invalid_options_fail_before_generation(self):
        import contextlib
        import io
        from unittest.mock import patch
        from agent_trajviz.awe._engine import config
        from agent_trajviz.awe._engine.cli import pipeline_args

        with patch.object(config, "AGENTS", ["a"]), contextlib.redirect_stderr(io.StringIO()):
            for args in [
                ["--limti", "1"],
                ["--agent", "missing"],
                ["--limit", "0"],
                ["--workers", "0"],
                ["--limit"],
                ["--max-md", "-1"],
            ]:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    pipeline_args(args)

    def test_full_solved_run_has_fresh_empty_artifacts(self):
        self._run([], solved=True)

    def test_failed_run_preserves_previous_canonical_output(self):
        self._run([], invalid=True)
