"""Plans must preserve diagnosis, scope, abstention and experiment uncertainty."""

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.remedy_plan import build_plan, write, to_markdown, insights_section
from agent_trajviz.awe._engine.runs import snapshot

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent

ROOT = _ENGINE_ROOT


def case(agent="a", task="i1", overlap=0.8):
    return {
        "case_id": agent + "__" + task,
        "agent": agent,
        "instance_id": task,
        "blame_status": "primary",
        "faults": [
            {
                "capability": "code_editing",
                "error_type": "incorrect_patch",
                "blame_weight": 1,
                "is_primary": True,
                "detector_evidence": {"region_overlap": overlap},
            }
        ],
    }


class PlanTests(unittest.TestCase):
    def test_variant_selection_and_case_specific_counts(self):
        plan = build_plan([case(), case("b", "i2", 0)])
        content, region = plan["candidates"]
        self.assertEqual(content["id"], "CM-EDIT-INCORRECT-1")
        self.assertEqual(content["case_ids"], ["a__i1"])
        self.assertEqual(region["case_ids"], ["b__i2"])
        self.assertEqual(plan["shortlist"]["by_agent"]["a"], [content["id"]])
        self.assertEqual(plan["shortlist"]["by_agent"]["b"], [region["id"]])

    def test_refuted_and_failed_audits_get_no_active_experiment(self):
        refuted, bad = case(), case(task="i2")
        refuted["arbiter"] = {
            "applied": "refuted_reassigned",
            "target": {"capability": "code_editing", "error_type": "incorrect_patch"},
        }
        bad["faults"][0]["audit"] = {"verdict": "mismatch"}
        plan = build_plan([refuted, bad])
        self.assertEqual(plan["candidates"], [])
        self.assertEqual(len(plan["review_queues"]["refuted_faults_without_advice"]), 1)
        self.assertEqual(len(plan["review_queues"]["uncertain_recipe_selection"]), 2)
        self.assertIn("No supported core candidate", to_markdown(plan))

    def test_counts_are_unique_cases_and_diagnosis_is_unchanged(self):
        c = case()
        c["faults"].append(copy.deepcopy(c["faults"][0]))
        old = copy.deepcopy(c)
        candidate = build_plan([c])["candidates"][0]
        self.assertEqual(candidate["matching_failed_cases"], 1)
        self.assertEqual(candidate["primary_matching_cases"], 1)
        self.assertEqual(c, old)

    def test_duplicate_and_out_of_scope_cases_fail(self):
        for cases, kwargs in (([case(), case()], {}), ([case()], {"agents": ["b"]}), ([case()], {"instances": ["i2"]})):
            with self.assertRaises(ValueError):
                build_plan(cases, **kwargs)

    def test_empty_solved_scope_does_not_borrow_previous_recommendations(self):
        plan = build_plan([], agents=["a", "b"], instances=["i1", "i2"], benchmark="Other")
        self.assertEqual(plan["scope"]["instance_ids"], ["i1", "i2"])
        self.assertEqual(plan["shortlist"], {"overall": [], "by_agent": {"a": [], "b": []}})
        self.assertEqual(plan["scope"]["diagnosed_failed_cases"], 0)

    def test_judge_grounded_entries_are_exploratory(self):
        c = case()
        c["faults"] = [
            {
                "capability": "requirement_understanding",
                "error_type": "misread_requirement",
                "blame_weight": 1,
                "is_primary": False,
            }
        ]
        plan = build_plan([c])
        self.assertTrue(plan["exploratory_recipe_ids"])
        self.assertEqual(plan["shortlist"]["overall"], [])

    def test_benchmark_suspects_and_unattributed_cases_keep_review_queue(self):
        suspect, unknown = case(), case(task="i2")
        suspect["fault_side"] = {"side": "grader", "certainty": "suspect"}
        unknown["blame_status"], unknown["faults"] = "unattributed", []
        plan = build_plan([suspect, unknown])
        self.assertEqual(plan["review_queues"]["benchmark_suspect_case_ids"], ["a__i1"])
        self.assertEqual(plan["review_queues"]["unattributed_case_ids"], ["a__i2"])
        self.assertEqual(plan["candidates"][0]["matching_failed_cases"], 1)
        self.assertIn("audit their harness evidence", to_markdown(plan))

    def test_prerequisites_contract_caveats_and_measurement_are_preserved(self):
        plan = build_plan([case()])
        candidate = plan["candidates"][0]
        self.assertEqual(candidate["deployment_status"], "prerequisites_unverified")
        self.assertEqual(candidate["recipe"]["status"], "untested")
        self.assertEqual(len(candidate["measurement"]["arms"]), 3)
        self.assertIn("P2P preservation", candidate["measurement"]["measure"])
        md = to_markdown(plan)
        for text in ("0.83", "escape", "F2P/P2P", "active control", "not causal validation"):
            self.assertIn(text, md)
        self.assertNotIn("[RECOMMENDATIONS.md](RECOMMENDATIONS.md)", md)
        self.assertIn("[RECOMMENDATIONS.md]", "\n".join(insights_section(plan)))
        self.assertIn("0.83", "\n".join(insights_section(plan)))

    def test_missing_variant_evidence_abstains(self):
        c = case(overlap=None)
        plan = build_plan([c])
        self.assertEqual(plan["candidates"], [])
        self.assertEqual(len(plan["review_queues"]["uncertain_recipe_selection"]), 2)

    def test_snapshot_validates_json_markdown_and_requested_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dossier = root / "dossier"
            plan = build_plan([case()], agents=["a"], instances=["i1"])
            write(plan, dossier)
            (dossier / "faults.json").write_text(json.dumps([case()]))
            dest = snapshot(
                "run",
                dossier_dir=dossier,
                reports_dir=root / "reports",
                runs_dir=root / "runs",
                extra={"agents": ["a"], "instance_ids": ["i1"]},
            )
            self.assertTrue((dest / "RECOMMENDATIONS.md").exists())
            self.assertTrue((dest / "recommendations.json").exists())
            self.assertEqual(json.loads((dest / "manifest.json").read_text())["recommendation_plan"]["n_candidates"], 1)
            with self.assertRaises(ValueError):
                snapshot("badscope", dossier_dir=dossier, runs_dir=root / "runs", extra={"agents": ["b"]})
            (dossier / "RECOMMENDATIONS.md").write_text("stale recommendation")
            with self.assertRaises(ValueError):
                snapshot("badmd", dossier_dir=dossier, runs_dir=root / "runs")
            write(plan, dossier)
            (dossier / "faults.json").write_text(json.dumps([case(overlap=0)]))
            with self.assertRaises(ValueError):
                snapshot("badjson", dossier_dir=dossier, runs_dir=root / "runs")
            self.assertEqual(len(list((root / "runs").iterdir())), 1)

    def test_offline_generator_preserves_explicit_scope_on_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan([case()], agents=["a", "b"], instances=["i1", "i2"])
            write(plan, root)
            (root / "faults.json").write_text(json.dumps([case(overlap=0)]))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/gen_recommendations.py"), "--dossier", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            fresh = json.loads((root / "recommendations.json").read_text())
            self.assertEqual(fresh["scope"]["agents"], ["a", "b"])
            self.assertEqual(fresh["scope"]["instance_ids"], ["i1", "i2"])
            self.assertEqual(fresh["shortlist"]["overall"], ["CM-EDIT-INCORRECT-2"])

    def test_digest_rejects_incompatible_plan_scope(self):
        from agent_trajviz.awe._engine.digest import build_insights_md
        from .test_digest_headline import _D

        agents, data = _D()
        with patch.object(config, "AGENTS", agents), self.assertRaisesRegex(ValueError, "scope differs"):
            build_insights_md(data, recommendations=build_plan([case()]))


if __name__ == "__main__":
    unittest.main()
