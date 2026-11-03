"""RSI contracts must preserve uncertainty, isolation and bounded policy behavior."""

import copy
from dataclasses import replace
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine.execution_context import for_case, load_manifest
from agent_trajviz.awe._engine.remedy_plan import build_plan, write
from agent_trajviz.awe._engine.rsi import build_exchange, prepare, decide, write_exchange, digest, POLICY
from agent_trajviz.awe._engine.runs import snapshot


def context():
    return {
        "agent": "runner",
        "benchmark": "Rust benchmark",
        "languages": ["Rust"],
        "model": "model-id",
        "provider": "provider",
        "scaffold_revision": "agent-v1",
        "capabilities": POLICY["required_capabilities"],
    }


def case():
    return {
        "case_id": "runner__task",
        "agent": "runner",
        "instance_id": "task",
        "benchmark": "Rust benchmark",
        "language": "Rust",
        "execution_context": context(),
        "blame_status": "primary",
        "faults": [
            {
                "capability": "code_editing",
                "error_type": "regression_introduced",
                "blame_weight": 1,
                "is_primary": True,
                "confidence": "high",
                "detector_evidence": {"p2p_broken": ["private-test-id"]},
                "code_evidence": {"gold": "PRIVATE REFERENCE PATCH"},
            }
        ],
    }


def exchange(c=None):
    cases = [c or case()]
    return build_exchange(cases, build_plan(cases, benchmark="Rust benchmark"))


class ContextTests(unittest.TestCase):
    def test_missing_identity_and_capabilities_remain_unknown(self):
        with patch.dict(os.environ, {}, clear=True):
            result = for_case("runner", "task", {"name": "Rust benchmark"})
        self.assertEqual(result["benchmark"], "Rust benchmark")
        self.assertIsNone(result["language"])
        self.assertIsNone(result["execution_context"]["capabilities"])
        self.assertIsNone(result["execution_context"]["model"])

    def test_instance_languages_override_agent_defaults_without_guessing(self):
        manifest = {
            "agents": {"runner": {"languages": ["Python"], "model": "m"}},
            "instances": {"task": {"languages": ["Java", "Rust"]}},
        }
        result = for_case("runner", "task", {"name": "benchmark"}, manifest)
        self.assertEqual(result["execution_context"]["languages"], ["Java", "Rust"])
        self.assertIsNone(result["language"])

    def test_manifest_rejects_credentials_and_malformed_capabilities(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "context.json"
            for value in ({"api_key": "not-a-real-key"}, {"capabilities": [{}]}, {"capabilities": ["x", "x"]}):
                path.write_text(json.dumps({"schema_version": 1, "agents": {"runner": value}}))
                with patch.dict(os.environ, {"AWE_EXECUTION_CONTEXT": str(path)}), self.assertRaises(ValueError):
                    load_manifest()


class ExchangeTests(unittest.TestCase):
    def test_deterministic_export_does_not_change_diagnoses_or_imply_installation(self):
        c = case()
        original = copy.deepcopy(c)
        proposals, feedback = exchange(c)
        self.assertEqual(c, original)
        self.assertEqual((proposals, feedback), exchange(c))
        self.assertEqual(proposals["proposals"][0]["compatibility"]["status"], "declared_compatible")
        self.assertEqual(proposals["proposals"][0]["deployment_status"], "not_installed")
        self.assertEqual(proposals["proposals"][0]["variant_efficacy_status"], "untested")
        self.assertFalse(proposals["efficacy_claimed"])
        self.assertFalse(feedback["runtime_input_allowed"])
        self.assertNotIn("PRIVATE REFERENCE PATCH", json.dumps(feedback))
        self.assertNotIn("private-test-id", json.dumps(feedback))
        self.assertEqual(feedback["cases"][0]["faults"][0]["support"], "supported")

    def test_unknown_incompatible_and_unregistered_adapters_are_explicit(self):
        c = case()
        c["execution_context"]["capabilities"] = None
        self.assertEqual(exchange(c)[0]["proposals"][0]["compatibility"]["status"], "unknown")
        c["execution_context"]["capabilities"] = []
        self.assertEqual(exchange(c)[0]["proposals"][0]["compatibility"]["status"], "incompatible")
        c["faults"][0].update(error_type="incorrect_patch", detector_evidence={"region_overlap": 0.5})
        self.assertEqual(exchange(c)[0]["proposals"][0]["compatibility"]["status"], "requires_adapter")

    def test_only_suspect_cases_cannot_prepare_intervention(self):
        c = case()
        c["fault_side"] = {"side": "grader"}
        proposal = exchange(c)[0]["proposals"][0]
        self.assertEqual(proposal["compatibility"]["status"], "review_required")
        self.assertFalse(proposal["evidence_refs"])
        with self.assertRaises(ValueError):
            prepare(proposal, context())

    def test_refuted_faults_do_not_supply_proposals(self):
        c = case()
        c["recommendation_support"] = "refuted"
        c["faults"][0]["recommendation_support"] = "refuted"
        self.assertEqual(exchange(c)[0]["proposals"], [])

    def test_contradictory_context_and_unexpected_context_fields_fail_closed(self):
        for fields in ({"benchmark": "different"}, {"languages": ["Java"]}, {"api_key": "not-a-key"}):
            c = case()
            c["execution_context"].update(fields)
            with self.assertRaises(ValueError):
                exchange(c)

    def test_stale_plan_and_scope_are_rejected(self):
        cases = [case()]
        plan = build_plan(cases)
        plan["candidates"][0]["measurement"]["design"] = "tampered"
        with self.assertRaisesRegex(ValueError, "stale"):
            build_exchange(cases, plan)

    def test_proposal_identity_changes_when_experiment_design_changes(self):
        from agent_trajviz.awe._engine.remedy_plan import MEASUREMENT

        original = exchange()[0]["proposals"][0]["proposal_id"]
        with patch(
            "agent_trajviz.awe._engine.remedy_plan.MEASUREMENT",
            {**MEASUREMENT, "design": "different prespecified design"},
        ):
            revised = exchange()[0]["proposals"][0]["proposal_id"]
        self.assertNotEqual(original, revised)

    def test_empty_scope_keeps_unknown_outcomes_and_no_proposals(self):
        proposals, feedback = build_exchange([], build_plan([], agents=["runner"], instances=["solved-or-excluded"]))
        self.assertEqual(proposals["proposals"], [])
        self.assertEqual(feedback["cases"], [])
        self.assertIn("no outcome", feedback["coverage"])

    def test_snapshot_archives_exchange_and_rejects_stale_feedback_before_copying(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = [case()]
            plan = build_plan(cases)
            write(plan, root)
            (root / "faults.json").write_text(json.dumps(cases))
            write_exchange(cases, plan, root)
            dest = snapshot("trial", dossier_dir=root, reports_dir=root / "reports", runs_dir=root / "runs")
            self.assertTrue((dest / "rsi_feedback.json").exists())
            bad = json.loads((root / "rsi_feedback.json").read_text())
            bad["runtime_input_allowed"] = True
            (root / "rsi_feedback.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "stale RSI"):
                snapshot("bad", dossier_dir=root, reports_dir=root / "reports", runs_dir=root / "runs")
            self.assertFalse((root / "runs/bad").exists())


class PolicyTests(unittest.TestCase):
    def prepared(self):
        return prepare(exchange()[0]["proposals"][0], context())

    def state(self, revision="old", status="passed"):
        return {
            "workspace_revision": "current",
            "tests_available": True,
            "verification": {"revision": revision, "status": status},
        }

    def test_prepare_checks_observed_identity_capabilities_and_recipe_hash(self):
        proposal = exchange()[0]["proposals"][0]
        for observed in ({**context(), "scaffold_revision": "different"}, {**context(), "capabilities": []}):
            with self.assertRaises(ValueError):
                prepare(proposal, observed)
        proposal["recipe_sha256"] = "changed"
        with self.assertRaises(ValueError):
            prepare(proposal, context())

    def test_recipe_refutation_and_target_mismatch_invalidate_old_proposal(self):
        from agent_trajviz.awe._engine.countermeasures import load_catalog

        proposal = exchange()[0]["proposals"][0]
        for status in ("refuted", "verified"):
            retired = [replace(e, status=status) if e.id == proposal["recipe_id"] else e for e in load_catalog()]
            with (
                patch("agent_trajviz.awe._engine.rsi.load_catalog", return_value=retired),
                self.assertRaises(ValueError),
            ):
                prepare(proposal, context())
        proposal["target"]["error_type"] = "different"
        with self.assertRaises(ValueError):
            prepare(proposal, context())

    def test_stale_and_explicit_not_run_request_tests_but_do_not_block(self):
        for state in (self.state(), self.state(None, "not_run")):
            result = decide(self.prepared(), state)
            self.assertEqual(result["action"], "request_project_tests")
            self.assertEqual(result["next_trigger_count"], 1)
            self.assertFalse(result["blocks_submission"])

    def test_fresh_failure_requests_repair_without_repeating_same_tests(self):
        result = decide(self.prepared(), self.state("current", "failed"))
        self.assertEqual(result["reason"], "repair_required")
        self.assertEqual(result["action"], "warn")
        self.assertEqual(result["next_trigger_count"], 0)
        self.assertEqual(decide(self.prepared(), self.state("current"))["action"], "allow")

    def test_unknown_unavailable_and_exhausted_do_not_deadlock(self):
        p = self.prepared()
        for state in (
            {},
            self.state(None),
            self.state("current", "unknown"),
            {**self.state(), "tests_available": False},
        ):
            self.assertEqual(decide(p, state)["action"], "warn")
        result = decide(p, self.state(), trigger_count=2)
        self.assertEqual(result["reason"], "trigger_budget_exhausted")
        self.assertEqual(result["next_trigger_count"], 2)

    def test_oracle_inputs_corrupt_policy_and_invalid_counters_are_rejected(self):
        p = self.prepared()
        for key in ("gold_patch", "FAIL_TO_PASS", "judge_verdict"):
            with self.assertRaises(ValueError):
                decide(p, {**self.state(), key: "private"})
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                decide(p, self.state(), trigger_count=count)
        p["policy"] = {**p["policy"], "max_triggers_per_run": 20}
        with self.assertRaisesRegex(ValueError, "integrity"):
            decide(p, self.state())

    def test_context_set_order_and_provenance_text_do_not_change_variant_identity(self):
        proposal = exchange()[0]["proposals"][0]
        observed = {
            **context(),
            "capabilities": list(reversed(context()["capabilities"])),
            "provenance": "observed native capabilities",
        }
        self.assertEqual(prepare(proposal, observed)["variant_id"], self.prepared()["variant_id"])

    def test_variant_changes_with_parent_revision_and_cannot_reuse_stale_code(self):
        c = case()
        c["execution_context"]["scaffold_revision"] = "agent-v2"
        second = prepare(exchange(c)[0]["proposals"][0], c["execution_context"])
        self.assertNotEqual(second["variant_id"], self.prepared()["variant_id"])
        p = self.prepared()
        p["adapter_code_sha256"] = "old"
        artifact = {k: p[k] for k in ("recipe_id", "recipe_sha256", "policy", "adapter_code_sha256", "context")}
        p["variant_id"] = digest(artifact)
        with self.assertRaisesRegex(ValueError, "stale"):
            decide(p, self.state())
