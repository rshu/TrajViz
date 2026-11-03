"""Synthetic experiments exercise validation; they do not verify shipped remedies."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine.countermeasures import countermeasures_for
from agent_trajviz.awe._engine.remedy_verification import validate_verification, verification_scope, recipe_digest
from agent_trajviz.awe._engine.remedies import select_remedies
from agent_trajviz.awe._engine import config, validate


class VerificationFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.entry = replace(countermeasures_for(error_type="incorrect_patch")[0], status="verified")
        self.protocol = {
            "experiment_id": "synthetic-only",
            "remedy_id": self.entry.id,
            "artifact_sha256": hashlib.sha256(self.entry.artifact_content.encode()).hexdigest(),
            "recipe_sha256": recipe_digest(self.entry),
            "taxonomy": self.save("configs/taxonomy.slice.yaml", "fixture"),
            "analysis_commit": "a" * 40,
            "judge_model": "objective",
            "scope": {"agents": ["agent"], "benchmarks": ["benchmark"], "languages": ["Java"]},
            "held_out_tasks": ["task1", "task2"],
            "allocation": "paired_randomized",
            "randomization_seed": 1,
            "oracle_free_runtime_reviewed": True,
            "endpoints": self.entry.expected_effect,
            "cost_unit": "USD",
            "max_cost_increase_fraction": 0,
            "max_p2p_drop": 0,
        }
        self.rows = []
        for task in self.protocol["held_out_tasks"]:
            for arm in ("baseline", "active_control", "treatment"):
                run = task + "-" + arm
                treatment = arm == "treatment"
                self.rows.append(
                    {
                        "run_id": run,
                        "task_id": task,
                        "arm": arm,
                        "seed": 1,
                        "agent": "agent",
                        "benchmark": "benchmark",
                        "language": "Java",
                        "resolved": int(treatment),
                        "failure_surfaced": int(treatment),
                        "target_fault": int(not treatment),
                        "p2p_pass_rate": 1,
                        "cost": 1,
                        "log": self.save(run + ".log", run),
                    }
                )

    def save(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value if isinstance(value, str) else json.dumps(value))
        return {"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def frozen(self):
        protocol = self.save("protocol.json", self.protocol)
        record = self.save(
            "results.json",
            {"experiment_id": "synthetic-only", "protocol_sha256": protocol["sha256"], "runs": self.rows},
        )
        return replace(self.entry, verification={"schema_version": 1, "protocol": protocol, "results": record})

    def result(self):
        return validate_verification(self.frozen(), self.root)


class VerificationTests(VerificationFixture):
    def test_complete_paired_record_passes_and_rederives_intervals(self):
        result = self.result()
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(result["comparisons"]), 8)
        self.assertEqual(result["comparisons"][0]["interval"], [1, 1])

    def test_empty_or_malformed_verification_fails_catalog_honesty(self):
        for value in (None, {}, [], {"schema_version": 1}, {"schema_version": 1, "protocol": []}):
            e = replace(self.entry, verification=value)
            self.assertFalse(validate_verification(e, self.root)["valid"])
            self.assertFalse(validate.countermeasure_catalog([e])["checks"]["honesty_invariant"])

    def test_hash_and_artifact_changes_invalidate_evidence(self):
        e = self.frozen()
        (self.root / "task1-baseline.log").write_text("tampered")
        self.assertFalse(validate_verification(e, self.root)["valid"])
        e = self.frozen()
        self.assertFalse(validate_verification(replace(e, artifact_content="changed"), self.root)["valid"])
        for fields in (
            {"error_type": "no_patch_produced"},
            {"hook_event": "pre_submit"},
            {"runtime_trigger": "changed"},
            {"selection_rules": []},
        ):
            self.assertEqual(
                validate_verification(replace(e, **fields), self.root)["reason"], "experiment_or_artifact_mismatch"
            )

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        e = self.frozen()
        for name in ("../outside.json", "/tmp/outside.json", "a/../protocol.json"):
            v = {**e.verification, "protocol": {"path": name, "sha256": "a" * 64}}
            self.assertFalse(validate_verification(replace(e, verification=v), self.root)["valid"])
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / "evidence.json"
            path.write_text("outside")
            (self.root / "escape.json").symlink_to(path)
            v = {
                **e.verification,
                "protocol": {"path": "escape.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            }
            self.assertFalse(validate_verification(replace(e, verification=v), self.root)["valid"])

    def test_unpaired_duplicate_and_out_of_scope_runs_fail(self):
        originals = list(self.rows)
        for rows in (originals[:-1], originals + [originals[0]], [{**r, "language": "Rust"} for r in originals]):
            self.rows = rows
            self.assertFalse(self.result()["valid"])

    def test_baseline_only_improvement_cannot_hide_active_control_failure(self):
        self.rows = [{**r, "resolved": 1} if r["arm"] == "active_control" else r for r in self.rows]
        self.assertFalse(self.result()["valid"])

    def test_target_fault_and_harm_checks_are_required(self):
        original = self.rows
        for field, value in (("target_fault", 1), ("p2p_pass_rate", 0), ("cost", 2), ("cost", True)):
            self.rows = [{**r, field: value} if r["arm"] == "treatment" else r for r in original]
            self.assertFalse(self.result()["valid"], field)

    def test_seed_repeats_do_not_supply_independent_tasks(self):
        self.protocol["held_out_tasks"] = ["task1"]
        self.rows = [r for r in self.rows if r["task_id"] == "task1"]
        self.assertFalse(self.result()["valid"])

    def test_unmeasured_endpoint_or_unbounded_scope_fails(self):
        self.protocol["endpoints"] = ["failure_surfacing"]
        self.assertFalse(self.result()["valid"])
        self.protocol["endpoints"] = self.entry.expected_effect
        self.protocol["scope"]["agents"] = ["all"]
        self.assertFalse(self.result()["valid"])

    def test_uncertain_benefit_and_mixed_context_harm_fail(self):
        self.rows = [
            {**r, "resolved": int(r["arm"] != "treatment")} if r["task_id"] == "task2" else r for r in self.rows
        ]
        self.assertFalse(self.result()["valid"])
        # Additional seeds of a profitable task cannot hide a failing task.
        self.rows += [{**r, "seed": 2, "run_id": r["run_id"] + "-repeat"} for r in self.rows if r["task_id"] == "task1"]
        self.assertFalse(self.result()["valid"])

    def test_nonfinite_derived_cost_cannot_pass_harm_check(self):
        self.protocol["max_cost_increase_fraction"] = 1e308
        self.rows = [{**r, "cost": 1e308} for r in self.rows]
        self.assertFalse(self.result()["valid"])

    def test_verified_selection_requires_known_in_scope_context(self):
        e = self.frozen()
        with patch.object(config, "PROJECT_ROOT", self.root):
            self.assertIsNone(verification_scope(e, {"agent": "agent", "benchmark": "benchmark", "language": "Java"}))
            self.assertEqual(verification_scope(e, {"agent": "agent"})[0], "uncertain")
            self.assertEqual(
                verification_scope(e, {"agent": "agent", "benchmark": "benchmark", "language": "Rust"})[0], "rejected"
            )
            fault = {
                "capability": e.capability,
                "error_type": e.error_type,
                "blame_weight": 1,
                "detector_evidence": {"region_overlap": 0.5},
            }
            r = select_remedies({"agent": "agent"}, fault, entries=[e])
            self.assertFalse(r["countermeasures"])
            self.assertEqual(r["countermeasure_decisions"][0]["reason"], "verification_scope_unavailable")


if __name__ == "__main__":
    unittest.main()


class GoalAlignedVerificationTests(VerificationFixture):
    """Version 2 adds explicit goal policy without changing legacy records."""

    def v2(self, effect="failure_surfacing"):
        self.entry = replace(self.entry, expected_effect=[effect])
        self.protocol.update(
            endpoints=[effect],
            recipe_sha256=recipe_digest(self.entry),
            verification_policy="goal_aligned_v2",
            max_resolution_drop=0,
            require_target_fault_reduction=False,
        )
        if effect == "failure_surfacing":
            self.protocol.update(max_false_alarm_increase=0, negative_control_tasks=["negative1", "negative2"])
            self.protocol["held_out_tasks"] += ["negative1", "negative2"]
            self.rows = [{**r, "resolved": 0, "target_fault": 1, "false_alarm_rate": 0} for r in self.rows]
            for task in self.protocol["negative_control_tasks"]:
                for arm in ("baseline", "active_control", "treatment"):
                    run = task + "-" + arm
                    self.rows.append(
                        {
                            **self.rows[0],
                            "task_id": task,
                            "run_id": run,
                            "arm": arm,
                            "resolved": 1,
                            "failure_surfaced": 0,
                            "target_fault": 0,
                            "log": self.save(run + ".log", run),
                        }
                    )
        else:
            self.rows = [
                {**r, "resolved": 0, "target_fault": 1, "cost": 1 if r["arm"] == "treatment" else 2} for r in self.rows
            ]

    def aligned_result(self):
        entry = self.frozen()
        entry = replace(entry, verification={**entry.verification, "schema_version": 2})
        return validate_verification(entry, self.root)

    def test_surfacing_can_help_without_reducing_underlying_fault(self):
        self.v2()
        result = self.aligned_result()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["verification_policy"], "goal_aligned_v2")
        faults = [c for c in result["comparisons"] if c["endpoint"] == "target_fault_reduction"]
        self.assertTrue(all(not c["required"] and c["interval"] == [0, 0] for c in faults))
        surfaced = [c for c in result["comparisons"] if c["endpoint"] == "failure_surfacing"]
        self.assertTrue(all(c["n_tasks"] == 2 and c["interval"] == [1, 1] for c in surfaced))

    def test_v1_and_requested_mechanism_check_still_require_fault_reduction(self):
        self.v2()
        self.assertFalse(self.result()["valid"])
        self.protocol["require_target_fault_reduction"] = True
        self.assertFalse(self.aligned_result()["valid"])

    def test_efficiency_requires_preserved_resolution_even_when_cost_improves(self):
        self.v2("budget_efficiency")
        self.assertTrue(self.aligned_result()["valid"])
        self.rows = [{**r, "resolved": 1} if r["arm"] != "treatment" else r for r in self.rows]
        self.assertFalse(self.aligned_result()["valid"])

    def test_false_alarms_cannot_be_diluted_by_positive_failure_tasks(self):
        self.v2()
        self.rows = [
            {**r, "false_alarm_rate": 1} if r["task_id"].startswith("negative") and r["arm"] == "treatment" else r
            for r in self.rows
        ]
        self.assertFalse(self.aligned_result()["valid"])

    def test_missing_false_alarm_records_or_controls_fail_closed(self):
        self.v2()
        original = [dict(r) for r in self.rows]
        self.rows = [{k: v for k, v in r.items() if k != "false_alarm_rate"} for r in original]
        self.assertFalse(self.aligned_result()["valid"])
        self.rows = original
        self.protocol["negative_control_tasks"] = ["negative1"]
        self.assertFalse(self.aligned_result()["valid"])

    def test_policy_and_margins_must_be_explicit_not_inferred_after_results(self):
        self.v2()
        for field, value in (
            ("verification_policy", "unknown"),
            ("max_resolution_drop", None),
            ("require_target_fault_reduction", None),
            ("max_false_alarm_increase", True),
        ):
            original = self.protocol[field]
            self.protocol[field] = value
            self.assertFalse(self.aligned_result()["valid"], field)
            self.protocol[field] = original
