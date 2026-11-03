import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from agent_trajviz.awe._engine.countermeasures import load_catalog
from agent_trajviz.awe._engine.remedies import entry_payload
from agent_trajviz.awe._engine.remedy_runtime import (
    GuardrailController,
    observation,
    symbol_overlap,
    test_status as _test_status,
    verification_gap,
    repair_stagnation,
    self_revert,
)
from agent_trajviz.awe._engine.remedy_verification import recipe_digest
from agent_trajviz.awe._engine.validate import countermeasure_catalog


def run(failed=(), executed=("pkg::TestRepair",), revision="content-1", exit_code=None):
    return {
        "completed": True,
        "exit_code": int(bool(failed)) if exit_code is None else exit_code,
        "executed_ids": list(executed),
        "failed_ids": list(failed),
        "errors": [],
        "revision": revision,
        "scope_ids": list(executed),
    }


class TestBoundedController(unittest.TestCase):
    def setUp(self):
        self.entries = load_catalog()
        self.rid = "CM-LOC-WRONG-1"  # maximum two prompts per whole run

    def test_recipe_limit_does_not_reset_between_files_or_retries(self):
        c = GuardrailController("run-1")
        result = [
            c.decide(self.rid, e, observation("match", "no_overlap"))
            for e in ("file-a-edit", "file-b-edit", "file-a-retry", "stop-reentry")
        ]
        self.assertEqual([r["action"] for r in result], ["warn", "warn", "allow", "allow"])
        self.assertEqual(result[-1]["reason"], "budget_exhausted")

    def test_global_budget_bounds_composed_recipes(self):
        c = GuardrailController("run-1", global_budget=1)
        self.assertEqual(c.decide(self.rid, "a", observation("match", "x"))["action"], "warn")
        self.assertEqual(c.decide("CM-VERIF-FINALTEST-1", "b", observation("match", "x"))["reason"], "budget_exhausted")

    def test_unknowns_are_bounded_not_failures(self):
        c = GuardrailController("run-1")
        result = [c.decide(self.rid, str(i), observation("unknown", "provider_missing")) for i in range(3)]
        self.assertEqual(result[0]["observation_status"], "unknown")
        self.assertEqual(result[2]["action"], "allow")

    def test_benign_and_exception_decisions_do_not_consume_budget(self):
        c = GuardrailController("run-1", global_budget=1)
        for i in range(10):
            self.assertEqual(c.decide(self.rid, str(i), observation("clear", "ok"))["action"], "allow")
            r = c.decide(self.rid, "exc" + str(i), observation("match", "proxy", "traced dependency"))
            self.assertEqual(r["receipt"], "traced dependency")
        self.assertEqual(c.decide(self.rid, "real", observation("match", "x"))["action"], "warn")

    def test_duplicate_delivery_is_not_reprompted_and_conflicts_are_rejected(self):
        c = GuardrailController("run-1")
        o = observation("match", "x")
        c.decide(self.rid, "event-1", o)
        self.assertTrue(c.decide(self.rid, "event-1", o)["replayed"])
        self.assertEqual(len(c.snapshot()["events"]), 1)
        with self.assertRaises(ValueError):
            c.decide(self.rid, "event-1", observation("match", "different observation"))

    def test_restart_preserves_run_budget(self):
        c = GuardrailController("run-1")
        for i in range(2):
            c.decide(self.rid, str(i), observation("match", "x"))
        restarted = GuardrailController("run-1", snapshot=c.snapshot())
        self.assertEqual(restarted.decide(self.rid, "retry", observation("match", "x"))["reason"], "budget_exhausted")
        with self.assertRaises(ValueError):
            GuardrailController("different-run", snapshot=c.snapshot())
        with self.assertRaises(ValueError):
            GuardrailController("run-1", global_budget=20, snapshot=c.snapshot())

    def test_snapshot_cannot_alias_controller_state(self):
        c = GuardrailController("run-1")
        snapshot = c.snapshot()
        snapshot["recipes"].clear()
        snapshot["events"].append({"recipe_id": self.rid})
        self.assertTrue(c.snapshot()["recipes"])
        self.assertEqual(c.snapshot()["events"], [])

    def test_invalid_or_duplicate_snapshot_events_are_rejected(self):
        c = GuardrailController("run-1")
        c.decide(self.rid, "a", observation("match", "x"))
        snapshot = c.snapshot()
        snapshot["events"].append(copy.deepcopy(snapshot["events"][0]))
        with self.assertRaises(ValueError):
            GuardrailController("run-1", snapshot=snapshot)

    def test_changed_recipe_semantics_invalidate_snapshot(self):
        c = GuardrailController("run-1")
        altered = [replace(e, runtime_trigger="different") if e.id == self.rid else e for e in self.entries]
        with self.assertRaises(ValueError):
            GuardrailController("run-1", entries=altered, snapshot=c.snapshot())

    def test_concurrent_delivery_respects_shared_budget(self):
        c = GuardrailController("run-1", global_budget=3)
        ids = [e.id for e in self.entries if e.intervention_type == "guardrail"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: c.decide(ids[i % len(ids)], str(i), observation("match", "x")), range(30))
            )
        self.assertEqual(sum(r["action"] == "warn" for r in results), 3)

    def test_disabled_budget_allows_without_reprompt(self):
        c = GuardrailController("run-1", global_budget=0)
        self.assertEqual(c.decide(self.rid, "a", observation("unknown", "x"))["reason"], "budget_exhausted")

    def test_invalid_runtime_inputs_rejected(self):
        for budget in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                GuardrailController("run-1", global_budget=budget)
        c = GuardrailController("run-1")
        for obs in (
            observation("failure", "x"),
            observation("match", ""),
            observation("match", "x", ""),
            {"status": "match", "reason": "x", "gold": "private"},
        ):
            with self.assertRaises(ValueError):
                c.decide(self.rid, "event", obs)


class TestObservationHelpers(unittest.TestCase):
    def test_symbol_free_issue_is_unknown_and_dependency_exception_allowed(self):
        self.assertEqual(symbol_overlap([], ["fix"])["status"], "unknown")
        self.assertEqual(symbol_overlap(["fn"], None)["status"], "unknown")
        self.assertEqual(symbol_overlap(["fn"], ["fn"])["status"], "clear")
        c = GuardrailController("run-1")
        result = c.decide("CM-LOC-WRONG-1", "a", symbol_overlap(["fn"], ["caller"], "traced fn caller"))
        self.assertEqual(result["action"], "allow")

    def test_revert_is_never_blocked_even_without_next_hypothesis(self):
        c = GuardrailController("run-1")
        self.assertEqual(c.decide("CM-REPAIR-SELFREVERT-1", "a", self_revert(True))["action"], "warn")
        self.assertEqual(
            c.decide("CM-REPAIR-SELFREVERT-1", "b", self_revert(True, "restore known-good"))["action"], "allow"
        )
        self.assertEqual(self_revert(False)["status"], "clear")

    def test_status_requires_actual_completed_execution(self):
        self.assertEqual(_test_status(run()), "passed")
        self.assertEqual(_test_status(run(failed=("pkg::TestRepair",))), "failed")
        for r in (
            run(executed=()),
            {**run(), "completed": False},
            {**run(), "errors": ["collection"]},
            run(exit_code=1),
            run(failed=("pkg::TestRepair",), exit_code=0),
            {**run(), "exit_code": True},
            run(failed=("unknown_id",)),
        ):
            self.assertEqual(_test_status(r), "unknown", r)

    def test_final_verification_handles_no_edits_and_unknowns(self):
        self.assertEqual(verification_gap(False, None, None, None)["status"], "clear")
        self.assertEqual(verification_gap(None, None, None, None)["status"], "unknown")
        self.assertEqual(verification_gap(True, "rev", None, None)["status"], "unknown")
        self.assertEqual(verification_gap(True, "rev", ["test"], None)["status"], "match")

    def test_fresh_results_bind_content_and_executed_scope(self):
        ids = ["pkg::TestRepair"]
        self.assertEqual(verification_gap(True, "content-1", ids, run())["status"], "clear")
        self.assertEqual(verification_gap(True, "content-2", ids, run())["status"], "match")
        self.assertEqual(verification_gap(True, "content-1", ids + ["other"], run())["status"], "match")
        # A failed assertion was executed: it must not provoke an identical missing-retest loop.
        self.assertEqual(verification_gap(True, "content-1", ids, run(failed=ids))["status"], "clear")
        self.assertEqual(verification_gap(True, "content-1", ids, {**run(), "completed": False})["status"], "unknown")

    def test_opaque_cross_language_test_ids_are_preserved(self):
        for tid in ("crate::module::tests::repair", "TestHTTP/GET_case", "package.Class#testFix"):
            self.assertEqual(verification_gap(True, "content-1", [tid], run(executed=[tid]))["status"], "clear")

    def test_exit_code_only_does_not_measure_repair_progress(self):
        self.assertEqual(repair_stagnation(3, [{"exit_code": 1}] * 4)["status"], "unknown")

    def test_repair_progress_and_scope_changes_are_not_stagnation(self):
        tests = ("a", "b", "c")
        records = [
            run(failed=f, executed=tests, revision=str(i)) for i, f in enumerate((tests, ("a", "b"), ("a",), ("a",)))
        ]
        self.assertEqual(repair_stagnation(3, records)["status"], "clear")
        records[-1] = run(failed=("a",), executed=("a",))
        self.assertEqual(repair_stagnation(3, records)["status"], "unknown")

    def test_loop_requires_comparable_stable_failure_evidence(self):
        records = [run(failed=("a",), executed=("a", "b"), revision=str(i)) for i in range(4)]
        self.assertEqual(repair_stagnation(3, records)["status"], "match")
        self.assertEqual(repair_stagnation(2, records)["status"], "clear")
        self.assertEqual(repair_stagnation(3, records[:3])["status"], "unknown")
        self.assertEqual(repair_stagnation(3, [records[0]] * 4)["status"], "unknown")
        changed = records[:3] + [run(failed=("b",), executed=("a", "b"), revision="3")]
        self.assertEqual(repair_stagnation(3, changed)["status"], "clear")


class TestCatalogHonesty(unittest.TestCase):
    def test_offline_report_refresh_preserves_scores_and_examples(self):
        from scripts.refresh_remedies import refresh_report_advice
        from agent_trajviz.awe._engine.report import remedy_lines

        e = load_catalog()[0]
        header = remedy_lines(e, 3)[0]
        old = "## Scores: 87.3\n\n" + header + "\n  obsolete advice\n- `task-id` — evidence\n"
        updated = refresh_report_advice(old)
        self.assertTrue(updated.startswith("## Scores: 87.3\n\n"))
        self.assertTrue(updated.endswith("- `task-id` — evidence\n"))
        self.assertNotIn("obsolete advice", updated)
        self.assertIn("_maturity:_ specification", updated)
        self.assertEqual(refresh_report_advice(updated), updated)
        renamed = updated.replace("**" + e.summary + "**", "**obsolete headline**")
        self.assertEqual(refresh_report_advice(renamed), updated)

    def test_evidence_and_maturity_exported_separately_from_status(self):
        for e in load_catalog():
            p = entry_payload(e)
            self.assertEqual(p["artifact_maturity"], "specification")
            self.assertEqual(p["evidence_assessment"]["exact_recipe_efficacy"], "unmeasured")
            self.assertEqual(p["status"], "untested")

    def test_false_efficacy_annotation_fails_validation(self):
        entry = replace(
            load_catalog()[0],
            evidence_assessment={"exact_recipe_efficacy": "scoped_verified", "scope_notes": "claimed"},
        )
        self.assertFalse(countermeasure_catalog([entry])["checks"]["evidence_scope"])

    def test_boolean_budget_and_runtime_contradiction_fail_validation(self):
        e = load_catalog()[0]
        invalid = replace(e, contract={**e.contract, "max_triggers": True})
        self.assertFalse(countermeasure_catalog([invalid])["checks"]["guardrail_contract"])
        invalid = replace(e, runtime_contract={**e.runtime_contract, "mode": "hard_block"})
        self.assertFalse(countermeasure_catalog([invalid])["checks"]["bounded_runtime"])

    def test_runtime_semantics_change_digest_but_evidence_prose_does_not(self):
        e = load_catalog()[0]
        different = replace(e, runtime_contract={**e.runtime_contract, "max_triggers": 1})
        self.assertNotEqual(recipe_digest(e), recipe_digest(different))
        self.assertEqual(recipe_digest(e), recipe_digest(replace(e, evidence_assessment={"scope_notes": "new prose"})))


if __name__ == "__main__":
    unittest.main()
