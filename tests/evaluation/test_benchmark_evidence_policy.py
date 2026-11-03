import unittest
from agent_trajviz.awe._engine.benchmark import resolve_profile
from agent_trajviz.awe._engine.diffutil import is_test_file
from agent_trajviz.awe._engine.gold import gold_from_requirement
from agent_trajviz.awe._engine.symptoms import analyze
from agent_trajviz.awe._engine.verification import verification_observations
from agent_trajviz.awe._engine.trajectory import NormalizedStep as S, NormalizedTrajectory as T
from agent_trajviz.awe._engine.fragile import suspect_solve, report_md


class BenchmarkEvidencePolicy(unittest.TestCase):
    def test_profile_overrides_roles_and_visibility(self):
        profile = resolve_profile(
            {
                "benchmark_profile": {
                    "gating_tests_visible": False,
                    "test_file_include": ["spec/**"],
                    "test_file_exclude": ["src/test_helpers.py"],
                }
            }
        )
        self.assertTrue(is_test_file("spec/clamp.rs", profile))
        self.assertFalse(is_test_file("src/test_helpers.py", profile))
        g = gold_from_requirement(
            "i",
            {
                "gold_patch": "--- a/src/test_helpers.py\n+++ b/src/test_helpers.py\n@@ -1 +1 @@\n-a\n+b\n",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": ["addition"],
                "benchmark_profile": profile,
            },
        )
        self.assertEqual(g.gold_files, ["src/test_helpers.py"])
        self.assertFalse(g.gating_tests_visible)
        self.assertTrue(g.opportunities["code_verification"])

    def test_malformed_profiles_rejected(self):
        for value in (
            {"evaluator_test_reset": True},
            {"test_file_include": "spec/**"},
            {"test_file_exclude": ["/absolute"]},
            {"resolution_policy": "any_pass"},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                resolve_profile({"benchmark_profile": value})

    def _traj(self, recorded, call_state=None, result_state=None, text="done"):
        return T(
            "a",
            "i",
            [
                S(
                    0,
                    0,
                    "agent",
                    "tool_call",
                    canonical_action="run_test",
                    test_run=True,
                    call_id="t",
                    args={"command": "go test ./...", **({"verification_state": call_state} if call_state else {})},
                ),
                S(
                    1,
                    0,
                    "tool",
                    "tool_result",
                    call_id="t",
                    output_excerpt=text,
                    args={
                        "verification_tests": recorded,
                        **({"verification_state": result_state} if result_state else {}),
                    },
                ),
            ],
        )

    def test_typed_nonpython_counts_and_identifiers(self):
        tests = {
            "complete": True,
            "results": [{"test_id": "TestClamp", "status": "passed"}, {"test_id": "TestAdd", "status": "skipped"}],
        }
        run = verification_observations(self._traj(tests))["runs"][0]
        self.assertEqual(run["summary"]["counts"]["executed"], 1)
        self.assertEqual(run["tests"]["results"][0]["test_id"], "TestClamp")
        self.assertEqual(run["summary"]["runner"], "explicit_test_records")

    def test_duplicate_or_partial_records_cannot_establish_zero(self):
        for tests in (
            {"complete": False, "results": []},
            {"complete": True, "results": [{"test_id": "x", "status": "passed"}] * 2},
        ):
            run = verification_observations(self._traj(tests))["runs"][0]
            self.assertEqual(run["summary"]["status"], "unknown")

    def test_state_conflict_is_unscored_and_not_a_causal_fault(self):
        traj = self._traj({"complete": True, "results": []}, {"revision": "a"}, {"revision": "b"})
        symptoms = analyze(traj)["symptoms"]
        conflict = next(s for s in symptoms if s["kind"] == "verification_state_conflict")
        self.assertEqual(conflict["evidence"]["conflicting_fields"], ["revision"])
        self.assertEqual(conflict["terminal_relevance"], "unknown")
        self.assertEqual(conflict["scoring"], "not_scored")

    def test_report_default_does_not_assume_evaluator_reset(self):
        record = suspect_solve({"gold_tests_edited": ["tests/a.py"]}, True)
        self.assertEqual(record["evaluator_test_reset"], "unknown")
        text = report_md([{**record, "agent": "a", "instance_id": "i"}], 1)
        self.assertIn("unknown", text)
        self.assertNotIn("all reset", text)

    def test_source_role_override_agrees_with_temporal_audit(self):
        from agent_trajviz.awe._engine.trajsignals import compute
        from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED

        profile = resolve_profile({"benchmark_profile": {"test_file_exclude": ["src/test_helpers.py"]}})
        gold = gold_from_requirement("i", {"FAIL_TO_PASS": ["fix"], "PASS_TO_PASS": [], "benchmark_profile": profile})
        traj = T(
            "a",
            "i",
            [
                S(0, 0, "agent", "tool_call", test_run=True, canonical_action="run_test", args={"command": "pytest"}),
                S(1, 1, "agent", "tool_call", canonical_action="edit", files=["src/test_helpers.py"]),
                S(2, 2, "agent", "message", text="done"),
            ],
        )
        self.assertTrue(compute(traj, gold).no_test_after_final_edit)
        fault = {
            "capability": "code_verification",
            "error_type": "no_test_after_final_edit",
            "confidence": "medium",
            "detector_evidence": {"no_test_after_final_edit": True},
        }
        audited = audit_fault(
            fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj, profile=profile
        )
        self.assertEqual(audited["verdict"], RE_DERIVED)
