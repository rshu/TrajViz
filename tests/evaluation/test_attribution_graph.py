"""Trace provenance, exact decisions, graph integrity and default run outputs."""

from pathlib import Path as _EnginePath
from agent_trajviz.awe import _engine as _engine_package

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trajviz.awe._engine.attribution_graph import build_graphs, validate_dag, write, render_html
from agent_trajviz.awe._engine.remedy_plan import cases_digest
from agent_trajviz.awe._engine.rootcause import build_blame
from agent_trajviz.awe._engine.runs import snapshot

_ENGINE_ROOT = _EnginePath(_engine_package.__file__).resolve().parent

ROOT = _ENGINE_ROOT


def detection(types=("incorrect_patch", "regression_introduced")):
    return {
        "resolved": False,
        "loc_flag": "right_place_wrong_change",
        "candidates": [{"capability": "code_editing", "error_type": et, "confidence": "high"} for et in types],
    }


def case(det=None, adjudicate=None):
    result = build_blame(det or detection(), adjudicate=adjudicate)
    return {
        "case_id": "a__i",
        "agent": "a",
        "instance_id": "i",
        "blame_status": result["status"],
        "attribution_trace": result["decision_trace"],
        "arbiter": result.get("arbiter"),
        "task": {"problem_statement": "Fix the behavior", "gold_files": ["pkg/file.rs"]},
        "faults": [
            {
                "capability": b["capability"],
                "error_type": b["error_type"],
                "confidence": b["confidence"],
                "blame_weight": round(b["weight"], 2),
                "is_primary": b is result["primary"],
                "detector_evidence": {"region_overlap": 0.6},
                "evidence_chain": {
                    "links": [
                        {"kind": "observation", "source": "patch", "statement": "File edited"},
                        {"kind": "conclusion", "source": "rule", "statement": "Candidate emitted"},
                    ],
                    "counterfactual": {"status": "not_tested", "what_would_confirm": "Compare paired interventions"},
                },
            }
            for b in result["blame"]
        ],
    }


class TraceTests(unittest.TestCase):
    def test_exact_tie_preserves_detector_order_without_changing_attribution(self):
        result = build_blame(detection())
        trace = result["decision_trace"]
        self.assertEqual(result["primary"]["error_type"], "incorrect_patch")
        self.assertEqual(trace["selection"]["exact_weight_ties"], ["incorrect_patch", "regression_introduced"])
        self.assertEqual(trace["selection"]["reason"], "stable_detector_order_for_exact_weight_tie")
        self.assertFalse(trace["weights_are_causal_probabilities"])

    def test_rounded_tie_is_not_an_exact_tie(self):
        det = detection()
        det["candidates"][1]["confidence"] = "medium"
        with patch("agent_trajviz.awe._engine.rootcause.CONF", {"high": 1, "medium": 0.9996}):
            c = case(det)
        graph = build_graphs([c])["graphs"][0]
        policy = next(n for n in graph["nodes"] if n["id"] == "policy")
        self.assertEqual(policy["details"]["selection"]["exact_weight_ties"], ["incorrect_patch"])
        self.assertNotIn("stable_detector_order_for_exact_weight_tie", " ".join(policy["lines"]))
        self.assertEqual(c["attribution_trace"]["dominance"]["leading_share"], 1)

    def test_refutation_records_zero_mass_and_cited_review_without_repeating_callback(self):
        det = detection(("relevant_change_omitted", "incorrect_patch"))
        calls = []

        def review(d, p):
            calls.append(p["error_type"])
            return {"support": "refute", "confidence": "high", "evidence_steps": [9], "rationale": "Edit exists"}

        c = case(det, review)
        self.assertEqual(calls, ["relevant_change_omitted"])
        trace = c["attribution_trace"]
        self.assertEqual(trace["adjudication"]["evidence_steps"], [9])
        self.assertEqual(trace["candidates"][0]["final_weight"], 0)
        graph = build_graphs([c])["graphs"][0]
        self.assertEqual(calls, ["relevant_change_omitted"])
        refuted = next(n for n in graph["nodes"] if n["id"] == "fault:1")
        self.assertEqual(refuted["status"], "refuted")
        self.assertTrue(any(e["relation"] == "updates_attribution" for e in graph["edges"]))
        self.assertFalse(any(e["relation"] == "proposes_check" and e["to"] == "fault:1:check" for e in graph["edges"]))
        self.assertEqual(next(n for n in graph["nodes"] if n["id"] == "fault:1:check")["status"], "inactive_refuted")

    def test_contributing_only_and_coverage_gap_do_not_get_a_primary(self):
        det = {
            "resolved": False,
            "loc_flag": "clean",
            "candidates": [
                {"capability": "code_verification", "error_type": "relevant_test_not_run", "confidence": "high"}
            ],
        }
        c = case(det)
        self.assertFalse(c["attribution_trace"]["dominance"]["passes"])
        self.assertFalse(c["attribution_trace"]["candidates"][0]["primary_eligible"])
        self.assertEqual(build_graphs([c])["graphs"][0]["primary_fault_ids"], [])
        empty = case({**det, "candidates": []})
        self.assertEqual(empty["blame_status"], "unattributed")
        self.assertEqual(build_graphs([empty])["graphs"][0]["primary_fault_ids"], [])


class GraphTests(unittest.TestCase):
    def test_reference_links_require_source_role_or_normalized_reference_source(self):
        for source, role, expected in (
            ("golden_runtime.log", None, False),
            ("ground_truth", None, True),
            ("oracle-spec.yaml", "reference", True),
        ):
            with self.subTest(source=source, role=role):
                c = case()
                f = c["faults"][0]
                f["detector_evidence"] = {"invariant_violations": 2}
                f["evidence_chain"]["links"] = [
                    {"kind": "observation", "source": source, "source_role": role, "statement": "Recorded check"}
                ]
                graph = build_graphs([c])["graphs"][0]
                self.assertEqual(
                    any(e["from"] == "reference" and e["to"] == "fault:0:evidence" for e in graph["edges"]), expected
                )

    def test_config_label_honors_profile_override_and_external_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / "profile.json"
            profile.write_text(json.dumps({"name": "Go repository benchmark"}))
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("AWE_BENCHMARK_NAME", "AWE_BENCHMARK_PROFILE", "AWE_ARGUS_ROOT")
            }
            for settings, expected in (
                ({}, "SWE-bench Verified"),
                ({"AWE_ARGUS_ROOT": td}, "unspecified"),
                ({"AWE_BENCHMARK_PROFILE": str(profile)}, "Go repository benchmark"),
                ({"AWE_BENCHMARK_PROFILE": str(profile), "AWE_BENCHMARK_NAME": "Display override"}, "Display override"),
            ):
                with self.subTest(settings=settings):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "from agent_trajviz.awe._engine.config import BENCHMARK_NAME; print(BENCHMARK_NAME)",
                        ],
                        cwd=ROOT,
                        env={**env, **settings},
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    self.assertEqual(result.stdout.strip(), expected)

    def test_standalone_exporter_preserves_scope_and_rejects_conflicting_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = [case()]
            (root / "faults.json").write_text(json.dumps(cases))
            env = {k: v for k, v in os.environ.items() if k not in ("AWE_BENCHMARK_NAME", "AWE_BENCHMARK_PROFILE")}
            command = [sys.executable, str(ROOT / "scripts/gen_attribution.py"), "--dossier", td]

            def run(extra=(), ok=True):
                result = subprocess.run(command + list(extra), cwd=ROOT, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode == 0, ok, result.stderr)
                return json.loads((root / "attribution_graphs.json").read_text())

            self.assertEqual(run()["scope"]["benchmark"], "unspecified")
            self.assertEqual(run(["--benchmark", "Rust benchmark"])["scope"]["benchmark"], "Rust benchmark")
            scope = {"agents": ["a"], "instance_ids": ["i", "solved"], "benchmark": "Rust benchmark"}
            (root / "recommendations.json").write_text(
                json.dumps({"source_cases_sha256": cases_digest(cases), "scope": scope})
            )
            bundle = run()
            self.assertEqual(bundle["scope"]["instance_ids"], ["i", "solved"])
            self.assertEqual(run(["--benchmark", "Wrong benchmark"], ok=False), bundle)

    def test_external_fault_origins_do_not_depend_on_taxonomy_names(self):
        for declared, strength, judge_metadata, expected in (
            ("objective", None, {}, "inference"),
            ("llm", None, {}, "llm"),
            (None, "associational", {}, "inference"),
            (None, "model_inferred", {}, "unknown"),
            (None, None, {}, "unknown"),
            (None, None, {"judge_model": "external-judge"}, "llm"),
            ("objective", None, {"judge_model": "external-judge"}, "unknown"),
        ):
            with self.subTest(declared=declared, strength=strength, judge_metadata=judge_metadata):
                c = case()
                del c["attribution_trace"]
                fault = c["faults"][0]
                fault.update(capability="external_runtime", error_type="protocol_invariant_violated")
                fault["detector_evidence"] = {"invariant_violations": 2, "hit": True, "recall": 0.8, **judge_metadata}
                chain = fault["evidence_chain"]
                if declared is not None:
                    chain["assessment_origin"] = declared
                if strength is not None:
                    chain["observation_strength"] = strength
                graph = build_graphs([c], benchmark="External JVM benchmark")["graphs"][0]
                assessment = next(n for n in graph["nodes"] if n["id"] == "fault:0:assessment")
                self.assertEqual(assessment["provenance"], expected)
                self.assertFalse(
                    any(e["from"] == "reference" and e["to"] == "fault:0:evidence" for e in graph["edges"])
                )

    def test_custom_metrics_and_opaque_test_ids_survive_summary_and_details(self):
        c = case()
        evidence = {
            "region_overlap": 0.4,
            "invariant_violations": 2,
            "f2p_pass_fraction": 0.5,
            "n_p2p_broken": 1,
            "p2p_broken": ["com.acme.CoreTest#preservesState[日本語]"],
            "failed_f2p_ids": ["TestFix/subcase=alpha"],
            "runtime": "JVM",
        }
        c["faults"][0]["detector_evidence"] = evidence
        graph = build_graphs([c], benchmark="External JVM benchmark")["graphs"][0]
        node = next(n for n in graph["nodes"] if n["id"] == "fault:0:evidence")
        self.assertIn("invariant_violations", " ".join(node["lines"]))
        self.assertIn("failed_f2p_ids", " ".join(node["lines"]))
        self.assertEqual(node["details"]["detector_evidence"], evidence)
        self.assertIn("com.acme.CoreTest#preservesState[日本語]", render_html(build_graphs([c])))

    def test_missing_official_metrics_are_not_invented(self):
        c = case()
        c["task"]["gold_files"] = []
        c["faults"][0]["detector_evidence"] = {"invariant_violations": 2}
        graph = build_graphs([c])["graphs"][0]
        self.assertIn("unavailable", next(n for n in graph["nodes"] if n["id"] == "reference")["lines"][0])
        node = next(n for n in graph["nodes"] if n["id"] == "fault:0:evidence")
        self.assertEqual(node["details"]["detector_evidence"], {"invariant_violations": 2})
        self.assertNotIn("f2p", " ".join(node["lines"]))
        self.assertNotIn("p2p", " ".join(node["lines"]))

    def test_build_is_deterministic_and_preserves_source_records(self):
        c = case()
        original = copy.deepcopy(c)
        bundle = build_graphs([c], agents=["a"], instances=["i", "solved"])
        self.assertEqual(bundle, build_graphs([c], agents=["a"], instances=["i", "solved"]))
        self.assertEqual(c, original)
        graph = bundle["graphs"][0]
        validate_dag(graph["nodes"], graph["edges"])
        self.assertTrue(all(not edge["causal"] for edge in graph["edges"]))
        self.assertEqual(next(n for n in graph["nodes"] if n["id"] == "fault:0:evidence")["number"], "2.1")
        self.assertTrue(all(n["source"]["artifact"] == "faults.json" for n in graph["nodes"]))

    def test_legacy_record_does_not_invent_exact_trace_or_tie_reason(self):
        c = case()
        del c["attribution_trace"]
        graph = build_graphs([c])["graphs"][0]
        self.assertFalse(graph["trace_available"])
        policy = next(n for n in graph["nodes"] if n["id"] == "policy")
        self.assertIn("exact decision trace unavailable", " ".join(policy["lines"]))

    def test_changed_trace_or_scope_fails_closed(self):
        c = case()
        c["attribution_trace"]["selection"]["primary"]["error_type"] = "different"
        with self.assertRaisesRegex(ValueError, "contradicts"):
            build_graphs([c])
        with self.assertRaisesRegex(ValueError, "scope"):
            build_graphs([case()], agents=["b"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_graphs([case(), case()])

    def test_cycle_dangling_and_duplicate_edges_are_rejected(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edge = {"from": "a", "to": "b", "relation": "documents"}
        for edges in (
            [edge, {"from": "b", "to": "a", "relation": "selects"}],
            [{"from": "a", "to": "missing", "relation": "documents"}],
            [edge, edge],
        ):
            with self.assertRaises(ValueError):
                validate_dag(nodes, edges)

    def test_action_edges_use_original_positions_and_threads(self):
        c = case()

        def ref(pos, thread):
            return {"position": pos, "step_id": 1, "thread": thread, "call_id": "reused"}

        c["event_ledger"] = {
            "available": True,
            "events": [
                {
                    "action": "run_test",
                    "links": [{"from": ref(0, "one"), "to": ref(1, "one"), "relation": "tool_result"}],
                },
                {
                    "action": "run_test",
                    "links": [{"from": ref(2, "two"), "to": ref(3, "two"), "relation": "tool_result"}],
                },
            ],
        }
        action = build_graphs([c])["graphs"][0]["action_graph"]
        self.assertEqual(len(action["nodes"]), 4)
        self.assertEqual(len({n["id"] for n in action["nodes"]}), 4)
        c["event_ledger"]["events"][0]["links"][0]["to"]["position"] = 0
        with self.assertRaisesRegex(ValueError, "order|Conflicting"):
            build_graphs([c])

    def test_judge_observations_are_not_presented_as_objective_rules(self):
        det = {
            "resolved": False,
            "loc_flag": "clean",
            "candidates": [{"capability": "task_planning", "error_type": "flawed_plan", "confidence": "high"}],
        }
        graph = build_graphs([case(det)])["graphs"][0]
        self.assertEqual(next(n for n in graph["nodes"] if n["phase"] == 3)["provenance"], "llm")
        self.assertFalse(any(e["from"] == "reference" for e in graph["edges"]))
        self.assertEqual(graph["primary_fault_ids"], [])

    def test_unavailable_sources_are_marked_and_no_steps_are_fabricated(self):
        c = case()
        del c["faults"][0]["detector_evidence"]
        graph = build_graphs([c])["graphs"][0]
        node = next(n for n in graph["nodes"] if n["id"] == "fault:0:evidence")
        self.assertFalse(node["source"]["available"])
        self.assertEqual(graph["trajectory_excerpt"], [])

    def test_objective_anchor_in_judge_capability_keeps_rule_provenance(self):
        det = {
            "resolved": False,
            "loc_flag": "loc_miss",
            "candidates": [
                {
                    "capability": "requirement_understanding",
                    "error_type": "solved_unrelated_problem",
                    "confidence": "medium",
                }
            ],
        }
        graph = build_graphs([case(det)])["graphs"][0]
        self.assertEqual(next(n for n in graph["nodes"] if n["phase"] == 3)["provenance"], "inference")
        self.assertEqual(graph["primary_fault_ids"], [])

    def test_embedded_issue_text_cannot_break_out_of_script(self):
        c = case()
        c["task"]["problem_statement"] = "</script><script>window.pwned=true</script>"
        html = render_html(build_graphs([c]))
        self.assertNotIn(c["task"]["problem_statement"], html)
        self.assertIn("\\u003c/script>", html)
        self.assertNotIn("fetch(", html)

    def test_empty_run_gets_fresh_viewer_and_is_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = build_graphs([], agents=["a"], instances=["i"])
            write(bundle, root / "dossier")
            (root / "dossier/faults.json").write_text("[]")
            dest = snapshot(
                "empty",
                dossier_dir=root / "dossier",
                reports_dir=root / "reports",
                runs_dir=root / "runs",
                extra={"agents": ["a"], "instance_ids": ["i"]},
            )
            manifest = json.loads((dest / "manifest.json").read_text())
            self.assertIn("ATTRIBUTION.html", manifest["artifacts"])
            self.assertEqual(manifest["attribution_graphs"]["n_graphs"], 0)
            self.assertIn("No diagnosed failures", (dest / "ATTRIBUTION.html").read_text())

    def test_snapshot_rejects_stale_viewer_and_scope_before_creating_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, c = Path(tmp), case()
            bundle = build_graphs([c])
            write(bundle, root / "dossier")
            (root / "dossier/faults.json").write_text(json.dumps([c]))
            (root / "dossier/ATTRIBUTION.html").write_text("stale")
            with self.assertRaisesRegex(ValueError, "HTML"):
                snapshot("bad", dossier_dir=root / "dossier", runs_dir=root / "runs")
            self.assertFalse((root / "runs").exists())
            write(bundle, root / "dossier")
            with self.assertRaisesRegex(ValueError, "scope"):
                snapshot("bad", dossier_dir=root / "dossier", runs_dir=root / "runs", extra={"agents": ["b"]})


if __name__ == "__main__":
    unittest.main()
