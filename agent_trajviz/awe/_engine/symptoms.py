"""Behavioral symptom records: evidence for review, never weighted fault labels."""

from __future__ import annotations

from .verification import verification_observations, tool_observations, ref
from .contradictions import contradictions
from .oracles import oracle_candidates

SCHEMA_VERSION = 2


def zero_test_symptoms(observations):
    """Known zero execution; missing counts are unknown, not missing verification.

    A later identical command with positive execution is a bounded observational
    fact, even when its tests fail. It does not establish terminal recovery.
    """
    rows = []
    runs = observations.get("runs", [])
    for i, run in enumerate(runs):
        summary = run["summary"]
        counts = summary.get("counts") or {}
        n = counts.get("executed")
        if summary.get("status") != "reported" or type(n) is not int or n != 0 or run.get("result") is None:
            continue
        skipped = counts.get("skipped", 0)
        kind = "all_tests_skipped" if type(skipped) is int and skipped > 0 else "zero_tests_executed"
        later = next(
            (
                r
                for r in runs[i + 1 :]
                if r["call"]["thread"] == run["call"]["thread"]
                and r["command"].strip() == run["command"].strip()
                and r["call"]["position"] > run["result"]["position"]
                and r["summary"].get("status") == "reported"
                and type((r["summary"].get("counts") or {}).get("executed")) is int
                and r["summary"]["counts"]["executed"] > 0
            ),
            None,
        )
        rows.append(
            {
                "symptom_id": f"{kind}:{run['call']['position']}",
                "kind": kind,
                "observation_strength": "reported_runner_summary",
                "call": run["call"],
                "result": run["result"],
                "command": run["command"],
                "evidence": {"runner": summary["runner"], "counts": dict(counts)},
                "lifecycle": "later_execution_observed" if later else "no_later_execution_observed",
                "later_execution": {"call": later["call"], "result": later["result"]} if later else None,
                "terminal_relevance": "unknown",
                "causal_validation": "not_tested",
                "scoring": "not_scored",
            }
        )
    return rows


def evidence_gap_symptoms(traj, verification):
    """Explicit gaps/conflicts; no assumption that an attempt caused failure."""
    rows = []

    def add(kind, call, result, evidence):
        rows.append(
            {
                "symptom_id": f"{kind}:{call['position']}",
                "kind": kind,
                "call": call,
                "result": result,
                "evidence": evidence,
                "observation_strength": "observed_evidence_structure",
                "lifecycle": "unknown",
                "terminal_relevance": "unknown",
                "causal_validation": "not_tested",
                "scoring": "not_scored",
            }
        )

    for cp, call, rp, result, linkage in tool_observations(traj):
        if result is None:
            kind = "ambiguous_tool_linkage" if "ambiguous" in linkage else "unanswered_tool_call"
            add(kind, ref(cp, call), None, {"linkage": linkage, "status": call.status})
    for run in verification["runs"]:
        if run["summary"].get("status") == "unknown":
            add(
                "verification_evidence_unknown",
                run["call"],
                run["result"],
                {"reason": run["summary"].get("reason"), "tests": run.get("tests", {})},
            )
        state = run["state"]
        values = {}
        for record in state["records"]:
            for key, value in record["fields"].items():
                values.setdefault(key, []).append(value)
        conflicts = [key for key, vals in values.items() if any(v != vals[0] for v in vals[1:])]
        if conflicts or state.get("conflicts"):
            add(
                "verification_state_conflict",
                run["call"],
                run["result"],
                {"conflicting_fields": conflicts, "recorded_state": state},
            )
    return rows


def analyze(traj, patch_text=""):
    verification = verification_observations(traj)
    oracle = oracle_candidates(patch_text)
    return {
        "schema_version": SCHEMA_VERSION,
        "available": traj is not None,
        "verification": verification,
        "oracle_coverage": oracle["coverage"],
        "symptoms": zero_test_symptoms(verification)
        + evidence_gap_symptoms(traj, verification)
        + contradictions(traj, verification)
        + oracle["symptoms"],
        "scope": "behavioral evidence for review; not fault candidates or causal claims",
        "scoring": "not_scored",
    }
