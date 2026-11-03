"""Run-specific experiment plans derived from the shared remedy selector.

Coverage ranks candidates; it never estimates intervention benefit or changes
attribution. Agent prerequisites and bundle composition remain unverified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from . import config
from .countermeasures import CATALOG_PATH, bundles
from .remedies import select_remedies

POLICY = (
    "Core diagnostic candidates first; then unique primary matching cases, "
    "then unique supported matching cases; ties by recipe ID. Coverage is "
    "measured within failed cases and does not predict benefit. "
    "Judge-grounded completeness recipes are exploratory, outside the shortlist."
)
PREREQUISITES = [
    "Confirm that this agent exposes the scaffold hook and required tools.",
    "Adapt build/test commands and path conventions to the project language and benchmark.",
    "Check the runtime trigger using only issue text, the working tree and observed events; exclude reference patches and hidden F2P/P2P information.",
    "Check resource limits, trigger caps and the escape hatch before enabling a gate.",
    "Implement and smoke-test one intervention; catalog text is a specification, not an installed policy.",
    "Keep solved tasks and benign traces as negative controls to detect unnecessary blocking.",
]
MEASUREMENT = {
    "design": "Prespecify scope/endpoints/margins; use held-out tasks and matched seeds with randomized execution order.",
    "arms": [
        "baseline: original agent",
        "active_control: neutral action with equivalent intervention budget",
        "treatment: the candidate intervention",
    ],
    "measure": [
        "declared expected_effect endpoints",
        "target-fault recurrence using a frozen analyzer",
        "F2P completion and overall resolution",
        "P2P preservation",
        "total cost and latency",
        "blocking/escape frequency on solved and benign controls",
    ],
    "evidence": "Preserve protocol, artifact, analyzer revision, provider/model, paired run IDs and hashed logs; report unsuccessful trials too.",
    "promotion": "Use docs/REMEDY_VERIFICATION.md for scoped evidence checks; no status is upgraded by generating this plan.",
}


def cases_digest(cases):
    return hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build_plan(cases, agents=None, instances=None, benchmark=None):
    """Return deterministic JSON, including explicit scope for empty/partial runs."""
    agents = sorted(set(agents if agents is not None else (c["agent"] for c in cases)))
    instances = sorted(set(instances if instances is not None else (c["instance_id"] for c in cases)))
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recommendation case IDs")
    if any(c["agent"] not in agents or c["instance_id"] not in instances for c in cases):
        raise ValueError("Recommendation cases fall outside the requested run scope")
    bundle_of = {member: bundle["id"] for bundle in bundles() for member in bundle["members"]}
    found, refuted, unavailable, suspect, unknown = {}, [], [], [], []
    decisions = Counter()
    for case in sorted(cases, key=lambda c: c["case_id"]):
        cid, agent = case["case_id"], case["agent"]
        if (case.get("fault_side") or {}).get("side") == "grader":
            suspect.append(cid)
        if case.get("blame_status") == "unattributed":
            unknown.append(cid)
        for fault in case.get("faults", []):
            selected = select_remedies(case, fault)
            if selected["recommendation_support"] == "refuted":
                refuted.append({"case_id": cid, "error_type": fault["error_type"]})
            for decision in selected["countermeasure_decisions"]:
                decisions[decision["decision"] + ":" + decision["reason"]] += 1
                if decision["decision"] == "uncertain":
                    unavailable.append({"case_id": cid, "error_type": fault["error_type"], **decision})
            for recipe in selected["countermeasures"]:
                rid = recipe["id"]
                if rid not in found:
                    found[rid] = {
                        "recipe": recipe,
                        "matched_cases": set(),
                        "primary_cases": set(),
                        "per_agent": {},
                        "fault_types": set(),
                        "benchmark_suspects": set(),
                    }
                item = found[rid]
                item["matched_cases"].add(cid)
                item["fault_types"].add(fault["error_type"])
                if cid in suspect:
                    item["benchmark_suspects"].add(cid)
                stats = item["per_agent"].setdefault(agent, {"matched_cases": set(), "primary_cases": set()})
                stats["matched_cases"].add(cid)
                if fault.get("is_primary"):
                    item["primary_cases"].add(cid)
                    stats["primary_cases"].add(cid)
    candidates = []
    for rid, item in sorted(found.items()):
        recipe = item["recipe"]

        def counts(stats):
            return {
                "matching_failed_cases": len(stats["matched_cases"]),
                "primary_matching_cases": len(stats["primary_cases"]),
                "case_ids": sorted(stats["matched_cases"]),
            }

        candidates.append(
            {
                "id": rid,
                "recipe": recipe,
                "bundle_id": bundle_of.get(rid),
                "bundle_status": "composition_unverified",
                "fault_types": sorted(item["fault_types"]),
                **counts(item),
                "primary_case_ids": sorted(item["primary_cases"]),
                "by_agent": {a: counts(s) for a, s in sorted(item["per_agent"].items())},
                "benchmark_suspect_case_ids": sorted(item["benchmark_suspects"]),
                "prerequisites": PREREQUISITES,
                "measurement": {**MEASUREMENT, "expected_effect": recipe["expected_effect"]},
                "next_step": "Check prerequisites, implement a pilot and evaluate against both controls.",
                "deployment_status": "prerequisites_unverified",
            }
        )
    core = [c for c in candidates if c["recipe"]["catalog_role"] == "core"]

    def rank(items, agent=None):
        def key(c):
            stats = c if agent is None else c["by_agent"][agent]
            return (-stats["primary_matching_cases"], -stats["matching_failed_cases"], c["id"])

        return [c["id"] for c in sorted(items, key=key)[:3]]

    return {
        "schema_version": 1,
        "source_cases_sha256": cases_digest(cases),
        "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "scope": {
            "agents": agents,
            "instance_ids": instances,
            "benchmark": benchmark or config.BENCHMARK_NAME,
            "diagnosed_failed_cases": len(cases),
        },
        "prioritization_policy": POLICY,
        "candidates": candidates,
        "shortlist": {
            "overall": rank(core),
            "by_agent": {a: rank([c for c in core if a in c["by_agent"]], a) for a in agents},
        },
        "exploratory_recipe_ids": [c["id"] for c in candidates if c["recipe"]["catalog_role"] != "core"],
        "selection_decisions": dict(sorted(decisions.items())),
        "review_queues": {
            "refuted_faults_without_advice": refuted,
            "uncertain_recipe_selection": unavailable,
            "benchmark_suspect_case_ids": suspect,
            "unattributed_case_ids": unknown,
        },
        "limitations": [
            "Matching diagnoses are candidates for intervention, not proven causal mechanisms.",
            "Unique case counts can overlap between remedies; do not sum them as prevented failures.",
            "Primary counts reflect AWE attribution policy; contributing faults may still have useful remedies.",
            "Language compatibility, hook availability and cost feasibility require checks before installation.",
            "Benchmark suspects remain in diagnostic counts; reproduce and audit harness evidence before assigning a remedy.",
            "Do not install all bundle members together without testing their interactions.",
            "No agent was rerun and no remedy efficacy was established by generating this plan.",
        ],
    }


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def insights_section(plan, include_link=True):
    """Short actionable entry point for INSIGHTS, without copying recipe payloads."""
    by_id = {c["id"]: c for c in plan["candidates"]}
    lines = [
        "## Recommended experiments",
        "",
        (
            "Read [RECOMMENDATIONS.md](RECOMMENDATIONS.md) for prerequisites, case links and controlled evaluation plans. "
            if include_link
            else "Up to three core candidates to consider first. "
        )
        + "These are proposed experiments; diagnostic frequency does not establish benefit.",
        "",
        "| Candidate | Matching failed cases | Primary matches | Next step |",
        "|---|---:|---:|---|",
    ]
    for rid in plan["shortlist"]["overall"]:
        c = by_id[rid]
        lines.append(
            f"| `{rid}` — {_text(c['recipe']['summary'])} | {c['matching_failed_cases']} | "
            f"{c['primary_matching_cases']} | Check prerequisites, then compare a pilot with baseline and active control |"
        )
    if not plan["shortlist"]["overall"]:
        lines.append(
            "| No supported core candidate | 0 | 0 | Check diagnosis coverage before proposing an intervention |"
        )
    lines += ["", "Ranking: " + plan["prioritization_policy"], ""]
    for rid in plan["shortlist"]["overall"]:
        calibration = by_id[rid]["recipe"].get("calibration") or {}
        if calibration.get("adjacent_negative"):
            lines += [f"**Calibration caution — `{rid}`:** {_text(calibration['adjacent_negative'])}", ""]
    return lines


def to_markdown(plan):
    scope, queues = plan["scope"], plan["review_queues"]
    lines = [
        "# Run recommendation plan",
        "",
        f"Scope: **{scope['diagnosed_failed_cases']} diagnosed failures**, {len(scope['agents'])} agents, "
        f"{len(scope['instance_ids'])} input tasks; {_text(scope['benchmark'])}. "
        "Input task count includes solved or excluded slots when full run scope is supplied.",
        "",
        "**Candidate experiments; prerequisites unverified.** This plan does not execute agents or prove efficacy. "
        "Counts describe matching failed cases; cases may occur under multiple candidates.",
        "",
    ]
    lines += insights_section(plan, include_link=False)
    by_id = {c["id"]: c for c in plan["candidates"]}
    lines += ["## Per-agent shortlist", "", "| Agent | Up to three core candidates |", "|---|---|"]
    for agent, ids in plan["shortlist"]["by_agent"].items():
        labels = []
        for rid in ids:
            stats = by_id[rid]["by_agent"][agent]
            labels.append(
                f"`{rid}` ({stats['matching_failed_cases']} matches; {stats['primary_matching_cases']} primary)"
            )
        lines.append(f"| `{agent}` | " + ("; ".join(labels) or "No supported core candidate") + " |")
    lines += [
        "",
        "## Review before changing the agent",
        "",
        f"- {len(queues['refuted_faults_without_advice'])} refuted fault entries: retained for audit, with no active advice.",
        f"- {len(queues['uncertain_recipe_selection'])} uncertain recipe decisions: obtain missing evidence or resolve audit/context problems first.",
        f"- {len(queues['benchmark_suspect_case_ids'])} benchmark-suspect cases: reproduce harness results, inspect per-test records and check undocumented constraints. Suspicion does not prove benchmark error.",
        f"- {len(queues['unattributed_case_ids'])} unattributed cases: collect missing evidence and revisit diagnosis; do not guess a remedy.",
        "",
        "Queue case IDs and reasons are preserved in [recommendations.json](recommendations.json).",
        "",
        "## Candidate specifications",
        "",
        "The shortlist is a starting point. The complete selected set follows; judge-grounded completeness entries are exploratory. "
        "Bundle IDs identify installation groups, whose composition is unverified. Test one intervention before combining policies.",
        "",
    ]
    for c in plan["candidates"]:
        r = c["recipe"]
        examples = (c["primary_case_ids"] + [i for i in c["case_ids"] if i not in c["primary_case_ids"]])[:3]
        lines += [
            f"### {r['id']} — {_text(r['summary'])}",
            "",
            f"- **Why selected:** {c['matching_failed_cases']} unique failed cases match `{', '.join(c['fault_types'])}` "
            f"and the diagnostic selector; {c['primary_matching_cases']} have primary attribution. "
            f"Evidence: `{r['evidence_signal']}`; role: `{r['catalog_role']}`. This is diagnostic support, not causal validation.",
            f"- **Status:** `{r['status']}`; provenance: `{r['provenance']}`; deployment prerequisites unverified.",
            f"- **Implementation maturity:** `{r.get('artifact_maturity', 'unspecified')}`. "
            f"Evidence scope: {_text((r.get('evidence_assessment') or {}).get('scope_notes', 'Unspecified; review before integration.'))}",
            "- **Examples:** "
            + ", ".join(f"`{i}`" for i in examples)
            + " (search `faults.json` for the evidence chain).",
            f"- **Change location:** `{r['scaffold_layer']}` at `{r['hook_event']}`; bundle `{c['bundle_id']}`.",
            f"- **Integration:** {_text(r['plug_in'])}",
            f"- **Proposed mechanism:** {_text(r['mechanism'])}",
            f"- **Diagnostic signature:** {_text(r['diagnostic_signature'])}",
            f"- **Runtime trigger:** {_text(r['runtime_trigger'])}",
            f"- **Feasibility to confirm:** {_text(r['feasibility'])}",
            f"- **Measure:** {', '.join(r['expected_effect'])}, target-fault recurrence, F2P/P2P, cost and latency against baseline and active control.",
            "",
        ]
        if r.get("contract"):
            lines += ["Guardrail contract: `" + _text(json.dumps(r["contract"], sort_keys=True)) + "`.", ""]
        if r.get("runtime_contract"):
            lines += ["Bounded runtime policy: `" + _text(json.dumps(r["runtime_contract"], sort_keys=True)) + "`.", ""]
        if r.get("calibration"):
            lines += ["Calibration and caveats: `" + _text(json.dumps(r["calibration"], sort_keys=True)) + "`.", ""]
        if c["benchmark_suspect_case_ids"]:
            lines += [
                f"Includes {len(c['benchmark_suspect_case_ids'])} benchmark suspects; audit their harness evidence before treating them as intervention targets.",
                "",
            ]
        lines += ["Implementation specification (requires agent integration):", ""]
        fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", r["artifact"])), default=0))
        lines += [fence, r["artifact"], fence, ""]
    lines += ["## Pilot checklist", ""] + ["- " + p for p in PREREQUISITES]
    lines += [
        "",
        "**Evaluation:** " + MEASUREMENT["design"],
        "",
        "- " + "; ".join(MEASUREMENT["arms"]) + ".",
        "- Measure " + "; ".join(MEASUREMENT["measure"]) + ".",
        "- " + MEASUREMENT["evidence"],
        "- " + MEASUREMENT["promotion"],
        "",
        "## Limits",
        "",
    ]
    lines += ["- " + limit for limit in plan["limitations"]]
    return "\n".join(lines) + "\n"


def write(plan, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recommendations.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RECOMMENDATIONS.md").write_text(to_markdown(plan), encoding="utf-8")


def showcase_summary(plan):
    """Small, reproducible summary for the published dashboard."""
    by_id = {c["id"]: c for c in plan["candidates"]}
    return {
        "scope": plan["scope"],
        "source_cases_sha256": plan["source_cases_sha256"],
        "catalog_sha256": plan["catalog_sha256"],
        "policy": plan["prioritization_policy"],
        "cautions": [
            {"id": rid, "text": by_id[rid]["recipe"]["calibration"]["adjacent_negative"]}
            for rid in plan["shortlist"]["overall"]
            if (by_id[rid]["recipe"].get("calibration") or {}).get("adjacent_negative")
        ],
        "shortlist": [
            {
                "id": rid,
                "summary": by_id[rid]["recipe"]["summary"],
                "status": by_id[rid]["recipe"]["status"],
                "matching_failed_cases": by_id[rid]["matching_failed_cases"],
                "primary_matching_cases": by_id[rid]["primary_matching_cases"],
                "hook_event": by_id[rid]["recipe"]["hook_event"],
                "expected_effect": by_id[rid]["recipe"]["expected_effect"],
            }
            for rid in plan["shortlist"]["overall"]
        ],
    }
