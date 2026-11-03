"""Versioned RSI exchange contracts and a bounded, oracle-free policy prototype.

Exports propose experiments, not installations or measured improvements. Runtime
inputs are intentionally separate from privileged diagnostic feedback.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

from . import config
from .countermeasures import load_catalog
from .remedy_plan import build_plan, cases_digest
from .remedies import fault_support
from .remedy_verification import recipe_digest, verification_scope

ADAPTERS = config.PROJECT_ROOT / "configs/rsi_adapters.json"
POLICY = {
    "schema_version": 1,
    "kind": "hook_policy",
    "event": "pre_submit",
    "operation": "request_fresh_local_verification",
    "allowed_inputs": ["workspace_revision", "verification", "tests_available"],
    "required_capabilities": [
        "hook.pre_submit",
        "tool.project_tests",
        "state.workspace_revision",
        "state.local_verification",
    ],
    "max_triggers_per_run": 2,
    "on_unknown": "warn",
    "on_exhaustion": "warn_and_continue",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def adapters():
    doc = json.loads(ADAPTERS.read_text(encoding="utf-8"))
    if (
        set(doc) != {"schema_version", "adapters"}
        or type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
    ):
        raise ValueError("Invalid RSI adapter registry")
    if not isinstance(doc["adapters"], dict):
        raise ValueError("Invalid RSI adapters")
    for entry in doc["adapters"].values():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"adapter_id", "implementation"}
            or entry["adapter_id"] != "awe.local-verification-policy.v1"
            or entry["implementation"] != POLICY
        ):
            raise ValueError("Unsupported or altered RSI adapter implementation")
    return doc["adapters"]


def _validate_context(context):
    allowed = {
        "agent",
        "benchmark",
        "languages",
        "model",
        "provider",
        "scaffold_revision",
        "capabilities",
        "provenance",
    }
    if not isinstance(context, dict) or set(context) - allowed:
        raise ValueError("Unknown RSI execution-context fields")
    for key, value in context.items():
        if value is None:
            continue
        if key in ("languages", "capabilities"):
            if (
                not isinstance(value, list)
                or any(not isinstance(v, str) or not v.strip() for v in value)
                or len(value) != len(set(value))
            ):
                raise ValueError("Invalid RSI execution-context list")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError("Invalid RSI execution-context identity")


def _normalized_context(context):
    _validate_context(context)
    return {
        k: sorted(v) if k in ("languages", "capabilities") and isinstance(v, list) else v
        for k, v in context.items()
        if k != "provenance"
    }


def compatibility(context, implementation):
    _validate_context(context)
    if implementation is None:
        return {"status": "requires_adapter", "reasons": ["no_executable_adapter_registered"]}
    missing = [
        k
        for k in ("benchmark", "languages", "model", "provider", "scaffold_revision")
        if not context.get(k) or (k == "benchmark" and context[k] == "unspecified")
    ]
    capabilities = context.get("capabilities")
    if capabilities is None:
        missing.append("capabilities")
    if missing:
        return {"status": "unknown", "reasons": ["missing_context:" + k for k in missing]}
    absent = sorted(set(implementation["required_capabilities"]) - set(capabilities))
    if absent:
        return {"status": "incompatible", "reasons": ["missing_capability:" + c for c in absent]}
    return {"status": "declared_compatible", "reasons": ["runtime_adapter_checks_still_required"]}


def build_exchange(cases, plan):
    scope = plan["scope"]
    expected = build_plan(cases, agents=scope["agents"], instances=scope["instance_ids"], benchmark=scope["benchmark"])
    if plan != expected:
        raise ValueError("Cannot export RSI contracts from a stale recommendation plan")
    registry = adapters()
    entries = {e.id: e for e in load_catalog()}
    source = cases_digest(cases)
    plan_hash = digest(plan)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    proposals, feedback = [], []
    indices = {c["case_id"]: i for i, c in enumerate(cases)}
    by_id = {c["case_id"]: c for c in cases}
    for candidate in plan["candidates"]:
        entry = entries[candidate["id"]]
        descriptor = registry.get(entry.id)
        implementation = descriptor["implementation"] if descriptor else None
        groups = {}
        for case_id in candidate["case_ids"]:
            case = by_id[case_id]
            context = {
                "benchmark": case.get("benchmark"),
                "languages": [case["language"]] if case.get("language") else [],
                "model": None,
                "provider": None,
                "scaffold_revision": None,
                "capabilities": None,
                **(case.get("execution_context") or {}),
                "agent": case["agent"],
            }
            context = _normalized_context(context)
            if (case.get("benchmark") and case["benchmark"] != context["benchmark"]) or (
                case.get("language") and context["languages"] != [case["language"]]
            ):
                raise ValueError("Case and RSI execution context disagree")
            key = digest(context)
            groups.setdefault(key, {"context": context, "eligible": [], "review": []})
            queue = "review" if case_id in candidate["benchmark_suspect_case_ids"] else "eligible"
            groups[key][queue].append(case_id)
        for key, group in sorted(groups.items()):
            context = group["context"]
            status = compatibility(context, implementation)
            if not group["eligible"]:
                status = {"status": "review_required", "reasons": ["only_benchmark_suspect_cases"]}
            proposals.append(
                {
                    "proposal_id": entry.id
                    + ":"
                    + digest([source, plan_hash, recipe_digest(entry), key, descriptor, code_hash])[:20],
                    "recipe_id": entry.id,
                    "recipe_sha256": recipe_digest(entry),
                    "source_cases_sha256": source,
                    "recommendation_plan_sha256": plan_hash,
                    "context": context,
                    "compatibility": status,
                    "target": {"capability": entry.capability, "error_type": entry.error_type},
                    "change": descriptor,
                    "adapter_code_sha256": code_hash if descriptor else None,
                    "evidence_refs": [
                        {"case_id": cid, "artifact": "faults.json", "json_pointer": "/" + str(indices[cid])}
                        for cid in sorted(group["eligible"])
                    ],
                    "review_case_ids": sorted(group["review"]),
                    "experiment": candidate["measurement"],
                    "constraints": {
                        "held_out_split_required": True,
                        "solved_negative_controls_required": True,
                        "single_intervention_first": True,
                        "promotion_requires_completed_trials": True,
                    },
                    "recipe_evidence_status": entry.status,
                    "variant_efficacy_status": "untested",
                    "artifact_maturity": entry.artifact_maturity,
                    "evidence_assessment": entry.evidence_assessment,
                    "runtime_contract": entry.runtime_contract,
                    "deployment_status": "not_installed",
                }
            )
    for index, case in enumerate(cases):
        feedback.append(
            {
                "case_id": case["case_id"],
                "agent": case["agent"],
                "instance_id": case["instance_id"],
                "outcome": "failed",
                "attribution_status": case["blame_status"],
                "context": case.get("execution_context"),
                "benchmark_suspect": (case.get("fault_side") or {}).get("side") == "grader",
                "faults": [
                    {
                        "capability": f["capability"],
                        "error_type": f["error_type"],
                        "primary": bool(f.get("is_primary")),
                        "support": fault_support(case, f),
                        "audit": (f.get("audit") or {}).get("verdict"),
                        "confidence": f.get("confidence"),
                        "observation_strength": (f.get("evidence_chain") or {}).get(
                            "observation_strength", (f.get("evidence_chain") or {}).get("strength")
                        ),
                        "causal_support": (f.get("evidence_chain") or {}).get("causal_support"),
                        "evidence_ref": {"artifact": "faults.json", "json_pointer": f"/{index}/faults/{j}"},
                    }
                    for j, f in enumerate(case.get("faults", []))
                ],
            }
        )
    common = {
        "schema_version": 1,
        "source_cases_sha256": source,
        "recommendation_plan_sha256": plan_hash,
        "scope": scope,
    }
    return (
        {
            **common,
            "adapter_registry_sha256": hashlib.sha256(ADAPTERS.read_bytes()).hexdigest(),
            "proposals": proposals,
            "agent_runs_performed": 0,
            "efficacy_claimed": False,
        },
        {
            **common,
            "access_class": "optimizer_development_only",
            "runtime_input_allowed": False,
            "cases": feedback,
            "coverage": "Diagnosed failed cases only; absent scoped tasks have no outcome in this packet.",
            "objective_results_source": "Use official evaluator records for resolution, F2P/P2P and costs; diagnostic counts are not rewards.",
        },
    )


def write_exchange(cases, plan, output):
    proposals, feedback = build_exchange(cases, plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("rsi_proposals.json", proposals), ("rsi_feedback.json", feedback)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(proposal, observed_context):
    """Prepare a portable policy after controller-reported capability checks.

    No process is started, no hook installed, and no measured benefit inferred.
    A native scaffold adapter must enforce isolation and report the actual install.
    """
    observed_context = _normalized_context(observed_context)
    registry = adapters()
    recipe = next((e for e in load_catalog() if e.id == proposal["recipe_id"]), None)
    descriptor = registry.get(proposal["recipe_id"])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        recipe is None
        or recipe.status not in ("untested", "verified")
        or proposal.get("target") != {"capability": recipe.capability, "error_type": recipe.error_type}
        or proposal["recipe_sha256"] != recipe_digest(recipe)
        or descriptor is None
        or descriptor != proposal["change"]
        or proposal.get("adapter_code_sha256") != code_hash
        or proposal["compatibility"]["status"] != "declared_compatible"
        or not proposal.get("evidence_refs")
    ):
        raise ValueError("Stale, unsupported or review-only RSI proposal")
    for key in ("agent", "benchmark", "languages", "model", "provider", "scaffold_revision"):
        if observed_context.get(key) != proposal["context"].get(key):
            raise ValueError("Observed execution context differs from proposal")
    if compatibility(observed_context, descriptor["implementation"])["status"] != "declared_compatible":
        raise ValueError("Runtime capabilities do not support this proposal")
    if recipe.status == "verified":
        scoped_case = {
            "agent": observed_context["agent"],
            "benchmark": observed_context["benchmark"],
            "language": observed_context["languages"][0] if len(observed_context["languages"]) == 1 else None,
        }
        if verification_scope(recipe, scoped_case):
            raise ValueError("Recipe verification is invalid or outside observed scope")
    artifact = {
        "recipe_id": recipe.id,
        "recipe_sha256": recipe_digest(recipe),
        "policy": descriptor["implementation"],
        "adapter_code_sha256": code_hash,
        "context": observed_context,
    }
    return {
        **artifact,
        "variant_id": digest(artifact),
        "status": "prepared_not_installed",
        "variant_efficacy_status": "untested",
    }


def decide(prepared, state, trigger_count=0):
    """Evaluate only allowlisted local state; return a bounded action request."""
    artifact = {k: prepared[k] for k in ("recipe_id", "recipe_sha256", "policy", "adapter_code_sha256", "context")}
    if prepared.get("variant_id") != digest(artifact) or prepared["policy"] != POLICY:
        raise ValueError("Prepared policy integrity mismatch")
    if prepared["adapter_code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Prepared adapter code is stale")
    if not isinstance(state, dict) or set(state) - set(POLICY["allowed_inputs"]):
        raise ValueError("Runtime state contains non-allowlisted inputs")
    if type(trigger_count) is not int or trigger_count < 0:
        raise ValueError("Invalid per-run trigger count")

    def result(action, reason, count=trigger_count):
        return {"action": action, "reason": reason, "next_trigger_count": count, "blocks_submission": False}

    revision, verification = state.get("workspace_revision"), state.get("verification")
    if (
        not isinstance(revision, str)
        or not revision
        or type(state.get("tests_available")) is not bool
        or not isinstance(verification, dict)
        or set(verification) != {"revision", "status"}
        or verification["status"] not in ("passed", "failed", "not_run", "unknown")
        or (verification["revision"] is not None and not isinstance(verification["revision"], str))
    ):
        return result("warn", "local_evidence_unknown")
    if not state["tests_available"]:
        return result("warn", "project_tests_unavailable")
    if verification["status"] == "unknown" or (
        verification["status"] in ("passed", "failed") and not verification["revision"]
    ):
        return result("warn", "local_evidence_unknown")
    if verification["revision"] == revision and verification["status"] in ("passed", "failed"):
        return result(
            "allow" if verification["status"] == "passed" else "warn",
            "fresh_local_verification" if verification["status"] == "passed" else "repair_required",
        )
    if trigger_count >= POLICY["max_triggers_per_run"]:
        return result("warn", "trigger_budget_exhausted")
    return result("request_project_tests", "local_verification_stale_or_not_run", trigger_count + 1)
