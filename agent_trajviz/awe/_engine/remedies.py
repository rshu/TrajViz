"""Case-specific remedy selection, separate from catalog lookup and attribution.

Selection is diagnostic-side: reference-region evidence may distinguish variants.
It never executes an intervention or evaluates its oracle-free runtime trigger.
"""

from __future__ import annotations

import math

from .countermeasures import countermeasures_for
from .remedy_verification import verification_scope

REFUTATIONS = {"demoted_to_conjunctive", "refuted_reassigned", "unattributed_after_refute"}


def valid_selection_rules(rules):
    if not isinstance(rules, list):
        return False
    return all(
        isinstance(r, dict)
        and set(r) == {"field", "op", "value"}
        and isinstance(r["field"], str)
        and bool(r["field"])
        and r["op"] in ("eq", "gt", "gte", "lt", "lte")
        and type(r["value"]) in (int, float)
        and math.isfinite(r["value"])
        for r in rules
    )


def _match(rules, evidence):
    if not valid_selection_rules(rules):
        return "uncertain", "invalid_selection_rules"
    unknown = False
    for rule in rules:
        value = evidence.get(rule["field"])
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (rule["field"] == "region_overlap" and not 0 <= value <= 1)
        ):
            unknown = True
            continue
        threshold = rule["value"]
        satisfied = {
            "eq": value == threshold,
            "gt": value > threshold,
            "gte": value >= threshold,
            "lt": value < threshold,
            "lte": value <= threshold,
        }[rule["op"]]
        if not satisfied:
            return "rejected", "diagnostic_condition_not_met"
    if unknown:
        return "uncertain", "diagnostic_evidence_unavailable"
    return "selected", "supported_fault_and_matching_signature"


def fault_support(case, fault, weight=None):
    """Use exact live weights when supplied; never infer support from confidence."""
    arb = case.get("arbiter") or {}
    target = arb.get("target") or {}
    if (
        arb.get("applied") in REFUTATIONS
        and target.get("capability") == fault.get("capability")
        and target.get("error_type") == fault.get("error_type")
    ):
        return "refuted"
    # Persisted records carry this decision so presentation rounding cannot
    # turn a small positive attribution into a supposed unsupported fault.
    state = fault.get("recommendation_support")
    if weight is None and state in ("supported", "refuted", "no_attribution"):
        return state
    value = fault.get("blame_weight") if weight is None else weight
    return "supported" if type(value) in (int, float) and math.isfinite(value) and value > 0 else "no_attribution"


def entry_payload(entry):
    """Complete deployment/measurement metadata, without an efficacy claim."""
    return {
        "id": entry.id,
        "type": entry.intervention_type,
        "summary": entry.summary,
        "strategies": entry.strategies,
        "artifact": entry.artifact_content,
        "plug_in": entry.artifact_plug_in,
        "applies_when": entry.runtime_trigger,
        "runtime_trigger": entry.runtime_trigger,
        "diagnostic_signature": entry.diagnostic_signature,
        "selection_rules": entry.selection_rules,
        "example_case": entry.example_case,
        "feasibility": entry.feasibility,
        "applicability": entry.applicability,
        "mechanism": entry.mechanism,
        "evidence_signal": entry.evidence_signal,
        "provenance": entry.provenance_kind,
        "status": entry.status,
        "verification": entry.verification,
        "empirical_support": entry.empirical_support,
        "scaffold_layer": entry.scaffold_layer,
        "hook_event": entry.hook_event,
        "expected_effect": entry.expected_effect,
        "contract": entry.contract,
        "calibration": entry.calibration,
        "catalog_role": entry.catalog_role,
        "references": entry.references,
        "artifact_maturity": entry.artifact_maturity,
        "evidence_assessment": entry.evidence_assessment,
        "runtime_contract": entry.runtime_contract,
        "deployment_status": "prerequisites_unverified",
    }


def select_remedies(case, fault, weight=None, entries=None):
    """Return selected recipes and explicit rejected/uncertain decisions.

    A selected recipe is an experiment to consider, not permission to install
    a gate. Scaffold capabilities and free-text prerequisites require checking.
    """
    support = fault_support(case, fault, weight)
    recipes = (
        countermeasures_for(capability=fault.get("capability"), error_type=fault.get("error_type"))
        if entries is None
        else entries
    )
    selected, decisions = [], []
    for entry in recipes:
        if entry.capability != fault.get("capability") or entry.error_type != fault.get("error_type"):
            continue
        if support != "supported":
            decision, reason = "rejected", "fault_" + support
        elif entry.status == "refuted":
            decision, reason = "rejected", "remedy_refuted"
        elif entry.status not in ("untested", "verified"):
            decision, reason = "uncertain", "remedy_not_active"
        elif entry.status == "verified" and (scope_decision := verification_scope(entry, case)):
            decision, reason = scope_decision
        elif (fault.get("audit") or {}).get("verdict") in ("mismatch", "unverifiable"):
            decision, reason = "uncertain", "fault_evidence_requires_review"
        else:
            agents = entry.applicability.get("agents", ["all"])
            if not isinstance(agents, list) or not agents:
                decision, reason = "uncertain", "invalid_agent_applicability"
            elif "all" not in agents and case.get("agent") not in agents:
                decision, reason = "rejected", "agent_outside_applicability"
            else:
                decision, reason = _match(entry.selection_rules, fault.get("detector_evidence") or {})
        decisions.append({"id": entry.id, "decision": decision, "reason": reason})
        if decision == "selected":
            selected.append(entry_payload(entry))
    return {
        "recommendation_support": support,
        "recommendation": [s for entry in selected for s in entry["strategies"]],
        "countermeasures": selected,
        "countermeasure_decisions": decisions,
    }


def refresh_case(case):
    """Refresh only recommendation fields in an existing diagnosis in place."""
    for fault in case.get("faults", []):
        fault.update(select_remedies(case, fault))
    return case
