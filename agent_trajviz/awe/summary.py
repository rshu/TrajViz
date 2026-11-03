"""Small, evidence-backed reading guide shared by native and saved attribution views."""
from __future__ import annotations

import html
import math

_CAP_LABELS = {
    "requirement_understanding": "Requirement Understanding",
    "task_planning": "Task Planning",
    "code_localization": "Code Localization",
    "code_editing": "Code Editing",
    "code_verification": "Code Verification",
    "self_repair_loop": "Self-Repair Loop",
    "tool_use": "Tool Use",
}


def _positive_weight(fault: dict) -> bool:
    if isinstance(fault.get("blame_weight"), bool):
        return False
    try:
        weight = float(fault.get("blame_weight") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(weight) and weight > 0


def _records(value: object) -> list[dict]:
    """Optional JSON arrays are evidence only when their container has that shape."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _fault_identity(fault: dict) -> tuple[str, str] | None:
    capability, error = fault.get("capability"), fault.get("error_type")
    if isinstance(capability, str) and capability.strip() and isinstance(error, str) and error.strip():
        return capability, error
    return None


def primary_fault(data: dict) -> dict | None:
    """Return a supported primary record, retaining explicit abstention and outcomes."""
    if data.get("resolved") is True:
        return None
    faults = [fault for fault in _records(data.get("faults"))
              if _positive_weight(fault) and _fault_identity(fault) is not None]
    if "primary" in data:
        primary = data.get("primary")
        identity = _fault_identity(primary) if isinstance(primary, dict) else None
        if identity is None:
            return None
        return next((fault for fault in faults if _fault_identity(fault) == identity), None)
    # Saved dossiers identify their primary fault on the fault itself.
    marked = [fault for fault in faults if fault.get("is_primary") is True]
    return marked[0] if len(marked) == 1 else None


def attribution_summary_html(data: dict) -> str:
    """Explain existing records without supplying a new verdict or unrecorded remedy."""
    fault = primary_fault(data)
    chain = (fault or {}).get("evidence_chain")
    chain = chain if isinstance(chain, dict) else {}
    if data.get("resolved") is True:
        happened = "The recorded outcome is solved."
    elif fault:
        capability = _CAP_LABELS.get(fault.get("capability"), str(fault.get("capability") or "Unknown capability"))
        error = str(fault.get("error_type") or "unspecified finding").replace("_", " ")
        happened = f"{capability}: {error}."
    elif data.get("resolved") is False or data.get("blame_status"):
        happened = "No single dominant fault is supported by the recorded findings."
    elif "analysis" in data:
        happened = "This record contains symptom observations without a fault attribution."
    else:
        happened = "The outcome is unknown; no dominant fault attribution is recorded."
    observations = []
    for link in _records(chain.get("links")):
        if link.get("kind") != "observation":
            continue
        statement = link.get("statement")
        if isinstance(statement, str) and statement.strip():
            source = link.get("source")
            excerpt = html.escape(statement.strip())
            if isinstance(source, str) and source.strip():
                excerpt += f" <span style='color:var(--ov-muted,#64748b)'>({html.escape(source.strip())})</span>"
            observations.append(excerpt)
    evidence = ("<ul style='margin:4px 0 0;padding-left:20px;list-style:disc;'>"
                + "".join(f"<li style='margin:3px 0;'>{observation}</li>" for observation in observations)
                + "</ul>" if observations else "Expand the recorded evidence below to inspect the available support.")
    counterfactual = chain.get("counterfactual")
    check = counterfactual.get("what_would_confirm") if isinstance(counterfactual, dict) else None
    if isinstance(check, str) and check.strip():
        raw_status = counterfactual.get("status")
        status = (raw_status.strip().replace("_", " ")
                  if isinstance(raw_status, str) and raw_status.strip() else "status not recorded")
        next_check = f"Recorded check ({html.escape(status)}): {html.escape(check.strip())}"
    elif data.get("resolved") is True:
        next_check = "Review the recorded test evidence before comparing this run with other attempts."
    else:
        next_check = "Review the evidence and test details; no confirming check was recorded."
    scorecard = [score for score in _records(data.get("scorecard"))
                 if isinstance(score.get("capability"), str) and score["capability"].strip()
                 and isinstance(score.get("assessed"), bool)]
    assessed = sum(score["assessed"] is True for score in scorecard)
    coverage = (f"{assessed} of {len(scorecard)} capabilities assessed; {len(scorecard) - assessed} unassessed. "
                if scorecard else "Capability assessment coverage was not recorded. ")
    limits = (coverage + "Not assessed does not mean passed. Attribution weights are not probabilities. "
              "Proposed checks require validation before treating a remedy as effective.")
    rows = (("What happened", html.escape(happened)), ("Evidence", evidence),
            ("Next check", next_check), ("Limits", html.escape(limits)))
    return ("<section class='awe-attribution-summary' aria-label='At a glance' "
            "style='border:1px solid var(--border-color-primary,#dbe3ed);border-radius:10px;padding:14px;margin:12px 0;'>"
            "<div style='font-size:15px;font-weight:600;'>At a glance</div>"
            + "".join(f"<div class='awe-summary-row' style='margin-top:8px;overflow-wrap:anywhere;'>"
                      f"<strong>{label}</strong><div>{body}</div></div>" for label, body in rows)
            + "</section>")
