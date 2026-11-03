"""Grounded countermeasure catalog: root cause -> targeted, mechanism-grounded remedy.

Source of truth is configs/countermeasures.json (stdlib json; the core package is
yaml-free). Supersedes the old recommendations.py map. Each entry pins to the SAME
diagnosis signal that grounds its label (grounds.evidence_signal) and carries a
required mechanism. Tier B (closed-loop verification) is the only path that may set
status='verified'; see docs/superpowers/specs/2026-06-20-countermeasure-catalog-design.md."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config

CATALOG_PATH = config.PROJECT_ROOT / "configs" / "countermeasures.json"

VALID_TYPES = {"prompt", "skill", "guardrail", "plugin"}
VALID_STATUS = {"proposed", "untested", "verified", "refuted"}
VALID_SOURCE_TYPES = {"paper", "preprint", "docs", "web"}

# Where in the agent scaffold the remedy plugs in — the developer's mental model
# ("which part of my agent do I edit?"). ORTHOGONAL to intervention type: that names
# the artifact FORM (prompt/guardrail/skill/plugin), this names the LOCATION in the
# agent loop. Lets the catalog be read "by where you change it", and lets a future
# interventional (patcher) loop target one layer at a time.
VALID_SCAFFOLD_LAYERS = {"system_prompt", "tool_definition", "control_flow", "retry_policy", "response_synthesis"}

# WHEN the artifact fires (v2) — the concrete scaffold event, finer than the layer.
# Bundles group entries sharing an event so one artifact implements many entries.
VALID_HOOK_EVENTS = {
    "task_start",
    "session",
    "first_edit",
    "per_edit",
    "per_tool_call",
    "tool_error",
    "post_test_run",
    "pre_submit",
}

# What a countermeasure is expected to DO. Attribution role and intervention effect
# are distinct: a contributing fault's remedy may improve resolution by enabling
# repair or candidate selection. Measure the declared endpoint (METH-VERIF-1).
VALID_EFFECTS = {"fix_rate", "failure_surfacing", "budget_efficiency"}


def provenance_kind_for(references: list) -> str:
    """The honesty ladder, computed from references (single source of truth)."""
    if any((r or {}).get("source_type") in ("paper", "preprint") for r in references):
        return "literature"
    if references:
        return "documented"
    return "heuristic"


def known_evidence_signals() -> set:
    """The valid grounding signals, derived from the diagnosis layer (not a local
    constant) so the catalog can never pin to a signal the system doesn't produce."""
    from .detect import grounding_signals

    return grounding_signals()


@dataclass
class Entry:
    id: str
    capability: str
    error_type: str
    intervention_type: str
    summary: str
    strategies: list[str]
    mechanism: str
    evidence_signal: str
    provenance_kind: str
    cites: list[str] = field(default_factory=list)
    applicability: dict = field(default_factory=dict)
    status: str = "untested"
    verification: dict | None = None
    artifact_type: str = ""
    artifact_content: str = ""
    artifact_plug_in: str = ""
    # v2: runtime_trigger is the DEPLOYABLE condition (oracle-free — issue text,
    # own trajectory, working tree only); diagnostic_signature is the post-hoc
    # detector condition (gold data allowed). Conflating them is how oracle
    # leakage enters an intervention (METH-ORACLE-1).
    runtime_trigger: str = ""
    diagnostic_signature: str = ""
    example_case: dict | None = None
    feasibility: str = ""
    empirical_support: int = 0
    references: list = field(default_factory=list)
    scaffold_layer: str = ""
    hook_event: str = ""
    expected_effect: list[str] = field(default_factory=list)
    contract: dict | None = None  # guardrails: {max_triggers, escape_hatch}
    calibration: dict | None = None  # in-repo cohort numbers / negative-result caveats
    selection_rules: list = field(default_factory=list)  # diagnostic-side predicates
    artifact_maturity: str = "unspecified"  # implementation readiness, independent of efficacy
    evidence_assessment: dict = field(default_factory=dict)
    runtime_contract: dict | None = None

    @property
    def artifact_applies_when(self) -> str:
        """Back-compat alias for pre-v2 surfaces (dossier/report/showcase): the
        deployable trigger is what those surfaces should show."""
        return self.runtime_trigger

    @property
    def catalog_role(self) -> str:
        """Derived, not stored: 'completeness' for judge-grounded entries (the
        judge layer measures near-noise — METH-JUDGE-1), 'core' for entries on
        objective signals. Filter to 'core' when selecting pilots."""
        return "completeness" if self.evidence_signal == "llm_judge" else "core"


def _to_entry(d: dict) -> Entry:
    rc = d.get("root_cause", {})
    iv = d.get("intervention", {})
    af = d.get("artifact") or {}
    return Entry(
        id=d.get("id", ""),
        capability=rc.get("capability", ""),
        error_type=rc.get("error_type", ""),
        intervention_type=iv.get("type", ""),
        summary=iv.get("summary", ""),
        strategies=list(iv.get("strategies", [])),
        mechanism=d.get("mechanism", ""),
        evidence_signal=(d.get("grounds") or {}).get("evidence_signal", ""),
        provenance_kind=(d.get("provenance") or {}).get("kind", ""),
        cites=list((d.get("provenance") or {}).get("cites", [])),
        applicability=dict(d.get("applicability") or {}),
        status=d.get("status", "untested"),
        verification=d.get("verification"),
        artifact_type=af.get("type", ""),
        artifact_content=af.get("content", ""),
        artifact_plug_in=af.get("plug_in", ""),
        # v1 catalogs carried a single applies_when; read it as the trigger so
        # external/legacy catalogs still load (the shipped catalog is v2).
        runtime_trigger=af.get("runtime_trigger", af.get("applies_when", "")),
        diagnostic_signature=af.get("diagnostic_signature", ""),
        example_case=d.get("example_case"),
        feasibility=d.get("feasibility", ""),
        empirical_support=d.get("empirical_support", 0),
        references=list(d.get("references", [])),
        scaffold_layer=d.get("scaffold_layer", ""),
        hook_event=d.get("hook_event", ""),
        expected_effect=list(d.get("expected_effect", [])),
        contract=d.get("contract"),
        calibration=d.get("calibration"),
        selection_rules=d.get("selection_rules", []),
        artifact_maturity=af.get("maturity", "unspecified"),
        evidence_assessment=d.get("evidence_assessment", {}),
        runtime_contract=d.get("runtime_contract"),
    )


_CACHE: list[Entry] | None = None
_DOC_CACHE: dict | None = None


def _load_raw(path: Path | None = None) -> dict:
    global _DOC_CACHE
    if path is None and _DOC_CACHE is not None:
        return _DOC_CACHE
    p = path or CATALOG_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RuntimeError(f"countermeasure catalog not found at {p}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"countermeasure catalog at {p} is not valid JSON: {e}") from e
    if path is None:
        _DOC_CACHE = raw
    return raw


def load_catalog(path: Path | None = None) -> list[Entry]:
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    entries = [_to_entry(e) for e in _load_raw(path).get("entries", [])]
    if path is None:
        _CACHE = entries
    return entries


def bundles(path: Path | None = None) -> list[dict]:
    """Hook-event groupings for integration planning, not installed implementations.
    Pilot one recipe first; bundle membership does not validate composition."""
    return list(_load_raw(path).get("bundles", []))


def methodology(path: Path | None = None) -> list[dict]:
    """v2 catalog-level methodology notes (oracle-free triggers, observational-RR
    limits, judge-role caveat, verification endpoint, composition costs, scout
    ingestion) — each with the same reference schema as entries."""
    return list(_load_raw(path).get("methodology", []))


def operational_annex(path: Path | None = None) -> list[dict]:
    """v2 annex: remedies for operational failure modes grounded on trajectory
    SIGNALS (e.g. truncated_run) rather than blamed fault types — exempt from the
    one-entry-per-error-type rule, validated against awe/trajsignals fields."""
    return list(_load_raw(path).get("operational_annex", []))


def benchmark_artifact_annex(path: Path | None = None) -> list[dict]:
    """v0.29 annex: harness-rigidity SUSPECTS — failed cases whose near-gold patch
    passed the agent's own verification. These are candidate evaluation artifacts
    (spec-oracle gap / strict format mismatch — METH-HARNESS-1), queued for manual
    review; they are never auto-excluded from fault counts. Grounded on real
    trajectory signals plus a patch-vs-gold content condition."""
    return list(_load_raw(path).get("benchmark_artifact_annex", []))


def countermeasures_for(
    capability: str | None = None, error_type: str | None = None, status: str | None = None
) -> list[Entry]:
    out = load_catalog()
    if capability is not None:
        out = [e for e in out if e.capability == capability]
    if error_type is not None:
        out = [e for e in out if e.error_type == error_type]
    if status is not None:
        out = [e for e in out if e.status == status]
    return out


def recommend(error_type: str) -> dict:
    """Back-compat shim for the old recommendations.recommend(): returns the target
    capability and the flattened strategy strings across all entries for this type."""
    entries = countermeasures_for(error_type=error_type)
    if not entries:
        return {"target": "unknown", "strategies": ["No mapped countermeasure; review manually."]}
    strategies: list[str] = []
    for e in entries:
        strategies.extend(e.strategies)
    return {"target": entries[0].capability, "strategies": strategies}


# --- Scaffold-layer view: the same remedies, indexed by WHERE a developer edits ---

SCAFFOLD_LAYER_LABEL = {
    "system_prompt": "System prompt",
    "tool_definition": "Tool definition",
    "control_flow": "Control flow",
    "retry_policy": "Retry policy",
    "response_synthesis": "Response synthesis",
}
# Display order ~ the agent loop: instruct -> equip -> orchestrate -> recover -> answer.
SCAFFOLD_LAYER_ORDER = ("system_prompt", "tool_definition", "control_flow", "retry_policy", "response_synthesis")


def by_scaffold_layer() -> dict:
    """Group catalog entries by the scaffold layer a developer edits — the
    'where do I change my agent?' view. Returns {layer: [Entry]} in loop order
    (any unassigned entries land under 'unassigned')."""
    buckets: dict = {layer: [] for layer in SCAFFOLD_LAYER_ORDER}
    for e in load_catalog():
        buckets.setdefault(e.scaffold_layer or "unassigned", []).append(e)
    return {layer: buckets[layer] for layer in buckets if buckets[layer]}


def scaffold_layer_view_md() -> str:
    """Render the catalog grouped by scaffold layer — the developer-facing 'by where
    you change it' index. Each remedy keeps its grounding (it only re-orders by
    intervention LOCATION; orthogonal to the artifact form)."""
    groups = by_scaffold_layer()
    statuses = sorted({e.status for e in load_catalog()})
    L = [
        "# Countermeasures by scaffold layer",
        "_The same grounded remedies, organized by **where in the agent you plug them "
        "in** rather than by which error they target — the developer's mental model. "
        "Orthogonal to the artifact form (prompt/guardrail/skill/plugin). Entry "
        f"statuses present: {', '.join(f'`{s}`' for s in statuses)} (`untested` = "
        "proposed, not efficacy-proven; only the Tier-B interventional loop may set "
        "`verified`)._",
        "",
    ]
    for layer, entries in groups.items():
        L += [
            f"## {SCAFFOLD_LAYER_LABEL.get(layer, layer)}  ({len(entries)})",
            "",
            "| Remedy | Targets (error type) | Form | Fires at | Runtime trigger (oracle-free) |",
            "|---|---|---|---|---|",
        ]
        for e in entries:
            L.append(
                f"| {e.summary} | `{e.error_type}` | {e.intervention_type} "
                f"| `{e.hook_event}` | {e.runtime_trigger[:80]} |"
            )
        L.append("")
    bs = bundles()
    if bs:
        L += [
            "## Bundles (integration planning only)",
            "",
            "_Pilot one recipe first. Membership does not establish safe composition. "
            "Use a shared run-level intervention budget (METH-COMPOSE-1)._",
            "",
            "| Bundle | Fires at | Members | Summary |",
            "|---|---|---|---|",
        ]
        for b in bs:
            evs = ", ".join(f"`{h}`" for h in b.get("hook_events", []))
            L.append(f"| {b['id']} | {evs} | {len(b.get('members', []))} | {b.get('summary', '')} |")
        L.append("")
    return "\n".join(L)
