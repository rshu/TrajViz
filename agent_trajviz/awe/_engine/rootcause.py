"""Blame distribution (the corrected 'root cause').

Instead of forcing a single root cause, distribute blame across the failed
task's candidate errors as weights that sum to 1, and mark a `primary` only when
one factor is clearly dominant (else `conjunctive`). The weighting is
DETERMINISTIC, driven by the triangulation's causal bucket:

  - if the agent never found the right files (loc_miss/loc_partial), the
    localization error is upstream-dominant — editing can't succeed on the
    wrong files, so editing/verification errors are downstream/secondary;
  - if it found AND edited the right files but still failed
    (right_place_wrong_change), localization is exonerated and editing dominates;
  - verification errors are contributing (failed to *catch* the failure), never
    the upstream cause.

Role multipliers are v0 heuristics, to be calibrated in the validation track.

v0.32 causal-logic corrections (see DESIGN.md §0ad):
  - Dominance is decided over CAUSAL capabilities only. Contributing mass
    (verification/self-repair/tool) still carries weight in the ledger — it
    measures failed-to-catch intensity for the scores — but it no longer
    dilutes the attribution test: previously 62 of 96 "conjunctive" cases had
    exactly one causal capability whose primary status was blocked purely by
    the NUMBER of contributing symptom types emitted.
  - Judge capabilities are never primary-eligible. The old rule let a
    HIGH-self-reported-confidence judge verdict seize primary and demote
    deductive facts; measured on the historical Pro cache, high-confidence judge
    flags fire on 0.72% of solved controls vs 0.29% of failures (2.5x
    ANTI-discriminative — scripts/judge_flag_rates.py), so self-reported
    confidence selects no reliable subset. Judge candidates still contribute
    weight; they just cannot own the attribution.
  - An arbiter refute now moves MASS, not just the label: the refuted fault's
    weight is zeroed, the ledger renormalizes, and dominance re-runs over the
    survivors. A deductive, non-interpretive runner-up that now dominates is
    promoted (facts need no adjudication); an interpretive/judge runner-up is
    NOT promoted (it has no arbiter verdict of its own) and the case stays
    conjunctive. Previously a refuted fault kept its full weight in
    capability_blame — the system said "this attribution is wrong" and
    "this fault carries the blame" about the same fault."""

from __future__ import annotations

from collections import defaultdict

# Causal role multipliers. v0 heuristics — NOT objectively calibratable without
# human/interventional ground truth; see agent_trajviz.awe._engine.calibrate.blame_sensitivity for the
# robustness check that justifies treating the bottleneck ranking as stable.
DEFAULT_ROLES = {"primary": 1.0, "verification": 0.5, "secondary": 0.4}
# 'low' is defined for completeness (injected/annotation paths); the judge layer
# gates low-confidence verdicts out before they ever become candidates.
CONF = {"high": 1.0, "medium": 0.6, "low": 0.3}
# Contributing capabilities: failed to catch/recover, but did not cause the wrong
# patch -> weighted like verification (the 'verification' role) and EXCLUDED from
# primary status even when they hold all the normalized weight (a lone
# relevant_test_not_run on a task with no loc/edit candidate must not become the
# headline root cause).
CONTRIBUTING = {"code_verification", "self_repair_loop", "tool_use"}

# Workflow ordering used for heuristic roles. Only Localization/Editing are
# primary-eligible; judge capabilities always keep a secondary role. This order
# does not establish causal effects from observing a requirement/planning label.
UPSTREAM_ORDER = ("requirement_understanding", "task_planning", "code_localization", "code_editing")

# The two judge capabilities are model_inferred (an interpretation, not a
# deduction). Since v0.32 they are NEVER primary-eligible: the old
# high-confidence override was refuted by historical control data (high-conf
# judge flags are ~2.5x MORE frequent on solved controls than on failures —
# self-reported confidence is anti-discriminative; scripts/judge_flag_rates.py).
# Judge candidates still contribute blame weight at the secondary role.
JUDGE_CAPS = ("requirement_understanding", "task_planning")

DOMINANCE = 0.5  # primary must hold >= this share
MARGIN = 0.2  # ...and beat the runner-up by this much


# Evidence strength is shared with evidence chains and score trust composition.
# Associational editing labels must not inherit the capability default here.
def _needs_adjudication(cand: dict) -> bool:
    from .evidence import strength_of, DEDUCTIVE

    return strength_of(cand.get("capability"), cand.get("error_type")) != DEDUCTIVE


def primary_capability(loc_flag: str) -> str | None:
    if loc_flag in ("loc_miss", "loc_partial"):
        return "code_localization"
    if loc_flag in ("right_place_wrong_change", "no_source_edit"):
        return "code_editing"
    return None


def _primary_cap(detection: dict) -> str | None:
    """The most-upstream capability eligible to be the primary cause. Judge
    capabilities (model_inferred) are never eligible (v0.32 — their
    high-confidence tier measures anti-discriminative on solved controls;
    scripts/judge_flag_rates.py); they contribute weight but cannot own the
    attribution. Falls back to the localization-flag mapping when only
    loc/edit are present."""
    present = {c["capability"] for c in detection["candidates"]}
    for cap in UPSTREAM_ORDER:
        if cap in JUDGE_CAPS:
            continue
        if cap in present:
            return cap
    return primary_capability(detection["loc_flag"])


def build_blame(detection: dict, roles: dict | None = None, adjudicate=None) -> dict:
    """``adjudicate`` (optional) is the LLM arbiter hook: a callable
    ``(detection, primary_candidate) -> verdict`` invoked ONLY when the primary
    cause is non-deductive (an omission / judge / contributing label). A confident,
    evidence-cited refute removes its attribution; no survivors means
    ``unattributed``, while competing survivors may remain ``conjunctive`` (we no longer
    claim a single confident root cause); a confirm marks it corroborated. Deductive
    loc/edit primaries are never sent to the arbiter, so facts are never overridden."""
    roles = roles or DEFAULT_ROLES
    res = detection["resolved"]
    cands = detection["candidates"]

    if res is not False:
        # Solved or unknown outcome -> no failure blame; surface quality flags only.
        return {
            "status": "resolved" if res else "unknown",
            "blame": [],
            "capability_blame": {},
            "primary": None,
            "quality_flags": [c["error_type"] for c in cands],
        }

    primary_cap = _primary_cap(detection)
    raws, multipliers = [], []
    for c in cands:
        if c["capability"] == primary_cap:
            role = roles["primary"]
        elif c["capability"] in CONTRIBUTING:
            # All contributing caps carry the 'verification' role (0.5): they
            # failed to catch/recover, they did not cause the wrong patch. (The
            # old code gave self_repair/tool the 0.4 secondary role, contradicting
            # this module's own documentation.)
            role = roles["verification"]
        else:
            role = roles["secondary"]
        raws.append(role * CONF.get(c["confidence"], 0.6))
        multipliers.append(role)

    total = sum(raws)
    if total == 0:
        # Failed but nothing detected -> honestly mark unattributed (coverage gap).
        out = {"status": "unattributed", "blame": [], "capability_blame": {"unattributed": 1.0}, "primary": None}
        out["decision_trace"] = _decision_trace(cands, multipliers, raws, [], out, primary_cap, [])
        return out

    blame = []
    cap_blame: dict[str, float] = defaultdict(float)
    for c, raw in zip(cands, raws, strict=False):
        w = raw / total
        blame.append(
            {"capability": c["capability"], "error_type": c["error_type"], "weight": w, "confidence": c["confidence"]}
        )
        cap_blame[c["capability"]] += w
    blame.sort(key=lambda x: -x["weight"])
    before = [dict(b) for b in blame]

    # Primary status is decided at CAPABILITY level, over CAUSAL capabilities
    # only (v0.32). Two same-capability faults splitting the mass 0.5/0.5 are
    # one dominant capability, not a conjunction. Contributing caps are never
    # primary AND their mass no longer dilutes the test: whether one causal
    # capability dominates cannot depend on how many failed-to-catch SYMPTOM
    # types happened to fire — previously each medium verification symptom
    # added rival mass, manufacturing "no dominant cause" in 62 of 96
    # conjunctive cases that had exactly one causal capability. Contributing
    # weights remain in blame/capability_blame (they measure catch/recovery
    # intensity for the scores); "conjunctive" now means genuinely competing
    # CAUSES.
    def _dominant_causal(exclude: tuple = ()) -> str | None:
        causal = {
            cap: m
            for cap, m in cap_blame.items()
            if cap not in CONTRIBUTING and cap not in JUDGE_CAPS and cap not in exclude and m > 0
        }
        total_causal = sum(causal.values())
        if not causal or total_causal <= 0:
            return None
        ranked = sorted(causal.items(), key=lambda kv: -kv[1])
        top_share = ranked[0][1] / total_causal
        runner_share = (ranked[1][1] / total_causal) if len(ranked) > 1 else 0.0
        if top_share >= DOMINANCE - 1e-9 and top_share - runner_share >= MARGIN - 1e-9:
            return ranked[0][0]
        return None

    top_cap = _dominant_causal()
    if top_cap is not None:
        status = "primary"
        primary = next(b for b in blame if b["capability"] == top_cap)
    else:
        status, primary = "conjunctive", None

    # LLM arbiter: corroborate or refute a NON-deductive primary (never a fact).
    result_arbiter = None
    arbiter_steps = []
    if status == "primary" and adjudicate is not None and _needs_adjudication(primary):
        v = adjudicate(detection, primary) or {}
        arbiter_steps = list(v.get("evidence_steps") or [])
        support, conf = v.get("support"), v.get("confidence")
        cited = bool(v.get("evidence_steps"))
        target = {"capability": primary["capability"], "error_type": primary["error_type"]}
        if support == "refute" and conf in ("high", "medium") and cited:
            # v0.32: a refute moves MASS, not just the label. The refuted
            # fault's weight is zeroed (kept in the ledger, flagged), the
            # survivors renormalize, and dominance re-runs. A deductive,
            # non-interpretive runner-up that now dominates is promoted —
            # facts need no adjudication. An interpretive/judge runner-up is
            # NOT promoted (it has no arbiter verdict of its own).
            refuted = primary
            refuted["weight"] = 0.0
            refuted["arbiter_refuted"] = True
            cap_blame[refuted["capability"]] = sum(
                b["weight"] for b in blame if b["capability"] == refuted["capability"]
            )
            remaining = sum(b["weight"] for b in blame)
            if remaining > 0:
                for b in blame:
                    b["weight"] = b["weight"] / remaining
                cap_blame = defaultdict(float)
                for b in blame:
                    cap_blame[b["capability"]] += b["weight"]
            blame.sort(key=lambda x: -x["weight"])

            new_cap = _dominant_causal()
            new_primary = None
            if new_cap is not None:
                cand = next(b for b in blame if b["capability"] == new_cap and b["weight"] > 0)
                if not _needs_adjudication(cand):
                    new_primary = cand
            if remaining == 0:
                cap_blame = defaultdict(float, {"unattributed": 1.0})
                applied = "unattributed_after_refute"
                status, primary = "unattributed", None
            elif new_primary is not None:
                applied = "refuted_reassigned"
                status, primary = "primary", new_primary
                primary["promoted_after_refute"] = True
            else:
                applied = "demoted_to_conjunctive"
                status, primary = "conjunctive", None
        else:
            corroborated = support == "confirm" and conf in ("high", "medium")
            applied = "corroborated" if corroborated else "noted"
            primary["arbiter"] = {
                "support": support,
                "confidence": conf,
                "corroborated": corroborated,
                "rationale": v.get("rationale", ""),
            }
        result_arbiter = {
            "target": target,
            "support": support,
            "confidence": conf,
            "cited": cited,
            "rationale": v.get("rationale", ""),
            "applied": applied,
            "cache_provenance": v.get("cache_provenance"),
        }

    out = {"status": status, "blame": blame, "capability_blame": dict(cap_blame), "primary": primary}
    if result_arbiter is not None:
        out["arbiter"] = result_arbiter
    out["decision_trace"] = _decision_trace(cands, multipliers, raws, before, out, primary_cap, arbiter_steps)
    return out


def _decision_trace(candidates, multipliers, raws, before, result, role_capability, arbiter_steps):
    """Record decisions without re-running detection or adjudication.

    Preserve detector order for exact ties; rounded ties never substitute for
    the full-precision weights actually used by the selection policy.
    """

    def identity(c):
        return c["capability"], c["error_type"]

    initial = {identity(b): b["weight"] for b in before}
    final = {identity(b): b["weight"] for b in result["blame"]}
    eligible = {
        cap: weight
        for cap, weight in result["capability_blame"].items()
        if cap not in CONTRIBUTING and cap not in JUDGE_CAPS and cap != "unattributed" and weight > 0
    }
    total = sum(eligible.values())
    ranked = sorted(eligible, key=lambda cap: -eligible[cap])
    shares = {cap: weight / total for cap, weight in eligible.items()} if total else {}
    lead = shares[ranked[0]] if ranked else 0
    runner = shares[ranked[1]] if len(ranked) > 1 else 0
    primary = result["primary"]
    ties, displayed_ties = [], []
    if primary:
        for c in candidates:
            if c["capability"] != primary["capability"]:
                continue
            weight = final.get(identity(c), 0)
            if weight == primary["weight"]:
                ties.append(c["error_type"])
            if round(weight, 2) == round(primary["weight"], 2):
                displayed_ties.append(c["error_type"])
    if primary:
        reason = (
            "stable_detector_order_for_exact_weight_tie"
            if len(ties) > 1
            else "highest_weight_in_dominant_eligible_capability"
        )
    else:
        reason = (
            "no_positive_candidate_weight"
            if result["status"] == "unattributed"
            else "no_eligible_dominance_or_unreviewed_replacement"
        )
    return {
        "schema_version": 1,
        "basis": "recorded_during_attribution",
        "role_capability": role_capability,
        "candidates": [
            {
                "detector_order": i,
                "capability": c["capability"],
                "error_type": c["error_type"],
                "confidence": c["confidence"],
                "role_multiplier": mult,
                "confidence_multiplier": CONF.get(c["confidence"], 0.6),
                "raw_score": raw,
                "weight_before_arbiter": initial.get(identity(c), 0),
                "final_weight": final.get(identity(c), 0),
                "primary_eligible": c["capability"] not in CONTRIBUTING and c["capability"] not in JUDGE_CAPS,
            }
            for i, (c, mult, raw) in enumerate(zip(candidates, multipliers, raws, strict=False))
        ],
        "dominance": {
            "eligible_capability_mass": eligible,
            "eligible_shares": shares,
            "leading_share": lead,
            "runner_up_share": runner,
            "required_share": DOMINANCE,
            "required_margin": MARGIN,
            "actual_margin": lead - runner,
            "passes": bool(ranked and lead >= DOMINANCE - 1e-9 and lead - runner >= MARGIN - 1e-9),
        },
        "adjudication": {**(result.get("arbiter") or {}), "evidence_steps": arbiter_steps}
        if result.get("arbiter")
        else None,
        "selection": {
            "status": result["status"],
            "reason": reason,
            "primary": {"capability": primary["capability"], "error_type": primary["error_type"]} if primary else None,
            "exact_weight_ties": ties,
            "displayed_weight_ties": displayed_ties,
        },
        "weights_are_causal_probabilities": False,
    }
