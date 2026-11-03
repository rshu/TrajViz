"""Cohort-contrast discovery — behaviors associated with failure vs success.

Borrowed from the Insights Generator paper (Manglik et al., 2026): instead of
attributing a single failure, contrast the solved vs failed cohorts and surface
process BEHAVIORS whose prevalence differs between them (e.g. "failed runs
hallucinate paths 3x more often"). This complements the deductive per-failure
diagnosis — it finds patterns the fixed taxonomy does not encode.

Honesty: these are ASSOCIATIONAL, not causal. The cohorts are objectively split by
the gold-anchored outcome (a strength over IG, whose labels are LLM-judged), the
effect is reported with a bootstrap CI + Holm-corrected p (a real significance gate,
not IG's qualitative buckets), but a difference in prevalence is not a proof of
cause — that needs the deferred interventional loop. This module is PURE over
per-instance feature vectors; the statistics import nothing from detect/rootcause.
"""

from __future__ import annotations

import random

from . import behaviors
from .significance import holm

# Binary process behaviors (each instance contributes 0.0/1.0). The rest are
# continuous magnitudes (counts / rates). Names are display-facing.
BINARY_FEATURES = {
    "ran_no_test",
    "encountered_tool_error",
    "tool_failure_not_retried",
    "hallucinated_path",
    "command_not_found",
    "repair_not_reverified",
    "repeated_ineffective_attempt",
    "claimed_success_in_final",
    "encountered_failure_event",
    "analysis_paralysis",
    "environment_error",
    "exploration_loop",
    "truncated_run",
    "retry_without_change",
    "hallucinated_path_unrecovered",
    "edited_before_any_run",
    "edit_reverted_own_change",
    "step_repetition",
    # v0.36 batch-8 candidates (calibrate-first; behavioral until the gate says
    # otherwise): delegation accounting, weak-oracle repro, suite-wide running.
    "delegated",
    "delegate_error_result",
    "delegate_no_result",
    "authored_test_ran_never_failed",
    "ran_suite_wide",
}


def features_of(sig) -> dict:
    """Map a TrajectorySignals to the contrast feature vector. Pure; takes the
    signals object so it is testable without ARGUS data."""
    subs = sig.tool_error_subtypes or {}
    rate = (sig.n_tool_call_errors / sig.n_tool_calls) if sig.n_tool_calls else 0.0
    return {
        # binary process behaviors
        "ran_no_test": float(not sig.ran_any_test),
        "encountered_tool_error": float(sig.encountered_tool_error),
        "tool_failure_not_retried": float(sig.tool_failure_not_retried),
        "hallucinated_path": float(subs.get("invalid_target", 0) > 0),
        "command_not_found": float(subs.get("command_not_found", 0) > 0),
        "repair_not_reverified": float(sig.repair_not_reverified),
        "repeated_ineffective_attempt": float(sig.repeated_ineffective_attempt),
        "claimed_success_in_final": float(sig.claimed_success_final),
        "encountered_failure_event": float(sig.encountered_failure),
        "exploration_loop": float(getattr(sig, "exploration_loop", False)),
        "step_repetition": float(getattr(sig, "step_repetition", False)),
        "truncated_run": float(getattr(sig, "truncated_run", False)),
        "retry_without_change": float(getattr(sig, "retry_without_change", False)),
        "hallucinated_path_unrecovered": float(getattr(sig, "hallucinated_path_unrecovered", False)),
        # calibration (v0.26): does NOT discriminate (43.7% solved vs 36.5%
        # failed, RR 0.83) — kept as a feature so the gate keeps saying so
        "edited_before_any_run": float(getattr(sig, "edited_before_any_run", False)),
        # calibration (v0.27): discriminates (5.5% failed vs 2.6% solved, RR 2.10)
        "edit_reverted_own_change": float(getattr(sig, "edit_reverted_own_change", False)),
        # scout-discovered mode (sig-only; analysis_paralysis is added in feature_rows
        # because it needs the authoritative patch, not the trajectory edit-set)
        "environment_error": float(behaviors.environment_error(sig)),
        # v0.36 batch-8 candidates
        "delegated": float(getattr(sig, "n_delegations", 0) > 0),
        # v20 plan-tool features (DESIGN §0ay) — behavioral until calibrated
        "plan_declared": float(getattr(sig, "plan_declared", False)),
        "plan_before_first_edit": float(getattr(sig, "plan_before_first_edit", False)),
        "plan_left_open": float(getattr(sig, "plan_items_open_at_end", 0) > 0),
        "n_plan_updates": float(getattr(sig, "n_plan_updates", 0)),
        "delegate_error_result": float(getattr(sig, "delegate_error_result", False)),
        "delegate_no_result": float(getattr(sig, "delegate_no_result", False)),
        "authored_test_ran_never_failed": float(getattr(sig, "authored_test_ran_never_failed", False)),
        "ran_suite_wide": float(getattr(sig, "ran_suite_wide", False)),
        # continuous process magnitudes
        "n_tool_calls": float(sig.n_tool_calls),
        "n_tool_call_errors": float(sig.n_tool_call_errors),
        "tool_error_rate": float(rate),
        "n_steps": float(sig.n_steps),
        "n_unproductive_repeats": float(getattr(sig, "n_unproductive_repeats", 0)),
        "n_test_runs": float(sig.n_test_runs),
        "n_failure_events": float(sig.n_failure_events),
        "read_set_size": float(len(sig.read_set)),
        "edit_set_size": float(len(sig.edit_set_traj)),
    }


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


MIN_COHORT = 5  # below this a bootstrap "significance" is an artifact, not evidence


def contrast(failed: list, solved: list, rng: random.Random, B: int = 2000) -> dict:
    """Unpaired bootstrap of the mean difference (failed − solved) for one feature.
    Two-sided percentile p = 2·min(share≤0, share≥0) with add-one smoothing
    ((k+1)/(B+1)) so p is never exactly 0. Either cohort empty → p=1.0; either
    cohort smaller than MIN_COHORT → p=1.0 (too few runs to claim an association)."""
    nf, ns = len(failed), len(solved)
    if nf == 0 or ns == 0:
        return {
            "mean_failed": _mean(failed),
            "mean_solved": _mean(solved),
            "diff": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "p": 1.0,
            "n_failed": nf,
            "n_solved": ns,
        }
    mf, ms = _mean(failed), _mean(solved)
    diffs = []
    for _ in range(B):
        bf = sum(failed[rng.randrange(nf)] for _ in range(nf)) / nf
        bs = sum(solved[rng.randrange(ns)] for _ in range(ns)) / ns
        diffs.append(bf - bs)
    diffs.sort()
    lo = diffs[min(B - 1, max(0, round(0.025 * (B - 1))))]
    hi = diffs[min(B - 1, max(0, round(0.975 * (B - 1))))]
    le = (sum(1 for d in diffs if d <= 0) + 1) / (B + 1)
    ge = (sum(1 for d in diffs if d >= 0) + 1) / (B + 1)
    p = min(1.0, 2 * min(le, ge))
    if nf < MIN_COHORT or ns < MIN_COHORT:
        p = 1.0
    return {
        "mean_failed": mf,
        "mean_solved": ms,
        "diff": mf - ms,
        "lo": lo,
        "hi": hi,
        "p": p,
        "n_failed": nf,
        "n_solved": ns,
    }


def binary_association(failed: list, solved: list) -> dict:
    """Risk ratio + odds ratio of FAILURE given a binary behavior is present.
    failed/solved are 0/1 lists (the behavior indicator within each cohort).
    a=present&failed, b=present&solved, c=absent&failed, d=absent&solved.

    A probability of 0.0 is DATA, not missing data: a perfectly protective
    behavior gets rr=0.0, a perfectly predictive one rr=float('inf') — only an
    undefined denominator (empty stratum) yields None. The old truthiness check
    (`if p_present and p_absent`) rendered exactly the most extreme associations
    as None."""
    a, b = sum(failed), sum(solved)
    c, d = len(failed) - a, len(solved) - b
    p_present = a / (a + b) if (a + b) else None  # P(fail | behavior present)
    p_absent = c / (c + d) if (c + d) else None  # P(fail | behavior absent)
    if p_present is None or p_absent is None:
        rr = None
    elif p_absent == 0.0:
        rr = float("inf") if p_present > 0 else None  # 0/0: undefined
    else:
        rr = p_present / p_absent
    odds = (a * d) / (b * c) if (b and c) else None
    return {
        "risk_ratio": rr,
        "odds_ratio": odds,
        "p_fail_if_present": p_present,
        "p_fail_if_absent": p_absent,
        "n_present": int(a + b),
    }


def discover(rows: list, seed: int = 0, B: int = 2000) -> dict:
    """Contrast every feature across the solved vs failed cohorts, Holm-correct the
    family of p-values, and rank. `rows` is a list of (resolved_bool, feature_dict).
    Pure — accepts the rows so it is testable without ARGUS data."""
    failed = [f for r, f in rows if r is False]
    solved = [f for r, f in rows if r is True]
    names = sorted(failed[0]) if failed else (sorted(solved[0]) if solved else [])
    rng = random.Random(seed)
    results, pvals = [], []
    for name in names:
        c = contrast([f[name] for f in failed], [f[name] for f in solved], rng, B)
        c["feature"] = name
        c["binary"] = name in BINARY_FEATURES
        if c["binary"]:
            c.update(binary_association([f[name] for f in failed], [f[name] for f in solved]))
        results.append(c)
        pvals.append(c["p"])
    for c, ap in zip(results, holm(pvals), strict=False):
        c["p_holm"] = ap
    results.sort(key=lambda c: (c["p_holm"], -abs(c["diff"])))
    return {"n_failed": len(failed), "n_solved": len(solved), "associations": results}


def feature_rows(agent: str, instances) -> list:
    """Build (resolved, features) rows for an agent over ALL its trajectories
    (solved + failed). Reads trajectory signals (cached); needs ARGUS data."""
    from .gold import build_gold_reference
    from .outcomes import resolved as _resolved
    from .trajsignals import signals_cached

    out = []
    for inst in instances:
        r = _resolved(agent, inst)
        if r is None:
            continue
        try:
            gold = build_gold_reference(inst)
        except Exception:
            continue
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        feats = features_of(sig)
        feats["analysis_paralysis"] = float(
            behaviors.analysis_paralysis(sig, behaviors._produced_source_patch(agent, inst))
        )
        out.append((r, feats))
    return out


def discover_agent(agent: str, instances, seed: int = 0, B: int = 2000) -> dict:
    d = discover(feature_rows(agent, instances), seed=seed, B=B)
    d["agent"] = agent
    return d


def cohort_report_md(disc: dict, alpha: float = 0.05) -> str:
    """Render the significant failure-associated behaviors as markdown. Only rows
    that survive the Holm-corrected gate are shown; direction is failure vs success."""
    L = [
        f"## Cohort-contrast behaviors — `{disc.get('agent', '?')}`",
        f"_Behaviors whose prevalence differs between this agent's "
        f"{disc['n_failed']} failed and {disc['n_solved']} solved traces. "
        f"**Associational, not causal** — significance-gated (Holm), but a "
        f"difference is not proof of cause (see the interventional loop). Cohorts "
        f"are split by the objective gold outcome._",
        "",
    ]
    sig = [a for a in disc["associations"] if a["p_holm"] < alpha and a["n_failed"] and a["n_solved"]]
    if not sig:
        L += ["_No behavior is significantly associated with failure after correction._", ""]
        return "\n".join(L)
    L += ["| Behavior | More common in | failed | solved | effect | p (Holm) |", "|---|---|---|---|---|---|"]
    for a in sig:
        direction = "**failures**" if a["diff"] > 0 else "successes"
        if a["binary"]:
            rr = a.get("risk_ratio")
            # rr=0.0 (perfectly protective) and rr=inf (perfectly predictive) are
            # the STRONGEST effects — render them, don't fall through to "—".
            if rr is None:
                eff = f"OR={a['odds_ratio']:.2f}" if a.get("odds_ratio") else "—"
            elif rr == float("inf"):
                eff = "RR=∞"
            else:
                eff = f"RR={rr:.2f}"
            fv, sv = f"{100 * a['mean_failed']:.0f}%", f"{100 * a['mean_solved']:.0f}%"
        else:
            eff = f"Δmean={a['diff']:+.2f}"
            fv, sv = f"{a['mean_failed']:.2f}", f"{a['mean_solved']:.2f}"
        L.append(f"| `{a['feature']}` | {direction} | {fv} | {sv} | {eff} | {a['p_holm']:.3f} |")
    L.append("")
    return "\n".join(L)
