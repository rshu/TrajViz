"""Calibration + robustness analysis.

over_editing threshold: genuinely data-calibratable from the solved-vs-failed
edit_size_ratio distribution. Blame role-multipliers: NOT objectively calibratable
without human/interventional ground truth, so we report a SENSITIVITY analysis
instead (does the bottleneck ranking survive varying the weights?)."""

from __future__ import annotations

from . import config
from .gold import build_gold_reference
from .patches import load_agent_patch
from .outcomes import resolved
from .signals import localization_signals
from .scoring import agent_capability_scores


def _pct(sorted_arr, p):
    if not sorted_arr:
        return None
    return sorted_arr[min(int(p / 100 * len(sorted_arr)), len(sorted_arr) - 1)]


def edit_ratio_distribution():
    solved, failed = [], []
    for agent in config.AGENTS:
        for inst in config.list_instances():
            res = resolved(agent, inst)
            if res is None:
                continue
            try:
                gold = build_gold_reference(inst)
            except Exception:
                continue
            r = localization_signals(gold, load_agent_patch(agent, inst), res)["edit_size_ratio"]
            if r is not None:
                (solved if res else failed).append(r)
    return sorted(solved), sorted(failed)


def suggest_over_edit_threshold():
    solved, failed = edit_ratio_distribution()
    allr = sorted(solved + failed)
    observed = (_pct(solved, 90), _pct(solved, 95), _pct(allr, 90))
    cand = sorted({3.0, 5.0, 8.0} | {round(value, 1) for value in observed if value is not None})
    fire = {}
    for thr in cand:
        sf = sum(r > thr for r in solved) / len(solved) if solved else None
        ff = sum(r > thr for r in failed) / len(failed) if failed else None
        fire[thr] = (sf, ff)
    return {
        "n_solved": len(solved),
        "n_failed": len(failed),
        "solved_pctiles": {p: round(_pct(solved, p), 2) if solved else None for p in (50, 75, 90, 95, 99)},
        "failed_pctiles": {p: round(_pct(failed, p), 2) if failed else None for p in (50, 75, 90, 95, 99)},
        "fire_rate_solved_failed": fire,
    }


def blame_sensitivity():
    settings = {
        "v0 (0.5/0.4)": {"primary": 1.0, "verification": 0.5, "secondary": 0.4},
        "low-leak (0.2/0.2)": {"primary": 1.0, "verification": 0.2, "secondary": 0.2},
        "high-leak (0.8/0.8)": {"primary": 1.0, "verification": 0.8, "secondary": 0.8},
        "primary-only (0/0)": {"primary": 1.0, "verification": 0.0, "secondary": 0.0},
    }
    out = {}
    for name, roles in settings.items():
        out[name] = {}
        for a in config.AGENTS:
            r = agent_capability_scores(a, use_trajectory=False, roles=roles, bootstrap=1)
            out[name][a] = {c: round(v["score"], 1) for c, v in r["capabilities"].items()}
    return out


def attribution_sensitivity():
    """v0.32 — sensitivity of the ATTRIBUTION counts (per-capability primaries
    + conjunctive count) to the uncalibrated role multipliers, measured on the
    FULL trajectory-enabled pipeline. blame_sensitivity() above covers only the
    score RANKING on the loc/edit slice, which never exercises the contributing
    candidates that drove primary/conjunctive instability — quoting it as
    robustness for the attribution counts was an overclaim (v0.32 review).
    With causal-only dominance the verification multiplier no longer enters
    the dominance test, so the counts should be near-invariant to it — this
    function is the check. The LLM arbiter is NOT invoked (rule layer only)."""
    from collections import Counter
    from .detect import detect
    from .rootcause import build_blame

    settings = {
        "v0 (0.5/0.4)": {"primary": 1.0, "verification": 0.5, "secondary": 0.4},
        "verif-0.3": {"primary": 1.0, "verification": 0.3, "secondary": 0.4},
        "verif-0.7": {"primary": 1.0, "verification": 0.7, "secondary": 0.4},
        "secondary-0.6": {"primary": 1.0, "verification": 0.5, "secondary": 0.6},
    }
    detections = []
    for a in config.AGENTS:
        for inst in config.list_instances():
            d = detect(inst, a, use_trajectory=True)
            if d["resolved"] is False:
                detections.append(d)
    out = {}
    for name, roles in settings.items():
        primaries: Counter = Counter()
        conjunctive = 0
        for d in detections:
            b = build_blame(d, roles=roles)
            if b["status"] == "primary":
                primaries[b["primary"]["capability"]] += 1
            elif b["status"] == "conjunctive":
                conjunctive += 1
        out[name] = {"primaries": dict(primaries), "conjunctive": conjunctive}
    return out
