"""Capability scoring — the corrected Section-12 method.

Per capability K and agent: error_rate(K) = mean over K-opportunity tasks of the
blame-share assigned to K (0 on solved/unblamed tasks). This is bounded [0,1],
opportunity-normalized, and reported with a bootstrap CI — not a hand-scaled
penalty. score(K) = 100*(1 - rate) for readability, but the rate + CI is the
primary number.

DECOMPOSITION CAVEAT: per-failed-task blame sums to 1 across capabilities, so the
blame MASS decomposes the failure count. The opportunity-normalized RATES only
decompose the failure rate when every capability shares the all-tasks denominator
(the loc/edit-only slice). Restricted-denominator capabilities (verification /
self-repair / tool-use / the judge caps) divide their mass by a smaller
opportunity count, so sum_K rate(K) exceeds the failure rate whenever they carry
mass — the rates are per-opportunity intensities, not shares of the failure rate.
The two judge caps have an additional comparability caveat: their opportunity is
verdict-availability (by default judged FAILED tasks only), so their denominators
are per-agent and outcome-conditioned — do not read cross-agent judge-cap rates
as same-denominator comparisons unless the judge was run identically (e.g. with
--include-resolved) for every agent compared."""

from __future__ import annotations

import random
from collections import Counter

from . import config
from .gold import build_gold_reference
from .outcomes import resolved
from .detect import detect
from .rootcause import build_blame
from .judge import JUDGE_CAPS, active as judge_active
from .arbiter import auto_adjudicate
from .evidence import strength_of

ALL_SLICE_CAPS = ["code_localization", "code_editing", "code_verification", "self_repair_loop", "tool_use"]
_TIERS = ("deductive", "associational", "model_inferred")


def compose_strength(blame: list) -> dict:
    """Roll a list of blame entries [{capability, weight}] up into evidence-strength
    shares — how much of the diagnosed blame rests on deductive vs weaker evidence.
    The trust profile: the headline should be deductive; weaker tiers are flagged."""
    out = {t: 0.0 for t in _TIERS}
    total = 0.0
    for b in blame:
        w = b.get("weight", 0.0) or 0.0
        out[strength_of(b["capability"], b.get("error_type"))] += w
        total += w
    return {t: (out[t] / total if total else 0.0) for t in _TIERS}


def strength_composition(
    agent: str, instances=None, use_trajectory: bool = True, use_judge: bool | None = None
) -> dict:
    """Per-agent diagnosis trust profile: the share of total blame mass (and of the
    PRIMARY causes) at each evidence-strength tier, over the agent's failures."""
    instances = list(instances if instances is not None else config.list_instances())
    if use_judge is None:
        use_judge = judge_active(agent)
    adjudicate = auto_adjudicate(agent)
    mass = {t: 0.0 for t in _TIERS}
    prim = {t: 0 for t in _TIERS}
    n_primary = 0
    for inst in instances:
        if resolved(agent, inst) is not False:
            continue
        d = detect(inst, agent, use_trajectory=use_trajectory, use_judge=use_judge)
        b = build_blame(d, adjudicate=adjudicate)
        for entry in b.get("blame", []):
            mass[strength_of(entry["capability"], entry.get("error_type"))] += entry["weight"]
        p = b.get("primary")
        if p:
            n_primary += 1
            prim[strength_of(p["capability"], p.get("error_type"))] += 1
    tot = sum(mass.values())
    return {
        "by_mass": {t: (mass[t] / tot if tot else 0.0) for t in _TIERS},
        "by_primary": prim,
        "n_primary": n_primary,
    }


def _bootstrap_ci(arr: list[float], rng: random.Random, B: int) -> tuple[float, float]:
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(B):
        means.append(sum(arr[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    # Nearest-rank percentile over B order statistics (the old int(0.025*B) was
    # off by one at the low end).
    lo_i = min(B - 1, max(0, round(0.025 * (B - 1))))
    hi_i = min(B - 1, max(0, round(0.975 * (B - 1))))
    return (means[lo_i], means[hi_i])


def agent_capability_scores(
    agent: str,
    instances=None,
    use_trajectory: bool = True,
    bootstrap: int = 1000,
    seed: int = 0,
    roles=None,
    use_judge: bool | None = None,
    return_vectors: bool = False,
) -> dict:
    instances = list(instances if instances is not None else config.list_instances())
    if use_judge is None:
        use_judge = judge_active(agent)
    adjudicate = auto_adjudicate(agent)  # LLM arbiter (no-op unless verdicts cached)
    caps = ALL_SLICE_CAPS if use_trajectory else ["code_localization", "code_editing"]
    if use_judge:
        caps = caps + list(JUDGE_CAPS)

    per_cap: dict[str, list[float]] = {k: [] for k in caps}
    per_cap_inst: dict[str, dict] = {k: {} for k in caps}
    fail_inst: dict[str, int] = {}
    err_mass: dict[str, Counter] = {k: Counter() for k in caps}
    incidence = {k: 0 for k in caps}
    unknown_mass = {k: 0.0 for k in caps}
    n = n_failed = unattributed = 0
    unscored = []
    if len(set(instances)) != len(instances):
        raise ValueError("Scoring instances must be unique")

    for inst in instances:
        # Aggregate benchmark exports namespace instances by configuration.  Check
        # membership before parsing the (potentially large) requirement payload;
        # this is equivalent to the former ordering because missing outcomes were
        # skipped immediately after the parse.
        res = resolved(agent, inst)
        if res is None:
            unscored.append(inst)
            continue
        try:
            build_gold_reference(inst)
        except Exception as exc:
            raise ValueError(f"Cannot score {agent}/{inst}: invalid gold reference") from exc
        n += 1
        if res is False:
            n_failed += 1
        fail_inst[inst] = 1 if res is False else 0
        d = detect(inst, agent, use_trajectory=use_trajectory, use_judge=use_judge)
        blame = build_blame(d, roles=roles, adjudicate=adjudicate)
        if blame["status"] == "unattributed":
            unattributed += 1
        cap_blame = blame.get("capability_blame", {})
        opp = d["opportunities"]
        supported_caps = {b["capability"] for b in blame.get("blame", []) if b["weight"] > 0}
        for k in caps:
            if opp.get(k):
                share = cap_blame.get(k, 0.0)
                per_cap[k].append(share)
                per_cap_inst[k][inst] = share
                incidence[k] += int(k in supported_caps)
                unknown_mass[k] += cap_blame.get("unattributed", 0.0)
        for b in blame.get("blame", []):
            if b["capability"] in err_mass and b["weight"] > 0:
                err_mass[b["capability"]][b["error_type"]] += b["weight"]

    out = {}
    for k in caps:
        arr = per_cap[k]
        # A capability with no opportunity for this agent is "not measured" —
        # omit it rather than report a vacuous score of 100.
        if not arr:
            continue
        rate = sum(arr) / len(arr) if arr else 0.0
        # One RNG stream PER capability: with a single shared stream, toggling the
        # judge caps on/off would shift every later capability's resamples and
        # perturb unrelated CIs.
        lo, hi = _bootstrap_ci(arr, random.Random(f"{seed}:{k}"), bootstrap)
        out[k] = {
            "rate": rate,
            "score": 100 * (1 - rate),
            "score_ci": (100 * (1 - hi), 100 * (1 - lo)),  # CI on the score
            "opportunity": len(arr),
            "fault_incidence": incidence[k] / len(arr),
            "unattributed_opportunity_mass": unknown_mass[k],
            # Bounds assign unresolved mass anywhere; these are not sampling CIs.
            "score_identification_bounds": (max(0.0, 100 * (1 - rate - unknown_mass[k] / len(arr))), 100 * (1 - rate)),
            "top_errors": err_mass[k].most_common(3),
        }
    result = {
        "agent": agent,
        "n_requested": len(instances),
        "unscored_instances": unscored,
        "n_tasks": n,
        "n_failed": n_failed,
        "failure_rate": n_failed / n if n else 0.0,
        "unattributed": unattributed,
        "capabilities": out,
    }
    if return_vectors:
        result["vectors"] = {k: per_cap_inst[k] for k in out}  # only measured caps
        result["fail_vector"] = fail_inst
    return result
