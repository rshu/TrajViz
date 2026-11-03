"""Paired significance testing for cross-agent comparisons.

All agents run the IDENTICAL task set, so agent-vs-agent comparisons are PAIRED on
shared instances. This module is PURE: it operates on per-instance vectors and
imports nothing from detect/scoring/rootcause, so it is testable in isolation and
cannot silently couple to the thing it is checking.

A "vector" is {instance_id: value} — a blame share in [0,1] (lower = better) or a
0/1 failure indicator. paired_bootstrap_diff pairs two vectors on shared keys and
bootstraps the mean difference; holm() corrects a family of p-values;
rank_with_significance decides a row's best/worst and the raw p separating each
from its nearest neighbour (Holm correction is applied by the caller across the
whole family)."""

from __future__ import annotations

import random


MIN_PAIRS = 5  # below this a bootstrap "significance" is an artifact, not evidence


def paired_bootstrap_diff(vec_a: dict, vec_b: dict, rng: random.Random, B: int = 2000) -> dict:
    """Bootstrap the mean of (a - b) over instances present in BOTH vectors.
    Two-sided percentile p = 2*min(share<=0, share>=0) with add-one smoothing
    ((k+1)/(B+1)), so p is never exactly 0 — a constant-sign diff vector (even a
    single pair) must not survive Holm as p=0. n_pairs == 0 -> p = 1.0 (not
    separable); n_pairs < MIN_PAIRS -> p = 1.0 (too few pairs to claim anything)."""
    keys = [k for k in vec_a if k in vec_b]
    n = len(keys)
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_two_sided": 1.0, "n_pairs": 0}
    diffs = [vec_a[k] - vec_b[k] for k in keys]
    point = sum(diffs) / n
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[min(B - 1, max(0, round(0.025 * (B - 1))))]
    hi = means[min(B - 1, max(0, round(0.975 * (B - 1))))]
    le = (sum(1 for m in means if m <= 0) + 1) / (B + 1)
    ge = (sum(1 for m in means if m >= 0) + 1) / (B + 1)
    p = min(1.0, 2 * min(le, ge))
    if n < MIN_PAIRS:
        p = 1.0
    return {"diff": point, "lo": lo, "hi": hi, "p_two_sided": p, "n_pairs": n}


def holm(pvals: list) -> list:
    """Holm-Bonferroni step-down adjusted p-values, returned in input order:
    sort ascending, multiply the k-th smallest by (m - k), enforce monotone
    non-decreasing, clamp to 1."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def rank_with_significance(scores: dict, vectors: dict, rng: random.Random, B: int = 2000) -> dict:
    """For one capability row: best = highest score (lowest rate), worst = lowest
    score. Returns the raw paired p separating best from runner-up and worst from
    next-worst. `scores`={agent: score}, `vectors`={agent: {inst: rate}} over the
    agents that HAVE this capability."""
    agents = list(scores)
    out = {"best": None, "worst": None, "p_best": 1.0, "p_worst": 1.0, "n_best": 0, "n_worst": 0}
    if not agents:
        return out
    if len(agents) == 1:
        out["best"] = out["worst"] = agents[0]
        return out
    by_score = sorted(agents, key=lambda a: scores[a], reverse=True)
    best, runner = by_score[0], by_score[1]
    worst, next_worst = by_score[-1], by_score[-2]
    rb = paired_bootstrap_diff(vectors.get(best, {}), vectors.get(runner, {}), rng, B)
    rw = paired_bootstrap_diff(vectors.get(worst, {}), vectors.get(next_worst, {}), rng, B)
    out.update(
        best=best,
        worst=worst,
        p_best=rb["p_two_sided"],
        n_best=rb["n_pairs"],
        p_worst=rw["p_two_sided"],
        n_worst=rw["n_pairs"],
    )
    return out
