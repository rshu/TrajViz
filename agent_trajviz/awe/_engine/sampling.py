"""Stratified sampler for the human-calibration set.

Failures are stratified by (agent x detector's primary capability) so annotators
see a balanced mix rather than ~60% incorrect_patch; a small solved control set
is included so humans can confirm 'no error' cases too."""

from __future__ import annotations

import random
from collections import defaultdict

from . import config
from .outcomes import resolved
from .detect import detect
from .rootcause import build_blame


def _primary_capability(agent: str, inst: str) -> str | None:
    cb = build_blame(detect(inst, agent, use_trajectory=False)).get("capability_blame", {})
    return max(cb, key=cb.get) if cb else None


def stratified_sample(n: int = 60, seed: int = 0, solved_fraction: float = 0.15):
    rng = random.Random(seed)
    failed_strata: dict = defaultdict(list)
    solved: list = []
    for agent in config.AGENTS:
        for inst in config.list_instances():
            res = resolved(agent, inst)
            if res is None:
                continue
            if res is False:
                failed_strata[(agent, _primary_capability(agent, inst))].append((agent, inst))
            else:
                solved.append((agent, inst))

    buckets = list(failed_strata.values())
    for b in buckets:
        rng.shuffle(b)
    n_solved = int(n * solved_fraction)
    n_failed = n - n_solved

    picked_failed = []
    i = 0
    while len(picked_failed) < n_failed and any(buckets):
        b = buckets[i % len(buckets)]
        if b:
            picked_failed.append(b.pop())
        i += 1
        if i > 100000:
            break
    rng.shuffle(solved)
    return picked_failed, solved[:n_solved]
