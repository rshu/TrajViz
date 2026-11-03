#!/usr/bin/env python3
"""Per-test (eval+) coverage report — which cases carry an official-harness
per-test record and what failure shapes they show (v0.39).

Reads data/pertest/<agent>/<instance_id>.json through agent_trajviz.awe._engine.outcomes.load_pertest
(the same loader the detector uses) and prints, per agent: coverage of the
failed cases, the shape distribution, and the instances in each non-trivial
shape. Reproduce the §0at pilot table with:

    AWE_ARGUS_ROOT=... python3 scripts/pertest_report.py [--agent claude_code]
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.outcomes import load_pertest

ORDER = (
    "unfixed",
    "unfixed_regressive",
    "partial_fix",
    "partial_fix_regressive",
    "regression_only",
    "resolved",
    "patch_apply_failed",
    "no_gating_tests",
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="per-test (eval+) coverage and failure shapes")
    ap.add_argument("--agent", choices=config.AGENTS, action="append", help="restrict to this agent (repeatable)")
    agents = ap.parse_args().agent or config.AGENTS
    for agent in agents:
        n_failed = n_cov = 0
        shapes: Counter = Counter()
        members: dict = defaultdict(list)
        for a, inst, res, _gold in iter_cases(agents=[agent]):
            if res is not False:
                continue
            n_failed += 1
            pt = load_pertest(a, inst)
            if pt is None:
                continue
            n_cov += 1
            shapes[pt["shape"]] += 1
            members[pt["shape"]].append((inst, pt["f2p_pass_fraction"], len(pt["p2p_broken"])))
        print(
            f"\n## {agent}: {n_cov}/{n_failed} failed cases have a per-test record"
            + ("" if n_cov else "  (none — run the official harness; see data/discovery/pertest_pilot.md)")
        )
        if not n_cov:
            continue
        print(f"{'shape':24s} {'n':>4s}")
        for s in ORDER:
            if shapes.get(s):
                print(f"{s:24s} {shapes[s]:4d}")
        for s in ("partial_fix", "partial_fix_regressive", "regression_only", "patch_apply_failed"):
            if members.get(s):
                print(f"\n  {s}:")
                for inst, frac, nb in sorted(members[s]):
                    print(f"    {inst:40s} f2p_pass_fraction={frac}  p2p_broken={nb}")


if __name__ == "__main__":
    main()
