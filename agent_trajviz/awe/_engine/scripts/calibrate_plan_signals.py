#!/usr/bin/env python3
"""Cohort calibration of the v20 PLAN signals (DESIGN.md §0ay).

The scaffolds' explicit plan tools (Claude Code TodoWrite, OpenCode todowrite,
Codex update_plan) were mapped to `other` with empty args until schema 20; this
is the first contrast of what they carry, failed vs solved, unpaired bootstrap
p with Holm over the family (the §0y discipline). Nothing here blames: a
feature that clears the gate becomes a candidate for the promotion chain, a
flat or protective one is recorded so it is not re-proposed.

  plan_declared           — the run updated a plan at least once (main thread)
  plan_before_first_edit  — the first plan update preceded the first edit
  plan_left_open          — the LAST plan still had pending / in_progress items
  plan_revised_3plus      — three or more plan updates (re-planning churn)
  edit_without_plan       — edited source but never declared a plan

Reproduce:  python3 scripts/calibrate_plan_signals.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.significance import holm
from agent_trajviz.awe._engine.trajsignals import signals_cached

CANDS = ("plan_declared", "plan_before_first_edit", "plan_left_open", "plan_revised_3plus", "edit_without_plan")


def rows_of() -> list:
    rows = []
    for agent, inst, res, gold in iter_cases():
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        rows.append(
            {
                "agent": agent,
                "inst": inst,
                "failed": 0 if res else 1,
                "plan_declared": int(sig.plan_declared),
                "plan_before_first_edit": int(sig.plan_before_first_edit),
                "plan_left_open": int(sig.plan_items_open_at_end > 0),
                "plan_revised_3plus": int(sig.n_plan_updates >= 3),
                "edit_without_plan": int(bool(sig.edit_set_traj) and not sig.plan_declared),
                "n_plan_updates": sig.n_plan_updates,
            }
        )
    return rows


def _table(title: str, rows: list, seed: int) -> None:
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"\n== {title}: failed {len(failed)}  solved {len(solved)} ==")
    if not failed or not solved:
        print("   (cohort too small)")
        return
    rng = random.Random(seed)
    stats, pvals = {}, []
    for c in CANDS:
        fv = [float(r[c]) for r in failed]
        sv = [float(r[c]) for r in solved]
        stats[c] = (binary_association(fv, sv), contrast(fv, sv, rng))
        pvals.append(stats[c][1]["p"])
    print(f"{'feature':26s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p':>7s} {'p_holm':>7s}")
    for c, p_h in zip(CANDS, holm(pvals), strict=False):
        a, ct = stats[c]
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{c:26s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {ct['p']:7.4f} {p_h:7.4f}"
        )


def main() -> None:
    rows = rows_of()
    if not rows:
        print("no cases found — is AWE_ARGUS_ROOT set?", file=sys.stderr)
        return
    print(f"cases={len(rows)}")
    print("\nplan-tool usage per agent (runs with >=1 plan update / all runs):")
    for agent in config.AGENTS:
        ar = [r for r in rows if r["agent"] == agent]
        if not ar:
            continue
        used = [r for r in ar if r["plan_declared"]]
        upd = Counter(r["n_plan_updates"] for r in used)
        print(
            f"  {agent:14s} {len(used):4d}/{len(ar):4d}  "
            f"left_open {sum(r['plan_left_open'] for r in used):3d}  "
            f"before_first_edit {sum(r['plan_before_first_edit'] for r in used):3d}  "
            f"median updates {sorted(upd.elements())[len(list(upd.elements())) // 2] if used else 0}"
        )
    _table("all agents", rows, seed=20)
    # Within-agent contrast: plan-tool usage is scaffold- and model-specific,
    # so the pooled RR can be a mixture of agent base rates (§0am strata).
    for agent in config.AGENTS:
        ar = [r for r in rows if r["agent"] == agent]
        if sum(r["plan_declared"] for r in ar) >= 20:
            _table(agent, ar, seed=21)


if __name__ == "__main__":
    main()
