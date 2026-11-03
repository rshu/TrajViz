#!/usr/bin/env python3
"""Threshold-stability sweep + the alt-solution bound (v0.36, DESIGN.md §0al).

The promoted anchors earned their thresholds through cohort contrast; four
hand-set constants never did. This sweep shows how the affected fault counts
move when each constant is perturbed one step either way — a stability check,
not a re-calibration (verdict-changing sensitivity would demand one).

  NEAR_GOLD_FRACTION (0.8)      — incorrect_patch medium/high split on
                                  right_place_wrong_change failures
  partial-loc editing cut (0.5) — incorrect_patch co-blame on loc_partial
  gave_up_early reads (<3)      — disengagement split of no-source-edit failures
  exploration_loop reads (>=4)  — behavioral flag prevalence, failed vs solved

Second section: the FAILED-ALT-SOLUTION BOUND. On solved cases the loc flags
show how often agents legitimately succeed off the gold path; that rate bounds
how often a failed case's localization blame could be a mislabeled valid
alternative (the confound outcome-conditioning documents but did not quantify).

Reproduce:  python3 scripts/threshold_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.trajsignals import signals_cached


def collect():
    rows = []
    for agent, inst, res, gold in iter_cases():
        patch = load_agent_patch(agent, inst)
        sig = signals_cached(agent, inst, gold)
        read_set = set(sig.read_set) if sig is not None else None
        loc = localization_signals(gold, patch, res, read_set=read_set)
        ov = loc.get("hit_overlap") or {}
        rows.append(
            {
                "agent": agent,
                "inst": inst,
                "resolved": res,
                "flag": loc["flag"],
                "hit": bool(loc["hit"]),
                "missed_gold": bool(loc["missed_gold"]),
                "frac": ov.get("reproduced_fraction"),
                "ran_any_test": sig.ran_any_test if sig else None,
                "files_read": len(sig.read_set) if sig else None,
                "truncated": sig.truncated_run if sig else None,
                "max_reads": getattr(sig, "max_reads_single_file", 0) if sig else 0,
            }
        )
    return rows


def main() -> None:
    rows = collect()
    failed = [r for r in rows if r["resolved"] is False]
    solved = [r for r in rows if r["resolved"] is True]
    print(f"cases={len(rows)} failed={len(failed)} solved={len(solved)}")
    if not failed or not solved:
        sys.exit(
            "need both failed AND solved cases to sweep — is the eval "
            "outcome data (eval_<agent>.json) present under "
            "AWE_ARGUS_ROOT? (a checkout without outcomes yields 0 cases)"
        )

    rpwc = [r for r in failed if r["flag"] == "right_place_wrong_change"]
    print(f"\nNEAR_GOLD_FRACTION sweep (right_place_wrong_change, n={len(rpwc)}):")
    for th in (0.7, 0.8, 0.9):
        near = sum(1 for r in rpwc if r["frac"] is not None and r["frac"] >= th)
        print(f"  >= {th:.1f}: medium(near-gold)={near}  high={len(rpwc) - near}")

    part = [r for r in failed if r["flag"] == "loc_partial" and r["hit"]]
    print(f"\npartial-loc editing-co-blame cut (loc_partial with a hit, n={len(part)}):")
    for th in (0.4, 0.5, 0.6):
        n = sum(1 for r in part if r["frac"] is not None and r["frac"] < th)
        print(f"  frac < {th:.1f}: incorrect_patch co-blamed on {n}")

    nse = [r for r in failed if r["flag"] == "no_source_edit" and r["ran_any_test"] is not None and not r["truncated"]]
    print(f"\ngave_up_early reads threshold (no_source_edit, untruncated, n={len(nse)}):")
    for th in (2, 3, 4):
        n = sum(1 for r in nse if not r["ran_any_test"] and (r["files_read"] or 0) < th)
        print(f"  < {th} files read: gave_up_early={n}  no_patch_produced={len(nse) - n}")

    print("\nexploration_loop threshold (behavioral, prevalence % failed/solved):")
    for th in (3, 4, 5):
        pf = 100 * sum(1 for r in failed if r["max_reads"] >= th) / len(failed)
        ps = 100 * sum(1 for r in solved if r["max_reads"] >= th) / len(solved)
        print(f"  >= {th} reads: {pf:.1f}% / {ps:.1f}%")

    print(f"\n--- failed-alt-solution bound (solved cases, n={len(solved)}) ---")
    alt = sum(1 for r in solved if r["flag"] == "resolved_alt_solution")
    full_alt = sum(1 for r in solved if r["flag"] == "resolved_alt_solution" and not r["hit"])
    print(f"resolved_alt_solution (some gold file unedited): {alt} ({100 * alt / len(solved):.1f}%)")
    print(f"  of which FULL alt path (zero gold files edited): {full_alt} ({100 * full_alt / len(solved):.1f}%)")
    print(
        "reading: the full-alt rate bounds how often a failed loc_miss could "
        "be a valid-but-failed alternative rather than a search miss."
    )


if __name__ == "__main__":
    main()
