#!/usr/bin/env python3
"""Robust over-edit metric calibration (v0.37, DESIGN.md §0am).

The v0.9 `over_editing` demotion left a recorded redesign: the ratio-to-gold
metric is unstable for tiny gold patches; the robust form is ABSOLUTE mass —
extra source LOC and non-gold source file count. This script calibrates both
(continuous contrast + binary cuts, Holm over the binary family) so the flag
either earns a failure association or the negative is recorded next to the
v0.9 one.

Reproduce:  python3 scripts/calibrate_overedit.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.evidence import OVEREDIT_EXTRA_LOC
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.significance import holm

# Cuts are DERIVED from the shipped threshold so re-running after a re-tune
# still calibrates the deployed detector (v0.37 review fix: the first draft
# re-derived the metric inline with its own hardcoded cuts — a definition
# drift channel between the calibration and the detector it licenses).
BINARY_CUTS = (
    ("extra_files>=3", lambda r: r["n_extra"] >= 3),
    ("extra_files>=5", lambda r: r["n_extra"] >= 5),
    (f"extra_loc>={OVEREDIT_EXTRA_LOC}", lambda r: r["extra_loc"] >= OVEREDIT_EXTRA_LOC),
    (f"extra_loc>={2 * OVEREDIT_EXTRA_LOC}", lambda r: r["extra_loc"] >= 2 * OVEREDIT_EXTRA_LOC),
    (f"extra_loc>={4 * OVEREDIT_EXTRA_LOC}", lambda r: r["extra_loc"] >= 4 * OVEREDIT_EXTRA_LOC),
)


def rows_of():
    """One row per (agent, case) with a patch, using the SHIPPED metric —
    ``localization_signals()['extra_src_loc']`` — so this script always
    calibrates exactly what the detector consumes."""
    rows = []
    for agent, inst, res, gold in iter_cases():
        patch = load_agent_patch(agent, inst)
        if not patch.exists:
            continue
        loc = localization_signals(gold, patch, res)
        rows.append(
            {
                "agent": agent,
                "inst": inst,
                "failed": 0 if res else 1,
                "n_extra": len(loc["extra_files"]),
                "extra_loc": loc["extra_src_loc"],
            }
        )
    return rows


def main() -> None:
    rows = rows_of()
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"cases with a patch: {len(rows)}  failed {len(failed)}  solved {len(solved)}")

    rng = random.Random(29)
    for metric in ("n_extra", "extra_loc"):
        ct = contrast([float(r[metric]) for r in failed], [float(r[metric]) for r in solved], rng)
        print(
            f"{metric:12s} mean failed {ct['mean_failed']:.2f} vs solved "
            f"{ct['mean_solved']:.2f}  diff {ct['diff']:+.2f} "
            f"[{ct['lo']:+.2f},{ct['hi']:+.2f}]  p={ct['p']:.4f}"
        )

    stats, pvals = [], []
    for name, pred in BINARY_CUTS:
        fv = [float(pred(r)) for r in failed]
        sv = [float(pred(r)) for r in solved]
        assoc, ct = binary_association(fv, sv), contrast(fv, sv, rng)
        stats.append((name, assoc, ct))
        pvals.append(ct["p"])
    print(f"\n{'cut':18s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p_holm':>7s}")
    for (name, a, ct), p_h in zip(stats, holm(pvals), strict=False):
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{name:18s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {p_h:7.4f}"
        )


if __name__ == "__main__":
    main()
