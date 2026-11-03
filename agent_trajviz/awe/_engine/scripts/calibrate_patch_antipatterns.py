#!/usr/bin/env python3
"""Patch anti-pattern calibration — CANDIDATE study, nothing promoted (v0.39,
DESIGN.md §0av).

Search-based program repair learned that plausible-but-incorrect patches share
syntactic anti-patterns (Tan, Yoshida, Prasad, Roychoudhury, FSE 2016:
anti-delete CFG exit node / control statement, anti-append early exit / trivial
condition). This script asks whether the LLM-agent analogues discriminate
failed from solved runs on this corpus, in two views:

  ALL cases            — confounded: editing OUTSIDE the gold set is already
                         failure-associated (excessive_extra_edits), so the
                         non-gold arms inherit that association.
  CONDITIONED          — cases whose patch edited EVERY gold source file (the
                         right_place_wrong_change population vs solves with full
                         recall), gold-file hunks only: the honest test of
                         "wrong kind of change in the right place".

Binary family per view, unpaired bootstrap p, Holm-corrected. A feature that
clears the gate here is a candidate ATTRIBUTE for incorrect_patch, pending the
§0am difficulty-strata and threshold-stability checks — never a type from this
script alone.

Reproduce:  AWE_ARGUS_ROOT=... python3 scripts/calibrate_patch_antipatterns.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.diffutil import is_test_file, parse_unified_diff
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.significance import holm

PATTERNS = {
    "A1 deleted raise": re.compile(r"^\s*raise\b"),
    "A1 deleted return": re.compile(r"^\s*return\b"),
    "A2 deleted if/elif": re.compile(r"^\s*(?:if|elif)\b"),
    "A2 deleted loop": re.compile(r"^\s*(?:for|while)\b"),
    "A6 added bare return": re.compile(r"^\s*return(?:\s+None)?\s*$"),
    "added try:": re.compile(r"^\s*try\s*:"),
    "added broad except": re.compile(r"^\s*except(?:\s*:|\s+(?:Exception|BaseException)\b)"),
    "added isinstance( guard": re.compile(r".*\bisinstance\("),
    "added hasattr/getattr": re.compile(r".*\b(?:hasattr|getattr)\("),
    "added `is None` guard": re.compile(r"^\s*if\s+\w+(?:\.\w+)*\s+is\s+None\s*:"),
}
_DELETED = {k for k in PATTERNS if "deleted" in k}


def features(text: str, gold_src: set, only_gold: bool) -> dict:
    out = {f"{k} ({'gold' if only_gold else 'nongold'})": False for k in PATTERNS}
    for fd in parse_unified_diff(text):
        if is_test_file(fd.path) or fd.path.endswith((".md", ".rst", ".txt")):
            continue
        in_gold = fd.path in gold_src
        if only_gold != in_gold:
            continue
        tag = "gold" if only_gold else "nongold"
        for k, rx in PATTERNS.items():
            lines = fd.removed_lines if k in _DELETED else fd.added_lines
            if any(rx.match(l) for l in lines):
                out[f"{k} ({tag})"] = True
    return out


def table(title, rows):
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"\n== {title}: failed {len(failed)}  solved {len(solved)} ==")
    if not rows:
        print("   (no cases in this cohort)")
        return
    feats = [k for k in rows[0] if k not in ("failed",)]
    rng = random.Random(41)
    stats, pvals = [], []
    for name in feats:
        fv = [float(r[name]) for r in failed]
        sv = [float(r[name]) for r in solved]
        assoc, ct = binary_association(fv, sv), contrast(fv, sv, rng)
        stats.append((name, assoc, ct))
        pvals.append(ct["p"])
    print(f"{'feature':40s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p_holm':>7s}")
    for (name, a, ct), p_h in zip(stats, holm(pvals), strict=False):
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{name:40s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {p_h:7.4f}"
        )


def main() -> None:
    all_rows, cond_rows = [], []
    for agent, inst, res, gold in iter_cases():
        patch = load_agent_patch(agent, inst)
        if not patch.exists:
            continue
        text = config.patch_path(agent, inst).read_text(errors="replace", encoding="utf-8")
        gold_src = set(gold.gold_files)
        row = {"failed": 0 if res else 1}
        row.update(features(text, gold_src, only_gold=True))
        row.update(features(text, gold_src, only_gold=False))
        all_rows.append(row)
        loc = localization_signals(gold, patch, res)
        if loc.get("recall") == 1.0:
            crow = {"failed": 0 if res else 1}
            crow.update(features(text, gold_src, only_gold=True))
            cond_rows.append(crow)
    table("ALL cases (non-gold arms confounded with excessive_extra_edits)", all_rows)
    table("CONDITIONED on full gold-file recall — gold-file hunks only", cond_rows)
    print(
        "\nVerdict rule: candidate attribute only if the CONDITIONED gold arm clears Holm; "
        "then run difficulty strata + threshold stability (§0am) before any promotion."
    )


if __name__ == "__main__":
    main()
