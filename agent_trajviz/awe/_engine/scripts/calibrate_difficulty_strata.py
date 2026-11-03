#!/usr/bin/env python3
"""Difficulty-stratified re-calibration (v0.37 — the recorded METH-OBS-1
follow-up): do the calibrated anchors hold WITHIN difficulty strata, or is
their solved-vs-failed contrast a difficulty proxy?

Labels: SWE-bench Verified's human difficulty annotations
(data/discovery/swebench_verified_difficulty.json, fetched from the HF
datasets-server 2026-08-28). Strata: '<15 min fix' / '15 min - 1 hour' /
'1-4 hours+' (the two largest strata plus the merged tail — 3 cases of
'>4 hours' cannot stand alone). Anchors re-checked: the three strongest
patch/content-grounded ones (solved_unrelated_candidate RR 8.09;
reverted_own_change RR 2.10; extra_src_loc>=50 RR 2.38).

Reproduce:  python3 scripts/calibrate_difficulty_strata.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cohort import binary_association
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.evidence import issue_code_tokens
from agent_trajviz.awe._engine.signals import localization_signals, OVEREDIT_EXTRA_LOC
from agent_trajviz.awe._engine.trajsignals import signals_cached

# Optional override for callers calibrating an in-memory synthetic corpus.
# Recorded labels belong to project data and are read only when requested.
LABELS: dict[str, str] | None = None
STRATA = (
    ("<15 min fix", ("<15 min fix",)),
    ("15 min - 1 hour", ("15 min - 1 hour",)),
    ("1-4 hours+", ("1-4 hours", ">4 hours")),
)


def load_labels(path: str | Path | None = None) -> dict[str, str]:
    source = Path(path) if path is not None else config.PROJECT_DATA / "discovery/swebench_verified_difficulty.json"
    with source.open(encoding="utf-8") as stream:
        labels = json.load(stream)
    if not isinstance(labels, dict):
        raise ValueError("Difficulty labels must be a JSON object keyed by instance ID.")
    return labels


def rows_of(labels: dict[str, str] | None = None):
    labels = labels if labels is not None else (LABELS if LABELS is not None else load_labels())
    rows = []
    for agent, inst, res, gold in iter_cases():
        if inst not in labels:
            continue
        patch = load_agent_patch(agent, inst)
        sig = signals_cached(agent, inst, gold)
        loc = localization_signals(gold, patch, res, read_set=set(sig.read_set) if sig else None)
        # OUTCOME-FREE form of the unrelated-problem anchor (the shipped
        # signal embeds `resolved is False` and would make the contrast
        # circular): total gold miss + zero identifier overlap, computed
        # from the same primitives without the outcome gate.
        unrelated_free = bool(
            not loc["hit"]
            and loc["gold_files"]
            and loc["edited_source_files"]
            and patch
            and patch.exists
            and patch.symbols
            and issue_code_tokens(gold.problem_statement)
            and not loc["issue_symbol_overlap"]
        )
        rows.append(
            {
                "difficulty": labels[inst],
                "failed": 0 if res else 1,
                "solved_unrelated": int(unrelated_free),
                "reverted_own_change": int(bool(sig and sig.edit_reverted_own_change)),
                "excessive_extra_edits": int(loc.get("extra_src_loc", 0) >= OVEREDIT_EXTRA_LOC),
            }
        )
    return rows


def rr(rows, key):
    """(risk_ratio, n_present) via the shared cohort primitive — the same
    definition every calibration table uses (closes a §0an cleanup debt)."""
    a = binary_association([r[key] for r in rows if r["failed"]], [r[key] for r in rows if not r["failed"]])
    return a["risk_ratio"], a["n_present"]


def main() -> None:
    rows = rows_of()
    if not rows:
        sys.exit(
            "0 cases collected — either the eval outcome data is absent "
            "under AWE_ARGUS_ROOT or the difficulty-label JSON shares no "
            "instance_ids with this corpus."
        )
    print(f"cases={len(rows)}  (pooled failure rate {100 * sum(r['failed'] for r in rows) / len(rows):.1f}%)")
    anchors = ("solved_unrelated", "reverted_own_change", "excessive_extra_edits")
    print(f"\n{'stratum':18s} {'n':>5s} {'fail%':>6s}  " + "  ".join(f"{a} RR(n)".rjust(28) for a in anchors))
    for name, labels in (("POOLED", None),) + STRATA:
        sub = rows if labels is None else [r for r in rows if r["difficulty"] in labels]
        if not sub:
            continue
        frate = 100 * sum(r["failed"] for r in sub) / len(sub)
        cells = []
        for a in anchors:
            r_, n = rr(sub, a)
            cells.append(
                (f"{r_:.2f}" if r_ not in (None, float("inf")) else ("inf" if r_ == float("inf") else "NA")) + f" ({n})"
            )
        print(f"{name:18s} {len(sub):5d} {frate:6.1f}  " + "  ".join(c.rjust(28) for c in cells))


if __name__ == "__main__":
    main()
