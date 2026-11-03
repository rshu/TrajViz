#!/usr/bin/env python3
"""Test-edit calibration (v0.39, DESIGN.md §0av).

Why: `gating_test_modified` (v0.22) predates the calibrate-first discipline
(§0y) and was never contrasted solved-vs-failed. This script measures its fact
and every sharper form of "the agent edited tests instead of code" over the
full corpus, using the SHIPPED signal for the promoted form
(``localization_signals()['assert_rewritten_files']``) so the calibration
always describes exactly what the detector consumes. Binary family, unpaired
bootstrap p, Holm-corrected.

Reproduce:  AWE_ARGUS_ROOT=... python3 scripts/calibrate_test_rewrite.py
            (add --no-traj to skip the trajectory-ordering features; the first
            run at a new SCHEMA_VERSION recomputes the trajsig cache)
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.diffutil import is_test_file, parse_unified_diff
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.significance import holm
from agent_trajviz.awe._engine import config

# Candidate forms that are NOT shipped signals (computed inline from the raw
# diff so their negatives are recorded next to the promoted form).
_SKIP_RE = re.compile(
    r"^\s*(?:@pytest\.mark\.(?:skip|xfail)|@unittest\.skip|pytest\.skip\(|"
    r"self\.skipTest\(|@skip)"
)
_TESTDEF_RE = re.compile(r"^\s*def test_\w+")


def rows_of(with_traj: bool):
    rows = []
    if with_traj:
        from agent_trajviz.awe._engine.trajsignals import signals_cached
    for agent, inst, res, gold in iter_cases():
        patch = load_agent_patch(agent, inst)
        if not patch.exists:
            continue
        loc = localization_signals(gold, patch, res)
        gold_t = set(gold.test_patch_files)
        # inline candidate forms over the raw diff (per path, sections summed)
        text = config.patch_path(agent, inst).read_text(errors="replace", encoding="utf-8")
        skip_any = skip_gold = def_del_any = def_del_gold = False
        for fd in parse_unified_diff(text):
            if not is_test_file(fd.path):
                continue
            if any(_SKIP_RE.match(l) for l in fd.added_lines):
                skip_any = True
                skip_gold |= fd.path in gold_t
            n_def_rem = sum(1 for l in fd.removed_lines if _TESTDEF_RE.match(l))
            n_def_add = sum(1 for l in fd.added_lines if _TESTDEF_RE.match(l))
            if n_def_rem > n_def_add:
                def_del_any = True
                def_del_gold |= fd.path in gold_t
        r = {
            "agent": agent,
            "inst": inst,
            "failed": 0 if res else 1,
            "any_test_edited": bool(patch.edited_test_files),
            "gold_test_edited (pre-v0.39 fact)": bool(loc["gold_tests_edited"]),
            "gold_test_modified_existing (old HIGH arm)": bool(loc["gold_tests_modified_existing"]),
            "assert_rewritten_files (SHIPPED)": bool(loc["assert_rewritten_files"]),
            "assert_rewritten_gold_tests": bool(loc["assert_rewritten_gold_tests"]),
            "assert_rewritten_other_tests": bool(loc["assert_rewritten_other_tests"]),
            "assert_net_removed_files": bool(loc["assert_net_removed_files"]),
            "skip/xfail marker added (any test)": skip_any,
            "skip/xfail marker added (gold test)": skip_gold,
            "test fn removed net (any test)": def_del_any,
            "test fn removed net (gold test)": def_del_gold,
        }
        if with_traj:
            t = signals_cached(agent, inst, gold)
            after = set(getattr(t, "test_files_edited_after_failure", []) or [])
            r["test file edited after observed failure"] = bool(after)
            r["rewrite AND after observed failure"] = bool(after & set(loc["assert_rewritten_files"]))
        rows.append(r)
    return rows


def main() -> None:
    with_traj = "--no-traj" not in sys.argv
    rows = rows_of(with_traj)
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"cases with a patch: {len(rows)}  failed {len(failed)}  solved {len(solved)}")
    if not rows:
        print("no patched cases found — is AWE_ARGUS_ROOT set?", file=sys.stderr)
        return
    feats = [k for k in rows[0] if k not in ("agent", "inst", "failed")]
    rng = random.Random(39)
    stats, pvals = [], []
    for name in feats:
        fv = [float(r[name]) for r in failed]
        sv = [float(r[name]) for r in solved]
        assoc, ct = binary_association(fv, sv), contrast(fv, sv, rng)
        stats.append((name, assoc, ct))
        pvals.append(ct["p"])
    print(f"\n{'feature':46s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p_holm':>7s}")
    for (name, a, ct), p_h in zip(stats, holm(pvals), strict=False):
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{name:46s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {p_h:7.4f}"
        )
    print(
        "\nReading: the SHIPPED signal (assert_rewritten_files) must clear the gate; the "
        "pre-v0.39 fact (gold_test_edited) is expected flat — that is why it was demoted."
    )


if __name__ == "__main__":
    main()
