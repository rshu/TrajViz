#!/usr/bin/env python3
"""Calibration of the 2026-08 literature-sweep fault candidates (DESIGN §0ar;
sources in data/discovery/litsweep_2026-08.md). The §0y discipline: promotion
only on Holm-surviving solved-vs-failed discrimination.

  authored_test_assertion_free — an agent-edited test file whose ADDED lines
      contain no assertion signal (All Smoke No Alarm, arXiv:2606.18168)
  claimed_change_not_in_patch  — a closing message names a .py file absent
      from the final diff (How Coding Agents Fail, arXiv:2605.29442)
  claim_stale_evidence         — success claim while no test follows the last
      source edit (Proof-or-Stop, arXiv:2607.14890) — conjunction of shipped
      signals
  recovered_detour             — non-gold source files edited during the run
      but absent from the final diff (TrajDebug lifecycle, arXiv:2608.06346;
      NB: the unrestricted form, traj_edit_not_landed, was rejected in
      batch 7 at RR 1.17)

Reproduce:  python3 scripts/calibrate_litsweep_candidates.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.diffutil import is_test_file
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.significance import holm
from agent_trajviz.awe._engine.trajsignals import signals_cached

CANDS = ("authored_test_assertion_free", "claimed_change_not_in_patch", "claim_stale_evidence", "recovered_detour")

# Assertion signals in added test lines (python-centric, matching the corpus):
# bare assert, unittest self.assert*/fail*, pytest.raises / warns, numpy
# assert_* helpers.
_ASSERT_RE = re.compile(
    r"\bassert\b|\.assert\w+\(|\.fail\w*\(|pytest\.(?:raises|warns)|"
    r"\bassert_\w+\(|\braises\("
)
_PYPATH_RE = re.compile(r"\b([\w./-]+\.py)\b")


def _assertion_free_authored_test(patch) -> bool:
    """>=1 agent-edited test file whose added lines carry NO assertion signal
    (files with zero added lines don't count — a pure deletion is not an
    authored test)."""
    if not (patch and patch.exists):
        return False
    for f in patch.edited_test_files:
        added = (patch.changed_lines.get(f) or {}).get("added") or {}
        lines = list(added)
        if lines and not any(_ASSERT_RE.search(l) for l in lines):
            return True
    return False


def _claimed_missing_file(agent, inst, patch) -> bool:
    """A .py path named in the run's CLOSING main-thread messages whose
    basename appears nowhere in the final diff."""
    traj = load_normalized(agent, inst)
    if traj is None:
        return False
    texts = [s for s in traj.assistant_texts() if s.subagent_id is None]
    tail = " ".join(s.text for s in texts[-3:])
    mentioned = {m.rsplit("/", 1)[-1] for m in _PYPATH_RE.findall(tail)}
    if not mentioned:
        return False
    edited = {f.rsplit("/", 1)[-1] for f in (patch.edited_files if patch and patch.exists else [])}
    return bool(mentioned - edited)


def rows_of():
    rows = []
    for agent, inst, res, gold in iter_cases():
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        patch = load_agent_patch(agent, inst)
        gold_src = set(gold.gold_files)
        final_files = set(patch.edited_files) if patch and patch.exists else set()
        detour = any(
            f not in final_files and f not in gold_src and not is_test_file(f) for f in (sig.edit_set_traj or []) if f
        )
        rows.append(
            {
                "agent": agent,
                "inst": inst,
                "failed": 0 if res else 1,
                "authored_test_assertion_free": int(_assertion_free_authored_test(patch)),
                "claimed_change_not_in_patch": int(_claimed_missing_file(agent, inst, patch)),
                "claim_stale_evidence": int(sig.claimed_success_final and sig.no_test_after_final_edit),
                "recovered_detour": int(detour),
            }
        )
    return rows


def main() -> None:
    rows = rows_of()
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"cases={len(rows)} failed={len(failed)} solved={len(solved)}")

    rng = random.Random(29)
    stats, pvals = [], []
    for c in CANDS:
        fv = [float(r[c]) for r in failed]
        sv = [float(r[c]) for r in solved]
        stats.append((c, binary_association(fv, sv), contrast(fv, sv, rng)))
        pvals.append(stats[-1][2]["p"])
    print(f"\n{'candidate':30s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p':>7s} {'p_holm':>7s}")
    for (c, a, ct), p_h in zip(stats, holm(pvals), strict=False):
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{c:30s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {ct['p']:7.4f} {p_h:7.4f}"
        )


if __name__ == "__main__":
    main()
