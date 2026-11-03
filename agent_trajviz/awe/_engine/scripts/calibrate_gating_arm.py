#!/usr/bin/env python3
"""Corpus replay for the v0.35 gating-run signal (DESIGN.md §0ak).

Measures, over the full corpus, the shapes that decide whether the reserved
``environment`` fault_side arm can be un-reserved on the tight per-gating-run
outcome (``gating_run_observed`` / ``last_gating_run_failed``, schema 14):

  - loose §0ai shape (rejected 2026-08-13): failed & fail_to_pass_referenced &
    last test of ANY kind green & success claimed & untruncated
  - tight shape: failed & last GATING-named run observed green & untruncated
    (with the success-claim conjunction)
  - the same shapes on SOLVED runs (direction check: a signature more common
    on solves is not failure evidence)
  - gating-specificity of the shipped test_result_misinterpreted firings
  - overlap with the BA-HARNESS-1 suspect queue

Reproduce:  python3 scripts/calibrate_gating_arm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.trajsignals import signals_cached
from agent_trajviz.awe._engine.diffutil import gold_added_coverage
from agent_trajviz.awe._engine.faultside import is_ba_harness_suspect


def collect():
    rows = []
    for agent, inst, res, gold in iter_cases():
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        cov = gold_added_coverage(gold, load_agent_patch(agent, inst))
        rows.append(
            {
                "agent": agent,
                "inst": inst,
                "resolved": res,
                "ftp_ref": sig.fail_to_pass_referenced,
                "ran": sig.ran_any_test,
                "ltf": sig.last_test_failed,
                "gro": sig.gating_run_observed,
                "lgrf": sig.last_gating_run_failed,
                "claim": sig.claimed_success_final,
                "trunc": sig.truncated_run,
                "ba": is_ba_harness_suspect(sig.ran_any_test, sig.last_test_failed, cov, sig.truncated_run),
            }
        )
    return rows


def main():
    rows = collect()
    failed = [r for r in rows if r["resolved"] is False]
    solved = [r for r in rows if r["resolved"] is True]
    n = len(failed)

    def c(rs, pred):
        return sum(1 for r in rs if pred(r))

    loose = lambda r: r["ftp_ref"] and r["ran"] and not r["ltf"] and r["claim"] and not r["trunc"]
    tight = lambda r: r["gro"] and not r["lgrf"] and not r["trunc"]
    tight_claim = lambda r: tight(r) and r["claim"]

    print(f"total {len(rows)}  failed {n}  solved {len(solved)}")
    print(f"\nFAILED (n={n})")
    print(f"  loose §0ai shape:            {c(failed, loose)}")
    print(f"  gating_run_observed:         {c(failed, lambda r: r['gro'])}")
    print(f"    last gating run FAILED:    {c(failed, lambda r: r['gro'] and r['lgrf'])}")
    print(f"  tight (gating green):        {c(failed, tight)}")
    print(f"  tight + success claim:       {c(failed, tight_claim)}")
    print(f"  BA-HARNESS-1 suspects:       {c(failed, lambda r: r['ba'])}")
    print(f"    with gating green:         {c(failed, lambda r: r['ba'] and tight(r))}")
    print(f"\nSOLVED (n={len(solved)})")
    print(f"  tight + success claim:       {c(solved, tight_claim)}")
    mis = [r for r in failed if r["claim"] and r["ftp_ref"] and r["ltf"]]
    print(f"\ntest_result_misinterpreted shape (n={len(mis)})")
    print(f"  last gating run failed too:  {c(mis, lambda r: r['gro'] and r['lgrf'])}")
    print(f"  last gating run green:       {c(mis, lambda r: r['gro'] and not r['lgrf'])}")
    print(f"  gating outcome unobserved:   {c(mis, lambda r: not r['gro'])}")


if __name__ == "__main__":
    main()
