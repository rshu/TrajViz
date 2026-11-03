#!/usr/bin/env python3
"""Batch 8 cohort calibration (v0.36, DESIGN.md §0al).

Solved-vs-failed risk ratios for the batch-8 fault candidates (unpaired
bootstrap p, Holm over the candidate family — the §0y discipline):

  delegated                     — run contains >=1 main-thread DELEGATE call
  delegate_error_result         — a DELEGATE result carries the scaffold error flag
  delegate_no_result            — a DELEGATE call has no result (truncation-guarded)
  authored_test_ran_never_failed— weak-oracle repro: authored test ran, never failed
  environment_error             — env/dependency error in a command/test result
                                  (candidate #2 of the Raj extraction)
  gating_exec_failed_seen       — a gating test FUNCTION was named failing in output

Also reports (not calibrated — impact accounting):
  - the corpus impact of the v0.36 suite-wide guard on the medium
    relevant_test_not_run arm (how many failed-case firings it suppresses)
  - the affirmative environment-divergence shape one tier up from §0ak:
    gating function observed in OUTPUT, last such observation green, eval failed

Reproduce:  python3 scripts/calibrate_batch8.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.corpusiter import iter_cases
from agent_trajviz.awe._engine.significance import holm
from agent_trajviz.awe._engine.trajsignals import signals_cached

CANDS = (
    "delegated",
    "delegate_error_result",
    "delegate_no_result",
    "authored_test_ran_never_failed",
    "environment_error",
    "gating_exec_failed_seen",
)


def rows_of():
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
                "delegated": int(sig.n_delegations > 0),
                "delegate_error_result": int(sig.delegate_error_result),
                "delegate_no_result": int(sig.delegate_no_result),
                "authored_test_ran_never_failed": int(sig.authored_test_ran_never_failed),
                "environment_error": int(sig.environment_error),
                "gating_exec_failed_seen": int(sig.gating_exec_observed and sig.last_gating_exec_failed),
                # impact / shape accounting
                "suite_guard_hit": int(
                    sig.ran_any_test
                    and gold.gating_tests_visible
                    and not sig.fail_to_pass_referenced
                    and sig.ran_suite_wide
                    and bool(gold.opportunities.get("code_verification"))
                ),
                "env_shape_exec": int(
                    sig.gating_exec_observed and not sig.last_gating_exec_failed and not sig.truncated_run
                ),
            }
        )
    return rows


def main() -> None:
    rows = rows_of()
    failed = [r for r in rows if r["failed"]]
    solved = [r for r in rows if not r["failed"]]
    print(f"cases={len(rows)} failed={len(failed)} solved={len(solved)}")

    rng = random.Random(29)
    stats, pvals = {}, []
    for c in CANDS:
        fv = [float(r[c]) for r in failed]
        sv = [float(r[c]) for r in solved]
        assoc, ct = binary_association(fv, sv), contrast(fv, sv, rng)
        stats[c] = (assoc, ct)
        pvals.append(ct["p"])
    print(f"\n{'candidate':32s} {'n_present':>9s} {'fail%':>6s} {'solv%':>6s} {'RR':>5s} {'p':>7s} {'p_holm':>7s}")
    for c, p_h in zip(CANDS, holm(pvals), strict=False):
        a, ct = stats[c]
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(
            f"{c:32s} {a['n_present']:9d} {100 * ct['mean_failed']:6.1f} "
            f"{100 * ct['mean_solved']:6.1f} {rr_s:>5s} {ct['p']:7.4f} {p_h:7.4f}"
        )

    # delegation candidates only exist where the scaffold delegates
    for agent in config.AGENTS:
        af = [r for r in rows if r["agent"] == agent and r["delegated"]]
        if af:
            afail = [r for r in af if r["failed"]]
            print(
                f"\n{agent}: delegated runs {len(af)} "
                f"(failed {len(afail)}); delegate_error_result "
                f"{sum(r['delegate_error_result'] for r in af)}, "
                f"delegate_no_result {sum(r['delegate_no_result'] for r in af)}"
            )

    print("\n--- impact accounting (failed cases, n=%d) ---" % len(failed))
    print("suite-wide guard suppressions of medium relevant_test_not_run:", sum(r["suite_guard_hit"] for r in failed))
    print(
        "exec-tier env-divergence shape (gating fn seen green in output):",
        sum(r["env_shape_exec"] for r in failed),
        "failed /",
        sum(r["env_shape_exec"] for r in solved),
        "solved",
    )


if __name__ == "__main__":
    main()
