"""Generate the BA-HARNESS-1 review queue: harness-rigidity suspects.

A suspect is a FAILED case where (a) the agent ran tests and its LAST test run
was green (its own verification accepted the patch), and (b) the final patch's
added-line content (normalized) covers >= --cov of the gold patch's added
source lines. These are candidate evaluation artifacts (spec-oracle gap /
strict format mismatch — METH-HARNESS-1, arXiv:2605.12270), queued for MANUAL
harness-diagnosis review; nothing is removed from fault counts.

Usage:  python3 scripts/harness_suspects.py [--cov 0.7] [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.diffutil import gold_added_coverage, is_test_file
from agent_trajviz.awe._engine.faultside import BA_HARNESS_COVERAGE, is_ba_harness_suspect
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.trajsignals import signals_cached

OUT = config.PROJECT_DATA / "discovery" / "harness_suspects.md"


def collect(cov_threshold: float) -> tuple[list[dict], int]:
    # ONE suspect predicate, shared with the in-record fault_side grader arm
    # (awe/faultside.py) — the queue and the dossier attribute are the same set
    # by construction. v0.33 adds the truncation gate to the shared predicate
    # (a dangling final test call means the "green" was never observed); no
    # shipped queue row is affected (both truncated runs sit in the solved
    # cohort).
    suspects, n_failed = [], 0
    for agent in config.AGENTS:
        for inst in config.list_instances():
            if resolved(agent, inst) is not False:
                continue
            n_failed += 1
            gold = build_gold_reference(inst)
            sig = signals_cached(agent, inst, gold)
            if not sig.ran_any_test or sig.last_test_failed or sig.truncated_run:
                continue  # cheap pre-gate: skip the patch load
            patch = load_agent_patch(agent, inst)
            cov = gold_added_coverage(gold, patch)
            if is_ba_harness_suspect(
                sig.ran_any_test, sig.last_test_failed, cov, truncated_run=sig.truncated_run, threshold=cov_threshold
            ):
                suspects.append(
                    {
                        "agent": agent,
                        "instance_id": inst,
                        "cov": cov,
                        "gold_files": sorted(f for f in gold.gold_changed_lines if not is_test_file(f)),
                    }
                )
    suspects.sort(key=lambda s: -s["cov"])
    return suspects, n_failed


def render(suspects: list[dict], n_failed: int, cov_threshold: float) -> str:
    L = [
        "# Harness-rigidity suspects (BA-HARNESS-1 review queue)",
        "",
        f"_Failed cases whose final patch covers >= {cov_threshold:.0%} of the gold",
        "patch's added source lines AND whose own last test run was green — the",
        "agent's verification accepted a near-gold patch that the harness rejected.",
        "Candidate evaluation artifacts (METH-HARNESS-1); each needs MANUAL",
        "harness-diagnosis review (inspect the official test_patch / FAIL_TO_PASS",
        "assertions) before any conclusion. Suspects stay in all fault counts._",
        "",
        f"**{len(suspects)} suspects / {n_failed} failed cases "
        f"({len(suspects) / n_failed:.1%})** — a conservative lower bound: "
        "content-level matching misses representation-only mismatches "
        "(e.g. str vs int) by construction.",
        "",
        "| Agent | Instance | Gold added-line coverage | Gold source files |",
        "|---|---|---|---|",
    ]
    for s in suspects:
        L.append(f"| {s['agent']} | {s['instance_id']} | {s['cov']:.2f} | {', '.join(s['gold_files'])} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cov", type=float, default=BA_HARNESS_COVERAGE, help="gold added-line coverage threshold (default 0.7)"
    )
    ap.add_argument("--write", action="store_true", help=f"write to {OUT} instead of stdout")
    args = ap.parse_args()
    suspects, n_failed = collect(args.cov)
    md = render(suspects, n_failed, args.cov)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(md, encoding="utf-8")
        print(f"wrote {OUT} ({len(suspects)} suspects / {n_failed} failed)")
    else:
        print(md)


if __name__ == "__main__":
    main()
