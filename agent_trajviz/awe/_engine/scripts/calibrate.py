#!/usr/bin/env python3
"""Calibrate the over_editing threshold + report blame-weight sensitivity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.calibrate import suggest_over_edit_threshold, blame_sensitivity, attribution_sensitivity


def main() -> None:
    print("\n=== over_editing threshold calibration (edit_size_ratio) ===")
    d = suggest_over_edit_threshold()
    print(f"  n_solved={d['n_solved']}  n_failed={d['n_failed']}")
    print(f"  solved edit_size_ratio pctiles: {d['solved_pctiles']}")
    print(f"  failed edit_size_ratio pctiles: {d['failed_pctiles']}")
    print(f"\n  {'threshold':>10}{'%solved fire':>14}{'%failed fire':>14}")
    for thr, (sf, ff) in sorted(d["fire_rate_solved_failed"].items()):
        solved_rate = f"{100 * sf:.0f}%" if sf is not None else "NA"
        failed_rate = f"{100 * ff:.0f}%" if ff is not None else "NA"
        print(f"  {thr:>10}{solved_rate:>14}{failed_rate:>14}")
    print("  (good threshold: low %solved firing, while still catching genuinely bloated patches)")

    print("\n=== blame role-multiplier sensitivity (capability scores, no-traj) ===")
    sens = blame_sensitivity()
    caps = ["code_localization", "code_editing"]
    short = {"code_localization": "Loc", "code_editing": "Edit"}
    head = f"  {'setting':<22}" + "".join(f"{a.replace('opencode_', 'oc_'):>16}" for a in config.AGENTS)
    print(head)
    for name, per_agent in sens.items():
        row = f"  {name:<22}"
        for a in config.AGENTS:
            sc = per_agent[a]
            row += f"{'/'.join(f'{short[c]}{sc[c]:.0f}' for c in caps):>16}"
        print(row)
    print("\n  Stability check: does Editing stay the bottleneck (lower than Loc) across all rows?")

    print("\n=== attribution sensitivity (primaries/conjunctive, FULL trajectory pipeline, no arbiter) ===")
    att = attribution_sensitivity()
    for name, row in att.items():
        prim = ", ".join(
            f"{c.replace('code_', '')}={n}" for c, n in sorted(row["primaries"].items(), key=lambda kv: -kv[1])
        )
        print(f"  {name:<16} primaries: {prim}   conjunctive: {row['conjunctive']}")
    print(
        "  Stability check: with causal-only dominance the counts should be "
        "near-invariant to the verification multiplier."
    )


if __name__ == "__main__":
    main()
