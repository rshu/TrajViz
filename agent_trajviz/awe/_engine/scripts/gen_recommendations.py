#!/usr/bin/env python3
"""Refresh experiment plans from an existing dossier without judging or agent runs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.remedy_plan import build_plan, write
from agent_trajviz.awe._engine.rsi import write_exchange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=config.OUTPUT_DATA / "dossier")
    args = parser.parse_args()
    cases = json.loads((args.dossier / "faults.json").read_text(encoding="utf-8"))
    # Preserve previously recorded input scope if the current cases fit it.
    # Without run metadata an offline generator claims only diagnosed-case scope.
    previous = args.dossier / "recommendations.json"
    scope = json.loads(previous.read_text(encoding="utf-8"))["scope"] if previous.exists() else {}
    if any(
        c["agent"] not in scope.get("agents", []) or c["instance_id"] not in scope.get("instance_ids", [])
        for c in cases
    ):
        scope = {}
    plan = build_plan(
        cases, agents=scope.get("agents"), instances=scope.get("instance_ids"), benchmark=scope.get("benchmark")
    )
    write(plan, args.dossier)
    write_exchange(cases, plan, args.dossier)
    print(
        f"Wrote {args.dossier / 'RECOMMENDATIONS.md'}; {len(plan['candidates'])} candidate experiments, no agent runs."
    )


if __name__ == "__main__":
    main()
