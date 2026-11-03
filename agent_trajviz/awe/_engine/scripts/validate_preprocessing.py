#!/usr/bin/env python3
"""Offline preprocessing/telemetry controls and optional read-only native replay."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.adapters import _parse_normalized
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.preprocessing_validation import controlled_validation, describe, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent")
    parser.add_argument("--instance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.agent) != bool(args.instance):
        parser.error("--agent and --instance must be supplied together")
    report = controlled_validation()
    if args.agent:
        config.safe_id(args.agent)
        config.safe_id(args.instance)
        traj = _parse_normalized(args.agent, args.instance)
        if traj is None:
            parser.error("trajectory is missing")
        gold = build_gold_reference(args.instance)
        report["native_replay"] = {
            mode: describe(project(traj, mode), gold.problem_statement)
            for mode in ("canonical", "legacy_lossy_proxy", "drop_results", "drop_result_text", "drop_state")
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"checks: {sum(report['checks'].values())}/{len(report['checks'])}; report: {args.output}")
    else:
        print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
