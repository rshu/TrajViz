#!/usr/bin/env python3
"""Export RSI proposals and privileged diagnostic feedback without running agents."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.rsi import write_exchange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=config.OUTPUT_DATA / "dossier")
    args = parser.parse_args()
    cases = json.loads((args.dossier / "faults.json").read_text(encoding="utf-8"))
    plan = json.loads((args.dossier / "recommendations.json").read_text(encoding="utf-8"))
    write_exchange(cases, plan, args.dossier)
    print(f"Wrote RSI exchange in {args.dossier}; no agent runs or installations")


if __name__ == "__main__":
    main()
