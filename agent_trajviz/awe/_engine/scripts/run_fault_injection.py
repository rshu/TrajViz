#!/usr/bin/env python3
"""Execute bundled fixture evaluators and record observed F2P/P2P outcomes."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_trajviz.awe._engine.fault_injection import run_pilot


def positive_timeout(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("all", "python", "javascript"), default="all")
    parser.add_argument(
        "--timeout", type=positive_timeout, default=10, help="seconds per evaluator process (default: 10)"
    )
    parser.add_argument("--output", type=Path, help="save complete JSON record at this path")
    args = parser.parse_args()
    languages = ("python", "javascript") if args.language == "all" else (args.language,)
    report = run_pilot(languages, timeout=args.timeout)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))
        print("Report: " + str(args.output.resolve()))
    else:
        print(text, end="")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
