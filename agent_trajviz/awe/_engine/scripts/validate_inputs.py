#!/usr/bin/env python3
"""Read-only benchmark preflight; JSON on stdout, exit 1 on invalid input."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine.preflight import validate_inputs


def main():
    agents, instances = selected_inputs(pipeline_args())
    report = validate_inputs(agents, instances)
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    sys.exit(main())
