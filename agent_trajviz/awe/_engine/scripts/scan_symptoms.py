#!/usr/bin/env python3
"""Export unscored symptoms and verification observations for all outcomes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine.symptom_exports import collect, write


def main():
    args = pipeline_args()
    agents, instances = selected_inputs(args)
    summary = write(collect(agents, instances, workers=args.workers), config.OUTPUT_DATA / "dossier")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
