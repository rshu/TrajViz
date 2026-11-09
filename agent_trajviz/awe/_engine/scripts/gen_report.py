#!/usr/bin/env python3
"""Generate matching Markdown and structured JSON diagnosis reports.

python scripts/gen_report.py [--agent NAME] [--limit N]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.report import agent_report_outputs

OUT = config.OUTPUT_DATA / "reports"


def main() -> None:
    args = pipeline_args()
    agents, instances = selected_inputs(args)

    OUT.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        md, scorecard = agent_report_outputs(agent, instances)
        path = OUT / f"{agent}.md"
        path.write_text(md, encoding="utf-8")
        print(f"wrote {path}")
        data_path = OUT / f"{agent}.json"
        data_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
