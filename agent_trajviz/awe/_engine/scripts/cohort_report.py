#!/usr/bin/env python3
"""Cohort-contrast discovery report — process behaviors associated with failure.

Contrasts each agent's solved vs failed traces and lists behaviors whose prevalence
differs (Holm-corrected). Associational, not causal — a complement to the deductive
per-failure diagnosis, surfacing patterns the fixed taxonomy does not encode.

Run:  python3 scripts/cohort_report.py [--agent NAME] [--write]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.cohort import discover_agent, cohort_report_md

OUT = Path(__file__).resolve().parent.parent / "data" / "discovery"


def main():
    args = sys.argv[1:]
    agents = [args[args.index("--agent") + 1]] if "--agent" in args else config.AGENTS
    instances = config.list_instances()
    blocks = [
        "# Cohort-contrast discovery",
        "_Behaviors associated with failure (solved vs failed cohorts). "
        "Associational and significance-gated (Holm); not causal._",
        "",
    ]
    for a in agents:
        disc = discover_agent(a, instances)
        md = cohort_report_md(disc)
        print(md)
        blocks.append(md)
    if "--write" in args:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "cohort_contrast.md").write_text("\n".join(blocks), encoding="utf-8")
        print(f"wrote {OUT / 'cohort_contrast.md'}")


if __name__ == "__main__":
    main()
