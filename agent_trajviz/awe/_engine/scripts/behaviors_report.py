#!/usr/bin/env python3
"""Discovered-mode behaviors report — objective prevalence of the scout-surfaced modes.

Per agent: how often `analysis_paralysis` / `environment_blocked` fire, and the failure
rate GIVEN the mode fired (the objective check that the scout-discovered mode tracks
failure). The cohort-contrast report (scripts/cohort_report.py) tests significance.

Run:  python3 scripts/behaviors_report.py [--write]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.behaviors import behaviors_summary, behaviors_report_md

OUT = Path(__file__).resolve().parent.parent / "data" / "discovery"


def main():
    instances = config.list_instances()
    summaries = [behaviors_summary(a, instances) for a in config.AGENTS]
    md = behaviors_report_md(summaries)
    print(md)
    if "--write" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "discovered_behaviors.md").write_text(md, encoding="utf-8")
        print(f"wrote {OUT / 'discovered_behaviors.md'}")


if __name__ == "__main__":
    main()
