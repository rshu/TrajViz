#!/usr/bin/env python3
"""Generate the comprehensive fault dossier: JSON (complete) + Markdown (human,
with code + thinking) + CSV (sortable triage index).

  python scripts/gen_dossier.py [--agent NAME] [--limit N] [--max-md N] [--workers N]
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.dossier import collect, to_markdown, to_csv_rows, CSV_COLUMNS
from agent_trajviz.awe._engine.symptom_exports import collect as collect_symptoms, write as write_symptoms
from agent_trajviz.awe._engine.remedy_plan import build_plan, write as write_plan
from agent_trajviz.awe._engine.attribution_graph import build_graphs, write as write_graphs
from agent_trajviz.awe._engine.rsi import write_exchange

OUT = config.OUTPUT_DATA / "dossier"


def main() -> None:
    args = pipeline_args()
    agents, instances = selected_inputs(args)
    max_md, workers = args.max_md, args.workers

    cases = collect(agents, instances, workers=workers)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "faults.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    rows = to_csv_rows(cases)
    with open(OUT / "faults.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for agent in agents:
        ac = [c for c in cases if c["agent"] == agent]
        (OUT / f"{agent}.md").write_text(to_markdown(ac, max_cases=max_md), encoding="utf-8")

    plan = build_plan(cases, agents=agents, instances=instances)
    write_plan(plan, OUT)
    write_exchange(cases, plan, OUT)
    write_graphs(build_graphs(cases, agents=agents, instances=instances), OUT)

    summary = write_symptoms(collect_symptoms(agents, instances, workers=workers), OUT)
    print(f"symptom review cases={summary['cases']} (all outcomes; not scored)")
    n_faults = sum(len(c["faults"]) for c in cases)
    print(f"cases={len(cases)}  faults={n_faults}")
    print(f"  JSON : {OUT / 'faults.json'}")
    print(f"  CSV  : {OUT / 'faults.csv'}  ({len(rows)} rows)")
    print(f"  MD   : {OUT}/<agent>.md")
    print(f"  PLAN : {OUT / 'RECOMMENDATIONS.md'}")
    print(f"  RSI  : {OUT / 'rsi_proposals.json'} + rsi_feedback.json")
    print(f"  DAG  : {OUT / 'ATTRIBUTION.html'}")


if __name__ == "__main__":
    main()
