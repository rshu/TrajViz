#!/usr/bin/env python3
"""Generate INSIGHTS.md, insights.json, and the failure-mode composition CSV."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.digest import build_digest, build_insights_md, discovery_summary, failure_mode_rows
from agent_trajviz.awe._engine.remedy_plan import build_plan
from agent_trajviz.awe._engine.attribution_graph import build_graphs

OUT = config.OUTPUT_DATA / "dossier"


def main() -> None:
    args = pipeline_args()
    config.AGENTS, instances = selected_inputs(args)
    workers = args.workers
    D = build_digest(instances, workers=workers)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        discovery = discovery_summary(instances)
    except Exception as e:  # discovery is additive; never block INSIGHTS
        print(f"warning: discovery signals skipped ({e})", file=sys.stderr)
        discovery = None
    recommendations = None
    plan_path, faults_path = OUT / "recommendations.json", OUT / "faults.json"
    if plan_path.exists() and faults_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        # A standalone filtered digest must not borrow advice from an older or
        # differently scoped dossier. Pipeline generation produces an exact match.
        try:
            cases = json.loads(faults_path.read_text(encoding="utf-8"))
            expected = build_plan(cases, agents=config.AGENTS, instances=instances)
            if plan != expected or len(cases) != sum(d["n_failed"] for d in D.values()):
                raise ValueError("stale or differently scoped recommendation plan")
            recommendations = plan
        except ValueError as error:
            print(f"warning: recommendation section skipped ({error})", file=sys.stderr)
    graphs = None
    graph_path = OUT / "attribution_graphs.json"
    if graph_path.exists() and faults_path.exists():
        try:
            cases = json.loads(faults_path.read_text(encoding="utf-8"))
            recorded = json.loads(graph_path.read_text(encoding="utf-8"))
            expected = build_graphs(cases, agents=config.AGENTS, instances=instances)
            if recorded != expected or len(cases) != sum(d["n_failed"] for d in D.values()):
                raise ValueError("stale or differently scoped attribution graphs")
            graphs = recorded
        except ValueError as error:
            print(f"warning: attribution link skipped ({error})", file=sys.stderr)
    (OUT / "INSIGHTS.md").write_text(build_insights_md(D, discovery, recommendations, graphs), encoding="utf-8")
    ser = {
        a: {
            **{k: v for k, v in d.items() if k not in ("primary", "allf", "examples")},
            "primary": dict(d["primary"]),
            "allf": dict(d["allf"]),
            "examples": {k: list(v) for k, v in d["examples"].items()},
        }
        for a, d in D.items()
    }
    (OUT / "insights.json").write_text(json.dumps(ser, indent=2), encoding="utf-8")
    mode_csv = OUT / "failure_composition_by_mode.csv"
    mode_rows = failure_mode_rows(D)
    mode_fields = (
        "phase",
        "phase_label",
        "root_cause_mode",
        "agent",
        "fault_entry_count",
        "attributed_blame_mass",
        "phase_blame_mass",
        "failed_tasks",
        "share_within_phase_pct",
        "share_of_agent_failures_pct",
    )
    with mode_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=mode_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(mode_rows)
    print(f"wrote {OUT / 'INSIGHTS.md'}")
    print(f"wrote {OUT / 'insights.json'}")
    print(f"wrote {mode_csv}")


if __name__ == "__main__":
    main()
