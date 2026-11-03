#!/usr/bin/env python3
"""Check committed publication integrity offline; no benchmark inputs or API key."""

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_trajviz.awe._engine.remedies import select_remedies
from agent_trajviz.awe._engine.remedy_plan import (
    build_plan,
    to_markdown as plan_markdown,
    insights_section,
    showcase_summary,
)
from agent_trajviz.awe._engine.attribution_graph import build_graphs, render_html


def check(root=ROOT):
    manifest = json.loads((root / "data/publication.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        path = root / name
        assert path.is_file(), f"Missing publication artifact: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Publication hash drift: {name}"
    cases = json.loads((root / "data/dossier/faults.json").read_text(encoding="utf-8"))
    insights = json.loads((root / "data/dossier/insights.json").read_text(encoding="utf-8"))
    symptoms = json.loads((root / "data/dossier/symptoms.json").read_text(encoding="utf-8"))
    stats = manifest["results"]
    assert len({c["case_id"] for c in cases}) == len(cases), "Duplicate dossier cases"
    assert len(cases) == stats["failed"], "Failure cohort drift"
    graph_path = root / "data/dossier/attribution_graphs.json"
    if graph_path.exists():
        graphs = json.loads(graph_path.read_text(encoding="utf-8"))
        scope = graphs["scope"]
        assert graphs == build_graphs(
            cases, agents=scope["agents"], instances=scope["instance_ids"], benchmark=scope["benchmark"]
        ), "Attribution graph drift"
        assert len(graphs["graphs"]) == len(cases)
        assert sorted(insights) == scope["agents"], "Attribution agent scope drift"
        assert len(scope["instance_ids"]) == manifest["cohort"]["input_tasks"], "Attribution task scope drift"
        assert (root / "data/dossier/ATTRIBUTION.html").read_text(encoding="utf-8") == render_html(graphs), "Attribution HTML drift"
    plan_path = root / "data/dossier/recommendations.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        scope = plan["scope"]
        assert plan == build_plan(
            cases, agents=scope["agents"], instances=scope["instance_ids"], benchmark=scope["benchmark"]
        ), "Recommendation plan drift"
        assert sorted(insights) == scope["agents"], "Recommendation agent scope drift"
        assert len(scope["instance_ids"]) == manifest["cohort"]["input_tasks"], "Recommendation task scope drift"
        assert (root / "data/dossier/RECOMMENDATIONS.md").read_text(encoding="utf-8") == plan_markdown(plan)
        assert "\n".join(insights_section(plan)) in (root / "data/dossier/INSIGHTS.md").read_text(encoding="utf-8")
    rsi_path = root / "data/dossier/rsi_proposals.json"
    if rsi_path.exists():
        from agent_trajviz.awe._engine.rsi import build_exchange

        assert (
            json.loads(rsi_path.read_text(encoding="utf-8")),
            json.loads((rsi_path.parent / "rsi_feedback.json").read_text(encoding="utf-8")),
        ) == build_exchange(cases, plan), "RSI exchange drift"
    assert dict(Counter(c["blame_status"] for c in cases)) == stats["attribution_statuses"]
    faults = [f for c in cases for f in c["faults"]]
    assert len(faults) == stats["fault_ledger_entries"]
    assert dict(Counter(f["error_type"] for f in faults)) == stats["fault_types"]
    for case in cases:
        assert sum(bool(f.get("is_primary")) for f in case["faults"]) == int(case["blame_status"] == "primary")
        for fault in case["faults"]:
            expected = select_remedies(case, fault)
            for field in ("recommendation_support", "recommendation", "countermeasures", "countermeasure_decisions"):
                assert fault[field] == expected[field], f"Remedy selection drift: {case['case_id']}/{field}"
    primary = [f for f in faults if f.get("is_primary")]
    assert dict(Counter(f["capability"] for f in primary)) == stats["primary_capabilities"]
    assert dict(Counter(f["error_type"] for f in primary)) == stats["primary_fault_types"]
    assert sum(d["n"] for d in insights.values()) == stats["scored_runs"]
    assert symptoms["summary"]["cases"] == manifest["cohort"]["input_slots"]
    assert math.isclose(sum(sum(d["cap_mass"].values()) for d in insights.values()), len(cases), abs_tol=1e-9), (
        "Full-precision blame mass drift"
    )
    for agent, d in insights.items():
        selected = [c for c in cases if c["agent"] == agent]
        assert len(selected) == d["n_failed"], f"{agent}: digest cohort drift"
        assert dict(Counter(c["blame_status"] for c in selected)) == d["attribution_statuses"]
        assert (
            dict(Counter(f["error_type"] for c in selected for f in c["faults"] if f.get("is_primary"))) == d["primary"]
        )
    with (root / "data/dossier/faults.csv").open(newline="", encoding="utf-8") as fp:
        assert sum(1 for _ in csv.DictReader(fp)) == len(faults), "Fault CSV drift"
    catalog = json.loads((root / "configs/countermeasures.json").read_text(encoding="utf-8"))
    counts = Counter(f["error_type"] for f in faults)
    for entry in catalog["entries"]:
        assert entry["empirical_support"] == counts[entry["root_cause"]["error_type"]]
    showcase = json.loads((root / "data/showcase_data.json").read_text(encoding="utf-8"))
    from agent_trajviz.awe._engine.showcase import countermeasure_catalog_payload
    from agent_trajviz.awe._engine.validate import countermeasure_catalog

    current_catalog = countermeasure_catalog()
    assert current_catalog["passed"] == current_catalog["n"], "Catalog contract failure"
    assert showcase["countermeasures"] == countermeasure_catalog_payload(), "Showcase catalog drift"
    assert manifest["validation"]["catalog"] == f"{current_catalog['passed']}/{current_catalog['n']}", (
        "Catalog validation metadata drift"
    )
    if plan_path.exists():
        assert showcase.get("recommendation_plan") == showcase_summary(plan), "Showcase recommendation drift"
    assert showcase["primary_capabilities"] == stats["primary_capabilities"]
    assert showcase["attribution_statuses"] == stats["attribution_statuses"]
    assert showcase["dataset"]["n_scored_runs"] == stats["scored_runs"]
    assert showcase["run_provenance"]["judge_model"] == manifest["judging"]["model"]
    print(
        f"Publication consistent: {stats['scored_runs']} scored runs, "
        f"{len(cases)} failures, {len(faults)} fault ledger entries; "
        f"{len(manifest['artifact_sha256'])} artifact hashes verified."
    )


if __name__ == "__main__":
    check()
