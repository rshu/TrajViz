"""Incomplete sidecar pairs count as unknown wherever chart data omit them."""

from __future__ import annotations

import copy
import json

from jsonschema import validate

import agent_trajviz as tv
from agent_trajviz.figures import figure
from agent_trajviz.insight.labels import aggregate_labels


def test_missing_phase_or_action_summary_matches_charts_and_export(tmp_path):
    records = [
        {"index": 0, "role": "user", "phase": "user", "action": "user_prompt"},
        {"index": 1, "phase": "understand", "action": "code_reading", "duration_s": 2},
        {"index": 2, "phase": "understand"},
        {"index": 3, "action": "code_reading"},
        {"index": 4},
        {"index": 5, "phase": "unknown", "action": "unknown", "label_source": "fallback"},
    ]
    for record in records:
        record.setdefault("role", "assistant")
    data = {
        "schema_version": "trajectory_labels.v2",
        "taxonomy_version": "v1",
        "model": "fixture-model",
        "steps": records,
    }
    before = copy.deepcopy(data)
    agg = aggregate_labels(data)
    assert agg["total"] == 5 and agg["classified"] == 1 and agg["unknown"] == 4
    assert agg["classification_rate"] == 20.0
    assert agg["phase_counts"] == {"understand": 1}
    assert agg["action_counts"] == {"code_reading": 1}
    assert [step["index"] for step in agg["steps"]] == [0, 1]
    assert data == before
    sidecar = tmp_path / "labels.json"
    sidecar.write_text(json.dumps(data))
    original = sidecar.read_bytes()
    views = tv.label_views(sidecar)
    schemas = tv.view_schemas()
    for name, view in views.items():
        validate(view.to_dict(), schemas[name])
    summary = views["labels_summary"].data
    assert summary["classified"] == sum(views["label_phase_counts"].data["counts"].values()) == 1
    assert summary["unknown"] == 4
    assert [step["index"] for step in views["label_timeline"].data["steps"]] == [0, 1]
    assert sum(figure(views["label_phase_counts"]).data[0].y) == 1
    assert sum(figure(views["label_action_counts"]).data[0].x) == 1
    output = tmp_path / "exported"
    tv.export_json(views, output)
    exported = json.loads((output / "labels_summary.json").read_text())
    assert exported["data"] == summary
    assert sidecar.read_bytes() == original


def test_custom_complete_taxonomy_is_not_reclassified_or_expanded():
    data = {"steps": [{"index": 7, "phase": "custom_phase", "action": "custom_action", "duration_s": 1}]}
    result = aggregate_labels(data)
    assert result["classified"] == 1 and result["unknown"] == 0
    assert result["phase_counts"] == {"custom_phase": 1}
    assert result["action_counts"] == {"custom_action": 1}
    assert [row["index"] for row in result["steps"]] == [7]
