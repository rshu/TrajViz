"""Label summaries distinguish original records from filtered chart/timeline rows."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from agent_trajviz.insight.labels import aggregate_labels
from agent_trajviz.insight.presenters.label_ui import build_label_ui_payload


def _record(index, role="assistant", *, unknown=False):
    return {
        "index": index,
        "role": role,
        "phase": "unknown" if unknown else "user" if role == "user" else "implement",
        "action": "unknown" if unknown else "user_prompt" if role == "user" else "implement_tests",
        "label_source": "fallback" if unknown else "default" if role == "user" else "llm",
        "duration_s": 1.0,
    }


def _write(tmp_path, steps):
    value = {
        "schema_version": "trajectory_labels.v2",
        "taxonomy_version": "v1",
        "model": "fixture-model",
        "steps": steps,
    }
    path = tmp_path / "session_labeled.json"
    path.write_text(json.dumps(value))
    return path, value


def test_run1_sized_coverage_explains_unknown_user_defaults_and_assistant_charts(tmp_path):
    # Synthetic counts mirror the audited204-record run without storing its data.
    steps = [_record(index, "user" if index in (0, 101) else "assistant", unknown=index == 181) for index in range(204)]
    phases = ("understand", "implement", "validate", "debug", "report")
    for record in steps:
        if record["role"] == "assistant" and record["phase"] != "unknown":
            record["phase"] = phases[record["index"] % len(phases)]
    path, value = _write(tmp_path, steps)
    before = path.read_bytes()
    run = SimpleNamespace(steps=[{"index": r["index"], "role": r["role"]} for r in steps])
    output = build_label_ui_payload(str(path), run=run)
    badge = output["badge_html"]
    assert "Labels loaded — 203/204 steps labeled; 1 unknown" in badge
    assert "assistant: 201/202 classified" in badge
    assert "user records: 2 (2 default labels)" in badge
    assert "201 classified assistant steps across 5 phases" in badge
    assert "Timeline: 203 steps, including 2 user prompts" in badge
    assert "Unknown assistant labels remain in the sidecar" in badge
    assert "fixture-model" in output["status_html"]
    assert path.read_bytes() == before
    agg = aggregate_labels(copy.deepcopy(value))
    assert agg["unknown"] == 1 and sum(agg["phase_counts"].values()) == 201
    assert len(agg["steps"]) == 203 and 181 not in [row["index"] for row in agg["steps"]]


def test_run2_sized_complete_coverage_does_not_count_users_as_assistant_classifications(tmp_path):
    steps = [_record(index, "user" if index in (0, 51) else "assistant") for index in range(104)]
    path, _ = _write(tmp_path, steps)
    badge = build_label_ui_payload(str(path))["badge_html"]
    assert "104/104 steps labeled; 0 unknown" in badge
    assert "assistant: 102/102 classified" in badge
    assert "Phase/action charts: 102 classified assistant steps" in badge
    assert "Timeline: 104 steps, including 2 user prompts" in badge


def test_sparse_sidecar_separates_missing_records_from_unknown_labels(tmp_path):
    path, _ = _write(tmp_path, [_record(1), _record(3, unknown=True)])
    run = SimpleNamespace(steps=[{"index": index, "role": "assistant"} for index in range(5)])
    badge = build_label_ui_payload(str(path), run=run)["badge_html"]
    assert "1/5 steps labeled; 1 unknown; 3 without label records" in badge
    assert "assistant: 1/2 classified" in badge
    assert "Timeline: 1 steps" in badge


def test_other_role_unknown_is_not_counted_as_an_assistant_failure(tmp_path):
    path, _ = _write(tmp_path, [_record(0, "user"), _record(1), _record(2, "system", unknown=True)])
    badge = build_label_ui_payload(str(path))["badge_html"]
    assert "2/3 steps labeled; 1 unknown" in badge
    assert "assistant: 1/1 classified" in badge and "other roles: 1" in badge


def test_legacy_roleless_records_and_empty_sidecars_keep_honest_denominators(tmp_path):
    record = _record(0)
    record.pop("role")
    path, _ = _write(tmp_path, [record])
    badge = build_label_ui_payload(str(path))["badge_html"]
    assert "1/1 steps labeled; 0 unknown" in badge and "assistant: 1/1 classified" in badge
    path, _ = _write(tmp_path, [])
    assert "0/0 steps labeled; 0 unknown" in build_label_ui_payload(str(path))["badge_html"]
