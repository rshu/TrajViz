"""Workflow navigation and visible, inert provenance for label overlays."""

from __future__ import annotations

import json

import pytest

from agent_trajviz.insight.presenters.label_ui import build_label_ui_payload
from agent_trajviz.insight.presenters.patterns import render_tool_sequences_html


def test_recurring_sequence_links_preserve_native_indices_and_ten_step_cap():
    output = render_tool_sequences_html(
        [{"sequence": ["Read", "Edit"], "frequency": 14, "step_indices": [0, *range(4, 17)]}]
    )
    assert output.count("class='insight-step-link'") == 10
    for index in [0, *range(4, 13)]:
        assert f"tvGotoWorkflowStep({index})" in output
        assert f">#{index}</span>" in output
    assert "tvGotoWorkflowStep(13)" not in output
    assert "(+4 more)" in output
    assert "Read → Edit" in output


def test_sequence_navigation_and_cells_cannot_execute_imported_html():
    payload = "<img src=x onerror=alert(1)>"
    output = render_tool_sequences_html([{"sequence": [payload], "frequency": payload, "step_indices": [payload, 9]}])
    assert "<img" not in output
    assert "&lt;img" in output
    assert payload not in output
    assert "tvGotoWorkflowStep(0)" in output
    assert "tvGotoWorkflowStep(9)" in output


@pytest.mark.parametrize("note_location", ["top-level", "metadata"])
def test_label_status_discloses_recorded_model_taxonomy_and_source_note(tmp_path, note_location):
    model = "OFFLINE HEURISTIC TEST OVERLAY <script>not_a_model()</script>"
    note = "Deterministic fixture & demonstration labels; not a model judgment."
    record = {
        "taxonomy_version": "v1",
        "model": model,
        "steps": [{"index": 1, "phase": "understand", "action": "code_reading", "duration_s": 1.0}],
    }
    if note_location == "metadata":
        record["metadata"] = {"source_note": note}
    else:
        record["source_note"] = note
    path = tmp_path / "fixture_labeled.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    panel = build_label_ui_payload(str(path))
    assert "Labels loaded" in panel["badge_html"]
    assert "Label provenance" in panel["status_html"]
    assert "OFFLINE HEURISTIC TEST OVERLAY" in panel["status_html"]
    assert "Taxonomy" in panel["status_html"] and ">v1</dd>" in panel["status_html"]
    assert "fixture &amp; demonstration" in panel["status_html"]
    assert "<script>" not in panel["status_html"]
    assert "&lt;script&gt;not_a_model()&lt;/script&gt;" in panel["status_html"]
    assert panel["phase_count_fig"].data


def test_label_provenance_does_not_invent_an_annotator_or_model(tmp_path):
    path = tmp_path / "manual_or_unknown_labeled.json"
    path.write_text(json.dumps({"taxonomy_version": "v1", "steps": []}), encoding="utf-8")
    output = build_label_ui_payload(str(path))["status_html"]
    assert "Taxonomy" in output
    assert "Model" not in output and "Source note" not in output
    assert "manual" not in output.lower() and "heuristic" not in output.lower()
