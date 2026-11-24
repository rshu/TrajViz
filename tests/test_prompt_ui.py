"""Prompt inspection keeps source text literal and clears stale selections."""

from __future__ import annotations

from types import SimpleNamespace

from agent_trajviz.insight.ui import prompt_tab


def _data():
    detail = {
        "id": "request-1", "scope": "native_prompt_state", "evidence": "reconstructed",
        "model": "test-model", "agent_id": "main", "step_index": 0,
        "system": [], "tools": [], "messages": [{"role": "user", "content": "hello"}],
        "components": [{"id": "message-1", "kind": "message", "role": "user", "estimated_tokens": 3,
                        "content": '<script>alert("literal")</script> Settings',
                        "sources": [{"source_path": '<img src=x onerror="alert(1)">'}]}],
        "limitations": [], "missing_references": [], "token_estimate": {"total": 3},
        "changes": {"available": False},
    }
    return {"available": True, "requests": [dict(detail)], "selected_request": detail, "limitations": []}


class Run:
    steps = [{"index": 0}]

    def __init__(self, data):
        self.data = data
        self.selection = None

    def view(self, name, **options):
        assert name == "prompt_construction"
        self.selection = options.get("request_id")
        return SimpleNamespace(data=self.data)


def test_prompt_html_escapes_evidence_but_request_json_keeps_original_content():
    data = _data()
    run = Run(data)
    selector, _chart, summary, _limits, components, _changes, detail = prompt_tab.outputs(run, "request-1")
    assert run.selection == "request-1" and selector["value"] == "request-1"
    assert "Reconstructed" in summary and "#tv-step-0" in summary
    assert "&lt;script&gt;" in components and "<script>" not in components
    assert "&lt;img" in components and "<img" not in components
    assert detail["components"][0]["content"] == '<script>alert("literal")</script> Settings'


def test_language_refresh_keeps_request_identity_and_literal_prompt():
    data = _data()
    run = Run(data)
    values = prompt_tab.outputs(run, "request-1", "zh-CN")
    assert values[0]["value"] == "request-1"
    assert "已记录的提示词状态" in values[0]["choices"][0][0]
    assert values[-1] == data["selected_request"]


def test_unload_or_failed_selection_clears_every_previous_prompt():
    values = prompt_tab.outputs(None)
    assert values[0]["value"] is None and values[-1] is None
    assert values[1]["visible"] is False
    assert not values[3] and not values[4] and not values[5]

    class BrokenRun(Run):
        def view(self, name, **options):
            raise ValueError("Selection belongs to another session")

    values = prompt_tab.outputs(BrokenRun(_data()), "old-request")
    assert values[0]["value"] is None and values[-1] is None
    assert "could not be completed" in values[2]


def test_request_evidence_is_visible_even_without_a_completed_response():
    run = Run(_data())
    run.steps = []
    assert prompt_tab.outputs(run)[-1]["id"] == "request-1"


def test_changes_show_retained_count_and_descriptive_additions():
    detail = _data()["selected_request"]
    detail["changes"] = {"available": True, "previous_request_id": "previous", "added": ["tool-new"],
                         "removed": [], "retained": 7, "estimated_token_delta": 35,
                         "added_components": [{"id": "tool-new", "kind": "tool", "label": "read<script>"}]}
    rendered = prompt_tab._changes(detail)
    assert "Added components" in rendered and "Unchanged components" in rendered
    assert "<p>7</p>" in rendered and "+35" in rendered
    assert "read&lt;script&gt;" in rendered and "read<script>" not in rendered


def test_generated_message_labels_in_changes_have_language_markers():
    detail = _data()["selected_request"]
    detail["changes"] = {"available": True, "added": ["new"], "removed": [], "retained": 0,
                         "added_components": [{"kind": "message", "role": "assistant", "label": "assistant 4"}]}
    assert 'data-tv-i18n="Assistant"' in prompt_tab._changes(detail)


def test_missing_request_inputs_show_partial_evidence_and_unknown_counts():
    data = _data()
    data["selected_request"].update(components=[], messages=[], tools=[], token_estimate={"total": 0},
                                    missing_references=["shape:missing", "blob:missing"])
    summary = prompt_tab.outputs(Run(data))[2]
    assert "Partial reconstruction" in summary and "pc-partial" in summary
    assert "≈ 0" not in summary and summary.count("—") >= 3


def test_large_prompt_preview_is_bounded_without_truncating_export_data():
    data = _data()
    parts = [dict(data["selected_request"]["components"][0], id=f"message-{index}", content="x" * 20000)
             for index in range(100)]
    data["selected_request"]["components"] = parts
    values = prompt_tab.outputs(Run(data))
    assert "Showing" in values[4] and "Preview shortened" in values[4]
    assert len(values[4]) < 350000
    assert len(values[-1]["components"]) == 100
    assert len(values[-1]["components"][50]["content"]) == 20000
