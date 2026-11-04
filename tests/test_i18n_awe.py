"""AWE translates its reading guide without changing recorded evidence."""
from __future__ import annotations

import copy
import html
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import Mock

import gradio as gr

from agent_trajviz.awe import render
from agent_trajviz.awe.inspection_render import inspection_html
from agent_trajviz.awe.runtime import available_operations
from agent_trajviz.awe.summary import attribution_summary_html
from agent_trajviz.i18n import translate
from agent_trajviz.i18n.catalog_awe import ZH_CN
from agent_trajviz.insight.ui import awe_tab


class _Markers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.keys = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "data-tv-i18n" in attributes:
            self.keys.append(attributes["data-tv-i18n"])


def _keys(body: str) -> list[str]:
    parser = _Markers()
    parser.feed(body)
    return parser.keys


def test_inspection_marks_known_findings_and_keeps_all_recorded_text():
    source = "At a glance <script>alert('recording')</script>"
    result = {
        "available": True,
        "workflow_indices": [2],
        "summary": {"steps": 3, "tool_calls": 2, "failed_calls": 2},
        "findings": [{"kind": "tool_failures", "title": "2 tool calls recorded failures",
                      "observation": "The recording contains failed statuses, nonzero exits, or recognizable tool-error evidence.",
                      "next_check": "Inspect the recorded error and surrounding actions before deciding whether it needs a change.",
                      "step_indices": [2], "evidence": [{"workflow_index": 2, "output_excerpt": source}]}],
        "files": [{"path": "At a glance", "step_indices": [2]}],
    }
    original = copy.deepcopy(result)
    body = inspection_html(result)
    keys = _keys(body)
    assert "{count} tool calls recorded failures" in keys
    assert "Recorded findings" in keys and "Recorded file targets ({count})" in keys
    assert "Next check:" in keys
    assert translate("{count} tool calls recorded failures", "zh-CN", count=2) == "2 次工具调用记录了失败"
    assert "<pre>" + html.escape(source, quote=True) + "</pre>" in body
    assert "<td>At a glance</td>" in body
    assert "<script>" not in body
    assert result == original


def test_unfamiliar_findings_are_not_mistaken_for_application_labels():
    body = inspection_html({"available": True, "findings": [
        {"kind": "plugin_note", "title": "At a glance", "observation": "Tool", "next_check": "Command"},
        {"kind": ["malformed"], "title": "2 tool calls recorded failures"},
    ]})
    assert "<div class='ai-finding-title'>1. At a glance</div>" in body
    assert "<p>Tool</p>" in body
    assert "<p class='ai-next'><strong>" in body and "</strong> Command</p>" in body
    assert "{count} tool calls recorded failures" not in _keys(body)


def test_saved_csv_and_markdown_are_not_translated_as_ui_chrome():
    run = SimpleNamespace(artifacts={
        "result.csv": "Agent,Outcome\nAt a glance,Recorded findings\n",
        "result.md": "# At a glance\n\nRecorded findings\n",
    })
    csv = render.report_html(run, "result.csv")
    markdown = render.report_html(run, "result.md")
    assert "<th scope='col'>Agent</th>" in csv
    assert "<td>At a glance</td>" in csv
    assert "Agent" not in _keys(csv)
    assert "At a glance" not in _keys(markdown)
    assert "<h4>At a glance</h4>" in markdown
    assert "Complete original artifact" in _keys(csv)


def test_computed_scorecard_marks_its_headers_but_preserves_agent_ids():
    run = SimpleNamespace(artifacts={"scorecard.json": {
        "agents": {"At a glance": {"scores": {"code_editing": 75}, "n_tasks": 2}},
    }})
    body = render.report_html(run, "scorecard.json")
    assert "Recorded capability scores" in _keys(body)
    assert "Editing" in _keys(body) and "Agent" in _keys(body)
    assert "<th scope='row'>At a glance</th>" in body


def test_attribution_reading_guide_preserves_original_observations_and_checks():
    check = '<img src=x onerror="alert(1)">'
    case = {
        "resolved": False,
        "faults": [{"capability": "code_editing", "error_type": "specific_error_id", "blame_weight": 1,
                    "is_primary": True, "evidence_chain": {
                        "links": [{"kind": "observation", "statement": "At a glance"}],
                        "counterfactual": {"status": "not_verified", "what_would_confirm": check},
                    }}],
        "scorecard": [{"capability": "code_editing", "assessed": True},
                      {"capability": "tool_use", "assessed": False}],
    }
    original = copy.deepcopy(case)
    body = attribution_summary_html(case)
    keys = _keys(body)
    assert "Code Editing" in keys and "What happened" in keys
    assert "Recorded check ({status}): {check}" in keys
    assert "{assessed} of {total} capabilities assessed; {unassessed} unassessed." in keys
    assert "<li style='margin:3px 0;'>At a glance</li>" in body
    assert "specific error id" in body and "&lt;img" in body and "<img" not in body
    assert case == original


def test_every_bundled_operation_has_chinese_display_and_help_labels():
    for descriptor in available_operations():
        label = descriptor["label"]
        combined = descriptor["group"] + " · " + label
        assert label in ZH_CN and combined in ZH_CN
        assert translate(label, "zh-CN") != label
        assert translate(combined, "zh-CN") != combined


def _saved_run():
    return SimpleNamespace(
        cases=[{"case_id": "case-agent", "agent": "Agent", "instance_id": "task-a", "faults": [], "resolved": False},
               {"case_id": "case-outcomes", "agent": "All outcomes", "instance_id": "task-b", "faults": []}],
        graphs=None, symptoms=None, recommendations=None, artifacts={}, manifest={}, warnings=(), fingerprints={},
    )


def _callback(app, name):
    functions = [function for function in app.fns.values() if getattr(function.fn, "__name__", "") == name]
    assert len(functions) == 1
    return functions[0]


def test_import_in_chinese_translates_only_owned_filter_sentinels():
    run = _saved_run()
    before = copy.deepcopy(vars(run))
    outputs = awe_tab._load_outputs(run, "Imported run", language="zh-CN")
    assert outputs[2].choices == [("所有智能体", "All agents"), ("Agent", "Agent"), ("All outcomes", "All outcomes")]
    assert outputs[2].value == "All agents"
    assert outputs[3]["choices"] == [("所有结果", "All outcomes"), ("已解决", "solved"), ("失败", "failed"), ("未知", "unknown")]
    assert outputs[3]["value"] == "All outcomes"
    assert outputs[5].choices == render.case_options(run)
    assert outputs[0] is run and vars(run) == before
    # Existing direct users retain the default English labels and canonical reset.
    english = awe_tab._load_outputs(run, "Imported run")
    assert english[2].choices[0] == ("All agents", "All agents") and english[3] == "All outcomes"


def test_language_change_rebuilds_current_choices_without_resetting_selection_or_running_work(monkeypatch):
    from agent_trajviz.awe import runtime, saved

    run = _saved_run()
    before = copy.deepcopy(vars(run))
    load = Mock(return_value=run)
    operation = Mock(side_effect=AssertionError("Changing language must not run evaluation"))
    monkeypatch.setattr(saved, "load_run", load)
    monkeypatch.setattr(runtime, "run_operation", operation)
    with gr.Blocks() as app:
        language = gr.Dropdown(choices=[("English", "en"), ("简体中文", "zh-CN")], value="en")
        with gr.Tabs():
            refs = awe_tab.layout()
        awe_tab.bind(refs, app, language=language)

    change = _callback(app, "on_awe_language_change")
    assert change.outputs == [refs.agent, refs.outcome]
    assert change.inputs == [refs.run, language]
    chinese = change.fn(run, "zh-CN")
    english = change.fn(run, "en")
    for updates in (chinese, english):
        # Choice labels can change, while selected canonical values (including
        # a specific imported agent/outcome) are never written by this event.
        assert all("value" not in update for update in updates)
        assert ("Agent", "Agent") in updates[0]["choices"]
        assert ("All outcomes", "All outcomes") in updates[0]["choices"]
        assert "failed" in [value for _, value in updates[1]["choices"]]
    assert chinese[0]["choices"][0] == ("所有智能体", "All agents")
    assert english[0]["choices"][0] == ("All agents", "All agents")
    load.assert_not_called()
    operation.assert_not_called()
    assert vars(run) == before

    # Both import paths pass the current language through to their display
    # projection, without treating recorded agent names as translation keys.
    imported = _callback(app, "on_awe_import").fn("saved.json", "", "zh-CN")
    generated = _callback(app, "on_awe_import_result").fn("completed-operation", "zh-CN")
    assert imported[2].choices == generated[2].choices == chinese[0]["choices"]
    assert imported[3]["choices"] == generated[3]["choices"] == chinese[1]["choices"]
    assert load.call_count == 2
    operation.assert_not_called()
