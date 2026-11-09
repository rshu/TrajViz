"""The combined analysis panel keeps configuration separate from model work."""

from __future__ import annotations

from unittest.mock import Mock

import gradio as gr
import pytest

from agent_trajviz.insight.llm_config import AnalysisLLMConfig, AnalysisSettings
from agent_trajviz.insight.ui import sidebar
from agent_trajviz.insight.ui.shared import SharedState


@pytest.fixture(scope="module")
def panel():
    with gr.Blocks(analytics_enabled=False) as app:
        shared = SharedState(
            state_steps=gr.State([]), state_raw=gr.State({}), state_dark=gr.State(False),
            state_analysis_brief=gr.State(""), state_run=gr.State(None),
            state_overview_issues=gr.State([]), state_settings=gr.State(None),
        )
        refs = sidebar.layout()
        load_event = gr.Button("Fixture load").click(fn=lambda: None)
        model_events = sidebar.bind(refs, shared, [load_event])
    handlers = {bf.fn.__name__: bf for bf in app.fns.values() if bf.fn is not None}
    return app, refs, shared, handlers, model_events


@pytest.fixture
def selected():
    return AnalysisSettings(
        config=AnalysisLLMConfig(
            base_url="https://session.example.test/v1", api_key="session-only-fixture-key",
            model="configured-model", provider="openai", temperature=None,
            max_tokens=1234, timeout=15, source="session",
        ),
        language="German",
    )


def _descendants(block):
    yield block
    for child in getattr(block, "children", ()):
        yield from _descendants(child)


def test_settings_live_once_inside_the_analysis_panel(panel):
    app, refs, _shared, _handlers, _events = panel
    tabs = {block.elem_id: block for block in app.blocks.values() if isinstance(block, gr.Tab)}
    settings_tab = tabs["analysis-settings-tab"]
    assert {tab.label for tab in tabs.values()} == {"Analysis", "Settings"}
    panel_ids = {block._id for block in _descendants(refs.analysis_sidebar)}
    settings_ids = {block._id for block in _descendants(settings_tab)}
    assert settings_tab._id in panel_ids
    assert refs.settings.api_key._id in settings_ids
    assert refs.settings.apply._id in settings_ids
    assert refs.settings.reset._id in settings_ids
    assert sum(isinstance(block, gr.BrowserState) for block in app.blocks.values()) == 1
    assert refs.settings.api_key.type == "password"
    assert refs.settings.api_key.value == ""


@pytest.mark.parametrize("history", [[], [{"role": "assistant", "content": "existing analysis"}]])
def test_opening_panel_or_settings_does_not_request_a_model(panel, selected, monkeypatch, history):
    app, _refs, _shared, handlers, _events = panel
    request = Mock(side_effect=AssertionError("Opening configuration must not request analysis"))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", request)
    result = handlers["on_sidebar_expand"].fn([{"index": 0}], "packed evidence", history, selected)
    assert result[:2] == ("packed evidence", history)
    assert "configured-model" in result[2]
    assert selected.config.api_key not in result[2]
    settings_tab = next(block for block in app.blocks.values()
                        if isinstance(block, gr.Tab) and block.elem_id == "analysis-settings-tab")
    assert not any(block_id == settings_tab._id for bf in app.fns.values() for block_id, _event in bf.targets)
    request.assert_not_called()


def test_loading_while_panel_is_open_only_replaces_local_evidence(panel, selected, monkeypatch):
    _app, _refs, _shared, handlers, _events = panel
    request = Mock(side_effect=AssertionError("Loading must not request analysis"))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", request)
    result = handlers["on_trajectory_loaded"].fn([{"index": 0}], "new evidence", selected)
    assert result[:2] == ("new evidence", [])
    assert "1 steps" in result[2] and "configured-model" in result[2]
    assert selected.config.api_key not in result[2]
    request.assert_not_called()


def test_explicit_analyze_is_a_click_using_current_session_settings(panel, selected, monkeypatch):
    _app, refs, shared, handlers, model_events = panel
    history = [{"role": "assistant", "content": "requested analysis"}]
    request = Mock(return_value=("packed evidence", history))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", request)
    trigger = handlers["on_analyze_trajectory"]
    assert trigger.targets == [(refs.analysis_run._id, "click")]
    assert trigger.inputs[-1] is shared.state_settings
    # Analyze, follow-up ask, and CodeAgent "AI 深入分析" from an Overview issue card.
    assert len(model_events) == 3
    result = trigger.fn([{"index": 0}], "packed evidence", selected)
    request.assert_called_once_with(
        [{"index": 0}], brief="packed evidence", config=selected.config, language="German",
    )
    assert result[:2] == ("packed evidence", history)
    assert selected.config.api_key not in result[2]


def test_explicit_analyze_without_a_trajectory_does_not_request_a_model(panel, selected, monkeypatch):
    _app, _refs, _shared, handlers, _events = panel
    request = Mock()
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", request)
    result = handlers["on_analyze_trajectory"].fn([], "", selected)
    assert result[:2] == ("", [])
    assert "load a trajectory first" in result[2]
    request.assert_not_called()


def test_followup_submission_keeps_its_current_session_connection(panel, selected, monkeypatch):
    _app, refs, _shared, handlers, _events = panel
    history = [{"role": "assistant", "content": "initial analysis"}]
    request = Mock(return_value=history)
    monkeypatch.setattr(sidebar, "answer_question", request)
    trigger = handlers["on_analysis_ask"]
    assert trigger.targets == [(refs.analysis_input._id, "submit")]
    assert trigger.fn("Why?", history, "packed evidence", selected) == (history, "")
    request.assert_called_once_with(
        "Why?", history, "packed evidence", config=selected.config, language="German",
    )
