"""Reply language offers an immediate choice without resetting connection state."""

from copy import deepcopy
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import pytest

from agent_trajviz.i18n import translate
from agent_trajviz.insight.llm_config import build_session_settings, request_reply_language
from agent_trajviz.insight.ui import settings
from agent_trajviz.insight.ui.shared import SharedState

_ENV = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "ANALYZE_BASE_URL": "https://environment.example.test/v1",
    "ANALYZE_API_KEY": "environment-private-key",
    "ANALYZE_MODEL": "environment-model",
    "ANALYZE_PROVIDER": "openai",
    "ANALYZE_LANGUAGE": "German",
}


def _function(app, name):
    matches = [function for function in app.fns.values() if getattr(function.fn, "__name__", "") == name]
    assert len(matches) == 1
    return matches[0]


def _output(function, values, component):
    return next(value for target, value in zip(function.outputs, values, strict=True) if target is component)


@pytest.fixture
def make_settings():
    applications = []

    def make(locale="en", *, with_selector=True):
        with patch.dict(os.environ, _ENV, clear=True), gr.Blocks(analytics_enabled=False) as app:
            previous = build_session_settings(
                provider="openai", base_url="https://active.example.test/v1", api_key="active-private-key",
                model="active-model", temperature=.2, max_tokens=1024, timeout=30, language="French",
            )
            shared = SharedState(gr.State([]), gr.State({}), gr.State(False), gr.State("evidence"),
                                 gr.State(None), gr.State(previous))
            sidebar = SimpleNamespace(analysis_chatbot=gr.Chatbot(value=[{"role": "assistant", "content": "old reply"}]),
                                      analysis_input=gr.Textbox(value="draft question"), analysis_status=gr.HTML("ready"))
            overview = SimpleNamespace(issues_html=gr.HTML("recorded issues"))
            selector = gr.Radio([("EN", "en"), ("中", "zh-CN")], value=locale) if with_selector else None
            refs = settings.layout()
            model_event = gr.Button("Analyze").click(lambda: "reply", outputs=[sidebar.analysis_status], api_name=False)
            settings.bind(refs, shared, sidebar, overview, app, model_events=(model_event,), language=selector)
        applications.append(app)
        return app, refs, selector, shared, sidebar, model_event

    yield make
    for app in applications:
        app.close()


@pytest.mark.parametrize(("locale", "expected"), [("en", "English"), ("zh-CN", "Simplified Chinese")])
def test_editable_reply_dropdown_initializes_from_selector_with_two_bilingual_choices(make_settings, locale, expected):
    app, refs, selector, _, _, _ = make_settings(locale)
    config = next(component["props"] for component in app.get_config_file()["components"]
                  if component["id"] == refs.language._id)
    assert isinstance(refs.language, gr.Dropdown)
    assert config["interactive"] is True and config["value"] == expected
    assert [tuple(choice) for choice in config["choices"]] == [("English", "English"), ("简体中文", "Simplified Chinese")]
    assert config["allow_custom_value"] is False
    assert refs.language.info == "Choose the language for the next reply. Switching EN | 中 updates this choice."
    assert translate(refs.language.info, "zh-CN") == "选择下一条回复的语言。切换 EN | 中 会更新此选项。"
    callback = _function(app, "on_reply_language_change")
    assert callback.inputs == [selector] and callback.outputs == [refs.language]
    assert callback.queue is False and callback.api_visibility == "private"


def test_switch_updates_only_reply_language_and_cancels_model_requests_without_mutating_state(make_settings):
    app, refs, selector, shared, sidebar, model_event = make_settings()
    refs.api_key.value = "unsaved-private-key"
    refs.browser_state.value = {"version": 1, "values": {"model": "saved-model", "language": "German"}}
    state_before = deepcopy(shared.state_settings.value)
    remembered_before = deepcopy(refs.browser_state.value)
    history_before = deepcopy(sidebar.analysis_chatbot.value)
    environment = dict(os.environ)
    callback = _function(app, "on_reply_language_change")
    assert callback.fn("zh-CN") == "Simplified Chinese"
    assert callback.fn("en") == "English"
    assert callback.fn("invalid-client-locale") == "English"
    assert refs.api_key.value == "unsaved-private-key"
    assert shared.state_settings.value == state_before
    assert refs.browser_state.value == remembered_before
    assert sidebar.analysis_chatbot.value == history_before and sidebar.analysis_input.value == "draft question"
    assert dict(os.environ) == environment
    assert any(model_event["id"] in function.cancels for function in app.fns.values()
               if (selector._id, "change") in function.targets)


def _saved(language="German"):
    return {"version": 1, "values": {
        "provider": "openai", "base_url": "https://saved.example.test/v1", "model": "saved-model",
        "temperature": .4, "max_tokens": 2048, "timeout": 60, "language": language,
        "api_key": "injected-browser-secret",
    }}


def test_restore_and_reset_preserve_manual_reply_choice_and_connection_secret_rules(make_settings):
    app, refs, _, shared, _, _ = make_settings("zh-CN")
    # A manual English draft takes priority over the header's Chinese default.
    refs.language.value = "English"
    restore = _function(app, "on_restore_settings")
    saved = _saved()
    saved_before = deepcopy(saved)
    restored = restore.fn(saved)
    assert _output(restore, restored, refs.language) == gr.skip()
    assert _output(restore, restored, refs.model) == "saved-model"
    assert _output(restore, restored, refs.base_url) == "https://saved.example.test/v1"
    assert _output(restore, restored, refs.api_key) == gr.update(value="", placeholder="")
    active = _output(restore, restored, shared.state_settings)
    assert active.language == "German" and active.config.api_key == ""
    remembered = _output(restore, restored, refs.browser_state)
    assert remembered["values"]["language"] == "German" and "api_key" not in remembered["values"]
    assert "injected-browser-secret" not in json.dumps(remembered) and saved == saved_before

    reset = _function(app, "on_reset_settings")
    with patch.dict(os.environ, _ENV, clear=True):
        reset_values = reset.fn([], None)
    assert _output(reset, reset_values, refs.language) == gr.skip()
    assert _output(reset, reset_values, refs.model) == "environment-model"
    assert _output(reset, reset_values, refs.browser_state) == {}
    assert _output(reset, reset_values, refs.api_key) == gr.update(value="", placeholder="********")
    assert _output(reset, reset_values, shared.state_settings) is None
    assert refs.language.value == "English"


@pytest.mark.parametrize("saved", [{}, {"version": 1, "values": {"language": "", "model": "bad\nmodel"}}])
def test_missing_or_invalid_saved_preferences_cannot_clobber_the_reply_dropdown(make_settings, saved):
    app, refs, _, _, _, _ = make_settings("zh-CN")
    restore = _function(app, "on_restore_settings")
    with patch.dict(os.environ, _ENV, clear=True):
        restored = restore.fn(saved)
    assert _output(restore, restored, refs.language) == gr.skip()
    assert _output(restore, restored, refs.api_key) == gr.update(value="", placeholder="********")
    assert refs.language.value == "Simplified Chinese"


@pytest.mark.parametrize("choice", ["English", "Simplified Chinese"])
def test_apply_remembers_only_nonsecret_values_and_does_not_rewrite_the_reply_choice(make_settings, choice):
    app, refs, _, shared, _, _ = make_settings("zh-CN")
    apply = _function(app, "on_apply_settings")
    applied = apply.fn("openai", "https://active.example.test/v1", "entered-private-key", "active-model",
                       .3, 1024, 30, choice, False, True, None, [], None)
    remembered = _output(apply, applied, refs.browser_state)
    assert remembered["values"]["language"] == choice
    assert "api_key" not in remembered["values"] and "entered-private-key" not in json.dumps(remembered)
    assert _output(apply, applied, shared.state_settings).config.api_key == "entered-private-key"
    assert _output(apply, applied, refs.api_key) == gr.update(value="", placeholder="********")
    assert refs.language not in apply.outputs


@pytest.mark.parametrize("key", ["short-secret", "a-much-longer-private-key-that-must-never-be-visible"])
def test_applied_key_presence_uses_constant_placeholder_without_exposing_key_or_length(make_settings, key):
    app, refs, _, shared, _, _ = make_settings()
    apply = _function(app, "on_apply_settings")
    applied = apply.fn("openai", "https://active.example.test/v1", key, "active-model",
                       .3, 1024, 30, "English", False, True, None, [], None)
    assert _output(apply, applied, refs.api_key) == gr.update(value="", placeholder="********")
    assert _output(apply, applied, shared.state_settings).config.api_key == key
    public = [value for component, value in zip(apply.outputs, applied, strict=True)
              if not isinstance(component, gr.State)]
    assert key not in json.dumps(public)
    assert "api_key" not in _output(apply, applied, refs.browser_state)["values"]


@pytest.mark.parametrize("changed_endpoint,clear", [(True, False), (False, True)])
def test_changed_or_cleared_endpoint_has_no_retained_key_placeholder(make_settings, changed_endpoint, clear):
    app, refs, _, shared, _, _ = make_settings()
    previous = shared.state_settings.value
    apply = _function(app, "on_apply_settings")
    applied = apply.fn("openai", "https://different.example.test/v1" if changed_endpoint else previous.config.base_url,
                       "", "active-model", .3, 1024, 30, "English", clear, False, previous, [], None)
    assert _output(apply, applied, shared.state_settings).config.api_key == ""
    assert _output(apply, applied, refs.api_key) == gr.update(value="", placeholder="")


def test_reset_with_no_environment_key_clears_the_presence_placeholder(make_settings):
    app, refs, _, _, _, _ = make_settings()
    reset = _function(app, "on_reset_settings")
    with patch.dict(os.environ, {**_ENV, "ANALYZE_API_KEY": ""}, clear=True):
        reset_values = reset.fn([], None)
    assert _output(reset, reset_values, refs.api_key) == gr.update(value="", placeholder="")


def test_standalone_legacy_preferences_normalize_dropdown_without_changing_stored_fallback(make_settings):
    app, refs, selector, shared, _, _ = make_settings(with_selector=False)
    assert selector is None and refs.language.interactive is True
    assert refs.language.value == "English"
    assert not any(getattr(function.fn, "__name__", "") == "on_reply_language_change" for function in app.fns.values())
    restore = _function(app, "on_restore_settings")
    for stored, projected in (("German", "English"), ("Spanish", "English"), ("English", "English"),
                              ("简体中文", "Simplified Chinese"), ("zh-CN", "Simplified Chinese"),
                              ("Simplified Chinese", "Simplified Chinese")):
        restored = restore.fn(_saved(stored))
        assert _output(restore, restored, refs.language) == projected
        assert _output(restore, restored, shared.state_settings).language == stored
    reset = _function(app, "on_reset_settings")
    with patch.dict(os.environ, _ENV, clear=True):
        reset_values = reset.fn([], None)
    assert _output(reset, reset_values, refs.language) == "English"


@pytest.mark.parametrize(("ui_language", "choice"), [
    ("en", "English"), ("en", "Simplified Chinese"), ("zh-CN", "English"), ("zh-CN", "Simplified Chinese"),
])
def test_manual_reply_dropdown_choice_wins_for_next_request_without_applying_settings(make_settings, ui_language, choice):
    _, refs, _, shared, _, _ = make_settings(ui_language)
    active_before = deepcopy(shared.state_settings.value)
    selected = refs.language.preprocess(choice)
    assert selected == choice
    assert request_reply_language(active_before.language, ui_language, selected) == choice
    assert shared.state_settings.value == active_before
    with pytest.raises(gr.Error, match="choices"):
        refs.language.preprocess("German")
