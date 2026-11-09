"""Each model request uses the current reply-language dropdown choice."""

from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import pytest
from gradio_client import Client, handle_file

from agent_trajviz.insight import assistant, insight, issue_judge
from agent_trajviz.insight.presenters.issues import OverviewIssue
from agent_trajviz.insight.ui import overview_tab

_TRAJECTORY = Path(__file__).parent / "fixtures" / "codearts_minimal.json"
_ENV = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "ANALYZE_PROVIDER": "openai",
    "ANALYZE_BASE_URL": "https://fixture.example.test/v1",
    "ANALYZE_API_KEY": "fixture-server-secret",
    "ANALYZE_MODEL": "fixture-server-model",
    "ANALYZE_LANGUAGE": "German",
}


def _handler(app, name):
    matches = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name]
    assert len(matches) == 1, name
    return matches[0]


def _is_state(component):
    return isinstance(component, (gr.State, gr.BrowserState))


def _invoke(app, client, name, **values):
    handler = _handler(app, name)
    inputs = [values.get(component.label, component.value) for component in handler.inputs if not _is_state(component)]
    return client.predict(*inputs, api_name="/" + handler.api_name)


def _output(app, name, result, label):
    components = [component for component in _handler(app, name).outputs if not _is_state(component)]
    values = [result] if len(components) == 1 else result
    return next(value for component, value in zip(components, values, strict=True) if component.label == label)


def _configure(app, client, *, model="session-model", key="session-secret"):
    return _invoke(app, client, "on_apply_settings", **{
        "Provider": "openai", "Base URL": "https://session.example.test/v1", "Model": model,
        "API key": key, "Temperature": 0.4, "Max output tokens": 777, "Timeout (seconds)": 45,
        "Reply language": "English", "Clear API key": False,
        "Remember connection details in this browser": False,
    })


def _load(app, client):
    return _invoke(app, client, "_do_load", **{
        "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_TRAJECTORY)),
        "Format": "", "Captured session": None,
    })


def _assert_language(call, expected):
    judge_language = "Simplified Chinese (Mandarin)" if expected == "Simplified Chinese" else expected
    instruction = (f'Write the values of "where", "fix", and "also" in {judge_language}.' if call["kind"] == "fix"
                   else f"Write the entire reply in {expected}")
    assert instruction in call["system"]
    assert "Write the entire reply in German" not in call["system"]
    if call["kind"] == "fix":
        assert expected in call["messages"][0]["content"]


@pytest.fixture(scope="module")
def served_app():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with patch.dict(os.environ, _ENV, clear=True), patch.object(insight, "load_env_files"):
        app = insight.build_ui()
        app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                   quiet=True, inbrowser=False)
        try:
            yield app, app.local_url
        finally:
            app.close()


@pytest.fixture
def recorded_transport(monkeypatch):
    calls = []

    def record(kind, config, system, messages):
        calls.append({"kind": kind, "config": config, "system": system, "messages": copy.deepcopy(messages)})
        chinese = "Simplified Chinese" in system
        if kind == "fix":
            return json.dumps({"where": "AGENTS.md", "fix": "先读取 src/cache.py" if chinese else "Read src/cache.py first",
                               "confidence": "high"})
        return "中文分析 raw_history/code_reading" if chinese else "English analysis raw_history/code_reading"

    monkeypatch.setattr(assistant, "complete_chat", lambda *args: record("chat", *args))
    monkeypatch.setattr(issue_judge, "complete_chat", lambda *args: record("fix", *args))
    issue = OverviewIssue(kind="error", title="fixture error", detail="command failed", steps=(1,), source_id="fixture")
    monkeypatch.setattr(overview_tab, "collect_overview_issues", lambda _session: [issue])
    return calls


def test_http_analyze_and_followups_use_current_dropdown_and_keep_existing_history(served_app, recorded_transport):
    app, url = served_app
    client = Client(url, verbose=False)
    _configure(app, client)
    _load(app, client)
    assert recorded_transport == [], "applying settings and loading data must not request a model"
    analyzed = _invoke(app, client, "on_analyze_trajectory", **{
        "Language / 语言": "zh-CN", "Reply language": "English",
    })
    history = _output(app, "on_analyze_trajectory", analyzed, "Analysis")
    assert "English analysis raw_history/code_reading" in json.dumps(history)
    _assert_language(recorded_transport[-1], "English")
    preserved = copy.deepcopy(history)

    # A manual choice takes effect immediately, without Apply, even when the
    # interface uses the other language. Existing replies retain their text.
    chinese = _invoke(app, client, "on_analysis_ask", **{
        "Question": "Why is code_reading repeated?", "Analysis": history,
        "Language / 语言": "en", "Reply language": "Simplified Chinese",
    })
    chinese_history = _output(app, "on_analysis_ask", chinese, "Analysis")
    _assert_language(recorded_transport[-1], "Simplified Chinese")
    assert chinese_history[:len(history)] == preserved
    assert "中文分析 raw_history/code_reading" in json.dumps(chinese_history, ensure_ascii=False)
    assert "English analysis raw_history/code_reading" in json.dumps(recorded_transport[-1]["messages"])

    english = _invoke(app, client, "on_analysis_ask", **{
        "Question": "What should I change next?", "Analysis": chinese_history,
        "Language / 语言": "zh-CN", "Reply language": "English",
    })
    english_history = _output(app, "on_analysis_ask", english, "Analysis")
    _assert_language(recorded_transport[-1], "English")
    assert english_history[:len(chinese_history)] == chinese_history
    assert len(recorded_transport) == 3
    for call in recorded_transport:
        config = call["config"]
        assert (config.model, config.api_key, config.base_url) == (
            "session-model", "session-secret", "https://session.example.test/v1",
        )
        assert (config.temperature, config.max_tokens, config.timeout) == (0.4, 777, 45)
    assert "session-secret" not in json.dumps(english)


def test_http_initial_analysis_uses_both_dropdown_options_without_reapplying_settings(served_app, recorded_transport):
    app, url = served_app
    client = Client(url, verbose=False)
    _configure(app, client)
    _load(app, client)
    for locale, reply in (("zh-CN", "English"), ("en", "Simplified Chinese")):
        result = _invoke(app, client, "on_analyze_trajectory", **{
            "Language / 语言": locale, "Reply language": reply,
        })
        _assert_language(recorded_transport[-1], reply)
        history = _output(app, "on_analyze_trajectory", result, "Analysis")
        expected = "中文分析 raw_history/code_reading" if reply == "Simplified Chinese" else "English analysis raw_history/code_reading"
        assert expected in json.dumps(history, ensure_ascii=False)
        assert recorded_transport[-1]["config"].api_key == "session-secret"
    assert len(recorded_transport) == 2


def test_http_sessions_interleave_locales_and_suggest_fixes_without_language_or_key_leakage(served_app,
                                                                                           recorded_transport):
    app, url = served_app
    first, second = Client(url, verbose=False), Client(url, verbose=False)
    _configure(app, first, model="first-model", key="first-private-key")
    _configure(app, second, model="second-model", key="second-private-key")
    _load(app, first)
    _load(app, second)
    assert recorded_transport == []
    requests = [(first, "en", "Simplified Chinese", "first-model", "first-private-key"),
                (second, "zh-CN", "English", "second-model", "second-private-key"),
                (first, "zh-CN", "English", "first-model", "first-private-key")]
    for client, locale, language, model, key in requests:
        _invoke(app, client, "on_analysis_ask", **{
            "Question": "Inspect this run", "Analysis": [], "Language / 语言": locale, "Reply language": language,
        })
        call = recorded_transport[-1]
        _assert_language(call, language)
        assert (call["config"].model, call["config"].api_key) == (model, key)

    for client, locale, language, model, key in (
        (second, "en", "Simplified Chinese", "second-model", "second-private-key"),
        (first, "zh-CN", "English", "first-model", "first-private-key"),
    ):
        result = _invoke(app, client, "on_suggest_fixes", **{
            "Language / 语言": locale, "Reply language": language,
        })
        call = recorded_transport[-1]
        assert call["kind"] == "fix"
        _assert_language(call, language)
        assert (call["config"].model, call["config"].api_key) == (model, key)
        assert ("先读取 src/cache.py" if language == "Simplified Chinese" else "Read src/cache.py first") in result
        assert "first-private-key" not in result and "second-private-key" not in result
    assert len(recorded_transport) == 5


def test_language_toggle_handlers_make_no_model_calls_and_preserve_active_state_history_and_form_values(served_app,
                                                                                                       recorded_transport):
    app, url = served_app
    client = Client(url, verbose=False)
    _configure(app, client)
    _load(app, client)
    selector = _handler(app, "on_analysis_ask").inputs[-2]
    active_state = _handler(app, "on_apply_settings").outputs[0]
    callbacks = [fn for fn in app.fns.values() if fn.fn is not None and fn.inputs == [selector]
                 and getattr(fn.fn, "__name__", "") in ("change", "on_reply_language_change")]
    assert {fn.fn.__name__ for fn in callbacks} == {"change", "on_reply_language_change"}
    for locale in ("zh-CN", "en", "zh-CN"):
        for callback in callbacks:
            values = callback.fn(locale)
            values = values if isinstance(values, tuple) else (values,)
            assert active_state not in callback.outputs
            for component, update in zip(callback.outputs, values, strict=True):
                if isinstance(component, gr.Textbox) and component.type == "password":
                    pytest.fail("language changes must not replace pending API key input")
                if isinstance(component, (gr.Textbox, gr.Number, gr.Chatbot)) and component.label != "Reply language":
                    assert "value" not in update, "language changes must preserve forms and existing chat"
    assert recorded_transport == []
    _invoke(app, client, "on_analysis_ask", **{
        "Question": "Verify retained configuration", "Analysis": [], "Language / 语言": "en",
        "Reply language": "Simplified Chinese",
    })
    _assert_language(recorded_transport[-1], "Simplified Chinese")
    assert recorded_transport[-1]["config"].api_key == "session-secret"
    assert recorded_transport[-1]["config"].model == "session-model"
