"""Connection checks are explicit, use current fields and never apply a key."""

from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr
import pytest
from gradio_client import Client, handle_file

from agent_trajviz.insight import assistant, insight
from agent_trajviz.insight.connection_test import ConnectionTestResult
from agent_trajviz.insight.llm_config import AnalysisLLMConfig, AnalysisSettings
from agent_trajviz.insight.ui import settings as settings_ui

_ENV = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "ANALYZE_BASE_URL": "https://environment.example.test/v1",
    "ANALYZE_API_KEY": "environment-key-must-not-be-returned",
    "ANALYZE_MODEL": "environment-model",
    "ANALYZE_PROVIDER": "openai",
}
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "codearts_minimal.json"


def _function(app, name):
    matches = [bf for bf in app.fns.values() if getattr(bf.fn, "__name__", "") == name]
    assert len(matches) == 1, f"expected one {name} handler"
    return matches[0]


@pytest.fixture(scope="module")
def ui():
    with patch.dict(os.environ, _ENV, clear=True), patch.object(insight, "load_env_files"):
        app = insight.build_ui()
    try:
        yield app
    finally:
        app.close()


@pytest.fixture
def previous():
    return AnalysisSettings(
        config=AnalysisLLMConfig(
            base_url="https://active.example.test/v1", api_key="active-private-key",
            model="active-model", provider="openai", temperature=0.7,
            max_tokens=987, timeout=65, source="session",
        ),
        language="German",
    )


def _arguments(previous=None, **overrides):
    values = {
        "provider": "anthropic", "base_url": "https://entered.example.test/v1",
        "model": "entered-model", "api_key": "entered-private-key",
        "clear_api_key": False, "timeout": 75, "previous": previous,
    }
    values.update(overrides)
    return values


def _probe(ui, previous=None, **overrides):
    return _function(ui, "on_test_connection").fn(**_arguments(previous, **overrides))


def test_probe_is_only_wired_to_explicit_button_and_has_no_trajectory_inputs(ui):
    trigger = _function(ui, "on_test_connection")
    assert len(trigger.targets) == 1 and trigger.targets[0][1] == "click"
    test_button = ui.blocks[trigger.targets[0][0]]
    assert isinstance(test_button, gr.Button)
    assert "test" in test_button.value.lower()
    assert [component.label for component in trigger.inputs if not isinstance(component, gr.State)] == [
        "Provider", "Base URL", "Model", "API key", "Clear API key", "Timeout (seconds)",
    ]
    assert sum(isinstance(component, gr.State) for component in trigger.inputs) == 1
    assert len(trigger.outputs) == 2
    assert trigger.outputs[0] is test_button
    assert isinstance(trigger.outputs[1], gr.HTML)
    assert not any(isinstance(component, (gr.State, gr.BrowserState, gr.Chatbot)) for component in trigger.outputs)


def test_probe_reports_progress_before_request_and_uses_current_form_connection(ui, previous, monkeypatch):
    request = Mock(return_value=ConnectionTestResult(True, "Connection successful."))
    monkeypatch.setattr(settings_ui, "check_connection", request)
    run = _probe(ui, previous)
    progress = next(run)
    request.assert_not_called()
    assert progress[0]["interactive"] is False
    assert "test" in progress[1].lower()
    finished = next(run)
    with pytest.raises(StopIteration):
        next(run)
    assert finished[0]["interactive"] is True
    assert "Connection successful." in finished[1]
    config = request.call_args.args[0]
    assert (config.provider, config.base_url, config.model, config.api_key) == (
        "anthropic", "https://entered.example.test/v1", "entered-model", "entered-private-key",
    )
    assert config.temperature is None and config.max_tokens == 16
    assert config.timeout == 75
    assert "entered-private-key" not in json.dumps([progress, finished])
    assert previous.config.api_key not in json.dumps([progress, finished])


def test_testing_does_not_change_applied_settings_browser_storage_or_environment(ui, previous, monkeypatch):
    initial = copy.deepcopy(previous)
    browser = next(block for block in ui.blocks.values() if isinstance(block, gr.BrowserState))
    stored = copy.deepcopy(browser.value)
    environment = dict(os.environ)
    monkeypatch.setattr(settings_ui, "check_connection", Mock(return_value=ConnectionTestResult(True, "Verified.")))
    list(_probe(ui, previous))
    assert previous == initial
    assert browser.value == stored
    assert dict(os.environ) == environment


def test_blank_key_tests_retained_key_only_for_same_endpoint_and_provider(ui, previous, monkeypatch):
    request = Mock(return_value=ConnectionTestResult(True, "Verified."))
    monkeypatch.setattr(settings_ui, "check_connection", request)
    finished = list(_probe(ui, previous, provider="openai", base_url=previous.config.base_url,
                           model="new-selected-model", api_key=""))[-1]
    assert request.call_args.args[0].api_key == previous.config.api_key
    assert request.call_args.args[0].model == "new-selected-model"
    assert previous.config.api_key not in finished[1]


@pytest.mark.parametrize("overrides", [
    {"provider": "openai", "api_key": ""},
    {"base_url": "https://active.example.test/v1", "api_key": ""},
    {"provider": "openai", "base_url": "https://active.example.test/v1", "api_key": "", "clear_api_key": True},
    {"provider": "openai", "base_url": "https://active.example.test/v1", "clear_api_key": True},
    {"base_url": ""}, {"model": ""}, {"api_key": "", "previous": None},
])
def test_incomplete_or_cleared_connection_never_sends_a_probe(ui, previous, monkeypatch, overrides):
    request = Mock()
    monkeypatch.setattr(settings_ui, "check_connection", request)
    values = _arguments(previous)
    values.update(overrides)
    result = list(_function(ui, "on_test_connection").fn(**values))[-1]
    request.assert_not_called()
    assert result[0]["interactive"] is True
    assert "API key" in result[1] or "Base URL" in result[1] or "Model" in result[1]
    assert "entered-private-key" not in result[1] and previous.config.api_key not in result[1]


@pytest.mark.parametrize("overrides", [
    {"base_url": "file:///tmp/private-key"},
    {"base_url": "https://user:url-secret@example.test/v1?token=query-secret"},
    {"provider": "unsupported"}, {"model": "bad\nmodel"}, {"timeout": "not-a-number"},
])
def test_invalid_connection_is_rejected_locally_without_echoing_secrets(ui, previous, monkeypatch, overrides):
    request = Mock()
    monkeypatch.setattr(settings_ui, "check_connection", request)
    result = list(_probe(ui, previous, **overrides))[-1]
    request.assert_not_called()
    assert result[0]["interactive"] is True
    for value in ("entered-private-key", previous.config.api_key, "url-secret", "query-secret"):
        assert value not in result[1]


def test_provider_result_is_visible_but_cannot_inject_html(ui, previous, monkeypatch):
    monkeypatch.setattr(settings_ui, "check_connection", Mock(return_value=ConnectionTestResult(
        False, "Request failed <script>alert(1)</script> & retry",
    )))
    result = list(_probe(ui, previous))[-1]
    assert result[0]["interactive"] is True
    assert "Request failed" in result[1]
    assert "<script>" not in result[1]
    assert "&lt;script&gt;" in result[1]


def test_unexpected_probe_exception_does_not_expose_connection_or_key(ui, previous, monkeypatch):
    request = Mock(side_effect=RuntimeError(
        "entered-private-key active-private-key https://entered.example.test/v1 provider-body-secret",
    ))
    monkeypatch.setattr(settings_ui, "check_connection", request)
    result = list(_probe(ui, previous))[-1]
    request.assert_called_once()
    assert result[0]["interactive"] is True
    assert "could not be completed" in result[1]
    for value in ("entered-private-key", "active-private-key", "entered.example.test", "provider-body-secret"):
        assert value not in result[1]


def test_connection_edits_cancel_and_clear_the_previous_result(ui):
    trigger = _function(ui, "on_test_connection")
    result_component = trigger.outputs[1]
    for component in trigger.inputs:
        if isinstance(component, gr.State):
            continue
        events = [fn for fn in ui.fns.values() if (component._id, "change") in fn.targets]
        assert any(result_component in fn.outputs for fn in events), component.label
        assert any(trigger._id in fn.cancels for fn in events), component.label


@pytest.mark.parametrize("name", ["on_apply_settings", "on_reset_settings"])
def test_apply_and_reset_cancel_the_probe_and_clear_old_verification(ui, name):
    probe = _function(ui, "on_test_connection")
    action = _function(ui, name)
    # Gradio installs cancellation as a sibling event on the same button.
    assert any(probe._id in fn.cancels for fn in ui.fns.values()
               if set(fn.targets) & set(action.targets))
    followups = [fn for fn in ui.fns.values()
                 if fn.trigger_after == action._id and probe.outputs[1] in fn.outputs and fn.fn is not None]
    assert followups
    reset_button, reset_status = followups[0].fn()
    assert reset_button["interactive"] is True
    assert "Not tested" in reset_status


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def served_ui():
    with patch.dict(os.environ, _ENV, clear=True), patch.object(insight, "load_env_files"):
        app = insight.build_ui()
        app.launch(server_name="127.0.0.1", server_port=_free_port(), prevent_thread_lock=True,
                   quiet=True, inbrowser=False)
        try:
            yield app, app.local_url.rstrip("/")
        finally:
            app.close()


def _invoke(app, client, name, values=None):
    fn = _function(app, name)
    values = values or {}
    inputs = [values.get(component.label, component.value) for component in fn.inputs
              if not isinstance(component, (gr.State, gr.BrowserState))]
    return client.predict(*inputs, api_name="/" + fn.api_name)


def test_http_probe_uses_current_fields_without_applying_them_or_exposing_keys(served_ui, monkeypatch):
    app, url = served_ui
    client = Client(url, verbose=False)
    applied_key = "http-applied-key-private"
    draft_key = "http-draft-key-private"
    _invoke(app, client, "on_apply_settings", {
        "Provider": "openai", "Base URL": "https://active.example.test/v1", "Model": "active-model",
        "API key": applied_key, "Remember connection details in this browser": False,
    })
    probes = []

    def check(config):
        probes.append(config)
        return ConnectionTestResult(True, "Connection successful.")

    monkeypatch.setattr(settings_ui, "check_connection", check)
    result = _invoke(app, client, "on_test_connection", {
        "Provider": "anthropic", "Base URL": "https://draft.example.test/v1", "Model": "draft-model",
        "API key": draft_key, "Clear API key": False, "Timeout (seconds)": 20,
    })
    assert (probes[-1].provider, probes[-1].model, probes[-1].api_key) == ("anthropic", "draft-model", draft_key)
    assert "Connection successful." in json.dumps(result)
    assert all(secret not in json.dumps(result) for secret in (draft_key, applied_key, _ENV["ANALYZE_API_KEY"]))
    calls = []

    def complete(config, _system, _messages):
        calls.append(config)
        return "Recorded active connection"

    monkeypatch.setattr(assistant, "complete_chat", complete)
    _invoke(app, client, "_do_load", {
        "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_FIXTURE)),
        "Format": "", "Captured session": None,
    })
    _invoke(app, client, "on_analysis_ask", {"Question": "Explain this run", "Analysis": []})
    assert (calls[-1].provider, calls[-1].model, calls[-1].api_key) == ("openai", "active-model", applied_key)


def test_http_sessions_cannot_reuse_another_sessions_retained_key(served_ui, monkeypatch):
    app, url = served_ui
    first, second = Client(url, verbose=False), Client(url, verbose=False)
    _invoke(app, first, "on_apply_settings", {
        "Provider": "openai", "Base URL": "https://active.example.test/v1", "Model": "active-model",
        "API key": "first-client-key", "Remember connection details in this browser": False,
    })
    request = Mock(return_value=ConnectionTestResult(True, "Verified."))
    monkeypatch.setattr(settings_ui, "check_connection", request)
    result = _invoke(app, second, "on_test_connection", {
        "Provider": "openai", "Base URL": "https://active.example.test/v1", "Model": "active-model",
        "API key": "", "Clear API key": False,
    })
    request.assert_not_called()
    assert "API key" in json.dumps(result)
    assert "first-client-key" not in json.dumps(result)
    _invoke(app, first, "on_test_connection", {
        "Provider": "openai", "Base URL": "https://active.example.test/v1", "Model": "active-model",
        "API key": "", "Clear API key": False,
    })
    assert request.call_args.args[0].api_key == "first-client-key"
