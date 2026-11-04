"""The served Settings feature must isolate connections and keep keys private."""

from __future__ import annotations

import json
import os
import socket
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from gradio_client import Client, handle_file

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "codearts_minimal.json"
_SERVER_KEY = "server-secret-not-for-browser"
_SERVER_URL = "https://server.example.test/v1"
_ENV = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "ANALYZE_BASE_URL": _SERVER_URL,
    "ANALYZE_API_KEY": _SERVER_KEY,
    "ANALYZE_MODEL": "server-model",
    "ANALYZE_PROVIDER": "openai",
    "ANALYZE_TEMPERATURE": "0.2",
    "ANALYZE_MAX_TOKENS": "2048",
    "ANALYZE_TIMEOUT": "120",
    "ANALYZE_LANGUAGE": "English",
}


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _function(app, name):
    matches = [bf for bf in app.fns.values() if getattr(bf.fn, "__name__", "") == name]
    assert len(matches) == 1, f"expected exactly one {name} handler"
    return matches[0]


def _visible_inputs(fn):
    return [component for component in fn.inputs if not isinstance(component, (gr.State, gr.BrowserState))]


def _invoke(app, client, name, values=None):
    fn = _function(app, name)
    values = values or {}
    inputs = [values.get(component.label, component.value) for component in _visible_inputs(fn)]
    return client.predict(*inputs, api_name="/" + fn.api_name)


def _apply(app, client, **overrides):
    values = {
        "Provider": "openai",
        "Base URL": "https://session.example.test/v1",
        "API key": "session-secret",
        "Model": "session-model",
        "Temperature": 0.4,
        "Max output tokens": 777,
        "Timeout (seconds)": 45,
        "Reply language": "English",
        "Clear API key": False,
        "Remember connection details in this browser": False,
    }
    values.update(overrides)
    return _invoke(app, client, "on_apply_settings", values)


class SettingsHttpTests(unittest.TestCase):
    """Use real HTTP sessions; the model transport is a recording local stub."""

    @classmethod
    def setUpClass(cls):
        from agent_trajviz.insight import insight

        cls._environment = patch.dict(os.environ, _ENV)
        cls._environment.start()
        cls._dotenv = patch.object(insight, "load_env_files")
        cls._dotenv.start()
        try:
            cls.app = insight.build_ui()
            cls.app.launch(
                server_name="127.0.0.1", server_port=_free_port(),
                prevent_thread_lock=True, quiet=True, inbrowser=False,
            )
            cls.base = cls.app.local_url.rstrip("/")
        except BaseException:
            cls._dotenv.stop()
            cls._environment.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls.app.close()
        finally:
            cls._dotenv.stop()
            cls._environment.stop()

    def setUp(self):
        self.calls = []

        def complete(config, system, messages):
            self.calls.append((config, system, messages))
            return f"Recorded model {config.model}"

        self.transport = patch("agent_trajviz.insight.assistant.complete_chat", side_effect=complete)
        self.transport.start()
        self.addCleanup(self.transport.stop)

    def client(self):
        return Client(self.base, verbose=False)

    def load(self, client):
        _invoke(self.app, client, "_do_load", {
            "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_FIXTURE)),
            "Format": "", "Captured session": None,
        })

    def ask(self, client, language="en", reply_language="English"):
        return _invoke(self.app, client, "on_analysis_ask", {
            "Question": "Explain this run", "Analysis": [], "Language / 语言": language,
            "Reply language": reply_language,
        })

    def test_server_key_is_absent_from_public_page_config_and_api_metadata(self):
        for path in ("/", "/config", "/gradio_api/info"):
            with self.subTest(path=path), urllib.request.urlopen(self.base + path, timeout=30) as response:  # noqa: S310 - loopback only
                self.assertNotIn(_SERVER_KEY, response.read().decode())
        settings = self.app.get_config_file()
        passwords = [component for component in settings["components"]
                     if component.get("props", {}).get("label") == "API key"]
        self.assertEqual(len(passwords), 1)
        self.assertEqual(passwords[0]["props"]["type"], "password")
        self.assertFalse(passwords[0]["props"].get("value"))

    def test_two_sessions_use_their_own_connection_and_language(self):
        first, second, untouched = self.client(), self.client(), self.client()
        _apply(self.app, first, **{"Model": "first-model", "API key": "first-key",
                                  "Reply language": "Simplified Chinese", "Provider": "anthropic"})
        _apply(self.app, second, **{"Model": "second-model", "API key": "second-key",
                                   "Reply language": "English"})
        self.assertEqual(self.calls, [], "applying settings must not call a provider")
        for client, language, reply_language in ((first, "en", "Simplified Chinese"),
                                                 (second, "zh-CN", "English"), (untouched, "en", "English")):
            self.load(client)
            self.ask(client, language, reply_language)
        self.assertEqual([call[0].model for call in self.calls], ["first-model", "second-model", "server-model"])
        self.assertEqual([call[0].api_key for call in self.calls], ["first-key", "second-key", _SERVER_KEY])
        self.assertEqual([call[0].provider for call in self.calls], ["anthropic", "openai", "openai"])
        for call, language in zip(self.calls, ("Simplified Chinese", "English", "English"), strict=True):
            self.assertIn(language, call[1])
            self.assertNotIn("Write the entire reply in German", call[1])
            self.assertNotIn("Write the entire reply in Spanish", call[1])
        self.assertEqual(self.calls[0][0].max_tokens, 777)
        self.assertEqual(self.calls[0][0].timeout, 45)
        self.assertEqual(self.calls[0][0].temperature, 0.4)

    def test_apply_outputs_clear_the_secret_and_do_not_echo_it(self):
        client = self.client()
        result = _apply(self.app, client, **{"API key": "entered-secret-not-for-output"})
        self.assertNotIn("entered-secret-not-for-output", json.dumps(result))
        self.assertNotIn(_SERVER_KEY, json.dumps(result))
        output_components = [component for component in _function(self.app, "on_apply_settings").outputs
                             if not isinstance(component, (gr.State, gr.BrowserState))]
        password_index = next(index for index, component in enumerate(output_components)
                              if component.label == "API key")
        self.assertEqual(result[password_index], "")
        self.assertEqual(self.calls, [])
        self.load(client)
        self.ask(client)
        self.assertEqual(self.calls[-1][0].api_key, "entered-secret-not-for-output")

    def test_invalid_apply_preserves_the_previous_active_settings(self):
        client = self.client()
        _apply(self.app, client, **{"Model": "keep-this-model"})
        try:
            invalid = _apply(self.app, client, **{"Base URL": "file:///tmp/private", "Model": "reject-this-model"})
        except Exception as exc:  # Gradio may surface validation as a user-facing Error.
            self.assertNotIn(_SERVER_KEY, str(exc))
        else:
            self.assertNotIn(_SERVER_KEY, json.dumps(invalid))
        self.assertEqual(self.calls, [])
        self.load(client)
        self.ask(client)
        self.assertEqual(self.calls[-1][0].model, "keep-this-model")
        self.assertEqual(self.calls[-1][0].base_url, "https://session.example.test/v1")

    def test_reset_restores_environment_without_calling_a_provider(self):
        client = self.client()
        _apply(self.app, client)
        result = _invoke(self.app, client, "on_reset_settings")
        self.assertNotIn(_SERVER_KEY, json.dumps(result))
        self.assertEqual(self.calls, [])
        self.load(client)
        self.ask(client)
        self.assertEqual(self.calls[-1][0].model, "server-model")
        self.assertEqual(self.calls[-1][0].api_key, _SERVER_KEY)

    def test_new_target_cannot_receive_an_inherited_server_secret(self):
        client = self.client()
        _apply(self.app, client, **{"API key": "", "Base URL": "https://different.example.test/v1"})
        self.load(client)
        result = self.ask(client)
        self.assertEqual(self.calls, [])
        self.assertNotIn(_SERVER_KEY, json.dumps(result))


    def test_remembered_details_restore_without_a_key_or_environment_fallback(self):
        from agent_trajviz.insight.llm_config import nonsecret_settings
        from agent_trajviz.insight.ui import sidebar

        apply = _function(self.app, "on_apply_settings")
        applied = apply.fn("openai", "https://saved.example.test/v1", "save-only-in-session", "saved-model",
                           0.6, 321, 25, "Spanish", False, True, None, [], None)
        remembered = next(value for component, value in zip(apply.outputs, applied, strict=True)
                          if isinstance(component, gr.BrowserState))
        self.assertEqual(remembered["values"]["model"], "saved-model")
        self.assertEqual(remembered["values"]["base_url"], "https://saved.example.test/v1")
        self.assertNotIn("api_key", remembered["values"])
        self.assertNotIn("save-only-in-session", json.dumps(remembered))
        # Local storage is client-controlled: even an injected key is ignored.
        remembered["values"]["api_key"] = "forged-browser-secret"
        restore = _function(self.app, "on_restore_settings")
        restored = restore.fn(remembered)
        active = next(value for component, value in zip(restore.outputs, restored, strict=True)
                      if isinstance(component, gr.State) and not isinstance(component, gr.BrowserState))
        self.assertEqual(active.config.model, "saved-model")
        self.assertEqual(active.config.api_key, "")
        self.assertFalse(active.config.ready)
        self.assertEqual(nonsecret_settings(active)["language"], "Spanish")
        for component, value in zip(restore.outputs, restored, strict=True):
            if component.label == "API key":
                self.assertEqual(value, "")
            if isinstance(component, gr.BrowserState):
                self.assertNotIn("api_key", value["values"])
                self.assertNotIn("forged-browser-secret", json.dumps(value))
        _, history, status = sidebar._run_analysis(
            [{"index": 0, "role": "user", "parts": [], "tool_calls": []}], "A synthetic run", active,
        )
        self.assertEqual(self.calls, [], "restoration must not call the environment provider")
        self.assertNotIn(_SERVER_KEY, json.dumps(history))
        self.assertIn("Not configured", status)

    def test_disabling_remember_clears_saved_connection_details(self):
        apply = _function(self.app, "on_apply_settings")
        result = apply.fn("openai", "https://session.example.test/v1", "session-only-key", "m",
                          None, 100, 15, "English", False, False, None, [], None)
        remembered = next(value for component, value in zip(apply.outputs, result, strict=True)
                          if isinstance(component, gr.BrowserState))
        self.assertEqual(remembered, {})
        self.assertEqual(self.calls, [])


class SettingsDefaultPrivacyTests(unittest.TestCase):
    def test_credential_bearing_environment_url_never_reaches_browser_defaults(self):
        from agent_trajviz.insight import insight

        environment = dict(_ENV, ANALYZE_BASE_URL="https://user:url-secret@example.test/v1?token=query-secret")
        with patch.dict(os.environ, environment), patch.object(insight, "load_env_files"):
            app = insight.build_ui()
        try:
            public = json.dumps(app.get_config_file(), default=str)
            self.assertNotIn("url-secret", public)
            self.assertNotIn("query-secret", public)
            self.assertNotIn(_SERVER_KEY, public)
        finally:
            app.close()
