"""Evaluation requests use TrajViz's credential boundary before any HTTP send."""
from __future__ import annotations

import copy
import io
import json
import urllib.error
import urllib.request

import pytest

from agent_trajviz.awe._engine import llm


class Reply:
    def __init__(self, value):
        self.body = json.dumps(value).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


@pytest.fixture
def connection(monkeypatch):
    key = "custom-session-credential"
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", key)
    monkeypatch.setattr(llm.config, "OPENROUTER_BASE_URL", "https://evaluation.example/v1")
    monkeypatch.setattr(llm.config, "JUDGE_MODEL", "test-model")
    monkeypatch.setattr(llm, "_backoff", lambda _attempt: None)
    return key


def _success():
    return Reply({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}})


def test_outbound_text_and_structured_fields_are_redacted_without_mutating_evidence(connection, monkeypatch):
    provider_key = "sk-" + "aB1" * 12
    password = "GeneratedPassword9Value"
    token = "Z9" * 14
    private_body = "SYNTHETIC-PRIVATE-MATERIAL"
    messages = [{"role": "system", "content": "Evaluate the public task."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"export API_KEY={provider_key}; TOKEN={token}; {connection}"},
                    {"type": "text", "text": f"https://user:{password}@example.test/"},
                    {"type": "text", "text": "-----BEGIN PRIVATE KEY-----\n" + private_body + "\n-----END PRIVATE KEY-----"}],
                 "tool_calls": [{"function": {"arguments": json.dumps({"header": "Authorization: Bearer " + token})}}],
                 "recorded": {"count": 7, "available": False, "extra": [provider_key]}}]
    original = copy.deepcopy(messages)
    observed = []
    def send(request, *, timeout):
        payload = json.loads(request.data)
        observed.append(payload)
        assert request.get_header("Authorization") == "Bearer " + connection
        assert request.get_header("X-title") == "TrajViz"
        assert request.get_header("Http-referer") == "https://github.com/rshu/TrajViz"
        assert timeout == 15
        return _success()
    monkeypatch.setattr(urllib.request, "urlopen", send)
    result = llm.chat(messages, timeout=15, retries=1)
    serialized = json.dumps(observed)
    for secret in (provider_key, password, token, connection, private_body):
        assert secret not in serialized
    assert "[REDACTED]" in serialized and "[REDACTED PRIVATE KEY]" in serialized
    assert observed[0]["messages"][1]["recorded"]["count"] == 7
    assert observed[0]["messages"][1]["recorded"]["available"] is False
    assert messages == original
    assert result["usage"]["total_tokens"] == 15
    assert "usage" not in observed[0], "custom compatible APIs must not receive router-only request fields"


def test_json_mode_fallback_retries_only_redacted_messages(connection, monkeypatch):
    attempts = []
    def send(request, *, timeout):
        payload = json.loads(request.data)
        attempts.append(payload)
        assert connection not in json.dumps(payload)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, "invalid", {},
                                         io.BytesIO(("response_format rejected " + connection).encode()))
        return _success()
    monkeypatch.setattr(urllib.request, "urlopen", send)
    llm.chat([{"role": "user", "content": connection}], response_format={"type": "json_object"}, retries=1)
    assert len(attempts) == 2
    assert attempts[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in attempts[1]


@pytest.mark.parametrize("kind", ["http", "body", "network"])
def test_provider_failures_never_expose_echoed_credentials(connection, monkeypatch, kind):
    private = "provider-private-response-value"
    def send(request, *, timeout):
        if kind == "http":
            raise urllib.error.HTTPError(request.full_url, 403, "rejected", {}, io.BytesIO((private + connection).encode()))
        if kind == "body":
            return Reply({"error": {"code": 401, "message": private + connection}})
        raise urllib.error.URLError(private + connection)
    monkeypatch.setattr(urllib.request, "urlopen", send)
    with pytest.raises(llm.LLMError) as raised:
        llm.chat([{"role": "user", "content": "Public task"}], retries=1)
    assert connection not in str(raised.value) and private not in str(raised.value)
    if kind in {"http", "body"}:
        assert raised.value.status_code == (403 if kind == "http" else 401)


def test_authorization_is_not_forwarded_to_a_redirect_target(connection, monkeypatch):
    def send(request, *, timeout):
        redirect = urllib.request.HTTPRedirectHandler().redirect_request(
            request, None, 302, "Found", {}, "https://other.example/receive")
        assert request.get_header("Authorization") == "Bearer " + connection
        assert redirect.get_header("Authorization") is None
        assert connection not in repr(redirect.header_items())
        assert redirect.data is None
        return _success()
    monkeypatch.setattr(urllib.request, "urlopen", send)
    llm.chat([{"role": "user", "content": "Public task"}], retries=1)


def test_missing_key_and_credential_bearing_url_never_send_requests(connection, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete or credential-bearing connection reached the HTTP client")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", "")
    with pytest.raises(llm.LLMError, match="API key"):
        llm.chat([], retries=1)
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", connection)
    monkeypatch.setattr(llm.config, "OPENROUTER_BASE_URL", "https://user:password@example.test/v1")
    with pytest.raises(llm.LLMError, match="Base URL"):
        llm.chat([], retries=1)


def test_short_development_key_does_not_rewrite_http_protocol_fields(connection, monkeypatch):
    monkeypatch.setattr(llm.config, "OPENROUTER_API_KEY", "system")
    def send(request, *, timeout):
        payload = json.loads(request.data)
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "The configured key is [REDACTED]."
        assert payload["model"] == "system-model"
        return _success()
    monkeypatch.setattr(urllib.request, "urlopen", send)
    llm.chat([{"role": "system", "content": "The configured key is system."}], model="system-model", retries=1)
