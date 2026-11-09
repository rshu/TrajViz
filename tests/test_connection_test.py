"""Connection checks use explicit credentials and never send trajectory data."""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from agent_trajviz.insight.assistant import complete_chat
from agent_trajviz.insight.connection_test import ConnectionTestResult, check_connection
from agent_trajviz.insight.llm_config import AnalysisLLMConfig, build_session_settings, resolve_analysis_config
from agent_trajviz.labeling.llm import call_llm


_KEY = "private-probe-key-not-real"
_MODEL = "connection-test-model"
_OPENAI = {"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]}
_ANTHROPIC = {"type": "message", "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn"}


def _config(base_url="https://test.example/v1", **kwargs):
    return dataclasses.replace(AnalysisLLMConfig(
        base_url=base_url, api_key=_KEY, model=_MODEL, provider="openai",
        temperature=0.8, max_tokens=8192, timeout=120, source="session",
    ), **kwargs)


@pytest.fixture
def endpoint(monkeypatch):
    # The fixture must never use an inherited corporate proxy for loopback data.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = {"calls": [], "status": 200, "payload": _OPENAI, "headers": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["calls"].append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
            payload = state.get("body", json.dumps(state["payload"]).encode())
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in state["headers"].items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            state["calls"].append({"path": self.path, "method": "GET"})
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("provider,suffix,path,payload", [
    ("openai", "/v1/", "/v1/chat/completions", _OPENAI),
    ("anthropic", "", "/v1/messages", _ANTHROPIC),
    ("anthropic", "/v1/", "/v1/messages", _ANTHROPIC),
    ("anthropic", "/gateway/anthropic/v1", "/gateway/anthropic/v1/messages", _ANTHROPIC),
])
def test_real_http_probe_uses_current_model_key_and_only_a_static_prompt(endpoint, provider, suffix, path, payload):
    base, state = endpoint
    state["payload"] = payload
    config = _config(base + suffix, provider=provider)
    before = dataclasses.asdict(config)
    result = check_connection(config)
    assert result.ok is True
    assert "accepted the key and selected model" in result.message
    assert len(state["calls"]) == 1
    call = state["calls"][0]
    assert call["path"] == path
    assert call["body"] == {"model": _MODEL, "max_tokens": 16,
                            "messages": [{"role": "user", "content": "Reply OK."}]}
    if provider == "anthropic":
        assert call["headers"]["x-api-key"] == _KEY
        assert call["headers"]["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in call["headers"]
    else:
        assert call["headers"]["Authorization"] == f"Bearer {_KEY}"
        assert "x-api-key" not in call["headers"]
    assert dataclasses.asdict(config) == before
    assert _KEY not in repr(result)
    assert base not in result.message


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}),
    ("openai", {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}),
    ("anthropic", {"content": [], "stop_reason": "max_tokens"}),
    ("anthropic", {"content": [{"type": "thinking", "thinking": "internal"}], "stop_reason": "max_tokens"}),
])
def test_reasoning_token_budget_stop_is_accepted_without_a_visible_reply(endpoint, provider, payload):
    base, state = endpoint
    state["payload"] = payload
    result = check_connection(_config(base, provider=provider))
    assert result.ok is True
    assert "token budget" in result.message


@pytest.mark.parametrize("status,explanation", [
    (400, "model compatibility"), (401, "rejected the key"), (402, "Billing or credits"),
    (403, "permissions"), (404, "Base URL and Model"), (418, "HTTP 418"),
    (422, "model compatibility"), (429, "Rate limit or quota"),
    (500, "service error"), (503, "service error"),
])
def test_http_failures_have_specific_public_messages_without_provider_bodies(endpoint, status, explanation):
    base, state = endpoint
    state.update(status=status, payload={"error": {"message": f"private-provider-body {base} {_KEY}"}})
    result = check_connection(_config(base))
    assert result.ok is False
    assert explanation in result.message
    assert str(status) in result.message
    assert _KEY not in result.message
    assert base not in result.message
    assert "private-provider-body" not in result.message
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("status", [302, 307, 308])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_redirect_is_not_followed_or_given_another_credentialed_request(endpoint, status, provider):
    base, state = endpoint
    state.update(status=status, headers={"Location": base + "/credential-sink"})
    result = check_connection(_config(base, provider=provider))
    assert result.ok is False
    assert "no redirect was followed" in result.message
    assert len(state["calls"]) == 1
    assert state["calls"][0]["path"] != "/credential-sink"


@pytest.mark.parametrize("provider,payload", [
    ("openai", []), ("openai", None), ("openai", {}),
    ("openai", {"choices": []}), ("openai", {"choices": [None]}),
    ("openai", {"choices": [{"message": {}}]}),
    ("openai", {"choices": [{"message": {"content": []}, "finish_reason": "length"}]}),
    ("openai", {"error": {"message": _KEY}, **_OPENAI}),
    ("anthropic", {}), ("anthropic", {"content": None, "stop_reason": "max_tokens"}),
    ("anthropic", {"content": [1], "stop_reason": "max_tokens"}),
    ("anthropic", {"content": [{"type": "text", "text": 1}], "stop_reason": "max_tokens"}),
    ("anthropic", {"type": "error", **_ANTHROPIC, "error": {"message": _KEY}}),
])
def test_malformed_or_error_envelopes_never_report_key_success(endpoint, provider, payload):
    base, state = endpoint
    state["payload"] = payload
    result = check_connection(_config(base, provider=provider))
    assert result.ok is False
    assert "unexpected response format" in result.message
    assert _KEY not in result.message


@pytest.mark.parametrize("provider,payload", [
    ("openai", {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}),
    ("openai", {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]}),
    ("anthropic", {"content": [], "stop_reason": "end_turn"}),
])
def test_empty_reply_does_not_misidentify_a_key_as_invalid_or_verified(endpoint, provider, payload):
    base, state = endpoint
    state["payload"] = payload
    result = check_connection(_config(base, provider=provider))
    assert result.ok is False
    assert "without a test reply" in result.message
    assert "could not be verified" in result.message


def test_non_json_response_does_not_leak_body_or_claim_success(endpoint):
    base, state = endpoint
    state["body"] = f"<html>private-provider-body {_KEY}</html>".encode()
    result = check_connection(_config(base))
    assert result.ok is False
    assert "unexpected response format" in result.message
    assert _KEY not in result.message


@pytest.mark.parametrize("exception,explanation", [
    (requests.Timeout("private-exception " + _KEY), "timed out"),
    (requests.ConnectionError("private-exception " + _KEY), "Could not reach"),
    (requests.exceptions.SSLError("private-exception " + _KEY), "TLS certificate"),
    (requests.ConnectionError(ReadTimeoutError(None, None, "private-exception " + _KEY)), "timed out"),
])
def test_transport_errors_are_sanitized_and_never_claim_the_key_is_bad(exception, explanation):
    with patch.object(requests.Session, "post", side_effect=exception) as post:
        result = check_connection(_config())
    assert result.ok is False
    assert explanation in result.message
    assert _KEY not in result.message
    assert "private-exception" not in result.message
    post.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"api_key": ""}, {"model": ""}, {"base_url": ""}, {"provider": "unsupported"},
    {"timeout": 0}, {"timeout": -1}, {"timeout": float("inf")}, {"timeout": True},
])
def test_missing_or_invalid_configuration_does_not_make_a_request(changes):
    with patch.object(requests.Session, "post") as post:
        result = check_connection(_config(**changes))
    assert result.ok is False
    post.assert_not_called()
    assert _KEY not in result.message


@pytest.mark.parametrize("timeout,expected", [(120, 30), (7, 7)])
@pytest.mark.parametrize("provider,payload", [("openai", _OPENAI), ("anthropic", _ANTHROPIC)])
def test_explicit_key_wins_over_netrc_while_environment_proxy_and_tls_are_preserved(
    monkeypatch, tmp_path, timeout, expected, provider, payload,
):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine test.example login private-netrc-user password private-netrc-password\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.example.test:8080")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    captured = []

    def send(session, request, **kwargs):
        captured.append((session, request, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    with patch.object(requests.Session, "send", send), patch(
        "requests.sessions.get_netrc_auth", side_effect=AssertionError("must not consult netrc"),
    ):
        result = check_connection(_config(provider=provider, timeout=timeout))
    assert result.ok is True
    assert len(captured) == 1
    session, request, options = captured[0]
    assert session.trust_env is True
    assert options["proxies"]["https"] == "http://proxy.example.test:8080"
    assert options["verify"] is True
    assert options["timeout"] == expected
    assert options["allow_redirects"] is False
    assert session.get_adapter(request.url).max_retries.total == 0
    assert "private-netrc" not in repr(dict(request.headers))
    assert request.headers["Authorization" if provider == "openai" else "x-api-key"] == (
        f"Bearer {_KEY}" if provider == "openai" else _KEY
    )


def test_result_is_a_small_immutable_public_value():
    result = ConnectionTestResult(True, "Connection successful.")
    assert dataclasses.asdict(result) == {"ok": True, "message": "Connection successful."}
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ok = False


def test_gateway_headers_are_bound_to_the_connection_through_test_and_analysis(endpoint, monkeypatch):
    base_url, state = endpoint
    gateway = f"{base_url}/gateway/v1"
    extra = '{"X-Gateway-Key":"synthetic-gateway-secret","authorization":"ignored","content-type":"ignored"}'
    for name, value in {
        "ANALYZE_BASE_URL": gateway + "/",
        "ANALYZE_PROVIDER": "openai",
        "ANALYZE_API_KEY": _KEY,
        "ANALYZE_MODEL": _MODEL,
        "ANALYZE_HEADERS": extra,
        "LABEL_BASE_URL": gateway,
        "LABEL_PROVIDER": "openai",
        "LABEL_HEADERS": '{"X-Label-Key":"synthetic-label-secret"}',
    }.items():
        monkeypatch.setenv(name, value)

    def settings(url, provider="openai"):
        return build_session_settings(base_url=url, api_key=_KEY, model=_MODEL, provider=provider,
                                      temperature=None, max_tokens=16, timeout=5, language="English").config

    def request(config):
        assert check_connection(config).ok
        assert complete_chat(config, "system", [{"role": "user", "content": "synthetic test"}]) == "OK"
        return [requests.structures.CaseInsensitiveDict(call["headers"]) for call in state["calls"][-2:]]

    for headers in request(settings(gateway)):
        assert headers["X-Gateway-Key"] == "synthetic-gateway-secret"
        assert headers["Authorization"] == f"Bearer {_KEY}"
        assert headers["Content-Type"] == "application/json"
        assert "X-Label-Key" not in headers

    other_url = f"{base_url}/other/v1"
    for headers in request(settings(other_url)):
        assert "X-Gateway-Key" not in headers and "X-Label-Key" not in headers

    state["payload"] = _ANTHROPIC
    for headers in request(settings(gateway, "anthropic")):
        assert "X-Gateway-Key" not in headers and "X-Label-Key" not in headers
        assert headers["x-api-key"] == _KEY

    monkeypatch.delenv("ANALYZE_HEADERS")
    assert resolve_analysis_config().extra_headers == {"X-Label-Key": "synthetic-label-secret"}
    monkeypatch.setenv("ANALYZE_BASE_URL", other_url)
    assert resolve_analysis_config().extra_headers == {}
    state["payload"] = _OPENAI
    assert call_llm(other_url, _KEY, _MODEL, "system", "user") == "OK"
    assert "X-Label-Key" not in state["calls"][-1]["headers"]


def test_model_completions_do_not_forward_gateway_headers_on_redirect(endpoint):
    base_url, state = endpoint
    state.update(status=307, headers={"Location": f"{base_url}/redirected"})
    config = _config(base_url, extra_headers={"X-Gateway-Key": "synthetic-gateway-secret"})
    with pytest.raises(requests.HTTPError, match="returned a redirect"):
        complete_chat(config, "system", [{"role": "user", "content": "synthetic test"}])
    with pytest.raises(requests.HTTPError, match="returned a redirect"):
        call_llm(base_url, _KEY, _MODEL, "system", "user", headers=config.extra_headers)
    assert [call["path"] for call in state["calls"]] == ["/chat/completions", "/chat/completions"]
