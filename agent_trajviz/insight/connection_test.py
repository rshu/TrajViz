"""Explicit, credential-safe connection checks without trajectory data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import requests
from requests.auth import AuthBase
from urllib3.exceptions import ReadTimeoutError

from .._endpoints import anthropic_messages_url
from .llm_config import AnalysisLLMConfig


@dataclass(frozen=True)
class ConnectionTestResult:
    ok: bool
    message: str


class _ExplicitCredentials(AuthBase):
    """Keep explicit headers authoritative without disabling environment proxies.

    A truthy auth handler prevents requests from replacing a Bearer header with
    a matching .netrc entry. The handler itself never changes the request.
    """

    def __call__(self, request):
        return request


_SUCCESS = "Connection successful. The API accepted the key and selected model."
_BUDGET_ACCEPTED = (
    "Connection successful. The API accepted the key and selected model; "
    "the small test token budget was exhausted."
)
_MALFORMED = (
    "The API returned an unexpected response format. Check Provider, Base URL and Model; "
    "the key could not be verified."
)
_EMPTY = (
    "The API responded without a test reply. The key could not be verified; "
    "check the selected model and provider."
)
_TIMEOUT = "The connection test timed out. Retry or check the endpoint; the key could not be verified."


def _http_failure(status: int) -> ConnectionTestResult:
    if 300 <= status < 400:
        message = "The API redirected the request. Enter the final Base URL; no redirect was followed."
    elif status == 401:
        message = "The API rejected the key (HTTP 401). Check the API key."
    elif status == 403:
        message = "Access denied (HTTP 403). Check the key's permissions and access to the selected model."
    elif status == 404:
        message = "Endpoint or model not found (HTTP 404). Check Base URL and Model."
    elif status == 429:
        message = "Rate limit or quota reached (HTTP 429). Retry later or check account limits; the key could not be verified."
    elif status == 402:
        message = "Billing or credits are required (HTTP 402). Check the provider account; the key could not be verified."
    elif status in {400, 422}:
        message = f"The API rejected the test request (HTTP {status}). Check Provider, Base URL and model compatibility."
    elif 500 <= status < 600:
        message = f"The provider returned a service error (HTTP {status}). Retry later; the key could not be verified."
    else:
        message = f"The API returned HTTP {status}. The key could not be verified."
    return ConnectionTestResult(False, message)


def _body_read_timed_out(exc: BaseException) -> bool:
    """Recognize the ConnectionError requests uses for a timed-out body read."""
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ReadTimeoutError):
            return True
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        pending.extend(link for link in (current.__cause__, current.__context__) if link is not None)
    return False


def _completion_result(data: object, provider: str) -> ConnectionTestResult:
    if not isinstance(data, dict) or data.get("error") is not None or data.get("type") == "error":
        return ConnectionTestResult(False, _MALFORMED)
    if provider == "anthropic":
        blocks = data.get("content")
        if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
            return ConnectionTestResult(False, _MALFORMED)
        texts = [block.get("text") for block in blocks if block.get("type") == "text"]
        if not all(isinstance(text, str) for text in texts):
            return ConnectionTestResult(False, _MALFORMED)
        budget_used = data.get("stop_reason") == "max_tokens"
        has_reply = any(text.strip() for text in texts)
    else:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ConnectionTestResult(False, _MALFORMED)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            return ConnectionTestResult(False, _MALFORMED)
        content = message["content"]
        if content is not None and not isinstance(content, str):
            return ConnectionTestResult(False, _MALFORMED)
        budget_used = choice.get("finish_reason") == "length"
        has_reply = bool(content and content.strip())
    if budget_used:
        return ConnectionTestResult(True, _BUDGET_ACCEPTED)
    if has_reply:
        return ConnectionTestResult(True, _SUCCESS)
    return ConnectionTestResult(False, _EMPTY)


def check_connection(config: AnalysisLLMConfig) -> ConnectionTestResult:
    """Make one small completion using an already validated connection.

    Tests only the supplied endpoint, model and credential. It never loads a
    trajectory, applies settings or persists a key. Environment proxies and TLS
    verification retain requests' defaults; redirects and retries are disabled.
    Public results contain static explanations, never provider response bodies.
    """
    if config.provider not in {"openai", "anthropic"}:
        return ConnectionTestResult(False, "Select OpenAI-compatible or Anthropic before testing.")
    if not config.ready:
        return ConnectionTestResult(False, "Enter Base URL, Model and API key before testing.")
    if (isinstance(config.timeout, bool) or not isinstance(config.timeout, (int, float))
            or not math.isfinite(config.timeout) or config.timeout <= 0):
        return ConnectionTestResult(False, "Enter a positive Timeout before testing.")
    body = {"model": config.model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 16}
    headers = {"Content-Type": "application/json"}
    if config.provider == "anthropic":
        url = anthropic_messages_url(config.base_url)
        headers.update({"x-api-key": config.api_key, "anthropic-version": "2023-06-01"})
    else:
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        with requests.Session() as session:
            session.auth = _ExplicitCredentials()
            response = session.post(url, json=body, headers=headers, timeout=min(config.timeout, 30),
                                    allow_redirects=False)
    except requests.Timeout:
        return ConnectionTestResult(False, _TIMEOUT)
    except requests.exceptions.SSLError:
        return ConnectionTestResult(False, "The TLS certificate could not be verified. Check Base URL and the server certificate.")
    except requests.RequestException as exc:
        if _body_read_timed_out(exc):
            return ConnectionTestResult(False, _TIMEOUT)
        return ConnectionTestResult(False, "Could not reach the API. Check Base URL and the network; the key could not be verified.")
    if not 200 <= response.status_code < 300:
        return _http_failure(response.status_code)
    try:
        data = response.json()
    except (ValueError, TypeError):
        return ConnectionTestResult(False, _MALFORMED)
    return _completion_result(data, config.provider)
