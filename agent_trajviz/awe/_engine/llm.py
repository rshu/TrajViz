"""Redacted OpenAI-compatible evaluation requests using only the standard library."""

from __future__ import annotations

import json
from http.client import IncompleteRead
import random
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from agent_trajviz._redact import PLACEHOLDER, redact

from . import config

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 524, 529}


def _backoff(attempt: int) -> None:
    """Jittered exponential backoff: N workers retrying a 429 with deterministic
    sleeps re-collide on every attempt."""
    time.sleep(min(2**attempt, 30) + random.random())


class LLMError(RuntimeError):
    """A safe connection failure, without provider bodies or credential values."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _outgoing(value, *, literal_secret=True):
    """Copy and redact every outbound text value, including structured prompts."""
    if isinstance(value, str):
        if literal_secret and config.OPENROUTER_API_KEY:
            value = value.replace(config.OPENROUTER_API_KEY, PLACEHOLDER)
        return redact(value)[0]
    if isinstance(value, dict):
        # Protocol field names and role/model identifiers are not trace text.
        # A short development key such as "system" must not rewrite the role.
        identifiers = {"role", "model", "type", "name", "id", "call_id", "tool_call_id"}
        return {key: _outgoing(item, literal_secret=key not in identifiers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_outgoing(item) for item in value]
    return value


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(_outgoing(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    # urllib forwards ordinary headers when following a 301/302/303 redirect.
    # Authorization belongs only to the explicitly configured endpoint.
    public_headers = {name: value for name, value in headers.items() if name.lower() != "authorization"}
    req = urllib.request.Request(url, data=data, headers=public_headers, method="POST")
    for name, value in headers.items():
        if name.lower() == "authorization":
            req.add_unredirected_header(name, value)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted URL)
        return json.loads(r.read().decode("utf-8"))


def chat(
    messages: list[dict],
    model: str | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 150.0,
    retries: int = 4,
) -> dict:
    """One chat completion. Returns ``{content, usage, model, raw}``.

    ``usage`` is OpenRouter's token accounting (prompt/completion/total, and a
    ``cost`` field in USD when the provider reports it). Raises ``LLMError`` on
    terminal failure after retries.
    """
    if not config.OPENROUTER_API_KEY:
        raise LLMError("Set the evaluation API key before requesting model assistance, or select mock mode.")
    model = model or config.JUDGE_MODEL
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": config.JUDGE_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.JUDGE_MAX_TOKENS,
    }
    endpoint = urlsplit(config.OPENROUTER_BASE_URL)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.netloc
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise LLMError("Evaluation Base URL must be an HTTP(S) endpoint without credentials, query or fragment.")
    # Other compatible APIs report token usage without this router-specific flag.
    if endpoint.hostname == "openrouter.ai":
        payload["usage"] = {"include": True}
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rshu/TrajViz",
        "X-Title": "TrajViz",
    }
    url = config.OPENROUTER_BASE_URL.rstrip("/") + "/chat/completions"

    last: LLMError | None = None
    attempt = 0
    while attempt < retries:
        try:
            resp = _post(url, payload, headers, timeout)
            # OpenRouter reports moderation/provider failures as HTTP 200 with an
            # `error` body and no `choices`. Treating that as content="" would let
            # the caller cache an all-`none` "no error" verdict permanently — fail
            # loudly instead (retrying transient provider codes).
            body_err = resp.get("error")
            if body_err or not resp.get("choices"):
                code = body_err.get("code") if isinstance(body_err, dict) else None
                status = code if type(code) is int else None
                last = LLMError(f"Evaluation provider error {status}." if status is not None
                                else "Evaluation provider returned no choices.", status_code=status)
                if attempt < retries - 1 and (code is None or code in _RETRY_STATUS):
                    _backoff(attempt)
                    attempt += 1
                    continue
                raise last from None
            choice = resp["choices"][0]
            content = ((choice.get("message") or {}).get("content")) or ""
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            return {
                "content": content,
                "usage": resp.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
                "model": resp.get("model", model),
                "raw": resp,
            }
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace") if e.fp else ""
            # Model rejected structured output -> drop it and retry WITHOUT consuming
            # an attempt, so the plain-mode fallback holds even on the last try.
            if e.code == 400 and response_format is not None and ("response_format" in body or "json" in body.lower()):
                payload.pop("response_format", None)
                response_format = None
                continue
            last = LLMError(f"Evaluation API returned HTTP {e.code}.", status_code=e.code)
            if e.code in _RETRY_STATUS and attempt < retries - 1:
                _backoff(attempt)
                attempt += 1
                continue
            raise last from None
        # OSError covers URLError, socket.timeout (a read timeout on 3.9 is NOT a
        # TimeoutError), ConnectionError, etc. — all must hit retry/backoff and be
        # wrapped as LLMError so the batch driver's `except LLMError` catches them.
        except (OSError, IncompleteRead, json.JSONDecodeError) as e:
            last = LLMError(f"Evaluation request failed ({type(e).__name__}).")
            if attempt < retries - 1:
                _backoff(attempt)
                attempt += 1
                continue
            raise last from None
    raise last or LLMError("Evaluation request failed after retries.")
