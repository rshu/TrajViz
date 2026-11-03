"""Configuration and HTTP calls for the labeling model (needs the ``llm`` extra).

Configuration comes from command-line flags, then ``LABEL_*`` environment
variables, then a ``.env`` file in the working directory (and, in a source
checkout, the repository root). ``requests`` is imported only when a call is
made, so the rest of the package works without the extra.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from .._endpoints import anthropic_messages_url
from .._llm_http import env_json_dict, merge_llm_headers
from .._redact import redact

logger = logging.getLogger(__name__)

# Match the dashboard's dotenv boundary: a trajectory directory must not
# configure unrelated process settings such as proxy or import locations.
_DOTENV_PREFIXES = ("ANALYZE_", "LABEL_")
_DOTENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _load_dotenv(path: str = ".env") -> None:
    """Fill unset LLM settings from a ``KEY=value`` file, if it exists."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not (_DOTENV_KEY.match(key) and key.startswith(_DOTENV_PREFIXES)):
                continue
            if key not in os.environ:
                os.environ[key] = _clean_env_scalar(value)


def load_env_files() -> None:
    """Load ``./.env`` and, in a source checkout, the repository's ``.env`` (fill-only).

    Called by the command-line entry point, never at import, so importing the
    package does not change the process environment.
    """
    _load_dotenv()
    # An installed wheel has no repository root: there, parents[2] of this module
    # is site-packages, so require the checkout's pyproject.toml first.
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        _load_dotenv(str(repo_root / ".env"))


def env_float(name: str, default: float) -> float:
    """A float from the environment, or *default* when unset, empty or invalid."""
    raw = os.getenv(name)
    if raw is None or _clean_env_scalar(raw) == "":
        return default
    try:
        return float(_clean_env_scalar(raw))
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


def env_int(name: str, default: int) -> int:
    """An int from the environment, or *default* when unset, empty or invalid."""
    raw = os.getenv(name)
    if raw is None or _clean_env_scalar(raw) == "":
        return default
    try:
        return int(_clean_env_scalar(raw))
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


class LabelBudgetExhausted(RuntimeError):
    """The model reached the output token limit before writing any answer text.

    Reasoning models spend the budget on hidden thinking first, so a small limit
    yields an empty reply. Raising instead of returning ``""`` lets the labeler
    retry the step and report it, rather than file it silently as unknown.
    ``limit`` is the ``max_tokens`` the request carried (``None`` when the request
    left the limit to the provider).
    """

    def __init__(self, limit: int | None) -> None:
        super().__init__("model hit the output token limit")
        self.limit = limit


def call_llm(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 120,
    headers: dict[str, str] | None = None,
) -> str:
    """Send one system + user message pair and return the model's text answer.

    *provider* is ``"openai"`` (any OpenAI-compatible ``/chat/completions``
    endpoint) or ``"anthropic"`` (the ``/v1/messages`` API). Both messages are
    redacted first (see :mod:`agent_trajviz._redact`): step text can carry
    credentials the agent printed. Raises :class:`LabelBudgetExhausted` when the
    model reached the token limit before writing any answer text.

    *headers* are extra gateway HTTP headers. When omitted, ``LABEL_HEADERS``
    from the environment is used.
    """
    system_prompt, in_system = redact(system_prompt)
    user_message, in_user = redact(user_message)
    if in_system + in_user:
        logger.info("redacted %d credential-shaped string(s) from the LLM request", in_system + in_user)
    extra = headers if headers is not None else env_json_dict("LABEL_HEADERS")
    if provider == "anthropic":
        return _call_anthropic(base_url, api_key, model, system_prompt, user_message,
                               temperature, max_tokens, timeout, extra)
    return _call_openai(base_url, api_key, model, system_prompt, user_message,
                        temperature, max_tokens, timeout, extra)


def _call_openai(
    base_url: str, api_key: str, model: str, system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
    extra_headers: dict[str, str],
) -> str:
    import requests

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body,
        headers=merge_llm_headers(extra_headers, auth={"Authorization": f"Bearer {api_key}"}),
        timeout=timeout,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    content = choice["message"]["content"]
    if content is not None and not isinstance(content, str):
        raise ValueError("OpenAI response content is not text")
    text = (content or "").strip()
    if not text and choice.get("finish_reason") == "length":
        raise LabelBudgetExhausted(max_tokens)
    return text


def _call_anthropic(
    base_url: str, api_key: str, model: str, system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
    extra_headers: dict[str, str],
) -> str:
    import requests

    body: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens or 4096,
    }
    if temperature is not None:
        body["temperature"] = temperature
    resp = requests.post(
        anthropic_messages_url(base_url),
        json=body,
        headers=merge_llm_headers(
            extra_headers,
            auth={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        ),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    text = "\n".join(texts).strip()
    if not text and data.get("stop_reason") == "max_tokens":
        raise LabelBudgetExhausted(body["max_tokens"])
    if not texts:
        raise ValueError("Anthropic response contained no text blocks")
    return text
