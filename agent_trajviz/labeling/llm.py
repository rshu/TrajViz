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
) -> str:
    """Send one system + user message pair and return the model's text answer.

    *provider* is ``"openai"`` (any OpenAI-compatible ``/chat/completions``
    endpoint) or ``"anthropic"`` (the ``/messages`` API). Both messages are
    redacted first (see :mod:`agent_trajviz._redact`): step text can carry
    credentials the agent printed.
    """
    system_prompt, in_system = redact(system_prompt)
    user_message, in_user = redact(user_message)
    if in_system + in_user:
        logger.info("redacted %d credential-shaped string(s) from the LLM request", in_system + in_user)
    if provider == "anthropic":
        return _call_anthropic(base_url, api_key, model, system_prompt, user_message,
                               temperature, max_tokens, timeout)
    return _call_openai(base_url, api_key, model, system_prompt, user_message,
                        temperature, max_tokens, timeout)


def _call_openai(
    base_url: str, api_key: str, model: str, system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def _call_anthropic(
    base_url: str, api_key: str, model: str, system_prompt: str, user_message: str,
    temperature: float | None, max_tokens: int | None, timeout: int,
) -> str:
    import requests

    body: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens or 1024,
    }
    if temperature is not None:
        body["temperature"] = temperature
    resp = requests.post(
        f"{base_url.rstrip('/')}/messages",
        json=body,
        headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=timeout,
    )
    resp.raise_for_status()
    texts = [b["text"] for b in resp.json().get("content", []) if b.get("type") == "text"]
    if not texts:
        raise ValueError("Anthropic response contained no text blocks")
    return "\n".join(texts).strip()
