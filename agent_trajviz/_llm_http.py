"""Shared helpers for LLM HTTP requests (env JSON headers, auth merge)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from requests import Response

logger = logging.getLogger(__name__)


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def env_json_dict(name: str) -> dict[str, str]:
    """Parse a JSON object env var into string headers (empty on missing/invalid)."""
    raw = os.getenv(name)
    if raw is None:
        return {}
    value = _clean_env_scalar(raw)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        logger.warning("invalid JSON in %s; ignoring", name)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("%s must be a JSON object; ignoring", name)
        return {}
    return {
        str(key): str(val)
        for key, val in parsed.items()
        if isinstance(val, (str, int, float, bool))
    }


def env_headers_for_connection(prefix: str, base_url: str, provider: str) -> dict[str, str]:
    """Read gateway credentials only for their configured destination."""
    configured_url = _clean_env_scalar(os.getenv(f"{prefix}_BASE_URL", "")).rstrip("/")
    configured_provider = _clean_env_scalar(os.getenv(f"{prefix}_PROVIDER", "")).lower() or "openai"
    if configured_url and configured_url == base_url.rstrip("/") and configured_provider == provider:
        return env_json_dict(f"{prefix}_HEADERS")
    return {}


def merge_llm_headers(
    extra: dict[str, str] | None = None,
    *,
    auth: dict[str, str],
) -> dict[str, str]:
    """Content-Type, optional gateway extras, then *auth* (auth wins on conflict)."""
    required = {"Content-Type": "application/json", **auth}
    reserved = {name.lower() for name in required}
    headers = {name: value for name, value in (extra or {}).items() if name.lower() not in reserved}
    headers.update(required)
    return headers


def raise_for_llm_status(response: Response) -> None:
    """Treat an unfollowed redirect as a failure, even if it has a JSON body."""
    if getattr(response, "is_redirect", False) is True:
        from requests import HTTPError

        raise HTTPError("Model endpoint returned a redirect.", response=response)
    response.raise_for_status()
