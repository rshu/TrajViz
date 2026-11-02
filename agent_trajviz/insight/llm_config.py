"""LLM settings for the dashboard analysis panel.

Reads ``ANALYZE_*`` from ``.env`` (fill-only, never overwrites a real
environment variable). Missing ``ANALYZE_*`` values fall back to the
``LABEL_*`` keys used by the step labeler (``agent-trajviz-label``) so one file can
drive both the labeler and the analysis chat.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


# The dashboard's .env configures the analysis LLM (ANALYZE_*) and shares the
# step labeler's settings (LABEL_*); that is its documented purpose (README,
# .env.example). It is read from the launch directory, so accepting any key let
# a .env in an untrusted folder set AWE_DECAF_PATH (which decides the code the
# Attribution tab imports) or anything else the process reads.
_DOTENV_PREFIXES = ("ANALYZE_", "LABEL_")
_DOTENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not (_DOTENV_KEY.match(key) and key.startswith(_DOTENV_PREFIXES)):
                continue
            value = _clean_env_scalar(value)
            if key not in os.environ:
                os.environ[key] = value


def load_env_files() -> None:
    """Load CWD ``.env`` then the repo-root ``.env`` (fill-only).

    Call from the dashboard entry / ``build_ui``, not at import time, so
    test collection does not mutate the process environment.
    """
    _load_dotenv(".env")
    # The repo-root .env exists only in a source checkout. In an installed
    # wheel parents[2] of this module is site-packages, so require the
    # checkout's pyproject.toml before reading anything there.
    if (_REPO_ROOT / "pyproject.toml").is_file():
        _load_dotenv(str(_REPO_ROOT / ".env"))


# Reply language of the AI sidebar and the Issues judge: English unless
# ANALYZE_LANGUAGE (any language name) says otherwise.
DEFAULT_ANALYSIS_LANGUAGE = "English"


def analysis_language() -> str:
    """The language the LLM features reply in."""
    return _env("ANALYZE_LANGUAGE") or DEFAULT_ANALYSIS_LANGUAGE


def is_chinese(language: str) -> bool:
    """True when *language* names Chinese, which has prompts of its own (Chinese headings)."""
    return language.strip().lower() in {"simplified chinese", "chinese", "zh", "zh-cn", "zh-hans", "简体中文", "中文"}


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _clean_env_scalar(raw)


def _env_float(name: str, default: float) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _clean_env_scalar(raw)
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = _clean_env_scalar(raw)
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AnalysisLLMConfig:
    """Resolved OpenAI-compatible or Anthropic chat settings."""

    base_url: str
    # Out of the repr, so printing or logging a config never shows the key.
    api_key: str = field(repr=False)
    model: str
    provider: str
    temperature: float | None
    max_tokens: int
    timeout: int
    source: str  # "analyze" | "label" | "mixed" | "missing" | "session"

    @property
    def ready(self) -> bool:
        """True when the endpoint, key and model are all set."""
        return bool(self.base_url and self.api_key and self.model)

    @property
    def missing(self) -> list[str]:
        """The environment variables still to set before :attr:`ready` (names only, never values)."""
        names: list[str] = []
        if not self.base_url:
            names.append("ANALYZE_BASE_URL (or LABEL_BASE_URL)")
        if not self.api_key:
            names.append("ANALYZE_API_KEY (or LABEL_API_KEY)")
        if not self.model:
            names.append("ANALYZE_MODEL (or LABEL_MODEL)")
        return names


def _pick(analyze_name: str, label_name: str, default: str = "") -> tuple[str, str]:
    """Return (value, which-prefix-won) for an ANALYZE_/LABEL_ pair."""
    analyze = _env(analyze_name)
    if analyze:
        return analyze, "analyze"
    label = _env(label_name)
    if label:
        return label, "label"
    return default, "missing"


def resolve_analysis_config() -> AnalysisLLMConfig:
    """Resolve chat settings. ``ANALYZE_*`` wins; ``LABEL_*`` fills gaps."""
    base_url, url_src = _pick("ANALYZE_BASE_URL", "LABEL_BASE_URL")
    api_key, key_src = _pick("ANALYZE_API_KEY", "LABEL_API_KEY")
    model, model_src = _pick("ANALYZE_MODEL", "LABEL_MODEL")
    provider, prov_src = _pick("ANALYZE_PROVIDER", "LABEL_PROVIDER", "openai")
    if not provider:
        provider = "openai"
    provider = provider.strip().lower()
    if provider not in {"openai", "anthropic"}:
        provider = "openai"

    sources = {url_src, key_src, model_src, prov_src} - {"missing"}
    if not (base_url and api_key and model):
        source = "missing"
    elif sources == {"analyze"}:
        source = "analyze"
    elif sources == {"label"}:
        source = "label"
    else:
        source = "mixed"

    if os.getenv("ANALYZE_TEMPERATURE") is not None:
        temperature = _env_float("ANALYZE_TEMPERATURE", 0.2)
    elif os.getenv("LABEL_TEMPERATURE") is not None:
        temperature = _env_float("LABEL_TEMPERATURE", 0.2)
    else:
        temperature = 0.2
    # Reasoning models spend part of the budget on hidden thinking before the visible
    # reply; at 2048 most first passes came back empty or cut off.
    max_tokens = _env_int("ANALYZE_MAX_TOKENS", 0) or _env_int("LABEL_MAX_TOKENS", 8192)
    if max_tokens <= 0:
        max_tokens = 8192
    timeout = _env_int("ANALYZE_TIMEOUT", 120)
    if timeout <= 0:
        timeout = 120

    return AnalysisLLMConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        source=source,
    )



@dataclass(frozen=True)
class AnalysisSettings:
    """A browser session's model connection and reply language.

    Keep this object in server-side session state. It must not be persisted in
    browser storage or returned as a component value: ``config`` holds the key.
    """

    config: AnalysisLLMConfig
    language: str


def resolve_session_settings(settings: AnalysisSettings | None = None) -> AnalysisSettings:
    """Return session overrides, or the launch environment's defaults."""
    if settings is None:
        return AnalysisSettings(resolve_analysis_config(), analysis_language())
    if not isinstance(settings, AnalysisSettings):
        raise ValueError("Invalid settings state. Reset settings and try again.")
    return settings


def _validated_base_url(value: str) -> str:
    """Validate an optional endpoint without echoing credentials in errors."""
    value = value.strip().rstrip("/")
    if not value:
        return ""
    message = "Base URL must be an HTTP or HTTPS endpoint without credentials, query parameters or fragments."
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError(message)
    try:
        parsed = urlparse(value)
        port = parsed.port
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or "?" in value or "#" in value or port == 0):
            raise ValueError(message)
    except ValueError:
        raise ValueError(message) from None
    return value


def nonsecret_settings(settings: AnalysisSettings | None = None) -> dict[str, object]:
    """Form defaults that never expose a key or credentials embedded in a URL."""
    resolved = resolve_session_settings(settings)
    cfg = resolved.config
    try:
        base_url = _validated_base_url(cfg.base_url)
    except ValueError:
        base_url = ""
    # The environment resolver preserves historical semantics; form defaults
    # must additionally be finite JSON values accepted by the Apply validator.
    temperature = cfg.temperature
    if temperature is not None:
        maximum = 1 if cfg.provider == "anthropic" else 2
        if isinstance(temperature, bool) or not math.isfinite(temperature) or not 0 <= temperature <= maximum:
            temperature = 0.2
    try:
        max_tokens = _positive_integer(cfg.max_tokens, "Maximum output tokens")
    except ValueError:
        max_tokens = 8192
    try:
        timeout = _positive_integer(cfg.timeout, "Timeout")
    except ValueError:
        timeout = 120
    return {
        "provider": cfg.provider,
        "base_url": base_url,
        "model": cfg.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "language": resolved.language,
    }


def _settings_text(value: object, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    value = value.strip()
    if (not allow_empty and not value) or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{label} must be nonempty text without control characters.")
    return value


def _positive_integer(value: object, label: str) -> int:
    message = f"{label} must be a positive whole number."
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(message)
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError:
            raise ValueError(message) from None
    else:
        raise ValueError(message)
    if result <= 0 or result > 2**31 - 1:
        raise ValueError(message)
    return result


def build_session_settings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    temperature: float | str | None,
    max_tokens: int | float | str,
    timeout: int | float | str,
    language: str,
    previous: AnalysisSettings | None = None,
    clear_api_key: bool = False,
) -> AnalysisSettings:
    """Validate form values without changing process environment or files.

    An empty key field preserves a configured key only at the same endpoint
    and provider. Changing either destination requires entering a new key;
    otherwise the result is not ready. ``clear_api_key`` explicitly removes it.
    Empty endpoint/model values are useful for configuring language and request
    options before a connection is available.
    """
    previous_config = resolve_session_settings(previous).config
    endpoint = _validated_base_url(_settings_text(base_url, "Base URL"))
    model = _settings_text(model, "Model")
    provider = _settings_text(provider, "Provider", allow_empty=False).lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError("Provider must be OpenAI-compatible or Anthropic.")
    language = _settings_text(language, "Reply language", allow_empty=False)
    api_key = _settings_text(api_key, "API key")
    if not isinstance(clear_api_key, bool):
        raise ValueError("Clear API key must be a checkbox value.")
    if clear_api_key:
        api_key = ""
    elif not api_key and (endpoint == previous_config.base_url.rstrip("/")
                          and provider == previous_config.provider):
        api_key = previous_config.api_key

    if temperature is None or (isinstance(temperature, str) and not temperature.strip()):
        parsed_temperature = None
    else:
        message = "Temperature must be a finite number, or blank to omit it."
        if isinstance(temperature, bool):
            raise ValueError(message)
        try:
            parsed_temperature = float(temperature)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(message) from None
        maximum = 1 if provider == "anthropic" else 2
        if not math.isfinite(parsed_temperature) or not 0 <= parsed_temperature <= maximum:
            raise ValueError(f"Temperature must be between 0 and {maximum} for this provider, or blank to omit it.")

    config = AnalysisLLMConfig(
        base_url=endpoint,
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=parsed_temperature,
        max_tokens=_positive_integer(max_tokens, "Maximum output tokens"),
        timeout=_positive_integer(timeout, "Timeout"),
        source="session",
    )
    return AnalysisSettings(config=config, language=language)


def config_status_html(*, loaded_steps: int | None = None, config: AnalysisLLMConfig | None = None) -> str:
    """Short HTML status for the analysis sidebar (never includes the key)."""
    cfg = config if config is not None else resolve_analysis_config()
    bits: list[str] = []
    if cfg.ready:
        # Name where the trajectory brief goes: the base URL comes from the
        # environment or .env and can point anywhere, so the host is the egress
        # the user is consenting to.
        # netloc includes URL userinfo; never fall back to the raw URL, whose
        # path/query (or malformed authority) can also contain credentials.
        try:
            endpoint = urlparse(cfg.base_url)
            hostname = endpoint.hostname
            port = endpoint.port
            host = f"[{hostname}]" if hostname and ":" in hostname else hostname
            if host and port is not None:
                host = f"{host}:{port}"
        except ValueError:
            host = None
        host = host or "(invalid endpoint)"
        bits.append(
            f"<span class='analysis-status-ok'>Ready</span> · "
            f"<code>{_escape(cfg.model)}</code> · {_escape(cfg.provider)} · "
            f"sends to <code>{_escape(host)}</code>"
        )
        if cfg.source == "session":
            bits.append("UI settings")
        elif cfg.source == "label":
            bits.append("using LABEL_*")
        elif cfg.source == "mixed":
            bits.append("ANALYZE_* + LABEL_*")
        else:
            bits.append("ANALYZE_*")
    elif cfg.source == "session":
        missing_fields = [label for label, value in (("base URL", cfg.base_url), ("API key", cfg.api_key),
                                                     ("model", cfg.model)) if not value]
        bits.append(
            "<span class='analysis-status-warn'>Not configured</span> · "
            f"set {_escape(', '.join(missing_fields))} in Settings"
        )
    elif len(cfg.missing) >= 3:
        bits.append(
            "<span class='analysis-status-warn'>Not configured</span> · "
            "open Settings, or set <code>ANALYZE_*</code> (or <code>LABEL_*</code>) in the environment "
            "or a <code>.env</code> file"
        )
    else:
        missing = ", ".join(cfg.missing)
        bits.append(
            f"<span class='analysis-status-warn'>Not configured</span> · "
            f"open Settings, or set {_escape(missing)} in <code>.env</code>"
        )
    if loaded_steps:
        bits.append(f"{loaded_steps} steps in context")
    elif loaded_steps == 0:
        bits.append("load a trajectory first")
    return "<div class='analysis-status'>" + " · ".join(bits) + "</div>"


def setup_help_text(cfg: AnalysisLLMConfig | None = None) -> str:
    """User-facing message when the panel cannot call a model."""
    cfg = cfg or resolve_analysis_config()
    missing = ", ".join(cfg.missing) if cfg.missing else "ANALYZE_BASE_URL, ANALYZE_API_KEY, ANALYZE_MODEL"
    return (
        "The analysis panel is not configured. Open Settings to configure the connection for this session.\n\n"
        f"Set {missing} in the environment, or in a `.env` file in the directory "
        "TrajViz was launched from (`.env.example` in the source repository lists "
        "every option).\n\n"
        "`LABEL_*` from the step labeler is used when `ANALYZE_*` is omitted."
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
