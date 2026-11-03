"""Session settings must isolate connections and keep server credentials private."""

from __future__ import annotations

import dataclasses
import json
import os
from unittest.mock import patch

import pytest

from agent_trajviz.insight.llm_config import (
    AnalysisLLMConfig,
    AnalysisSettings,
    build_session_settings,
    config_status_html,
    nonsecret_settings,
    resolve_session_settings,
    setup_help_text,
)

ENV = {
    "ANALYZE_BASE_URL": "https://server.example/v1",
    "ANALYZE_API_KEY": "server-key-private-sentinel",
    "ANALYZE_MODEL": "server-model",
    "ANALYZE_PROVIDER": "openai",
    "ANALYZE_LANGUAGE": "German",
}


def values(**overrides):
    result = dict(base_url="https://server.example/v1", api_key="", model="session-model", provider="openai",
                  temperature=0.3, max_tokens=4096, timeout=90, language="French")
    result.update(overrides)
    return result


def test_environment_defaults_are_resolved_without_exposing_the_key():
    with patch.dict(os.environ, ENV, clear=True):
        settings = resolve_session_settings()
        defaults = nonsecret_settings()
    assert isinstance(settings, AnalysisSettings)
    assert settings.config.api_key == ENV["ANALYZE_API_KEY"]
    assert settings.language == "German"
    assert "api_key" not in defaults
    assert ENV["ANALYZE_API_KEY"] not in repr(defaults)
    assert ENV["ANALYZE_API_KEY"] not in repr(settings)
    assert defaults["base_url"] == ENV["ANALYZE_BASE_URL"]


def test_apply_is_session_scoped_and_does_not_mutate_the_environment():
    with patch.dict(os.environ, ENV, clear=True):
        before = dict(os.environ)
        first = build_session_settings(**values(api_key="first-session-key"))
        second = build_session_settings(**values(base_url="http://localhost:8080/v1", api_key="second-session-key"))
        assert dict(os.environ) == before
        assert resolve_session_settings(first) is first
        assert resolve_session_settings(second) is second
        assert resolve_session_settings().config.api_key == ENV["ANALYZE_API_KEY"]
    assert first.config.api_key == "first-session-key"
    assert second.config.api_key == "second-session-key"
    assert first.config.source == second.config.source == "session"


@pytest.mark.parametrize("previous_is_session", [False, True])
@pytest.mark.parametrize("edit", [{"base_url": "https://elsewhere.example/v1"}, {"provider": "anthropic"}])
def test_blank_key_never_moves_an_inherited_secret_to_a_changed_destination(previous_is_session, edit):
    with patch.dict(os.environ, ENV, clear=True):
        previous = build_session_settings(**values(api_key="session-key")) if previous_is_session else None
        changed = build_session_settings(**values(**edit), previous=previous)
    assert changed.config.api_key == ""
    assert not changed.config.ready


def test_blank_key_retains_same_destination_and_clear_explicitly_removes_it():
    with patch.dict(os.environ, ENV, clear=True):
        initial = build_session_settings(**values(base_url="https://server.example/v1/"))
        assert initial.config.api_key == ENV["ANALYZE_API_KEY"]
        changed = build_session_settings(**values(model="another-model", language="Japanese"), previous=initial)
        assert changed.config.api_key == ENV["ANALYZE_API_KEY"]
        cleared = build_session_settings(**values(api_key="ignored-new-key"), previous=changed, clear_api_key=True)
    assert cleared.config.api_key == ""
    assert not cleared.config.ready


def test_explicit_new_key_can_configure_a_new_destination():
    with patch.dict(os.environ, ENV, clear=True):
        settings = build_session_settings(**values(base_url="https://other.example/api", api_key="new-key",
                                                   provider="anthropic", temperature=None))
    assert settings.config.ready
    assert settings.config.api_key == "new-key"
    assert settings.config.temperature is None


def test_partial_settings_support_language_before_a_connection_is_configured():
    with patch.dict(os.environ, {}, clear=True):
        settings = build_session_settings(**values(base_url="", model="", api_key="", temperature=""))
    assert settings.language == "French"
    assert settings.config.temperature is None
    assert not settings.config.ready
    status = config_status_html(config=settings.config, loaded_steps=4)
    assert "Settings" in status and "Not configured" in status and "4 steps" in status


@pytest.mark.parametrize("url", [
    "https://name:private-password@example.test/v1",
    "https://example.test/v1?api_key=private-query",
    "https://example.test/v1#private-fragment",
    "https://example.test:bad-port/v1",
    "https://[broken/v1",
    "private-malformed-value",
])
def test_unsafe_environment_url_is_never_prefilled_in_the_browser(url):
    with patch.dict(os.environ, ENV | {"ANALYZE_BASE_URL": url}, clear=True):
        defaults = nonsecret_settings()
    assert defaults["base_url"] == ""
    assert "private" not in repr(defaults)
    assert "api_key" not in defaults


@pytest.mark.parametrize("url", [
    "file:///tmp/model", "ftp://example.test/v1", "example.test/v1", "https:///v1",
    "https://name:private@example.test/v1", "https://@example.test/v1",
    "https://example.test/v1?token=private", "https://example.test/v1#private",
    "https://example.test:0/v1", "https://example.test:99999/v1", "https://example.test:bad/v1",
    "https://[invalid/v1", "https://bad host/v1", "https://example.test/\nprivate",
])
def test_invalid_ui_endpoints_fail_without_echoing_input_or_secrets(url):
    with patch.dict(os.environ, ENV, clear=True), pytest.raises(ValueError) as exc:
        build_session_settings(**values(base_url=url, api_key="private-key"))
    assert "Base URL" in str(exc.value)
    assert "private" not in str(exc.value)
    assert ENV["ANALYZE_API_KEY"] not in str(exc.value)


def test_local_ipv6_endpoint_and_integral_numbers_are_supported():
    with patch.dict(os.environ, {}, clear=True):
        settings = build_session_settings(**values(base_url=" http://[::1]:8080/v1/ ", api_key="local-key",
                                                   max_tokens=512.0, timeout="30", temperature="0"))
    assert settings.config.base_url == "http://[::1]:8080/v1"
    assert settings.config.max_tokens == 512 and settings.config.timeout == 30
    assert settings.config.temperature == 0


@pytest.mark.parametrize("field", ["max_tokens", "timeout"])
@pytest.mark.parametrize("invalid", [None, True, 0, -1, 1.5, float("inf"), float("nan"), "", "1.5", "lots", 10**400])
def test_positive_integer_parameters_reject_invalid_values(field, invalid):
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="positive whole number"):
        build_session_settings(**values(**{field: invalid}))


@pytest.mark.parametrize("temperature", [True, "nope", float("inf"), float("nan"), -0.1, 2.1, 10**400])
def test_temperature_must_be_finite_and_within_protocol_bounds(temperature):
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="Temperature"):
        build_session_settings(**values(temperature=temperature))


def test_anthropic_temperature_has_its_own_protocol_range():
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="between 0 and 1"):
        build_session_settings(**values(provider="anthropic", temperature=1.1))


@pytest.mark.parametrize("edit", [
    {"provider": "unsupported"}, {"provider": None}, {"model": []}, {"language": ""},
    {"language": "English\nprivate"}, {"api_key": "key\nprivate"}, {"clear_api_key": "yes"},
])
def test_other_invalid_form_values_have_safe_validation_errors(edit):
    with patch.dict(os.environ, ENV, clear=True), pytest.raises(ValueError) as exc:
        build_session_settings(**values(**edit))
    assert "private" not in str(exc.value)
    assert ENV["ANALYZE_API_KEY"] not in str(exc.value)


def test_status_uses_explicit_config_and_does_not_expose_the_key():
    config = AnalysisLLMConfig(base_url="https://viewer:private-password@example.test:8443/private?key=private-query",
                               api_key="private-api-key", model="<model>", provider="openai", temperature=None,
                               max_tokens=42, timeout=12, source="session")
    with patch("agent_trajviz.insight.llm_config.resolve_analysis_config", side_effect=AssertionError("no env read")):
        status = config_status_html(config=config, loaded_steps=3)
    assert "UI settings" in status
    assert "example.test:8443" in status
    assert "&lt;model&gt;" in status
    assert "private" not in status
    assert "3 steps" in status


def test_invalid_state_fails_without_reading_environment():
    with (
        patch("agent_trajviz.insight.llm_config.resolve_analysis_config", side_effect=AssertionError("no env read")),
        pytest.raises(ValueError, match="Invalid settings state"),
    ):
        resolve_session_settings({})


def test_defaults_are_independent_mutable_snapshots():
    with patch.dict(os.environ, ENV, clear=True):
        settings = resolve_session_settings()
        defaults = nonsecret_settings(settings)
        defaults["base_url"] = "https://elsewhere.example/v1"
    assert settings.config.base_url == ENV["ANALYZE_BASE_URL"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.language = "Spanish"


def test_setup_help_includes_ui_settings_and_legacy_environment_names():
    with patch.dict(os.environ, {}, clear=True):
        help_text = setup_help_text()
    assert "Settings" in help_text
    assert "ANALYZE_API_KEY" in help_text and "LABEL_API_KEY" in help_text
    assert ".env" in help_text


@pytest.mark.parametrize("overrides", [
    {"ANALYZE_TEMPERATURE": "nan"}, {"ANALYZE_TEMPERATURE": "inf"}, {"ANALYZE_TEMPERATURE": "-inf"},
    {"ANALYZE_TEMPERATURE": "2.1"}, {"ANALYZE_TEMPERATURE": "-0.1"},
    {"ANALYZE_TEMPERATURE": "1.1", "ANALYZE_PROVIDER": "anthropic"},
])
def test_invalid_environment_temperature_has_finite_form_default_without_mutating_config(overrides):
    with patch.dict(os.environ, ENV | overrides, clear=True):
        config_before = resolve_session_settings().config
        defaults = nonsecret_settings()
        assert defaults["temperature"] == 0.2
        json.dumps(defaults, allow_nan=False)
        # The UI normalizes its defaults, not the launch environment.
        assert os.environ["ANALYZE_TEMPERATURE"] == overrides["ANALYZE_TEMPERATURE"]
        config_after = resolve_session_settings().config
        assert repr(config_before) == repr(config_after)


@pytest.mark.parametrize("field,default", [("MAX_TOKENS", 8192), ("TIMEOUT", 120)])
def test_overflow_environment_integers_have_valid_form_defaults(field, default):
    with patch.dict(os.environ, ENV | {"ANALYZE_" + field: str(10**400)}, clear=True):
        original = resolve_session_settings().config
        defaults = nonsecret_settings()
        assert defaults[field.lower()] == default
        assert getattr(original, field.lower()) == 10**400
        json.dumps(defaults, allow_nan=False)


def test_default_form_budget_leaves_room_for_reasoning():
    with patch.dict(os.environ, {}, clear=True):
        assert nonsecret_settings()["max_tokens"] == 8192


def test_max_output_tokens_field_explains_the_reasoning_budget():
    import gradio as gr

    from agent_trajviz.insight.ui import settings as settings_ui

    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANALYZE_", "LABEL_"))}
    with patch.dict(os.environ, env, clear=True), gr.Blocks(analytics_enabled=False):
        refs = settings_ui.layout()
    assert refs.max_tokens.value == 8192
    guidance = refs.max_tokens.info.lower()
    assert "hidden thinking" in guidance
    assert "empty" in guidance and "cut off" in guidance


def test_omitted_environment_temperature_stays_omitted_in_form_defaults():
    with patch.dict(os.environ, ENV | {"ANALYZE_TEMPERATURE": ""}, clear=True):
        defaults = nonsecret_settings()
    assert defaults["temperature"] is None


@pytest.mark.parametrize("env", [{}, {"ANALYZE_BASE_URL": "https://server.example/v1"},
                                 ENV | {"ANALYZE_API_KEY": ""}])
def test_all_not_ready_statuses_offer_settings(env):
    with patch.dict(os.environ, env, clear=True):
        status = config_status_html()
    assert "Settings" in status and "Not configured" in status
