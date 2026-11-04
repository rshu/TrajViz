"""Session settings reach every model feature without changing process config."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import pytest

import agent_trajviz as tv
from agent_trajviz.insight import assistant, issue_judge
from agent_trajviz.insight.insight import build_ui
from agent_trajviz.insight.llm_config import AnalysisLLMConfig, AnalysisSettings
from agent_trajviz.insight.presenters.issues import OverviewIssue
from agent_trajviz.insight.ui import overview_tab, sidebar


def _settings(*, model="session-model", language="French", api_key="private-session-key"):
    return AnalysisSettings(
        config=AnalysisLLMConfig(
            base_url="http://127.0.0.1:18999/v1", api_key=api_key,
            model=model, provider="anthropic", temperature=None,
            max_tokens=731, timeout=19, source="session",
        ),
        language=language,
    )


@pytest.fixture(scope="module")
def handlers():
    # This test uses explicit fixture settings and must not load developer keys.
    with patch("agent_trajviz.insight.insight.load_env_files"):
        app = build_ui()
    return {
        getattr(bf.fn, "__name__", ""): bf
        for bf in app.fns.values() if bf.fn is not None
    }


def test_every_model_callback_receives_the_same_session_state(handlers):
    callbacks = [handlers[name] for name in (
        "on_trajectory_loaded", "on_sidebar_expand", "on_analyze_trajectory", "on_analysis_ask", "on_suggest_fixes",
    )]
    state = callbacks[0].inputs[-1]
    assert isinstance(state, gr.State)
    assert all(state in bf.inputs for bf in callbacks)
    language = callbacks[2].inputs[-2]
    assert isinstance(language, gr.Radio)
    assert language.choices == [("EN", "en"), ("中", "zh-CN")]
    reply_language = callbacks[2].inputs[-1]
    assert isinstance(reply_language, gr.Dropdown)
    assert reply_language.choices == [("English", "English"), ("简体中文", "Simplified Chinese")]
    assert all(bf.inputs[-2] is language for bf in callbacks[2:])
    assert all(bf.inputs[-1] is reply_language for bf in callbacks[2:])
    assert all(bf.inputs[-3] is state for bf in callbacks[2:])


def test_closed_sidebar_load_does_not_request_analysis_with_ui_settings(handlers):
    settings = _settings()
    with patch.object(sidebar, "analyze_loaded_trajectory") as call:
        packed, history, status = handlers["on_trajectory_loaded"].fn(
            [{"index": 0}], "packed evidence", settings,
        )
    call.assert_not_called()
    assert (packed, history) == ("packed evidence", [])
    assert "session-model" in status and "127.0.0.1:18999" in status
    assert settings.config.api_key not in status


def test_explicit_analysis_uses_session_connection_and_language(handlers):
    settings = _settings()
    steps = [{"index": 0}]
    history = [{"role": "assistant", "content": "fixture analysis"}]
    with patch.object(sidebar, "analyze_loaded_trajectory", return_value=("packed", history)) as call:
        result = handlers["on_analyze_trajectory"].fn(steps, "packed", settings)
    call.assert_called_once_with(
        steps, brief="packed", config=settings.config, language=settings.language,
    )
    assert result[1] == history
    assert settings.config.api_key not in result[2]


def test_existing_sidebar_history_is_not_reanalysed(handlers):
    settings = _settings()
    history = [{"role": "assistant", "content": "already analysed"}]
    with patch.object(sidebar, "analyze_loaded_trajectory") as call:
        result = handlers["on_sidebar_expand"].fn([{"index": 0}], "packed", history, settings)
    call.assert_not_called()
    assert result[1] == history
    assert "session-model" in result[2]


def test_followups_keep_sessions_isolated_and_leave_environment_untouched(handlers):
    first, second = _settings(model="first"), _settings(model="second", language="German")
    environment = dict(os.environ)
    with patch.object(sidebar, "answer_question", return_value=[]) as call:
        for settings in (first, second):
            assert handlers["on_analysis_ask"].fn("Why?", [], "packed", settings) == ([], "")
    assert [c.kwargs["config"] for c in call.call_args_list] == [first.config, second.config]
    assert [c.kwargs["language"] for c in call.call_args_list] == ["French", "German"]
    assert dict(os.environ) == environment


@pytest.mark.parametrize("ui_language,expected", [("en", "English"), ("zh-CN", "Simplified Chinese")])
def test_optional_missing_reply_choice_defaults_to_current_ui_language(handlers, ui_language, expected):
    settings = _settings(language="German")
    with patch.object(sidebar, "answer_question", return_value=[]) as call:
        handlers["on_analysis_ask"].fn("Why?", [], "packed", settings, ui_language)
    assert call.call_args.kwargs["language"] == expected
    assert call.call_args.kwargs["config"] is settings.config


def test_initial_analysis_prompt_uses_explicit_language_over_environment():
    settings = _settings(language="German")
    captured = {}

    def chat(config, system, messages):
        captured.update(config=config, system=system, messages=messages)
        return "fixture reply"

    with patch.dict(os.environ, {"ANALYZE_LANGUAGE": "French"}):
        _, history = assistant.analyze_loaded_trajectory(
            [{"index": 0}], brief="packed", config=settings.config,
            language=settings.language, chat_fn=chat,
        )
    assert captured["config"] is settings.config
    assert "Write the entire reply in German" in captured["system"]
    assert history[-1]["content"] == "fixture reply"


def test_issue_suggestions_use_session_connection_and_language(handlers):
    settings = _settings(language="German")
    run = tv.load(Path(__file__).parent / "fixtures" / "codearts_minimal.json")
    issue = OverviewIssue(kind="error", title="fixture error", detail="command failed", steps=(1,), source_id="fixture")
    captured = {}

    def chat(config, system, messages):
        captured.update(config=config, system=system, messages=messages)
        return '{"where":"AGENTS.md","fix":"Check the command","confidence":"high"}'

    with (
        patch.object(overview_tab, "collect_overview_issues", return_value=[issue]),
        patch.object(issue_judge, "complete_chat", side_effect=chat),
    ):
        results = list(handlers["on_suggest_fixes"].fn(run, settings))
    assert captured["config"] is settings.config
    assert "German" in captured["system"] and "German" in captured["messages"][0]["content"]
    assert "Check the command" in results[-1]
    assert settings.config.api_key not in "".join(results)


def test_unconfigured_issue_suggestions_point_to_settings_without_calling_model(handlers):
    run = tv.load(Path(__file__).parent / "fixtures" / "codearts_minimal.json")
    with patch.object(issue_judge, "complete_chat") as call:
        results = list(handlers["on_suggest_fixes"].fn(run, _settings(api_key="")))
    call.assert_not_called()
    assert "Settings" in results[-1]
