"""Session model settings with optional nonsecret browser preferences."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..llm_config import (
    build_session_settings,
    config_status_html,
    nonsecret_settings,
)
from ..presenters.issues import build_overview_issues_html
from ..session import session_from_run
from .shared import SharedState

_FIELDS = ("provider", "base_url", "model", "temperature", "max_tokens", "timeout", "language")


@dataclass
class SettingsRefs:
    provider: gr.Dropdown
    base_url: gr.Textbox
    api_key: gr.Textbox
    model: gr.Textbox
    temperature: gr.Number
    max_tokens: gr.Number
    timeout: gr.Number
    language: gr.Textbox
    clear_api_key: gr.Checkbox
    remember: gr.Checkbox
    apply: gr.Button
    reset: gr.Button
    status: gr.Markdown
    browser_state: gr.BrowserState

    def form_fields(self) -> list:
        return [getattr(self, name) for name in _FIELDS]


def layout() -> SettingsRefs:
    defaults = nonsecret_settings()
    # Only public connection preferences enter this storage. A stable identifier
    # lets those preferences survive app restarts; this is never a key store.
    browser_state = gr.BrowserState({}, storage_key="trajviz-connection-settings-v1",
                                    secret="trajviz-nonsecret-preferences-v1")
    with gr.Accordion("Settings", open=False, elem_id="app-settings"):
        gr.Markdown("Configure the model used by AI analysis and Suggest fixes. "
                    "Changes take effect when you apply them; applying settings sends no model request.")
        with gr.Row():
            provider = gr.Dropdown(label="Provider", value=defaults["provider"],
                                   choices=[("OpenAI-compatible", "openai"), ("Anthropic", "anthropic")], scale=1)
            base_url = gr.Textbox(label="Base URL", value=defaults["base_url"],
                                  placeholder="https://api.example.com/v1", scale=3)
        with gr.Row():
            model = gr.Textbox(label="Model", value=defaults["model"], placeholder="Model ID")
            api_key = gr.Textbox(label="API key", value="", type="password", html_attributes={"autocomplete": "off"},
                                 info="Leave blank to retain the active key for this endpoint. Keys stay in this session.")
        clear_api_key = gr.Checkbox(label="Clear API key", value=False)
        with gr.Accordion("Request options", open=False):
            with gr.Row():
                temperature = gr.Number(label="Temperature", value=defaults["temperature"],
                                        info="Leave blank to use the provider default.")
                max_tokens = gr.Number(
                    label="Max output tokens", value=defaults["max_tokens"], precision=0,
                    info=("Reasoning models spend part of this on hidden thinking; "
                          "raise it if replies come back empty or cut off."),
                )
                timeout = gr.Number(label="Timeout (seconds)", value=defaults["timeout"], precision=0)
            language = gr.Textbox(label="Reply language", value=defaults["language"], placeholder="English")
        remember = gr.Checkbox(label="Remember connection details in this browser", value=True,
                                info="Saves these fields on this device. API keys are never remembered.")
        with gr.Row():
            apply = gr.Button("Apply settings", variant="primary", scale=0, min_width=140, elem_classes=["settings-action"])
            reset = gr.Button("Use environment defaults", scale=0, min_width=220, elem_classes=["settings-action"])
        status = gr.Markdown("Using environment defaults. API keys stay in the current session.")
    return SettingsRefs(provider, base_url, api_key, model, temperature, max_tokens, timeout,
                        language, clear_api_key, remember, apply, reset, status, browser_state)


def _offline_issues(run) -> str:
    return build_overview_issues_html(session_from_run(run, "")) if run is not None and run.steps else ""


def bind(refs: SettingsRefs, shared: SharedState, sidebar, overview, app, *, model_events=()) -> None:
    assert shared.state_settings is not None
    state = shared.state_settings
    outputs = [state, refs.api_key, refs.clear_api_key, refs.status, refs.browser_state,
               sidebar.analysis_chatbot, sidebar.analysis_input, sidebar.analysis_status, overview.issues_html]

    def on_apply_settings(provider, base_url, api_key, model, temperature, max_tokens, timeout,
                          language, clear_api_key, remember, previous, steps, run):
        if not isinstance(remember, bool):
            raise gr.Error("Remember connection details must be a checkbox value.")
        try:
            selected = build_session_settings(
                provider=provider, base_url=base_url, api_key=api_key, model=model,
                temperature=temperature, max_tokens=max_tokens, timeout=timeout, language=language,
                clear_api_key=clear_api_key, previous=previous,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        remembered = {"version": 1, "values": nonsecret_settings(selected)} if remember else {}
        status = "Settings applied for this session. "
        if not selected.config.ready:
            status += "Connection incomplete; enter the missing connection fields before requesting analysis. "
        status += "API keys are session-only. No model request was sent."
        return (selected, "", False, status, remembered, [], "",
                config_status_html(config=selected.config, loaded_steps=len(steps or [])), _offline_issues(run))

    def on_reset_settings(steps, run):
        defaults = nonsecret_settings()
        values = (None, "", False, "Using environment defaults. Remembered connection details cleared.",
                  {}, [], "", config_status_html(loaded_steps=len(steps or [])), _offline_issues(run))
        return (*values, *(defaults[name] for name in _FIELDS), False)

    refs.apply.click(
        fn=on_apply_settings,
        inputs=[refs.provider, refs.base_url, refs.api_key, refs.model, refs.temperature, refs.max_tokens,
                refs.timeout, refs.language, refs.clear_api_key, refs.remember, state,
                shared.state_steps, shared.state_run],
        outputs=outputs, cancels=list(model_events), api_name="apply_settings",
    )
    refs.reset.click(
        fn=on_reset_settings, inputs=[shared.state_steps, shared.state_run],
        outputs=[*outputs, *refs.form_fields(), refs.remember],
        cancels=list(model_events), api_name="reset_settings",
    )

    def on_restore_settings(saved):
        defaults = nonsecret_settings()
        message = "Using environment defaults. API keys stay in the current session."
        remember = True
        cleaned = {}
        active = None
        if isinstance(saved, dict) and saved.get("version") == 1 and isinstance(saved.get("values"), dict):
            candidate = {name: saved["values"].get(name, defaults[name]) for name in _FIELDS}
            try:
                restored = build_session_settings(**candidate, api_key="", clear_api_key=True)
            except ValueError:
                message = "Saved connection details were invalid; using environment defaults."
            else:
                active = restored
                defaults = nonsecret_settings(restored)
                cleaned = {"version": 1, "values": defaults}
                message = "Connection details restored. Enter your API key and apply settings to enable analysis."
        return (active, *(defaults[name] for name in _FIELDS), "", False, remember, message, cleaned,
                config_status_html(config=active.config if active is not None else None, loaded_steps=0))

    app.load(fn=on_restore_settings, inputs=[refs.browser_state],
             outputs=[state, *refs.form_fields(), refs.api_key, refs.clear_api_key, refs.remember,
                      refs.status, refs.browser_state, sidebar.analysis_status], api_name=False)
