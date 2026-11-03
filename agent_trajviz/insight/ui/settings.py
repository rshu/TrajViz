"""Session model settings with optional nonsecret browser preferences."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import gradio as gr

from ..connection_test import ConnectionTestResult, check_connection
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
    test: gr.Button
    test_status: gr.HTML
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
    with gr.Column(elem_id="app-settings"):
        gr.Markdown("Configure AI analysis and Suggest fixes. Apply changes before running analysis.",
                    elem_classes=["settings-intro"])
        with gr.Row(elem_classes=["settings-field-row"]):
            with gr.Column(min_width=0):
                provider = gr.Dropdown(label="Provider", value=defaults["provider"],
                                       choices=[("OpenAI-compatible", "openai"), ("Anthropic", "anthropic")],
                                       elem_classes=["settings-field"])
            with gr.Column(min_width=0):
                base_url = gr.Textbox(label="Base URL", value=defaults["base_url"],
                                      placeholder="https://api.example.com/v1", elem_classes=["settings-field"])
        with gr.Row(elem_classes=["settings-field-row"]):
            with gr.Column(min_width=0):
                model = gr.Textbox(label="Model", value=defaults["model"], placeholder="Model ID",
                                   elem_classes=["settings-field"])
            with gr.Column(min_width=0):
                api_key = gr.Textbox(
                    label="API key", value="", type="password", html_attributes={"autocomplete": "off"},
                    info="Leave blank to keep the active key for this endpoint. Session only.",
                    elem_classes=["settings-field"],
                )
        with gr.Row(elem_classes=["settings-field-row", "settings-test-row"]):
            with gr.Column(min_width=0):
                clear_api_key = gr.Checkbox(label="Clear API key", value=False, elem_classes=["settings-check"])
            with gr.Column(min_width=0):
                test = gr.Button("Test API key", variant="secondary", min_width=0,
                                 elem_classes=["settings-test-action"])
        test_status = gr.HTML(_test_status_html(), elem_classes=["settings-test-status"])
        with gr.Accordion("Request options", open=False, elem_classes=["settings-options"]):
            with gr.Row(elem_classes=["settings-field-row"]):
                with gr.Column(min_width=0):
                    temperature = gr.Number(label="Temperature", value=defaults["temperature"],
                                            info="Leave blank to use the provider default.",
                                            elem_classes=["settings-field"])
                with gr.Column(min_width=0):
                    max_tokens = gr.Number(
                        label="Max output tokens", value=defaults["max_tokens"], precision=0,
                        info="Includes hidden thinking. Increase if replies are empty or cut off.",
                        elem_classes=["settings-field"],
                    )
            with gr.Row(elem_classes=["settings-field-row"]):
                with gr.Column(min_width=0):
                    timeout = gr.Number(label="Timeout (seconds)", value=defaults["timeout"], precision=0,
                                        elem_classes=["settings-field"])
                with gr.Column(min_width=0):
                    language = gr.Textbox(label="Reply language", value=defaults["language"], placeholder="English",
                                          elem_classes=["settings-field"])
        remember = gr.Checkbox(label="Remember connection details in this browser", value=True,
                                info="API keys stay in this session and are never remembered.",
                                elem_classes=["settings-check"])
        with gr.Row(elem_classes=["settings-actions"]):
            apply = gr.Button("Apply settings", variant="primary", scale=0, min_width=140, elem_classes=["settings-action"])
            reset = gr.Button("Use environment defaults", scale=0, min_width=220, elem_classes=["settings-action"])
        status = gr.Markdown("Using environment defaults. API keys stay in the current session.",
                             elem_classes=["settings-status"])
    return SettingsRefs(provider, base_url, api_key, model, temperature, max_tokens, timeout,
                        language, clear_api_key, remember, test, test_status, apply, reset, status, browser_state)


def _test_status_html(result: ConnectionTestResult | None = None, *, testing: bool = False) -> str:
    if testing:
        kind, text = "testing", "Testing connection…"
    elif result is None:
        kind, text = "idle", "Not tested. Sends a small model request; no trajectory data. Testing does not apply settings."
    else:
        kind, text = ("success" if result.ok else "error"), result.message
    return (f'<div class="settings-connection-result is-{kind}" role="status" aria-live="polite" '
            f'aria-busy="{str(testing).lower()}">{escape(text)}</div>')


def _offline_issues(run) -> str:
    return build_overview_issues_html(session_from_run(run, "")) if run is not None and run.steps else ""


def bind(refs: SettingsRefs, shared: SharedState, sidebar, overview, app, *, model_events=()) -> None:
    assert shared.state_settings is not None
    state = shared.state_settings
    outputs = [state, refs.api_key, refs.clear_api_key, refs.status, refs.browser_state,
               sidebar.analysis_chatbot, sidebar.analysis_input, sidebar.analysis_status, overview.issues_html]

    def on_test_connection(provider, base_url, model, api_key, clear_api_key, timeout, previous):
        try:
            selected = build_session_settings(
                provider=provider, base_url=base_url, api_key=api_key, model=model,
                temperature=None, max_tokens=16, timeout=timeout, language="English",
                clear_api_key=clear_api_key, previous=previous,
            )
        except ValueError as exc:
            yield gr.update(interactive=True), _test_status_html(ConnectionTestResult(False, str(exc)))
            return
        if not selected.config.ready:
            missing = [label for label, value in (("Base URL", selected.config.base_url),
                                                  ("Model", selected.config.model),
                                                  ("API key", selected.config.api_key)) if not value]
            message = f"Enter {', '.join(missing)} before testing."
            yield gr.update(interactive=True), _test_status_html(ConnectionTestResult(False, message))
            return
        yield gr.update(interactive=False), _test_status_html(testing=True)
        try:
            result = check_connection(selected.config)
        except Exception:
            # No exception/provider text enters the UI: it may contain credentials.
            result = ConnectionTestResult(False, "The connection test could not be completed. Please retry.")
        yield gr.update(interactive=True), _test_status_html(result)

    test_event = refs.test.click(
        fn=on_test_connection,
        inputs=[refs.provider, refs.base_url, refs.model, refs.api_key, refs.clear_api_key, refs.timeout, state],
        outputs=[refs.test, refs.test_status], api_name="test_connection", show_progress="hidden",
    )

    def reset_connection_test():
        return gr.update(interactive=True), _test_status_html()

    # Cancel the generator before it can display a result for fields the user changed.
    for field in (refs.provider, refs.base_url, refs.model, refs.api_key, refs.clear_api_key, refs.timeout):
        field.change(fn=reset_connection_test, outputs=[refs.test, refs.test_status],
                     cancels=[test_event], queue=False, show_progress="hidden", api_name=False)

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

    apply_event = refs.apply.click(
        fn=on_apply_settings,
        inputs=[refs.provider, refs.base_url, refs.api_key, refs.model, refs.temperature, refs.max_tokens,
                refs.timeout, refs.language, refs.clear_api_key, refs.remember, state,
                shared.state_steps, shared.state_run],
        outputs=outputs, cancels=[*model_events, test_event], api_name="apply_settings",
    )
    reset_event = refs.reset.click(
        fn=on_reset_settings, inputs=[shared.state_steps, shared.state_run],
        outputs=[*outputs, *refs.form_fields(), refs.remember],
        cancels=[*model_events, test_event], api_name="reset_settings",
    )
    for event in (apply_event, reset_event):
        event.then(fn=reset_connection_test, outputs=[refs.test, refs.test_status],
                   queue=False, show_progress="hidden", api_name=False)

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
