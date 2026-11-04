"""Analysis sidebar layout and chat callbacks."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..assistant import analyze_loaded_trajectory, answer_question
from ..llm_config import AnalysisSettings, config_status_html, request_reply_language, resolve_session_settings
from . import settings as settings_ui
from .shared import SharedState


@dataclass
class SidebarRefs:
    analysis_sidebar: gr.Sidebar
    analysis_status: gr.HTML
    analysis_chatbot: gr.Chatbot
    analysis_input: gr.Textbox
    analysis_clear: gr.Button
    analysis_run: gr.Button
    settings: settings_ui.SettingsRefs


def layout() -> SidebarRefs:
    with gr.Sidebar(
        label="AI Trajectory Analysis",
        position="right",
        width="min(560px, calc(100vw - 40px))",
        open=False,
        elem_id="analysis-sidebar",
        elem_classes=["analysis-sidebar"],
    ) as analysis_sidebar:
        gr.HTML(
            "<div class='analysis-panel-title'>🤖 AI Trajectory Analysis</div>"
        )
        with gr.Tabs(elem_id="analysis-tabs"):
            with gr.Tab("Analysis", elem_id="analysis-chat-tab"):
                analysis_status = gr.HTML(config_status_html(loaded_steps=0))
                analysis_run = gr.Button(
                    "Analyze trajectory", size="sm", variant="primary", elem_id="analysis-run",
                )
                analysis_chatbot = gr.Chatbot(
                    value=[],
                    label="Analysis",
                    show_label=False,
                    height=380,
                    resizable=True,
                    layout="panel",
                    placeholder=(
                        "Load a trajectory and choose Analyze trajectory, or ask a question below. "
                        "Requests send trajectory excerpts to the configured provider."
                    ),
                    buttons=["copy"],
                    feedback_options=None,
                    elem_classes=["analysis-chatbot"],
                )
                with gr.Row():
                    analysis_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What is the biggest performance bottleneck?",
                        scale=4,
                        container=False,
                        submit_btn=True,
                        elem_id="analysis-input",
                    )
                analysis_clear = gr.Button("Clear chat", size="sm", variant="secondary")
            with gr.Tab("Settings", elem_id="analysis-settings-tab"):
                settings_refs = settings_ui.layout()
    return SidebarRefs(
        analysis_sidebar=analysis_sidebar,
        analysis_status=analysis_status,
        analysis_chatbot=analysis_chatbot,
        analysis_input=analysis_input,
        analysis_clear=analysis_clear,
        analysis_run=analysis_run,
        settings=settings_refs,
    )


def _run_analysis(steps, brief, settings: AnalysisSettings | None = None, ui_language=None, reply_language=None):
    """First LLM pass using this session's settings and packed trajectory brief."""
    active = resolve_session_settings(settings)
    if not steps:
        return "", [], config_status_html(loaded_steps=0, config=active.config)
    packed = brief if isinstance(brief, str) else ""
    brief, history = analyze_loaded_trajectory(
        steps, brief=packed, config=active.config,
        language=request_reply_language(active.language, ui_language, reply_language),
    )
    return brief, history, config_status_html(loaded_steps=len(steps), config=active.config)


def bind(refs: SidebarRefs, shared: SharedState, load_events, *, language: gr.Radio | None = None,
         reply_language: gr.Dropdown | None = None) -> tuple:
    """Bind chat and return model event handles so settings can cancel stale work."""
    state_analysis_brief = shared.state_analysis_brief
    settings_inputs = [shared.state_settings] if shared.state_settings is not None else []
    # An optional locale input must not occupy the settings argument's slot in
    # an embedder that has no session overrides.
    if (language is not None or reply_language is not None) and not settings_inputs:
        settings_inputs = [gr.State(None)]
    language_inputs = [language] if language is not None else []
    if reply_language is not None:
        if language is None:
            language_inputs = [gr.State(None)]
        language_inputs.append(reply_language)
    model_events = []

    def on_trajectory_loaded(steps, brief, settings=None):
        """Refresh local evidence and status without requesting a model."""
        active = resolve_session_settings(settings)
        if not steps:
            return "", [], config_status_html(loaded_steps=0, config=active.config)
        packed = brief if isinstance(brief, str) else ""
        status = config_status_html(loaded_steps=len(steps), config=active.config)
        return packed, [], status

    for _ev in load_events:
        _ev.then(
            fn=on_trajectory_loaded,
            inputs=[shared.state_steps, state_analysis_brief, *settings_inputs],
            outputs=[state_analysis_brief, refs.analysis_chatbot, refs.analysis_status],
        )

    def on_sidebar_expand(steps, brief, history, settings=None):
        active = resolve_session_settings(settings)
        return (
            brief if isinstance(brief, str) else "",
            history or [],
            config_status_html(loaded_steps=len(steps or []), config=active.config),
        )

    # Opening the panel also exposes Settings, so it must never start analysis.
    refs.analysis_sidebar.expand(
        fn=on_sidebar_expand,
        inputs=[shared.state_steps, state_analysis_brief, refs.analysis_chatbot, *settings_inputs],
        outputs=[
            state_analysis_brief,
            refs.analysis_chatbot,
            refs.analysis_status,
        ],
    )

    def on_analyze_trajectory(steps, brief, settings=None, ui_language=None, reply_language=None):
        return _run_analysis(steps, brief, settings, ui_language, reply_language)

    model_events.append(refs.analysis_run.click(
        fn=on_analyze_trajectory,
        inputs=[shared.state_steps, state_analysis_brief, *settings_inputs, *language_inputs],
        outputs=[state_analysis_brief, refs.analysis_chatbot, refs.analysis_status],
    ))

    def on_analysis_ask(question, history, brief, settings=None, ui_language=None, reply_language=None):
        active = resolve_session_settings(settings)
        return answer_question(
            question, history, brief, config=active.config,
            language=request_reply_language(active.language, ui_language, reply_language),
        ), ""

    model_events.append(refs.analysis_input.submit(
        fn=on_analysis_ask,
        inputs=[refs.analysis_input, refs.analysis_chatbot, state_analysis_brief, *settings_inputs, *language_inputs],
        outputs=[refs.analysis_chatbot, refs.analysis_input],
    ))
    refs.analysis_clear.click(fn=lambda: [], outputs=[refs.analysis_chatbot])
    return tuple(model_events)
