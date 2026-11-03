"""Overview tab: performance through labels, plus pressure-agent and label callbacks."""

from __future__ import annotations

import html
import logging
import os
from dataclasses import dataclass

import gradio as gr
import plotly.graph_objects as go

from ..charts import build_context_pressure_chart, build_execution_gantt_chart
from ..charts.gantt import GANTT_AXIS_SKIP_IDLE, GANTT_AXIS_WALLCLOCK
from ..context_usage import (
    DEFAULT_CONTEXT_WINDOW_LIMIT,
    PRESSURE_ALL_AGENTS,
    SNAPSHOT_CURRENT,
    context_pressure_series,
    parse_usage_snapshot,
    usage_snapshot_choices,
)
from ..formatting import format_context_pressure_html
from ..issue_export import write_issues_agent_export
from ..presenters.label_ui import build_label_ui_payload
from ..presenters.issues import (
    build_overview_issues_html,
    collect_overview_issues,
    overview_issue_from_payload,
    overview_issues_payload,
    rank_issues,
)
from ..presenters.overview import (
    build_chart_outputs,
    build_diagnostics_outputs,
    build_overview_outputs,
    build_token_summary_html,
    empty_plotly_fig,
    load_warnings_html,
    render_section_guide,
)
from ..token_explain import build_token_step_panel_html
from ..issue_judge import JUDGE_ISSUE_CAP, iter_judge_overview_issues
from ..llm_config import resolve_analysis_config, resolve_session_settings
from ..session import LoadedSession, session_from_run
from .shared import SharedState, is_latest_run, safe_callback
from .upload import (
    UploadRefs,
    _TEMP_ISSUES_EXPORT_PREFIX,
    _export_off,
    stage_gradio_download,
)

logger = logging.getLogger(__name__)


@dataclass
class OverviewRefs:
    session_detail_html: gr.HTML
    overview_kpi_html: gr.HTML
    overview_section: gr.Radio
    performance_section: gr.Column
    context_section: gr.Column
    tools_section: gr.Column
    agents_section: gr.Column
    diagnostics_section: gr.Column
    deep_dive_section: gr.Column
    labels_section: gr.Column
    issues_html: gr.HTML
    suggest_fixes_btn: gr.Button
    export_issues_btn: gr.DownloadButton
    metrics_md: gr.Markdown
    token_chart: gr.Plot
    token_summary_html: gr.HTML
    token_step_html: gr.HTML
    duration_chart: gr.Plot
    gantt_chart: gr.Plot
    gantt_axis_mode: gr.Radio
    behavior_md: gr.Markdown
    tool_chart: gr.Plot
    tool_duration_chart: gr.Plot
    skill_chart: gr.Plot
    tool_outcome_chart: gr.Plot
    agent_summary_html: gr.HTML
    agent_token_chart: gr.Plot
    agent_swimlane_chart: gr.Plot
    diag_summary_html: gr.HTML
    diag_file_chart: gr.Plot
    diag_pressure_html: gr.HTML
    diag_pressure_agent: gr.Dropdown
    diag_window_limit: gr.Number
    diag_usage_snapshot: gr.Dropdown
    diag_pressure_chart: gr.Plot
    diag_rootcause_html: gr.HTML
    plan_timeline_chart: gr.Plot
    hotspots_md: gr.Markdown
    per_message_md: gr.Markdown
    label_status_html: gr.HTML
    label_charts_row1: gr.Row
    label_phase_count_chart: gr.Plot
    label_action_count_chart: gr.Plot
    label_charts_row2: gr.Row
    label_phase_dur_chart: gr.Plot
    label_action_dur_chart: gr.Plot
    label_timeline_row: gr.Row
    label_timeline_chart: gr.Plot


# Nav label -> the `OverviewRefs` field holding that section's Column, in nav
# order. One structure, because `show_overview_section` zips the names against
# the Columns purely by position: as two parallel lists, reordering a Column or
# inserting a section silently showed the wrong panel.
OVERVIEW_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Summary", "performance_section"),
    ("Tools", "tools_section"),
    ("Agents", "agents_section"),
    ("Context Utilization", "context_section"),
    ("Diagnostics", "diagnostics_section"),
    ("Deep Dive", "deep_dive_section"),
    ("Labels", "labels_section"),
)

OVERVIEW_SECTION_NAMES = [name for name, _ in OVERVIEW_SECTIONS]


def _resolve_overview_sections(refs: OverviewRefs) -> tuple[gr.Column, ...]:
    """The Columns named by `OVERVIEW_SECTIONS`, in nav order.

    A missing field or two names resolving to the same Column would show the
    wrong panel (or fail as an opaque Gradio output-count error), so fail here
    with the mapping that is wrong.
    """
    missing = [attr for _, attr in OVERVIEW_SECTIONS if not hasattr(refs, attr)]
    if missing:
        raise ValueError(f"OVERVIEW_SECTIONS names fields OverviewRefs does not have: {missing}")
    sections = tuple(getattr(refs, attr) for _, attr in OVERVIEW_SECTIONS)
    if len({id(section) for section in sections}) != len(OVERVIEW_SECTIONS):
        raise ValueError("OVERVIEW_SECTIONS maps two nav names onto the same Column")
    return sections


def layout(overview_kpi_html: gr.HTML) -> OverviewRefs:
    with gr.TabItem("Overview"):
        # Debug hero: Issues first; KPI strip lives above the main Tabs.
        issues_html = gr.HTML("")
        # Judging sends trajectory excerpts (prompts, commands, file paths,
        # tool output) to the configured LLM endpoint. It runs automatically
        # after load when ANALYZE_*/LABEL_* are set; the button re-runs it.
        with gr.Row():
            suggest_fixes_btn = gr.Button(
                "Suggest fixes with LLM (sends trajectory excerpts to the configured provider)",
                size="sm",
                variant="secondary",
                scale=3,
            )
            export_issues_btn = gr.DownloadButton(
                label="Export issues into JSON",
                variant="secondary",
                size="sm",
                interactive=False,
                scale=1,
                min_width=160,
            )
        session_detail_html = gr.HTML("")

        overview_section_names = OVERVIEW_SECTION_NAMES
        with gr.Row(elem_classes=["overview-content-layout"]):
            with gr.Column(scale=0, min_width=160, elem_classes=["overview-section-nav"]):
                gr.HTML("<div class='overview-nav-title'>Contents</div>")
                overview_section = gr.Radio(
                    choices=overview_section_names,
                    value="Summary",
                    show_label=False,
                    container=False,
                    elem_classes=["overview-section-radio"],
                )

            with gr.Column(scale=1, min_width=0, elem_classes=["overview-section-content"]):
                with gr.Column(visible=True) as performance_section:
                    gr.HTML(render_section_guide("performance_section"))
                    token_summary_html = gr.HTML(
                        "",
                        elem_classes=["token-summary-wrap"],
                    )
                    with gr.Row(equal_height=True):
                        token_chart = gr.Plot(
                            show_label=False,
                            label="Token Usage",
                            elem_id="token-chart",
                        )
                        duration_chart = gr.Plot(
                            show_label=False,
                            label="Step Duration",
                            elem_id="duration-chart",
                        )
                    token_step_html = gr.HTML(
                        build_token_step_panel_html([]),
                        elem_classes=["token-step-explain"],
                    )
                    with gr.Row(equal_height=True):
                        gantt_chart = gr.Plot(
                            show_label=False,
                            label="Execution Timeline",
                            elem_id="gantt-chart",
                        )
                    gantt_axis_mode = gr.Radio(
                        label="Timeline axis",
                        choices=[
                            ("Skip idle gaps (default)", GANTT_AXIS_SKIP_IDLE),
                            ("True wall-clock", GANTT_AXIS_WALLCLOCK),
                        ],
                        value=GANTT_AXIS_SKIP_IDLE,
                        interactive=True,
                    )

                with gr.Column(visible=False) as context_section:
                    gr.HTML(render_section_guide("context_section"))
                    with gr.Row():
                        diag_pressure_agent = gr.Dropdown(
                            label="Agent",
                            choices=[("All agents", PRESSURE_ALL_AGENTS)],
                            value=PRESSURE_ALL_AGENTS,
                            visible=False,
                            interactive=True,
                            scale=2,
                        )
                        diag_usage_snapshot = gr.Dropdown(
                            label="Window snapshot",
                            choices=[("Current window", SNAPSHOT_CURRENT)],
                            value=SNAPSHOT_CURRENT,
                            visible=False,
                            interactive=True,
                            scale=2,
                        )
                        diag_window_limit = gr.Number(
                            label="Window limit (tokens)",
                            info=f"Empty = detected, else an assumed {DEFAULT_CONTEXT_WINDOW_LIMIT:,}",
                            value=None,
                            precision=0,
                            minimum=1,
                            step=1000,
                            interactive=True,
                            scale=1,
                        )
                    diag_pressure_html = gr.HTML("")
                    diag_pressure_chart = gr.Plot(
                        show_label=False,
                        label="Context Window Pressure",
                    )

                with gr.Column(visible=False) as tools_section:
                    gr.HTML(render_section_guide("tools_section"))
                    behavior_md = gr.Markdown("")
                    with gr.Row(equal_height=True):
                        tool_chart = gr.Plot(show_label=False, label="Tool Call Frequency")
                        tool_duration_chart = gr.Plot(
                            show_label=False,
                            label="Tool Call Duration",
                            elem_id="tool-duration-chart",
                        )
                    with gr.Row(equal_height=True):
                        tool_outcome_chart = gr.Plot(
                            show_label=False,
                            label="Tool Outcome Timeline",
                            elem_id="tool-outcome-chart",
                        )
                    with gr.Row(equal_height=True):
                        skill_chart = gr.Plot(show_label=False, label="Skill Calls by Agent")
                        gr.Column(scale=1)

                with gr.Column(visible=False) as agents_section:
                    gr.HTML(render_section_guide("agents_section"))
                    agent_summary_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        agent_swimlane_chart = gr.Plot(
                            show_label=False,
                            label="Agent Swimlane",
                            elem_id="agent-swimlane-chart",
                        )
                    with gr.Row(equal_height=True):
                        agent_token_chart = gr.Plot(show_label=False, label="Token Breakdown by Agent")
                        gr.Column(scale=1)

                with gr.Column(visible=False) as diagnostics_section:
                    gr.HTML(render_section_guide("diagnostics_section"))
                    diag_summary_html = gr.HTML("")
                    diag_file_chart = gr.Plot(
                        show_label=False,
                        label="File Interaction Timeline",
                        elem_id="diag-file-chart",
                        elem_classes=["resizable-chart"],
                    )
                    diag_rootcause_html = gr.HTML("")
                    plan_timeline_chart = gr.Plot(show_label=False, label="Plan Progress Timeline")

                with gr.Column(visible=False) as deep_dive_section:
                    gr.HTML(render_section_guide("deep_dive_section"))
                    metrics_md = gr.Markdown("")
                    hotspots_md = gr.Markdown("")
                    per_message_md = gr.Markdown("")

                with gr.Column(visible=False) as labels_section:
                    gr.HTML(render_section_guide("labels_section"))
                    label_status_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                        "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
                    )
                    with gr.Row(equal_height=True, visible=False) as label_charts_row1:
                        label_phase_count_chart = gr.Plot(show_label=False, label="Phase Count Distribution")
                        label_action_count_chart = gr.Plot(show_label=False, label="Action Count Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_charts_row2:
                        label_phase_dur_chart = gr.Plot(show_label=False, label="Phase Duration Distribution")
                        label_action_dur_chart = gr.Plot(show_label=False, label="Action Duration Distribution")
                    with gr.Row(equal_height=True, visible=False) as label_timeline_row:
                        label_timeline_chart = gr.Plot(show_label=False, label="Step Timeline")

    return OverviewRefs(
        session_detail_html=session_detail_html,
        overview_kpi_html=overview_kpi_html,
        overview_section=overview_section,
        performance_section=performance_section,
        context_section=context_section,
        tools_section=tools_section,
        agents_section=agents_section,
        diagnostics_section=diagnostics_section,
        deep_dive_section=deep_dive_section,
        labels_section=labels_section,
        issues_html=issues_html,
        suggest_fixes_btn=suggest_fixes_btn,
        export_issues_btn=export_issues_btn,
        metrics_md=metrics_md,
        token_chart=token_chart,
        token_summary_html=token_summary_html,
        token_step_html=token_step_html,
        duration_chart=duration_chart,
        gantt_chart=gantt_chart,
        gantt_axis_mode=gantt_axis_mode,
        behavior_md=behavior_md,
        tool_chart=tool_chart,
        tool_duration_chart=tool_duration_chart,
        skill_chart=skill_chart,
        tool_outcome_chart=tool_outcome_chart,
        agent_summary_html=agent_summary_html,
        agent_token_chart=agent_token_chart,
        agent_swimlane_chart=agent_swimlane_chart,
        diag_summary_html=diag_summary_html,
        diag_file_chart=diag_file_chart,
        diag_pressure_html=diag_pressure_html,
        diag_pressure_agent=diag_pressure_agent,
        diag_window_limit=diag_window_limit,
        diag_usage_snapshot=diag_usage_snapshot,
        diag_pressure_chart=diag_pressure_chart,
        diag_rootcause_html=diag_rootcause_html,
        plan_timeline_chart=plan_timeline_chart,
        hotspots_md=hotspots_md,
        per_message_md=per_message_md,
        label_status_html=label_status_html,
        label_charts_row1=label_charts_row1,
        label_phase_count_chart=label_phase_count_chart,
        label_action_count_chart=label_action_count_chart,
        label_charts_row2=label_charts_row2,
        label_phase_dur_chart=label_phase_dur_chart,
        label_action_dur_chart=label_action_dur_chart,
        label_timeline_row=label_timeline_row,
        label_timeline_chart=label_timeline_chart,
    )


def load_slots(refs: OverviewRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "overview_kpi_html": refs.overview_kpi_html,
        "session_detail_html": refs.session_detail_html,
        "issues_html": refs.issues_html,
        "export_issues_btn": refs.export_issues_btn,
        "metrics_md": refs.metrics_md,
        "token_chart": refs.token_chart,
        "token_summary_html": refs.token_summary_html,
        "token_step_html": refs.token_step_html,
        "duration_chart": refs.duration_chart,
        "gantt_chart": refs.gantt_chart,
        "gantt_axis_mode": refs.gantt_axis_mode,
        "behavior_md": refs.behavior_md,
        "tool_chart": refs.tool_chart,
        "tool_duration_chart": refs.tool_duration_chart,
        "skill_chart": refs.skill_chart,
        "tool_outcome_chart": refs.tool_outcome_chart,
        "agent_summary_html": refs.agent_summary_html,
        "agent_token_chart": refs.agent_token_chart,
        "agent_swimlane_chart": refs.agent_swimlane_chart,
        "diag_summary_html": refs.diag_summary_html,
        "diag_pressure_html": refs.diag_pressure_html,
        "diag_pressure_agent": refs.diag_pressure_agent,
        "diag_window_limit": refs.diag_window_limit,
        "diag_usage_snapshot": refs.diag_usage_snapshot,
        "diag_pressure_chart": refs.diag_pressure_chart,
        "diag_file_chart": refs.diag_file_chart,
        "diag_rootcause_html": refs.diag_rootcause_html,
        "plan_timeline_chart": refs.plan_timeline_chart,
        "hotspots_md": refs.hotspots_md,
        "per_message_md": refs.per_message_md,
    }


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named Overview values for a load (empty when *session* is None)."""
    del banner
    fig = empty_plotly_fig()
    export_off = _export_off()
    if session is None:
        return {
            "overview_kpi_html": gr.update(value="", visible=False),
            "session_detail_html": "",
            "issues_html": "",
            "export_issues_btn": export_off,
            "state_overview_issues": [],
            "metrics_md": "",
            "token_chart": fig,
            "token_summary_html": "",
            "token_step_html": build_token_step_panel_html([]),
            "duration_chart": fig,
            "gantt_chart": fig,
            "gantt_axis_mode": GANTT_AXIS_SKIP_IDLE,
            "behavior_md": "",
            "tool_chart": fig,
            "tool_duration_chart": fig,
            "skill_chart": fig,
            "tool_outcome_chart": fig,
            "agent_summary_html": "",
            "agent_token_chart": fig,
            "agent_swimlane_chart": fig,
            "diag_summary_html": "",
            "diag_pressure_html": "",
            "diag_pressure_agent": gr.update(
                choices=[("All agents", PRESSURE_ALL_AGENTS)],
                value=PRESSURE_ALL_AGENTS,
                visible=False,
            ),
            "diag_window_limit": None,
            "diag_usage_snapshot": gr.update(
                choices=[("Current window", SNAPSHOT_CURRENT)],
                value=SNAPSHOT_CURRENT,
                visible=False,
            ),
            "diag_pressure_chart": fig,
            "diag_file_chart": fig,
            "diag_rootcause_html": "",
            "plan_timeline_chart": fig,
            "hotspots_md": "",
            "per_message_md": "",
        }

    # With AI configured the 任务/结局 rows start as a spinner placeholder
    # (on_suggest_fixes fills them in right after load); without AI they are
    # omitted entirely.
    ov = build_overview_outputs(session, ai_pending=resolve_analysis_config().ready)
    ch = build_chart_outputs(session, dark=dark)
    dg = build_diagnostics_outputs(session, dark=dark)
    return {
        "overview_kpi_html": gr.update(value=ov["kpi_html"], visible=True),
        "session_detail_html": ov["session_detail"],
        "issues_html": ov["issues_html"],
        # Disarmed here; load.success / post-judge prepare re-enables the button.
        "export_issues_btn": export_off,
        "state_overview_issues": ov["issues_payload"],
        "metrics_md": ov["metrics_text"],
        "token_chart": ch["tok_fig"],
        "token_summary_html": build_token_summary_html(session),
        "token_step_html": build_token_step_panel_html(session.steps),
        "duration_chart": ch["dur_fig"],
        "gantt_chart": ch["gantt_fig"],
        "gantt_axis_mode": GANTT_AXIS_SKIP_IDLE,
        "behavior_md": ov["behavior_text"],
        "tool_chart": ch["tl_fig"],
        "tool_duration_chart": ch["tool_dur_fig"],
        "skill_chart": ch["skill_fig"],
        "tool_outcome_chart": ch["tool_outcome_fig"],
        "agent_summary_html": ch["agent_cards_html"],
        "agent_token_chart": ch["agent_tok_fig"],
        "agent_swimlane_chart": ch["swimlane_fig"],
        "diag_summary_html": dg["diag_summary_html"],
        "diag_pressure_html": dg["diag_pressure_html"],
        "diag_pressure_agent": gr.update(
            choices=dg["diag_pressure_dropdown"]["choices"],
            value=dg["diag_pressure_dropdown"]["value"],
            visible=dg["diag_pressure_dropdown"]["visible"],
        ),
        # Pre-fill only a window that was detected. Programmatic updates fire
        # `.change`, which re-applies the field as a USER override — so a
        # pre-filled 128k fallback would silently become "user-set", lose its
        # "assumed" label, and bring back a percentage the export disproves.
        "diag_window_limit": (
            session.pressure_series.get("window_limit")
            if session.pressure_series.get("window_source") in ("declared", "model")
            else None
        ),
        "diag_usage_snapshot": _snapshot_dropdown_update(session.steps),
        "diag_pressure_chart": dg["diag_pressure_chart"],
        "diag_file_chart": dg["diag_file_chart"],
        "diag_rootcause_html": dg["diag_rootcause_html"],
        "plan_timeline_chart": ch["plan_timeline_fig"],
        "hotspots_md": ov["hotspots_text"],
        "per_message_md": ov["per_message_text"],
    }


def _snapshot_dropdown_update(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    value: str = SNAPSHOT_CURRENT,
) -> dict:
    choices = usage_snapshot_choices(steps, agent_key=agent_key)
    values = {choice_value for _label, choice_value in choices}
    selected = value if value in values else SNAPSHOT_CURRENT
    return gr.update(
        choices=choices,
        value=selected,
        visible=len(choices) > 1,
    )


def prepare_issues_agent_download(run, issues_rows, request: gr.Request | None = None):
    """Stage agent-ready issues JSON for the Overview DownloadButton."""
    if run is None or not getattr(run, "steps", None):
        return _export_off()
    try:
        session = session_from_run(run)
        issues = None
        if isinstance(issues_rows, list) and issues_rows:
            issues = [
                overview_issue_from_payload(row, max_steps=None)
                for row in issues_rows
                if isinstance(row, dict)
            ]
        return stage_gradio_download(
            lambda directory: write_issues_agent_export(
                session, directory, issues=issues,
            ),
            prefix=_TEMP_ISSUES_EXPORT_PREFIX,
            request=request,
        )
    except Exception:
        logger.exception("writing issues agent export failed")
        return _export_off()


def on_suggest_fixes(run, settings=None, request: gr.Request | None = None):
    """Run the LLM run summary + Issues judge after load (or on button click); stream progress into the panel.

    The stream paints the Issues panel across many yields while the run it
    started with may already have been replaced (upload row cleared, another
    trajectory loaded). Every resume checks ``shared.is_latest_run`` and stops
    itself instead of overwriting the new trace's panel with the old one's
    content.
    """
    from ..presenters.issues import render_overview_issues_html
    from ..issue_judge import judge_run_summary, _judge_error_text

    session_key = getattr(request, "session_hash", None)

    def _superseded() -> bool:
        return not is_latest_run(session_key, run)

    if run is None or not run.steps:
        yield render_overview_issues_html([]), []
        return

    if _superseded():
        return

    active = resolve_session_settings(settings)
    cfg = active.config
    session = session_from_run(run, "")
    # Prefer collect+rank so tests can patch ``collect_overview_issues`` the
    # same way as upstream; production still gets ranking before judging.
    ranked = rank_issues(collect_overview_issues(session))
    payload = overview_issues_payload(ranked)

    if not cfg.ready:
        missing = ", ".join(cfg.missing)
        banner = f"Open Settings or configure {missing} in .env to auto-suggest fixes"
        if _superseded():
            return
        yield build_overview_issues_html(session, issues=ranked, banner=banner), payload
        return

    # AI run summary first — one call, before issue judging. While it runs
    # the 任务/结局 rows show a spinner placeholder. Progress line stays empty:
    # the 任务/结局 spinner is the sole status.
    ai_summary = None
    summary_error = ""
    if _superseded():
        return
    yield build_overview_issues_html(
        session, issues=ranked, ai_pending=True,
    ), gr.skip()
    if _superseded():
        return
    try:
        ai_summary = judge_run_summary(session, config=cfg, language=active.language)
    except Exception as exc:  # noqa: BLE001 — rows stay hidden; banner explains
        summary_error = _judge_error_text(exc)

    if not ranked:
        if _superseded():
            return
        banner = f"Run summary failed: {summary_error[:160]}" if summary_error else ""
        yield build_overview_issues_html(
            session, issues=ranked, ai_summary=ai_summary, banner=banner,
        ), payload
        return

    judged = ranked
    errors: list[str] = []
    for progress in iter_judge_overview_issues(
        session, ranked, config=cfg, limit=JUDGE_ISSUE_CAP, language=active.language,
    ):
        judged = progress.issues
        errors = progress.errors
        if not progress.finished:
            short = progress.current_title
            if len(short) > 48:
                short = short[:45] + "…"
            progress_line = (
                f"Suggesting fixes {progress.current}/{progress.total} — {short}"
            )
            # Stream HTML only; refresh State once when judgments are final.
            if _superseded():
                return
            yield build_overview_issues_html(
                session, issues=judged, progress=progress_line,
                ai_summary=ai_summary,
            ), gr.skip()
            continue

        ok = sum(1 for i in judged if i.judgment is not None)
        if errors and ok == 0:
            banner = f"Judge failed ({len(errors)}). First: {errors[0][:160]}"
            if _superseded():
                return
            yield build_overview_issues_html(
                session, issues=ranked, banner=banner, ai_summary=ai_summary,
            ), payload
            return

        banner = ""
        if summary_error:
            banner = f"Run summary failed: {summary_error[:160]}"
        elif errors:
            banner = (
                f"{len(errors)} issue(s) could not be judged"
                f" ({ok}/{min(len(ranked), JUDGE_ISSUE_CAP)} ok)."
            )
        elif len(ranked) > JUDGE_ISSUE_CAP:
            banner = f"Judged {ok}/{JUDGE_ISSUE_CAP} (capped)."
        if _superseded():
            return
        yield build_overview_issues_html(
            session, issues=judged, banner=banner, ai_summary=ai_summary,
        ), overview_issues_payload(judged)


def bind(
    refs: OverviewRefs,
    shared: SharedState,
    upload: UploadRefs,
    load_events: tuple = (),  # kept for the uniform tab-bind signature
) -> tuple:
    overview_section_names = OVERVIEW_SECTION_NAMES
    overview_sections = _resolve_overview_sections(refs)

    def show_overview_section(selected):
        return tuple(gr.update(visible=name == selected) for name in overview_section_names)

    refs.overview_section.change(
        fn=show_overview_section,
        inputs=[refs.overview_section],
        outputs=list(overview_sections),
    ).then(
        fn=None,
        js="() => { if (window.tvExpandFileTimeline) { window.tvExpandFileTimeline(); setTimeout(window.tvExpandFileTimeline, 200); } }",
    )

    @safe_callback("Execution Timeline")
    def on_gantt_axis_change(axis_mode, steps, dark):
        """Rebuild the Gantt when the user toggles idle-gap compression."""
        fig = build_execution_gantt_chart(
            steps or [],
            dark=dark,
            axis_mode=axis_mode or GANTT_AXIS_SKIP_IDLE,
        )
        return fig or empty_plotly_fig()

    refs.gantt_axis_mode.change(
        fn=on_gantt_axis_change,
        inputs=[refs.gantt_axis_mode, shared.state_steps, shared.state_dark],
        outputs=[refs.gantt_chart],
    )

    settings_inputs = (
        [shared.state_settings] if shared.state_settings is not None else []
    )
    judge_event = dict(
        fn=on_suggest_fixes,
        inputs=[shared.state_run, *settings_inputs],
        outputs=[refs.issues_html, shared.state_overview_issues],
        show_progress="minimal",
        concurrency_id="issues_judge",
    )
    export_event = dict(
        fn=prepare_issues_agent_download,
        inputs=[shared.state_run, shared.state_overview_issues],
        outputs=[refs.export_issues_btn],
        show_progress="minimal",
        show_progress_on=refs.export_issues_btn,
    )
    # Arm the download as soon as load succeeds (heuristic issues already in
    # State) — same pattern as Export HTML — so a slow/failing judge cannot
    # leave the button stuck disarmed. Refresh after judge (load path and
    # button click) so judgments land in the file when available.
    # Without ANALYZE_*/LABEL_* configured, judge only shows the setup banner.
    for _ev in load_events:
        _ev.success(**export_event)
        _ev.then(**judge_event).then(**export_event)
    judge_dependency = refs.suggest_fixes_btn.click(**judge_event).then(**export_event)

    @safe_callback("Context utilization")
    def _rebuild_utilization(agent_key, window_limit, snapshot_key, steps, raw, dark):
        if not steps:
            return empty_plotly_fig(), ""
        key = agent_key or PRESSURE_ALL_AGENTS
        raw_dict = raw if isinstance(raw, dict) else None
        snapshot_step = (
            None if key == PRESSURE_ALL_AGENTS else parse_usage_snapshot(snapshot_key)
        )
        series = context_pressure_series(
            steps,
            agent_key=key,
            raw=raw_dict,
            window_limit=window_limit,
        )
        fig = build_context_pressure_chart(
            steps,
            dark=bool(dark),
            series=series,
            highlight_step=snapshot_step,
        )
        html_strip = format_context_pressure_html(
            series,
            steps=steps,
            raw=raw_dict,
            agent_key=key,
            snapshot_step=snapshot_step,
        )
        return fig, html_strip

    @safe_callback("Context utilization")
    def on_agent_change(agent_key, window_limit, steps, raw, dark):
        fig, html_strip = _rebuild_utilization(
            agent_key, window_limit, SNAPSHOT_CURRENT, steps, raw, dark,
        )
        return fig, html_strip, _snapshot_dropdown_update(steps or [], agent_key=agent_key)

    util_state = [shared.state_steps, shared.state_raw, shared.state_dark]
    util_outputs = [refs.diag_pressure_chart, refs.diag_pressure_html]
    refs.diag_pressure_agent.change(
        fn=on_agent_change,
        inputs=[refs.diag_pressure_agent, refs.diag_window_limit, *util_state],
        outputs=[*util_outputs, refs.diag_usage_snapshot],
    )
    refs.diag_window_limit.change(
        fn=_rebuild_utilization,
        inputs=[
            refs.diag_pressure_agent,
            refs.diag_window_limit,
            refs.diag_usage_snapshot,
            *util_state,
        ],
        outputs=util_outputs,
    )
    refs.diag_usage_snapshot.change(
        fn=_rebuild_utilization,
        inputs=[
            refs.diag_pressure_agent,
            refs.diag_window_limit,
            refs.diag_usage_snapshot,
            *util_state,
        ],
        outputs=util_outputs,
    )

    _empty_label_fig = go.Figure()
    _empty_label_fig.update_layout(
        template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    _badge_off = gr.update(value="", visible=False)

    def do_load_labels(upload_obj, dark=False, run=None, warning_html=None):
        """Load labels and reveal the summary parent only when it has content."""
        file_path = (upload_obj if isinstance(upload_obj, str) else upload_obj.name) if upload_obj is not None else None
        empty = _empty_label_fig
        has_warnings = (bool(isinstance(warning_html, str) and warning_html.strip())
                        or bool(run is not None and load_warnings_html(session_from_run(run, "")).strip()))

        def no_charts(badge, message, visible):
            return (badge, message, gr.update(visible=False), empty, empty,
                    gr.update(visible=False), empty, empty, gr.update(visible=False), empty,
                    gr.update(visible=visible))

        if not file_path or not os.path.isfile(file_path):
            return no_charts("", "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                             "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>",
                             has_warnings)
        try:
            payload = build_label_ui_payload(file_path, dark=bool(dark), run=run)
        except Exception as exc:
            error = ("<div role='alert' style='padding:0.5em;color:#dc2626;'>"
                     f"Labels could not be loaded: {html.escape(str(exc))}</div>")
            return no_charts(error, error, True)

        return (payload["badge_html"], payload["status_html"], gr.update(visible=True),
                payload["phase_count_fig"], payload["action_count_fig"], gr.update(visible=True),
                payload["phase_duration_fig"], payload["action_duration_fig"], gr.update(visible=True),
                payload["timeline_fig"], gr.update(visible=True))

    label_outputs = [
        upload.label_badge_html,
        refs.label_status_html,
        refs.label_charts_row1,
        refs.label_phase_count_chart,
        refs.label_action_count_chart,
        refs.label_charts_row2,
        refs.label_phase_dur_chart,
        refs.label_action_dur_chart,
        refs.label_timeline_row,
        refs.label_timeline_chart,
    ]
    label_inputs = [upload.label_file_upload, shared.state_dark, shared.state_run, upload.summary_banner]
    upload.label_load_btn.click(
        fn=do_load_labels,
        inputs=label_inputs,
        outputs=[*label_outputs, upload.summary_area],
    )
    upload.label_file_upload.change(
        fn=do_load_labels,
        inputs=label_inputs,
        outputs=[*label_outputs, upload.summary_area],
    )

    def _reset_labels():
        empty = _empty_label_fig
        return (
            _badge_off,
            (
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
            ),
            gr.update(visible=False),
            empty,
            empty,
            gr.update(visible=False),
            empty,
            empty,
            gr.update(visible=False),
            empty,
            gr.update(value=None),
        )

    # One trigger only: choosing a trajectory re-clicks Load in the browser
    # rather than running a second load (see `load.bind_load`).
    _reset_outputs = label_outputs + [upload.label_file_upload]
    if load_events:
        # Clear after the new run is stored, so the file-clear callback cannot
        # restore the previous run's warning visibility over the new load.
        for event in load_events:
            event.then(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
    else:
        upload.load_btn.click(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
    return (judge_dependency,)
