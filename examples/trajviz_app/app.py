"""An example dashboard built on the public agent_trajviz API.

    pip install "agent-trajviz[app]"
    python examples/trajviz_app/app.py           # http://127.0.0.1:7870

The example includes an Overview, Patterns, Attribution, Comparison,
Workflow, Raw Data, and an AI sidebar. It uses `agent_trajviz` to load runs
and read views, `agent_trajviz.figures` for charts, `agent_trajviz.llm` for
AI analysis, and `agent_trajviz.attribution` for failure attribution.

With a model connection configured through environment variables, opening
the AI sidebar for a loaded run with an empty chat sends trajectory excerpts
to the configured provider.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any

import gradio as gr
import panels
import styles

import agent_trajviz as tv
from agent_trajviz import attribution, figures, llm

SECTIONS = ["Summary", "Tools", "Agents", "Context Utilization", "Diagnostics", "Deep Dive", "Labels"]
FORMAT_CHOICES = [("Auto-detect", "")] + [(label, fid) for fid, label in tv.FORMATS.items()]
ROLE_FILTERS = ["Assistant", "User"]
FEATURE_FILTERS = ["Tool Calls", "Errors", "Reasoning"]
SUBTITLES = {
    "Summary": "Cost charts for this run. Issues (above) list what went wrong.",
    "Tools": "Tool usage, timing and outcomes.",
    "Agents": "Multi-agent breakdown, spawning relationships, and per-agent performance.",
    "Context Utilization": "Context-window occupancy over the run, compactions, and what fills the window.",
    "Diagnostics": "Failure chains, root causes, file activity and plan progress.",
    "Deep Dive": "Every metric, the hotspots and the per-step table.",
    "Labels": "Phase and action classification from a labeled JSON sidecar.",
}


def _path(upload: Any) -> str | None:
    if upload is None:
        return None
    return upload if isinstance(upload, str) else getattr(upload, "name", str(upload))


def _inspect_sessions(upload):
    """Inspect a saved capture without choosing a multi-session run implicitly."""
    path = _path(upload)
    try:
        sessions = tv.list_sessions(path, format="otlp") if path else []
    except (tv.LoadError, OSError, ValueError):
        sessions = []
    return gr.update(choices=[(row.get("label") or row["id"], row["id"]) for row in sessions],
                     value=sessions[0]["id"] if len(sessions) == 1 else None, visible=bool(sessions))


class Refs:
    """Every component the handlers update, by name."""


def build_app() -> gr.Blocks:  # noqa: C901 - one layout, read top to bottom
    r = Refs()
    with gr.Blocks(title="TrajViz", analytics_enabled=False) as app:
        state = gr.State(None)  # Loaded run, views, labels, and display state.

        with gr.Sidebar(label="AI Trajectory Analysis", open=False, position="right") as sidebar:
            gr.Markdown("### 🤖 AI Trajectory Analysis")
            r.ai_status = gr.HTML(_ai_status(None))
            r.chat = gr.Chatbot(height=460, label="Analysis")
            r.question = gr.Textbox(label="Question", placeholder="e.g. What is the biggest performance bottleneck?")
            r.clear_chat = gr.Button("Clear chat", size="sm", variant="secondary")

        gr.HTML("<div class='tv-header'><span class='tv-title'>TrajViz</span>"
                "<span class='tv-subtitle'>Coding Agent Trajectory Analysis Dashboard</span></div>")
        with gr.Row():
            r.fmt = gr.Dropdown(FORMAT_CHOICES, value="", label="Format", scale=1)
            with gr.Column(scale=2):
                r.traj_file = gr.File(label="Trajectory (.json / .jsonl / .zstd / .zip)",
                                      file_types=[".json", ".jsonl", ".zstd", ".zip"])
                r.session_selector = gr.Dropdown(label="Captured session", choices=[], value=None, visible=False)
                with gr.Row():
                    r.load_btn = gr.Button("Load Trajectory", variant="primary")
                    r.export_btn = gr.Button("Export HTML", interactive=False)
            with gr.Column(scale=2):
                r.labels_file = gr.File(label="Labels (optional)", file_types=[".json"])
                r.labels_btn = gr.Button("Load Labels")
        r.export_file = gr.File(label="HTML report", visible=False)
        r.banner = gr.HTML("")
        r.kpis = gr.HTML("")

        with gr.Tabs(visible=False) as r.tabs:
            with gr.Tab("Overview"):
                r.issues = gr.HTML("")
                r.fixes_btn = gr.Button("Suggest fixes with LLM (sends trajectory excerpts to the configured provider)",
                                        size="sm")
                r.session = gr.HTML("")
                with gr.Row():
                    with gr.Column(scale=1, min_width=160):
                        r.nav = gr.Radio(SECTIONS, value="Summary", label="Contents")
                    with gr.Column(scale=7):
                        r.sections = {}
                        with gr.Column(visible=True) as col:
                            r.sections["Summary"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Summary']}</div>")
                            with gr.Row():
                                r.tok_plot = gr.Plot(show_label=False)
                                r.dur_plot = gr.Plot(show_label=False)
                        with gr.Column(visible=False) as col:
                            r.sections["Tools"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Tools']}</div>")
                            r.tools_md = gr.Markdown("")
                            with gr.Row():
                                r.tool_plot = gr.Plot(show_label=False)
                                r.tool_dur_plot = gr.Plot(show_label=False)
                            r.outcome_plot = gr.Plot(show_label=False)
                            r.skill_plot = gr.Plot(show_label=False)
                        with gr.Column(visible=False) as col:
                            r.sections["Agents"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Agents']}</div>")
                            r.agent_cards = gr.HTML("")
                            r.swimlane_plot = gr.Plot(show_label=False)
                            r.agent_tok_plot = gr.Plot(show_label=False)
                        with gr.Column(visible=False) as col:
                            r.sections["Context Utilization"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Context Utilization']}</div>")
                            with gr.Row():
                                r.ctx_agent = gr.Dropdown(label="Agent", choices=[], value=None)
                                r.ctx_snapshot = gr.Dropdown(label="Window snapshot", choices=[], value=None)
                                r.ctx_window = gr.Number(label="Window limit (tokens)", value=None, precision=0,
                                                         info="Empty = detected, else an assumed 128,000")
                            r.ctx_html = gr.HTML("")
                            r.ctx_plot = gr.Plot(show_label=False)
                        with gr.Column(visible=False) as col:
                            r.sections["Diagnostics"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Diagnostics']}</div>")
                            r.diag_html = gr.HTML("")
                            r.files_plot = gr.Plot(show_label=False)
                            r.roots_html = gr.HTML("")
                            r.plan_plot = gr.Plot(show_label=False)
                        with gr.Column(visible=False) as col:
                            r.sections["Deep Dive"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Deep Dive']}</div>")
                            with gr.Accordion("Performance", open=True):
                                r.perf_md = gr.Markdown("")
                            with gr.Accordion("Hotspots", open=False):
                                r.hot_md = gr.Markdown("")
                            with gr.Accordion("Per-step metrics", open=False):
                                r.steps_md = gr.Markdown("")
                        with gr.Column(visible=False) as col:
                            r.sections["Labels"] = col
                            gr.HTML(f"<div class='tv-section-subtitle'>{SUBTITLES['Labels']}</div>")
                            r.labels_html = gr.HTML("<div class='tv-muted'>Upload a labels file and press Load Labels.</div>")
                            with gr.Row():
                                r.lab_phase = gr.Plot(show_label=False)
                                r.lab_action = gr.Plot(show_label=False)
                            with gr.Row():
                                r.lab_phase_dur = gr.Plot(show_label=False)
                                r.lab_action_dur = gr.Plot(show_label=False)
                            r.lab_timeline = gr.Plot(show_label=False)

            with gr.Tab("Patterns"):
                with gr.Accordion("Tool Sequence Patterns", open=True):
                    r.seq_html = gr.HTML("")
                with gr.Accordion("Failure Patterns", open=True):
                    r.fail_html = gr.HTML("")
                with gr.Accordion("Anti-Pattern Summary", open=True):
                    r.anti_html = gr.HTML("")

            with gr.Tab("Attribution"):
                gr.HTML("<div class='tv-section-subtitle'>Capability failure attribution — which workflow "
                        "capability broke, with tiered evidence. Needs the task's reference data.</div>")
                r.attr_btn = gr.Button("Diagnose failure", variant="primary", size="sm")
                with gr.Accordion("Override agent / instance / reference data", open=False), gr.Row():
                    r.attr_agent = gr.Textbox(label="Agent", placeholder="auto-detected from path")
                    r.attr_inst = gr.Textbox(label="Instance id", placeholder="auto-detected from path")
                    r.attr_root = gr.Textbox(label="Reference data root", placeholder="default")
                r.attr_html = gr.HTML("")

            with gr.Tab("Comparison"):
                with gr.Accordion("Run group (N trajectories)", open=True):
                    gr.Markdown("The loaded trajectory is the baseline; upload one or more runs of the same task.")
                    with gr.Row():
                        r.rg_fmt = gr.Dropdown(FORMAT_CHOICES, value="", label="Format", scale=1)
                        r.rg_files = gr.File(label="Runs", file_count="multiple", scale=3)
                    r.rg_btn = gr.Button("Build scorecard", variant="primary")
                    r.rg_html = gr.HTML("")
                    r.rg_plot = gr.Plot(show_label=False)
                    r.rg_behavior = gr.HTML("")
                with gr.Accordion("Pairwise comparison", open=False):
                    gr.Markdown("The loaded trajectory is compared against a reference run of the same task.")
                    with gr.Row():
                        r.pw_fmt = gr.Dropdown(FORMAT_CHOICES, value="", label="Reference format", scale=1)
                        r.pw_ref = gr.File(label="Reference trajectory", scale=2)
                        r.pw_anchor = gr.File(label="Anchor patch (optional)", file_types=[".patch", ".diff", ".txt"],
                                              scale=2)
                        r.pw_ref_labels = gr.File(label="Reference labels (optional)", file_types=[".json"], scale=2)
                    r.pw_btn = gr.Button("Compare", variant="primary")
                    r.pw_html = gr.HTML("")
                    with gr.Row():
                        r.pw_counts = gr.Plot(show_label=False)
                        r.pw_durations = gr.Plot(show_label=False)

            with gr.Tab("Workflow"):
                with gr.Row():
                    r.wf_roles = gr.CheckboxGroup(ROLE_FILTERS, value=ROLE_FILTERS, label="Role")
                    r.wf_features = gr.CheckboxGroup(FEATURE_FILTERS, value=[], label="Contains (any)")
                    r.wf_agents = gr.CheckboxGroup([], value=[], label="Agent", visible=False)
                    r.wf_search = gr.Textbox(label="Search", placeholder="text in a step")
                r.wf_count = gr.HTML("")
                with gr.Row():
                    with gr.Column(scale=1, min_width=140):
                        r.wf_toc = gr.HTML("")
                    with gr.Column(scale=3):
                        r.wf_cards = gr.HTML("")
                    with gr.Column(scale=2):
                        r.wf_step = gr.Dropdown(label="Step", choices=[], value=None)
                        r.wf_detail = gr.HTML(panels.step_detail(None))

            with gr.Tab("Raw Data"):
                r.raw = gr.Code(language="json", label="Converted trajectory", max_lines=40)

        _wire(app, r, state, sidebar)
    return app


# -- handlers ---------------------------------------------------------------------------

def _ai_status(ctx: dict | None) -> str:
    cfg = llm.config_from_env()
    if not cfg.ready:
        status = "Not configured · set " + ", ".join(cfg.missing)
    else:
        status = f"Ready · {cfg.model} · {cfg.provider}"
    if ctx:
        status += f" · {len(ctx['run'].steps)} steps in context"
    return f"<div class='tv-muted'>{panels.E(status)}</div>"


def _figure(view: tv.View) -> Any:
    return figures.figure(view)


def _context_outputs(ctx: dict, agent: str | None, window: Any, snapshot: str | None) -> tuple[str, Any]:
    run = ctx["run"]
    win = int(window) if isinstance(window, (int, float)) and window > 0 else None
    if win is not None and win == ctx.get("detected_window"):
        win = None  # the prefilled, detected window is not an override
    snap = int(snapshot) if snapshot not in (None, "", "current") and str(snapshot).isdigit() else None
    pressure = run.view("context_pressure", agent=agent, window_limit=win, snapshot_step=snap)
    usage = run.view("context_usage", agent=agent, snapshot_step=snap, window_limit=win)
    return panels.context_strip(pressure.data, usage.data), _figure(pressure)


def _snapshot_choices(ctx: dict, agent: str | None) -> dict:
    usage = ctx["run"].view("context_usage", agent=agent)
    choices = [(s["label"], s["value"]) for s in usage.data["snapshots"]]
    return gr.update(choices=choices, value=choices[0][1] if choices else None, visible=len(choices) > 1)


def _visible_steps(ctx: dict, roles: list[str], features: list[str], agents: list[str], search: str) -> list[int]:
    wf = ctx["views"]["workflow"].data
    keyword = (search or "").strip().lower()
    out = []
    for row, step in zip(wf["steps"], ctx["run"].steps, strict=True):
        if not set(roles) & set(row["filters"]):
            continue
        if features and not set(features) & set(row["filters"]):
            continue
        if agents and row["agent"] not in agents:
            continue
        if keyword and keyword not in json.dumps(step, ensure_ascii=False, default=str).lower():
            continue
        out.append(row["position"])
    return out


def _wire(app: gr.Blocks, r: Refs, state: gr.State, sidebar: Any) -> None:
    load_outputs = [state, r.banner, r.kpis, r.tabs, r.export_btn, r.issues, r.session, r.tok_plot, r.dur_plot,
                    r.tools_md, r.tool_plot, r.tool_dur_plot, r.outcome_plot, r.skill_plot, r.agent_cards,
                    r.swimlane_plot, r.agent_tok_plot, r.ctx_agent, r.ctx_snapshot, r.ctx_window, r.ctx_html,
                    r.ctx_plot, r.diag_html, r.files_plot, r.roots_html, r.plan_plot, r.perf_md, r.hot_md,
                    r.steps_md, r.seq_html, r.fail_html, r.anti_html, r.wf_agents, r.wf_count, r.wf_toc, r.wf_cards,
                    r.wf_step, r.wf_detail, r.raw, r.attr_html, r.ai_status, r.chat,
                    r.export_file, r.labels_file, r.labels_html, r.lab_phase, r.lab_action,
                    r.lab_phase_dur, r.lab_action_dur, r.lab_timeline, r.rg_html, r.rg_plot, r.rg_behavior,
                    r.pw_html, r.pw_counts, r.pw_durations, r.wf_roles, r.wf_features, r.wf_search]

    def reset_derived():
        # Labels, downloads and comparison panels describe one loaded baseline.
        # A new run must not inherit them, even if its upload fails or is cleared.
        return {
            r.export_file: gr.update(value=None, visible=False),
            r.labels_file: gr.update(value=None),
            r.labels_html: "<div class='tv-muted'>Upload a labels file and press Load Labels.</div>",
            **dict.fromkeys((r.lab_phase, r.lab_action, r.lab_phase_dur, r.lab_action_dur, r.lab_timeline,
                             r.rg_plot, r.pw_counts, r.pw_durations)),
            r.rg_html: "", r.rg_behavior: "", r.pw_html: "",
            r.wf_roles: ROLE_FILTERS, r.wf_features: [], r.wf_search: "",
        }

    def empty_load(message):
        text = (r.kpis, r.issues, r.session, r.tools_md, r.agent_cards, r.ctx_html, r.diag_html,
                r.roots_html, r.perf_md, r.hot_md, r.steps_md, r.seq_html, r.fail_html, r.anti_html,
                r.wf_count, r.wf_toc, r.wf_cards, r.raw, r.attr_html)
        plots = (r.tok_plot, r.dur_plot, r.tool_plot, r.tool_dur_plot, r.outcome_plot, r.skill_plot,
                 r.swimlane_plot, r.agent_tok_plot, r.ctx_plot, r.files_plot, r.plan_plot)
        return {
            **reset_derived(), **dict.fromkeys(text, ""), **dict.fromkeys(plots),
            state: None, r.banner: panels.error_banner(message), r.tabs: gr.update(visible=False),
            r.export_btn: gr.update(interactive=False), r.ctx_window: None,
            r.ctx_agent: gr.update(choices=[], value=None, visible=False),
            r.ctx_snapshot: gr.update(choices=[], value=None, visible=False),
            r.wf_agents: gr.update(choices=[], value=[], visible=False),
            r.wf_step: gr.update(choices=[], value=None), r.wf_detail: panels.step_detail(None),
            r.ai_status: _ai_status(None), r.chat: [],
        }

    def on_load(upload, fmt, session_id=None):
        path = _path(upload)
        if path is None:
            return empty_load("Choose a trajectory file first.")
        try:
            run = tv.load(path, format=fmt or None, session_id=session_id or None)
        except tv.LoadError as exc:
            return empty_load(f"Could not load {os.path.basename(path)}: {exc}")
        views = run.views()
        d = {vid: v.data for vid, v in views.items()}
        stats = d["context_pressure"]["stats"]
        detected = stats.get("window_limit") if stats.get("window_source") in ("declared", "model") else None
        limited = run.format == "otlp" or bool(d["metrics"].get("tool_unknown_outcomes"))
        ctx = {"run": run, "views": views, "name": os.path.basename(path), "labels": None,
               "detected_window": detected, "limited_evidence": limited}
        pressure = views["context_pressure"]
        agents = [(a["label"], a["key"]) for a in d["context_pressure"]["agents"]]
        ctx_html, ctx_fig = _context_outputs(ctx, None, None, None)
        wf_agents = [(a["label"], a["id"]) for a in d["workflow"]["agents"]]
        visible = _visible_steps(ctx, ROLE_FILTERS, [], [], "")
        step_choices = [(f"#{row['step']} {row['role']}", row["position"]) for row in d["workflow"]["steps"]]
        return {
            **reset_derived(), state: ctx,
            r.banner: panels.banner(ctx["name"], d["summary"], list(run.warnings)),
            r.kpis: panels.kpi_strip(d["summary"], d["metrics"], d["health"], d["issues"], d["agents"], d["steps"],
                                     limited_evidence=limited),
            r.tabs: gr.update(visible=True),
            r.export_btn: gr.update(interactive=True),
            r.issues: panels.issues_panel(d["issues"], limited_evidence=limited),
            r.session: panels.session_details(d["session"]),
            r.tok_plot: _figure(views["token_usage"]),
            r.dur_plot: _figure(views["step_durations"]),
            r.tools_md: panels.behavior_md(d["metrics"], d["diagnostics"]),
            r.tool_plot: _figure(views["tool_usage"]),
            r.tool_dur_plot: _figure(views["tool_durations"]),
            r.outcome_plot: _figure(views["tool_outcomes"]),
            r.skill_plot: _figure(views["skill_calls"]),
            r.agent_cards: panels.agent_cards(d["agents"]),
            r.swimlane_plot: _figure(views["agent_swimlane"]),
            r.agent_tok_plot: _figure(views["agent_tokens"]),
            r.ctx_agent: gr.update(choices=agents,
                                   value=pressure.data["agent"] if any(key == pressure.data["agent"] for _, key in agents) else None,
                                   visible=len(agents) > 2),
            r.ctx_snapshot: gr.update(choices=[("Current window", "current")], value="current", visible=False),
            r.ctx_window: detected,
            r.ctx_html: ctx_html,
            r.ctx_plot: ctx_fig,
            r.diag_html: panels.diagnostics_summary(d["failure_chains"], d["error_clusters"], d["bottlenecks"],
                                                    d["file_interactions"], d["context_pressure"], limited_evidence=limited),
            r.files_plot: _figure(views["file_interactions"]),
            r.roots_html: panels.root_causes(d["error_clusters"]),
            r.plan_plot: _figure(views["plan"]),
            r.perf_md: panels.performance_md(d["metrics"]),
            r.hot_md: panels.hotspots_md(d["hotspots"]),
            r.steps_md: panels.per_step_md(d["steps"]),
            r.seq_html: panels.tool_sequences_html(d["tool_sequences"]),
            r.fail_html: panels.failure_patterns_html(d["failure_patterns"], limited_evidence=limited),
            r.anti_html: panels.antipatterns_html(d, limited_evidence=limited),
            r.wf_agents: gr.update(choices=wf_agents, value=[], visible=bool(wf_agents)),
            r.wf_count: f"<div class='tv-muted'>Showing {len(visible)} of {len(run.steps)} steps</div>",
            r.wf_toc: panels.workflow_toc(d["workflow"], visible),
            r.wf_cards: panels.workflow_cards(d["workflow"], run.steps, visible),
            r.wf_step: gr.update(choices=step_choices, value=None),
            r.wf_detail: panels.step_detail(None),
            r.raw: json.dumps(run.raw, ensure_ascii=False, indent=2, default=str)[:2_000_000],
            r.attr_html: panels.attribution_html(attribution.diagnose(run)),
            r.ai_status: _ai_status(ctx),
            r.chat: [],
        }

    load_inputs = [r.traj_file, r.fmt, r.session_selector]
    r.load_btn.click(on_load, inputs=load_inputs, outputs=load_outputs, api_name="load")
    inspected = r.traj_file.change(_inspect_sessions, inputs=r.traj_file, outputs=r.session_selector,
                                   api_name="sessions")
    inspected.then(on_load, inputs=load_inputs, outputs=load_outputs)
    r.session_selector.input(on_load, inputs=load_inputs, outputs=load_outputs, api_name="select_session")

    def on_nav(section):
        return [gr.update(visible=name == section) for name in SECTIONS]

    r.nav.change(on_nav, inputs=r.nav, outputs=[r.sections[name] for name in SECTIONS])

    def on_ctx_agent(ctx, agent, window):
        if not ctx:
            return "", None, gr.update()
        html_out, fig = _context_outputs(ctx, agent, window, None)
        return html_out, fig, _snapshot_choices(ctx, agent)

    def on_ctx_change(ctx, agent, window, snapshot):
        if not ctx:
            return "", None
        return _context_outputs(ctx, agent, window, snapshot)

    r.ctx_agent.change(on_ctx_agent, inputs=[state, r.ctx_agent, r.ctx_window],
                       outputs=[r.ctx_html, r.ctx_plot, r.ctx_snapshot])
    for control in (r.ctx_window, r.ctx_snapshot):
        control.change(on_ctx_change, inputs=[state, r.ctx_agent, r.ctx_window, r.ctx_snapshot],
                       outputs=[r.ctx_html, r.ctx_plot])

    def on_labels(ctx, upload):
        path = _path(upload)
        if path is None:
            return {r.labels_html: "<div class='tv-muted'>Choose a labels file first.</div>"}
        try:
            views = tv.label_views(path)
        except tv.LoadError as exc:
            return {r.labels_html: panels.error_banner(str(exc))}
        if ctx:
            ctx["labels"] = path
        return {
            r.labels_html: panels.labels_badge(views["labels_summary"].data),
            r.lab_phase: _figure(views["label_phase_counts"]),
            r.lab_action: _figure(views["label_action_counts"]),
            r.lab_phase_dur: _figure(views["label_phase_durations"]),
            r.lab_action_dur: _figure(views["label_action_durations"]),
            r.lab_timeline: _figure(views["label_timeline"]),
            r.nav: "Labels",
        }

    r.labels_btn.click(on_labels, inputs=[state, r.labels_file],
                       outputs=[r.labels_html, r.lab_phase, r.lab_action, r.lab_phase_dur, r.lab_action_dur,
                                r.lab_timeline, r.nav], api_name="labels")

    def on_export(ctx):
        if not ctx:
            raise gr.Error("Load a trajectory first.")
        path = tv.write_report(ctx["run"], tempfile.mkdtemp(prefix="trajviz-report-"))
        return gr.update(value=str(path), visible=True)

    r.export_btn.click(on_export, inputs=state, outputs=r.export_file, api_name="export")

    def on_fixes(ctx):
        if not ctx:
            raise gr.Error("Load a trajectory first.")
        issues, errors = llm.suggest_fixes(ctx["run"])
        note = "".join(f"<div class='tv-muted'>⚠ {panels.E(e)}</div>" for e in errors)
        return panels.issues_panel(issues, limited_evidence=ctx.get("limited_evidence", False)) + note

    r.fixes_btn.click(on_fixes, inputs=state, outputs=r.issues, api_name="suggest_fixes")

    def on_attr(ctx, agent, inst, root):
        if not ctx:
            return "<div class='tv-muted'>Load a trajectory first.</div>"
        return panels.attribution_html(attribution.diagnose(ctx["run"], agent=agent or None,
                                                            instance_id=inst or None, reference_root=root or None))

    r.attr_btn.click(on_attr, inputs=[state, r.attr_agent, r.attr_inst, r.attr_root], outputs=r.attr_html)

    def on_run_group(ctx, uploads, fmt):
        if not ctx:
            return "<div class='tv-muted'>Load a trajectory first — it is the baseline.</div>", None, ""
        paths = [_path(u) for u in (uploads or []) if _path(u)]
        if not paths:
            return "<div class='tv-muted'>Upload at least one more run.</div>", None, ""
        try:
            result = tv.compare_runs([ctx["run"], *paths], format=fmt or None)
        except ValueError as exc:
            return panels.error_banner(str(exc)), None, ""
        views = tv.run_group_views(result)
        return (panels.scorecard_html(views["run_scorecard"].data), _figure(views["run_timelines"]),
                panels.behavior_html(views["run_behavior"].data))

    r.rg_btn.click(on_run_group, inputs=[state, r.rg_files, r.rg_fmt], outputs=[r.rg_html, r.rg_plot, r.rg_behavior],
                   api_name="run_group")

    def on_pairwise(ctx, ref_upload, anchor_upload, ref_labels_upload, fmt):
        if not ctx:
            return "<div class='tv-muted'>Load a trajectory first — it is the compared run.</div>", None, None
        ref_path = _path(ref_upload)
        if ref_path is None:
            return "<div class='tv-muted'>Upload a reference trajectory.</div>", None, None
        try:
            reference = tv.load(ref_path, format=fmt or None)
            report = tv.compare(reference, ctx["run"], anchor_patch=_path(anchor_upload))
        except (tv.LoadError, FileNotFoundError) as exc:
            return panels.error_banner(str(exc)), None, None
        views = tv.comparison_views(report)
        html_out = panels.comparison_html(views["comparison_summary"].data, views["evaluation_layers"].data)
        counts = durations = None
        ref_labels = _path(ref_labels_upload)
        if ref_labels and ctx.get("labels"):
            try:
                lv = tv.label_comparison_views(ref_labels, ctx["labels"], reference_name=os.path.basename(ref_path),
                                               compared_name=ctx["name"])
                counts = _figure(lv["label_phase_count_comparison"])
                durations = _figure(lv["label_phase_duration_comparison"])
            except tv.LoadError as exc:
                html_out += panels.error_banner(str(exc))
        return html_out, counts, durations

    r.pw_btn.click(on_pairwise, inputs=[state, r.pw_ref, r.pw_anchor, r.pw_ref_labels, r.pw_fmt],
                   outputs=[r.pw_html, r.pw_counts, r.pw_durations], api_name="compare")

    def on_workflow_filter(ctx, roles, features, agents, search):
        if not ctx:
            return "", "", ""
        visible = _visible_steps(ctx, roles or [], features or [], agents or [], search)
        workflow = ctx["views"]["workflow"].data
        return (f"<div class='tv-muted'>Showing {len(visible)} of {len(ctx['run'].steps)} steps</div>",
                panels.workflow_toc(workflow, visible),
                panels.workflow_cards(workflow, ctx["run"].steps, visible))

    for control in (r.wf_roles, r.wf_features, r.wf_agents, r.wf_search):
        control.change(on_workflow_filter, inputs=[state, r.wf_roles, r.wf_features, r.wf_agents, r.wf_search],
                       outputs=[r.wf_count, r.wf_toc, r.wf_cards])

    def on_step(ctx, position):
        if not ctx or position is None:
            return panels.step_detail(None)
        return panels.step_detail(ctx["run"].steps[int(position)])

    r.wf_step.change(on_step, inputs=[state, r.wf_step], outputs=r.wf_detail, api_name="step")

    # AI sidebar: the first pass runs when the panel is opened with a run loaded.
    def on_sidebar_open(ctx, history):
        if not ctx or history or not llm.config_from_env().ready:
            return history or []
        return llm.ask(ctx["run"], llm.first_question())

    sidebar.expand(on_sidebar_open, inputs=[state, r.chat], outputs=r.chat)

    def on_question(ctx, question, history):
        if not ctx:
            return (history or []) + [{"role": "assistant", "content": "Load a trajectory first."}], ""
        return llm.ask(ctx["run"], question, history), ""

    r.question.submit(on_question, inputs=[state, r.question, r.chat], outputs=[r.chat, r.question], api_name="ask")
    r.clear_chat.click(lambda: [], outputs=r.chat)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Example TrajViz dashboard built on the public API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args(argv)
    build_app().launch(server_name=args.host, server_port=args.port, css=styles.CSS)


if __name__ == "__main__":
    main()
