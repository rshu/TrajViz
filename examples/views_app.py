"""A small Gradio app of your own, built from the JSON views.

    pip install "agent-trajviz[app]"
    python examples/views_app.py             # http://127.0.0.1:7871

Load a trajectory to see its summary, health checks and issues; pick any chart
or table to see it drawn and as JSON; download every view as JSON; compare
two runs of the same task. Nothing here reaches into agent_trajviz internals:
the analysis comes from views, the charts from agent_trajviz.figures and the
components from agent_trajviz.dashboard.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

import agent_trajviz as tv
from agent_trajviz import dashboard, figures

CHART_IDS = [vid for vid, info in tv.VIEW_CATALOG.items() if info["kind"] == "chart"]
VIEW_CHOICES = [(f"{info['title']} ({info['kind']})", vid) for vid, info in tv.VIEW_CATALOG.items()]


def _upload_path(upload) -> str | None:
    if upload is None:
        return None
    return upload if isinstance(upload, str) else getattr(upload, "name", str(upload))


def _table(view: tv.View) -> tuple[list[str], list[list]]:
    rows = view.data if isinstance(view.data, list) else []
    headers: list[str] = []
    for row in rows:
        headers.extend(k for k in row if k not in headers)
    return headers, [[row.get(h) if not isinstance(row.get(h), (dict, list)) else str(row.get(h))
                      for h in headers] for row in rows]


def inspect_sessions(upload):
    """List saved OpenTelemetry sessions, selecting a sole session automatically."""
    path = _upload_path(upload)
    try:
        sessions = tv.list_sessions(path, format="otlp") if path else []
    except (tv.LoadError, OSError, ValueError):
        sessions = []
    return gr.update(choices=[(row.get("label") or row["id"], row["id"]) for row in sessions],
                     value=sessions[0]["id"] if len(sessions) == 1 else None, visible=bool(sessions))


def load_run(upload, session_id=None):
    """Load a trajectory; return (state, summary markdown, health rows, issue rows)."""
    path = _upload_path(upload)
    if path is None:
        return None, "Upload a trajectory file.", gr.update(value=None), gr.update(value=None)
    try:
        run = tv.load(path, session_id=session_id or None)
    except tv.LoadError as exc:
        return None, f"**Could not load the file:** {exc}", gr.update(value=None), gr.update(value=None)
    s = run.view("summary").data
    rate = s["tool_success_rate"]
    tokens = (s['tokens'] or {}).get('total')
    token_text = 'n/a' if tokens is None else f'{tokens:,}'
    if run.tokens_estimated:
        token_text += ' estimated'
    limited = run.format == 'otlp' or bool(run.metrics.get('tool_unknown_outcomes'))
    summary = (
        f"### {Path(path).name}\n"
        f"**{s['format_label']}** · {s['steps']} steps · {s['agents']} agent(s) · "
        f"{token_text} tokens · {s['tool_calls']} tool calls "
        f"({'n/a' if rate is None else f'{rate}%'} success) · {s['issues']} issue(s)"
        + (" · limited evidence" if limited else "")
    )
    health_headers, health_rows = _table(run.view("health"))
    issue_view = run.view("issues")
    issue_rows = [[i["kind"], i["title"], i["detail"], ", ".join(map(str, i["steps"]))] for i in issue_view.data]
    return (
        run,
        summary,
        gr.update(value=health_rows, headers=health_headers),
        gr.update(value=issue_rows, headers=["kind", "title", "detail", "steps"]),
    )


def show_view(run, view_id: str):
    """Draw one view: (plot update, table update, JSON text)."""
    if run is None or not view_id:
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False), ""
    view = run.view(view_id)
    plot = gr.update(value=figures.figure(view), visible=True) if view_id in figures.FIGURE_VIEWS \
        else gr.update(value=None, visible=False)
    headers, rows = _table(view)
    table = gr.update(value=rows, headers=headers, visible=True) if rows else gr.update(value=None, visible=False)
    return plot, table, view.to_json(indent=2)


def export_views(run):
    """Every view as JSON files, zipped; returns the zip path."""
    if run is None:
        raise gr.Error("Load a trajectory first.")
    work = Path(tempfile.mkdtemp(prefix="trajviz-views-"))
    tv.export_json(run, work / "views")
    return shutil.make_archive(str(work / "views"), "zip", work / "views")


def compare_runs(ref_upload, cmp_upload):
    """Compare two runs: (summary JSON, milestone figure, divergence figure)."""
    ref, cmp_ = _upload_path(ref_upload), _upload_path(cmp_upload)
    if ref is None or cmp_ is None:
        raise gr.Error("Upload both a reference and a compared trajectory.")
    try:
        views = tv.comparison_views(tv.compare(ref, cmp_))
    except tv.LoadError as exc:
        raise gr.Error(f"Could not load a file: {exc}") from None
    return (
        views["comparison_summary"].data,
        figures.figure(views["milestone_timeline"]),
        figures.figure(views["divergence"]),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TrajViz example", analytics_enabled=False) as app:
        gr.Markdown("# TrajViz example app\nBuilt on the public `agent_trajviz` API only.")
        state = gr.State(None)
        with gr.Tab("Analyze"):
            upload = gr.File(label="Trajectory", file_types=[".json", ".jsonl", ".zstd", ".zip"])
            session = gr.Dropdown(label="Captured session", choices=[], value=None, visible=False)
            summary = gr.Markdown()
            with gr.Row():
                health = gr.Dataframe(label="Health checks", interactive=False, wrap=True)
                issues = gr.Dataframe(label="Issues", interactive=False, wrap=True)
            view_id = gr.Dropdown(choices=VIEW_CHOICES, value="token_usage", label="View")
            plot = gr.Plot(visible=False, show_label=False)
            table = gr.Dataframe(visible=False, interactive=False, wrap=True, show_label=False)
            with gr.Accordion("View JSON", open=False):
                view_json = gr.Code(language="json", label="JSON")
            export_btn = gr.Button("Download every view as JSON")
            export_file = gr.File(label="views.zip")

            inspected = upload.change(inspect_sessions, inputs=upload, outputs=session, api_name="sessions")
            loaded = inspected.then(load_run, inputs=[upload, session], outputs=[state, summary, health, issues],
                                    api_name="load")
            selected = session.input(load_run, inputs=[upload, session], outputs=[state, summary, health, issues],
                                     api_name="select_session")
            for event in (loaded, selected):
                event.then(show_view, inputs=[state, view_id], outputs=[plot, table, view_json])
                event.then(lambda: gr.update(value=None), outputs=export_file)
            view_id.change(show_view, inputs=[state, view_id], outputs=[plot, table, view_json], api_name="view")
            export_btn.click(export_views, inputs=state, outputs=export_file, api_name="export")
        with gr.Tab("Compare"):
            with gr.Row():
                ref_file = gr.File(label="Reference trajectory")
                cmp_file = gr.File(label="Compared trajectory")
            compare_btn = gr.Button("Compare", variant="primary")
            comparison = gr.JSON(label="Comparison summary")
            with gr.Row():
                milestones = gr.Plot(show_label=False)
                divergence = gr.Plot(show_label=False)
            compare_btn.click(compare_runs, inputs=[ref_file, cmp_file],
                              outputs=[comparison, milestones, divergence], api_name="compare")
        with gr.Tab("All views") as all_views_tab:
            gr.Markdown("Every view of the loaded trajectory, rendered by `agent_trajviz.dashboard`.")

            # Rendered when the tab is opened, not on every load: a re-render
            # inside a tab would otherwise pull the browser onto that tab.
            @gr.render(inputs=state, triggers=[all_views_tab.select])
            def render_all(run):
                if run is None:
                    gr.Markdown("Load a trajectory in the Analyze tab.")
                    return
                dashboard.render_views(run.views(), open_first=0)

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7871)
