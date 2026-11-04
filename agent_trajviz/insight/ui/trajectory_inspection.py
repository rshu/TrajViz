"""Reference-free AWE inspection of the trajectory already loaded in this session."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gradio as gr

from ...i18n import text
from ..session import MAX_STEPS
from .shared import SharedState

logger = logging.getLogger(__name__)

_EMPTY = (
    "<div class='attr-banner' role='status'><div class='attr-banner-head'>" + text('Inspect a loaded trajectory') + "</div><div class='attr-banner-sub'>" + text('Load a session above to review its recorded actions, tool failures, recovery attempts, files and tests.') + '</div></div>'
)


@dataclass
class InspectionRefs:
    result_html: gr.HTML
    status: gr.HTML
    run_btn: gr.Button
    result: gr.State
    optional_controls: gr.Column


def layout() -> InspectionRefs:
    """Build the default AWE view; its only input is the shared loaded session."""
    gr.Markdown(
        "Inspect the session already loaded above. AWE runs locally on load; "
        "no extra files or model connection are required."
    )
    run_btn = gr.Button("Inspect trajectory", variant="primary", size="sm", scale=0,
                        elem_id="awe-inspect-trajectory")
    status = gr.HTML("<div class='attr-banner-sub'>" + text('No trajectory loaded.') + '</div>')
    with gr.Column(min_width=0) as optional_controls:
        pass
    result_html = gr.HTML(_EMPTY, elem_id="awe-inspection-result")
    result = gr.State(None)
    return InspectionRefs(result_html, status, run_btn, result, optional_controls)


def bind(refs: InspectionRefs, shared: SharedState, load_events) -> None:
    """Inspect after each load and clear old evidence when loading fails."""
    def on_inspect_trajectory(run):
        if run is None or not getattr(run, "steps", None):
            return _EMPTY, ("<div class='attr-banner-sub'>" + text('No trajectory loaded.') + '</div>'), None
        try:
            from ...awe import inspect_trajectory
            from ...awe.inspection_render import inspection_html

            result = inspect_trajectory(run)
            # Workflow renders a bounded prefix. Evidence from later rows remains
            # readable, but must not link to a card that the dashboard did not draw.
            displayed = dict(result)
            displayed["workflow_indices"] = [
                step["index"] for step in run.steps[:MAX_STEPS]
                if type(step.get("index")) is int and step["index"] >= 0
            ]
            rendered = inspection_html(displayed, workflow_links=True)
        except Exception:  # noqa: BLE001 -- clear stale evidence even on malformed plugin data
            logger.exception("Trajectory inspection failed")
            return (
                ("<div class='attr-banner' role='status'>" + text('Inspection could not be completed. Reload the session and try again; details are in the server log.') + '</div>'),
                ("<div class='attr-banner-sub'>" + text('Inspection unavailable.') + '</div>'), None,
            )
        status = (
            "<div class='attr-banner-sub' role='status'>"
            + text("Inspection complete — {count} recorded steps.", count=f"{len(run.steps):,}") + "</div>"
        )
        return rendered, status, result

    outputs = [refs.result_html, refs.status, refs.result]
    refs.run_btn.click(fn=on_inspect_trajectory, inputs=shared.state_run, outputs=outputs,
                       concurrency_id="awe-inspection", trigger_mode="always_last")
    for event in load_events:
        event.then(fn=on_inspect_trajectory, inputs=shared.state_run, outputs=outputs,
                   concurrency_id="awe-inspection", trigger_mode="always_last")
