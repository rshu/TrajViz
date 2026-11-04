"""Attribution tab diagnosis callbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr

from ..loaders import detect_format
from .shared import SharedState, safe_callback
from .upload import UploadRefs


@dataclass
class AttributionRefs:
    attr_status_html: gr.HTML
    attr_run_btn: gr.Button
    attr_agent_override: gr.Textbox
    attr_inst_override: gr.Textbox
    attr_root_override: gr.Textbox
    attr_result_html: gr.HTML


def _diagnosis_identity(overview_raw, agent_override="", inst_override=""):
    """Infer editable case identifiers without treating an upload cache as an agent."""
    raw = overview_raw if isinstance(overview_raw, dict) else {}
    src = raw.get("_source_path") or ""
    src = os.fspath(src) if isinstance(src, (str, os.PathLike)) else ""
    detected = detect_format(overview_raw)
    fmt = None if detected == "unknown" else detected
    agent = (agent_override or "").strip()
    inst = (inst_override or "").strip()
    if src:
        if not inst:
            inst = os.path.splitext(os.path.basename(src))[0]
        parent = os.path.dirname(os.path.normpath(src))
        if not agent and os.path.basename(os.path.dirname(parent)) == "trajectory":
            agent = os.path.basename(parent)
    if not agent and fmt not in {None, "otlp"}:
        agent = fmt
    return src, fmt, agent, inst


def layout(*, allow_server_paths: bool = True, embedded: bool = False) -> AttributionRefs:
    """Build the attribution controls, optionally inside the shared AWE workspace."""
    with (gr.Column(min_width=0, elem_classes=["attribution-tab"]) if embedded
          else gr.TabItem("Attribution", elem_classes=["attribution-tab"])):
        gr.HTML(
            "<div class='section-subtitle'>Explain a failed attempt using its "
            "trajectory, reference solution and recorded test results.</div>"
        )
        gr.Markdown(
            "**1. Load a trajectory** above. **2. Confirm the task and reference data** below. "
            "**3. Read the finding, evidence and next check.** "
            "Choose Diagnose failure to run reference checks locally; no model request is sent."
        )
        with gr.Accordion(
            "Task and reference data", open=True,
            elem_classes=["per-message-acc", "attr-overrides"],
        ):
            gr.Markdown(
                "Agent and Instance id are filled from the session format and filename. "
                "Edit the detected values to match your reference data. "
                "The reference folder must contain the matching task, final agent patch, "
                "original trajectory and recorded test outcome."
            )
            with gr.Row(equal_height=True):
                attr_agent_override = gr.Textbox(
                    label="Agent", placeholder="e.g. opencode or codex", scale=1,
                )
                attr_inst_override = gr.Textbox(
                    label="Instance id", placeholder="Benchmark task ID", scale=2,
                )
                attr_root_override = gr.Textbox(
                    label="Reference data root", placeholder="Path to the benchmark folder", scale=2,
                    visible=allow_server_paths, interactive=allow_server_paths,
                )
            with gr.Accordion("Reference folder format", open=False):
                gr.Markdown(
                    "Use one matching task across these files. The reference trajectory must "
                    "match the uploaded session. Step labels are optional.\n\n"
                    "```text\nbenchmark/\n  data/\n    requirements/<task>.json\n"
                    "    patch/<agent>/<task>.diff\n    trajectory/<agent>/<task>.jsonl\n"
                    "    eval_<agent>.json\n```\n"
                    "The requirements file contains the task and reference patch; "
                    "the evaluation file records the test outcome. Native JSON sessions "
                    "can use a .json extension."
                )
        with gr.Row(equal_height=True):
            attr_run_btn = gr.Button("Diagnose failure", variant="primary", size="sm", scale=1, min_width=140)
        attr_status_html = gr.HTML(
            "<div class='attr-banner-sub'>Load a trajectory above to begin. "
            "Uploaded files may need an agent, task ID and matching reference folder.</div>"
        )
        attr_result_html = gr.HTML("")
    return AttributionRefs(
        attr_status_html=attr_status_html,
        attr_run_btn=attr_run_btn,
        attr_agent_override=attr_agent_override,
        attr_inst_override=attr_inst_override,
        attr_root_override=attr_root_override,
        attr_result_html=attr_result_html,
    )


def bind(
    refs: AttributionRefs,
    shared: SharedState,
    upload: UploadRefs,
    load_events,
    *,
    allow_server_paths: bool = True,
    auto_diagnose: bool = True,
) -> None:
    @safe_callback("Attribution diagnosis")
    def on_diagnose(overview_raw, agent_override, inst_override, root_override):
        from dataclasses import asdict

        from .. import attribution as _attr
        from ..rendering import build_attribution_html

        if not overview_raw:
            return (
                build_attribution_html(
                    {"available": False, "reason": "Load a trajectory in the Overview tab first."}
                ),
                "<div style='color:var(--ov-warn);padding:0.5em;font-size:13px;'>No trajectory loaded.</div>",
            )

        src, fmt, agent, inst = _diagnosis_identity(overview_raw, agent_override, inst_override)
        src_sha = overview_raw.get("_source_sha256") if isinstance(overview_raw, dict) else None
        # Enforced here, not only by hiding the box: a client can call the
        # event endpoint with any value it likes.
        root = (root_override or "").strip() if allow_server_paths else ""
        # Native parent/child trees carry one composite content identity. Keep
        # their merge context while the evaluator verifies the complete source
        # fingerprint and refuses any stale or mismatched displayed content.
        merged = overview_raw.get("_source_merged_count") or 0 if isinstance(overview_raw, dict) else 0

        result = _attr.diagnose(
            agent=agent or None,
            instance_id=inst or None,
            source_path=src or None,
            fmt=fmt or None,
            expected_sha=src_sha,
            merged_sources=merged,
            reference_root=root or None,
        )
        html_out = build_attribution_html(asdict(result))
        import html as _html

        ident = _html.escape(f"{result.agent}/{result.instance_id}")
        status = (
            "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>"
            f"Diagnosis complete &mdash; {ident}.</div>"
            if result.available
            else "<div style='color:var(--ov-muted);padding:0.5em;font-size:13px;'>"
            "No gold-grounded attribution &mdash; see the note below.</div>"
        )
        return html_out, status

    refs.attr_run_btn.click(
        fn=on_diagnose,
        inputs=[shared.state_raw, refs.attr_agent_override, refs.attr_inst_override, refs.attr_root_override],
        outputs=[refs.attr_result_html, refs.attr_status_html],
        concurrency_id="attribution",
    )

    def _clear_attribution():
        return (
            "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
            + ("Diagnosing the loaded trajectory&hellip;" if auto_diagnose
               else "Resetting optional reference attribution&hellip;") + "</div>",
            "",
        )

    def on_diagnose_autoload(overview_raw, root_override):
        if auto_diagnose:
            html_out, status = on_diagnose(overview_raw, "", "", root_override)
        else:
            html_out = ""
            status = (
                "<div class='attr-banner-sub'>Optional: confirm the matching reference data, "
                "then choose Diagnose failure to assess solution correctness.</div>"
                if overview_raw else "<div class='attr-banner-sub'>Load a trajectory above to begin.</div>"
            )
        _, _, agent, inst = _diagnosis_identity(overview_raw)
        return html_out, status, agent, inst

    # Keep one ordered load -> clear -> diagnose chain. A separate click
    # listener can clear a completed diagnosis when Gradio schedules it late.
    for _ev in load_events:
        cleared = _ev.then(
            fn=_clear_attribution,
            outputs=[refs.attr_result_html, refs.attr_status_html],
            concurrency_id="attribution",
        )
        cleared.then(
            fn=on_diagnose_autoload,
            inputs=[shared.state_raw, refs.attr_root_override],
            outputs=[refs.attr_result_html, refs.attr_status_html, refs.attr_agent_override, refs.attr_inst_override],
            concurrency_id="attribution",
        )
