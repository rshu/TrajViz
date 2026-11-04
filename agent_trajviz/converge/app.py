"""Standalone Gradio app for Converge trajectory comparison.

The three views -- pairwise, batch and before/after -- are module-level
functions that :func:`build_ui` wires to the page, so each can be exercised
without a browser.

Errors render as a message, never a traceback. A failure the user can act on
(a bad file, a bad manifest, a bad rate) shows its reason, with server paths
cut back to the name the user knows the file by; anything else shows only its
type and goes to the server log -- the same rule as the Insight dashboard.
"""

from __future__ import annotations

import html
import logging
import math
import os
from pathlib import Path

import gradio as gr

from ..errors import TrajVizError
from ..i18n import text as ui_text
from ..i18n.ui import bind_language, language_selector, presentation
from .styles import CONVERGE_CSS

logger = logging.getLogger(__name__)

# Gradio 6 takes `css` / `theme` at launch(), not in the Blocks constructor
# (passing them there is deprecated and warned about). Same pattern as the
# Insight app: build_ui() is unstyled and every launcher spreads this.
LAUNCH_PRESENTATION = presentation({
    "css": CONVERGE_CSS,
    "theme": gr.themes.Soft(),
})

# Failures whose message describes the user's input rather than the program.
_EXPECTED_ERRORS = (ValueError, OSError, TrajVizError)


def _upload_path(upload) -> str | None:
    if upload is None:
        return None
    return upload.name if hasattr(upload, "name") else str(upload)


def _token_rate(value) -> float:
    """The Token Rate box as a positive finite float; a cleared box is ``None``."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        rate = math.nan
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Token rate must be a positive number.")
    return rate


def _upload_names(*paths: str | None) -> dict[str, str]:
    """Map each upload's server path (as given and resolved) to its file name."""
    shown: dict[str, str] = {}
    for path in paths:
        if path:
            for form in (path, os.path.realpath(path)):
                shown[form] = os.path.basename(path)
    return shown


def _root_names(root: Path) -> dict[str, str]:
    """Map the data root's prefix (as given and resolved) to nothing."""
    return {f"{form}{os.sep}": "" for form in (str(root), os.path.realpath(root))}


def _public(text: str, shown: dict[str, str]) -> str:
    # Longest first, so a directory prefix cannot pre-empt a file inside it.
    for server, public in sorted(shown.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(server, public)
    return text


def _error_html(action: str, exc: BaseException, shown: dict[str, str]) -> str:
    if isinstance(exc, _EXPECTED_ERRORS):
        detail = _public(str(exc), shown)
        detail_html = html.escape(detail)
    else:
        logger.error("%s failed", action, exc_info=exc)
        detail_html = ui_text("unexpected {error_type}. See the server log.", error_type=type(exc).__name__)
    return (
        "<div class='cvg-report'><div class='cvg-warning'>"
        f"<b>{ui_text(action + ' failed:')}</b> {detail_html}"
        "</div></div>"
    )


def _note_html(text: str) -> str:
    return f"<div class='cvg-report'><p>{ui_text(text)}</p></div>"


def _input_errors_html(errors: list[dict], shown: dict[str, str]) -> str:
    """Show failed batch inputs without exposing their server-side paths."""
    if not errors:
        return ""
    rows = []
    for error in errors:
        side = f"{error['side']}: " if error.get("side") else ""
        task = f"{side}{error.get('task_id', '(unknown task)')}"
        detail = _public(str(error.get("error") or "Comparison report is unavailable."), shown)
        rows.append(f"<li><b>{html.escape(task)}</b>: {html.escape(detail)}</li>")
    return f"<h3>{ui_text('Input errors ({count})', count=len(errors))}</h3><ul class='cvg-warning'>{''.join(rows)}</ul>"


def compare_uploads(ref_upload, cmp_upload, anchor_upload, rate, fuzzy) -> tuple:
    """Pairwise view: report HTML, three figures and the anchor-chart update."""
    from .charts import (
        _empty_figure,
        build_anchor_class_chart,
        build_divergence_waterfall_chart,
        build_milestone_timeline_chart,
        build_segment_cost_chart,
    )

    def _failed(report_html: str, message: str) -> tuple:
        empty = _empty_figure(message=message)
        return report_html, empty, empty, empty, gr.update(value=empty, visible=False)

    ref_path, cmp_path = _upload_path(ref_upload), _upload_path(cmp_upload)
    if ref_path is None or cmp_path is None:
        return _failed(
            _note_html("Please upload both a reference and compared trajectory."),
            "Upload both trajectory files to compare",
        )
    anchor_path = _upload_path(anchor_upload)
    try:
        from . import alignment
        from .rendering import build_comparison_report_html

        report = alignment.build_comparison_report(
            ref_file=ref_path,
            cmp_file=cmp_path,
            token_rate=_token_rate(rate),
            fuzzy_commands=bool(fuzzy),
            anchor_patch=anchor_path,
            task_id="",
        )
        # Milestones and segments are read from the report (single source of truth).
        seg_data = report.get("segments", {})
        anchor_analysis = report.get("anchor_analysis")
        return (
            build_comparison_report_html(report),
            build_milestone_timeline_chart(report.get("ref_milestones", {}), report.get("cmp_milestones", {})),
            build_segment_cost_chart(seg_data, seg_data.get("milestone_order_matches", False)),
            build_divergence_waterfall_chart(report.get("patterns", [])),
            gr.update(value=build_anchor_class_chart(anchor_analysis), visible=anchor_analysis is not None),
        )
    except Exception as exc:  # noqa: BLE001 - rendered, never raised into Gradio
        return _failed(
            _error_html("Comparison", exc, _upload_names(ref_path, cmp_path, anchor_path)),
            "Comparison failed",
        )


def _run_manifest(manifest_path: str, rate: float, fuzzy: bool, data_root: Path) -> list[dict]:
    from .batch import parse_manifest, run_batch

    entries = parse_manifest(manifest_path, base_dir=data_root, root=data_root)
    return run_batch(entries, token_rate=rate, fuzzy_commands=bool(fuzzy))


def batch_report_html(manifest_upload, rate, fuzzy, *, data_root: str | os.PathLike) -> str:
    """Batch view: run every task in an uploaded manifest and summarise them.

    Manifest paths resolve against *data_root* and must stay inside it.
    """
    from .batch import aggregate_reports, compute_pattern_frequency, promote_patterns

    manifest_path = _upload_path(manifest_upload)
    if manifest_path is None:
        return _note_html("Please upload a batch manifest JSON file.")
    root = Path(data_root)
    try:
        results = _run_manifest(manifest_path, _token_rate(rate), fuzzy, root)
        aggregate = aggregate_reports(results)
        frequency = compute_pattern_frequency(results)
        promoted = promote_patterns(frequency)
    except Exception as exc:  # noqa: BLE001 - rendered, never raised into Gradio
        return _error_html("Batch", exc, {**_root_names(root), **_upload_names(manifest_path)})

    parts = ['<div class="cvg-report">', f"<h2>{ui_text('Batch Report')}</h2>"]
    parts.append('<p>' + ui_text('{compared} of {total} tasks compared',
                                 compared=aggregate.get('compared_count', 0), total=aggregate.get('task_count', 0)) + '</p>')
    parts.append(_input_errors_html(
        [{"task_id": result.task_id, "error": result.error} for result in results if result.report is None],
        {**_root_names(root), **_upload_names(manifest_path)},
    ))

    metrics = aggregate.get("metrics", {})
    if metrics:
        parts.append(f"<h3>{ui_text('Aggregate Metrics')}</h3>")
        parts.append('<table class="cvg-outcome-table"><thead><tr>'
                     f'<th>{ui_text("Metric")}</th><th>{ui_text("Mean")}</th>'
                     f'<th>{ui_text("Median")}</th><th>P95</th></tr></thead><tbody>')
        for name, stats in metrics.items():
            parts.append(f"<tr><td>{html.escape(name)}</td>"
                         f"<td>{stats.get('mean', 0):.4f}</td>"
                         f"<td>{stats.get('median', 0):.4f}</td>"
                         f"<td>{stats.get('p95', 0):.4f}</td></tr>")
        parts.append("</tbody></table>")

    if frequency:
        parts.append(f"<h3>{ui_text('Pattern Frequency')}</h3>")
        parts.append('<table class="cvg-outcome-table"><thead><tr>'
                     f'<th>{ui_text("Pattern")}</th><th>{ui_text("Tasks")}</th>'
                     f'<th>{ui_text("Prevalence")}</th><th>{ui_text("Level")}</th></tr></thead><tbody>')
        for ptype, data in sorted(frequency.items(), key=lambda x: -x[1].get("count", 0)):
            level = promoted.get(ptype, "hypothesis")
            badge = "supported_finding" if level == "supported_finding" else "hypothesis"
            parts.append(f"<tr><td>{html.escape(ptype)}</td>"
                         f"<td>{data.get('count', 0)}</td>"
                         f"<td>{data.get('prevalence', 0)*100:.0f}%</td>"
                         f"<td><span class='cvg-badge'>{html.escape(badge)}</span></td></tr>")
        parts.append("</tbody></table>")

    parts.append("</div>")
    return "".join(parts)


def before_after_report_html(before_upload, after_upload, rate, fuzzy, *, data_root: str | os.PathLike) -> str:
    """Before/after view: pair two manifests' tasks and report the deltas.

    Manifest paths resolve against *data_root* and must stay inside it.
    """
    from .intervention import (
        build_intervention_report,
        compute_metric_deltas,
        compute_pattern_deltas,
        detect_guardrail_regressions,
        pair_tasks,
    )

    before_path, after_path = _upload_path(before_upload), _upload_path(after_upload)
    if before_path is None or after_path is None:
        return _note_html("Please upload both before and after manifests.")
    root = Path(data_root)
    try:
        token_rate = _token_rate(rate)
        before_results = _run_manifest(before_path, token_rate, fuzzy, root)
        after_results = _run_manifest(after_path, token_rate, fuzzy, root)
        paired, before_only, after_only = pair_tasks(before_results, after_results)
        metric_deltas = compute_metric_deltas(paired)
        pattern_deltas = compute_pattern_deltas(paired)
        guardrails = detect_guardrail_regressions(metric_deltas)
        report = build_intervention_report(
            before_path, after_path, paired, before_only, after_only,
            metric_deltas, pattern_deltas, guardrails)
    except Exception as exc:  # noqa: BLE001 - rendered, never raised into Gradio
        return _error_html("Before/after comparison", exc,
                           {**_root_names(root), **_upload_names(before_path, after_path)})

    parts = ['<div class="cvg-report">', f"<h2>{ui_text('Intervention Report')}</h2>"]
    parts.append('<p>' + ui_text('Paired tasks: {paired} · Unpaired before: {before} · Unpaired after: {after}',
                                 paired=report['paired_tasks'], before=report['unpaired_before'],
                                 after=report['unpaired_after']) + '</p>')
    parts.append(_input_errors_html(report.get("input_errors", []),
                                   {**_root_names(root), **_upload_names(before_path, after_path)}))

    rec = report.get("recommendation", "")
    if rec:
        parts.append(f'<div class="cvg-note"><b>{ui_text("Recommendation:")}</b> {html.escape(rec)}</div>')

    for g in report.get("guardrail_warnings", []):
        parts.append(f'<div class="cvg-warning">! {html.escape(g["warning"])}</div>')

    deltas = report.get("intervention_effect", {})
    if deltas:
        parts.append(f"<h3>{ui_text('Metric Deltas')}</h3>")
        parts.append('<table class="cvg-outcome-table"><thead><tr>'
                     f'<th>{ui_text("Metric")}</th><th>{ui_text("Before")}</th><th>{ui_text("After")}</th>'
                     f'<th>{ui_text("Delta")}</th><th>{ui_text("Direction")}</th></tr></thead><tbody>')
        for name, d in deltas.items():
            color = "#059669" if d["direction"] == "improved" else (
                "#dc2626" if d["direction"] == "regressed" else "#6b7280")
            parts.append(
                f'<tr><td>{html.escape(name)}</td>'
                f'<td>{d["before_mean"]:.4f}</td>'
                f'<td>{d["after_mean"]:.4f}</td>'
                f'<td>{d["delta"]:+.4f}</td>'
                f'<td style="color:{color};">{html.escape(str(d["direction"]))}</td></tr>')
        parts.append("</tbody></table>")

    parts.append("</div>")
    return "".join(parts)


def build_ui(*, data_root: str | os.PathLike | None = None) -> gr.Blocks:
    """Build and return the Converge Gradio Blocks app (style it via LAUNCH_PRESENTATION).

    *data_root* (default: the current directory) is where batch manifests'
    paths resolve; a manifest naming a file outside it is refused.
    """
    root = Path(data_root) if data_root is not None else Path.cwd()

    # No Gradio telemetry, as in the Insight app.
    with gr.Blocks(title="Converge", analytics_enabled=False) as app:
        with gr.Row(elem_id="tv-converge-header", equal_height=False):
            gr.Markdown("# Converge\nCross-agent trajectory comparison and alignment analysis.")
            language = language_selector()

        with gr.Row():
            with gr.Column(scale=1):
                ref_file = gr.File(label="Reference Trajectory", file_types=[".json", ".jsonl", ".zstd", ".zip"])
            with gr.Column(scale=1):
                cmp_file = gr.File(label="Compared Trajectory", file_types=[".json", ".jsonl", ".zstd", ".zip"])

        with gr.Row():
            anchor_patch = gr.File(label="Anchor Patch (optional)", file_types=[".patch", ".diff", ".txt"])
            token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            fuzzy_commands = gr.Checkbox(label="Fuzzy Commands", value=False)

        compare_btn = gr.Button("Compare", variant="primary")
        report_html = gr.HTML(label="Comparison Report")
        with gr.Row():
            milestone_plot = gr.Plot(show_label=False, label="Milestone Timeline")
        with gr.Row():
            segment_plot = gr.Plot(show_label=False, label="Segment Costs")
        with gr.Row():
            waterfall_plot = gr.Plot(show_label=False, label="Divergence Waterfall")
        with gr.Row():
            anchor_class_plot = gr.Plot(show_label=False, label="Anchor Write Recall by Class", visible=False)

        compare_btn.click(
            fn=compare_uploads,
            inputs=[ref_file, cmp_file, anchor_patch, token_rate, fuzzy_commands],
            outputs=[report_html, milestone_plot, segment_plot, waterfall_plot, anchor_class_plot],
        )

        # ── Batch ──────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Batch Comparison")
        gr.Markdown(
            "Upload a manifest JSON file to compare multiple trajectory pairs. Its paths are "
            "resolved against the directory the app was started in (`--data-root`) and must stay inside it."
        )
        with gr.Row():
            batch_file = gr.File(label="Batch Manifest (JSON)", file_types=[".json"])
            batch_token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            batch_fuzzy = gr.Checkbox(label="Fuzzy Commands", value=False)
        batch_btn = gr.Button("Run Batch", variant="primary")
        batch_report = gr.HTML(label="Batch Report")

        def do_batch(manifest_upload, rate, fuzzy):
            return batch_report_html(manifest_upload, rate, fuzzy, data_root=root)

        batch_btn.click(fn=do_batch, inputs=[batch_file, batch_token_rate, batch_fuzzy], outputs=[batch_report])

        # ── Before/After ───────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Before/After Comparison")
        gr.Markdown("Upload two manifest files to compare pre- and post-intervention batches.")
        with gr.Row():
            before_file = gr.File(label="Before Manifest (JSON)", file_types=[".json"])
            after_file = gr.File(label="After Manifest (JSON)", file_types=[".json"])
        with gr.Row():
            ba_token_rate = gr.Number(label="Token Rate", value=50.0, precision=1)
            ba_fuzzy = gr.Checkbox(label="Fuzzy Commands", value=False)
        ba_btn = gr.Button("Compare Before/After", variant="primary")
        ba_report = gr.HTML(label="Intervention Report")

        def do_before_after(before_upload, after_upload, rate, fuzzy):
            return before_after_report_html(before_upload, after_upload, rate, fuzzy, data_root=root)

        ba_btn.click(
            fn=do_before_after,
            inputs=[before_file, after_file, ba_token_rate, ba_fuzzy],
            outputs=[ba_report],
        )
        bind_language(app, language)

    return app
