"""Labels section payload: phase/action charts and the upload badge."""

from __future__ import annotations

import html

from ..charts import (
    build_label_action_count_chart,
    build_label_action_duration_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_label_timeline_chart,
)
from ..labels import aggregate_labels, load_labeled_json
from ..palette import LABEL_PHASE_COLORS


def validate_labels_for_run(data: dict, run) -> None:
    """Reject contradictory identity evidence; sparse and legacy sidecars remain valid."""
    if run is None:
        return
    steps = {step["index"]: step for step in run.steps}
    seen = set()
    for record in data.get("steps", []):
        if not isinstance(record, dict):
            raise ValueError("Label records must be objects.")
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in steps:
            raise ValueError("Labels contain an index absent from the loaded trajectory.")
        if index in seen:
            raise ValueError("Labels contain duplicate step indices.")
        seen.add(index)
        step = steps[index]
        if record.get("role") and record["role"] != step.get("role"):
            raise ValueError("Label roles do not match the loaded trajectory.")
        if "raw_index" in record and "raw_index" in step and record["raw_index"] != step["raw_index"]:
            raise ValueError("Label source indices do not match the loaded trajectory.")
        if record.get("session_id") and step.get("session_id") and record["session_id"] != step["session_id"]:
            raise ValueError("Labels belong to a different captured session.")


def _label_provenance_html(data: dict) -> str:
    """Show recorded label origin without inferring how labels were produced."""
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    rows = [("Model", data.get("model")), ("Taxonomy", data.get("taxonomy_version")),
            ("Source note", data.get("source_note") or metadata.get("source_note"))]
    entries = "".join(
        f"<dt style='font-weight:600;'>{label}</dt>"
        f"<dd style='margin:0;overflow-wrap:anywhere;'>{html.escape(str(value))}</dd>"
        for label, value in rows if value is not None and str(value).strip()
    )
    if not entries:
        return ""
    return (
        "<details class='label-provenance' open style='font-size:12px;margin:6px 0;color:var(--ov-muted);'>"
        "<summary style='cursor:pointer;'>Label provenance</summary>"
        "<dl style='display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 12px;margin:6px 0;'>"
        f"{entries}</dl></details>"
    )


def build_label_ui_payload(file_path: str, dark: bool = False, *, run=None) -> dict:
    """Build UI-facing label payload for a *_labeled.json label file."""
    data = load_labeled_json(file_path)
    validate_labels_for_run(data, run)
    agg = aggregate_labels(data)
    pc_fig = build_label_phase_count_chart(agg["phase_counts"], dark=dark)
    ac_fig = build_label_action_count_chart(agg["action_counts"], agg["action_to_phase"], dark=dark)
    pd_fig = build_label_phase_duration_chart(agg["phase_durations"], dark=dark)
    ad_fig = build_label_action_duration_chart(agg["action_durations"], agg["action_to_phase"], dark=dark)
    tl_fig = build_label_timeline_chart(agg["steps"], dark=dark)

    n_steps = len(agg.get("steps", []))
    phase_counts = agg.get("phase_counts", {})
    n_phases = len(phase_counts)

    bar_segments = "".join(
        f"<span style='flex:{count};background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};height:8px;'"
        f" title='{html.escape(str(phase))}: {count}'></span>"
        for phase, count in phase_counts.items()
        if count > 0
    )
    phase_bar = (
        (
            "<div style='display:flex;border-radius:4px;overflow:hidden;width:200px;margin-top:4px;'>"
            f"{bar_segments}</div>"
        )
        if bar_segments
        else ""
    )

    phase_chips = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;"
        f"background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};color:white;'>"
        f"{html.escape(str(phase))}: {count}</span>"
        for phase, count in phase_counts.items()
        if count > 0
    )

    badge = (
        "<div style='display:flex;flex-direction:column;gap:6px;margin:6px 0;'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        "<span style='background:#059669;color:white;"
        "padding:4px 12px;border-radius:12px;font-size:13px;'>"
        f"Labels loaded — {n_steps} steps, {n_phases} phases</span>"
        "<a href='#' onclick=\"var t=document.querySelectorAll('button[role=tab]');"
        "for(var i=0;i<t.length;i++){if(t[i].textContent.trim()==='Overview'){t[i].click();break;}}"
        "var r=document.querySelector(&quot;.overview-section-radio input[value='Labels']&quot;);"
        "if(r){r.click();r.scrollIntoView({behavior:'smooth',block:'center'});}"
        "return false;\" style='font-size:12px;color:#059669;text-decoration:underline;cursor:pointer;'>"
        "Jump to Labels</a>"
        "</div>"
        f"<div style='display:flex;gap:4px;flex-wrap:wrap;'>{phase_chips}</div>"
        f"{phase_bar}"
        "</div>"
    )
    return {
        "badge_html": badge,
        "status_html": _label_provenance_html(data),
        "phase_count_fig": pc_fig,
        "action_count_fig": ac_fig,
        "phase_duration_fig": pd_fig,
        "action_duration_fig": ad_fig,
        "timeline_fig": tl_fig,
    }
