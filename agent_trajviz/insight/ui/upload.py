"""Upload row, title, and load warnings."""

from __future__ import annotations

import atexit
import os
import shutil
import logging
import tempfile
from dataclasses import dataclass
from threading import Lock
from typing import Any

import gradio as gr

from ..presenters.overview import load_warnings_html
from ..report import ReportError, write_report_file
from ..session import LoadedSession
from .shared import SharedState, format_choices

logger = logging.getLogger(__name__)

# Picking a file re-clicks this button from the browser instead of binding a
# second Python load path; see `load.bind_load`.
LOAD_BTN_ELEM_ID = "tv-load-btn"


@dataclass
class UploadRefs:
    upload_accordion: gr.Column
    format_selector: gr.Dropdown
    file_upload: gr.File
    load_btn: gr.Button
    export_btn: gr.DownloadButton
    label_file_upload: gr.File
    label_load_btn: gr.Button
    summary_area: gr.Column
    summary_banner: gr.HTML
    label_badge_html: gr.HTML


def layout() -> UploadRefs:
    gr.HTML(
        "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px;'>"
        "<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;'>TrajViz</span>"
        "<span style='font-size:12px;color:var(--ov-muted);'>Coding Agent Trajectory Analysis Dashboard</span>"
        "</div>"
    )

    with gr.Column(elem_classes=["upload-row"]) as upload_accordion, gr.Row(equal_height=True):
        format_selector = gr.Dropdown(
            label="Format",
            choices=format_choices(),
            value="",
            interactive=True,
            scale=1,
            min_width=140,
        )
        with gr.Column(scale=2, min_width=200):
            file_upload = gr.File(
                label="Trajectory (.json / .jsonl / .zip)",
                file_types=[".json", ".jsonl", ".zip"],
                height=110,
            )
            with gr.Row():
                load_btn = gr.Button(
                    "Load Trajectory",
                    variant="primary",
                    size="sm",
                    min_width=120,
                    elem_id=LOAD_BTN_ELEM_ID,
                )
                export_btn = gr.DownloadButton(
                    label="Export HTML",
                    variant="secondary",
                    size="sm",
                    min_width=120,
                    interactive=False,
                )
        with gr.Column(scale=2, min_width=200):
            label_file_upload = gr.File(
                label="Labels (optional)",
                file_types=[".json"],
                height=110,
            )
            label_load_btn = gr.Button("Load Labels", variant="secondary", size="sm", min_width=120)

    with gr.Column(visible=False) as summary_area:
        summary_banner = gr.HTML("", elem_classes=["summary-banner"])
        label_badge_html = gr.HTML("")

    return UploadRefs(
        upload_accordion=upload_accordion,
        format_selector=format_selector,
        file_upload=file_upload,
        load_btn=load_btn,
        export_btn=export_btn,
        label_file_upload=label_file_upload,
        label_load_btn=label_load_btn,
        summary_area=summary_area,
        summary_banner=summary_banner,
        label_badge_html=label_badge_html,
    )


def load_slots(refs: UploadRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "summary_area": refs.summary_area,
        "upload_accordion": refs.upload_accordion,
        "summary_banner": refs.summary_banner,
        "export_btn": refs.export_btn,
    }


_TEMP_EXPORT_PREFIX = "trajviz-report-"
# Keyed by Gradio session hash (``None`` for direct/library callers). One process
# serves every viewer and each export deletes the directory it recorded, so a
# single module-level string meant the next viewer's export rmtree'd the report
# the previous viewer's Export HTML button was still pointing at. The cost of
# keying it is that abandoned sessions need cleanup on unload, with a normal
# interpreter-exit sweep as a fallback when the browser never sends unload.
_last_temp_export_dirs: dict[str | None, str] = {}
_temp_export_lock = Lock()


def _remove_temp_export_dir(directory: str | None) -> None:
    if directory and os.path.basename(directory).startswith(_TEMP_EXPORT_PREFIX):
        shutil.rmtree(directory, ignore_errors=True)


def cleanup_session_export(request: gr.Request) -> None:
    """Release only the departing browser session's temporary report."""
    session = getattr(request, "session_hash", None)
    if session is None:
        return
    with _temp_export_lock:
        _remove_temp_export_dir(_last_temp_export_dirs.pop(session, None))


def _cleanup_temp_exports() -> None:
    """Clean UI-owned reports only; standalone library/CLI exports stay usable."""
    with _temp_export_lock:
        for directory in _last_temp_export_dirs.values():
            _remove_temp_export_dir(directory)
        _last_temp_export_dirs.clear()


atexit.register(_cleanup_temp_exports)


def _export_off():
    return gr.update(value=None, interactive=False)


def _replace_temp_export_dir(new_path: str, session: str | None = None) -> None:
    """Drop *this session's* previous mkdtemp export dir once its replacement is written."""
    parent = os.path.dirname(os.path.abspath(new_path))
    with _temp_export_lock:
        previous = _last_temp_export_dirs.get(session)
        if os.path.basename(parent).startswith(_TEMP_EXPORT_PREFIX):
            _last_temp_export_dirs[session] = parent
        if previous != parent:
            _remove_temp_export_dir(previous)


def prepare_html_export(source, _steps=None, dark=False, request: gr.Request | None = None):
    """Build the HTML snapshot after a trajectory loads.

    Gradio's DownloadButton only saves in the same click that already has a
    file URL. Generating on click (then JS-clicking a hidden button) is
    blocked as a non-gesture download, and ``value=callable`` toasts on page
    load. So the report is prepared here and the button just downloads.
    """
    from ...api import Trajectory
    from ..session import session_from_run

    if isinstance(source, Trajectory):
        if not source.steps:
            return _export_off()
        payload: Any = session_from_run(source)
    else:
        payload = source if isinstance(source, dict) else {}
        if not payload or payload.get("_error"):
            return _export_off()
    directory = path = None
    try:
        # Gradio otherwise copies DownloadButton files into its shared,
        # content-addressed cache after this callback returns. Create the report
        # inside that cache so postprocessing serves this session-owned path
        # directly; cleanup then removes the actual served file too.
        cache_root = gr.utils.get_upload_folder()
        os.makedirs(cache_root, exist_ok=True)
        directory = tempfile.mkdtemp(prefix=_TEMP_EXPORT_PREFIX, dir=cache_root)
        path = write_report_file(payload, dest=directory, dark=bool(dark))
    except ReportError:
        return _export_off()
    except Exception:
        logger.exception("writing the HTML report failed")
        return _export_off()
    finally:
        if path is None:
            _remove_temp_export_dir(directory)
    # Gradio injects *request* on a browser call (annotation-driven) and leaves
    # it None for a direct call, which is the key library callers share.
    _replace_temp_export_dir(path, getattr(request, "session_hash", None))
    return gr.update(value=path, interactive=True)


def bind_export(refs: UploadRefs, shared: SharedState, load_events: tuple) -> None:
    """Enable Export HTML after a successful load (not as DownloadButton value=)."""
    for event in load_events:
        event.success(
            fn=prepare_html_export,
            inputs=[shared.state_run, shared.state_steps, shared.state_dark],
            outputs=refs.export_btn,
            show_progress="minimal",
            show_progress_on=refs.export_btn,
        )


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named upload-row values for a load (error banner when *session* is None).

    Export is always disarmed here so a reload cannot download the previous
    file; ``prepare_html_export`` re-enables the button after a successful load.
    """
    del dark
    export_off = _export_off()
    if session is None:
        has_banner = bool(banner)
        return {
            "summary_area": gr.update(visible=has_banner),
            "upload_accordion": gr.update(),
            "summary_banner": gr.update(value=banner, visible=has_banner),
            "export_btn": export_off,
        }
    warnings = load_warnings_html(session)
    # `summary_area` is a plain Column above the Tabs — never an accordion — so a
    # step-truncation notice (session.MAX_STEPS) cannot be collapsed out of
    # sight, and `report.py` repeats the same strip in the exported header.
    return {
        # Keep area visible so label badges can appear after Load Labels.
        "summary_area": gr.update(visible=True),
        "upload_accordion": gr.update(),
        "summary_banner": gr.update(value=warnings, visible=bool(warnings.strip())),
        "export_btn": export_off,
    }
