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
from .shared import SharedState, drop_latest_run, format_choices

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
    session_selector: gr.Dropdown


def layout() -> UploadRefs:
    gr.HTML(
        "<div class='tv-header'>"
        "<span class='tv-brand'>TrajViz</span>"
        "<span class='tv-subtitle'>Coding Agent Trajectory Analysis Dashboard</span>"
        "</div>",
        elem_classes=["app-title"],
    )

    with gr.Column(elem_classes=["upload-row"]) as upload_accordion, gr.Row(equal_height=False):
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
                label="Trajectory (.json / .jsonl / .zstd / .zip)",
                file_types=[".json", ".jsonl", ".zstd", ".zip"],
                height=110,
            )
            gr.Markdown(
                "Subagents: upload a Claude Code ZIP or a consolidated session export. "
                "OTLP captures: upload traces or a manifest bundle with companion logs.",
            )
            session_selector = gr.Dropdown(label="Captured session", choices=[], value=None,
                                           visible=False, interactive=True)
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
            label_load_btn = gr.Button("Load Labels", variant="secondary", size="sm", min_width=120, scale=0)

    with gr.Column(visible=False) as summary_area:
        summary_banner = gr.HTML("", container=False, padding=False)
        # Hidden until Load Labels fills it — an empty gr.HTML still paints a
        # padded block (~20px) between the upload row and the KPI strip.
        label_badge_html = gr.HTML("", visible=False)

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
        session_selector=session_selector,
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
_TEMP_ISSUES_EXPORT_PREFIX = "trajviz-issues-"
_TEMP_EXPORT_PREFIXES = (_TEMP_EXPORT_PREFIX, _TEMP_ISSUES_EXPORT_PREFIX)
_EXPORT_KIND_BY_PREFIX = {
    _TEMP_EXPORT_PREFIX: "report",
    _TEMP_ISSUES_EXPORT_PREFIX: "issues",
}
# Keyed by (Gradio session hash, export kind). One process serves every viewer;
# replacing an export of the same kind deletes only that kind's previous dir so
# Export HTML and Export issues can coexist. Abandoned sessions need cleanup on
# unload, with a normal interpreter-exit sweep as a fallback.
_last_temp_export_dirs: dict[tuple[str | None, str], str] = {}
_temp_export_lock = Lock()


def _is_temp_export_dir(directory: str | None) -> bool:
    if not directory:
        return False
    base = os.path.basename(directory)
    return any(base.startswith(prefix) for prefix in _TEMP_EXPORT_PREFIXES)


def _remove_temp_export_dir(directory: str | None) -> None:
    if _is_temp_export_dir(directory):
        shutil.rmtree(directory, ignore_errors=True)


def cleanup_session_export(request: gr.Request) -> None:
    """Release only the departing browser session's temporary exports and load record."""
    session = getattr(request, "session_hash", None)
    if session is None:
        return
    drop_latest_run(session)
    with _temp_export_lock:
        stale = [key for key in _last_temp_export_dirs if key[0] == session]
        for key in stale:
            _remove_temp_export_dir(_last_temp_export_dirs.pop(key, None))


def _cleanup_temp_exports() -> None:
    """Clean UI-owned reports only; standalone library/CLI exports stay usable."""
    with _temp_export_lock:
        for directory in _last_temp_export_dirs.values():
            _remove_temp_export_dir(directory)
        _last_temp_export_dirs.clear()


atexit.register(_cleanup_temp_exports)


def _export_off():
    return gr.update(value=None, interactive=False)


def _replace_temp_export_dir(
    new_path: str,
    session: str | None = None,
    *,
    kind: str = "report",
) -> None:
    """Drop *this session's* previous mkdtemp dir for *kind* once its replacement is written."""
    parent = os.path.dirname(os.path.abspath(new_path))
    key = (session, kind)
    with _temp_export_lock:
        previous = _last_temp_export_dirs.get(key)
        if _is_temp_export_dir(parent):
            _last_temp_export_dirs[key] = parent
        if previous != parent:
            _remove_temp_export_dir(previous)


def stage_gradio_download(
    writer,
    *,
    prefix: str = _TEMP_EXPORT_PREFIX,
    request: gr.Request | None = None,
):
    """Write a DownloadButton file under Gradio's upload cache; return ``gr.update``.

    *writer* receives the temporary directory path and must return the absolute
    file path to serve. On failure the temp dir is removed and the button is
    disarmed.
    """
    kind = _EXPORT_KIND_BY_PREFIX.get(prefix)
    if kind is None:
        raise ValueError(f"unknown export temp prefix: {prefix!r}")
    directory = path = None
    try:
        cache_root = gr.utils.get_upload_folder()
        os.makedirs(cache_root, exist_ok=True)
        directory = tempfile.mkdtemp(prefix=prefix, dir=cache_root)
        path = writer(directory)
    except Exception:
        logger.exception("staging Gradio download failed")
        return _export_off()
    finally:
        if path is None:
            _remove_temp_export_dir(directory)
    _replace_temp_export_dir(
        path, getattr(request, "session_hash", None), kind=kind,
    )
    return gr.update(value=path, interactive=True)


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
            "summary_banner": gr.update(value=f'<div class="summary-banner">{banner}</div>' if has_banner else "", visible=has_banner),
            "export_btn": export_off,
        }
    warnings = load_warnings_html(session)
    # `summary_area` is a plain Column above the Tabs — never an accordion — so a
    # step-truncation notice (session.MAX_STEPS) cannot be collapsed out of
    # sight, and `report.py` repeats the same strip in the exported header.
    # Kept visible after load so label badges can appear (badge HTML starts hidden).
    return {
        # Label callbacks reveal this parent when their own badge has content.
        "summary_area": gr.update(visible=bool(warnings.strip())),
        "upload_accordion": gr.update(),
        "summary_banner": gr.update(value=f'<div class="summary-banner">{warnings}</div>' if warnings.strip() else "", visible=bool(warnings.strip())),
        "export_btn": export_off,
    }
