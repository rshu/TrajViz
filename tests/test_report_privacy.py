"""Temporary dashboard reports must not outlive their owning sessions."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import gradio as gr
from gradio.processing_utils import async_move_files_to_cache

from agent_trajviz import Trajectory
from agent_trajviz.insight import report
from agent_trajviz.insight.session import session_from_run
from agent_trajviz.insight.ui import upload


def _owned_report(root, name):
    directory = root / f"trajviz-report-{name}"
    directory.mkdir()
    path = directory / "report.html"
    path.write_text("synthetic report", encoding="utf-8")
    return path


def test_session_unload_preserves_other_viewers_and_direct_callers(tmp_path, monkeypatch):
    monkeypatch.setattr(upload, "_last_temp_export_dirs", {})
    first = _owned_report(tmp_path, "first")
    second = _owned_report(tmp_path, "second")
    direct = _owned_report(tmp_path, "direct")
    upload._replace_temp_export_dir(str(first), "first")
    upload._replace_temp_export_dir(str(second), "second")
    upload._replace_temp_export_dir(str(direct))

    upload.cleanup_session_export(SimpleNamespace(session_hash="first"))
    upload.cleanup_session_export(SimpleNamespace(session_hash="first"))
    upload.cleanup_session_export(SimpleNamespace(session_hash=None))

    assert not first.parent.exists()
    assert second.is_file()
    assert direct.is_file()
    upload._cleanup_temp_exports()
    assert not second.parent.exists()
    assert not direct.parent.exists()
    assert not upload._last_temp_export_dirs


def test_unload_handler_is_bound_without_a_public_api(tmp_path, monkeypatch):
    from agent_trajviz.insight.insight import build_ui

    monkeypatch.chdir(tmp_path)
    app = build_ui()
    callbacks = [fn for fn in app.fns.values() if fn.fn is upload.cleanup_session_export]
    assert len(callbacks) == 1
    assert any(target[1] == "unload" for target in callbacks[0].targets)
    assert callbacks[0].api_visibility == "private"


def test_normal_process_exit_removes_only_dashboard_owned_reports(tmp_path):
    # Exercise actual atexit registration in a fresh interpreter. CLI/library
    # reports returned to the caller must remain usable after that process ends.
    script = """
import json
import asyncio
import tempfile
import gradio as gr
from gradio.processing_utils import async_move_files_to_cache
from agent_trajviz import Trajectory
from agent_trajviz.insight.report import write_report_file
from agent_trajviz.insight.session import session_from_run
from agent_trajviz.insight.ui.upload import prepare_html_export
tempfile.tempdir = __import__('sys').argv[1]
run = Trajectory.from_steps([{'role': 'assistant', 'text_preview': 'synthetic'}])
session = session_from_run(run)
standalone = write_report_file(session)
update = prepare_html_export(run)
button = gr.DownloadButton()
served = asyncio.run(async_move_files_to_cache(button.postprocess(update['value']), button, postprocess=True))
dashboard = served['path']
assert dashboard == update['value']
print(json.dumps([standalone, dashboard]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False", "GRADIO_TEMP_DIR": str(tmp_path / "gradio-cache")},
        text=True, capture_output=True, check=True,
    )
    standalone, dashboard = map(Path, json.loads(result.stdout.strip()))
    assert standalone.is_file()
    assert not dashboard.parent.exists()


def test_gradio_serves_session_owned_files_without_leaving_cached_copies(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-cache"))
    monkeypatch.setattr(upload, "_last_temp_export_dirs", {})
    # Identical report bytes would normally share Gradio's content-hash cache.
    # Neither viewer may lose a download when the other unloads.
    monkeypatch.setattr(report, "build_report_html", lambda *args, **kwargs: "synthetic identical report")
    run = Trajectory.from_steps([{"role": "assistant", "text_preview": "synthetic"}])
    button = gr.DownloadButton()

    def prepare(session):
        update = upload.prepare_html_export(run, request=SimpleNamespace(session_hash=session))
        served = asyncio.run(async_move_files_to_cache(button.postprocess(update["value"]), button, postprocess=True))
        assert served["path"] == update["value"]
        return Path(served["path"])

    first, second = prepare("first"), prepare("second")
    assert first != second
    assert first.read_bytes() == second.read_bytes()
    assert set(tmp_path.rglob("*.html")) == {first, second}

    upload.cleanup_session_export(SimpleNamespace(session_hash="first"))
    assert not first.exists()
    assert second.is_file()
    assert set(tmp_path.rglob("*.html")) == {second}

    replacement = prepare("second")
    assert not second.exists()
    assert set(tmp_path.rglob("*.html")) == {replacement}
    upload.cleanup_session_export(SimpleNamespace(session_hash="second"))
    assert list(tmp_path.rglob("*.html")) == []


def test_failed_dashboard_export_leaves_no_report_in_gradio_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-cache"))

    def fail_render(*args, **kwargs):
        raise report.ReportError("synthetic render failure")

    monkeypatch.setattr(report, "build_report_html", fail_render)
    run = Trajectory.from_steps([{"role": "assistant", "text_preview": "synthetic"}])
    update = upload.prepare_html_export(run)
    assert update["value"] is None
    assert list((tmp_path / "gradio-cache").iterdir()) == []


def test_render_failure_allocates_no_scratch_directory(tmp_path, monkeypatch):
    session = session_from_run(Trajectory.from_steps([{"role": "assistant", "text_preview": "synthetic"}]))
    monkeypatch.setattr(report.tempfile, "tempdir", str(tmp_path))

    def fail_render(*args, **kwargs):
        raise report.ReportError("synthetic render failure")

    monkeypatch.setattr(report, "build_report_html", fail_render)
    with pytest.raises(report.ReportError):
        report.write_report_file(session)
    assert list(tmp_path.iterdir()) == []


def test_write_failure_removes_only_its_new_scratch_directory(tmp_path, monkeypatch):
    session = session_from_run(Trajectory.from_steps([{"role": "assistant", "text_preview": "synthetic"}]))
    monkeypatch.setattr(report.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(report, "build_report_html", lambda *args, **kwargs: "synthetic report")

    def fail_write(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(report, "open", fail_write, raising=False)
    with pytest.raises(OSError):
        report.write_report_file(session)
    assert list(tmp_path.iterdir()) == []

    user_directory = tmp_path / "user-output"
    user_directory.mkdir()
    keepsake = user_directory / "keep.txt"
    keepsake.write_text("preserve", encoding="utf-8")
    with pytest.raises(OSError):
        report.write_report_file(session, str(user_directory / "report.html"))
    assert keepsake.read_text(encoding="utf-8") == "preserve"
