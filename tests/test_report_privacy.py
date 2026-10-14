"""Temporary dashboard reports must not outlive their owning sessions."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
import tempfile
from agent_trajviz import Trajectory
from agent_trajviz.insight.report import write_report_file
from agent_trajviz.insight.session import session_from_run
from agent_trajviz.insight.ui.upload import _replace_temp_export_dir
tempfile.tempdir = __import__('sys').argv[1]
session = session_from_run(Trajectory.from_steps([{'role': 'assistant', 'text_preview': 'synthetic'}]))
standalone = write_report_file(session)
dashboard = write_report_file(session)
_replace_temp_export_dir(dashboard, 'synthetic-session')
print(json.dumps([standalone, dashboard]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"},
        text=True, capture_output=True, check=True,
    )
    standalone, dashboard = map(Path, json.loads(result.stdout.strip()))
    assert standalone.is_file()
    assert not dashboard.parent.exists()


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
