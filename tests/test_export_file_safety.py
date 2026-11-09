"""Exports preserve unrelated files and fail without corrupting completed output."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

import agent_trajviz as tv


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["trajectory.json", "index.json", "metrics.json"])
def test_export_refuses_to_replace_loaded_source_before_writing_any_file(tmp_path, filename):
    source = tmp_path / filename
    original = (REPO / "tests/fixtures/codearts_minimal.json").read_bytes()
    source.write_bytes(original)
    run = tv.load(source)
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(run, tmp_path)
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_export_checks_file_aliases_before_writing_any_output(tmp_path, link):
    source = tmp_path / "source.json"
    original = (REPO / "tests/fixtures/codearts_minimal.json").read_bytes()
    source.write_bytes(original)
    output = tmp_path / "export"
    output.mkdir()
    alias = output / "metrics.json"
    try:
        alias.symlink_to(source) if link == "symbolic" else os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"File links unavailable: {exc}")
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(tv.load(source), output)
    assert source.read_bytes() == alias.read_bytes() == original
    assert list(output.iterdir()) == [alias]


def test_json_cli_refuses_source_collision_with_a_controlled_error(tmp_path):
    source = tmp_path / "trajectory.json"
    original = (REPO / "tests/fixtures/codearts_minimal.json").read_bytes()
    source.write_bytes(original)
    result = subprocess.run([sys.executable, "-m", "agent_trajviz.insight", "--json", str(source), "-o", str(tmp_path)],
                            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and result.stdout == ""
    assert "overwrite input source" in result.stderr and "Traceback" not in result.stderr
    assert source.read_bytes() == original and list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("link", ["source", "symbolic", "hard"])
def test_native_html_report_refuses_input_aliases_before_rendering(tmp_path, link):
    source = tmp_path / "native.json"
    original = (REPO / "tests/fixtures/codearts_minimal.json").read_bytes()
    source.write_bytes(original)
    output = source if link == "source" else tmp_path / "report.html"
    if link != "source":
        try:
            output.symlink_to(source) if link == "symbolic" else os.link(source, output)
        except OSError as exc:
            pytest.skip(f"File links unavailable: {exc}")
    run = tv.load(source)
    with mock.patch("agent_trajviz.insight.report.build_report_html", side_effect=AssertionError("rendered before preflight")), \
            pytest.raises(tv.ReportError, match="overwrite input source"):
        tv.write_report(run, output)
    assert source.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_native_report_replacement_failure_preserves_previous_output(tmp_path):
    previous = tmp_path / "report.html"
    previous.write_bytes(b"previous report")
    with mock.patch("agent_trajviz._fileio.os.replace", side_effect=OSError("controlled report replacement failure")), \
            pytest.raises(OSError, match="controlled report replacement failure"):
        tv.write_report(tv.load(REPO / "tests/fixtures/codearts_minimal.json"), previous)
    assert previous.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [previous]


def test_native_report_cli_refuses_source_collision_without_a_traceback(tmp_path):
    source = tmp_path / "native.json"
    original = (REPO / "tests/fixtures/codearts_minimal.json").read_bytes()
    source.write_bytes(original)
    result = subprocess.run([sys.executable, "-m", "agent_trajviz.insight", "--report", str(source), "-o", str(source)],
                            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and result.stdout == ""
    assert "overwrite input source" in result.stderr and "Traceback" not in result.stderr
    assert source.read_bytes() == original


def _views(value):
    return {"probe": tv.View("probe", "Probe", "summary", "", {"value": value})}


def test_export_preserves_unrelated_temporary_files(tmp_path):
    sentinel = tmp_path / "probe.json.tmp"
    sentinel.write_text("unrelated user content", encoding="utf-8")
    tv.export_json(_views("new"), tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "unrelated user content"
    assert json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))["data"] == {"value": "new"}
    assert set(p.name for p in tmp_path.iterdir()) == {"probe.json.tmp", "probe.json", "index.json"}


def test_replace_failure_keeps_previous_output_and_cleans_owned_scratch(tmp_path):
    previous = tmp_path / "probe.json"
    previous.write_text('{"previous": true}', encoding="utf-8")
    sentinel = tmp_path / "probe.json.tmp"
    sentinel.write_text("unrelated", encoding="utf-8")
    with mock.patch("agent_trajviz.views.os.replace", side_effect=OSError("controlled replace failure")), \
            pytest.raises(OSError, match="controlled replace failure"):
        tv.export_json(_views("new"), tmp_path)
    assert previous.read_text(encoding="utf-8") == '{"previous": true}'
    assert sentinel.read_text(encoding="utf-8") == "unrelated"
    assert {p.name for p in tmp_path.iterdir()} == {"probe.json", "probe.json.tmp"}


def test_serialization_failure_leaves_no_partial_file(tmp_path):
    previous = tmp_path / "probe.json"
    previous.write_text("existing output", encoding="utf-8")
    with pytest.raises(TypeError):
        tv.export_json(_views(object()), tmp_path)
    assert previous.read_text(encoding="utf-8") == "existing output"
    assert list(tmp_path.iterdir()) == [previous]


def test_simultaneous_exports_write_complete_files_without_shared_scratch(tmp_path):
    def export(i):
        tv.export_json(_views({"writer": i, "text": "record " * 1500}), tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(export, range(24)))
    data = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))["data"]["value"]
    assert data["writer"] in range(24)
    assert data["text"] == "record " * 1500
    assert json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["views"][0]["file"] == "probe.json"
    assert {p.name for p in tmp_path.iterdir()} == {"probe.json", "index.json"}


@pytest.mark.parametrize("identity", ["../escape", "a/b", r"a\b", "index", "Index", "trajectory", "TRAJECTORY", "", 42])
def test_mapping_exports_refuse_unsafe_or_reserved_view_ids_before_writing(tmp_path, identity):
    outside = tmp_path / "escape.json"
    outside.write_text("unrelated", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="invalid view id"):
        tv.export_json({"alias": tv.View(identity, "Invalid", "summary", "", {})}, out)
    assert not out.exists()
    assert outside.read_text(encoding="utf-8") == "unrelated"


@pytest.mark.parametrize("other_id", ["same", "Same"])
def test_mapping_exports_refuse_duplicate_file_ids_before_writing(tmp_path, other_id):
    views = {"first": tv.View("same", "First", "summary", "", {}),
             "second": tv.View(other_id, "Second", "summary", "", {})}
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="duplicate view id"):
        tv.export_json(views, out)
    assert not out.exists()


def test_report_cli_output_path_error_has_no_traceback(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("unrelated", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "agent_trajviz.insight", "--report",
                             str(REPO / "tests/fixtures/codearts_minimal.json"), "-o", str(blocked / "report.html")],
                            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
    assert blocked.read_text(encoding="utf-8") == "unrelated"


def test_view_json_accepts_explicit_strict_serialization_options():
    view = tv.View("probe", "Probe", "summary", "", {"text": "café"})
    assert json.loads(view.to_json(allow_nan=False, ensure_ascii=True, indent=2))["data"] == {"text": "café"}
    invalid = tv.View("probe", "Probe", "summary", "", {"value": float("nan")})
    for options in ({}, {"allow_nan": False}):
        with pytest.raises(ValueError):
            invalid.to_json(**options)
