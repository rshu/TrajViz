"""Exports and labeling protect every file actually consumed by a loader."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

import agent_trajviz as tv
from agent_trajviz.labeling import labeler

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("format_name", ["claude", "codex", "dsh", "bundle"])
def test_labeler_refuses_native_child_or_bundle_input_before_any_model_call(tmp_path, format_name):
    if format_name == "claude":
        root = tmp_path / "claude"
        shutil.copytree(FIXTURES / "otlp/native", root)
        source, output = root / "demo-session.jsonl", root / "demo-session/subagents/agent-agent-child.jsonl"
    elif format_name == "codex":
        root = tmp_path / "codex"
        shutil.copytree(FIXTURES / "codex_current", root)
        source, output = root / "root.jsonl", root / "root/subagents/agent-child.jsonl"
    elif format_name == "dsh":
        root = tmp_path / "dsh"
        shutil.copytree(FIXTURES / "dsh_native_v4", root)
        source, output = root / "root/session.v4.jsonl", root / "reviewer-a/session.v4.jsonl"
    else:
        root = tmp_path / "bundle"
        shutil.copytree(FIXTURES / "otlp", root)
        source, output = root / "manifest.json", root / "claude_code_traces.jsonl"
    original = output.read_bytes()
    with patch.object(labeler.llm, "call_llm", side_effect=AssertionError("No model call allowed")) as client, \
            pytest.raises(labeler.OutputSafetyError, match="overwrite input trajectory"):
        labeler.label_trajectory(str(source), str(output), base_url="http://127.0.0.1:9/v1",
                                 api_key="offline-test", model="offline-test")
    client.assert_not_called()
    assert output.read_bytes() == original


def test_labeler_refuses_to_overwrite_its_taxonomy_before_loading_or_calling_model(tmp_path):
    taxonomy = tmp_path / "taxonomy.md"
    original = labeler.DEFAULT_TAXONOMY.read_bytes()
    taxonomy.write_bytes(original)
    with patch.object(labeler, "load_all_steps") as load, patch.object(labeler.llm, "call_llm") as client, \
            pytest.raises(labeler.OutputSafetyError, match="input taxonomy"):
        labeler.label_trajectory(str(FIXTURES / "codearts_minimal.json"), str(taxonomy),
                                 taxonomy_path=str(taxonomy), base_url="http://127.0.0.1:9/v1",
                                 api_key="offline-test", model="offline-test")
    load.assert_not_called()
    client.assert_not_called()
    assert taxonomy.read_bytes() == original


@pytest.mark.parametrize("mode", ["json", "html"])
def test_bundle_member_is_protected_when_its_name_matches_an_export(tmp_path, mode):
    bundle = tmp_path / "bundle"
    shutil.copytree(FIXTURES / "otlp", bundle)
    manifest = bundle / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    previous = bundle / value["traces"][0]
    target = bundle / "metrics.json"
    previous.rename(target)
    value["traces"][0] = target.name
    manifest.write_text(json.dumps(value), encoding="utf-8")
    run = tv.load(manifest)
    original = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, tv.ReportError), match="overwrite input source"):
        tv.export_json(run, bundle) if mode == "json" else tv.write_report(run, target)
    assert {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()} == original


def test_reloaded_normalized_document_cannot_be_replaced_during_export(tmp_path):
    source = tmp_path / "trajectory.json"
    source.write_text(json.dumps(tv.load(FIXTURES / "codearts_minimal.json").to_dict(include_raw=True)), encoding="utf-8")
    original = source.read_bytes()
    run = tv.load(source)
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(run, tmp_path)
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("normalized", [False, True])
def test_relative_input_stays_protected_after_working_directory_changes(tmp_path, monkeypatch, normalized):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "trajectory.json"
    document = json.loads((FIXTURES / "codearts_minimal.json").read_text(encoding="utf-8"))
    if normalized:
        document = tv.load(FIXTURES / "codearts_minimal.json").to_dict(include_raw=True)
    source.write_text(json.dumps(document), encoding="utf-8")
    original = source.read_bytes()
    monkeypatch.chdir(source_dir)
    run = tv.load("trajectory.json")
    exported = run.to_dict(include_raw=True)
    assert "_input_files" not in exported["raw"]
    assert tv.Trajectory.from_dict(exported).to_dict(include_raw=True) == exported
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(run, source_dir)
    assert source.read_bytes() == original
    assert list(source_dir.iterdir()) == [source]
