"""Normalized exports and custom steps must stay inside the public error contract."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.session import LoadError as SessionLoadError, load_session

FIXTURE = Path(__file__).parent / "fixtures" / "codearts_minimal.json"


def _document():
    return tv.load(FIXTURE).to_dict(include_raw=True)


@pytest.mark.parametrize("patch", [
    {"steps": [None]},
    {"steps": [{}]},
    {"schema_version": True},
    {"schema_version": 1.0},
    {"warnings": 42},
    {"warnings": "not an array"},
    {"warnings": {"warning": "not an array"}},
])
def test_invalid_export_containers_raise_public_errors(tmp_path, patch):
    document = {**_document(), **patch}
    with pytest.raises(ValueError):
        tv.Trajectory.from_dict(document)
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(tv.LoadError) as caught:
        tv.load(path)
    assert caught.value.code == "unsupported_schema"
    result = load_session(str(path))
    assert isinstance(result, SessionLoadError) and result.code == "read_error"


@pytest.mark.parametrize(("key", "value"), [
    ("tokens", 12), ("parts", 42), ("parts", [None]),
    ("tool_calls", [None]), ("tool_calls", [{}]),
    ("tool_calls", [{"tool_name": "bash", "status": {}}]),
    ("tool_calls", [{"tool_name": "bash", "status": "success", "metadata": True}]),
    ("index", "zero"), ("raw_index", True), ("tool_call_count", None),
    ("error_count", -1), ("role", []),
])
def test_invalid_normalized_step_fields_are_rejected(key, value):
    document = _document()
    document["steps"][0][key] = value
    with pytest.raises(ValueError, match="invalid trajectory step 0"):
        tv.Trajectory.from_dict(document)


def test_empty_export_has_the_same_no_steps_error_as_native_input(tmp_path):
    document = _document()
    document["steps"] = []
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(tv.LoadError) as caught:
        tv.load(path)
    assert caught.value.code == "no_steps"


def test_cli_rejects_bad_export_without_a_traceback(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**_document(), "warnings": 42}), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "agent_trajviz.insight", "--json", str(path),
                             "-o", str(tmp_path / "out")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "warnings must be an array" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("raw_index", [-1, 800])
def test_valid_export_keeps_indices_timing_and_source_payloads_exact(raw_index):
    document = _document()
    document["steps"][0].update(index=min(80, raw_index), raw_index=raw_index, duration=None)
    before = deepcopy(document)
    restored = tv.Trajectory.from_dict(document)
    assert restored.to_dict(include_raw=True) == before
    assert document == before


@pytest.mark.parametrize("parts", [42, None, [None, "text", 42]])
def test_custom_steps_complete_malformed_parts_without_crashing(parts):
    original = {"role": "assistant", "parts": parts}
    run = tv.Trajectory.from_steps([original])
    assert run.steps[0]["parts"] == []
    assert original == {"role": "assistant", "parts": parts}
    assert len(run.views()) == len(tv.VIEW_CATALOG)


def test_incomplete_custom_tool_call_gets_its_name_before_preview_rendering():
    original = {"role": "assistant", "tool_calls": [{}]}
    run = tv.Trajectory.from_steps([original])
    assert original == {"role": "assistant", "tool_calls": [{}]}
    assert run.steps[0]["tool_calls"][0]["tool_name"] == "?"
    assert len(run.views()) == len(tv.VIEW_CATALOG)
    restored = tv.Trajectory.from_dict(run.to_dict(include_raw=True))
    assert restored.steps == run.steps


@pytest.mark.parametrize("include_raw", [False, True])
def test_large_sorted_normalized_exports_remain_objects_with_jsonl_names(tmp_path, include_raw):
    run = tv.Trajectory.from_steps([{"role": "assistant", "raw_index": -1, "text_preview": "x" * 10000}])
    if include_raw:
        run.raw["extra"] = "y" * 10000
    document = run.to_dict(include_raw=include_raw)
    payload = json.dumps(document, sort_keys=True)
    assert payload.index('"type": "agent-trajviz-trajectory"') > 4096
    path = tmp_path / "sorted.jsonl"
    path.write_text(payload, encoding="utf-8")
    restored = tv.load(path)
    assert restored.to_dict(include_raw=include_raw) == document
    assert restored.views().keys() == run.views().keys()


@pytest.mark.parametrize("field", ["total", "input", "output", "cache_read", "cache_write", "reasoning"])
def test_custom_steps_handle_unrepresentable_counts_without_an_overflow(field):
    tokens = {"total": 12, "input": 10, "output": 2, "cache_read": 0, "cache_write": 0}
    tokens[field] = 10 ** 400
    run = tv.Trajectory.from_steps([{"role": "assistant", "tokens": tokens}])
    assert tokens[field] == 10 ** 400
    assert len(run.views()) == len(tv.VIEW_CATALOG)
    json.dumps({name: view.to_dict() for name, view in run.views().items()}, allow_nan=False)
    if field == "total":
        # A valid total can still be derived from the four observed components.
        assert run.steps[0]["tokens"]["total"] == 12
    elif field != "reasoning":
        labels = {"input": "Input Tokens", "output": "Output Tokens",
                  "cache_read": "Cache Read", "cache_write": "Cache Write"}
        assert labels[field] in run.steps[0]["_metrics_unavailable_fields"]


@pytest.mark.parametrize("field", ["duration", "time_created_ms", "time_completed_ms"])
def test_custom_steps_drop_unrepresentable_clocks(field):
    run = tv.Trajectory.from_steps([{"role": "assistant", field: 10 ** 400}])
    assert run.steps[0][field] is None
    assert len(run.views()) == len(tv.VIEW_CATALOG)


@pytest.mark.parametrize(("field", "value"), [("type", []), ("text", None)])
def test_malformed_normalized_part_fields_are_rejected(field, value):
    document = _document()
    document["steps"][0]["parts"] = [{"type": "text", "text": "Message", field: value}]
    with pytest.raises(ValueError, match=f"part {field} must be a string"):
        tv.Trajectory.from_dict(document)


def test_unknown_forward_tool_metadata_stays_exact():
    run = tv.Trajectory.from_steps([{"role": "assistant", "tool_calls": [{"tool_name": "custom",
        "metadata": {"future": {"values": [None, "x"], "flag": True}}}]}])
    document = run.to_dict(include_raw=True)
    assert tv.Trajectory.from_dict(document).to_dict(include_raw=True) == document


def test_absent_optional_source_index_stays_absent_on_roundtrip():
    document = _document()
    for step in document["steps"]:
        step.pop("raw_index", None)
    restored = tv.Trajectory.from_dict(document)
    assert restored.to_dict(include_raw=True) == document
    assert all("raw_index" not in step for step in restored.steps)
    assert len(restored.views()) == len(tv.VIEW_CATALOG)
