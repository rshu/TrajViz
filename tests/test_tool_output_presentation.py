"""Structured Codex results stay intact while failures read as diagnostics."""
from __future__ import annotations

from copy import deepcopy
import html
import json

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.assistant import _error_step_lines
from agent_trajviz.insight.rendering import _format_tool_call_detail
from agent_trajviz.insight.tool_output import tool_output_text


def _block(text):
    return {"type": "input_text", "text": text}


def _result(code, output):
    return _block(json.dumps({"chunk_id": "controlled", "exit_code": code,
                             "output": output, "wall_time_seconds": 0.125}))


def _load(tmp_path, output, source='text(await tools.exec_command({cmd: "ruff check example.py"}));'):
    events = [
        {"type": "session_meta", "payload": {"id": "test", "cli_version": "0.164.0"}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "call_id": "cmd",
            "input": source}},
        {"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "cmd", "output": output}},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    return tv.load(path)


@pytest.mark.parametrize(("code", "stderr", "pattern"), [
    (7, "", "Process exited with code 7"),
    (7, "example.py:8:1: F401 imported but unused\nFound 1 error.", "example.py:8:1: F401 imported but unused"),
    (0, "Error: deliberate fixture text\nAll checks passed.", None),
])
def test_native_error_views_use_diagnostics_without_mutating_output(tmp_path, code, stderr, pattern):
    output = [_block("Script completed\nWall time 0.1 seconds\nOutput:\n"), _result(code, stderr)]
    original = deepcopy(output)
    run = _load(tmp_path, output)
    call = run.steps[0]["tool_calls"][0]
    assert call["status"] == ("error" if code else "success")
    clusters = run.view("error_clusters").data["clusters"]
    if pattern:
        assert clusters[0]["pattern"] == pattern
        assert pattern in "\n".join(_error_step_lines(run.steps))
    else:
        assert clusters == []
        assert run.metrics["tool_success"] == 1
    detail = _format_tool_call_detail(call)
    assert f"Process exited with code {code}" in detail
    assert html.escape(stderr) in detail
    assert "input_text" not in detail
    assert call["output"] == original
    assert run.raw["messages"][0]["parts"][0]["output"] == original
    for raw in [False, True]:
        restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=raw))))
        assert restored.steps[0]["tool_calls"][0]["output"] == original
        assert restored.view("error_clusters").data == run.view("error_clusters").data


def test_summary_prefers_failed_command_after_a_successful_batch_member(tmp_path):
    output = [_block("Script completed\nOutput:\n"), _result(0, "150 passed"),
              _result(7, "example.py:8:1: F841 local variable is unused\nFound 1 error.")]
    run = _load(tmp_path, output)
    assert run.view("error_clusters").data["clusters"][0]["pattern"] == "example.py:8:1: F841 local variable is unused"
    preview = "\n".join(_error_step_lines(run.steps))
    assert "F841" in preview
    assert "150 passed" not in preview
    detail = _format_tool_call_detail(run.steps[0]["tool_calls"][0])
    assert "150 passed" in detail and "F841" in detail


def test_outer_runtime_failure_remains_the_error_summary(tmp_path):
    output = [_result(0, "earlier command passed"), _block("Script failed\nError executing code: controlled crash")]
    run = _load(tmp_path, output)
    assert run.view("error_clusters").data["clusters"][0]["pattern"] == "Error executing code: controlled crash"


def test_detail_escapes_markup_and_keeps_full_native_payload(tmp_path):
    stderr = "<img src=x onerror=boom>\n" + "line\n" * 1000 + "native-tail"
    output = [_result(7, stderr)]
    run = _load(tmp_path, output)
    call = run.steps[0]["tool_calls"][0]
    detail = _format_tool_call_detail(call)
    assert "<img src=x" not in detail
    assert "&lt;img src=x" in detail
    # Detail keeps the full native payload (no mid-size truncation) while escaping markup.
    assert "native-tail" in detail
    assert "truncated" not in detail
    assert json.loads(call["output"][0]["text"])["output"].endswith("native-tail")
    assert run.to_dict(include_raw=True)["raw"]["messages"][0]["parts"][0]["output"] == output


def test_plain_text_and_opaque_blocks_keep_their_recorded_content():
    plain = "Error: original stderr\nextra detail"
    assert tool_output_text(plain, errors_only=True) == plain
    output = [_block("plain block"), {"type": "input_image", "image_url": "test-image"}]
    text = tool_output_text(output)
    assert "plain block" in text and '"image_url": "test-image"' in text
    assert output[1]["image_url"] == "test-image"


def test_single_quoted_native_command_is_readable_without_executing_javascript(tmp_path):
    source = "text(await tools.exec_command({cmd:'python -c \"raise SystemExit(7)\"'}));\n"
    run = _load(tmp_path, [_result(7, "")], source=source)
    call = run.steps[0]["tool_calls"][0]
    assert call["input"]["command"] == 'python -c "raise SystemExit(7)"'
    assert call["input"]["raw_input"] == source
    assert run.view("error_clusters").data["clusters"][0]["tool"] == "python"
