"""Native v2 tool results can fail despite a completed transport invocation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile

import pytest

import agent_trajviz as tv
from agent_trajviz.awe.inspection import inspect_trajectory
from agent_trajviz.awe.inspection_render import inspection_html
from agent_trajviz.insight.tool_failure import tool_call_failed
from agent_trajviz.insight.rendering import format_step_detail, render_workflow_html
from agent_trajviz.labeling.taxonomy import build_step_message


def capture(name: str, metadata: dict, output: str = "Tool output") -> dict:
    return {
        "info": {"id": "ses_native", "title": "Native tool outcome", "location": {"directory": "/workspace"}},
        "messages": [
            {"id": "msg_user", "type": "user", "sessionID": "ses_native",
             "time": {"created": 1000}, "text": "Run the focused check."},
            {"id": "msg_assistant", "type": "assistant", "sessionID": "ses_native",
             "time": {"created": 1100, "completed": 3000}, "finish": "tool-calls",
             "content": [{"type": "tool", "id": "call_native", "name": name,
                          "time": {"created": 1200, "ran": 1300, "completed": 2900},
                          "state": {"status": "completed", "input": {"command": "pytest -q"},
                                    "content": [{"type": "text", "text": output}], "metadata": metadata}}]},
        ],
    }


def load(native: dict):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native.json"
        path.write_text(json.dumps(native), encoding="utf-8")
        return tv.load(path)


@pytest.mark.parametrize(("name", "metadata", "output", "error"), [
    ("execute", {"error": True, "toolCalls": []}, "SyntaxError: Illegal return statement", "Tool reported an error"),
    ("shell", {"status": "completed", "timeout": True, "truncated": False},
     "Timed out: command exceeded its timeout", "Tool timed out"),
])
def test_native_boolean_failure_is_visible_to_metrics_and_label_prompt(name, metadata, output, error):
    native = capture(name, metadata, output)
    original = deepcopy(native)
    run = load(native)
    step = run.steps[1]
    call = step["tool_calls"][0]

    assert run.format == "opencode"
    assert call["status"] == "error"
    assert call["error"] == error
    assert call["output"] == output
    assert tool_call_failed(call)
    assert run.metrics["tool_fail"] == 1
    assert run.metrics["tool_success"] == 0
    assert step["error_count"] == 1
    assert "1 tool err" in render_workflow_html(run.steps)
    assert "Errors 1" in " ".join(re.sub(r"<[^>]+>", " ", format_step_detail(step)).split())
    prompt = build_step_message(step)
    assert f"{name} (error)" in prompt
    assert f"ERROR: {error}" in prompt

    # Retain the real completed invocation and its exact native result flags.
    assert call["metadata"]["opencode_v2_tool"] == original["messages"][1]["content"][0]
    assert {key: call["metadata"][key] for key in metadata} == metadata
    assert run.raw["_opencode_v2_messages"] == original["messages"]
    assert native == original


@pytest.mark.parametrize("key", ["error", "timeout"])
@pytest.mark.parametrize("value", [False, None, 0, 1, "true", {}, []])
def test_only_literal_true_flags_change_native_outcome(key, value):
    run = load(capture("execute" if key == "error" else "shell", {key: value}))
    call = run.steps[1]["tool_calls"][0]
    assert call["status"] == "completed"
    assert call["error"] is None
    assert not tool_call_failed(call)
    assert run.metrics["tool_fail"] == 0
    assert run.metrics["tool_success"] == 1
    assert run.steps[1]["error_count"] == 0


def test_absent_flags_and_nested_arbitrary_flags_are_not_failure_signals():
    for metadata in [{}, {"custom": {"error": True, "timeout": True}}]:
        run = load(capture("shell", metadata))
        assert run.steps[1]["tool_calls"][0]["status"] == "completed"
        assert run.metrics["tool_fail"] == 0


@pytest.mark.parametrize(("exit_code", "failures"), [(0, 0), (1, 1)])
def test_native_exit_semantics_remain_unchanged(exit_code, failures):
    run = load(capture("shell", {"exit": exit_code}))
    call = run.steps[1]["tool_calls"][0]
    assert call["status"] == "completed"
    assert call["error"] is None
    assert tool_call_failed(call) is bool(failures)
    assert run.metrics["tool_fail"] == failures
    assert run.metrics["tool_success"] == 1 - failures
    assert run.steps[1]["error_count"] == failures
    detail_html = format_step_detail(run.steps[1])
    if failures:
        assert "1 tool err" in render_workflow_html(run.steps)
        detail = " ".join(re.sub(r"<[^>]+>", " ", detail_html).split())
        assert "Errors 1" in detail
        assert "dp-section-tool-error" in detail_html
        assert "Exit code 1" in detail
        expanded = render_workflow_html(run.steps, expanded=True)
        assert "status: <code>error</code>" in expanded
        assert "<pre>Exit code 1</pre>" in expanded
        inspection = inspect_trajectory(run)
        failure, = inspection["failures"]
        assert failure["status"] == "completed"
        rendered = inspection_html(inspection)
        assert re.search(r"<td><span [^>]*>Error</span></td><td>1</td>", rendered)
    else:
        assert "dp-section-tool-error" not in detail_html


def test_genuine_empty_interruption_remains_a_request_error_without_tool_errors():
    native = capture("shell", {})
    assistant = native["messages"][1]
    assistant.update({"content": [], "finish": "error", "error": {"type": "interrupted", "message": "Interrupted"}})
    run = load(native)
    step = run.steps[1]
    assert step["tool_calls"] == []
    assert step["error_count"] == 0
    assert step["finish"] == "error"
    assert run.metrics["tool_fail"] == 0
    rendered = render_workflow_html(run.steps).lower()
    assert "system err" in rendered
    assert "tool err</span>" not in rendered


def test_existing_native_error_message_is_preserved():
    native = capture("execute", {"error": True})
    native["messages"][1]["content"][0]["state"]["error"] = {"type": "execution", "message": "Precise native failure"}
    call = load(native).steps[1]["tool_calls"][0]
    assert call["status"] == "error"
    assert call["error"] == "Precise native failure"


def test_opencode_v1_metadata_is_not_reinterpreted():
    native = {
        "info": {"id": "ses_legacy", "title": "Legacy metadata remains unchanged"},
        "messages": [{"info": {"id": "msg_legacy", "role": "assistant", "sessionID": "ses_legacy"},
                      "parts": [{"type": "tool", "tool": "execute", "callID": "call_legacy",
                                 "state": {"status": "completed", "metadata": {"error": True, "timeout": True}}}]}],
    }
    run = load(native)
    assert run.steps[0]["tool_calls"][0]["status"] == "completed"
    assert run.metrics["tool_fail"] == 0
