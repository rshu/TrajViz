"""Normal OpenCode cancellation is history, not a global import warning."""

from copy import deepcopy
import json

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.presenters.overview import load_warnings_html
from agent_trajviz.insight.session import session_from_run


def _capture(error=None, *, resumed=False, idle_error=None, failed_tool=False):
    # Native OpenCode 2.2.5 error and lifecycle shapes from a real resumed run.
    error = {"type": "aborted", "message": "Step interrupted"} if error is None else error
    assistant = {
        "id": "msg_interrupted", "type": "assistant", "sessionID": "ses_root", "agent": "build",
        "time": {"created": 1100, "completed": 2000}, "finish": "error", "error": error, "content": [],
    }
    if failed_tool:
        assistant["content"] = [{
            "type": "tool", "id": "call_cancelled", "name": "shell",
            "time": {"created": 1200, "ran": 1300, "completed": 1900},
            "state": {"status": "error", "input": {"command": "pytest -q"},
                      "error": {"type": "aborted", "message": "Tool execution interrupted"}},
        }]
    idle = {"id": "msg_idle", "type": "idle", "outcome": "interrupted", "time": {"created": 2100}}
    if idle_error is not None:
        idle["error"] = idle_error
    messages = [
        {"id": "msg_user", "type": "user", "time": {"created": 1000}, "text": "Inspect the parser."},
        assistant, idle,
    ]
    if resumed:
        messages.extend([
            {"id": "msg_resume", "type": "user", "time": {"created": 2200}, "text": "Continue."},
            {"id": "msg_done", "type": "assistant", "agent": "build", "finish": "stop",
             "time": {"created": 2300, "completed": 3000}, "content": [{"type": "text", "text": "Done."}]},
            {"id": "msg_succeeded", "type": "idle", "outcome": "succeeded", "time": {"created": 3100}},
        ])
    return {"info": {"id": "ses_root", "title": "Resumed native run", "version": "2.2.5",
                     "time": {"created": 1000, "updated": 3100}}, "messages": messages}


def _load(tmp_path, native):
    path = tmp_path / "session.json"
    path.write_text(json.dumps(native), encoding="utf-8")
    original_bytes = path.read_bytes()
    run = tv.load(path)
    assert path.read_bytes() == original_bytes
    assert run.raw["_opencode_v2_messages"] == native["messages"]
    return run


@pytest.mark.parametrize("explicit_idle_error", [False, True])
def test_resumed_normal_interruption_does_not_show_model_error_warning(tmp_path, explicit_idle_error):
    error = {"type": "aborted", "message": "Step interrupted"}
    native = _capture(resumed=True, idle_error=error if explicit_idle_error else None)
    original = deepcopy(native)
    run = _load(tmp_path, native)
    assert run.warnings == ()
    assert load_warnings_html(session_from_run(run, "")) == ""
    assert run.raw["_opencode_v2_history_errors"] == (
        ["Step interrupted"] if explicit_idle_error else ["Step interrupted", "Session turn interrupted."]
    )
    assert run.raw["output"]["error"] is None
    assert run.metrics["patch_error"] is None
    assert run.steps[1]["finish"] == "error"
    assert run.raw["messages"][1]["info"]["error"] == error
    assert native == original


def test_terminal_interruption_retains_its_outcome_without_inventing_success(tmp_path):
    run = _load(tmp_path, _capture())
    assert run.warnings == ()
    assert run.steps[-1]["finish"] == "error"
    assert run.raw["output"]["error"] == "Session turn interrupted."
    assert run.metrics["patch_error"] == "Session turn interrupted."
    assert run.metrics["tool_call_count"] == 0
    assert run.metrics["tool_fail"] == 0


@pytest.mark.parametrize("error", [
    {"type": "provider.auth", "message": "Step interrupted", "status": 401},
    {"type": "provider.error", "message": "Step interrupted"},
    {"type": "aborted", "message": "Step interrupted", "status": 503},
    {"type": "aborted", "message": "Step interrupted", "response": {"body": "Upstream unavailable"}},
    {"type": "aborted", "message": "The operation was aborted."},
    {"type": "aborted", "message": "Step interrupted: upstream transport failed"},
    {"type": "aborted", "message": "step interrupted"},
    {"message": "Step interrupted"},
    "Step interrupted",
])
def test_error_type_or_provider_evidence_prevents_broad_interruption_suppression(tmp_path, error):
    run = _load(tmp_path, _capture(error, resumed=True, idle_error=error))
    message = error["message"] if isinstance(error, dict) else error
    assert run.warnings == (f"OpenCode model/session error: {message}",)
    assert message in load_warnings_html(session_from_run(run, ""))
    assert run.raw["messages"][1]["info"]["error"] == error
    assert run.raw["output"]["error"] is None  # the later resumed turn succeeded


def test_genuine_idle_failure_is_not_hidden_by_matching_interruption_text(tmp_path):
    native = _capture(idle_error={"type": "aborted", "message": "Step interrupted"})
    native["messages"][-1]["outcome"] = "failed"
    run = _load(tmp_path, native)
    assert run.warnings == ("OpenCode model/session error: Step interrupted",)
    assert run.raw["output"]["error"] == "Step interrupted"


def test_real_provider_failure_after_normal_interruption_remains_a_warning_and_terminal_error(tmp_path):
    native = _capture(resumed=True)
    error = {"type": "provider.auth", "message": "Authentication failed", "status": 401}
    native["messages"][-2].update(finish="error", error=error)
    native["messages"][-1].update(outcome="failed", error=error)
    run = _load(tmp_path, native)
    assert run.warnings == ("OpenCode model/session error: Authentication failed",)
    assert run.metrics["patch_error"] == "Authentication failed"
    assert run.raw["_opencode_v2_history_errors"] == [
        "Step interrupted", "Session turn interrupted.", "Authentication failed",
    ]
    assert run.raw["messages"][-1]["info"]["error"] == error


def test_normal_session_interruption_does_not_reclassify_failed_tools(tmp_path):
    native = _capture(resumed=True, failed_tool=True)
    run = _load(tmp_path, native)
    assert run.warnings == ()
    assert run.metrics["tool_fail"] == 1
    assert run.metrics["tool_success"] == 0
    assert run.steps[1]["error_count"] == 1
    call = run.steps[1]["tool_calls"][0]
    assert call["status"] == "error"
    assert call["error"] == "Tool execution interrupted"
    assert call["metadata"]["opencode_v2_tool"] == native["messages"][1]["content"][0]
