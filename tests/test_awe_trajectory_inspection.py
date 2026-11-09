"""AWE inspection consumes recorded sessions without reference or model inputs."""
from __future__ import annotations

import copy
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_trajviz import awe
from agent_trajviz.awe import native
from agent_trajviz.awe._engine import eventledger, gold, outcomes, symptoms


def call(command="python -m pytest tests/test_a.py", *, code=0, output="1 passed in 0.02s", status="completed", identity="call-1", name="Bash", **metadata):
    return {"type": "tool_call", "tool_name": name, "tool_id": identity,
            "input": {"command": command}, "output": output, "status": status,
            "metadata": {**({"exit": code} if code is not None else {}), **metadata}}


def step(index, calls=(), **extra):
    return {"index": index, "role": "assistant", "parts": [], "tool_calls": list(calls),
            "duration": None, "tokens": {}, **extra}


def run(*steps, fmt="codex"):
    return SimpleNamespace(steps=list(steps), source="/deleted/session.jsonl", sha256="loaded-sha", format=fmt)


def test_uses_bundled_ledger_and_symptoms_without_reference_execution_or_source_reads(monkeypatch):
    sample = run(step(17, [call(code=1, output="1 failed in 0.02s")]), step(29, [call()]))
    before = copy.deepcopy(sample.steps)
    calls = {"ledger": 0, "symptoms": 0}
    original_ledger, original_symptoms = eventledger.event_ledger, symptoms.analyze

    def ledger(value):
        calls["ledger"] += 1
        return original_ledger(value)

    def analyze(value, patch_text=""):
        calls["symptoms"] += 1
        return original_symptoms(value, patch_text)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must consume the loaded snapshot only")

    monkeypatch.setattr(eventledger, "event_ledger", ledger)
    monkeypatch.setattr(symptoms, "analyze", analyze)
    monkeypatch.setattr(native, "_fingerprints", forbidden)
    monkeypatch.setattr(native, "adapter_code_hash", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(gold, "build_gold_reference", forbidden)
    monkeypatch.setattr(outcomes, "load_outcomes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = awe.inspect_trajectory(sample)
    assert result["mode"] == "trajectory_only"
    assert calls == {"ledger": 1, "symptoms": 1}
    assert sample.steps == before
    assert result["source"]["identity_basis"] == "loaded_snapshot"
    assert result["outcome"]["status"] == "unknown"
    assert result["workflow_indices"] == [17, 29]
    assert result["summary"]["failed_calls"] == 1
    assert result["summary"]["later_successes"] == 1
    assert result["recovery"][0]["step_indices"] == [17, 29]
    json.dumps(result, allow_nan=False)


def test_shared_native_failure_metadata_and_timeout_are_preserved():
    result = awe.inspect_trajectory(run(step(1, [call(code=1, output="", status="completed")]),
                                       step(2, [call(code=None, output="timeout", status="timeout")]),
                                       step(3, [call(code=None, output="aborted", status="cancelled")])))
    assert result["summary"]["failed_calls"] == 3
    assert result["failures"][0]["exit_code"] == 1
    assert [row["workflow_index"] for row in result["failures"]] == [1, 2, 3]


def test_same_native_ids_across_threads_never_establish_recovery():
    result = awe.inspect_trajectory(run(step(2, [call(code=1, output="1 failed in 0.01s")]),
                                       step(3, [call()], is_sub_agent=True, agent="child-7")))
    assert result["summary"]["threads"] == 2
    assert result["summary"]["later_successes"] == 0
    assert result["recovery"][0]["thread_id"] == "main"
    assert result["tests"][1]["thread_id"] == "child-7"


def test_pending_call_reusing_id_cannot_consume_later_result():
    result = awe.inspect_trajectory(run(step(0, [call(code=None, output=None, status="running")]),
                                       step(1, [call()])))
    assert result["summary"]["tool_calls"] == 2
    assert result["summary"]["tool_results"] == 1
    assert result["summary"]["missing_results"] == 1
    assert result["tests"][0]["summary_status"] == "unknown"
    assert result["tests"][1]["summary_status"] == "reported"
    assert result["summary"]["later_successes"] == 0


def test_idless_calls_are_counted_once_when_parts_and_canonical_calls_coexist():
    a, b = call("cat a.py", identity=""), call("cat b.py", identity="")
    result = awe.inspect_trajectory(run(step(0, [a, b], parts=[copy.deepcopy(a)])))
    assert result["summary"]["tool_calls"] == 2
    assert result["summary"]["tool_results"] == 2
    assert {row["path"] for row in result["files"]} == {"a.py", "b.py"}


def test_explicit_read_and_write_actions_survive_original_command_metadata():
    read = call("sed -n '1,8p' src/app.py", name="Read")
    read["input"]["file_path"] = "src/app.py"
    write = call("*** Begin Patch\n*** Update File: src/app.py\n*** End Patch", name="Write", identity="write")
    write["input"]["file_path"] = "src/app.py"
    result = awe.inspect_trajectory(run(step(4, [read]), step(8, [write])))
    assert result["files"] == [{"path": "src/app.py", "actions": ["read", "write"], "step_indices": [4, 8],
                                "threads": ["main"], "first_step": 4, "last_step": 8}]
    assert result["summary"]["edited_files"] == 1


def test_a_direct_test_command_uses_reported_runner_summary_without_claiming_task_success():
    result = awe.inspect_trajectory(run(step(2, [call()])))
    test = result["tests"][0]
    assert test["summary_status"] == "reported"
    assert test["counts"]["executed"] == 1 and test["counts"]["passed"] == 1
    assert test["process_status"] == "zero_exit"
    assert result["summary"]["test_summaries_reported"] == 1
    assert result["outcome"]["status"] == "unknown"


def test_exit_zero_without_runner_summary_never_becomes_passed_tests():
    result = awe.inspect_trajectory(run(step(2, [call(output="Finished command.")])))
    test = result["tests"][0]
    assert test["summary_status"] == "unknown" and test["counts"] is None
    assert test["output_summary"] is None
    assert test["process_status"] == "zero_exit"
    assert result["summary"]["test_summaries_reported"] == 0
    assert result["outcome"]["status"] == "unknown"


def test_compound_or_quoted_summary_text_is_inspectable_without_execution_credit():
    command = "python -m pytest tests/test_a.py | tail -20; echo '10 passed in 0.01s'"
    result = awe.inspect_trajectory(run(step(5, [call(command, output="10 passed in 0.01s")])))
    test = result["tests"][0]
    assert test["summary_status"] == "unknown" and test["counts"] is None
    assert test["output_summary"]["counts"]["passed"] == 10
    assert test["output_summary"]["basis"] == "unattributed_output_text"
    assert test["status"] == "output_summary_only"
    assert result["summary"]["test_summaries_reported"] == 0
    assert result["summary"]["test_summary_texts"] == 1
    assert not any(f["kind"] == "reported_test_failures" for f in result["findings"])
    assert result["outcome"]["status"] == "unknown"


def test_zero_execution_and_missing_result_are_unscored_observations():
    result = awe.inspect_trajectory(run(step(2, [call(output="1 skipped in 0.02s")]),
                                       step(3, [call(output=None, code=None, status="pending")])))
    assert any(f["kind"] == "all_tests_skipped" for f in result["findings"])
    assert any(f["kind"] == "missing_results" for f in result["findings"])
    assert all(s.get("scoring") == "not_scored" for s in result["symptoms"]["symptoms"])
    assert result["outcome"]["status"] == "unknown"


def test_recorded_timing_is_optional_and_does_not_invent_cache_or_latency_causes():
    result = awe.inspect_trajectory(run(step(0, duration=18.5, timing_source="model_request"),
                                       step(1, duration=None), step(2, duration=float("nan"))))
    assert len(result["timing"]) == 1
    assert result["timing"][0]["duration_seconds"] == 18.5
    assert result["coverage"]["step_timing"] == {"observed": 1, "total": 3}
    assert "weak cache" not in json.dumps(result).lower()
    assert result["outcome"]["status"] == "unknown"


@pytest.mark.parametrize("index", [[], {}, True, None])
def test_invalid_custom_indices_never_gain_workflow_links(index):
    result = awe.inspect_trajectory(run(step(index, [call(code=1, output="failed")]), step(7)))
    assert result["workflow_indices"] == [7]
    assert result["failures"][0]["workflow_index"] is None
    assert result["findings"][0]["step_indices"] == []


def test_duplicate_indices_never_gain_ambiguous_workflow_links():
    result = awe.inspect_trajectory(run(step(3, [call(code=1)]), step(3)))
    assert result["workflow_indices"] == []
    assert result["failures"][0]["workflow_index"] is None


def test_empty_recording_has_a_clear_load_prompt():
    result = awe.inspect_trajectory(None)
    assert result["available"] is False
    assert result["reason"].startswith("Load a trajectory")
    assert not result["findings"]


def test_hidden_tool_fields_do_not_turn_placeholders_into_failure_or_file_evidence():
    tool = call("cat hidden.py", code=1, output="1 failed in 0.01s")
    tool["metadata"]["_otel"] = {"fields": {"tool_inputs": "unavailable", "tool_outputs": "unavailable", "tool_outcomes": "unavailable"}}
    result = awe.inspect_trajectory(run(step(0, [tool]), fmt="otlp"))
    assert result["summary"]["failed_calls"] == 0
    assert result["summary"]["missing_results"] == 1
    assert result["files"] == []
    assert result["coverage"]["tool_inputs"]["observed"] == 0
    assert result["coverage"]["tool_outcomes"]["observed"] == 0


def test_source_file_can_disappear_after_loading(tmp_path):
    import agent_trajviz as tv

    source = tmp_path / "session.json"
    fixture = Path(__file__).parent / "fixtures" / "codearts_minimal.json"
    source.write_bytes(fixture.read_bytes())
    loaded = tv.load(source)
    source.unlink()
    result = awe.inspect_trajectory(loaded)
    assert result["available"] and result["summary"]["steps"] == len(loaded.steps)
    assert result["source"]["sha256"] == loaded.sha256


def test_memory_projection_preserves_default_reference_provenance_path(monkeypatch):
    seen = []
    monkeypatch.setattr(native, "adapter_code_hash", lambda: seen.append("code") or "code-sha")
    projected = native.project_native(run(step(0)), "codex", "task")
    assert projected.source["adapter_sha256"] == "code-sha" and seen == ["code"]
    seen.clear()
    projected = native.project_native(run(step(0)), "codex", "task", verify_source=False)
    assert seen == [] and "adapter_sha256" not in projected.source
    assert projected.source["identity_basis"] == "loaded_snapshot"


def test_error_looking_prose_in_a_successful_output_never_becomes_a_failed_call():
    tool = call("cat debugging-notes.txt", code=None, output="Traceback (most recent call last):\nModuleNotFoundError: example", status="success")
    result = awe.inspect_trajectory(run(step(1, [tool])))
    assert result["summary"]["failed_calls"] == 0
    assert result["summary"]["failure_episodes"] == 0
    assert result["failures"] == []
    assert not any(f["kind"] == "tool_failures" for f in result["findings"])


def test_running_process_response_does_not_establish_a_successful_recovery():
    failed = call("python -m pytest", code=1, output="1 failed in 0.02s")
    running = call("python -m pytest", code=None,
                   output="Process running with session ID 12345\nOriginal token count: 0\nOutput:\n", status="success")
    result = awe.inspect_trajectory(run(step(1, [failed]), step(2, [running])))
    assert result["summary"]["failed_calls"] == 1
    assert result["summary"]["later_successes"] == 0
    assert result["tests"][1]["process_status"] == "success"
    assert result["tests"][1]["exit_code"] is None


@pytest.mark.parametrize("fmt", ["codex", "opencode", "claude_code"])
@pytest.mark.parametrize("output", ["Exit code 1\nExample documentation text",
                                   "Process exited with code 1\nExample documentation text",
                                   '{"metadata":{"exit_code":1},"output":"Example documentation text"}'])
def test_successful_file_read_never_interprets_document_contents_as_process_outcome(fmt, output):
    tool = call("", code=None, name="Read", output=output)
    tool["input"] = {"file_path": "notes.txt"}
    result = awe.inspect_trajectory(run(step(1, [tool]), fmt=fmt))
    assert result["summary"]["failed_calls"] == 0
    assert result["summary"]["failure_episodes"] == 0
    projected = native.project_native(run(step(1, [tool]), fmt=fmt), fmt, "task")
    recorded_result = projected.steps[-1]
    assert "exit_code" not in recorded_result.args
    assert recorded_result.output_excerpt == output


@pytest.mark.parametrize("output", ["Exit code 1", "Process exited with code 1",
                                   '{"metadata":{"exit_code":1},"output":"quoted example"}'])
def test_successful_shell_stdout_cannot_create_a_new_failure(output):
    tool = call("printf 'quoted example'", code=None, output=output)
    result = awe.inspect_trajectory(run(step(1, [tool])))
    assert result["summary"]["failed_calls"] == 0
    assert result["summary"]["failure_episodes"] == 0
    assert result["failures"] == []


@pytest.mark.parametrize("code", [0, 1])
def test_structured_exit_counts_as_observed_even_when_status_is_missing(code):
    result = awe.inspect_trajectory(run(step(1, [call(code=code, status="")]), fmt="opencode"))
    assert result["coverage"]["tool_outcomes"] == {"observed": 1, "total": 1}
    assert result["summary"]["failed_calls"] == int(code != 0)
    assert result["tests"][0]["process_status"] == ("zero_exit" if code == 0 else "nonzero_exit")


@pytest.mark.parametrize("tagged", [True, False])
def test_distinct_sessions_with_the_same_agent_name_cannot_cross_recover(tagged):
    result = awe.inspect_trajectory(run(step(4, [call(code=1)], agent="worker", session_id="session-A", is_sub_agent=tagged),
                                       step(8, [call()], agent="worker", session_id="session-B", is_sub_agent=tagged), fmt="opencode"))
    assert result["summary"]["threads"] == 2
    assert result["summary"]["later_successes"] == 0
    assert result["recovery"][0]["thread_id"] == "session-A"


def test_same_root_session_mode_change_preserves_its_recovery_evidence():
    result = awe.inspect_trajectory(run(step(4, [call(code=1)], agent="plan", session_id="session-A", is_sub_agent=False),
                                       step(8, [call()], agent="build", session_id="session-A", is_sub_agent=False), fmt="opencode"))
    assert result["summary"]["threads"] == 1
    assert result["summary"]["later_successes"] == 1


def test_claude_children_sharing_root_session_stay_separate_from_each_other():
    result = awe.inspect_trajectory(run(step(4, [call(code=1)], agent="agent-child-a", session_id="root-session", is_sub_agent=True),
                                       step(8, [call()], agent="agent-child-b", session_id="root-session", is_sub_agent=True), fmt="claude_code"))
    assert result["summary"]["threads"] == 2
    assert result["summary"]["later_successes"] == 0
