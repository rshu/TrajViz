"""Native projection invariants and optional real-backend interoperability."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import agent_trajviz as tv
from agent_trajviz.awe.native import project_native


@pytest.fixture
def ontology(monkeypatch):
    package = ModuleType("agent_trajviz.awe._engine")
    trajectory = ModuleType("agent_trajviz.awe._engine.trajectory")
    for name in ("READ", "EDIT", "WRITE", "SEARCH", "DELEGATE", "PLAN", "OTHER"):
        setattr(trajectory, name, name.lower())
    trajectory.NormalizedTrajectory = lambda agent, instance: SimpleNamespace(agent=agent, instance_id=instance, steps=[], source={})
    trajectory.NormalizedStep = lambda **kw: SimpleNamespace(**kw)
    trajectory.plan_items = lambda value: value or []
    trajectory.repo_relative = lambda path, *args: path
    trajectory.claude_code_exit_code = lambda text: None
    adapters = ModuleType("agent_trajviz.awe._engine.adapters")
    adapters._recorded_verification_state = lambda *records: {}
    adapters._edit_content_args = lambda old, new: {"old_content": old, "new_content": new}
    package.trajectory, package.adapters = trajectory, adapters
    monkeypatch.setitem(sys.modules, "agent_trajviz.awe._engine", package)
    monkeypatch.setitem(sys.modules, "agent_trajviz.awe._engine.trajectory", trajectory)
    monkeypatch.setitem(sys.modules, "agent_trajviz.awe._engine.adapters", adapters)


def _row(*calls):
    return {"index": 7, "raw_index": 3, "role": "assistant", "agent": "reviewer", "is_sub_agent": True,
            "parts": [{"type": "text", "text": "Observed reasoning summary."}], "tool_calls": list(calls)}


def test_observed_results_keep_threads_and_never_infer_an_exit_code(ontology):
    call = {"type": "tool_call", "tool_name": "Read", "tool_id": "read-1", "input": {"file_path": "src/a.py"},
            "output": "file content", "status": "success"}
    run = tv.Trajectory(format="cursor", steps=[_row(call)])
    result = project_native(run, "cursor", "case-1")
    assert [step.type for step in result.steps] == ["message", "tool_call", "tool_result"]
    assert result.steps[1].files == ["src/a.py"]
    assert all(step.subagent_id == "reviewer" for step in result.steps)
    assert result.steps[2].call_id == "read-1" and "exit_code" not in result.steps[2].args
    assert result.steps[1].source_ref["workflow_step"] == 7
    assert result.steps[1].source_ref["kind"] == "trajviz_workflow_projection"


def test_pending_call_has_no_invented_result_and_parts_do_not_duplicate_calls(ontology):
    call = {"type": "tool_call", "tool_name": "Read", "tool_id": "pending-1", "input": {"file_path": "a.py"},
            "output": "", "status": "pending"}
    step = _row(call)
    step["parts"].append(call)
    result = project_native(tv.Trajectory(format="claude_code", steps=[step]), "claude_code", "case-1")
    assert [row.type for row in result.steps].count("tool_call") == 1
    assert not any(row.type == "tool_result" for row in result.steps)


def test_empty_completed_result_is_retained_without_claiming_tests_passed(ontology):
    call = {"type": "tool_call", "tool_name": "Read", "tool_id": "empty-1", "input": {}, "output": "", "status": "completed"}
    result = project_native(tv.Trajectory(format="pi", steps=[_row(call)]), "pi", "case-1")
    output = result.steps[-1]
    assert output.type == "tool_result" and output.output_excerpt == "" and output.args == {}


@pytest.mark.parametrize("part", [
    {"type": "reasoning", "text": "[encrypted reasoning]", "observability": "hidden"},
    {"type": "reasoning", "text": "", "redacted": True},
    {"type": "thinking", "text": "", "encrypted": True},
])
def test_hidden_or_empty_redacted_reasoning_stays_hidden(ontology, part):
    row = _row()
    row["parts"] = [part]
    result = project_native(tv.Trajectory(format="codex", steps=[row]), "codex", "case-1")
    assert len(result.steps) == 1
    assert result.steps[0].type == "reasoning" and result.steps[0].observability == "hidden"
    assert result.steps[0].text == part["text"]
    assert result.steps[0].subagent_id == "reviewer"


@pytest.mark.parametrize("exit_code", [0, 1, None, True])
def test_codex_json_results_retain_only_recorded_integer_exit_codes(ontology, exit_code):
    from agent_trajviz.awe._engine import trajectory
    trajectory.codex_result_header = lambda text: (None, None, text)
    output = json.dumps({"metadata": {"exit_code": exit_code}, "output": "2 passed"})
    call = {"type": "tool_call", "tool_name": "exec_command", "tool_id": "json-result", "input": {},
            "output": output, "metadata": {}, "status": "success"}
    result = project_native(tv.Trajectory(format="codex", steps=[_row(call)]), "codex", "case-1")
    observed = result.steps[-1]
    assert observed.type == "tool_result" and observed.output_excerpt == "2 passed"
    if type(exit_code) is int:
        assert observed.args["exit_code"] == exit_code
    else:
        assert "exit_code" not in observed.args


def test_command_reads_resolve_recorded_workdir_under_the_workspace(ontology):
    from agent_trajviz.awe._engine import adapters, trajectory
    trajectory.classify_command = lambda command: trajectory.READ
    trajectory.command_read_targets = lambda command: ["src/a.py"]
    trajectory.is_test_command = lambda command: False
    adapters._Builder = lambda agent, instance: SimpleNamespace(agent=agent, instance_id=instance)
    adapters._command_cwd = lambda command, builder, initial="": command[3:] if command.startswith("cd ") else initial
    adapters._resolve_read = lambda builder, cwd, path: f"{cwd}/{path}" if cwd else path
    call = {"type": "tool_call", "tool_name": "exec_command", "tool_id": "workdir-read",
            "input": {"cmd": "cat src/a.py", "workdir": "/workspace/demo/pkg"}, "output": "content", "status": "success"}
    row = _row(call)
    row["cwd"] = "/workspace/demo"
    result = project_native(tv.Trajectory(format="cursor", steps=[row]), "cursor", "case-1")
    projected = next(step for step in result.steps if step.type == "tool_call")
    assert projected.files == ["pkg/src/a.py"]
    assert projected.native_input["workdir"] == "/workspace/demo/pkg"


def test_real_backend_native_claude_and_children_are_content_bound(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = r'''
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from agent_trajviz.awe._engine import config
from agent_trajviz.awe.native import install_native_adapters
reference=Path(sys.argv[2]); parent=reference/'data/trajectory/claude_code'
parent.mkdir(parents=True)
fixtures=Path(sys.argv[1])/'tests/fixtures/claude_current'
shutil.copyfile(fixtures/'session-current.jsonl',parent/'session-current.jsonl')
shutil.copytree(fixtures/'session-current',parent/'session-current')
config.TRAJECTORY_DIR=reference/'data/trajectory'
install_native_adapters()
from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.provenance import trajectory_fingerprint
before=trajectory_fingerprint('claude_code','session-current')
run=load_normalized('claude_code','session-current')
assert run.steps and any(step.subagent_id for step in run.steps)
assert any(step.type=='tool_call' and step.canonical_action=='delegate' for step in run.steps)
assert all(step.source_ref.get('kind')=='trajviz_workflow_projection' for step in run.steps)
child=next((parent/'session-current').rglob('*.jsonl'))
child.write_text(child.read_text()+'\n')
assert trajectory_fingerprint('claude_code','session-current')!=before
assert run.source['adapter']=='trajviz-native-v1'
print(json.dumps({'steps':len(run.steps),'children':len(run.source['files'])-1}))
'''
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "AWE_PROJECT_DATA": str(tmp_path / "project")}
    result = subprocess.run([sys.executable, "-I", "-c", script, str(project), str(tmp_path / "reference")],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["children"] > 0
