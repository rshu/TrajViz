"""Native OpenCode ``shell`` calls retain the shared shell analysis semantics."""

from copy import deepcopy

import pytest

from agent_trajviz import Trajectory
from agent_trajviz.converge.canonical import assign_effect_labels, canonicalize_steps
from agent_trajviz.converge.milestones import extract_milestones
from agent_trajviz.insight.patterns import _is_search_call
from agent_trajviz.insight.shell_cmd import tool_chart_name
from agent_trajviz.tool_vocab import BASH_TOOL_NAMES


def _step(command, *, exit_code=0, observed=True):
    metadata = {"exit": exit_code} if exit_code is not None else {}
    if not observed:
        metadata["_outcome_observed"] = False
    return Trajectory.from_steps([{
        "role": "assistant", "tokens": {"total": 100},
        "tool_calls": [{"tool_name": "shell", "tool_id": "native-shell",
                        "input": {"command": command}, "status": "success",
                        "metadata": metadata, "output": "completed"}],
    }]).steps[0]


@pytest.mark.parametrize(("command", "action_type", "target", "chart"), [
    ("cd /repo && python -m pytest -q tests/", "COMMAND", "pytest", "pytest"),
    ("cd /repo && python - <<'PY'\nprint('pytest')\nPY", "COMMAND", "python", "python"),
    ("cd /repo", "REASON", "", "cd"),
    ("cd /repo && ls", "REASON", "", "ls"),
    ("MODE=ci env OTHER=1 timeout 30 python -m pytest -q", "COMMAND", "pytest", "pytest"),
    ("sh -lc 'cd /repo && python -m pytest -q'", "COMMAND", "pytest", "pytest"),
    ("cd /repo && git -C libs/core diff --check", "COMMAND", "git", "git"),
    ("cd /repo && sed -n '1,30p' /repo/src/app.py", "FILE_READ", "/repo/src/app.py", "sed"),
    ("cd /repo && grep -n 'def run' /repo/src/app.py", "SEARCH", "def run@/repo/src/app.py", "grep"),
    ("env -u grep /usr/bin/grep needle /repo/src/app.py", "SEARCH", "needle@/repo/src/app.py", "grep"),
    ("cd /repo && sed -n '1,30p' src/app.py", "COMMAND", "sed", "sed"),
])
def test_native_shell_classification_preserves_work_and_original_record(command, action_type, target, chart):
    step = _step(command)
    original = deepcopy(step)
    actions = canonicalize_steps([step])
    assert len(actions) == 1
    assert (actions[0].action_type, actions[0].target) == (action_type, target)
    assert actions[0].tool == "shell"
    assert actions[0].tool_call_id == "native-shell"
    assert actions[0].args == {"command": command}
    assert tool_chart_name(step["tool_calls"][0]) == chart
    assert step == original


@pytest.mark.parametrize(("exit_code", "observed", "effect", "milestone"), [
    (0, True, "justified", 0),
    (1, True, "failed", None),
    (None, False, "unknown", None),
])
def test_shell_validation_keeps_exit_and_missing_evidence_semantics(exit_code, observed, effect, milestone):
    step = _step("cd /repo && python -m pytest -q tests/", exit_code=exit_code, observed=observed)
    actions = canonicalize_steps([step])
    assign_effect_labels(actions, [step])
    assert actions[0].action_type == "COMMAND"
    assert actions[0].effect_label == effect
    assert extract_milestones(actions, steps=[step])["first_passing_validation"] == milestone


def test_native_shell_uses_existing_search_detection_without_renaming():
    assert "shell" in BASH_TOOL_NAMES
    search = _step("cd /repo && grep needle src/app.py")["tool_calls"][0]
    validation = _step("cd /repo && python -m pytest -q")["tool_calls"][0]
    heredoc = _step("cd /repo && python - <<'PY'\nprint('grep needle')\nPY")["tool_calls"][0]
    assert _is_search_call(search)
    assert not _is_search_call(validation)
    assert not _is_search_call(heredoc)
    assert search["tool_name"] == "shell"


def test_unresolved_shell_script_arguments_remain_a_command():
    actions = canonicalize_steps([_step("sh -lc 'cd /repo && grep needle src/app.py'")])
    assert (actions[0].action_type, actions[0].target) == ("COMMAND", "grep")
