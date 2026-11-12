"""Passing validation requires a runner invocation and a completed outcome."""

import pytest

from agent_trajviz import Trajectory
from agent_trajviz.converge.alignment import build_comparison_report_from_steps
from agent_trajviz.converge.canonical import CanonicalAction, assign_effect_labels, canonicalize_steps
from agent_trajviz.converge.milestones import extract_milestones


def _step(index, command=None, *, output="", status="success", metadata=None,
          tool="Bash", session="root", process=None):
    args = {"command": command} if command is not None else {"session_id": process}
    step = Trajectory.from_steps([{
        "index": index, "role": "assistant", "session_id": session, "parts": [],
        "tool_calls": [{"tool_name": tool, "input": args, "status": status,
                        "metadata": metadata or {}, "output": output}],
    }]).steps[0]
    # Construct parsed-step fixtures with source positions and unknown statuses;
    # from_steps otherwise reindexes a one-step sequence and fills blank status.
    step["index"] = index
    step["tool_calls"][0]["status"] = status
    return step


def _milestone(*steps):
    actions = canonicalize_steps(list(steps))
    assign_effect_labels(actions, list(steps))
    return extract_milestones(actions, steps=list(steps))["first_passing_validation"]


@pytest.mark.parametrize("command", [
    "python - <<'PY'\nmods = ['pytest', 'numpy']\nfor mod in mods:\n    __import__(mod)\nPY",
    "python -c 'import pytest; print(pytest.__version__)'",
    "which pytest", "command -v pytest", "cat pytest.ini", "sed -n '1,20p' tests/test_check.py",
    "echo 'pytest tests/'", "pip install pytest", "git show HEAD:pytest.ini",
    "pytest --version", "python -m pytest -V", "pytest --collect-only",
    "ruff format src/", "npm install vitest", "python -m pip install pytest",
    "pytest || true", "pytest; echo done", "pytest | cat", "pytest &",
    "! pytest -q", "pytest --markers", "pytest --setup-plan", "pytest --setup-only",
    "pytest --fixtures-per-test", "pytest --cache-show=lastfailed", "make test -n",
    "make test --dry-run", "ruff check --show-settings src/a.py", "ruff check --show-files src/",
    "make test -kn",
    "ruff rule E501", "ruff config", "ruff clean", "ruff",
    "tox -l", "nox --list", "ctest -N", "jest --listTests", "vitest list", "eslint --print-config index.js",
])
def test_dependency_probes_file_reads_and_runner_metadata_are_not_validation(command):
    assert _milestone(_step(17, command)) is None


@pytest.mark.parametrize("command", [
    "pytest -q tests/", "python -m pytest -v tests/", "/usr/bin/python3 -I -m unittest discover",
    "ruff check src/", "ruff format --check src/", "git -C src diff --check",
    "go test ./...", "cargo check", "npm run lint", "make test", "ctest",
    "env MODE=ci timeout 30 python -m pytest -q", "uv run --with pytest python -m pytest -q",
    "poetry run pytest", "cd src && pytest", "sh -lc 'python -m pytest -q'",
    "echo preparing; pytest -q", "pytest -q && echo done",
    "pytest -q 2>&1", "pytest -q &>/tmp/test-output.log",
    'cmd /c "python -m pytest -q"', 'pwsh -NoProfile -Command "python -m pytest -q"',
])
def test_executed_validation_runners_with_observed_success_are_retained(command):
    assert _milestone(_step(9, command)) == 9


@pytest.mark.parametrize("changes", [
    {"status": "error"}, {"status": "running"}, {"status": ""},
    {"metadata": {"exit": 1}}, {"metadata": {"exit_code": 2}},
    {"metadata": {"_outcome_observed": False}},
    {"output": "Process exited with code 1\n1 failed"},
])
def test_failed_running_and_unobserved_validation_cannot_pass(changes):
    assert _milestone(_step(5, "pytest -q", **changes)) is None


@pytest.mark.parametrize("command", [
    "cd work && timeout 30 python -m pytest -q 2>&1 | tail -30",
    "cd work && python - <<'PY'\nprint('prepare')\nPY\npython -m pytest -q 2>&1 | tail -4",
])
def test_pytest_piped_to_tail_uses_the_runner_summary_instead_of_the_pipeline_exit(command):
    failed = _step(5, command, metadata={"exit": 0}, output="1 failed, 3 passed in 0.31s")
    passed = _step(9, command, metadata={"exit": 0}, output="24 passed in 4.90s\n")
    assert _milestone(failed) is None
    assert _milestone(failed, passed) == 9


@pytest.mark.parametrize("changes", [
    {"output": ""},
    {"output": "4 skipped in 0.02s"},
    {"output": "2 passed in 0.07s\n2 passed in 0.09s"},
    {"output": "2 passed in 0.07s\ninterrupted"},
    {"metadata": {"exit": 1}},
    {"metadata": {"truncated": True}},
    {"metadata": {"_outcome_observed": False}},
    {"command": "pytest && echo summary | tail -4"},
    {"command": "pytest | tail input.log"},
    {"command": "pytest | tail -4 || true"},
])
def test_pytest_pipeline_abstains_when_the_pass_summary_is_missing_or_ambiguous(changes):
    args = {"command": "pytest | tail -4", "output": "2 passed in 0.07s", **changes}
    assert _milestone(_step(5, **args)) is None


def _launch(index=67, *, session="root"):
    return _step(index, "python -m pytest -q tests/test_sample.py",
                 output="Process running with session ID 7011\nOutput:\n", session=session)


def _poll(index=68, *, session="root", process=7011, output="Process exited with code 0\n3 passed in 0.09s", **kwargs):
    return _step(index, tool="write_stdin", process=process, session=session, output=output, **kwargs)


def test_async_validation_milestone_is_observed_completion_after_the_import_probe():
    probe = _step(17, "python - <<'PY'\nimport pytest\nPY")
    assert _milestone(probe, _launch()) is None
    assert _milestone(probe, _launch(), _poll()) == 68


def test_poll_success_without_process_completion_remains_unknown():
    assert _milestone(_launch(), _poll(output="")) is None
    assert _milestone(_launch(), _poll(output="Process running with session ID 7011")) is None
    assert _milestone(_launch(), _poll(metadata={"exit": 0}, output="")) == 68


def test_failed_poll_never_turns_a_later_endpoint_success_into_a_pass():
    failed = _poll(output="Process exited with code 1\n1 failed", status="error")
    assert _milestone(_launch(), failed, _poll(69)) is None


def test_async_completion_is_scoped_to_its_process_and_agent_session():
    assert _milestone(_launch(), _poll(process=1234)) is None
    assert _milestone(_launch(session="child"), _poll(session="root")) is None
    assert _milestone(_step(67, "which pytest", output="Process running with session ID 7011"), _poll()) is None


def test_comparison_pipeline_supplies_outcome_evidence_for_both_runs():
    steps = [_step(17, "python -c 'import pytest'"), _launch(), _poll()]
    report = build_comparison_report_from_steps({}, {}, steps, steps)
    assert report["ref_milestones"]["first_passing_validation"] == 68
    assert report["cmp_milestones"]["first_passing_validation"] == 68
    assert report["milestones"]["first_passing_validation_delta"] == 0


def test_standalone_actions_only_callers_keep_the_existing_contract():
    actions = [CanonicalAction(step_index=5, action_type="COMMAND", target="pytest",
                               effect_label="justified", effect_detail={"reason": "validation_command"})]
    assert extract_milestones(actions)["first_passing_validation"] == 5
    assert extract_milestones(actions, steps=[])["first_passing_validation"] is None
