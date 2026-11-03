"""Comparison file identity uses recorded session directories without changing evidence."""

import json
from copy import deepcopy

import pytest

from agent_trajviz import compare, load
from dataclasses import asdict

from agent_trajviz.converge.canonical import _comparison_actions, canonicalize_steps, semantic_equivalent
from agent_trajviz.insight.run_group import build_behavioral_comparison, build_run_group_scorecard_html
from agent_trajviz.views import compare_runs


def _action(path):
    return canonicalize_steps([{
        "index": 0, "tokens": {"total": 10}, "parts": [],
        "tool_calls": [{"tool_name": "Edit", "input": {"file_path": path}, "status": "success"}],
    }])[0]


@pytest.mark.parametrize(("directory", "path", "expected"), [
    ("/runs/one/repo", "/runs/one/repo/src/app.py", "src/app.py"),
    ("/runs/one/repo/", "/runs/one/repo/src/../app.py", "app.py"),
    ("/runs/one/repo", "/runs/one/repository/app.py", "/runs/one/repository/app.py"),
    ("/runs/one/repo", "/tmp/debug.py", "/tmp/debug.py"),
    ("/runs/one/repo", "src/app.py", "src/app.py"),
    ("/runs/one/repo/pkg", "/runs/one/repo/pkg/app.py", "app.py"),
    ("/runs/one/repo/pkg", "/runs/one/repo/other.py", "/runs/one/repo/other.py"),
    ("/runs/one/repo", "/runs/one/repo/pkg/app.py", "pkg/app.py"),
    (r"C:\runs\repo", r"C:\runs\repo\src\app.py", "src/app.py"),
    (r"C:\runs\repo", r"D:\runs\repo\src\app.py", "/runs/repo/src/app.py"),
    (r"C:\runs\repo", r"C:\runs\repository\app.py", "/runs/repository/app.py"),
])
def test_file_identity_stays_within_the_recorded_directory(directory, path, expected):
    action = _action(path)
    original = deepcopy(action)
    action.effect_label = "survived"
    action.effect_detail = {"reason": "source_evidence"}
    compared = _comparison_actions([action], {"metadata": {"directory": directory}})[0]
    assert compared.target == expected
    assert compared.args == original.args
    assert compared.tool == original.tool
    assert compared.effect_label == "survived"
    assert compared.effect_detail == {"reason": "source_evidence"}
    assert action.target == original.target
    assert set(asdict(compared)) == set(asdict(action))


@pytest.mark.parametrize("raw", [{}, {"metadata": {}}, {"metadata": {"directory": ""}},
                                 {"metadata": {"directory": "relative/repo"}}])
def test_missing_or_relative_session_directory_keeps_existing_targets(raw):
    actions = [_action("/runs/one/repo/src/app.py")]
    assert _comparison_actions(actions, raw) is actions


@pytest.mark.parametrize(("other_path", "other_directory"), [
    ("/runs/outside/app.py", "/runs/two/repo"),
    ("/unknown/app.py", None),
    ("/runs/two/repo/pkg/app.py", "/runs/two/repo"),
])
def test_resolved_workspace_file_does_not_suffix_match_outside_unknown_or_other_relative_paths(other_path, other_directory):
    first = _comparison_actions([_action("/runs/one/repo/app.py")], {"metadata": {"directory": "/runs/one/repo"}})
    raw = {"metadata": {"directory": other_directory}} if other_directory else {}
    other = _comparison_actions([_action(other_path)], raw)
    assert not semantic_equivalent(first[0], other[0])
    behavior = build_behavioral_comparison([
        {"run_id": "one", "label": "one", "actions": first},
        {"run_id": "two", "label": "two", "actions": other},
    ])
    assert behavior["file_matrix_total"] == 2
    assert all(row["n_runs"] == 1 for row in behavior["file_matrix"])


def test_fuzzy_command_file_identity_uses_its_own_recorded_workspace():
    command = canonicalize_steps([{
        "index": 0, "tokens": {"total": 10}, "parts": [],
        "tool_calls": [{"tool_name": "shell", "input": {
            "command": "printf ignored | cat /runs/one/repo/src/app.py"}, "status": "success"}],
    }])
    command = _comparison_actions(command, {"metadata": {"directory": "/runs/one/repo"}})[0]
    read = _action("/runs/two/repo/src/app.py")
    read.action_type = "FILE_READ"
    read = _comparison_actions([read], {"metadata": {"directory": "/runs/two/repo"}})[0]
    assert semantic_equivalent(command, read, fuzzy_commands=True)


def test_scorecard_explains_model_inferred_context_percentage():
    output = build_run_group_scorecard_html({"rows": [{"run_id": "one", "peak_pct": 8.9,
                                                      "peak_occupancy": 93700}]}, include_behavior=False)
    assert "capacity inferred from the model name" in output
    assert "these percentages are estimates" in output
    assert "title='Context percentages" in output


def _native_capture(directory):
    return {
        "info": {"id": "session", "location": {"directory": directory},
                 "time": {"created": 1000, "updated": 2000}},
        "messages": [{
            "id": "assistant", "type": "assistant", "finish": "stop",
            "time": {"created": 1000, "streamed": 1500, "completed": 2000},
            "tokens": {"input": 10, "output": 5, "reasoning": 0,
                       "cache": {"read": 0, "write": 0}},
            "content": [{"type": "tool", "name": "edit", "id": "write-app",
                         "state": {"status": "completed", "input": {"filePath": directory + "/src/app.py"},
                                   "content": [{"type": "text", "text": "updated"}]}}],
        }],
    }


def test_pairwise_and_run_group_share_workspace_identity_without_changing_source_bytes(tmp_path):
    sources = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_native_capture(f"/runs/{name}/repo")))
        sources.append(path)
    source_bytes = [path.read_bytes() for path in sources]
    runs = [load(path) for path in sources]
    steps = deepcopy([run.steps for run in runs])
    raw = deepcopy([run.raw for run in runs])

    pairwise = compare(*runs)
    group = compare_runs(runs)
    assert pairwise["alignment"]["alignment_f1"] == 1
    assert group["behavior"]["similarity"]["one"]["two"] == 1
    assert group["behavior"]["file_matrix_total"] == 1
    row = group["behavior"]["file_matrix"][0]
    assert (row["path"], row["n_runs"], row["kind"]) == ("src/app.py", 2, "consensus")
    assert all(cell["write"] == 1 for cell in row["cells"].values())
    assert [run.steps for run in runs] == steps
    assert [run.raw for run in runs] == raw
    assert [path.read_bytes() for path in sources] == source_bytes


def test_run_group_does_not_merge_similar_filenames_without_a_recorded_directory(tmp_path):
    paths = []
    for name in ("one", "two"):
        capture = _native_capture(f"/runs/{name}/repo")
        capture["info"].pop("location")
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(capture))
        paths.append(path)
    result = compare_runs(paths)
    assert result["behavior"]["file_matrix_total"] == 2
    assert all(row["n_runs"] == 1 for row in result["behavior"]["file_matrix"])
