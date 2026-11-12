"""File evidence comes from command operands, not formatter flags or executables."""
from __future__ import annotations

import json

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.diagnostics import _extract_bash_paths, extract_file_interactions
from agent_trajviz.insight.formats.codex import _build_codex_tool_input, _classify_codex_command


@pytest.mark.parametrize(("command", "pattern", "path"), [
    ("rg -n 'column' astropy | head -n 80", "column", "astropy"),
    ("rg -n 'a|b' 'source folder' | sed -n '1,200p'", "a|b", "source folder"),
    ("rg -n query src -g '*.c' | head -n 80", "query", "src"),
    ("rg -A 80 query src", "query", "src"),
    ("rg --context=80 query src", "query", "src"),
    ("rg -e 'two words' --glob '*.py' src", "two words", "src"),
    ("grep -n -e query source.py 2>/dev/null", "query", "source.py"),
    ("grep -T --color query source.py", "query", "source.py"),
    ("rg --color never query source.py", "query", "source.py"),
    ("rg -f patterns.txt src", "patterns.txt", "src"),
    ("rg query 80", "query", "80"),
    ("rg -- -pattern 80", "-pattern", "80"),
    ("rg query | head -n 80", "query", ""),
])
def test_codex_search_scope_excludes_pipeline_and_option_arguments(command, pattern, path):
    result = _build_codex_tool_input("Grep", command, {})
    assert (result["pattern"], result["path"]) == (pattern, path)
    assert result["command"] == command


@pytest.mark.parametrize(("command", "path"), [
    ("head -n 80 'source folder/file.py' | tail -n 3", "source folder/file.py"),
    ("head --lines=80 source.py", "source.py"),
    ("tail -c 80 source.py", "source.py"),
    ("sed -n '1,200p' source.py | head -n 80", "source.py"),
    ("sed -n -e '1,200p' source.py", "source.py"),
    ("sed --expression='1,200p' source.py", "source.py"),
    ("cat -- 80", "80"),
    ("cat README", "README"),
    ("head -n 80", ""),
])
def test_codex_read_operands_preserve_real_filenames(command, path):
    assert _build_codex_tool_input("Read", command, {})["file_path"] == path


@pytest.mark.parametrize(("command", "paths"), [
    ("/opt/venv/bin/python --version", []),
    ("CACHE=/tmp/cache /opt/venv/bin/python /repo/script.py", ["/repo/script.py"]),
    ("env -u SECRET CACHE=/tmp/cache /opt/venv/bin/python /repo/script.py", ["/repo/script.py"]),
    ("cd /repo && /usr/bin/python -m pytest /repo/tests/test_api.py", ["/repo/tests/test_api.py"]),
    ("/usr/bin/cat '/repo/source file.py'", ["/repo/source file.py"]),
    (r'type "C:\repo\source.py"', ["C:/repo/source.py"]),
    (r'cat "C:\repo\source.py"', ["C:/repo/source.py"]),
    (r'cmd /d /c "type C:\repo\source.py"', ["C:/repo/source.py"]),
    (r'pwsh -Command "Get-Content C:\repo\source.py"', ["C:/repo/source.py"]),
    (r'pwsh -File C:\repo\script.ps1', ["C:/repo/script.ps1"]),
    ('pwsh -File -', []),
    ("bash -lc 'cat /repo/source.py'", ["/repo/source.py"]),
    ("which /usr/bin/python", []),
    ("python - <<'PY'\npaths = ['/tmp/not-shell-a', '/tmp/not-shell-b']\nPY\ncat /repo/after.py", ["/repo/after.py"]),
    ("echo '<<EOF'\ncat /repo/after.py", ["/repo/after.py"]),
    ("cat <<< literal\ncat /repo/after.py", ["/repo/after.py"]),
])
def test_bash_file_references_exclude_non_file_arguments(command, paths):
    assert _extract_bash_paths(command) == paths


@pytest.mark.parametrize(("command", "workdir", "interactions"), [
    ("cd '/capture/repo root' && sed -n '4,10p' 'src/file one.py' | head -n 80", "", [
        ("/capture/repo root/src/file one.py", "read"),
    ]),
    ("cd /capture/repo && grep -n -A 3 needle src/a.py src/b.py && cat README", "", [
        ("/capture/repo/src/a.py", "search"), ("/capture/repo/src/b.py", "search"),
        ("/capture/repo/README", "read"),
    ]),
    ("cat ./source.py /tmp/input.py > /tmp/output.py", "/capture/repo", [
        ("/capture/repo/source.py", "read"), ("/tmp/input.py", "read"), ("/tmp/output.py", "write"),
    ]),
    ("cd /capture/repo && cat /capture/repo/source.py && rm /capture/repo/source.py", "", [
        ("/capture/repo/source.py", "read"), ("/capture/repo/source.py", "write"),
    ]),
    ("cat ./source.py > ./source.py", "/capture/repo", [
        ("/capture/repo/source.py", "read"), ("./source.py", "write"),
    ]),
    ("cd /capture/repo || cat src/file.py", "", []),
    ("tools/cd /capture/repo && cat src/file.py", "", []),
    ("cd /capture/repo && cat README; cat ambiguous.py", "", [("/capture/repo/README", "read")]),
    ("cd /capture/repo && cd subdir && cat src/file.py", "", []),
    ("cd /capture/repo && cat README # ignore adjacent.py", "", [("/capture/repo/README", "read")]),
    ("cat missing_scope.py", "relative-workdir", []),
    ("cd /capture/repo && cat *.py \"$INPUT\"", "", []),
    ("cd /capture/repo && sed -i 's/a/b/' target.py", "", []),
])
def test_shell_file_diagnostics_uses_scoped_reader_and_search_operands(command, workdir, interactions):
    steps = [{"index": 0, "tokens": {"total": 10}, "tool_calls": [
        {"tool_name": "shell", "input": {"command": command, "workdir": workdir}},
    ]}]
    assert [(row["path"], row["type"]) for row in extract_file_interactions(steps)] == interactions


@pytest.mark.parametrize("scope", [".", "..", "./", "../", "/"])
def test_generic_search_roots_do_not_become_file_rows(scope):
    steps = [{"index": 0, "tokens": {"total": 10}, "tool_calls": [
        {"tool_name": "Grep", "input": {"path": scope, "pattern": "query"}},
    ]}]
    assert extract_file_interactions(steps) == []


def test_explicit_numeric_file_is_not_filtered_as_a_count():
    steps = [{"index": 0, "tokens": {"total": 10}, "tool_calls": [
        {"tool_name": "Read", "input": {"file_path": "80"}},
        {"tool_name": "Grep", "input": {"path": "80", "pattern": "query"}},
    ]}]
    assert [row["path"] for row in extract_file_interactions(steps)] == ["80", "80"]


def test_native_codex_file_view_uses_real_scope_not_head_count(tmp_path):
    commands = [
        "rg -n 'column' astropy | head -n 80",
        "rg -n 'query' astropy/io/fits -g '*.c' | sed -n '1,200p'",
        "find .. -maxdepth 3 -name python | head -n 80",
        "/opt/venv/bin/python --version",
        "CACHE=/tmp/cache /opt/venv/bin/python -m pytest /repo/tests/test_api.py",
    ]
    events = [{"type": "session_meta", "payload": {"id": "file-evidence", "cli_version": "0.164.0"}}]
    for number, command in enumerate(commands):
        events.extend([
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                "call_id": str(number), "arguments": json.dumps({"cmd": command})}},
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": str(number),
                "output": "Process exited with code 0"}},
        ])
    events.append({"type": "event_msg", "payload": {"type": "task_complete"}})
    source = tmp_path / "rollout.jsonl"
    source.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    run = tv.load(source)
    data = run.view("file_interactions").data
    assert set(data["files"]) == {"astropy", "astropy/io/fits", "/repo/tests/test_api.py"}
    assert len(data["interactions"]) == 3
    restored = tv.Trajectory.from_dict(run.to_dict(include_raw=True))
    assert restored.view("file_interactions").data == data


@pytest.mark.parametrize(("command", "path"), [
    ("cat > 'output folder/result.py' <<'PY'\nvalue = 1\nPY", "output folder/result.py"),
    ("cat >>80", "80"),
    ("cat >/repo/result.py", "/repo/result.py"),
])
def test_direct_cat_redirects_preserve_actual_write_targets(command, path):
    name = _classify_codex_command("exec_command", command)
    assert name == "Write"
    assert _build_codex_tool_input(name, command, {})["file_path"] == path
