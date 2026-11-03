"""Bundled evaluators import without toy workspaces, optional harnesses or data."""

from __future__ import annotations

import builtins
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ENGINE = Path(__file__).resolve().parents[1] / "agent_trajviz" / "awe" / "_engine"
_FILES = (
    "fixtures/fault_injection/python/evaluate.py",
    "scripts/calibrate_difficulty_strata.py",
    "scripts/compare_grounding.py",
    "scripts/grader_log_audit.py",
)


def _load(relative: str):
    spec = importlib.util.spec_from_file_location("evaluation_import_safety_" + Path(relative).stem, _ENGINE / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", _FILES)
def test_import_does_not_require_data_or_optional_workspace(relative, monkeypatch, tmp_path, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "core" or name.startswith("swebench"):
            raise AssertionError(f"optional execution dependency imported: {name}")
        return original_import(name, *args, **kwargs)

    def forbid_data_read(*args, **kwargs):
        raise AssertionError("script read a data file while importing")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", forbid_data_read)
    _load(relative)
    assert capsys.readouterr().out == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "body, resolved, crashes",
    [
        ("def clamp(x, lo, hi): return max(lo, min(x, hi))\ndef add(a,b): return a+b\n", True, False),
        ("def clamp(x, lo, hi): return x\ndef add(a,b): return a+b\n", False, False),
        ("raise RuntimeError('toy workspace crash')\n", None, True),
    ],
)
def test_fault_fixture_main_preserves_grades_and_crashes(body, resolved, crashes, tmp_path):
    evaluator = tmp_path / "evaluate.py"
    shutil.copy2(_ENGINE / _FILES[0], evaluator)
    (tmp_path / "core.py").write_text(body, encoding="utf-8")
    result = subprocess.run([sys.executable, str(evaluator)], cwd=tmp_path, capture_output=True, text=True, check=False)
    if crashes:
        assert result.returncode != 0
        assert result.stdout == ""
        assert "toy workspace crash" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["resolved"] is resolved
        assert [test["id"] for test in report["tests"]] == [
            "lower_bound",
            "upper_bound",
            "inside_interval",
            "addition",
        ]
        assert all(test["passed"] for test in report["tests"]) is resolved


def test_difficulty_rows_load_current_project_data_lazily(monkeypatch, tmp_path):
    module = _load(_FILES[1])
    monkeypatch.setattr(module.config, "PROJECT_DATA", tmp_path)
    labels = tmp_path / "discovery" / "swebench_verified_difficulty.json"
    labels.parent.mkdir()
    labels.write_text(json.dumps({"case-1": "<15 min fix"}), encoding="utf-8")
    monkeypatch.setattr(module, "iter_cases", lambda: [("agent", "case-1", False, SimpleNamespace())])
    monkeypatch.setattr(module, "load_agent_patch", lambda *args: None)
    monkeypatch.setattr(module, "signals_cached", lambda *args: None)
    monkeypatch.setattr(module, "localization_signals", lambda *args, **kwargs: {"hit": False, "gold_files": []})
    expected = [
        {
            "difficulty": "<15 min fix",
            "failed": 1,
            "solved_unrelated": 0,
            "reverted_own_change": 0,
            "excessive_extra_edits": 0,
        }
    ]
    assert module.rows_of() == expected
    labels.unlink()
    assert module.rows_of({"case-1": "<15 min fix"}) == expected
    module.LABELS = {"case-1": "<15 min fix"}
    assert module.rows_of() == expected
    assert module.rows_of({}) == []


def test_grounding_collect_is_objective_and_main_writes_only_configured_output(monkeypatch, tmp_path, capsys):
    module = _load(_FILES[2])
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "outputs"
    monkeypatch.setattr(module.config, "OUTPUT_DATA", destination)
    monkeypatch.setattr(module.config, "AGENTS", ["agent"])
    monkeypatch.setattr(module.config, "list_instances", lambda: ["failed", "solved"])
    calls = []

    def scores(agent, **options):
        calls.append((agent, options))
        return {"failure_rate": 0.5, "capabilities": {key: {"score": 80.1} for key in module.CAPS}}

    def detect(instance, agent, **options):
        assert instance == "failed" and agent == "agent"
        assert options == {"use_trajectory": True, "use_judge": False}
        return {"candidates": [{"error_type": "relevant_change_omitted"}]}

    monkeypatch.setattr(module, "agent_capability_scores", scores)
    monkeypatch.setattr(module, "resolved", lambda agent, iid: iid == "solved")
    monkeypatch.setattr(module, "detect", detect)
    monkeypatch.setattr(module, "build_blame", lambda dossier: {"primary": {"capability": "code_editing"}})
    expected = {
        "agent": {
            "fail_pct": 50.0,
            "scores": dict.fromkeys(module.CAPS, 80.1),
            "primary": {"code_editing": 1},
            "cases_with_omission": 1,
        }
    }
    assert module.collect() == expected
    assert calls == [("agent", {"use_trajectory": True, "use_judge": False})]
    assert not destination.exists()
    module.main()
    assert json.loads((destination / "grounding_compare.json").read_text()) == expected
    assert list(tmp_path.iterdir()) == [destination]
    assert "code_editing: 1" in capsys.readouterr().out


def test_grader_loads_optional_harness_only_for_existing_logs_and_preserves_registry(monkeypatch, tmp_path):
    module = _load(_FILES[3])
    monkeypatch.setattr(module.config, "PROJECT_DATA", tmp_path)
    assert module.audit_case("agent", "owner__repo-1") is None
    records = tmp_path / "pertest" / "agent"
    records.mkdir(parents=True)
    (records / "owner__repo-1.json").write_text(
        json.dumps(
            {
                "owner__repo-1": {
                    "tests_status": {"FAIL_TO_PASS": {"success": ["test_one"]}},
                }
            }
        )
    )
    assert module.audit_case("agent", "owner__repo-1")["eval_status"] == "no_log"
    assert module._HARNESS_CONSTANTS is None
    logs = tmp_path / "cache" / "harness" / "agent" / "owner__repo-1"
    logs.mkdir(parents=True)
    (logs / "test_output.txt").write_text("setup\nSTART_TEST_OUTPUT\ntest_one ... ok\n")
    parser_calls = []

    def custom_parser(body, test_spec):
        parser_calls.append((body, test_spec))
        return {"test_one": "PASSED"}

    module.MAP_REPO_TO_PARSER["owner/repo"] = custom_parser
    constants = ModuleType("swebench.harness.constants")
    constants.START_TEST_OUTPUT = "START_TEST_OUTPUT"
    constants.APPLY_PATCH_FAIL = "APPLY_PATCH_FAIL"
    constants.TESTS_TIMEOUT = "TESTS_TIMEOUT"
    constants.TestStatus = SimpleNamespace(ERROR=SimpleNamespace(value="ERROR"))
    parsers = ModuleType("swebench.harness.log_parsers")
    parsers.MAP_REPO_TO_PARSER = {"owner/repo": lambda *args: {"test_one": "FAILED"}}
    harness = ModuleType("swebench.harness")
    harness.constants, harness.log_parsers = constants, parsers
    package = ModuleType("swebench")
    package.harness = harness
    for name, value in {
        "swebench": package,
        "swebench.harness": harness,
        "swebench.harness.constants": constants,
        "swebench.harness.log_parsers": parsers,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    result = module.audit_case("agent", "owner__repo-1")
    assert result["eval_status"] == "graded"
    assert result["gating"] == {"test_one": "PASSED"}
    assert result["parsed_counts"] == {"PASSED": 1}
    assert parser_calls == [("START_TEST_OUTPUT\ntest_one ... ok\n", None)]
    assert module.MAP_REPO_TO_PARSER["owner/repo"] is custom_parser
