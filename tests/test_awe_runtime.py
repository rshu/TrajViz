"""AWE jobs isolate code, secrets, network policy and output artifacts."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_trajviz.awe import runtime


@pytest.fixture
def engine(tmp_path, monkeypatch):
    package = tmp_path / "package" / "agent_trajviz"
    root = package / "awe" / "_engine"
    root.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    (root / "__init__.py").write_text("")
    script = '''import json, os, socket
from pathlib import Path
out = Path(__file__).resolve().parent.parent / 'data'
out.mkdir(exist_ok=True)
network = 'allowed'
try:
    socket.create_connection(('127.0.0.1', 9), timeout=.1)
except PermissionError:
    network = 'blocked'
except OSError:
    network = 'attempted'
data = {'network': network, 'inherited_secret': 'UNRELATED_API_KEY' in os.environ,
        'had_key': bool(os.environ.get('OPENROUTER_API_KEY')),
        'strict_cache': os.environ.get('AWE_STRICT_VERDICT_PROVENANCE'),
        'args': __import__('sys').argv[1:]}
(out / 'pipeline.json').write_text(json.dumps(data))
(out / 'cohort.json').write_text(json.dumps(os.environ.get('AWE_AGENTS', '').split(',')))
print('log-key:' + os.environ.get('OPENROUTER_API_KEY', 'none'))
'''
    for name in ("run_awe.py", "run_judge.py", "gen_attribution.py"):
        (root / "scripts" / name).write_text(script)
    (root / "scripts" / "run_judge.py").write_text(
        script + "\ndef target_instances(agent,include_resolved,limit):\n    return ['public-case']\n"
        + "def estimate_tokens(agent,instance):\n    return 42\n")
    (root / "config.py").write_text("import os\nAGENTS=os.environ.get('AWE_AGENTS','claude_code').split(',')\nJUDGE_MODEL='mock'\nJUDGE_TEMPERATURE=0\nJUDGE_MAX_TOKENS=2400\n")
    (root / "judge.py").write_text("mock_client=None\ndef judge_one(*args,**kwargs):\n    return {}\n")
    (root / "arbiter.py").write_text("")
    (root / "scout.py").write_text("")
    reference = tmp_path / "reference"
    reference.mkdir()
    # A private package copy tests the real process entry and descendant policy,
    # independently of the optional engine's trajectory adapter implementation.
    (package / "awe").mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "awe" / "__init__.py").write_text("")
    for name in ("runtime.py", "_worker.py"):
        shutil.copyfile(Path(runtime.__file__).with_name(name), package / "awe" / name)
    (package / "awe" / "native.py").write_text("def install_native_adapters():\n    pass\n")
    monkeypatch.setattr(runtime, "__file__", str(package / "awe" / "runtime.py"))
    monkeypatch.setenv("UNRELATED_API_KEY", "private-unrelated-key")
    settings = runtime.EvaluationSettings(reference, output_root=tmp_path / "jobs", agents=("claude_code",))
    return root, settings


def test_catalog_reports_capability_gaps_without_importing_backend(engine):
    root, settings = engine
    entries = {entry["id"]: entry for entry in runtime.available_operations(settings)}
    assert entries["pipeline"]["available"] is True
    assert entries["arbiter"]["available"] is False
    assert entries["judge"]["requires_network"] is True
    assert not (root / "data").exists()
    assert entries["pertest_harness"]["requires_execution"] is True
    assert "confirm_execution" in entries["pertest_harness"]["options"]


def test_offline_pipeline_has_isolated_code_inputs_and_descendant_network_policy(engine):
    root, settings = engine
    result = runtime.run_operation("pipeline", settings, agent="claude_code", limit=2)
    assert result.success and result.available
    assert result.output_dir and Path(result.output_dir).is_dir()
    data = json.loads((Path(result.output_dir) / "pipeline.json").read_text())
    assert data == {"network": "blocked", "inherited_secret": False, "had_key": False,
                    "strict_cache": "1", "args": ["--agent", "claude_code", "--limit", "2", "--workers", "1"]}
    assert not (root / "data").exists()
    manifest = json.loads((Path(result.output_dir) / "operation-manifest.json").read_text())
    assert manifest["network_enabled"] is False and manifest["api_key_stored"] is False
    assert "pipeline.json" in manifest["artifact_sha256"]


def test_private_worker_home_resolves_on_windows_without_inheriting_user_profile(engine, monkeypatch):
    root, settings = engine
    monkeypatch.setenv("HOME", "/unrelated-original-home")
    monkeypatch.setenv("USERPROFILE", "C:/unrelated-original-profile")
    (root / "scripts" / "run_awe.py").write_text(
        "import json,ntpath,os\nfrom pathlib import Path\n"
        "out=Path(__file__).parent.parent/'data'/'home.json'\n"
        "out.write_text(json.dumps({'home':os.environ.get('HOME'),"
        "'profile':os.environ.get('USERPROFILE'),'windows_home':ntpath.expanduser('~')}),encoding='utf-8')\n",
        encoding="utf-8")
    result = runtime.run_operation("pipeline", settings)
    assert result.success, (result.reason, result.logs)
    observed = json.loads((Path(result.output_dir) / "home.json").read_text(encoding="utf-8"))
    private_home = str(Path(result.project_data).parents[1])
    assert observed == {"home": private_home, "profile": private_home, "windows_home": private_home}
    assert "unrelated-original" not in json.dumps(result.to_dict())


def test_model_operations_need_explicit_network_or_offline_mode(engine):
    _, settings = engine
    rejected = runtime.run_operation("judge", settings)
    assert not rejected.available and "explicitly" in rejected.reason
    offline = runtime.run_operation("judge", settings, payload={"mock": True})
    assert offline.success
    assert offline.data["mode"] == "mock" and offline.data["verdicts"] == 1


def test_model_credentials_are_session_only_and_redacted_from_logs_and_manifest(engine):
    root, settings = engine
    selected = dataclasses.replace(settings, api_key="secret-session-key", base_url="http://127.0.0.1:9",
                                   judge_model="test-model", allow_network=True)
    assert "secret-session-key" not in repr(selected)
    assert "api_key" not in selected.nonsecret()
    result = runtime.run_operation("judge", selected)
    assert result.success and "[redacted]" in result.logs
    assert "secret-session-key" not in json.dumps(result.to_dict())
    for name in ("operation.log", "operation-manifest.json", "operation-result.json"):
        assert "secret-session-key" not in (Path(result.output_dir) / name).read_text()
    assert not (root / "data").exists()


def test_instance_selection_filters_model_targets_before_limit(engine):
    root, settings = engine
    (root / "scripts" / "run_judge.py").write_text(
        "def target_instances(agent,include_resolved,limit):\n"
        "    return ['case-first','case-selected'][:limit] if limit else ['case-first','case-selected']\n"
        "def estimate_tokens(agent,instance):\n    return 42\n")
    (root / "judge.py").write_text(
        "from pathlib import Path\nimport json\nmock_client=None\n"
        "def judge_one(agent,instance,**kwargs):\n"
        "    out=Path(__file__).parent/'data'/'selected.json'\n"
        "    out.write_text(json.dumps([instance]))\n    return {}\n")
    result = runtime.run_operation("judge", settings, instance_id="case-selected", limit=1, payload={"mock": True})
    assert result.success and result.data["targets"] == result.data["verdicts"] == 1
    assert json.loads((Path(result.output_dir) / "selected.json").read_text()) == ["case-selected"]
    dry = runtime.run_operation("judge", settings, instance_id="case-selected", limit=1, payload={"dry_run": True})
    assert dry.success and dry.data["targets"] == 1 and dry.data["model_requests"] == 0
    assert not (Path(dry.output_dir) / "selected.json").exists()


def test_explicit_agent_narrows_a_preconfigured_cohort(engine):
    _, settings = engine
    result = runtime.run_operation("pipeline", dataclasses.replace(settings, agents=("first", "second")), agent="selected")
    assert result.success
    assert json.loads((Path(result.output_dir) / "cohort.json").read_text()) == ["selected"]


def test_unsupported_scope_selectors_are_not_silently_ignored(engine):
    _, settings = engine
    result = runtime.run_operation("pipeline", settings, instance_id="selected")
    assert not result.success and "instance selector" in result.reason
    result = runtime.run_operation("graphs", settings, limit=1)
    assert not result.success and "case limit" in result.reason


def test_mock_and_dry_run_never_receive_the_session_credential(engine):
    _, settings = engine
    selected = dataclasses.replace(settings, api_key="session-credential", base_url="https://example.test/v1",
                                   judge_model="test-model", allow_network=True)
    for payload in ({"mock": True}, {"dry_run": True}):
        result = runtime.run_operation("judge", selected, payload=payload)
        assert result.success
        assert json.loads((Path(result.output_dir) / "pipeline.json").read_text())["had_key"] is False


@pytest.mark.parametrize("operation,options", [("unknown", {}), ("pipeline", {"shell": "touch bad"}),
    ("pipeline", {"workers": True}), ("judge", {"mock": "true"}), ("judge", {"panel": 100})])
def test_unknown_and_untyped_operation_options_never_start(engine, operation, options):
    _, settings = engine
    result = runtime.run_operation(operation, settings, payload=options)
    assert not result.available and not result.success and result.output_dir is None
    assert not Path(settings.output_root).exists()


def test_dossier_refresh_copies_preserved_inputs_without_mutating_them(engine, tmp_path):
    _, settings = engine
    dossier = tmp_path / "saved"
    dossier.mkdir()
    (dossier / "faults.json").write_text('[{"unknown_extension":{"value":1}}]')
    result = runtime.run_operation("graphs", settings, payload={"dossier": str(dossier)})
    assert result.success
    assert (Path(result.output_dir) / "dossier" / "faults.json").read_bytes() == (dossier / "faults.json").read_bytes()


def test_output_root_cannot_be_placed_inside_readonly_inputs(engine):
    root, settings = engine
    result = runtime.run_operation("pipeline", dataclasses.replace(settings, output_root=root / "outputs"))
    assert not result.available and "outside" in result.reason
    assert not (root / "outputs").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup check")
@pytest.mark.parametrize("ignore_termination", [False, True])
def test_timeout_stops_descendant_processes(engine, tmp_path, ignore_termination):
    root, settings = engine
    marker = tmp_path / "orphan-marker"
    child = ("import signal,time;from pathlib import Path;"
             + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_termination else "")
             + f"time.sleep(2);Path({str(marker)!r}).write_text('orphan')")
    script = ("import subprocess,sys,time\n"
              f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
              "time.sleep(20)\n")
    (root / "scripts" / "run_awe.py").write_text(script)
    result = runtime.run_operation("pipeline", dataclasses.replace(settings, timeout=1))
    assert result.available and not result.success and "timed out" in result.reason
    time.sleep(1.3)
    assert not marker.exists()


def test_windows_timeout_uses_process_tree_termination_without_a_shell(monkeypatch):
    process = Mock(pid=4321)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime._stop_process_tree(process)
    assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    assert not run.call_args.kwargs.get("shell", False)
    process.terminate.assert_not_called()


def test_windows_cleanup_failure_reports_unconfirmed_descendants(monkeypatch):
    process = Mock(pid=4321)
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(runtime.subprocess, "run", Mock(side_effect=OSError("not available")))
    assert not runtime._stop_process_tree(process, force=True)
    process.kill.assert_called_once()


@pytest.mark.parametrize("supplied", ["missing", "file", "blank"])
def test_paper_status_rejects_missing_directories_before_creating_a_job(tmp_path, supplied):
    path = tmp_path / "papers"
    if supplied == "file":
        path.write_text("This is a file, not a paper directory", encoding="utf-8")
    value = "  " if supplied == "blank" else str(path)
    result = runtime.run_operation("papers_status", runtime.EvaluationSettings(output_root=tmp_path / "jobs"),
                                   payload={"papers_dir": value})
    assert not result.success and not result.available
    assert "existing papers_dir" in result.reason
    assert not (tmp_path / "jobs").exists()


def test_posix_cleanup_tolerates_worker_exit_races(monkeypatch):
    process = Mock(pid=4321)
    killpg = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(runtime, "os", SimpleNamespace(name="posix", killpg=killpg))
    assert runtime._stop_process_tree(process, force=True)
    killpg.assert_called_once_with(4321, runtime.signal.SIGKILL)


def test_symlinked_backend_code_is_rejected(engine, tmp_path):
    root, settings = engine
    target = tmp_path / "external.py"
    target.write_text("raise AssertionError('external code')")
    (root / "external.py").symlink_to(target)
    result = runtime.run_operation("pipeline", settings)
    assert not result.available and "Symbolic links" in result.reason


@pytest.mark.parametrize("reply", ["raise RuntimeError('private-provider-response')",
    "return {'parse_failed':True,'finish_reason':'length'}"])
def test_model_failures_and_invalid_verdicts_cannot_report_success(engine, reply):
    root, settings = engine
    (root / "config.py").write_text("AGENTS=['claude_code']\nJUDGE_MODEL='mock'\nJUDGE_TEMPERATURE=0\n")
    (root / "judge.py").write_text("mock_client=None\ndef judge_one(*args,**kwargs):\n    " + reply + "\n")
    (root / "arbiter.py").write_text("")
    (root / "scout.py").write_text("")
    (root / "scripts" / "run_judge.py").write_text("def target_instances(agent,include_resolved,limit):\n    return ['public-case']\n")
    result = runtime.run_operation("judge", settings, payload={"mock": True})
    assert result.available and not result.success
    assert result.data["verdicts"] == 0
    assert len(result.data["errors"]) + result.data["invalid_responses"] == 1
    assert "private-provider-response" not in json.dumps(result.to_dict())


@pytest.fixture
def bundled_engine(tmp_path):
    assert (runtime._engine_root() / "provenance.py").is_file(), "Bundled evaluation engine must be installed"
    reference = Path(__file__).resolve().parent / "fixtures" / "reference"
    if not reference.is_dir():
        pytest.skip("public benchmark fixtures unavailable")
    return runtime.EvaluationSettings(reference, project_data=tmp_path / "empty-inputs",
                                   output_root=tmp_path / "jobs", agents=("claude_code",),
                                   benchmark_name="SWE-bench Verified", judge_model="mock", timeout=90)


def test_bundled_engine_native_claude_full_dossier_and_provenance(bundled_engine, tmp_path):
    source = Path(__file__).resolve().parent / "fixtures"
    reference = tmp_path / "reference"
    shutil.copytree(source / "reference", reference)
    case = "django__django-11477"
    canonical = reference / "data" / "trajectory" / "claude_code"
    (canonical / (case + ".json")).unlink()
    shutil.copyfile(source / "native_claude_code" / (case + ".jsonl"), canonical / (case + ".jsonl"))
    result = runtime.run_operation("diagnose", dataclasses.replace(bundled_engine, reference_root=reference),
                                   agent="claude_code", instance_id=case)
    assert result.success and result.data["available"] is True
    full = result.data["case"]
    assert {"attribution_trace", "event_ledger", "fault_side", "thinking_process", "symptom_analysis",
            "execution_context", "benchmark", "task"} <= full.keys()
    assert all("audit" in fault and "countermeasure_decisions" in fault for fault in full["faults"])
    assert result.data["trajectory_source"]["native_format"] == "claude_code"
    assert result.data["trajectory_source"]["files"]
    assert result.data["fresh_model_requests"] == 0


def test_bundled_engine_pipeline_returns_one_immutable_complete_run(bundled_engine):
    result = runtime.run_operation("pipeline", bundled_engine, limit=1)
    assert result.success
    archive = Path(result.output_dir)
    assert archive.parent.name == "runs" and result.project_data != result.output_dir
    for name in ("manifest.json", "faults.json", "attribution_graphs.json", "recommendations.json",
                 "rsi_proposals.json", "rsi_feedback.json", "symptoms.json", "insights.json", "INSIGHTS.md"):
        assert (archive / name).is_file(), name
    assert list((archive / "reports").glob("*.md"))
    assert len(json.loads((archive / "faults.json").read_text())) == 1


def test_bundled_engine_verified_mock_cache_reaches_full_diagnosis(bundled_engine):
    judge = runtime.run_operation("judge", bundled_engine, limit=1, payload={"mock": True})
    assert judge.success and judge.data["verdicts"] == 1 and judge.data["errors"] == []
    with_cache = dataclasses.replace(bundled_engine, project_data=judge.project_data)
    result = runtime.run_operation("diagnose", with_cache, agent="claude_code", instance_id="astropy__astropy-13033")
    assert result.success and result.data["used_judge"] is True
    assert result.data["judge_policy"] == "verified_cache_only" and result.data["fresh_model_requests"] == 0


def test_bundled_engine_catalog_and_controlled_python_faults(bundled_engine):
    catalog = runtime.run_operation("catalog", bundled_engine)
    assert catalog.success and len(catalog.data["entries"]) >= 33
    checks = catalog.data["quality_checks"]
    assert checks["passed"] == checks["n"]
    faults = runtime.run_operation("fault_injection", bundled_engine,
                                   payload={"language": "python", "case_timeout": 5})
    assert faults.success
    report = json.loads((Path(faults.output_dir) / "fault-injection.json").read_text())
    assert report["summary"]["passed"] is True


def test_bundled_engine_guardrail_snapshot_preserves_duplicate_event_budget(bundled_engine):
    payload = {"run_id": "public-runtime-test", "recipe_id": "CM-EDIT-REGRESSION-1", "event_id": "submit-1",
               "observed": {"status": "unknown", "reason": "verification_unavailable", "exception": None}}
    first = runtime.run_operation("guardrail_decide", bundled_engine, payload=payload)
    assert first.success and first.data["decision"]["action"] == "warn"
    second = runtime.run_operation("guardrail_decide", bundled_engine,
                                   payload={**payload, "snapshot": first.data["snapshot"]})
    assert second.success and second.data["decision"]["replayed"] is True
    assert second.data["snapshot"] == first.data["snapshot"]
    assert second.data["installed_hooks"] is False


def test_reference_free_catalog_uses_private_empty_inputs(tmp_path):
    result = runtime.run_operation("catalog", runtime.EvaluationSettings(output_root=tmp_path / "jobs"))
    assert result.success and len(result.data["entries"]) >= 33
    assert (Path(result.output_dir).parents[1] / "reference-inputs" / "data").is_dir()
    preflight = runtime.run_operation("preflight", runtime.EvaluationSettings())
    assert not preflight.success and "reference data directory" in preflight.reason


def test_empty_agent_settings_derive_only_official_outcome_cohort(bundled_engine, tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(Path(bundled_engine.reference_root), reference)
    data = reference / "data"
    for file in data.glob("eval_*.json"):
        file.unlink()
    instance = "astropy__astropy-13033"
    for agent in ("pi", "cursor-custom"):
        (data / ("eval_" + agent + ".json")).write_text(json.dumps({instance: False}))
        for directory, suffix in (("patch", ".diff"), ("trajectory", ".json")):
            target = data / directory / agent
            target.mkdir()
            shutil.copyfile(data / directory / "claude_code" / (instance + suffix), target / (instance + suffix))
    result = runtime.run_operation("pipeline", dataclasses.replace(bundled_engine, reference_root=reference, agents=()), limit=1)
    assert result.success
    manifest = json.loads((Path(result.output_dir) / "manifest.json").read_text())
    assert sorted(manifest["agents"]) == ["cursor-custom", "pi"]
    assert {case["agent"] for case in json.loads((Path(result.output_dir) / "faults.json").read_text())} == {"cursor-custom", "pi"}


def test_bundled_engine_harbor_ingestion_preserves_inputs_and_excludes_unfinished_trials(bundled_engine, tmp_path):
    harbor = tmp_path / "harbor"
    group = harbor / "public-run"
    bundle = group / "preflight" / "task_bundles" / "fixtures" / "task-one"
    for name in ("solution", "environment", "tests/private"):
        (bundle / name).mkdir(parents=True, exist_ok=True)
    patch = "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-return 0\n+return 1\n"
    (bundle / "solution" / "solution.patch").write_text(patch)
    (bundle / "instruction.md").write_text("Return one for the recorded input.")
    (bundle / "tests" / "private" / "f2p.txt").write_text("tests/test_a.py::test_one\n")
    (bundle / "environment" / "metadata.json").write_text(json.dumps({"repo": "example/project", "base_commit": "a" * 40}))
    trial = group / "agent" / "task-one" / "job" / "trial__one"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "agent").mkdir()
    recorded = {"task_name": "task-one", "finished_at": "2026-01-01T00:00:00Z", "verifier_result": {"rewards": {"resolved": 0}}}
    (trial / "result.json").write_text(json.dumps(recorded))
    (trial / "artifacts" / "solution.patch").write_text(patch)
    (trial / "agent" / "trajectory.json").write_text('{"trajectory":[{"role":"assistant","content":"Checking the public input."}]}')
    unfinished = trial.parent / "trial__unfinished"
    unfinished.mkdir()
    (unfinished / "result.json").write_text('{"task_name":"task-one"}')
    original = (trial / "result.json").read_bytes()
    result = runtime.run_operation("ingest_harbor", bundled_engine,
                                   payload={"harbor_root": str(harbor), "run_groups": ["public-run"], "agent_name": "harbor-agent"})
    assert result.success
    canonical = Path(result.data["canonical_root"])
    assert json.loads((canonical / "data" / "eval_harbor-agent.json").read_text()) == {"trial__one": False}
    assert result.data["ingestion"]["n_ingested"] == 1 and result.data["ingestion"]["n_skipped"] == 1
    assert (canonical / "data" / "patch" / "harbor-agent" / "trial__one.diff").read_text() == patch
    assert (trial / "result.json").read_bytes() == original


def test_bundled_engine_rsi_prepare_and_decide_preserve_experiment_boundaries(bundled_engine, tmp_path):
    context = {"agent": "claude_code", "benchmark": "SWE-bench Verified", "languages": ["Python"],
               "model": "public-test-model", "provider": "public-test-provider", "scaffold_revision": "revision-1",
               "capabilities": ["hook.pre_submit", "tool.project_tests", "state.workspace_revision", "state.local_verification"]}
    case = {"case_id": "claude_code__public-case", "agent": "claude_code", "instance_id": "public-case",
            "benchmark": "SWE-bench Verified", "language": "Python", "execution_context": context,
            "blame_status": "primary", "faults": [{"capability": "code_editing", "error_type": "regression_introduced",
            "blame_weight": 1, "is_primary": True, "confidence": "high", "detector_evidence": {"p2p_broken": ["private-test-id"]},
            "code_evidence": {"gold": "PRIVATE REFERENCE PATCH"}}]}
    dossier = tmp_path / "saved-dossier"
    dossier.mkdir()
    (dossier / "faults.json").write_text(json.dumps([case]))
    exported = runtime.run_operation("recommendations", bundled_engine, payload={"dossier": str(dossier)})
    assert exported.success
    proposals = json.loads((Path(exported.output_dir) / "dossier" / "rsi_proposals.json").read_text())
    proposal = next(item for item in proposals["proposals"] if item["recipe_id"] == "CM-EDIT-REGRESSION-1")
    prepared = runtime.run_operation("rsi_prepare", bundled_engine,
                                     payload={"proposal": proposal, "observed_context": context})
    assert prepared.success and prepared.data["status"] == "prepared_not_installed"
    state = {"workspace_revision": "current", "tests_available": True,
             "verification": {"revision": "old", "status": "passed"}}
    decided = runtime.run_operation("rsi_decide", bundled_engine, payload={"prepared": prepared.data, "state": state})
    assert decided.success and decided.data["action"] == "request_project_tests"
    assert decided.data["next_trigger_count"] == 1 and decided.data["blocks_submission"] is False
    feedback = (Path(exported.output_dir) / "dossier" / "rsi_feedback.json").read_text()
    assert "PRIVATE REFERENCE PATCH" not in feedback and "private-test-id" not in feedback
    assert json.loads(feedback)["runtime_input_allowed"] is False


@pytest.mark.parametrize("failure,phase,errno", [
    ("start", "starting isolated worker", 13),
    ("wait", "waiting for isolated worker", 5),
    ("cleanup", "stopping timed-out process tree", 1),
])
def test_parent_oserror_reports_phase_and_errno_without_private_details(engine, monkeypatch, failure, phase, errno):
    _, settings = engine
    private = "private-command-and-environment-detail"
    error = PermissionError(errno, private, "/private/input-path") if errno != 5 else OSError(errno, private)
    process = Mock()
    popen = Mock(return_value=process)
    if failure == "start":
        popen.side_effect = error
    elif failure == "wait":
        process.communicate.side_effect = error
    else:
        process.communicate.side_effect = subprocess.TimeoutExpired(private, 1)
        monkeypatch.setattr(runtime, "_stop_process_tree", Mock(side_effect=error))
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    result = runtime.run_operation("pipeline", settings)
    assert not result.success and not result.available
    assert phase in result.reason and f"errno={errno}" in result.reason
    assert type(error).__name__ in result.reason
    assert private not in result.reason and "/private/input-path" not in result.reason
    assert result.logs == "" and result.output_dir is None
