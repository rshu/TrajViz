"""attribution API drift must preserve isolation and the no-unverified-verdict rule.

These small synthetic APIs cover both supported attribution generations without a
checkout, credentials, network, developer data or cached paid model results.
Optional modern-source checks use only the public reference fixtures and the
isolated cache configuration installed by conftest.py.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from agent_trajviz.insight import attribution as seam


@pytest.fixture
def attribution(tmp_path, monkeypatch):
    package = ModuleType("agent_trajviz.awe._engine")
    package.__path__ = []
    import agent_trajviz.awe as evaluation_package
    monkeypatch.setattr(evaluation_package, "_engine", package, raising=False)
    monkeypatch.setitem(sys.modules, "agent_trajviz.awe._engine", package)
    modules = {}
    for name in ("config", "gold", "outcomes", "adapters", "detect", "dossier", "evidence", "rootcause"):
        module = ModuleType("agent_trajviz.awe._engine." + name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, "agent_trajviz.awe._engine." + name, module)
        setattr(package, name, module)
    config = modules["config"]
    config.AGENTS = ["claude_code", "codex", "pi"]
    config.requirements_path = lambda name: config.REQUIREMENTS_DIR / (name + ".json")
    config.patch_path = lambda agent, name: config.PATCH_DIR / agent / (name + ".patch")
    config.eval_path = lambda agent: config.DATA / ("eval_" + agent + ".json")
    config.PROJECT_DATA = tmp_path / "project-data"
    monkeypatch.setattr(seam, "_cfg", config, raising=False)
    monkeypatch.setattr(seam, "ATTRIBUTION_AVAILABLE", True)
    monkeypatch.setattr(seam, "_DEFAULT_ROOT", tmp_path)
    gold = SimpleNamespace(problem_statement="Synthetic task", gold_files=["src/a.py"],
                           is_multi_file=False, is_cross_module=False)
    modules["gold"].build_gold_reference = lambda name: gold
    modules["outcomes"].load_outcomes = lambda agent: {"case-1": False}
    seam.configure(tmp_path)
    config.REQUIREMENTS_DIR.mkdir(parents=True)
    config.TRAJECTORY_DIR.mkdir(parents=True)
    config.requirements_path("case-1").write_text(json.dumps({"gold_patch": "synthetic diff"}))
    path = config.TRAJECTORY_DIR / "claude_code" / "case-1.json"
    path.parent.mkdir()
    path.write_text('{"trajectory": [{"role": "assistant", "content": "Synthetic workflow step"}]}')
    stamp = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    step = SimpleNamespace(step_id=0, text="Synthetic workflow step", output_excerpt=None)
    modules["adapters"].load_normalized = lambda agent, name: SimpleNamespace(source=stamp, steps=[step])
    calls = []
    def detect(name, agent, *, use_judge=None):
        calls.append(("detect", use_judge))
        return {"resolved": False, "opportunities": {}, "candidates": [{
            "capability": "code_editing", "error_type": "incorrect_patch", "confidence": "high", "evidence": {}}]}
    modules["detect"].detect = detect
    def dossier(agent, name):
        pytest.fail("modern case_record must not auto-read unverified model caches")
    modules["dossier"].case_record = dossier
    modules["dossier"].file_hunks = lambda text: {}
    modules["evidence"].changed_lines = lambda text: {}
    modules["evidence"].build_chain = lambda *args, **kwargs: {"strength": "deductive", "links": []}
    def build_blame(detection, *, adjudicate=None):
        calls.append(("blame", adjudicate))
        primary = {"capability": "code_editing", "error_type": "incorrect_patch", "weight": 1.0}
        return {"status": "primary", "primary": primary, "blame": [primary]}
    modules["rootcause"].build_blame = build_blame
    return SimpleNamespace(root=tmp_path, modules=modules, config=config, path=path, stamp=stamp, calls=calls)


def test_uncached_and_content_keyed_cached_functions_both_configure(attribution):
    cleared = []
    attribution.modules["gold"].build_gold_reference.cache_clear = lambda: cleared.append("gold")
    attribution.modules["outcomes"]._read_json = SimpleNamespace(cache_clear=lambda: cleared.append("json"))
    attribution.modules["outcomes"]._read_outcomes = SimpleNamespace(cache_clear=lambda: cleared.append("outcomes"))
    seam.configure(attribution.root)
    assert cleared == ["gold", "json", "outcomes"]


def test_modern_api_returns_real_objective_blame_without_model_cache_reads(attribution):
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert result.available and result.primary == {"capability": "code_editing", "error_type": "incorrect_patch"}
    assert result.used_judge is False and result.arbiter is None
    assert result.scorecard[3].blamed
    assert attribution.calls == [("detect", False), ("blame", None)]
    assert result.notes and "without cached LLM verdicts" in result.notes[0]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("normalized", [None, SimpleNamespace(steps=[])])
def test_unrecognized_or_empty_normalization_never_issues_blame(attribution, normalized, legacy):
    attribution.modules["adapters"].load_normalized = lambda agent, name: normalized
    if legacy:
        def legacy_record(agent, name, *, use_judge=None, adjudicate=None):
            pytest.fail("an unsupported trajectory must not reach the legacy dossier")
        attribution.modules["dossier"].case_record = legacy_record
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert result.available is False
    assert result.mode == "reference" and result.primary is None and result.faults == []
    assert "could not extract workflow steps" in result.reason
    assert attribution.calls == []


def test_legacy_api_keeps_per_call_provenance_controls(attribution, monkeypatch):
    marker = object()
    monkeypatch.setattr(seam, "_llm_layer_policy", lambda *args: (False, marker, ["verified policy"]))
    def legacy(agent, name, *, use_judge=None, adjudicate=None):
        assert use_judge is False and adjudicate is marker
        return {"agent": agent, "instance_id": name, "blame_status": "unattributed", "faults": []}
    attribution.modules["dossier"].case_record = legacy
    attribution.modules["adapters"].canonical_trajectory_path = lambda agent, name: attribution.path
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert result.available and result.notes == ["verified policy"]
    assert attribution.calls == [("detect", False)]


@pytest.mark.parametrize("policy,available,expected", [
    (None, True, True),     # Auto mode found a usable cached verdict.
    (None, False, False),   # No usable verdict, including a failed parse.
    (None, None, False),    # Judge inactive: no availability signal at all.
    (False, True, False),   # Provenance policy disables the layer explicitly.
    (True, False, False),   # Enabling the layer does not establish coverage.
])
def test_legacy_judge_usage_reports_actual_coverage_not_the_policy_flag(
        attribution, monkeypatch, policy, available, expected):
    monkeypatch.setattr(seam, "_llm_layer_policy", lambda *args: (policy, None, []))

    def legacy(agent, name, *, use_judge=None, adjudicate=None):
        assert use_judge is policy and adjudicate is None
        return {"agent": agent, "instance_id": name, "blame_status": "unattributed", "faults": []}

    def detect(name, agent, *, use_judge=None):
        attribution.calls.append(("detect", use_judge))
        # A task opportunity alone does not mean a cached judge assessed it.
        signals = {} if available is None else {"judge_available": available}
        return {"resolved": False, "opportunities": {"requirement_understanding": True},
                "signals": signals}

    attribution.modules["dossier"].case_record = legacy
    attribution.modules["detect"].detect = detect
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert result.available and result.used_judge is expected
    assert attribution.calls == [("detect", policy)]


def test_current_canonical_resolution_matches_codex_and_jsonl_fallback(attribution):
    for agent in ("codex", "pi"):
        parent = attribution.config.TRAJECTORY_DIR / agent
        parent.mkdir()
        jsonl = parent / "case-1.jsonl"
        jsonl.write_text('{}\n')
        assert seam._canonical_trajectory_path(agent, "case-1") == jsonl
        ordinary = parent / "case-1.json"
        ordinary.write_text('{}')
        assert seam._canonical_trajectory_path(agent, "case-1") == (jsonl if agent == "codex" else ordinary)


def test_modern_adapter_source_stamp_must_match_approved_canonical_bytes(attribution):
    attribution.stamp["sha256"] = "0" * 64
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available and not attribution.calls


def test_modern_adapter_must_not_report_a_different_canonical_path(attribution):
    attribution.stamp["path"] = str(attribution.root / "another.json")
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available and not attribution.calls


def test_reference_mutation_during_diagnosis_refuses_old_gold_result(attribution):
    original = attribution.modules["detect"].detect
    def mutating_detect(*args, **kwargs):
        attribution.config.requirements_path("case-1").write_text('{"gold_patch":"changed"}')
        return original(*args, **kwargs)
    attribution.modules["detect"].detect = mutating_detect
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available and "reference task changed" in result.reason


@pytest.mark.parametrize("input_kind,operation", [("patch", "replace"), ("patch", "remove"),
    ("patch", "create"), ("outcome", "replace"), ("outcome", "remove"), ("outcome", "create")])
def test_patch_and_outcome_mutation_refuses_mixed_reference_evidence(attribution, input_kind, operation):
    path = (attribution.config.patch_path("claude_code", "case-1") if input_kind == "patch"
            else attribution.config.eval_path("claude_code"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if operation != "create":
        path.write_text("original synthetic evidence")
    original = attribution.modules["detect"].detect
    def mutating_detect(*args, **kwargs):
        if operation == "remove":
            path.unlink()
        else:
            path.write_text("changed synthetic evidence")
        return original(*args, **kwargs)
    attribution.modules["detect"].detect = mutating_detect
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available
    assert ("agent patch changed" if input_kind == "patch" else "benchmark outcome changed") in result.reason


def test_objective_capability_opportunities_do_not_claim_a_model_judge_was_used(attribution):
    original = attribution.modules["detect"].detect
    def objective_detect(*args, **kwargs):
        detection = original(*args, **kwargs)
        # Real attribution always opens requirement_understanding and may open planning;
        # these describe task opportunities, not whether a model judge ran.
        detection["opportunities"] = {"requirement_understanding": True, "task_planning": True}
        return detection
    attribution.modules["detect"].detect = objective_detect
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert result.available and result.used_judge is False
    assert attribution.calls == [("detect", False), ("blame", None)]


@pytest.mark.parametrize("relative", ["overrides/resolved_overrides.json",
    "pertest/claude_code/case-1.json", "pertest/claude_code/case-1.log.json"])
def test_optional_evaluation_inputs_cannot_change_mid_diagnosis(attribution, relative):
    path = attribution.config.PROJECT_DATA / relative
    original = attribution.modules["detect"].detect
    def mutating_detect(*args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new synthetic evaluation evidence")
        return original(*args, **kwargs)
    attribution.modules["detect"].detect = mutating_detect
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available and "optional evaluation evidence changed" in result.reason


def test_unknown_api_drift_is_safe_and_does_not_echo_private_exception(attribution):
    def broken_config(name):
        raise ValueError("synthetic-private-endpoint-and-key")
    attribution.config.requirements_path = broken_config
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=attribution.path)
    assert not result.available and "ValueError" in result.reason
    assert "synthetic-private" not in result.reason


def test_bundled_engine_ignores_malicious_external_paths(tmp_path):
    dependency = tmp_path / "dependency"
    package = dependency / "awe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    marker = tmp_path / "executed-external-code"
    (package / "config.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('executed')\n"
        + 'raise ValueError("synthetic-private-import-value")\n')
    source_root = Path(__file__).resolve().parents[1]
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); "
        "from agent_trajviz.attribution import available; "
        "from agent_trajviz.insight.attribution import diagnose; "
        "result = diagnose(agent=None, instance_id=None); "
        "print(json.dumps({'engine_available': available()[0], 'available': result.available, 'reason': result.reason}))"
    )
    output = subprocess.check_output(
        [sys.executable, "-I", "-c", script, str(source_root)], cwd=tmp_path,
        env={"PATH": os.defpath, "HOME": str(tmp_path), "AWE_PATH": str(dependency),
             "TRAJVIZ_REFERENCE_ROOT": str(tmp_path / "reference")},
        text=True, timeout=15,
    )
    result = json.loads(output)
    assert result["engine_available"] is True and result["available"] is False
    assert not marker.exists()
    assert "synthetic-private" not in result["reason"] and str(tmp_path) not in result["reason"]


@pytest.mark.parametrize("instance_id", ["astropy__astropy-13033", "django__django-11477"])
def test_real_modern_dependency_uses_only_objective_evidence(instance_id, monkeypatch):
    assert seam.ATTRIBUTION_AVAILABLE, "Bundled attribution engine must be installed"
    from agent_trajviz.awe._engine.dossier import case_record
    if seam._supports_layer_controls(case_record):
        pytest.skip("legacy attribution generation uses verified per-call model controls")
    root = Path(__file__).resolve().parent / "fixtures" / "reference"
    source = root / "data" / "trajectory" / "claude_code" / (instance_id + ".json")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = seam.diagnose(agent="claude_code", instance_id=instance_id,
                           reference_root=root, source_path=source)
    assert result.available is True
    assert result.used_judge is False and result.arbiter is None
    assert result.primary is not None and result.faults
    assert any("without cached LLM verdicts" in note for note in result.notes)


def test_native_claude_session_receives_bundled_public_attribution(tmp_path):
    assert seam.ATTRIBUTION_AVAILABLE, "Bundled attribution engine must be installed"
    import agent_trajviz as tv
    from agent_trajviz import attribution as public_attribution
    from agent_trajviz.awe._engine.adapters import load_normalized

    fixtures = Path(__file__).resolve().parent / "fixtures"
    root = tmp_path / "reference"
    shutil.copytree(fixtures / "reference", root)
    instance_id = "django__django-11477"
    canonical = root / "data" / "trajectory" / "claude_code"
    (canonical / (instance_id + ".json")).unlink()
    native = canonical / (instance_id + ".jsonl")
    shutil.copyfile(fixtures / "native_claude_code" / (instance_id + ".jsonl"), native)
    run = tv.load(native)
    assert len(run.steps) > 0
    try:
        seam.configure(root)
        normalized = load_normalized("claude_code", instance_id)
        assert normalized is not None and normalized.steps
        result = public_attribution.diagnose(run, agent="claude_code", instance_id=instance_id,
                                            reference_root=root)
    finally:
        seam.configure(seam._DEFAULT_ROOT)
    assert result["available"] is True, result["reason"]
    assert result["dossier"]["event_ledger"] and result["faults"]



@pytest.fixture
def backend_case(request):
    return request.getfixturevalue("attribution")

def _install_full_objective_helpers(backend_case):
    """A backend contract with full dossier fields but no cached model reads."""
    dossier = backend_case.modules["dossier"]
    original_gold = backend_case.modules["gold"].build_gold_reference
    def full_gold(name):
        gold = original_gold(name)
        gold.repo = "synthetic/repo"
        gold.test_patch_files = ["tests/test_a.py"]
        gold.fail_to_pass = ["tests/test_a.py::test_regression"]
        gold.benchmark_profile = {"name": "Synthetic benchmark"}
        return gold
    backend_case.modules["gold"].build_gold_reference = full_gold
    backend_case.modules["outcomes"].load_pertest = lambda agent, name: {"f2p_pass_fraction": 0.5}
    original_blame = backend_case.modules["rootcause"].build_blame
    def full_blame(*args, **kwargs):
        return {**original_blame(*args, **kwargs), "decision_trace": {"steps": [{"policy": "objective"}]}}
    backend_case.modules["rootcause"].build_blame = full_blame
    def audit(fault, **kwargs):
        assert kwargs["pertest"] == {"f2p_pass_fraction": 0.5}
        assert kwargs["fail_to_pass"] == ["tests/test_a.py::test_regression"]
        assert kwargs["test_files"] == ["tests/test_a.py"]
        return {"verdict": "supported", "mismatches": [], "additional": "not a public audit field"}
    dossier.audit_fault = audit
    dossier.select_remedies = lambda case, fault, weight: {
        "recommendation_support": "supported", "countermeasures": [{"id": "CM-SYNTHETIC"}],
        "countermeasure_decisions": [{"decision": "selected"}], "recommendation": ["Check a controlled trial"]}
    dossier.CAP_LABEL = {"code_editing": "Code Editing"}
    dossier.for_case = lambda agent, instance, profile: {
        "benchmark": profile["name"], "language": "Python",
        "execution_context": {"model": "declared-model", "capabilities": None}}
    dossier.fault_side_assess = lambda signals: {"side": "model", "certainty": "unknown"}
    dossier.thinking_excerpt = lambda trajectory: [{"step": 0, "kind": "says", "text": trajectory.steps[0].text}]
    dossier.event_ledger = lambda trajectory: {"available": True, "events": [], "missing_result_calls": 0}
    dossier.analyze_symptoms = lambda trajectory, patch: {"available": True, "symptoms": [], "verification": {"runs": []}}
    dossier.compact_symptoms = lambda value: dict(value)
    return dossier


def test_modern_objective_dossier_preserves_full_evidence_and_context(backend_case):
    _install_full_objective_helpers(backend_case)
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=backend_case.path)
    assert result.available and result.used_judge is False and result.arbiter is None
    dossier = result.dossier
    assert dossier["case_id"] == "claude_code__case-1" and dossier["repo"] == "synthetic/repo"
    assert dossier["attribution_trace"] == {"steps": [{"policy": "objective"}]}
    assert dossier["benchmark"] == "Synthetic benchmark" and dossier["language"] == "Python"
    assert dossier["execution_context"]["model"] == "declared-model"
    assert dossier["fault_side"] == {"side": "model", "certainty": "unknown"}
    assert dossier["event_ledger"]["available"] is True
    assert dossier["thinking_process"][0]["text"] == "Synthetic workflow step"
    assert dossier["symptom_analysis"]["patch_available"] is False
    assert dossier["reasoning_observability"] == "observed"
    assert dossier["faults"][0]["audit"] == {"verdict": "supported", "mismatches": []}
    assert dossier["faults"][0]["recommendation_support"] == "supported"
    assert dossier["faults"][0]["label"] == "Code Editing → incorrect_patch"
    assert backend_case.calls == [("detect", False), ("blame", None)]
    assert json.loads(json.dumps(result.dossier)) == result.dossier


def test_shape_keeps_unknown_backend_dossier_fields_without_sharing_mutable_data():
    record = {"agent": "custom", "instance_id": "task", "faults": [],
              "new_backend_metadata": {"preserve": ["original"]}}
    result = seam._shape(record, {}, [], used_judge=False)
    assert result.dossier == record
    record["new_backend_metadata"]["preserve"].append("changed")
    assert result.dossier["new_backend_metadata"] == {"preserve": ["original"]}


def test_full_dossier_helper_errors_are_not_silently_dropped(backend_case):
    dossier = _install_full_objective_helpers(backend_case)
    def broken_ledger(trajectory):
        raise ValueError("synthetic-private-ledger-text")
    dossier.event_ledger = broken_ledger
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=backend_case.path)
    assert result.available is False and result.dossier is None
    assert "ValueError" in result.reason and "synthetic-private" not in result.reason


def test_execution_context_mutation_refuses_mixed_dossier_metadata(backend_case, monkeypatch):
    dossier = _install_full_objective_helpers(backend_case)
    path = backend_case.root / "execution-context.json"
    path.write_text('{"schema_version":1,"agents":{}}')
    monkeypatch.setenv("AWE_EXECUTION_CONTEXT", str(path))
    original = dossier.for_case
    def mutating_context(*args):
        path.write_text('{"schema_version":1,"agents":{"changed":{}}}')
        return original(*args)
    dossier.for_case = mutating_context
    result = seam.diagnose(agent="claude_code", instance_id="case-1", source_path=backend_case.path)
    assert result.available is False and result.dossier is None
    assert "optional evaluation evidence changed" in result.reason


def test_real_modern_objective_dossier_retains_current_backend_fields():
    assert seam.ATTRIBUTION_AVAILABLE, "Bundled attribution engine must be installed"
    from agent_trajviz.awe._engine.dossier import case_record
    if seam._supports_layer_controls(case_record):
        pytest.skip("full additive fields belong to the modern dossier API")
    root = Path(__file__).resolve().parent / "fixtures" / "reference"
    source = root / "data" / "trajectory" / "claude_code" / "astropy__astropy-13033.json"
    result = seam.diagnose(agent="claude_code", instance_id="astropy__astropy-13033",
                           reference_root=root, source_path=source)
    assert result.available and result.dossier is not None
    assert {"attribution_trace", "fault_side", "benchmark", "language", "execution_context", "repo",
            "thinking_process", "event_ledger", "symptom_analysis"} <= result.dossier.keys()
    assert {"audit", "countermeasures", "countermeasure_decisions", "recommendation_support",
            "recommendation", "label"} <= result.dossier["faults"][0].keys()
    assert result.dossier["arbiter"] is None and result.used_judge is False
