"""Saved AWE results retain evidence and reject mixed or unsafe run artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import zipfile
import pytest

from agent_trajviz.awe import saved


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@pytest.fixture
def artifacts():
    cases = [{"case_id": "claude_code__case-1", "agent": "claude_code", "instance_id": "case-1",
              "blame_status": "primary", "unknown_future_field": {"raw": "原始 evidence"},
              "faults": [{"capability": "code_editing", "error_type": "incorrect_patch",
                          "blame_weight": 1, "is_primary": True,
                          "code_evidence": {"reference_fix": {"a.py": ["+ fixed"]}},
                          "audit": {"verdict": "supported", "mismatches": []}}]}]
    scope = {"agents": ["claude_code"], "instance_ids": ["case-1", "case-2"],
             "diagnosed_failed_cases": 1, "benchmark": "Synthetic benchmark"}
    graphs = {"schema_version": 1, "source_cases_sha256": _digest(cases), "scope": scope,
              "graphs": [{"case_id": "claude_code__case-1", "agent": "claude_code", "instance_id": "case-1",
                          "nodes": [{"id": "input", "title": "<script>not executed</script>"},
                                    {"id": "conclusion", "title": "Diagnosis"}],
                          "edges": [{"from": "input", "to": "conclusion", "relation": "documents", "causal": False}],
                          "action_graph": {"nodes": [], "edges": []}}]}
    recommendations = {"schema_version": 1, "source_cases_sha256": _digest(cases), "scope": scope,
                       "candidates": [{"id": "CM-1", "case_ids": ["claude_code__case-1"],
                                       "deployment_status": "prerequisites_unverified"}],
                       "shortlist": {"overall": ["CM-1"], "by_agent": {"claude_code": ["CM-1"]}}}
    symptoms = {"schema_version": 1, "summary": {"scoring": "not_scored"},
                "cases": [{"case_id": "claude_code__case-1", "agent": "claude_code", "instance_id": "case-1",
                           "resolved": False, "analysis": {"symptoms": []}},
                          {"case_id": "claude_code__case-2", "agent": "claude_code", "instance_id": "case-2",
                           "resolved": True, "analysis": {"symptoms": [{"kind": "zero_tests_reported"}]}}]}
    raw = {"faults.json": _bytes(cases), "attribution_graphs.json": _bytes(graphs),
           "recommendations.json": _bytes(recommendations), "symptoms.json": _bytes(symptoms),
           "INSIGHTS.md": b"# Observed results\n", "reports/claude_code.md": b"# Scorecard\n",
           "ATTRIBUTION.html": b"<script>must never execute</script>", "attachment.bin": b"\xff\x00\xfe"}
    manifest = {"run_id": "run-1", "agents": scope["agents"], "instance_ids": scope["instance_ids"],
                "n_failed_cases": 1, "n_faults": 1, "artifacts": list(raw),
                "attribution_graphs": {"source_cases_sha256": _digest(cases), "scope": scope},
                "recommendation_plan": {"source_cases_sha256": _digest(cases), "scope": scope},
                "unrecognized_provenance": {"revision": "source-revision"}}
    return {**raw, "manifest.json": _bytes(manifest)}


def _write(directory, artifacts):
    for name, body in artifacts.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return directory


def test_directory_import_preserves_full_artifacts_bytes_and_provenance(tmp_path, artifacts):
    directory = _write(tmp_path / "run", artifacts)
    run = saved.load_run(directory)
    assert len(run.cases) == 1
    assert run.cases[0]["unknown_future_field"] == {"raw": "原始 evidence"}
    assert run.graphs["graphs"][0]["nodes"][0]["title"].startswith("<script>")
    assert run.symptoms["cases"][1]["resolved"] is True
    assert run.recommendations["candidates"][0]["deployment_status"] == "prerequisites_unverified"
    assert run.manifest["unrecognized_provenance"]["revision"] == "source-revision"
    assert run.raw_artifacts == artifacts
    assert run.fingerprints["faults.json"]["sha256"] == hashlib.sha256(artifacts["faults.json"]).hexdigest()
    assert run.artifacts["attachment.bin"]["encoding"] == "binary"
    assert run.warnings == ()


def test_selected_manifest_or_faults_loads_siblings(tmp_path, artifacts):
    directory = _write(tmp_path / "run", artifacts)
    assert saved.load_run(directory / "faults.json").raw_artifacts == artifacts
    assert saved.load_run(directory / "manifest.json").raw_artifacts == artifacts


def test_portable_json_and_zip_roundtrip_keep_original_bytes(tmp_path, artifacts):
    original = saved.load_run(_write(tmp_path / "run", artifacts))
    envelope = original.to_dict()
    assert envelope["imported"] is True
    envelope["additive_envelope_field"] = {"keep": True}
    assert saved.load_run(envelope).to_dict()["additive_envelope_field"] == {"keep": True}
    restored = saved.load_run(original.write(tmp_path / "portable.json"))
    assert restored.artifacts == original.artifacts and restored.raw_artifacts == artifacts
    from_zip = saved.load_run(restored.export(tmp_path / "portable.zip"))
    assert from_zip.raw_artifacts == artifacts


def test_wrapped_zip_import_and_current_output_root_exclude_caches(tmp_path, artifacts):
    archive = tmp_path / "run.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, body in artifacts.items():
            stream.writestr("run-1/" + name, body)
    assert saved.load_run(archive).raw_artifacts == artifacts
    current = _write(tmp_path / "data", {f"dossier/{name}": body for name, body in artifacts.items()
                                          if name != "manifest.json" and not name.startswith("reports/")})
    _write(current, {"reports/claude_code.md": artifacts["reports/claude_code.md"],
                     "cache/judge/key.json": b"not an artifact", "runs/old/faults.json": b"old run"})
    loaded = saved.load_run(current)
    assert "dossier/faults.json" in loaded.artifacts
    assert not any(name.startswith(("cache/", "runs/")) for name in loaded.artifacts)


@pytest.mark.parametrize("name", ["attribution_graphs.json", "symptoms.json", "recommendations.json"])
def test_standalone_artifacts_are_preserved_without_fabricating_cases(name, artifacts):
    document = json.loads(artifacts[name])
    result = saved.load_run(document)
    assert result.cases == []
    assert result.artifacts[name] == document
    assert any("No fault dossier" in warning for warning in result.warnings)
    if name != "symptoms.json":
        assert any("cannot be verified" in warning for warning in result.warnings)


def test_single_case_and_empty_dossier_are_supported(artifacts):
    case = json.loads(artifacts["faults.json"])[0]
    assert saved.load_run(case).cases == [case]
    assert saved.load_run([]).cases == []


@pytest.mark.parametrize("field", ["artifacts", "raw_artifacts", "fingerprints"])
def test_modified_envelope_fails_identity_validation(field, artifacts, tmp_path):
    document = saved.load_run(_write(tmp_path / "run", artifacts)).to_dict()
    if field == "artifacts":
        document[field]["faults.json"][0]["blame_status"] = "changed"
    elif field == "raw_artifacts":
        document[field]["faults.json"] = "bm90IHRoZSBzYW1l"
    else:
        document[field]["faults.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        saved.load_run(document)


def test_mutating_in_memory_evidence_cannot_export_as_original(tmp_path, artifacts):
    result = saved.load_run(_write(tmp_path / "run", artifacts))
    result.cases[0]["blame_status"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        result.to_dict()


@pytest.mark.parametrize("name", ["attribution_graphs.json", "recommendations.json"])
def test_stale_derived_artifacts_rejected(tmp_path, artifacts, name):
    document = json.loads(artifacts[name])
    document["source_cases_sha256"] = "0" * 64
    artifacts[name] = _bytes(document)
    with pytest.raises(ValueError, match="stale"):
        saved.load_run(_write(tmp_path / "run", artifacts))


@pytest.mark.parametrize("change,expected", [
    ("cycle", "cycle"), ("dangling", "dangling"), ("duplicate", "duplicate"), ("wrong_case", "case IDs"),
])
def test_invalid_graphs_do_not_reach_viewers(tmp_path, artifacts, change, expected):
    document = json.loads(artifacts["attribution_graphs.json"])
    graph = document["graphs"][0]
    if change == "cycle":
        graph["edges"].append({"from": "conclusion", "to": "input", "relation": "selects"})
    elif change == "dangling":
        graph["edges"][0]["to"] = "missing"
    elif change == "duplicate":
        graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))
    else:
        graph["case_id"] = "different"
    artifacts["attribution_graphs.json"] = _bytes(document)
    with pytest.raises(ValueError, match=expected):
        saved.load_run(_write(tmp_path / "run", artifacts))


def test_recommendation_scope_and_unknown_case_references_are_rejected(tmp_path, artifacts):
    document = json.loads(artifacts["recommendations.json"])
    document["candidates"][0]["case_ids"] = ["another-run__case-1"]
    artifacts["recommendations.json"] = _bytes(document)
    with pytest.raises(ValueError, match="outside"):
        saved.load_run(_write(tmp_path / "run", artifacts))


def test_all_outcome_symptoms_must_stay_in_input_scope(tmp_path, artifacts):
    document = json.loads(artifacts["symptoms.json"])
    document["cases"][1]["instance_id"] = "other-task"
    artifacts["symptoms.json"] = _bytes(document)
    with pytest.raises(ValueError, match="scope"):
        saved.load_run(_write(tmp_path / "run", artifacts))


def test_missing_manifest_files_and_ambiguous_runs_fail(tmp_path, artifacts):
    broken = dict(artifacts)
    broken.pop("INSIGHTS.md")
    with pytest.raises(ValueError, match="missing artifact"):
        saved.load_run(_write(tmp_path / "incomplete", broken))
    with pytest.raises(ValueError, match="Multiple faults.json"):
        saved.load_run(_write(tmp_path / "mixed", {"run-1/faults.json": artifacts["faults.json"],
                                                    "run-2/faults.json": artifacts["faults.json"]}))


@pytest.mark.parametrize("unsafe", ["../faults.json", "/faults.json", "C:/faults.json", "a\\faults.json", "a//faults.json"])
def test_unsafe_zip_paths_are_never_extracted(tmp_path, unsafe):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(unsafe, b"[]")
    with pytest.raises(ValueError, match="relative paths"):
        saved.load_run(path)


def test_symlink_and_duplicate_zip_entries_rejected(tmp_path):
    linked = tmp_path / "linked.zip"
    info = zipfile.ZipInfo("faults.json")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(linked, "w") as archive:
        archive.writestr(info, "other-file")
    with pytest.raises(ValueError, match="links"):
        saved.load_run(linked)
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive, pytest.warns(UserWarning):
        archive.writestr("faults.json", b"[]")
        archive.writestr("faults.json", b"[]")
    with pytest.raises(ValueError, match="duplicate"):
        saved.load_run(duplicate)


def test_directory_symlinks_and_artifact_size_limits(tmp_path, monkeypatch):
    directory = _write(tmp_path / "run", {"faults.json": b"[]"})
    (directory / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        saved.load_run(directory)
    (directory / "linked").unlink()
    monkeypatch.setattr(saved, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        saved.load_run(directory)


@pytest.mark.parametrize("body", [b"{broken", b"NaN", b"[NaN]", b'{"x":1,"x":2}', b"null"])
def test_invalid_nonfinite_duplicate_or_null_json_rejected(tmp_path, body):
    path = tmp_path / "faults.json"
    path.write_bytes(body)
    with pytest.raises(ValueError):
        saved.load_run(path)


def test_unknown_additive_graph_relations_are_kept(artifacts):
    graphs = json.loads(artifacts["attribution_graphs.json"])
    graphs["future"] = {"keep": "value"}
    graphs["graphs"][0]["edges"][0]["relation"] = "future_relation"
    graphs["schema_version"] = 2
    run = saved.load_run(graphs)
    assert run.graphs == graphs
    assert any("unfamiliar schema" in warning for warning in run.warnings)


def test_recursive_python_documents_are_rejected():
    recursive = {}
    recursive["cycle"] = recursive
    with pytest.raises(ValueError):
        saved.load_run(recursive)


def test_saved_import_works_without_awe_backend(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__
    def no_backend(name, *args, **kwargs):
        if name == "awe" or name.startswith("awe."):
            raise AssertionError("Saved import must not import AWE")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_backend)
    assert saved.load_run([]).cases == []


def test_validation_only_operation_outputs_are_importable_and_fingerprinted(tmp_path):
    raw = {"operation-result.json": _bytes({"passed": True, "checks": {"strict_provenance": True}}),
           "operation.log": b"Controls passed.\n"}
    manifest = {"schema_version": 1, "operation": "preflight", "success": True,
                "backend_source_sha256": "a" * 64, "api_key_stored": False,
                "artifact_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in raw.items()}}
    raw["operation-manifest.json"] = _bytes(manifest)
    directory = _write(tmp_path / "operation", raw)
    result = saved.load_run(directory / "operation-manifest.json")
    assert result.cases == [] and result.manifest == manifest
    assert result.raw_artifacts == raw
    (directory / "operation.log").write_text("Changed after the operation.")
    with pytest.raises(ValueError, match="fingerprint"):
        saved.load_run(directory)


def test_operation_provenance_is_preserved_with_dossier_outputs(tmp_path):
    body = b"[]\n"
    raw = {"dossier/faults.json": body, "operation-result.json": b'{"available":false}',
           "operation.log": b"No failed cases.\n"}
    manifest = {"operation": "diagnose", "artifact_sha256": {
        name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}}
    raw["operation-manifest.json"] = _bytes(manifest)
    result = saved.load_run(_write(tmp_path / "operation", raw))
    assert result.manifest == manifest and result.raw_artifacts == raw


def test_standalone_rsi_feedback_keeps_privileged_runtime_boundary():
    value = {"schema_version": 1, "source_cases_sha256": "0" * 64,
             "runtime_input_allowed": False, "access_class": "optimizer_development_only", "cases": []}
    result = saved.load_run(value)
    assert result.artifacts["rsi_feedback.json"] == value and result.cases == []


@pytest.mark.parametrize("field,value", [("agent", "different-agent"), ("instance_id", "case-2")])
def test_graph_case_id_cannot_override_known_dossier_identity(tmp_path, artifacts, field, value):
    graphs = json.loads(artifacts["attribution_graphs.json"])
    graphs["graphs"][0][field] = value
    artifacts["attribution_graphs.json"] = _bytes(graphs)
    with pytest.raises(ValueError, match="conflicting case identities"):
        saved.load_run(_write(tmp_path / "run", artifacts))


def test_symptom_case_identity_must_match_the_fault_dossier(tmp_path, artifacts):
    symptoms = json.loads(artifacts["symptoms.json"])
    symptoms["cases"][0]["agent"] = "different-agent"
    artifacts["symptoms.json"] = _bytes(symptoms)
    with pytest.raises(ValueError, match="conflicting case identities"):
        saved.load_run(_write(tmp_path / "run", artifacts))


def test_solved_symptom_record_cannot_be_joined_to_a_failed_dossier_case(tmp_path, artifacts):
    symptoms = json.loads(artifacts["symptoms.json"])
    symptoms["cases"][0]["resolved"] = True
    artifacts["symptoms.json"] = _bytes(symptoms)
    with pytest.raises(ValueError, match="conflicting case outcomes"):
        saved.load_run(_write(tmp_path / "run", artifacts))


@pytest.mark.parametrize("value", [0, 1, "false", []])
def test_recorded_outcomes_are_strict_booleans_or_null(value):
    symptoms = {"schema_version": 1, "summary": {}, "cases": [
        {"case_id": "demo__task", "agent": "demo", "instance_id": "task", "resolved": value, "analysis": {}}]}
    with pytest.raises(ValueError, match="booleans or null"):
        saved.load_run(symptoms)


def test_known_graph_and_symptom_identities_agree_without_a_fault_dossier(tmp_path, artifacts):
    graphs = json.loads(artifacts["attribution_graphs.json"])
    graphs.pop("scope")
    symptoms = json.loads(artifacts["symptoms.json"])
    symptoms["cases"][0]["agent"] = "other-agent"
    raw = {"attribution_graphs.json": _bytes(graphs), "symptoms.json": _bytes(symptoms)}
    with pytest.raises(ValueError, match="conflicting case identities"):
        saved.load_run(_write(tmp_path / "run", raw))
