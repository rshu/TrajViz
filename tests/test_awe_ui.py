"""Saved evidence stays complete, readable and inert in the AWE interface."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import gradio as gr
import pytest

from agent_trajviz.awe import render
from agent_trajviz.insight.ui import awe_tab


@dataclass
class Run:
    cases: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    graphs: dict | None = None
    symptoms: dict | None = None
    recommendations: dict | None = None
    manifest: dict | None = None
    warnings: tuple = ()

    def to_dict(self):
        return {"type": "trajviz-awe-run", "schema_version": 1, "artifacts": self.artifacts,
                "source_sha256": "a" * 64}

    def export(self, destination):
        destination.write_text(json.dumps(self.to_dict()))


@pytest.fixture
def run():
    case = {"case_id": "agent__task", "agent": "agent", "instance_id": "task", "blame_status": "primary",
            "task": {"problem_statement": "TASK-EVIDENCE <script>task()</script>", "gold_files": ["src/a.py"]},
            "faults": [{"capability": "code_editing", "error_type": "incorrect_patch", "confidence": "high",
                        "blame_weight": 1.0, "is_primary": True,
                        "detector_evidence": {"metric": "DETECTOR-EVIDENCE <img src=x onerror=alert(1)>"},
                        "code_evidence": {"reference_fix": "PATCH-EVIDENCE", "what_the_agent_changed": "AGENT-PATCH"},
                        "audit": {"verdict": "mismatch", "mismatches": ["AUDIT-MISMATCH"]},
                        "countermeasures": [{"id": "CM-1", "runtime_contract": "RUNTIME-CONTRACT"}],
                        "countermeasure_decisions": {"reason": "SELECTION-REASON"},
                        "evidence_chain": {"strength": "deductive", "counterfactual": {"status": "not_tested"}}}],
            "attribution_trace": {"selection": {"status": "primary"}},
            "event_ledger": {"events": [{"origin": {"position": 3, "step_id": 7, "thread": "child", "call_id": "call-1"}}]},
            "fault_side": {"repair_locus": "scaffold"}, "execution_context": {"model": "model-a"},
            "thinking_process": [{"step": 7, "text": "RECORDED-EXCERPT"}], "custom_future_field": "PRESERVED-UNKNOWN"}
    graph = {"case_id": "agent__task", "agent": "agent", "instance_id": "task",
             "nodes": [{"id": "task' onclick='bad()", "number": "1.1", "phase": 1, "title": "<img src=x>",
                        "lines": ["Task observation"], "details": {"field": "NODE-DETAIL"},
                        "source": {"artifact": "faults.json", "json_pointer": "/0/task"}},
                       {"id": "result", "number": "6.1", "phase": 6, "title": "Conclusion", "lines": []}],
             "edges": [{"from": "task' onclick='bad()", "to": "result", "relation": "selects", "causal": False}],
             "action_graph": {"nodes": [{"id": "event:3", "position": 3, "title": "Original step 7", "reference": {"thread": "child"}}], "edges": []}}
    symptoms = {"summary": {"by_outcome": {"solved": {"cases": 1}, "failed": {"cases": 1}}},
                "cases": [{"case_id": "agent__task", "agent": "agent", "instance_id": "task", "resolved": False,
                           "analysis": {"symptoms": [{"kind": "zero_test_execution", "scoring": "not_scored"}]}},
                          {"case_id": "agent__solved", "agent": "agent", "instance_id": "solved", "resolved": True,
                           "analysis": {"symptoms": [{"kind": "verification_state_conflict", "scoring": "not_scored"}]}}]}
    plan = {"shortlist": {"overall": ["CM-1"]}, "candidates": [{"id": "CM-1", "recipe": {"name": "Proposed check"},
                                                               "case_ids": ["agent__task"], "prerequisites": ["PREREQUISITE"],
                                                               "measurement": {"expected_effect": "UNTESTED"}}],
            "limitations": ["LIMITATION"], "review_queues": {"uncertain_recipe_selection": []}}
    return Run(cases=[case], graphs={"graphs": [graph]}, symptoms=symptoms, recommendations=plan,
               artifacts={"faults.json": [case], "reports/agent.md": "95% CI recorded in upstream scorecard",
                          "INSIGHTS.md": "Cross-agent digest", "discovery/cohort_contrast.md": "Cohort report",
                          "ATTRIBUTION.html": "<script>untrusted()</script>",
                          "rsi_proposals.json": {"proposals": [{"compatibility": {"status": "requires_adapter"}}]},
                          "rsi_feedback.json": {"runtime_input_allowed": False}},
               manifest={"git_commit": "a" * 40}, warnings=("Missing optional discovery output",))


def test_complete_case_evidence_and_unknown_fields_remain_visible_and_escaped(run):
    output = render.case_html(run.cases[0])
    for marker in ("TASK-EVIDENCE", "DETECTOR-EVIDENCE", "PATCH-EVIDENCE", "AGENT-PATCH", "AUDIT-MISMATCH",
                   "RUNTIME-CONTRACT", "SELECTION-REASON", "RECORDED-EXCERPT", "PRESERVED-UNKNOWN", "call-1", "scaffold"):
        assert marker in output
    assert "<script>" not in output and "<img src=x" not in output
    assert "&lt;script&gt;task()&lt;/script&gt;" in output
    assert "Exact attribution policy trace" in output


def test_all_outcome_filters_include_symptom_only_cases_without_adding_blame(run):
    assert len(render.case_options(run)) == 2
    solved = render.case_options(run, result="solved")
    assert solved[0][1] == "agent__solved"
    case = render.selected_case(run, solved[0][1])
    assert "faults" not in case
    assert "does not assign fault blame" in render.case_html(case)
    assert render.case_options(run, agent="other") == []
    assert render.case_options(run, query="task")[0][1] == "agent__task"


def test_graph_navigation_preserves_numbers_relations_and_exact_source_pointers(run):
    graph = render.case_graph(run, run.cases[0])
    options = render.node_options(graph)
    selected = options[0][1]
    output = render.graph_html(graph, selected)
    assert "1.1" in output and "6.1" in output and "selects" in output
    assert "do not establish causal effects" in output
    assert "<img src=x" not in output and "onclick='bad()'" not in output
    assert "href='#awe-node-" in output
    assert "<svg " in output and "marker-end='url(#awe-arrow-" in output
    assert "aria-label='Numbered AWE provenance graph'" in output
    assert "<foreignObject" not in output and "<script" not in output
    details = render.node_html(graph, selected)
    assert "/0/task" in details and "NODE-DETAIL" in details
    action = render.case_graph(run, run.cases[0], "Action/result episodes")
    assert action["nodes"][0]["reference"]["thread"] == "child"


def test_graph_sources_resolve_exact_original_values_without_reading_server_files(run):
    node = run.graphs["graphs"][0]["nodes"][0]
    resolved = render.source_details(run, node["source"])
    assert resolved["available"] is True
    assert resolved["value"] == run.cases[0]["task"]
    output = render.node_html(run.graphs["graphs"][0], node["id"], run=run)
    assert "Original artifact value" in output and "TASK-EVIDENCE" in output
    assert "<script>task()" not in output
    missing = render.source_details(run, {"artifact": "/etc/passwd", "json_pointer": ""})
    assert missing["available"] is False
    absent = render.source_details(run, {"artifact": "faults.json", "json_pointer": "/999/task"})
    assert absent["available"] is False
    run.artifacts["custom.json"] = {"with/slash": {"with~tilde": "exact"}}
    assert render.source_details(run, {"artifact": "custom.json", "json_pointer": "/with~1slash/with~0tilde"})["value"] == "exact"


def test_graph_matching_uses_case_identity_when_agent_metadata_is_absent():
    run = Run(cases=[{"case_id": "second", "faults": []}], graphs={"graphs": [
        {"case_id": "first", "nodes": []}, {"case_id": "second", "nodes": [{"id": "correct"}]}]})
    assert render.case_graph(run, run.cases[0])["nodes"][0]["id"] == "correct"


def test_reports_are_inert_and_all_output_families_are_accessible(run):
    options = render.report_options(run)
    assert "reports/agent.md" in options and "discovery/cohort_contrast.md" in options
    output = render.report_html(run, "ATTRIBUTION.html")
    assert "<script>" not in output and "&lt;script&gt;untrusted()" in output
    assert "never executed" in output
    assert "zero_test_execution" in render.symptoms_html(run, run.cases[0])
    assert "verification_state_conflict" in render.symptoms_html(run)
    assert "PREREQUISITE" in render.recommendations_html(run, run.cases[0])
    assert "LIMITATION" in render.recommendations_html(run)
    assert "requires_adapter" in render.rsi_html(run)
    assert "runtime_input_allowed" in render.rsi_html(run)
    assert "a" * 40 in render.provenance_html(run)


def test_provenance_never_serializes_or_embeds_original_binary_payload(run):
    def forbidden():
        pytest.fail("Rendering provenance must not construct a full portable envelope")

    run.to_dict = forbidden
    run.raw_artifacts = {"large.png": b"binary-payload"}
    run.fingerprints = {"large.png": {"sha256": "b" * 64, "bytes": 14}}
    output = render.provenance_html(run)
    assert "b" * 64 in output
    assert "binary-payload" not in output and "raw_artifacts" not in output


def _app(monkeypatch, *, allow_server_paths=True, mock_capabilities=True):
    from agent_trajviz.awe import runtime

    if mock_capabilities:
        monkeypatch.setattr(runtime, "available_operations", lambda settings=None: [
            {"id": "preflight", "label": "Validate inputs", "available": False, "reason": "Inputs not configured"},
            {"id": "pipeline", "label": "Batch analysis", "available": True}])
    with gr.Blocks() as app:
        with gr.Tabs():
            refs = awe_tab.layout(allow_server_paths=allow_server_paths)
        awe_tab.bind(refs, app, allow_server_paths=allow_server_paths)
    return app, refs


def _callback(app, name):
    found = [entry.fn for entry in app.fns.values() if getattr(entry.fn, "__name__", "") == name]
    assert len(found) == 1
    return found[0]


def test_evaluation_settings_use_bundled_logic_without_checkout_selection(monkeypatch):
    app, refs = _app(monkeypatch)
    assert "backend_path" not in refs.runtime and "backend_path" not in awe_tab._PREF_FIELDS
    labels = [str(getattr(component, "label", "")).lower() for component in app.blocks.values()]
    assert not any("checkout" in label for label in labels)
    assert "Bundled support" in awe_tab._capabilities(None)


def test_import_failure_clears_run_case_graph_reports_and_export(monkeypatch, run):
    from agent_trajviz.awe import saved

    app, _ = _app(monkeypatch)
    callback = _callback(app, "on_awe_import")
    monkeypatch.setattr(saved, "load_run", lambda path: run)
    # bind captures the loader, so rebuild after replacing it.
    app, _ = _app(monkeypatch)
    callback = _callback(app, "on_awe_import")
    good = callback("saved.json", "")
    assert good[0] is run and "Imported 2 cases" in good[1]
    bad = callback(None, "")
    assert bad[0] is None and "Import failed" in bad[1]
    assert bad[6] == "" and bad[8] == "" and bad[13] == "" and bad[-1] is None
    assert len(good) == len(bad) == 17


def test_remote_launch_rejects_server_paths_and_every_backend_entry_point(monkeypatch):
    app, refs = _app(monkeypatch, allow_server_paths=False)
    assert refs.directory.visible is False
    imported = _callback(app, "on_awe_import")(None, "/private/server/data")
    assert imported[0] is None and "Server paths are disabled" in imported[1]
    values = [awe_tab._DEFAULTS[name] for name in awe_tab._PREF_FIELDS]
    with pytest.raises(gr.Error, match="local launch"):
        _callback(app, "on_awe_apply")(*values, "secret", False, False, None)
    with pytest.raises(gr.Error, match="local launch"):
        _callback(app, "on_awe_operation")(None, "pipeline", "", "", None, "{}", False)
    with pytest.raises(gr.Error, match="local launch"):
        _callback(app, "on_awe_import_result")("/private/server/data")


def test_settings_do_not_forward_a_retained_key_to_a_new_endpoint(monkeypatch):
    from agent_trajviz.awe import runtime

    monkeypatch.setattr(runtime, "EvaluationSettings", lambda **kwargs: SimpleNamespace(**kwargs))
    values = dict(awe_tab._DEFAULTS, reference_root="/reference", base_url="https://first.test/v1")
    first = awe_tab._settings(values, "session-secret")
    same = awe_tab._settings(values, "", first)
    assert same.api_key == "session-secret"
    moved = awe_tab._settings(dict(values, base_url="https://second.test/v1"), "", first)
    assert moved.api_key == ""
    cleared = awe_tab._settings(values, "", first, clear_key=True)
    assert cleared.api_key == ""


def test_apply_remembers_only_connection_preferences_and_never_runs_an_operation(monkeypatch):
    from agent_trajviz.awe import runtime

    monkeypatch.setattr(runtime, "EvaluationSettings", lambda **kwargs: SimpleNamespace(**kwargs))
    app, refs = _app(monkeypatch)
    monkeypatch.setattr(runtime, "run_operation", lambda *args, **kwargs: pytest.fail("Applying settings must be offline"))
    values = [awe_tab._DEFAULTS[name] for name in awe_tab._PREF_FIELDS]
    result = _callback(app, "on_awe_apply")(*values, "session-secret", False, True, None)
    assert result[0].api_key == "session-secret" and result[1] == ""
    assert "session-secret" not in json.dumps(result[4])
    assert "api_key" not in result[4]["values"]
    assert "No operation was started" in result[3]
    assert refs.runtime["api_key"].value == ""
    assert refs.runtime["allow_network"].value is False


def test_restored_browser_preferences_are_cleaned_of_unrecognized_secret_fields(monkeypatch):
    app, _ = _app(monkeypatch)
    saved = {"version": 1, "values": {**awe_tab._DEFAULTS, "judge_model": "saved-model", "api_key": "must-not-persist"}}
    result = _callback(app, "on_awe_restore")(saved)
    assert result[0] is None
    assert result[-1]["values"]["judge_model"] == "saved-model"
    assert "api_key" not in result[-1]["values"]
    assert "must-not-persist" not in json.dumps(result[-1])


def test_operation_caches_continue_within_session_and_external_consent_is_one_shot(monkeypatch):
    from agent_trajviz.awe import runtime

    app, _ = _app(monkeypatch)
    calls = []
    def operation(name, config, **kwargs):
        calls.append((name, config))
        return runtime.OperationResult(success=True, available=True, reason=None, operation=name,
                                       project_data="/private/session-working-cache", data={"completed": True})

    monkeypatch.setattr(runtime, "run_operation", operation)
    selected = runtime.EvaluationSettings(reference_root="/reference",
                                        api_key="session-secret", base_url="https://example.test/v1", judge_model="model")
    result = _callback(app, "on_awe_operation")(selected, "judge", "agent", "task", None, "{}", True)
    assert calls[0][1].allow_network is True
    active = result[3]
    assert active.project_data == "/private/session-working-cache"
    assert active.api_key == "session-secret" and active.allow_network is False
    assert result[4] is False
    assert "session-secret" not in json.dumps(result[0])
    _callback(app, "on_awe_operation")(active, "pipeline", "agent", "", None, "{}", False)
    assert calls[1][1].project_data == "/private/session-working-cache"
    assert calls[1][1].allow_network is False


@pytest.mark.parametrize("raw,expected", [(None, None), ("", None), ("   ", None), (" 3 ", 3), ("1", 1)])
def test_operation_limit_is_optional_and_positive(monkeypatch, raw, expected):
    from agent_trajviz.awe import runtime

    app, refs = _app(monkeypatch)
    assert isinstance(refs.runtime["limit"], gr.Textbox)
    assert refs.runtime["limit"].value == ""
    calls = []

    def operation(name, settings, **kwargs):
        calls.append(kwargs["limit"])
        return runtime.OperationResult(success=True, available=True, reason=None, operation=name)

    monkeypatch.setattr(runtime, "run_operation", operation)
    selected = runtime.EvaluationSettings()
    _callback(app, "on_awe_operation")(selected, "catalog" if expected is None else "pipeline", "", "", raw, "{}", False)
    assert calls == [expected]


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "unlimited"])
def test_invalid_operation_limit_is_rejected_before_runtime(monkeypatch, raw):
    from agent_trajviz.awe import runtime

    app, _ = _app(monkeypatch)
    monkeypatch.setattr(runtime, "run_operation", lambda *args, **kwargs: pytest.fail("Invalid limit reached runtime"))
    with pytest.raises(gr.Error, match="positive integer"):
        _callback(app, "on_awe_operation")(runtime.EvaluationSettings(), "pipeline", "", "", raw, "{}", False)


def test_harbor_import_updates_active_dataset_and_visible_settings(monkeypatch):
    from agent_trajviz.awe import runtime

    app, _ = _app(monkeypatch)
    selected = runtime.EvaluationSettings(reference_root="/old-reference")
    generated = runtime.OperationResult(success=True, available=True, reason=None, operation="ingest_harbor",
                                         project_data="/session/project-data",
                                         data={"canonical_root": "/session/canonical", "agents": ["harbor-agent"], "benchmark_name": "LoLBench"})
    monkeypatch.setattr(runtime, "run_operation", lambda *args, **kwargs: generated)
    result = _callback(app, "on_awe_operation")(selected, "ingest_harbor", "", "", None, "{}", False)
    active = result[3]
    assert active.reference_root == "/session/canonical"
    assert active.agents == ("harbor-agent",) and active.benchmark_name == "LoLBench"
    assert result[5:] == ("/session/project-data", "/session/canonical", "harbor-agent", "LoLBench")


@pytest.mark.parametrize("field,value", [("timeout", -1), ("max_tokens", 0), ("temperature", float("nan")),
                                         ("judge_min_agree", 1.1), ("judge_min_agree", float("nan")),
                                         ("base_url", "https://user:secret@example.test/v1"),
                                         ("base_url", "https://example.test/v1?api_key=secret")])
def test_invalid_request_preferences_are_rejected_before_browser_storage(field, value):
    values = dict(awe_tab._DEFAULTS)
    values[field] = value
    with pytest.raises(ValueError):
        awe_tab._settings(values, "")


def test_saved_import_export_and_session_isolation_over_http(tmp_path, monkeypatch):
    import socket
    import zipfile

    from gradio_client import Client, handle_file
    from gradio_client.exceptions import AppError

    from agent_trajviz.awe.saved import load_run

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    source = tmp_path / "faults.json"
    source.write_text(json.dumps([{"case_id": "agent__http-task", "agent": "agent", "instance_id": "http-task",
                                   "blame_status": "unattributed", "task": {"problem_statement": "HTTP-CASE-EVIDENCE"},
                                   "faults": [], "future_field": {"preserved": True}}]))
    app, _ = _app(monkeypatch, mock_capabilities=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app.launch(server_name="127.0.0.1", server_port=port, quiet=True, prevent_thread_lock=True, inbrowser=False)
    try:
        first, second = Client(app.local_url, verbose=False), Client(app.local_url, verbose=False)
        result = first.predict(handle_file(str(source)), "", api_name="/awe_import")
        assert "HTTP-CASE-EVIDENCE" in str(result)
        empty = second.predict(None, "Attribution", api_name="/awe_case")
        assert "HTTP-CASE-EVIDENCE" not in str(empty)
        chosen = first.predict("agent__http-task", "Attribution", api_name="/awe_case")
        assert "HTTP-CASE-EVIDENCE" in str(chosen)
        download = first.predict(api_name="/awe_export")
        exported = load_run(download)
        assert exported.cases[0]["future_field"] == {"preserved": True}
        assert zipfile.is_zipfile(download)
        invalid = tmp_path / "invalid.json"
        invalid.write_text("{broken")
        failure = first.predict(handle_file(str(invalid)), "", api_name="/awe_import")
        assert "Import failed" in str(failure)
        cleared = first.predict(None, "Attribution", api_name="/awe_case")
        assert "HTTP-CASE-EVIDENCE" not in str(cleared)
        # Exercise the actual settings/state transport too. A second client's
        # key must never replace the connection used by the first client.
        from agent_trajviz.awe import runtime

        # Exercise an actual bundled, reference-free operation with an untouched
        # blank field before using a spy to check per-session model settings.
        values = dict(awe_tab._DEFAULTS)
        first.predict(*(values[name] for name in awe_tab._PREF_FIELDS), "", False, False,
                      api_name="/awe_apply_settings")
        catalog = first.predict("catalog", "", "", "", "{}", False, api_name="/awe_run_operation")
        assert catalog[0]["success"] is True

        calls = []
        limits = []
        def operation(name, settings, **kwargs):
            calls.append((settings.judge_model, settings.api_key))
            limits.append(kwargs["limit"])
            return runtime.OperationResult(success=True, available=True, reason=None, operation=name,
                                           data={"model": settings.judge_model})

        monkeypatch.setattr(runtime, "run_operation", operation)
        for client, model, key in ((first, "first-model", "first-session-key"), (second, "second-model", "second-session-key")):
            values = dict(awe_tab._DEFAULTS, reference_root="/reference",
                          judge_model=model, base_url="https://example.test/v1")
            applied = client.predict(*(values[name] for name in awe_tab._PREF_FIELDS), key, False, True,
                                     api_name="/awe_apply_settings")
            assert key not in str(applied)
        first_result = first.predict("preflight", "", "", "", "{}", False, api_name="/awe_run_operation")
        second_result = second.predict("preflight", "", "", "", "{}", False, api_name="/awe_run_operation")
        assert "first-model" in str(first_result) and "second-model" in str(second_result)
        assert calls == [("first-model", "first-session-key"), ("second-model", "second-session-key")]
        assert limits == [None, None]
        first.predict("pipeline", "", "", " 3 ", "{}", False, api_name="/awe_run_operation")
        assert limits == [None, None, 3]
        with pytest.raises(AppError, match="positive integer"):
            first.predict("pipeline", "", "", "0", "{}", False, api_name="/awe_run_operation")
        assert limits == [None, None, 3]
        assert "first-session-key" not in str(first_result) and "second-session-key" not in str(second_result)
        import urllib.request
        with urllib.request.urlopen(app.local_url + "config", timeout=10) as response:  # noqa: S310 - loopback server only
            public_config = response.read().decode()
        assert "first-session-key" not in public_config and "second-session-key" not in public_config
    finally:
        app.close()


def test_native_attribution_panels_retain_full_context_and_do_not_invent_refutation():
    from agent_trajviz.insight.rendering import build_attribution_html

    case = {"task": {"problem_statement": "TASK-PANEL"}, "attribution_trace": {"selection": "POLICY-TRACE"},
            "fault_side": {"locus": "scaffold"}, "event_ledger": {"events": [{"call_id": "original-call"}]}}
    result = {"available": True, "blame_status": "unattributed", "scorecard": [], "dossier": case,
              "faults": [{"capability": "code_editing", "error_type": "candidate", "blame_weight": 0,
                          "confidence": "medium", "detector_evidence": {"metric": "DETECTOR-PANEL"},
                          "code_evidence": {"patch": "PATCH-PANEL"}, "audit": {"mismatches": ["AUDIT-PANEL"]}}]}
    output = build_attribution_html(result)
    assert all(value in output for value in ("TASK-PANEL", "POLICY-TRACE", "original-call", "DETECTOR-PANEL", "PATCH-PANEL", "AUDIT-PANEL", "medium"))
    assert "Refuted candidates" not in output and ">refuted<" not in output
