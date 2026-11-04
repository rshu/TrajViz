"""One AWE workspace defaults to trajectory inspection and retains optional attribution."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gradio as gr
import pytest
from gradio.state_holder import SessionState
from gradio_client import Client, handle_file

from agent_trajviz.awe import render
from agent_trajviz.awe.saved import load_run
from agent_trajviz.insight import attribution, insight
from agent_trajviz.insight.ui import sidebar


def _function(app, name, **keywords):
    matches = []
    for block_fn in app.fns.values():
        fn = block_fn.fn
        partial_keywords = {}
        while isinstance(fn, functools.partial):
            partial_keywords.update(fn.keywords)
            fn = fn.func
        if getattr(fn, "__name__", "") == name and all(partial_keywords.get(k) == v for k, v in keywords.items()):
            matches.append(block_fn)
    assert len(matches) == 1, (name, keywords, len(matches))
    return matches[0]


def _block(app, elem_id):
    matches = [block for block in app.blocks.values() if getattr(block, "elem_id", None) == elem_id]
    assert len(matches) == 1
    return matches[0]


def _descendants(block):
    yield block
    for child in getattr(block, "children", ()):
        yield from _descendants(child)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    for prefix in ("ANALYZE", "LABEL"):
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            monkeypatch.setenv(f"{prefix}_{suffix}", "")
    monkeypatch.setattr(insight, "load_env_files", lambda: None)
    return insight.build_ui()


def test_one_main_awe_tab_defaults_to_inspection_with_optional_reference_attribution(app):
    main = _block(app, "main-tabs")
    main_tabs = [block for block in main.children if isinstance(block, gr.Tab)]
    assert sum(tab.label == "AWE" for tab in main_tabs) == 1
    assert not any(tab.label == "Attribution" for tab in main_tabs)
    workspace = _block(app, "awe-workspace")
    sources = _block(app, "awe-workspace-tabs")
    native = _block(app, "awe-trajectory-workspace")
    saved = _block(app, "awe-saved-workspace")
    assert sources in set(_descendants(workspace))
    assert set(sources.children) == {native, saved}
    assert {tab.label for tab in sources.children} == {"Trajectory", "Saved runs & evaluation"}
    assert sources.selected == "inspection"
    native_labels = {getattr(block, "label", None) for block in _descendants(native)}
    native_values = {getattr(block, "value", None) for block in _descendants(native)
                     if isinstance(getattr(block, "value", None), str)}
    saved_labels = {getattr(block, "label", None) for block in _descendants(saved)}
    saved_values = {getattr(block, "value", None) for block in _descendants(saved)
                    if isinstance(getattr(block, "value", None), str)}
    assert {"Agent", "Instance id", "Reference data root"} <= native_labels
    assert "Diagnose failure" in native_values and "Import saved run" not in native_values
    assert "Inspect trajectory" in native_values
    optional = [block for block in _descendants(native)
                if isinstance(block, gr.Accordion) and block.label == "Reference-based attribution (optional)"]
    assert len(optional) == 1 and optional[0].open is False
    reference_ids = {block._id for block in _descendants(optional[0])}
    assert {block._id for block in _descendants(native)
            if getattr(block, "label", None) in {"Agent", "Instance id", "Reference data root"}} <= reference_ids
    assert {"Saved AWE run (.zip / .json)", "AWE case", "Graph node", "Saved report or artifact"} <= saved_labels
    assert {"Import saved run", "Prepare run download", "Run AWE operation"} <= saved_values
    saved_tab_labels = {block.label for block in _descendants(saved) if isinstance(block, gr.Tab)}
    assert {"Case evidence", "Provenance graph", "Symptoms", "Recommendations", "Reports & discovery", "Provenance", "RSI exchange"} <= saved_tab_labels


@pytest.mark.parametrize("mode,expected", [("inspection", [True, True, True, True]),
                                           ("saved", [False, False, False, False])])
def test_source_selection_only_changes_workspace_visibility(app, monkeypatch, mode, expected):
    calls = Mock(side_effect=AssertionError("Switching evidence sources must not run a model or operation"))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", calls)
    from agent_trajviz.awe import runtime
    monkeypatch.setattr(runtime, "run_operation", calls)
    fn = _function(app, "on_awe_workspace_select", mode=mode)
    result = fn.fn(SimpleNamespace(steps=[{"index": 0}]))
    assert result[0] == mode
    assert [update["visible"] for update in result[1:]] == expected
    assert isinstance(fn.outputs[0], gr.State) and fn.outputs[0].value == "inspection"
    assert len(fn.inputs) == 1 and isinstance(fn.inputs[0], gr.State)
    calls.assert_not_called()


@pytest.mark.parametrize("mode", ["inspection", "saved"])
def test_returning_to_awe_restores_the_selected_source(app, mode):
    result = _function(app, "on_main_tab_select", tab_label="AWE").fn(SimpleNamespace(steps=[{}]), mode)
    assert result[-1]["selected"] == mode
    expected = mode != "saved"
    assert [value["visible"] for value in result[:4]] == [expected] * 4
    other = _function(app, "on_main_tab_select", tab_label="Overview").fn(SimpleNamespace(steps=[{}]), mode)
    assert [value["visible"] for value in other[:4]] == [True] * 4
    assert "selected" not in other[-1]


def test_inspection_without_a_loaded_run_retains_upload_controls(app):
    result = _function(app, "on_main_tab_select", tab_label="AWE").fn(None, "inspection")
    assert [value["visible"] for value in result[:4]] == [True, True, False, False]
    assert result[-1]["selected"] == "inspection"


def test_source_selection_is_session_state_not_a_global_mode(app):
    select_saved = _function(app, "on_awe_workspace_select", mode="saved")
    show_awe = _function(app, "on_main_tab_select", tab_label="AWE")
    first, second = SessionState(app), SessionState(app)
    run_state, mode_state = select_saved.inputs[0], select_saved.outputs[0]
    for state in (first, second):
        state[run_state._id] = SimpleNamespace(steps=[{}])
    asyncio.run(app.process_api(select_saved, [None], state=first, session_hash="first"))
    assert first[mode_state._id] == "saved"
    assert second[mode_state._id] == "inspection"
    for state, expected in ((first, "saved"), (second, "inspection")):
        result = asyncio.run(app.process_api(show_awe, [None, None], state=state, session_hash=expected))
        assert result["data"][-1]["selected"] == expected


def test_embedded_remote_workspace_keeps_server_paths_and_execution_disabled(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(insight, "load_env_files", lambda: None)
    app = insight.build_ui(allow_server_paths=False)
    root_boxes = [block for block in app.blocks.values()
                  if isinstance(block, gr.Textbox) and block.label in {"Reference data root", "Saved run folder"}]
    assert len(root_boxes) == 2
    assert all(box.visible is False and box.interactive is False for box in root_boxes)
    result = _function(app, "on_awe_import").fn(None, "/private/server/run")
    assert result[0] is None and "Server paths are disabled" in result[1]
    with pytest.raises(gr.Error, match="local launch"):
        _function(app, "on_awe_operation").fn(None, "pipeline", "", "", None, "{}", False)


def _saved_bundle(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    cases = [
        {"case_id": "agent_a__failed-task", "agent": "agent_a", "instance_id": "failed-task", "resolved": False,
         "blame_status": "primary", "task": {"problem_statement": "MERGED-SAVED-CASE"},
         "faults": [{"capability": "code_editing", "error_type": "incorrect_patch", "blame_weight": 1, "is_primary": True}]},
        {"case_id": "agent_a__solved-task", "agent": "agent_a", "instance_id": "solved-task", "resolved": True,
         "faults": []},
        {"case_id": "agent_b__unknown-task", "agent": "agent_b", "instance_id": "unknown-task", "faults": []},
    ]
    # Solved cases belong in symptom records, never in failed-case dossiers.
    faults = [cases[0]]
    digest = hashlib.sha256(json.dumps(faults, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    graphs = {"source_cases_sha256": digest, "graphs": [{
        "case_id": cases[0]["case_id"], "agent": "agent_a", "instance_id": "failed-task",
        "nodes": [{"id": "task", "number": "1.1", "title": "Task observation", "lines": ["MERGED-GRAPH-NODE"],
                   "source": {"artifact": "faults.json", "json_pointer": "/0/task"}},
                  {"id": "result", "number": "6.1", "title": "Recorded conclusion", "lines": []}],
        "edges": [{"from": "task", "to": "result", "relation": "documents", "causal": False}],
        "action_graph": {"nodes": [{"id": "event:1", "position": 1, "title": "Recorded call"}], "edges": []},
    }]}
    for name, value in {"faults.json": faults, "attribution_graphs.json": graphs,
                        "symptoms.json": {"cases": cases[1:]}}.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "INSIGHTS.md").write_text("# MERGED-SAVED-REPORT\n", encoding="utf-8")
    bundle = tmp_path / "run.zip"
    load_run(root).export(bundle)
    return bundle


def _invoke(app, client, name, values=None):
    fn = _function(app, name)
    values = values or {}
    visible = [component for component in fn.inputs if not isinstance(component, (gr.State, gr.BrowserState))]
    return client.predict(*(values.get(component.label, component.value) for component in visible),
                          api_name="/" + fn.api_name)


@pytest.mark.parametrize("outcome,case_id", [("failed", "agent_a__failed-task"),
                                            ("solved", "agent_a__solved-task"),
                                            ("unknown", "agent_b__unknown-task")])
def test_merged_saved_filters_keep_all_recorded_outcome_types(app, tmp_path, outcome, case_id):
    run = load_run(_saved_bundle(tmp_path))
    handlers = [handler for handler in app.fns.values()
                if getattr(handler.fn, "__name__", "") == "on_awe_filter"]
    assert len(handlers) == 3  # Agent, outcome, and query controls each refresh the saved evidence.
    saved_ids = {block._id for block in _descendants(_block(app, "awe-saved-workspace"))}
    for handler in handlers:
        result = handler.fn(run, "All agents", outcome, "", "Attribution")
        assert result[0].value == case_id
        assert {component._id for component in handler.inputs + handler.outputs} <= saved_ids
        assert handler.inputs[0] is not _function(app, "on_diagnose").inputs[0]


def test_combined_served_app_keeps_native_diagnosis_and_saved_evidence_independent(app, tmp_path, monkeypatch):
    calls = []

    def diagnose(**kwargs):
        calls.append(kwargs)
        return attribution.AttributionResult(False, reason="NATIVE-ATTRIBUTION-CHECK")

    monkeypatch.setattr(attribution, "diagnose", diagnose)
    no_model = Mock(side_effect=AssertionError("Inspecting merged evidence must not call a model"))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", no_model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, inbrowser=False)
    try:
        first, second = Client(app.local_url, verbose=False), Client(app.local_url, verbose=False)
        native = Path(__file__).resolve().parent / "fixtures" / "codearts_minimal.json"
        _invoke(app, first, "_do_load", {
            "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(native)),
            "Format": "", "Captured session": None,
        })
        diagnosed = _invoke(app, first, "on_diagnose", {"Agent": "fixture_agent", "Instance id": "fixture_task"})
        assert "NATIVE-ATTRIBUTION-CHECK" in str(diagnosed)
        assert calls[-1]["source_path"] and calls[-1]["expected_sha"]
        identity = (calls[-1]["source_path"], calls[-1]["expected_sha"])
        imported = first.predict(handle_file(str(_saved_bundle(tmp_path))), "", api_name="/awe_import")
        assert "Imported 3 cases" in str(imported) and "MERGED-SAVED-CASE" in str(imported)
        case = first.predict("agent_a__failed-task", "Attribution", api_name="/awe_case")
        assert "MERGED-SAVED-CASE" in str(case) and "MERGED-GRAPH-NODE" in str(case)
        node = first.predict("agent_a__failed-task", "Attribution", "task", api_name="/awe_node")
        assert "Original artifact value" in str(node) and "MERGED-SAVED-CASE" in str(node)
        report = first.predict("INSIGHTS.md", api_name="/awe_report")
        assert "MERGED-SAVED-REPORT" in report
        exported = first.predict(api_name="/awe_export")
        assert len(render.all_cases(load_run(exported))) == 3
        empty = second.predict(None, "Attribution", api_name="/awe_case")
        assert "MERGED-SAVED-CASE" not in str(empty)
        _invoke(app, first, "on_diagnose", {"Agent": "fixture_agent", "Instance id": "fixture_task"})
        assert (calls[-1]["source_path"], calls[-1]["expected_sha"]) == identity
        # A new native load also leaves the independently imported saved run intact.
        _invoke(app, first, "_do_load", {
            "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(native)),
            "Format": "", "Captured session": None,
        })
        preserved = first.predict("agent_a__failed-task", "Attribution", api_name="/awe_case")
        assert "MERGED-SAVED-CASE" in str(preserved)
        no_model.assert_not_called()
    finally:
        app.close()
