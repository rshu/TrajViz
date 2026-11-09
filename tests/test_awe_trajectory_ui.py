"""Loaded sessions inspect locally; reference attribution remains an explicit action."""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gradio as gr
import pytest
from gradio.state_holder import SessionState
from gradio_client import Client, handle_file

import agent_trajviz as tv
from agent_trajviz.insight import attribution, insight
from agent_trajviz.insight.ui import sidebar


_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "codearts_minimal.json"


def _fn(app, name):
    matches = [block for block in app.fns.values() if getattr(block.fn, "__name__", "") == name]
    if len(matches) > 1 and name == "on_inspect_trajectory":
        matches = [block for block in matches if any(event == "click" for _, event in block.targets)]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _invoke(app, client, name, values=None):
    fn = _fn(app, name)
    values = values or {}
    visible = [component for component in fn.inputs if not isinstance(component, (gr.State, gr.BrowserState))]
    return client.predict(*(values.get(component.label, component.value) for component in visible),
                          api_name="/" + fn.api_name)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(insight, "load_env_files", lambda: None)
    return insight.build_ui()


def test_inspection_uses_only_loaded_run_state_and_runs_after_load(app):
    inspect = _fn(app, "on_inspect_trajectory")
    assert len(inspect.inputs) == 1 and isinstance(inspect.inputs[0], gr.State)
    assert len(inspect.outputs) == 3 and isinstance(inspect.outputs[2], gr.State)
    assert inspect.outputs[0].elem_id == "awe-inspection-result"
    config = app.get_config_file()
    load_id = next(identity for identity, fn in app.fns.items()
                   if getattr(fn.fn, "__name__", "") == "_do_load")
    inspect_id = next(identity for identity, fn in app.fns.items() if fn is inspect)
    # The manual button and load follow-up can share one function object. Each
    # registration must have the same in-memory run input, never reference fields.
    related = [fn for fn in app.fns.values() if fn.fn is inspect.fn]
    assert all(fn.inputs == inspect.inputs and fn.outputs == inspect.outputs for fn in related)
    dependencies = config["dependencies"]
    follows = [event for event in dependencies if event.get("trigger_after") == load_id
               and event["outputs"] == [block._id for block in inspect.outputs]]
    assert len(follows) == 1
    assert inspect.concurrency_id == "awe-inspection"
    assert inspect_id in {event["id"] for event in dependencies}


def test_trajectory_inspection_does_not_run_reference_or_model_operations(app, monkeypatch):
    no_external = Mock(side_effect=AssertionError("Trajectory inspection must use the loaded recording only"))
    monkeypatch.setattr(attribution, "diagnose", no_external)
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", no_external)
    from agent_trajviz.awe import runtime
    monkeypatch.setattr(runtime, "run_operation", no_external)
    run = tv.load(_FIXTURE)
    html, status, result = _fn(app, "on_inspect_trajectory").fn(run)
    assert result["mode"] == "trajectory_only"
    assert html and status
    assert "score-dim-grid" not in html
    assert "Primary cause" not in html
    json.dumps(result, allow_nan=False)
    no_external.assert_not_called()


def test_inspection_empty_load_clears_previous_result(app):
    inspect = _fn(app, "on_inspect_trajectory")
    run = tv.load(_FIXTURE)
    first_html, _, first_result = inspect.fn(run)
    empty_html, _, empty_result = inspect.fn(None)
    assert first_result["mode"] == "trajectory_only"
    assert empty_result is None or not empty_result
    assert first_html != empty_html
    assert "Load" in empty_html or "load" in empty_html


def test_inspection_state_is_separate_for_each_browser_session(app):
    inspect = _fn(app, "on_inspect_trajectory")
    first, second = SessionState(app), SessionState(app)
    first[inspect.inputs[0]._id] = tv.load(_FIXTURE)
    second[inspect.inputs[0]._id] = None
    asyncio.run(app.process_api(inspect, [None], state=first, session_hash="inspection-first"))
    result_state = inspect.outputs[2]._id
    assert first[result_state]["mode"] == "trajectory_only"
    assert not second[result_state]
    asyncio.run(app.process_api(inspect, [None], state=second, session_hash="inspection-second"))
    assert not second[result_state]
    assert first[result_state]["mode"] == "trajectory_only"


def test_inspection_refresh_and_failed_load_replace_the_session_result(app):
    inspect = _fn(app, "on_inspect_trajectory")
    state = SessionState(app)
    state[inspect.inputs[0]._id] = tv.load(_FIXTURE)
    asyncio.run(app.process_api(inspect, [None], state=state, session_hash="inspection-refresh"))
    assert state[inspect.outputs[2]._id]["mode"] == "trajectory_only"
    state[inspect.inputs[0]._id] = None
    asyncio.run(app.process_api(inspect, [None], state=state, session_hash="inspection-refresh"))
    assert not state[inspect.outputs[2]._id]


@pytest.mark.parametrize("layer", ["inspect", "render"])
def test_inspection_failure_clears_the_result_without_exposing_exception_text(app, monkeypatch, layer):
    from agent_trajviz import awe
    from agent_trajviz.awe import inspection_render

    def fail(*args, **kwargs):
        raise ValueError("PRIVATE-INSPECTION-DETAIL")

    if layer == "inspect":
        monkeypatch.setattr(awe, "inspect_trajectory", fail)
    else:
        monkeypatch.setattr(inspection_render, "inspection_html", fail)
    html, status, result = _fn(app, "on_inspect_trajectory").fn(tv.load(_FIXTURE))
    assert result is None
    assert "unavailable" in status.lower()
    assert "PRIVATE-INSPECTION-DETAIL" not in html + status


def test_workflow_navigation_is_bounded_to_rendered_steps_without_mutating_result(app, monkeypatch):
    from agent_trajviz import awe
    from agent_trajviz.awe import inspection_render
    from agent_trajviz.insight.ui import trajectory_inspection

    original = {"mode": "trajectory_only", "workflow_indices": [0, 2, 5]}
    rendered = []
    monkeypatch.setattr(awe, "inspect_trajectory", lambda run: original)
    monkeypatch.setattr(trajectory_inspection, "MAX_STEPS", 2)

    def render(result, *, workflow_links):
        rendered.append((result, workflow_links))
        return "BOUNDED-INSPECTION"

    monkeypatch.setattr(inspection_render, "inspection_html", render)
    run = SimpleNamespace(steps=[{"index": 0}, {"index": 2}, {"index": 5}])
    html, _, result = _fn(app, "on_inspect_trajectory").fn(run)
    assert html == "BOUNDED-INSPECTION" and result is original
    assert rendered[0][0]["workflow_indices"] == [0, 2] and rendered[0][1] is True
    assert original["workflow_indices"] == [0, 2, 5]


def test_optional_reference_identity_refresh_never_diagnoses_on_load(app, monkeypatch):
    no_reference = Mock(side_effect=AssertionError("Reference attribution is an explicit optional action"))
    monkeypatch.setattr(attribution, "diagnose", no_reference)
    raw = {"_source_format": "codex", "_source_path": "/tmp/upload-cache/instance.jsonl"}
    result = _fn(app, "on_diagnose_autoload").fn(raw, "/no/reference/needed")
    assert result[2:] == ("codex", "instance")
    assert "score-dim-grid" not in result[0]
    no_reference.assert_not_called()


def test_served_native_session_inspects_without_reference_and_manual_attribution_still_works(app, monkeypatch, tmp_path):
    calls = []

    def diagnose(**kwargs):
        calls.append(kwargs)
        return attribution.AttributionResult(False, reason="EXPLICIT-REFERENCE-RESULT")

    monkeypatch.setattr(attribution, "diagnose", diagnose)
    no_model = Mock(side_effect=AssertionError("Inspection must not send model requests"))
    monkeypatch.setattr(sidebar, "analyze_loaded_trajectory", no_model)
    from agent_trajviz.awe import runtime
    monkeypatch.setattr(runtime, "run_operation", no_model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, inbrowser=False)
    try:
        first, second = Client(app.local_url, verbose=False), Client(app.local_url, verbose=False)
        _invoke(app, first, "_do_load", {"Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_FIXTURE)),
                                         "Format": "", "Captured session": None})
        # Gradio's API client does not fire frontend chained events, so invoke
        # the same follow-up handlers whose load dependencies are pinned above.
        _invoke(app, first, "on_diagnose_autoload", {"Reference data root": "/missing-reference"})
        inspected = _invoke(app, first, "on_inspect_trajectory")
        assert inspected and "Primary cause" not in str(inspected)
        assert "No gold reference" not in str(inspected)
        assert calls == []
        empty = _invoke(app, second, "on_inspect_trajectory")
        assert "load" in str(empty).lower()
        explicit = _invoke(app, first, "on_diagnose", {"Agent": "custom-agent", "Instance id": "custom-task",
                                                         "Reference data root": "/reference"})
        assert "EXPLICIT-REFERENCE-RESULT" in str(explicit)
        assert len(calls) == 1
        assert calls[0]["agent"] == "custom-agent" and calls[0]["instance_id"] == "custom-task"
        assert calls[0]["source_path"] and calls[0]["expected_sha"]
        _invoke(app, first, "on_inspect_trajectory")
        assert len(calls) == 1
        broken = tmp_path / "not-a-trajectory.json"
        broken.write_text('{"unrecognized": "input"}', encoding="utf-8")
        _invoke(app, first, "_do_load", {"Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(broken)),
                                        "Format": "", "Captured session": None})
        cleared = _invoke(app, first, "on_inspect_trajectory")
        assert "load" in str(cleared).lower() and str(cleared) != str(inspected)
        assert len(calls) == 1
        no_model.assert_not_called()
    finally:
        app.close()
