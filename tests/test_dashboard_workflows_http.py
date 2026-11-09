"""Exercise composed dashboard workflows through real HTTP and public fixtures."""

from __future__ import annotations

import json
import os
import shutil
import socket
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from gradio_client import Client, handle_file

import agent_trajviz as tv

_FIXTURES = Path(os.environ.get("TRAJVIZ_HTTP_FIXTURE_ROOT", str(Path(__file__).parent / "fixtures")))
_TRAJECTORY = _FIXTURES / "native_claude_code" / "astropy__astropy-13033.jsonl"


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _function(app, name):
    matches = [fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name]
    assert matches, f"No endpoint for {name}"
    return matches[0]


def _key(component, index):
    return (
        getattr(component, "elem_id", None)
        or getattr(component, "label", None)
        or f"{type(component).__name__}_{index}"
    )


def _call(app, client, name, values=None):
    fn = _function(app, name)
    values = values or {}
    inputs = [component for component in fn.inputs if not component.skip_api]
    args = [
        next(
            (values[key] for key in [_key(component, index), getattr(component, "label", None)] if key in values),
            component.value,
        )
        for index, component in enumerate(inputs)
    ]
    result = client.predict(*args, api_name="/" + fn.api_name)
    outputs = [component for component in fn.outputs if not component.skip_api]
    unpacked = [] if not outputs else ([result] if len(outputs) == 1 else list(result))
    assert len(outputs) == len(unpacked)
    mapped = {}
    for index, (component, value) in enumerate(zip(outputs, unpacked, strict=True)):
        mapped[_key(component, index)] = value
        if getattr(component, "label", None):
            mapped[component.label] = value
    return mapped


def _event_outputs(app, client, name, data):
    """The public SSE API retains layout updates hidden by gradio_client."""
    fn = _function(app, name)
    endpoint = app.local_url.rstrip("/") + "/gradio_api/call/" + fn.api_name
    payload = json.dumps({"data": data, "session_hash": client.session_hash}).encode()
    request = urllib.request.Request(endpoint, payload, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - loopback only
        assert json.load(response)["event_id"]
    # Gradio6.14 keys call streams by session; newer versions also resolve
    # event IDs. Keep this browser session so AWE preservation is exercised.
    with urllib.request.urlopen(endpoint + "/" + client.session_hash, timeout=30) as response:  # noqa: S310 - loopback only
        lines = response.read().decode().splitlines()
    assert "event: complete" in lines
    return json.loads(next(line.removeprefix("data: ") for line in lines if line.startswith("data: ")))


def _section_updates(app, client, selected):
    return _event_outputs(app, client, "show_overview_section", [selected])


def _figure(value):
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, dict) and "plot" in value:
        return json.loads(value["plot"])
    return value


def _loaded(app, client):
    return _call(
        app,
        client,
        "_do_load",
        {
            "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_TRAJECTORY)),
            "Format": "",
            "Captured session": None,
        },
    )


@pytest.fixture(scope="module")
def dashboard():
    from agent_trajviz.insight import insight

    expected = os.environ.get("TRAJVIZ_EXPECT_PACKAGE_PATH")
    if expected:
        assert Path(tv.__file__).resolve().is_relative_to(Path(expected).resolve())
    with (
        patch.dict(
            os.environ,
            {"GRADIO_ANALYTICS_ENABLED": "False", "ANALYZE_API_KEY": "", "ANALYZE_BASE_URL": "", "ANALYZE_MODEL": ""},
        ),
        patch.object(insight, "load_env_files"),
    ):
        app = insight.build_ui()
        app.launch(server_name="127.0.0.1", server_port=_port(), prevent_thread_lock=True, quiet=True, inbrowser=False)
        try:
            yield app
        finally:
            app.close()


@pytest.fixture
def labels(tmp_path):
    run = tv.load(_TRAJECTORY)
    records = []
    for step in run.steps:
        phase, action = ("understand", "code_reading") if step["index"] % 2 else ("implement", "code_editing")
        records.append(
            {
                "index": step["index"],
                "raw_index": step.get("raw_index", step["index"]),
                "role": step["role"],
                "phase": phase,
                "action": action,
                "duration_s": step.get("duration") or 0,
            }
        )
    path = tmp_path / "walkthrough_labeled.json"
    path.write_text(json.dumps({"taxonomy_version": "v1", "steps": records}), encoding="utf-8")
    return path


def test_loaded_dashboard_populates_charts_raw_data_and_every_overview_section(dashboard):
    from agent_trajviz.insight.ui.overview_tab import OVERVIEW_SECTION_NAMES

    client = Client(dashboard.local_url, verbose=False)
    result = _loaded(dashboard, client)
    assert "Error loading" not in str(result)
    assert json.loads(result["Full trajectory JSON"])
    for chart in [
        "Token Usage",
        "Step Duration",
        "Tool Call Frequency",
        "Tool Call Duration",
        "Tool Outcome Timeline",
        "File Interaction Timeline",
        "Context Window Pressure",
    ]:
        figure = _figure(result[chart])
        assert isinstance(figure, dict) and figure.get("data"), chart
    for section in OVERVIEW_SECTION_NAMES:
        shown = _section_updates(dashboard, client, section)
        assert [value["visible"] for value in shown] == [name == section for name in OVERVIEW_SECTION_NAMES]
    assert "wf-card" in str(result) and "wf-toc" in str(result)


def test_workflow_search_and_context_overrides_use_loaded_session(dashboard):
    client = Client(dashboard.local_url, verbose=False)
    loaded = _loaded(dashboard, client)
    original = _call(
        dashboard,
        client,
        "do_filter_workflow",
        {
            "wf-filter-hidden": "Assistant,User,All",
            "Search": "",
            "HTML_2": loaded["wf-toc-container"],
        },
    )
    absent = _call(
        dashboard,
        client,
        "do_filter_workflow",
        {
            "wf-filter-hidden": "Assistant,User,All",
            "Search": "__missing_walkthrough_marker__",
            "HTML_2": loaded["wf-toc-container"],
        },
    )
    assert len(str(absent)) < len(str(original))
    assert "__missing_walkthrough_marker__" not in str(absent)
    for override in [None, 128000]:
        result = _call(
            dashboard,
            client,
            "_rebuild_utilization",
            {
                "Agent": "__all__",
                "Window limit (tokens)": override,
                "Window snapshot": "current",
            },
        )
        assert _figure(result["Context Window Pressure"])["data"]
        assert "Error" not in str(result)
        if override:
            assert "128,000" in str(result) or "128000" in str(result)


def test_labels_pairwise_and_group_comparison_over_http(dashboard, labels):
    client = Client(dashboard.local_url, verbose=False)
    _loaded(dashboard, client)
    rendered = _call(dashboard, client, "do_load_labels", {"Labels (optional)": handle_file(str(labels))})
    assert "Labels loaded" in str(rendered)
    for chart in [
        "Phase Count Distribution",
        "Action Count Distribution",
        "Phase Duration Distribution",
        "Action Duration Distribution",
        "Step Timeline",
    ]:
        assert _figure(rendered[chart])["data"], chart
    compared = _call(
        dashboard,
        client,
        "on_run_comparison",
        {
            "Reference Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_TRAJECTORY)),
            "Reference Labels (optional, *_labeled.json)": handle_file(str(labels)),
            "Labels (optional)": handle_file(str(labels)),
            "Reference captured session": None,
        },
    )
    assert "Comparison complete" in str(compared)
    assert "Label comparison unavailable" not in str(compared)
    assert _figure(compared["Step Count by Phase — Reference vs Compared"])["data"]
    assert _figure(compared["Duration by Phase — Reference vs Compared"])["data"]
    group = _call(
        dashboard,
        client,
        "on_run_group_scorecard",
        {
            "Comparison runs (.json / .jsonl / .zstd / .zip) — select one or more": [handle_file(str(_TRAJECTORY))],
        },
    )
    assert "2 of 2 run(s) loaded" in str(group)
    assert "<table" in str(group)
    assert _figure(group["Agent timeline (by run)"])["data"]
    cleared = _call(dashboard, client, "do_load_labels", {"Labels (optional)": None})
    assert "Labels loaded" not in str(cleared)
    assert not _figure(cleared["Phase Count Distribution"])["data"]


def test_failed_native_upload_clears_charts_comparison_and_export(dashboard, tmp_path):
    client = Client(dashboard.local_url, verbose=False)
    _loaded(dashboard, client)
    saved = tmp_path / "saved" / "faults.json"
    saved.parent.mkdir()
    saved.write_text(
        json.dumps(
            [
                {
                    "case_id": "agent__independent",
                    "agent": "agent",
                    "instance_id": "independent",
                    "task": {"problem_statement": "INDEPENDENT-AWE-EVIDENCE"},
                    "faults": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    imported = _call(dashboard, client, "on_awe_import", {"Saved AWE run (.zip / .json)": handle_file(str(saved))})
    assert "INDEPENDENT-AWE-EVIDENCE" in str(imported)
    invalid = tmp_path / "malformed-session.json"
    invalid.write_text("{broken", encoding="utf-8")
    failed = _call(
        dashboard,
        client,
        "_do_load",
        {
            "Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(invalid)),
            "Format": "",
            "Captured session": None,
        },
    )
    assert "could not" in str(failed).lower() or "error loading" in str(failed).lower()
    assert not _figure(failed["Token Usage"])["data"]
    assert not _figure(failed["Context Window Pressure"])["data"]
    compared = _call(
        dashboard,
        client,
        "on_run_comparison",
        {
            "Reference Trajectory (.json / .jsonl / .zstd / .zip)": handle_file(str(_TRAJECTORY)),
        },
    )
    assert "Load a trajectory in the Overview tab first" in str(compared)
    export = _call(dashboard, client, "prepare_html_export")["Export HTML"]
    assert export["interactive"] is False and export["value"] is None
    # Inspect the actual layout update too: gradio_client hides Tabs outputs.
    import requests

    with invalid.open("rb") as stream:
        uploaded = requests.post(
            dashboard.local_url.rstrip("/") + "/gradio_api/upload",
            files={"files": (invalid.name, stream, "application/json")},
            timeout=30,
        )
    uploaded.raise_for_status()
    file_value = {"path": uploaded.json()[0], "orig_name": invalid.name, "meta": {"_type": "gradio.FileData"}}
    updates = _event_outputs(dashboard, client, "_do_load", [file_value, False, "", None])
    fn = _function(dashboard, "_do_load")
    tabs_index = next(index for index, component in enumerate(fn.outputs) if component.get_block_name() == "tabs")
    assert updates[tabs_index]["visible"] is True
    still_saved = client.predict("agent__independent", "Attribution", api_name="/awe_case")
    assert "INDEPENDENT-AWE-EVIDENCE" in str(still_saved)


def test_export_download_is_real_html_and_model_features_use_only_mock_transport(dashboard, monkeypatch):
    from agent_trajviz.insight import assistant, issue_judge

    client = Client(dashboard.local_url, verbose=False)
    _loaded(dashboard, client)
    exported = _call(dashboard, client, "prepare_html_export")["Export HTML"]
    target = exported.get("value", exported) if isinstance(exported, dict) else exported
    if isinstance(target, dict):
        target = target.get("path")
    document = Path(target).read_text(encoding="utf-8")
    assert "TrajViz" in document and "<html" in document
    assert "wf-card" in document
    calls = []

    def complete(config, system, messages):
        calls.append(config)
        if '"where"' in system:
            return json.dumps(
                {
                    "where": "AGENTS.md",
                    "fix": "Offline fixture: verify after the final edit.",
                    "also": "No provider request",
                    "confidence": "medium",
                }
            )
        return "Offline fixture analysis: inspect recorded failed commands."

    monkeypatch.setattr(assistant, "complete_chat", complete)
    monkeypatch.setattr(issue_judge, "complete_chat", complete)
    _call(
        dashboard,
        client,
        "on_apply_settings",
        {
            "Provider": "openai",
            "Base URL": "https://offline.invalid/v1",
            "API key": "http-fixture-secret",
            "Model": "http-fixture-model",
            "Reply language": "English",
            "Remember connection details in this browser": False,
        },
    )
    answer = _call(dashboard, client, "on_analysis_ask", {"Question": "Explain the failed commands", "Analysis": []})
    assert "Offline fixture analysis" in str(answer)
    findings = _call(dashboard, client, "on_suggest_fixes")
    assert "Offline fixture: verify" in str(findings)
    assert calls and all(config.model == "http-fixture-model" for config in calls)
    assert "http-fixture-secret" not in str(answer) + str(findings)


def test_converge_pairwise_batch_and_intervention_endpoints(tmp_path):
    from agent_trajviz.converge.app import build_ui

    root = tmp_path / "data"
    root.mkdir()
    for name in ["reference.jsonl", "compared.jsonl"]:
        shutil.copy2(_TRAJECTORY, root / name)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps([{"task_id": "public-astropy-case", "reference": "reference.jsonl", "compared": "compared.jsonl"}]),
        encoding="utf-8",
    )
    app = build_ui(data_root=root)
    app.launch(server_name="127.0.0.1", server_port=_port(), prevent_thread_lock=True, quiet=True, inbrowser=False)
    try:
        client = Client(app.local_url, verbose=False)
        pair = _call(
            app,
            client,
            "compare_uploads",
            {
                "Reference Trajectory": handle_file(str(root / "reference.jsonl")),
                "Compared Trajectory": handle_file(str(root / "compared.jsonl")),
            },
        )
        assert "Comparison failed" not in str(pair)
        for chart in ["Milestone Timeline", "Segment Costs", "Divergence Waterfall"]:
            assert isinstance(_figure(pair[chart]), dict), chart
        invalid_rate = _call(
            app,
            client,
            "compare_uploads",
            {
                "Reference Trajectory": handle_file(str(root / "reference.jsonl")),
                "Compared Trajectory": handle_file(str(root / "compared.jsonl")),
                "Token Rate": None,
            },
        )
        assert "Token rate must be a positive number" in str(invalid_rate)
        assert "Traceback" not in str(invalid_rate) and str(root) not in str(invalid_rate)
        assert not _figure(invalid_rate["Milestone Timeline"])["data"]
        assert not _figure(invalid_rate["Segment Costs"])["data"]
        batch = _call(app, client, "do_batch", {"Batch Manifest (JSON)": handle_file(str(manifest))})
        assert "1 of 1 tasks compared" in str(batch)
        intervention = _call(
            app,
            client,
            "do_before_after",
            {
                "Before Manifest (JSON)": handle_file(str(manifest)),
                "After Manifest (JSON)": handle_file(str(manifest)),
            },
        )
        assert "Paired tasks: 1" in str(intervention)
        assert "Metric Deltas" in str(intervention)
    finally:
        app.close()
