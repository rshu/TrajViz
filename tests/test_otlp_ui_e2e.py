"""Saved-capture UI transitions through inspection, loading and HTML export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.issues import OverviewIssue
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from agent_trajviz.insight.session import session_from_run
from agent_trajviz.insight.rendering import format_step_detail, render_workflow_html
from agent_trajviz.insight.ui import load, upload
from tests.test_otlp_e2e import FIXTURE, _bundle
from tests.test_otlp_mapping_e2e import _log, _span, _write


@pytest.fixture
def browser_session(tmp_path, monkeypatch):
    request = SimpleNamespace(session_hash=f"otlp-ui-{tmp_path.name}")
    cache = tmp_path / "ui-cache"
    monkeypatch.setattr(upload.gr.utils, "get_upload_folder", lambda: str(cache))
    yield request
    upload.cleanup_session_export(request)


@pytest.mark.parametrize("partial", [False, True], ids=["observed", "partial"])
def test_otlp_issues_panel_never_claims_healthy_empty_state(tmp_path, partial):
    directory, _ = _bundle(tmp_path, partial=partial)
    session = session_from_run(tv.load(directory, session_id="demo-session"))
    for panel in (build_overview_issues_html(session, issues=[]),
                  build_overview_issues_html(session, issues=[], banner="Provider not configured"),
                  load.do_load(str(directory), session_id="demo-session")["issues_html"]):
        assert "limited evidence" in panel
        assert "none detected" not in panel.lower()
        assert "No major workflow issues detected" not in panel
        assert "Some workflow checks are unavailable" in panel


def test_observed_issues_remain_visible_with_limited_coverage(tmp_path):
    directory, _ = _bundle(tmp_path, partial=True)
    session = session_from_run(tv.load(directory, session_id="demo-session"))
    issue = OverviewIssue("error", "Observed command failed", "The captured result reports an error", steps=(2,))
    panel = build_overview_issues_html(session, issues=[issue], progress="Reviewing captured error")
    assert "1 issue · limited evidence" in panel
    assert "Observed command failed" in panel
    assert "Reviewing captured error" in panel
    assert "none detected" not in panel


@pytest.mark.parametrize("archive", [False, True], ids=["directory", "zip"])
def test_observed_partial_invalid_native_ui_transition_has_no_stale_state(tmp_path, browser_session, archive):
    observed_root, partial_root = tmp_path / "observed", tmp_path / "partial"
    observed_root.mkdir()
    partial_root.mkdir()
    observed = _bundle(observed_root)
    partial = _bundle(partial_root, partial=True)
    chosen = 1 if archive else 0
    old_export = None
    for bundle, tokens in ((observed[chosen], 57), (partial[chosen], None)):
        inspection = load._inspect_sessions(str(bundle))
        assert inspection["visible"] is True
        assert inspection["value"] is None  # A previous capture's selection is cleared.
        assert {value for _label, value in inspection["choices"]} == {"demo-session", "second-session"}
        packed = load.do_load(str(bundle), session_id="second-session")
        run = packed["state_run"]
        assert run.format == "otlp" and run.raw["_otel"]["session_id"] == "second-session"
        assert run.metrics["tokens"]["total"] == tokens
        assert packed["state_raw"] is run.raw
        assert packed["state_steps"] == run.steps
        assert packed["main_tabs"]["visible"] is True
        assert "limited evidence" in packed["issues_html"]
        assert "none detected" not in packed["issues_html"].lower()
        assert packed["export_btn"]["value"] is None
        assert packed["export_btn"]["interactive"] is False
        exported = upload.prepare_html_export(run, request=browser_session)
        assert exported["interactive"] is True
        path = Path(exported["value"])
        html = path.read_text()
        assert run.sha256 in html[:4096]
        assert "OpenTelemetry capture" in html
        if tokens is None:
            assert "No errors detected" not in html
            assert "Failure-pattern checks unavailable" in html
            for key in ("issues_html", "metrics_md", "behavior_md", "diag_summary_html"):
                assert "No errors detected" not in packed[key]
                assert "No diagnostic issues detected" not in packed[key]
            assert packed["token_chart"].layout.annotations
            if old_export:
                assert not old_export.exists()
        old_export = path

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"resourceSpans": "not an array"}))
    inspection = load._inspect_sessions(str(invalid))
    assert inspection["choices"] == [] and inspection["value"] is None
    assert inspection["visible"] is False
    cleared = load.do_load(str(invalid))
    assert cleared["state_run"] is None and cleared["state_raw"] == {} and cleared["state_steps"] == []
    # A native error must not make the independent saved-AWE workspace disappear.
    assert cleared["main_tabs"]["visible"] is True
    assert cleared["overview_kpi_html"]["value"] == ""
    assert cleared["overview_kpi_html"]["visible"] is False
    assert cleared["issues_html"] == ""
    assert cleared["metrics_md"] == ""
    assert cleared["token_chart"].data == ()
    assert cleared["agent_swimlane_chart"].data == ()
    assert cleared["export_btn"]["value"] is None
    assert cleared["export_btn"]["interactive"] is False
    failed_export = upload.prepare_html_export(cleared["state_run"], request=browser_session)
    assert failed_export["value"] is None and failed_export["interactive"] is False

    native = FIXTURE / "native" / "demo-session.jsonl"
    inspection = load._inspect_sessions(str(native))
    assert inspection["choices"] == [] and inspection["value"] is None and inspection["visible"] is False
    packed = load.do_load(str(native), selected_format="claude_code", session_id=inspection["value"])
    run = packed["state_run"]
    assert run.format == "claude_code" and "_otel" not in run.raw
    assert len(run.agents) == 2
    assert packed["main_tabs"]["visible"] is True
    assert "Limited telemetry evidence" not in packed["overview_kpi_html"]["value"]
    assert "limited evidence" not in packed["issues_html"]
    exported = upload.prepare_html_export(run, request=browser_session)
    assert exported["interactive"] is True
    html = Path(exported["value"]).read_text()
    assert run.sha256 in html[:4096]
    assert "OpenTelemetry capture" not in html
    assert old_export and not old_export.exists()


def test_selection_error_clears_the_previous_capture_and_export(tmp_path, browser_session):
    directory, _ = _bundle(tmp_path)
    selected = load.do_load(str(directory), session_id="demo-session")
    export = upload.prepare_html_export(selected["state_run"], request=browser_session)
    assert export["interactive"] is True
    failed = load.do_load(str(directory), session_id="previous-capture-session")
    assert failed["state_run"] is None
    assert failed["state_steps"] == [] and failed["state_raw"] == {}
    assert failed["issues_html"] == "" and failed["metrics_md"] == ""
    assert "previous-capture-session" in failed["summary_banner"]["value"]
    assert failed["export_btn"]["value"] is None and failed["export_btn"]["interactive"] is False
    disabled = upload.prepare_html_export(failed["state_run"], request=browser_session)
    assert disabled["value"] is None and disabled["interactive"] is False


def test_late_otlp_envelope_ui_session_inspection_is_not_prefix_limited(tmp_path):
    source = tmp_path / "late-capture.json"
    batches = [json.loads(line) for line in (FIXTURE / "claude_code_traces.jsonl").read_text().splitlines()]
    second = _span(900, "llm_request", session="second-session", input_tokens=1, output_tokens=1,
                   cache_read_tokens=0, cache_creation_tokens=0)
    source.write_text(json.dumps({"future_field": "x" * (1024 * 1024 + 1), "resourceSpans": [
        *(resource for batch in batches for resource in batch["resourceSpans"]),
        {"scopeSpans": [{"spans": [second]}]},
    ]}))
    inspected = load._inspect_sessions(str(source))
    assert inspected["visible"] is True
    assert inspected["value"] is None
    assert {identity for _label, identity in inspected["choices"]} == {"demo-session", "second-session"}
    packed = load.do_load(str(source), session_id="second-session")
    assert packed["state_run"].raw["_otel"]["session_id"] == "second-session"
    assert packed["state_run"].metrics["tokens"]["total"] == 2


@pytest.mark.parametrize("source", ["attributes", "span-status", "api-error"])
def test_failed_model_error_details_are_visible_and_escaped(tmp_path, source):
    message = 'Request failed <img src=x onerror="alert(1)"> & timed out'
    span = _span(1, "llm_request", request_id="failed-request", **({
        "success": False, "error": message, "error_class": "TimeoutError", "status_code": 504,
    } if source == "attributes" else {}))
    if source == "span-status":
        span["status"] = {"code": 2, "message": message}
    logs = [_log("api_error", request_id="failed-request", error=message,
                 error_class="TimeoutError", status_code=504)] if source == "api-error" else []
    run = tv.load(_write(tmp_path, [span], logs))
    step = run.steps[0]
    assert step["finish"] == "error"
    assert step["tool_calls"] == []
    for rendered in (render_workflow_html(run.steps), format_step_detail(step)):
        assert "Model request failed" in rendered
        assert "Request failed &lt;img" in rendered
        assert "<img src=x" not in rendered
        assert "&amp; timed out" in rendered
    detail = format_step_detail(step)
    assert '&quot;_otel&quot;' in detail
    assert '&quot;error&quot;' in detail
    if source != "span-status":
        assert "TimeoutError" in detail and "Status 504" in detail


def test_missing_otlp_usage_is_unavailable_in_workflow_cards_and_detail(tmp_path):
    run = tv.load(_write(tmp_path, [_span(1, "llm_request", request_id="unmeasured")]))
    assert "n/a tok" in render_workflow_html(run.steps)
    detail = format_step_detail(run.steps[0])
    for metric in ("Total Tokens", "Input Tokens", "Output Tokens", "Reasoning Tokens", "Cache Read", "Cache Write",
                   "Throughput", "Cache Ratio"):
        assert f"<td>{metric}</td><td>n/a</td>" in detail
    measured = tv.load(_write(tmp_path / "zero", [_span(2, "llm_request", input_tokens=0, output_tokens=0,
                                                         cache_read_tokens=0, cache_creation_tokens=0)]))
    assert "0 tok" in render_workflow_html(measured.steps)
    assert "<td>Total Tokens</td><td>0</td>" in format_step_detail(measured.steps[0])
