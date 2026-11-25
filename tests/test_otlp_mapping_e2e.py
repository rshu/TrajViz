"""Public saved-capture audit against Claude's documented operation semantics.

The matrix models two sessions sharing an inbound trace, overlapping delegation,
permission waits, a rejected call, failures, log replay and a second prompt trace.
Expected totals below are declared independently of the projection implementation.
Source: https://code.claude.com/docs/en/monitoring-usage#span-attributes
"""

from __future__ import annotations

import json
from copy import deepcopy

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.figures import figure

BASE = 1_792_963_200_123_456_789
TRACE = "a" * 32
TRACE_TWO = "b" * 32
SESSION = "matrix-session"
OTHER = "unrelated-session"
SECRET = "unrelated-session-content-must-not-leak"
USAGE = {"input_tokens": 10, "output_tokens": 8, "cache_read_tokens": 30, "cache_creation_tokens": 2}
TOKEN_KEYS = set(USAGE)


def _value(value):
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": _attributes(value)}}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_value(item) for item in value]}}
    return {"stringValue": value}


def _attributes(values):
    return [{"key": key, "value": _value(value)} for key, value in values.items()]


def _span(number, kind, *, parent=0, trace=TRACE, session=SESSION, start=0, end=1, status=0, **attrs):
    return {
        "traceId": trace, "spanId": f"{number:016x}",
        **({"parentSpanId": f"{parent:016x}"} if parent else {}),
        "name": f"claude_code.{kind}",
        "startTimeUnixNano": str(BASE + round(start * 1e9)),
        "endTimeUnixNano": str(BASE + round(end * 1e9)),
        "attributes": _attributes({**({"session.id": session} if session else {}), **attrs}),
        "status": {"code": status},
    }


def _log(name, *, trace=TRACE, span=0, session=SESSION, **attrs):
    return {
        **({"traceId": trace} if trace else {}),
        **({"spanId": f"{span:016x}"} if span else {}),
        "timeUnixNano": str(BASE), "body": {"stringValue": f"claude_code.{name}"},
        "attributes": _attributes({"event.name": name, **({"session.id": session} if session else {}), **attrs}),
    }


def _batch(records, *, logs=False):
    signal, scope, entries = ("resourceLogs", "scopeLogs", "logRecords") if logs else (
        "resourceSpans", "scopeSpans", "spans")
    return {signal: [{"resource": {"attributes": _attributes({"service.name": "claude-code"})},
                     scope: [{"scope": {"name": "claude-code", "version": "2.1.310"}, entries: records}]}]}


def _write(tmp_path, spans, logs=(), *, replay=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Child-first batches and mixed session order exercise graph resolution.
    midpoint = len(spans) // 2
    batches = [_batch(list(reversed(spans[midpoint:]))), _batch(list(reversed(spans[:midpoint])))]
    if replay:
        batches += deepcopy(batches)
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(batch) for batch in batches) + "\n")
    log_batches = [_batch(list(logs), logs=True)]
    if replay:
        log_batches += deepcopy(log_batches)
    (tmp_path / "logs.jsonl").write_text("\n".join(json.dumps(batch) for batch in log_batches) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"type": "agent-trajviz-otlp-bundle", "schema_version": 1,
                                    "profile": "claude_code", "traces": ["traces.jsonl"],
                                    "logs": ["logs.jsonl"]}))
    return manifest


def _matrix():
    spans = [
        _span(1, "interaction", start=0, end=18, user_prompt="Audit a production parser and review its tests."),
        _span(2, "llm_request", parent=1, start=0.1, end=2.1, request_id="request-one", **USAGE,
              model="claude-test", success=True, stop_reason="tool_use"),
        _span(3, "tool", parent=1, start=2.1, end=9.1, tool_name="Agent", tool_use_id="spawn-reviewer",
              tool_input='{"subagent_type":"reviewer","prompt":"Review parser invariants"}'),
        _span(4, "tool.execution", parent=3, start=2.2, end=9.1, success=True),
        _span(5, "llm_request", parent=3, session="", start=3, end=6, agent_id="reviewer", request_id="request-child",
              input_tokens=3, output_tokens=5, cache_read_tokens=4, cache_creation_tokens=1),
        _span(6, "tool", parent=3, start=6, end=8, agent_id="reviewer", tool_name="Read", tool_use_id="read-source",
              tool_input='{"file_path":"parser.py"}'),
        _span(7, "tool.blocked_on_user", parent=6, start=6, end=6.5, decision="accept"),
        _span(8, "tool.execution", parent=6, start=6.5, end=8, success=True),
        _span(9, "tool", parent=1, start=10, end=14.5, tool_name="Bash", tool_use_id="test-command",
              tool_input='{"command":"python -m pytest tests/test_parser.py"}'),
        _span(10, "tool.blocked_on_user", parent=9, start=10, end=11, decision="accept"),
        _span(11, "tool.execution", parent=9, start=11, end=14.5, success=False, error="ShellError", status=2),
        _span(12, "tool", parent=1, start=14.5, end=15.5, tool_name="Read", tool_use_id="denied-read"),
        _span(13, "tool.blocked_on_user", parent=12, start=14.5, end=15.5, decision="reject"),
        _span(14, "interaction", trace=TRACE_TWO, start=20, end=23, user_prompt="Summarize the review."),
        _span(15, "llm_request", trace=TRACE_TWO, parent=14, start=20.1, end=21.1,
              request_id="request-final", input_tokens=0, output_tokens=1, cache_read_tokens=43,
              cache_creation_tokens=0, response="Review complete.", success=True, stop_reason="end_turn"),
        _span(16, "compaction", trace=TRACE_TWO, parent=14, start=22, end=23, summary="Retained review notes"),
        # Two Claude subprocesses can share one inbound TRACEPARENT while
        # retaining distinct session.id values and colliding producer IDs.
        _span(101, "interaction", session=OTHER, start=1, end=30, user_prompt=SECRET),
        _span(102, "llm_request", parent=101, session=OTHER, start=2, end=3,
              request_id="request-one", response=SECRET, input_tokens=900, output_tokens=800,
              cache_read_tokens=700, cache_creation_tokens=600),
    ]
    spans[1]["endTimeUnixNano"] = str(int(spans[1]["endTimeUnixNano"]) + 7)
    spans[5]["events"] = [{"name": "tool.output", "timeUnixNano": str(BASE + 8_000_000_000),
                           "attributes": _attributes({"content": "def parse(...): [TRUNCATED]",
                                                       "content_truncated": True})}]
    logs = [
        _log("api_request", request_id="request-one", **USAGE),
        _log("assistant_response", request_id="request-one", response="I will inspect, test and delegate a review.",
             **{"message.uuid": "response-one"}),
        _log("tool_result", tool_use_id="spawn-reviewer", request_id="request-one", success=True),
        _log("assistant_response", request_id="request-child", agent_id="reviewer", response="One edge case found."),
        _log("tool_result", tool_use_id="read-source", request_id="request-child", agent_id="reviewer", success=True),
        _log("tool_result", tool_use_id="test-command", request_id="request-one", success=False,
             error="ShellError", duration_ms=3500),
        _log("tool_decision", tool_use_id="denied-read", request_id="request-one", decision="reject"),
        _log("api_request", session=OTHER, request_id="request-one", input_tokens=900, output_tokens=800,
             cache_read_tokens=700, cache_creation_tokens=600),
        _log("assistant_response", session=OTHER, request_id="request-one", response=SECRET),
    ]
    return spans, logs


def _attrs(record):
    return {entry["key"]: next(iter(entry["value"].values())) for entry in record.get("attributes", [])}


@pytest.mark.parametrize("variant", ["complete", "partial", "missing"])
@pytest.mark.parametrize("replay", [False, True])
def test_interleaved_capture_matrix_public_contract(tmp_path, variant, replay):
    spans, logs = _matrix()
    if variant != "complete":
        for record in spans + logs:
            attrs = _attrs(record)
            kind = record.get("name", "").removeprefix("claude_code.")
            if variant == "missing" or attrs.get("request_id") in {"request-child", "request-final"}:
                record["attributes"] = [entry for entry in record["attributes"] if entry["key"] not in TOKEN_KEYS]
            if variant == "missing":
                record["attributes"] = [entry for entry in record["attributes"] if entry["key"] not in {
                    "success", "response", "user_prompt", "tool_input", "summary", "error",
                }]
                if kind:
                    record["startTimeUnixNano"] = record["endTimeUnixNano"] = "0"
                    record["status"] = {"code": 0}
                    record.pop("events", None)
    source = _write(tmp_path, spans, logs, replay=replay)
    sessions = tv.list_sessions(source)
    assert {row["id"] for row in sessions} == {SESSION, OTHER}
    run = tv.load(source, session_id=SESSION)
    assert run.raw["_otel"]["trace_ids"] == [TRACE, TRACE_TWO]
    assert run.raw["metadata"]["sub_agent_count"] == 1
    assert {agent["agent_id"] for agent in run.raw["_otel"]["agents"]} == {"reviewer"}
    assert SECRET not in json.dumps(run.to_dict(include_raw=True))
    assert {step["session_id"] for step in run.steps} == {SESSION}
    assert run.metrics["tool_call_count"] == 4
    assert len([step for step in run.steps if step["role"] == "assistant"]) == 3
    expected_tokens = ({"input": 13, "output": 14, "cache_read": 77, "cache_write": 3, "total": 107}
                       if variant == "complete" else dict.fromkeys(("input", "output", "cache_read", "cache_write", "total"),
                                                                 None) if variant == "missing" else
                       {"input": 10, "output": 8, "cache_read": 30, "cache_write": 2, "total": 50})
    for key, expected in expected_tokens.items():
        assert run.metrics["tokens"][key] == expected
    assert run.metrics["tokens"]["reasoning"] is None
    calls = {call["tool_id"]: call for step in run.steps for call in step["tool_calls"]}
    assert calls["denied-read"]["status"] == "unknown"
    assert calls["denied-read"]["duration_ms"] is None
    if variant != "missing":
        assert run.metrics["total_duration"] == 6.0
        assert run.metrics["tool_fail"] == 1 and run.metrics["tool_unknown_outcomes"] == 1
        assert calls["test-command"]["duration_ms"] == 3500
        assert calls["test-command"]["metadata"]["_otel"]["permission_wait_ms"] == 1000
        assert calls["read-source"]["metadata"]["_otel"]["fields"]["tool_outputs"] == "partial"
        assert run.steps[1]["_otel"]["start_time_unix_nano"] == BASE + 100_000_000
        assert run.steps[1]["_otel"]["end_time_unix_nano"] == BASE + 2_100_000_007
        assert run.steps[1]["duration"] == 2.000000007
    else:
        assert run.metrics["total_duration"] is None
        assert run.metrics["tool_fail"] is None
    views = run.views()
    assert len(views) == len(tv.VIEW_CATALOG)
    chart_count = 0
    for view_id, view in views.items():
        document = view.to_dict()
        jsonschema.validate(document, tv.view_schema(view_id))
        json.dumps(document, allow_nan=False)
        if view.kind == "chart":
            chart_count += 1
            chart = figure(view)
            json.loads(chart.to_json())
            if view.availability["status"] == "unavailable":
                assert not chart.data
            if view.availability["status"] == "partial":
                assert any("Partial data" in annotation.text for annotation in chart.layout.annotations)
    assert chart_count == 12
    for include_raw in (False, True):
        document = json.loads(json.dumps(run.to_dict(include_raw=include_raw), allow_nan=False))
        restored = tv.Trajectory.from_dict(document)
        assert restored.sha256 == run.sha256
        assert restored.metrics == run.metrics
        for view_id, view in views.items():
            assert restored.view(view_id).to_dict() == view.to_dict()


@pytest.mark.parametrize("include_agent", [False, True])
@pytest.mark.parametrize("traceless", [False, True])
def test_repeated_request_and_tool_ids_require_unambiguous_agent_evidence(tmp_path, include_agent, traceless):
    spans = [_span(1, "llm_request", request_id="repeated"),
             _span(2, "llm_request", request_id="repeated", agent_id="reviewer"),
             _span(3, "tool", tool_name="Read", tool_use_id="repeated-call"),
             _span(4, "tool", tool_name="Read", tool_use_id="repeated-call", agent_id="reviewer")]
    lane = {"agent_id": "reviewer"} if include_agent else {}
    trace = "" if traceless else TRACE
    logs = [_log("api_request", trace=trace, request_id="repeated", **lane, **USAGE),
            _log("tool_result", trace=trace, tool_use_id="repeated-call", **lane, success=False)]
    run = tv.load(_write(tmp_path, spans, logs))
    for step in run.steps:
        observed = include_agent and step["agent"] == "reviewer"
        if step["role"] == "assistant":
            assert (step["_otel"]["fields"]["tokens.total"] == "observed") is observed
            assert step["tokens"]["total"] == (50 if observed else 0)
        else:
            assert step["tool_calls"][0]["status"] == ("error" if observed else "unknown")


def test_span_count_aliases_win_and_disclose_conflicting_log_aliases(tmp_path):
    span = _span(1, "llm_request", request_id="request", input_tokens=1, output_tokens=2,
                 cache_read_input_tokens=3, cache_creation_input_tokens=4)
    logs = [_log("api_request", request_id="request", input_tokens=1, output_tokens=2,
                 cache_read_tokens=300, cache_creation_tokens=400)]
    run = tv.load(_write(tmp_path, [span], logs))
    assert run.metrics["tokens"]["total"] == 10
    assert run.metrics["tokens"]["cache_read"] == 3
    assert run.metrics["tokens"]["cache_write"] == 4
    assert run.steps[0]["_otel"]["conflicts"] == ["cache_creation_tokens", "cache_read_tokens"]
    assert any("Conflicting span/log usage" in warning for warning in run.warnings)


def test_valid_log_usage_can_fill_invalid_span_fields_without_inventing_totals(tmp_path):
    span = _span(1, "llm_request", request_id="request", input_tokens="invalid", output_tokens=2)
    run = tv.load(_write(tmp_path, [span], [_log("api_request", request_id="request", input_tokens=10)]))
    assert run.metrics["tokens"]["input"] == 10
    assert run.metrics["tokens"]["total"] is None


@pytest.mark.parametrize("source", ["success_attribute", "span_status", "api_error"])
def test_model_failures_are_visible_without_fabricated_tool_failures(tmp_path, source):
    span = _span(1, "llm_request", request_id="failed-request", **USAGE,
                 **({"success": False, "error": "Request timeout"} if source == "success_attribute" else {}))
    if source == "span_status":
        span["status"] = {"code": 2, "message": "Request timeout"}
    logs = [_log("api_error", request_id="failed-request", error="Request timeout", status_code=504)] if (
        source == "api_error") else []
    run = tv.load(_write(tmp_path, [span], logs))
    assert run.steps[0]["finish"] == "error"
    assert run.steps[0]["_otel"]["success"] is False
    assert run.steps[0]["_otel"]["error"] == "Request timeout"
    assert run.metrics["tool_call_count"] == 0
    assert run.metrics["tool_fail"] is None
    workflow = run.view("workflow").data["steps"][0]
    assert workflow["category"] == "system_error"
    assert workflow["error_kind"] == "system"
    assert "Errors" in workflow["filters"]


def test_client_request_id_correlates_timeout_without_server_id_and_deduplicates_logs(tmp_path):
    span = _span(1, "llm_request", client_request_id="client-timeout")
    logs = [_log("api_error", client_request_id="client-timeout", error="Connection timeout", status_code=504),
            _log("api_error", client_request_id="different", error="Unrelated error")]
    run = tv.load(_write(tmp_path, [span], logs, replay=True))
    assert run.steps[0]["finish"] == "error"
    assert run.steps[0]["_otel"]["error"] == "Connection timeout"
    assert len(run.steps[0]["_otel"]["correlated_logs"]) == 1
    assert run.steps[0]["_otel"]["client_request_id"] == "client-timeout"
    assert run.metrics["tokens"]["total"] is None


def test_request_and_client_ids_do_not_attach_the_same_log_twice(tmp_path):
    span = _span(1, "llm_request", client_request_id="client", request_id="server")
    logs = [_log("api_request", client_request_id="client", request_id="server", **USAGE)]
    run = tv.load(_write(tmp_path, [span], logs))
    assert run.metrics["tokens"]["total"] == 50
    assert len(run.steps[0]["_otel"]["correlated_logs"]) == 1


def test_result_input_takes_precedence_over_permission_proposal(tmp_path):
    span = _span(1, "tool", tool_name="Bash", tool_use_id="rewritten")
    logs = [_log("tool_decision", tool_use_id="rewritten", decision="accept", tool_input='{"command":"proposed"}'),
            _log("tool_result", tool_use_id="rewritten", success=True, tool_input='{"command":"executed"}')]
    run = tv.load(_write(tmp_path, [span], logs))
    assert run.steps[0]["tool_calls"][0]["input"]["command"] == "executed"


def test_redacted_output_event_does_not_erase_recorded_span_content(tmp_path):
    span = _span(1, "tool", tool_name="Read", tool_use_id="read", new_context="Complete retained content")
    span["events"] = [{"name": "tool.output", "attributes": _attributes({"content": "<REDACTED>"})}]
    run = tv.load(_write(tmp_path, [span]))
    call = run.steps[0]["tool_calls"][0]
    assert call["output"] == "Complete retained content"
    assert call["metadata"]["_otel"]["fields"]["tool_outputs"] == "observed"
    assert "tool_outputs" not in call["metadata"]["_otel"]["reasons"]


def test_partial_token_population_is_drawn_with_partial_notice(tmp_path):
    run = tv.load(_write(tmp_path, [_span(1, "llm_request", **USAGE), _span(2, "llm_request")]))
    view = run.view("token_usage")
    assert view.availability["status"] == "partial"
    chart = figure(view)
    assert chart.data
    assert any("Partial data" in annotation.text for annotation in chart.layout.annotations)


def test_tool_duration_availability_uses_execution_evidence_not_model_timing(tmp_path):
    spans = [_span(1, "llm_request", **USAGE), _span(2, "tool", tool_name="Read", tool_use_id="read")]
    run = tv.load(_write(tmp_path, spans))
    assert run.view("tool_durations").availability["status"] == "unavailable"
    assert not figure(run.view("tool_durations")).data
    spans.append(_span(3, "tool.execution", parent=2, success=True))
    complete = tv.load(_write(tmp_path / "complete", spans))
    assert complete.view("tool_durations").availability["status"] == "observed"
    assert figure(complete.view("tool_durations")).data


def test_client_correlated_response_supplies_explicit_tool_association(tmp_path):
    spans = [_span(1, "llm_request", client_request_id="client-only"),
             _span(2, "tool", tool_name="Read", tool_use_id="read")]
    logs = [_log("api_request", request_id="server", client_request_id="client-only", **USAGE),
            _log("tool_result", tool_use_id="read", request_id="server", success=True)]
    run = tv.load(_write(tmp_path, spans, logs))
    assert len(run.steps) == 1
    assert run.steps[0]["_otel"]["request_id"] == "server"
    assert run.steps[0]["tool_calls"][0]["tool_id"] == "read"


def test_successful_request_is_not_reclassified_by_an_intermediate_error_log(tmp_path):
    span = _span(1, "llm_request", request_id="recovered", success=True, stop_reason="end_turn", **USAGE)
    logs = [_log("api_error", request_id="recovered", error="Transient retry"),
            _log("api_request", request_id="recovered", **USAGE)]
    run = tv.load(_write(tmp_path, [span], logs))
    assert run.steps[0]["finish"] == "end_turn"
    assert run.steps[0]["_otel"]["success"] is True
    assert run.view("workflow").data["steps"][0]["error_kind"] is None


@pytest.mark.parametrize("source", ["execution", "log"])
@pytest.mark.parametrize("duration_ms", [0.5, 0.0])
def test_exact_and_zero_tool_timings_reach_public_analytics(tmp_path, source, duration_ms):
    from agent_trajviz.insight.metrics import tool_call_duration_ms

    spans = [_span(1, "tool", tool_name="Read", tool_use_id="precise-read")]
    logs = []
    if source == "execution":
        execution = _span(2, "tool.execution", parent=1, start=0, end=0, success=True)
        execution["endTimeUnixNano"] = str(BASE + round(duration_ms * 1_000_000))
        spans.append(execution)
    else:
        logs = [_log("tool_result", tool_use_id="precise-read", duration_ms=duration_ms, success=True)]
    run = tv.load(_write(tmp_path, spans, logs))
    call = run.steps[0]["tool_calls"][0]
    assert call["duration_ms"] == duration_ms
    if source == "execution":
        # Both display timestamps floor to the same integer millisecond. The
        # nanosecond-derived duration still proves a half-millisecond interval.
        assert call["time_end"] == call["time_start"]
    assert tool_call_duration_ms(call) == duration_ms
    assert run.view("step_analytics").data[0]["tool_time_ms"] == duration_ms
    assert run.view("steps").data[0]["tool_time_sum"] == duration_ms / 1000
    tool = run.view("tool_durations").data["tools"][0]
    assert tool["total_s"] == duration_ms / 1000
    assert tool["calls"][0]["seconds"] == duration_ms / 1000
    assert run.metrics["avg_tool_duration"] is not None
    assert run.metrics["tool_time_total"] is not None
    for include_raw in (False, True):
        restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=include_raw))))
        assert restored.view("tool_durations").to_dict() == run.view("tool_durations").to_dict()
        assert restored.view("step_analytics").to_dict() == run.view("step_analytics").to_dict()


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), 10**400])
def test_invalid_otlp_duration_cannot_override_valid_execution_bounds(invalid):
    from agent_trajviz.insight.metrics import tool_call_duration_ms

    call = {"duration_ms": invalid, "time_start": 10, "time_end": 15,
            "metadata": {"_otel": {"fields": {"duration": "observed"}}}}
    assert tool_call_duration_ms(call) == 5


def test_native_timing_precedence_and_unobserved_otlp_timing_remain_unchanged():
    from agent_trajviz.insight.metrics import tool_call_duration_ms

    call = {"duration_ms": 0.5, "time_created": 1, "time_updated": 5, "time_start": 2, "time_end": 3}
    assert tool_call_duration_ms(call) == 4
    assert tool_call_duration_ms({"duration_ms": 0}) is None
    call["metadata"] = {"_otel": {"fields": {"duration": "unavailable"}}}
    assert tool_call_duration_ms(call) is None



def test_parallel_tool_windows_union_nanoseconds_before_display_rounding(tmp_path):
    spans = [_span(1, "llm_request", request_id="parallel", **USAGE)]
    logs = []
    for tool_number, start_ns in ((2, 0), (4, 250_000)):
        tool_id = f"read-{tool_number}"
        spans.append(_span(tool_number, "tool", tool_name="Read", tool_use_id=tool_id))
        execution = _span(tool_number + 1, "tool.execution", parent=tool_number, success=True)
        execution["startTimeUnixNano"] = str(BASE + start_ns)
        execution["endTimeUnixNano"] = str(BASE + start_ns + 500_000)
        spans.append(execution)
        logs.append(_log("tool_result", tool_use_id=tool_id, request_id="parallel", success=True))
    run = tv.load(_write(tmp_path, spans, logs))
    assert len(run.steps) == 1
    assert run.view("steps").data[0]["tool_time_sum"] == 0.00075
    # Per-call analytics sum work; the step wall-time uses the interval union.
    assert run.view("step_analytics").data[0]["tool_time_ms"] == 1.0
    assert run.view("tool_durations").data["tools"][0]["total_s"] == 0.001
    for include_raw in (False, True):
        restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=include_raw))))
        assert restored.view("steps").data[0]["tool_time_sum"] == 0.00075
