"""Public saved-OTLP loading, selection, exports and evidence contracts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import jsonschema
import pytest

import agent_trajviz as tv

FIXTURE = Path(__file__).parent / "fixtures" / "otlp"
BASE = 1_792_963_200_123_456_789


def _value(value):
    if isinstance(value, int):
        return {"intValue": str(value)}
    return {"stringValue": value}


def _span(trace, identity, kind, session, *, parent="", offset=0, **attrs):
    return {
        "traceId": trace, "spanId": identity,
        **({"parentSpanId": parent} if parent else {}),
        "name": f"claude_code.{kind}",
        "startTimeUnixNano": str(BASE + offset), "endTimeUnixNano": str(BASE + offset + 1000),
        "attributes": [{"key": key, "value": _value(value)} for key, value in {
            "session.id": session, **attrs,
        }.items()],
    }


def _session(index, tokens=10):
    trace = f"{index:032x}"
    parent = f"{index * 2:016x}"
    return [
        _span(trace, parent, "interaction", f"session-{index}", offset=index * 10_000),
        _span(trace, f"{index * 2 + 1:016x}", "llm_request", f"session-{index}",
              parent=parent, offset=index * 10_000 + 1,
              input_tokens=tokens, output_tokens=5, cache_read_tokens=0, cache_creation_tokens=0,
              model="claude-test", request_id=f"request-{index}"),
    ]


def _payload(*sessions):
    return {"resourceSpans": [{"resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "claude-code"}},
    ]}, "scopeSpans": [{"scope": {"name": "claude-code"}, "spans": [
        span for session in sessions for span in session
    ]}]}]}


def _write(tmp_path, *sessions):
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps(_payload(*sessions)) + "\n")
    return path


def test_standalone_autodetection_and_explicit_format_agree(tmp_path):
    source = _write(tmp_path, _session(1))
    detected = tv.load(source)
    selected = tv.load(source, format="otlp")
    assert "otlp" in tv.FORMATS
    assert detected.format == "otlp" and detected.format_label == "OpenTelemetry"
    assert detected.to_dict(include_raw=True) == selected.to_dict(include_raw=True)
    assert detected.metrics["tokens"]["total"] == 15
    assert len(detected.steps) == 2
    assert detected.raw["_otel"]["session_id"] == "session-1"
    assert detected.raw["_otel"]["spans"][0]["start_time_unix_nano"] == BASE + 10_000


def test_partial_usage_does_not_create_context_growth(tmp_path):
    source = _write(tmp_path, [
        _span(f"{1:032x}", f"{index + 1:016x}", "llm_request", "partial",
              offset=index * 10_000, input_tokens=count, output_tokens=5)
        for index, count in enumerate((10, 20))
    ])
    trajectory = tv.load(source)
    rows = trajectory.view("steps").data
    assert [(row["tokens_input"], row["tokens_total"]) for row in rows] == [(10, None), (20, None)]
    assert trajectory.view("hotspots").data["top_growth"] == []


def test_format_gate_rejects_otlp_as_native(tmp_path):
    source = _write(tmp_path, _session(1))
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="claude_code")
    assert caught.value.code == "mismatch" and caught.value.detected == "otlp"


def test_format_gate_rejects_native_as_otlp_and_keeps_native_loading():
    source = FIXTURE / "native" / "demo-session.jsonl"
    native = tv.load(source, format="claude_code")
    assert native.format == "claude_code"
    assert native.metrics["tokens"]["total"] == 57
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="otlp")
    assert caught.value.code == "mismatch" and caught.value.detected == "claude_code"


def test_multiple_sessions_are_discoverable_and_never_implicitly_merged(tmp_path):
    source = _write(tmp_path, _session(1, 10), _session(2, 20))
    sessions = tv.list_sessions(source)
    assert [row["id"] for row in sessions] == ["session-1", "session-2"]
    assert [row["interaction_count"] for row in sessions] == [1, 1]
    assert all(row["profile"] == "claude_code" for row in sessions)
    assert tv.list_sessions(source, format="otlp") == sessions
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source)
    assert caught.value.code == "ambiguous_session"
    assert caught.value.sessions == sessions
    first = tv.load(source, session_id="session-1")
    second = tv.load(source, format="otlp", session_id="session-2")
    assert first.metrics["tokens"]["total"] == 15
    assert second.metrics["tokens"]["total"] == 25
    assert first.sha256 != second.sha256
    assert first.raw["_otel"]["capture_sha256"] == second.raw["_otel"]["capture_sha256"]
    assert len(first.steps) == len(second.steps) == 2
    assert {step["session_id"] for step in first.steps} == {"session-1"}
    assert {step["session_id"] for step in second.steps} == {"session-2"}
    assert set(first.raw["_otel"]["trace_ids"]).isdisjoint(second.raw["_otel"]["trace_ids"])
    assert first.raw["_otel"]["source"] == second.raw["_otel"]["source"]


def test_multiple_interaction_traces_share_one_explicit_session(tmp_path):
    one, two = _session(1, 10), _session(2, 20)
    for span in two:
        span["attributes"][0]["value"] = {"stringValue": "session-1"}
    source = _write(tmp_path, one, two)
    sessions = tv.list_sessions(source)
    assert len(sessions) == 1 and sessions[0]["interaction_count"] == 2
    run = tv.load(source)
    assert run.metrics["tokens"]["total"] == 40
    assert len(run.steps) == 4


@pytest.mark.parametrize("value", ["", 3, False])
def test_invalid_selector_values_are_rejected_early(tmp_path, value):
    source = _write(tmp_path, _session(1))
    with pytest.raises(ValueError, match="non-empty string"):
        tv.load(source, session_id=value)


def test_unknown_selector_and_non_otlp_selection_have_actionable_errors(tmp_path):
    source = _write(tmp_path, _session(1))
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, session_id="absent")
    assert caught.value.code == "session_not_found"
    with pytest.raises(tv.LoadError) as caught:
        tv.load(FIXTURE / "native" / "demo-session.jsonl", session_id="demo-session")
    assert caught.value.code == "unsupported_selection"
    with pytest.raises(ValueError):
        tv.list_sessions(source, format="claude_code")


def test_selected_session_survives_normalized_export_and_view_roundtrip(tmp_path):
    source = _write(tmp_path, _session(1), _session(2, 20))
    run = tv.load(source, session_id="session-2")
    document = run.to_dict(include_raw=True)
    restored = tv.Trajectory.from_dict(json.loads(json.dumps(document)))
    assert restored.to_dict(include_raw=True) == document
    export_path = tmp_path / "trajectory.json"
    export_path.write_text(json.dumps(document))
    assert tv.load(export_path, session_id="session-2").to_dict(include_raw=True) == document
    with pytest.raises(tv.LoadError) as caught:
        tv.load(export_path, session_id="session-1")
    assert caught.value.code == "session_not_found"
    for view_id in ("metrics", "summary", "steps", "health", "data_quality", "agent_swimlane"):
        assert restored.view(view_id).to_dict() == run.view(view_id).to_dict()
    assert restored.raw["_otel"]["spans"][0]["start_time_unix_nano"] == BASE + 20_000


def test_unknown_metrics_and_context_are_visible_and_validate_schema_v2(tmp_path):
    spans = _session(1)
    spans[1]["attributes"] = [entry for entry in spans[1]["attributes"] if not entry["key"].endswith("tokens")]
    run = tv.load(_write(tmp_path, spans))
    assert run.metrics["tokens"]["total"] is None
    quality = run.view("data_quality").data
    assert quality["source"] == "otlp" and quality["capture_complete"] is None
    assert quality["fields"]["tokens.total"]["status"] in {"unavailable", "unknown"}
    assert quality["fields"]["duration"]["status"] == "observed"
    assert run.view("context_pressure").availability["status"] == "unavailable"
    for view_id in ("metrics", "summary", "steps", "data_quality", "health", "context_pressure", "token_usage"):
        exported = run.view(view_id).to_dict()
        assert exported["schema_version"] == 2
        jsonschema.validate(exported, tv.view_schema(view_id))
        json.dumps(exported, allow_nan=False)


def test_lightweight_roundtrip_keeps_quality_session_and_identity(tmp_path):
    run = tv.load(FIXTURE)
    document = run.to_dict()
    assert "raw" not in document
    assert "spans" not in document["telemetry"] and "logs" not in document["telemetry"]
    restored = tv.Trajectory.from_dict(json.loads(json.dumps(document)))
    assert restored.sha256 == run.sha256
    assert restored.raw["_source_sha256"] == run.sha256
    assert restored.raw["_otel"]["session_id"] == run.raw["_otel"]["session_id"]
    assert restored.metrics == run.metrics
    assert restored.view("data_quality").data == run.view("data_quality").data
    source = tmp_path / "lightweight.json"
    source.write_text(json.dumps(document))
    assert tv.load(source, session_id=run.raw["_otel"]["session_id"]).metrics == run.metrics


@pytest.mark.parametrize("include_raw", [False, True], ids=["lightweight", "full"])
@pytest.mark.parametrize("selection", ["auto", "format", "session", "both"])
def test_sorted_normalized_exports_reload_when_type_follows_large_steps(tmp_path, include_raw, selection):
    run = tv.load(FIXTURE)
    document = run.to_dict(include_raw=include_raw)
    data = json.dumps(document, sort_keys=True).encode()
    # Force the full-document recognition path; a prefix-only export probe
    # cannot see the discriminator after sorted steps/raw fields.
    assert document["type"].encode() not in data[:4096]
    source = tmp_path / "sorted-trajectory.json"
    source.write_bytes(data)
    options = {}
    if selection in ("format", "both"):
        options["format"] = "otlp"
    if selection in ("session", "both"):
        options["session_id"] = run.raw["_otel"]["session_id"]
    restored = tv.load(source, **options)
    assert restored.sha256 == run.sha256
    assert restored.metrics == run.metrics
    assert restored.view("data_quality").data == run.view("data_quality").data
    assert restored.raw["_otel"]["session_id"] == run.raw["_otel"]["session_id"]
    if include_raw:
        assert restored.raw["_otel"]["spans"] == run.raw["_otel"]["spans"]
        assert restored.raw["_otel"]["logs"] == run.raw["_otel"]["logs"]


def test_late_top_level_envelope_has_same_selected_identity_auto_and_forced(tmp_path):
    from agent_trajviz.insight.formats.otlp_files import looks_like_otlp_source

    payload = {"unknown_future_field": "x" * (1024 * 1024 + 1), **_payload(_session(1))}
    source = tmp_path / "late-envelope.json"
    source.write_text(json.dumps(payload))
    assert not looks_like_otlp_source(source)
    detected = tv.load(source)
    forced = tv.load(source, format="otlp")
    assert detected.sha256 == forced.sha256
    assert detected.raw["_source_sha256"] == detected.raw["_otel"]["run_sha256"] == forced.sha256
    assert detected.raw["_otel"]["capture_sha256"] == forced.raw["_otel"]["capture_sha256"]
    assert detected.metrics == forced.metrics


def test_late_top_level_envelope_supports_automatic_session_selection(tmp_path):
    source = tmp_path / "late-multi-session.json"
    source.write_text(json.dumps({"unknown_future_field": "x" * (1024 * 1024 + 1),
                                 **_payload(_session(1), _session(2))}))
    selected = tv.load(source, session_id="session-2")
    forced = tv.load(source, format="otlp", session_id="session-2")
    assert selected.to_dict(include_raw=True) == forced.to_dict(include_raw=True)
    assert selected.metrics["tokens"]["total"] == 15
    assert selected.raw["_otel"]["session_id"] == "session-2"


@pytest.mark.parametrize("format", ["claude_code", "cursor"])
def test_late_otlp_session_selection_honors_native_format_gate(tmp_path, format):
    source = tmp_path / "late-multi-session.json"
    source.write_text(json.dumps({"unknown_future_field": "x" * (1024 * 1024 + 1),
                                 **_payload(_session(1), _session(2))}))
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format=format, session_id="session-2")
    assert caught.value.code == "mismatch"
    assert caught.value.detected == "otlp"
    assert caught.value.path == str(source)


@pytest.mark.parametrize("data,code", [
    (b"{malformed", "invalid_otlp"),
    (b"\xff", "invalid_otlp"),
    (b'{"resourceSpans": "invalid collection"}', "invalid_otlp"),
    (b'{"unknown": true}', "unsupported_otlp"),
])
def test_forced_malformed_bounded_reads_fail_with_public_actionable_errors(tmp_path, data, code):
    source = tmp_path / "broken.json"
    source.write_bytes(data)
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="otlp")
    assert caught.value.code == code
    assert caught.value.path == str(source)
    assert str(caught.value)


def test_forced_oversized_unknown_source_is_rejected_before_native_conversion(tmp_path, monkeypatch):
    from agent_trajviz.insight import loaders
    from agent_trajviz.insight.formats import otlp_files

    source = tmp_path / "oversized.json"
    source.write_text(json.dumps({"padding": "x" * 1000}))
    monkeypatch.setattr(otlp_files, "MAX_MEMBER_BYTES", 256)

    def unexpected_native_conversion(*_args, **_kwargs):
        pytest.fail("forcing OTLP must not convert the source as a native trajectory")

    monkeypatch.setattr(loaders, "_apply_format", unexpected_native_conversion)
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="otlp")
    assert caught.value.code == "size_limit" and caught.value.path == str(source)


def test_forced_special_file_is_rejected_before_a_blocking_open(tmp_path, monkeypatch):
    import builtins
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no named pipes")
    source = tmp_path / "named-pipe.json"
    os.mkfifo(source)
    original_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        if os.fspath(path) == str(source):
            pytest.fail("opening this named pipe would block before bounded-source validation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="otlp")
    assert caught.value.code == "load_failed" and caught.value.path == str(source)


def test_fixture_manifest_directory_and_zip_expose_equivalent_views(tmp_path):
    direct = tv.load(FIXTURE)
    manifest = tv.load(FIXTURE / "manifest.json", format="otlp")
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in ("manifest.json", "claude_code_traces.jsonl", "claude_code_logs.jsonl"):
            archive.write(FIXTURE / name, f"capture/{name}")
    zipped = tv.load(archive_path)
    assert direct.sha256 == manifest.sha256 == zipped.sha256
    assert direct.metrics["tokens"]["total"] == manifest.metrics["tokens"]["total"] == zipped.metrics["tokens"]["total"] == 57
    assert direct.metrics["tool_call_count"] == zipped.metrics["tool_call_count"] == 3
    assert tv.list_sessions(archive_path) == tv.list_sessions(FIXTURE)
    assert len(zipped.raw["_otel"]["agents"]) == 1
    assert any("!/" in span["source"]["path"] for span in zipped.raw["_otel"]["spans"])


def test_forced_empty_directory_uses_otlp_bundle_validation(tmp_path):
    with pytest.raises(tv.LoadError) as caught:
        tv.load(tmp_path, format="otlp")
    assert caught.value.code == "invalid_otlp_bundle"
    assert "requires exactly one" in str(caught.value)


def test_unrecognized_directory_session_selection_is_not_a_file_probe(tmp_path):
    with pytest.raises(tv.LoadError) as caught:
        tv.load(tmp_path, session_id="absent")
    assert caught.value.code == "unsupported_selection"


def test_json_exports_include_coverage_and_reload_raw_provenance(tmp_path):
    run = tv.load(FIXTURE)
    destination = tmp_path / "export"
    paths = tv.export_json(run, destination, only=["metrics", "data_quality"])
    assert destination / "metrics.json" in paths
    assert destination / "data_quality.json" in paths
    restored = tv.load(destination / "trajectory.json")
    assert restored.raw["_otel"] == run.raw["_otel"]
    assert restored.view("data_quality").data == run.view("data_quality").data
    for view_id in ("metrics", "data_quality"):
        exported = json.loads((destination / f"{view_id}.json").read_text())
        jsonschema.validate(exported, tv.view_schema(view_id))


def test_logs_and_metrics_alone_do_not_fabricate_trajectories(tmp_path):
    source = tmp_path / "logs.json"
    source.write_text(json.dumps({"resourceLogs": [{"scopeLogs": [{"logRecords": [{
        "body": {"stringValue": "claude_code.api_request"},
        "attributes": [{"key": "input_tokens", "value": {"intValue": "100"}}],
    }]}]}]}))
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source)
    assert caught.value.code == "missing_traces"
    source.write_text(json.dumps({"resourceMetrics": []}))
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source, format="otlp")
    assert caught.value.code == "unsupported_otlp"


def test_generic_otlp_is_not_silently_claimed_as_claude(tmp_path):
    spans = _session(1)
    for span in spans:
        span["name"] = "other.producer"
    source = _write(tmp_path, spans)
    with pytest.raises(tv.LoadError) as caught:
        tv.load(source)
    assert caught.value.code == "unsupported_producer"
