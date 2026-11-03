"""AWE case defaults come from declared session identity, never upload cache names."""
import pytest

from agent_trajviz.insight.ui.attribution_tab import _diagnosis_identity


def raw(fmt="codex", path="/private/tmp/gradio/0123456789abcdef/task-17.jsonl"):
    return {"_source_format": fmt, "_source_path": path, "_source_sha256": "displayed-source-sha"}


@pytest.mark.parametrize("fmt", ["codex", "claude_code", "opencode", "cursor", "pi", "dsh"])
def test_native_format_supplies_agent_for_uploaded_session(fmt):
    source = raw(fmt)
    assert _diagnosis_identity(source) == (source["_source_path"], fmt, fmt, "task-17")


def test_declared_canonical_agent_precedes_native_format():
    source = raw("codex", "/benchmark/data/trajectory/custom-eval-agent/task-17.jsonl")
    assert _diagnosis_identity(source) == (source["_source_path"], "codex", "custom-eval-agent", "task-17")


def test_explicit_overrides_precede_detected_defaults():
    source = raw("codex", "/benchmark/data/trajectory/codex/task-17.jsonl")
    assert _diagnosis_identity(source, " custom-agent ", " matching-task ") == (
        source["_source_path"], "codex", "custom-agent", "matching-task")


@pytest.mark.parametrize("source,fmt", [({"_source_path": "/tmp/gradio/abcdef12/upload.json"}, None),
                                        (raw("otlp"), "otlp")])
def test_unknown_or_telemetry_format_never_uses_an_arbitrary_cache_folder_as_agent(source, fmt):
    _, detected, agent, _ = _diagnosis_identity(source)
    assert detected == fmt and agent == ""


def test_missing_source_still_uses_format_without_inventing_task():
    assert _diagnosis_identity({"_source_format": "codex"}) == ("", "codex", "codex", "")
    assert _diagnosis_identity(None) == ("", None, "", "")


def test_parent_named_like_an_agent_is_not_treated_as_a_declared_evaluation_identity():
    source = raw("opencode", "/tmp/codex/task-17.json")
    assert _diagnosis_identity(source)[2] == "opencode"


def test_autoload_publishes_fresh_defaults_and_preserves_source_integrity_gate(monkeypatch):
    from agent_trajviz.insight import attribution
    from agent_trajviz.insight.insight import build_ui

    calls = []

    def diagnose(**kwargs):
        calls.append(kwargs)
        return attribution.AttributionResult(False, reason="Matching reference data required.")

    monkeypatch.setattr(attribution, "diagnose", diagnose)
    app = build_ui()
    handler = next(block.fn for block in app.fns.values()
                   if getattr(block.fn, "__name__", "") == "on_diagnose_autoload")
    first = raw("codex", "/tmp/gradio/hash-one/task-one.jsonl")
    first["_source_merged_count"] = 2
    assert handler(first, "/reference")[2:] == ("codex", "task-one")
    assert calls[-1] == {"agent": "codex", "instance_id": "task-one", "source_path": first["_source_path"],
                         "fmt": "codex", "expected_sha": "displayed-source-sha", "merged_sources": 2,
                         "reference_root": "/reference"}
    second = raw("opencode", "/tmp/gradio/hash-two/task-two.json")
    assert handler(second, "/reference")[2:] == ("opencode", "task-two")
    assert calls[-1]["agent"] == "opencode" and calls[-1]["instance_id"] == "task-two"
    assert handler({}, "/reference")[2:] == ("", "")
