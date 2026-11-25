"""Claude request evidence stays separate from conversation and accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.formats.claude_code import convert_claude_code_events
from agent_trajviz.insight.prompt_claude import inspect_claude


def transcript(*records):
    return [{"type": "user", "sessionId": "s", "uuid": "u", "message": {"role": "user", "content": "Task"}},
            *records,
            {"type": "assistant", "sessionId": "s", "uuid": "a", "message": {
                "role": "assistant", "id": "response", "model": "test", "content": "Done",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }}]


def shape(**extra):
    return {"type": "api-request-shape", "sessionId": "s", "shapeHash": "shape", "shape": {
        "system": [{"type": "text", "text": "System", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Read", "input_schema": {"type": "object", "properties": {
            "api_key": {"type": "string"}}}}],
    }, **extra}


def blob(identity, text, **extra):
    return {"type": "api-request-blob", "sessionId": "s", "hash": identity,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]}, **extra}


def request(identity, tail, **extra):
    return {"type": "api-request", "id": identity, "sessionId": "s", "shapeHash": "shape",
            "params": {"model": "test", "max_tokens": 100}, "tail": tail, **extra}


def inspect(*records, selection=None):
    raw = convert_claude_code_events(transcript(*records), source_path="session.jsonl")
    return inspect_claude(raw, raw["_claude_code_parsed_steps"], selection)


def test_explicit_snapshot_retains_ordered_blocks_parameters_and_sources():
    value = inspect(shape(), blob("b", "First"), blob("c", "Second"), request("r", ["b", "c"]))
    assert value["available"] and len(value["requests"]) == 1
    detail = value["selected_request"]
    assert detail["evidence"] == "reconstructed" and detail["scope"] == "recorded_request"
    assert detail["missing_references"] == []
    assert [m["content"][0]["text"] for m in detail["messages"]] == ["First", "Second"]
    assert detail["message_indices"] == [0, 1]
    assert detail["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert detail["tools"][0]["input_schema"]["properties"]["api_key"] == {"type": "string"}
    assert detail["parameters"] == {"model": "test", "max_tokens": 100}
    assert {source["kind"] for source in detail["sources"]} == {"shape", "blob", "request"}
    assert detail["step_index"] is None  # A response association is not invented.
    assert all(source["source_path"] == "session.jsonl" for source in detail["sources"])


def test_explicit_base_prefix_replay_and_reset_keep_zero():
    value = inspect(shape(), blob("a", "A"), blob("b", "B"), blob("c", "C"),
                    request("one", ["a", "b"]), request("two", ["c"], base="one", keep=1),
                    request("three", ["b"], base="two", keep=0))
    row = value["requests"][1]
    second = inspect(shape(), blob("a", "A"), blob("b", "B"), blob("c", "C"),
                     request("one", ["a", "b"]), request("two", ["c"], base="one", keep=1),
                     selection=row["id"])["selected_request"]
    assert [m["content"][0]["text"] for m in second["messages"]] == ["A", "C"]
    assert second["previous_request_id"] == value["requests"][0]["id"]
    assert any("not been independently validated" in line for line in second["limitations"])
    assert value["selected_request"]["previous_request_id"] == value["requests"][1]["id"]
    assert [m["content"][0]["text"] for m in value["selected_request"]["messages"]] == ["B"]


def test_prefix_does_not_inherit_discarded_message_gaps_or_previous_shape():
    value = inspect(blob("a", "A"), request("one", ["a", "missing"], shapeHash="old-missing"),
                    shape(), request("two", [], base="one", keep=1))
    assert set(value["requests"][0]["missing_references"]) == {"shape:old-missing", "blob:missing"}
    assert value["selected_request"]["missing_references"] == []
    assert [m["content"][0]["text"] for m in value["selected_request"]["messages"]] == ["A"]


def test_kept_missing_message_remains_partial_after_a_second_delta():
    value = inspect(shape(), blob("a", "A"), request("one", ["missing", "a"]),
                    request("two", ["a"], base="one", keep=1),
                    request("three", [], base="two", keep=1))
    assert value["selected_request"]["messages"] == []
    assert value["selected_request"]["missing_references"] == ["blob:missing"]


def test_evolution_replays_only_explicit_baseline_and_distinguishes_replaced_tail():
    changed = shape(shapeHash="changed", shape={"system": "Updated instructions", "tools": [
        {"name": "Edit", "input_schema": {"type": "object"}},
    ]})
    value = inspect(shape(), blob("a", "A"), blob("b", "B"), blob("c", "C"),
                    request("one", ["a", "b"]), changed,
                    request("two", ["c"], base="one", keep=1, shapeHash="changed"))
    detail = value["selected_request"]
    kinds = {event["kind"] for event in detail["prompt_events"]}
    assert {"system_changed", "tools_changed", "context_prefix_retained", "message_tail_replaced"} <= kinds
    tools = next(event for event in detail["prompt_events"] if event["kind"] == "tools_changed")
    assert tools["added_tools"] == ["Edit"] and tools["removed_tools"] == ["Read"]
    tool_events = [(event["kind"], event["target_id"]) for event in detail["prompt_events"]
                   if event["kind"] in {"tool_added", "tool_removed"}]
    assert tool_events == [("tool_added", "Edit"), ("tool_removed", "Read")]
    transition = detail["transition"]
    assert transition["baseline_evidence"] == "explicit_base"
    assert transition["retained_message_count"] == 1 and transition["known_tail_message_count"] == 1
    assert transition["removed_message_count"] == 1 and transition["added_message_count"] == 1
    assert transition["reset"] is False
    assert all(event["source_refs"] for event in detail["prompt_events"])


def test_evolution_reset_uses_explicit_base_without_retaining_its_messages():
    value = inspect(shape(), blob("a", "A"), blob("b", "B"), request("one", ["a", "b"]),
                    request("two", ["b"], base="one", keep=0))
    detail = value["selected_request"]
    assert detail["transition"]["reset"] is True
    assert detail["transition"]["retained_message_count"] == 0
    assert detail["transition"]["removed_message_count"] == 2
    assert any(event["kind"] == "context_reset" for event in detail["prompt_events"])
    assert [m["content"][0]["text"] for m in detail["messages"]] == ["B"]


def test_evolution_does_not_guess_changes_from_neighboring_requests():
    changed = shape(shapeHash="changed", shape={"system": "Updated", "tools": []})
    value = inspect(shape(), blob("a", "A"), request("one", ["a"]), changed,
                    request("two", ["a"], keep=1, shapeHash="changed"))
    detail = value["selected_request"]
    assert detail["transition"]["baseline_evidence"] == "unavailable"
    assert detail["transition"]["retained_message_count"] is None
    assert detail["transition"]["added_message_count"] is None
    assert detail["transition"]["known_tail_message_count"] == 1
    assert not any(event["kind"] in {"system_changed", "tools_changed", "context_reset"}
                   for event in detail["prompt_events"])
    assert any(event["kind"] == "baseline_unavailable" for event in detail["prompt_events"])


def test_implicit_previous_request_is_never_assumed():
    value = inspect(shape(), blob("a", "A"), blob("b", "B"), request("one", ["a"]),
                    request("two", ["b"], keep=1))
    detail = value["selected_request"]
    assert "base:unrecorded" in detail["missing_references"]
    assert detail["message_indices"] == [1]
    assert [m["content"][0]["text"] for m in detail["messages"]] == ["B"]
    assert "previous_request_id" not in detail


def test_future_references_do_not_backfill_past_requests():
    value = inspect(request("early", ["future"]), shape(), blob("future", "Later"), request("late", ["future"]))
    early = inspect(request("early", ["future"]), shape(), blob("future", "Later"),
                    selection=value["requests"][0]["id"])["selected_request"]
    assert early["messages"] == [] and early["system"] == []
    assert set(early["missing_references"]) == {"shape:shape", "blob:future"}
    assert value["selected_request"]["missing_references"] == []


def test_conflicting_hash_is_ambiguous_only_from_redefinition_onward():
    value = inspect(shape(), blob("a", "Original"), request("before", ["a"]),
                    blob("a", "Changed"), request("after", ["a"]))
    assert value["requests"][0]["missing_references"] == []
    assert value["selected_request"]["messages"] == []
    assert "blob:a" in value["selected_request"]["missing_references"]
    assert inspect(shape(), shape(), blob("a", "Same"), blob("a", "Same"), request("r", ["a"]))[
        "selected_request"]["missing_references"] == []


def test_child_identity_and_dictionaries_are_isolated():
    parent = transcript(shape(), blob("a", "Parent"), request("same", ["a"]))
    children = {"child": {"source_path": "session/subagents/agent-child.jsonl", "events":
                           transcript(request("same", ["a"], base="same", keep=1))}}
    raw = convert_claude_code_events(parent, source_path="session.jsonl", child_event_lists=children)
    value = inspect_claude(raw, raw["_claude_code_parsed_steps"])
    assert len(value["requests"]) == 2 and len({row["id"] for row in value["requests"]}) == 2
    child = value["selected_request"]
    assert child["agent_id"] == "child" and child["source_path"].endswith("agent-child.jsonl")
    assert child["messages"] == []
    assert set(child["missing_references"]) == {"shape:shape", "blob:a", "base:same"}
    assert "previous_request_id" not in child


def test_session_identity_isolates_hashes_even_for_same_agent():
    value = inspect(shape(), blob("a", "Parent"), request("one", ["a"]),
                    request("two", ["a"], sessionId="foreign"))
    assert value["selected_request"]["messages"] == []
    assert set(value["selected_request"]["missing_references"]) == {"shape:shape", "blob:a"}


def test_duplicate_native_ids_disable_ambiguous_future_lineage():
    value = inspect(shape(), blob("a", "A"), request("same", ["a"]), request("same", []),
                    request("last", [], base="same", keep=1))
    assert len({row["id"] for row in value["requests"]}) == 3
    assert "base:same" in value["selected_request"]["missing_references"]


@pytest.mark.parametrize("keep", [-1, True, 1.5, 10**20])
def test_malformed_prefix_is_explicit_and_bounded(keep):
    value = inspect(shape(), blob("a", "A"), request("r", ["a"], keep=keep))
    assert "prefix:invalid" in value["selected_request"]["missing_references"]
    assert len(value["selected_request"]["messages"]) == 1


def test_unknown_fields_and_missing_shapes_are_visible():
    incomplete = shape(shape={"system": [], "tools": [], "unrecognized_transform": {"value": "unknown"}})
    value = inspect(incomplete, request("r", [], future_delta=1))
    detail = value["selected_request"]
    assert detail["unsupported_fields"] == ["future_delta", "shape.unrecognized_transform"]
    assert any("Unsupported request or shape fields" in line for line in detail["limitations"])
    assert inspect(request("r", []))["selected_request"]["missing_references"] == ["shape:shape"]


def test_malformed_record_types_roles_shapes_and_payloads_stay_partial():
    value = inspect({"type": ["unexpected"]},
                    shape(shape={"system": {"unsupported": True}, "tools": "invalid"}),
                    blob("bad", "Unused", message={"role": [], "content": "Invalid"}),
                    blob("missing", "Unused", message={"role": "user"}),
                    request("r", ["bad", "missing"], params={"model": "test", "input": "unhandled"}))
    detail = value["selected_request"]
    assert detail["system"] == [] and detail["tools"] == [] and detail["messages"] == []
    assert set(detail["missing_references"]) == {"shape:shape", "blob:bad", "blob:missing"}
    assert detail["unsupported_fields"] == ["params.input"]


def test_primitive_shape_blocks_do_not_escape_the_detail_schema():
    value = inspect(shape(shape={"system": ["not a block"], "tools": [123]}), request("r", []))
    detail = value["selected_request"]
    assert detail["system"] == [] and detail["tools"] == []
    assert detail["missing_references"] == ["shape:shape"]


def test_transport_credentials_are_omitted_without_changing_prompt_body():
    native = request("r", ["a"], headers={"Authorization": "never-export"},
                     params={"model": "test", "api_key": "never-export", "headers": {"x-api-key": "never-export"},
                             "safeguards": {"private_extra": "not-a-model-option"}})
    raw = convert_claude_code_events(transcript(shape(), blob("a", "Literal api_key text"), native))
    assert "never-export" not in json.dumps(raw)
    detail = inspect_claude(raw, raw["_claude_code_parsed_steps"])["selected_request"]
    assert detail["parameters"] == {"model": "test"}
    assert detail["messages"][0]["content"][0]["text"] == "Literal api_key text"
    assert any("credentials and headers are omitted" in line for line in detail["limitations"])
    assert "private_extra" not in json.dumps(detail)


def test_shape_transport_boundary_omits_credentials_but_preserves_tool_schema():
    record = shape()
    record["shape"]["headers"] = {"Authorization": "never-export-shape"}
    raw = convert_claude_code_events(transcript(record, request("r", [])))
    assert "never-export-shape" not in json.dumps(raw)
    detail = inspect_claude(raw, raw["_claude_code_parsed_steps"])["selected_request"]
    assert detail["tools"][0]["input_schema"]["properties"]["api_key"] == {"type": "string"}
    assert any("credentials and headers are omitted" in line for line in detail["limitations"])


def test_current_fixture_and_export_roundtrip_do_not_change_usage_or_steps():
    path = Path(__file__).parent / "fixtures/claude_current/session-current.jsonl"
    run = tv.load(path)
    assert len(run.steps) == 6 and run.metrics["tokens"]["total"] == 323
    assert run.metrics["tool_call_count"] == 2
    assert len(run.raw["_claude_prompt_records"]) == 4
    value = inspect_claude(run.raw, run.steps)
    assert len(value["requests"]) == 2
    restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=True))))
    assert inspect_claude(restored.raw, restored.steps) == value
    copied = copy.deepcopy(value)
    value["selected_request"]["messages"].append({"role": "user", "content": "mutated output"})
    assert inspect_claude(run.raw, run.steps) == copied


def test_unknown_selection_and_unavailable_legacy_data():
    assert inspect()["available"] is False
    with pytest.raises(ValueError, match="Unknown prompt construction request_id"):
        inspect(selection="nonexistent")
    with pytest.raises(ValueError):
        inspect(shape(), request("r", []), selection="nonexistent")
    with pytest.raises(ValueError):
        inspect(shape(), request("r", []), selection="")


def test_long_explicit_chain_materializes_selected_input_without_expanding_raw():
    records = [shape(), blob("a", "A"), request("r0", ["a"])]
    records.extend(request(f"r{i}", ["a"], base=f"r{i-1}", keep=i) for i in range(1, 1500))
    raw = convert_claude_code_events(transcript(*records))
    value = inspect_claude(raw, raw["_claude_code_parsed_steps"])
    assert len(value["requests"]) == 1500
    assert len(value["selected_request"]["messages"]) == 1500
    assert len(raw["_claude_prompt_records"]) == len(records)
    assert len(json.dumps(raw)) < 1_000_000


@pytest.mark.parametrize("missing", [False, True])
def test_fixed_length_chains_reuse_bounded_prefix_inspection(monkeypatch, missing):
    from agent_trajviz.insight import prompt_claude

    monkeypatch.setattr(prompt_claude, "_MAX_PREFIX_WALK", 4)
    records = [shape(), blob("a", "A"), request("r0", ["missing" if missing else "a", "a"])]
    records.extend(request(f"r{i}", ["a"], base=f"r{i-1}", keep=1) for i in range(1, 2500))
    value = inspect(*records)
    assert value["selected_request"]["missing_references"] == (["blob:missing"] if missing else [])
    assert "prefix:inspection-limit" not in value["selected_request"]["missing_references"]


def test_uncached_adversarial_prefix_walk_is_bounded_and_explicit(monkeypatch):
    from agent_trajviz.insight import prompt_claude

    monkeypatch.setattr(prompt_claude, "_MAX_PREFIX_WALK", 4)
    records = [shape(), request("r0", ["missing"])]
    records.extend(request(f"r{i}", ["missing"], base=f"r{i-1}", keep=i) for i in range(1, 20))
    records.append(request("last", [], base="r19", keep=1))
    detail = inspect(*records)["selected_request"]
    assert "prefix:inspection-limit" in detail["missing_references"]
    assert any("safety limit" in line for line in detail["limitations"])


def test_evidence_quality_separates_absent_blobs_from_unsupported_settings_and_estimation():
    value = inspect(shape(), blob("known", "Known input"),
                    request("initial", ["known"], cacheControl={"type": "ephemeral"}),
                    request("second", ["absent-one", "absent-two"], base="initial", keep=1),
                    request("third", ["absent-three"], base="second", keep=3))
    initial, second, third = value["requests"]
    quality = initial["evidence_quality"]
    assert quality["status"] == "partial" and quality["content_complete"]
    assert quality["missing_references"] == []
    assert quality["unsupported_settings"] == ["cacheControl"]
    assert quality["estimation"] == {"token_counts": "character_estimate", "provider_request_verified": False}
    second_issues = second["evidence_quality"]["missing_references"]
    assert {(issue["kind"], issue["reference"], issue["reason"]) for issue in second_issues} == {
        ("message", "absent-one", "not_recorded"), ("message", "absent-two", "not_recorded")}
    third_issues = third["evidence_quality"]["missing_references"]
    assert len(third_issues) == 3 and not third["evidence_quality"]["content_complete"]
    assert third_issues[:2] == second_issues  # Sources remain where the absent references originated.
    assert third_issues[0]["source_refs"][0]["reference"] == "second"
    assert third_issues[2]["source_refs"][0]["reference"] == "third"
    assert value["selected_request"]["evidence_quality"] == third["evidence_quality"]
    assert value["selected_request"]["messages"][0]["content"][0]["text"] == "Known input"


def test_evidence_quality_distinguishes_invalid_conflicting_and_future_definitions():
    value = inspect(request("early", ["future"]), shape(), blob("future", "Future input"),
                    blob("invalid", "Unused", message={"role": "user"}),
                    blob("conflict", "First input"), blob("conflict", "Second input"),
                    request("late", ["future", "invalid", "conflict"]))
    early_quality = value["requests"][0]["evidence_quality"]
    assert {(issue["kind"], issue["reason"]) for issue in early_quality["missing_references"]} == {
        ("shape", "not_recorded"), ("message", "not_recorded")}
    quality = value["selected_request"]["evidence_quality"]
    assert {(issue["reference"], issue["reason"]) for issue in quality["missing_references"]} == {
        ("invalid", "invalid"), ("conflict", "conflicting")}
    collision = next(issue for issue in quality["missing_references"] if issue["reason"] == "conflicting")
    assert len(collision["source_refs"]) == 2
    assert len({source["raw_index"] for source in collision["source_refs"]}) == 2
    assert "First input" not in json.dumps(quality) and "Second input" not in json.dumps(quality)
    assert [message["content"][0]["text"] for message in value["selected_request"]["messages"]] == ["Future input"]


@pytest.mark.parametrize("record,kind,reason", [
    (request("r", [], base="absent", keep=1), "base", "not_recorded"),
    (request("r", [], keep=-1), "prefix", "invalid"),
    (request("r", "invalid"), "message", "invalid"),
])
def test_evidence_quality_distinguishes_missing_lineage_and_invalid_record_shapes(record, kind, reason):
    quality = inspect(shape(), record)["selected_request"]["evidence_quality"]
    assert not quality["content_complete"]
    assert any(issue["kind"] == kind and issue["reason"] == reason for issue in quality["missing_references"])


def test_evidence_quality_marks_duplicate_base_conflicts_and_oversized_prefixes():
    value = inspect(shape(), blob("known", "Known input"), request("same", ["known"]),
                    request("same", []), request("r", [], base="same", keep=1))
    assert value["selected_request"]["evidence_quality"]["missing_references"][0]["reason"] == "conflicting"
    oversized = inspect(shape(), request("first", []), request("second", [], base="first", keep=1))
    issue = oversized["selected_request"]["evidence_quality"]["missing_references"][0]
    assert issue["kind"] == "base" and issue["reason"] == "invalid"


def test_evidence_quality_discards_unretained_gaps_and_bounds_details(monkeypatch):
    from agent_trajviz.insight import prompt_claude

    value = inspect(shape(), blob("known", "Known input"), request("first", ["known", "absent"]),
                    request("second", [], base="first", keep=1))
    assert value["selected_request"]["evidence_quality"]["content_complete"]
    assert value["selected_request"]["evidence_quality"]["missing_references"] == []
    monkeypatch.setattr(prompt_claude, "_MAX_MISSING", 3)
    many = inspect(shape(), request("r", [f"absent-{index}" for index in range(20)]))
    issues = many["selected_request"]["evidence_quality"]["missing_references"]
    assert len(issues) == 4
    assert issues[-1]["kind"] == "catalog" and issues[-1]["reason"] == "inspection_limit"
    assert all(len(issue["source_refs"]) <= 3 for issue in issues)


def test_evidence_quality_reports_transport_omission_without_missing_prompt_content():
    quality = inspect(shape(headers={"authorization": "placeholder"}), blob("known", "Known input"),
                      request("r", ["known"]))["selected_request"]["evidence_quality"]
    assert quality["status"] == "complete" and quality["content_complete"]
    assert quality["transport_omitted"] and not quality["missing_references"]
    assert "placeholder" not in json.dumps(quality)


def test_evidence_quality_transport_flag_follows_only_retained_blob_prefixes():
    value = inspect(shape(), blob("clean", "Clean input"),
                    blob("omitted", "Recorded input", headers={"authorization": "placeholder"}),
                    request("first", ["clean", "omitted"]),
                    request("retained", [], base="first", keep=2),
                    request("discarded", [], base="retained", keep=1),
                    request("reset", ["clean"], base="retained", keep=0))
    assert [row["evidence_quality"]["transport_omitted"] for row in value["requests"]] == [True, True, False, False]
    assert all(row["evidence_quality"]["content_complete"] for row in value["requests"])
    assert "placeholder" not in json.dumps(value)
