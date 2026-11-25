"""Semantic categories preserve source identity and partition recorded evidence."""

from __future__ import annotations

import copy
import json

import pytest

from agent_trajviz.insight.prompt_components import CATEGORIES, composition_summary
from agent_trajviz.insight.prompt_construction import _components, inspect_prompt_construction


def detail(**extra):
    return {"id": "claude:session:request", "scope": "recorded_request", "evidence": "reconstructed",
            "session_id": "session", "agent_id": "", "system": [], "tools": [], "messages": [], **extra}


def pointer(value, path):
    for key in path.split("/")[1:]:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def test_semantic_segments_partition_escaped_json_and_cjk_with_original_pointers():
    raw = detail(messages=[{"role": "user", "content": [
        {"type": "text", "text": "ordinary \\\" text 中文"},
        {"type": "text", "text": "# AGENTS.md instructions\nUse these rules"},
        {"type": "text", "text": "<skill>\nRecorded skill content\n</skill>"},
        {"type": "image", "data": "secret image"},
        {"type": "text", "text": "last original index"}]}])
    original = copy.deepcopy(raw)
    components, estimate = _components(raw)
    component = components[0]
    assert component["role"] == "user" and component["category"] == "conversation"
    segments = component["segments"]
    assert sum(segment["estimated_tokens"] for segment in segments) == component["estimated_tokens"]
    assert {segment["category"] for segment in segments} >= {"rules", "skills", "conversation"}
    content = [segment for segment in segments if segment["kind"] == "content"]
    assert [segment["json_pointer"] for segment in content] == ["/content/0", "/content/1", "/content/2", "/content/4"]
    assert all(pointer(component["content"], segment["json_pointer"]) for segment in content)
    spans = [segment["serialized_span"] for segment in segments]
    assert spans[0]["start"] == 0 and all(left["end"] == right["start"] for left, right in zip(spans, spans[1:], strict=False))
    summary = composition_summary(components, estimate)
    assert summary["total"] == estimate["total"] and summary["partial"]
    assert sum(row["estimated_tokens"] for row in summary["categories"]) == estimate["total"]
    assert "secret image" not in json.dumps(summary)
    assert raw == original


def test_native_identity_version_and_occurrence_are_independent_and_append_stable():
    first = detail(id="pi:session:a:2", scope="native_prompt_state", branch_id="a",
                   system=[{"type": "text", "section": "rules", "text": "Original rule"}],
                   system_refs=[{"source_path": "session.jsonl", "record_id": "s", "raw_index": 1}])
    second = copy.deepcopy(first)
    second.update(id="pi:session:b:4", branch_id="b")
    second["system"][0]["text"] = "Updated rule"
    second["system_refs"][0].update(record_id="patch", raw_index=3)
    before = _components(first)[0][0]
    after = _components(second)[0][0]
    assert before["identity"] == after["identity"] == "pi:system:rules"
    assert before["logical_id"] == after["logical_id"]
    assert before["content_hash"] != after["content_hash"]
    assert before["occurrence_id"] != after["occurrence_id"]
    assert after["category"] == "rules" and after["category_basis"] == "recorded_section_name"
    assert after["provenance"][0]["record_id"] == "patch"


def test_recorded_tool_association_classifies_skill_mcp_outputs_without_renaming_roles():
    raw = detail(messages=[
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "r", "name": "Read", "input": {"file_path": "/skills/pdf/SKILL.md"}},
            {"type": "tool_use", "id": "m", "name": "mcp__files__read", "input": {}},
            {"type": "tool_use", "id": "s", "name": "spawn_agent", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "r", "content": "Recorded skill"},
            {"type": "tool_result", "tool_use_id": "m", "content": "Recorded MCP result"},
            {"type": "tool_result", "tool_use_id": "s", "content": "Recorded child result"}]}])
    components, _estimate = _components(raw)
    results = [segment for segment in components[1]["segments"] if segment["kind"] == "content"]
    assert [segment["category"] for segment in results] == ["skills", "mcp", "tool_outputs"]
    assert all(segment["role"] == "user" for segment in results)
    assert results[0]["category_basis"] == "inferred_recorded_skill_file_read"
    assert "subagents" not in {segment["category"] for component in components for segment in component["segments"]}


def test_message_identity_does_not_claim_stable_identity_for_shifted_content_blocks():
    first = detail(id="pi:session:a:3", scope="native_prompt_state", messages=[{
        "role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}],
        message_refs=[{"entry_id": "user", "origin": "message", "record_id": "user", "source_path": "run.jsonl"}])
    second = copy.deepcopy(first)
    second["id"] = "pi:session:b:5"
    second["messages"][0]["content"].insert(0, {"type": "text", "text": "Inserted"})
    before = _components(first)[0][0]
    after = _components(second)[0][0]
    assert before["logical_id"] == after["logical_id"] and before["logical_id"]
    before_segments = [part for part in before["segments"] if part["kind"] == "content"]
    after_segments = [part for part in after["segments"] if part["kind"] == "content"]
    assert pointer(before["content"], before_segments[0]["json_pointer"])["text"] == "A"
    assert pointer(after["content"], after_segments[0]["json_pointer"])["text"] == "Inserted"
    assert all(part["logical_id"] is None and part["identity_basis"] == "unavailable"
               for part in before_segments + after_segments)
    assert set(part["occurrence_id"] for part in before_segments).isdisjoint(
        part["occurrence_id"] for part in after_segments)


def test_pi_named_tool_result_uses_matching_recorded_read_arguments_for_skill_category():
    raw = detail(messages=[
        {"role": "assistant", "content": [{"type": "toolCall", "id": "read-skill", "name": "Read",
                                             "arguments": {"path": "/skills/testing/SKILL.md"}}]},
        {"role": "toolResult", "toolCallId": "read-skill", "toolName": "read",
         "content": [{"type": "text", "text": "Recorded testing instructions"}]}])
    component = _components(raw)[0][-1]
    assert component["role"] == "toolResult" and component["category"] == "skills"
    assert component["category_basis"] == "inferred_recorded_skill_file_read"
    assert all(segment["category"] == "skills" for segment in component["segments"])
    # A different recorded name cannot use Read's path to claim a skill source.
    raw["messages"][-1]["toolName"] = "shell"
    assert _components(raw)[0][-1]["category"] == "tool_outputs"


def test_future_or_ambiguous_tool_calls_never_supply_result_classification():
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "Output"}]}
    call = {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "mcp__files__read"}]}
    future = _components(detail(messages=[result, call]))[0]
    assert next(segment for segment in future[0]["segments"] if segment["kind"] == "content")["category"] == "tool_outputs"
    other = copy.deepcopy(call)
    other["content"][0]["name"] = "Skill"
    ambiguous = _components(detail(messages=[call, other, result]))[0]
    assert next(segment for segment in ambiguous[-1]["segments"] if segment["kind"] == "content")["category"] == "tool_outputs"


def test_plain_mentions_do_not_become_instruction_definitions_and_scalar_location_resolves():
    raw = detail(system="Ordinary system text", messages=[{"role": "user", "content": "Please explain rules and skills"}])
    components, _estimate = _components(raw)
    assert components[0]["location"]["json_pointer"] == "/system"
    assert pointer(raw, components[0]["location"]["json_pointer"]) == components[0]["content"]
    assert {segment["category"] for segment in components[1]["segments"]} == {"conversation"}
    assert all(component["provenance"] for component in components)
    assert all(component["logical_id"] is None for component in components)


def test_transcript_sources_are_aligned_stable_and_preserve_tool_part_locations():
    steps = [
        {"index": 0, "raw_index": 5, "message_id": "user-id", "source_path": "run.jsonl", "role": "user",
         "parts": [{"type": "text", "text": "Question"}]},
        {"index": 1, "raw_index": 6, "source_path": "run.jsonl", "role": "assistant", "time_created_ms": 2,
         "tool_calls": [{"tool_id": "c", "tool_name": "Read", "input": {"path": "/tmp/SKILL.md"},
                         "output": "Recorded skill text", "time_start": 3, "time_end": 4}]},
        {"index": 2, "role": "assistant", "source_path": "run.jsonl", "time_created_ms": 5},
    ]
    last = inspect_prompt_construction({}, steps)["selected_request"]
    first = inspect_prompt_construction({}, steps, last["previous_candidate_id"])["selected_request"]
    assert first["components"][0]["logical_id"] == last["components"][0]["logical_id"]
    assert last["components"][0]["provenance"][0]["raw_index"] == 5
    result = last["components"][-1]
    segment = next(segment for segment in result["segments"] if segment["kind"] == "content")
    assert segment["category"] == "skills"
    assert segment["provenance"][0]["value_field"] == "output"
    assert segment["provenance"][0]["index"] == 0


def test_normalized_identity_survives_timeline_reindexing_with_same_recorded_source():
    first = detail(context_id="candidate:main", messages=[{"role": "user", "content": "Task"}],
                   message_refs=[{"record": "normalized_step", "source_path": "run.jsonl",
                                  "raw_index": 5, "step_index": 2, "field": "message"}])
    second = copy.deepcopy(first)
    second["id"] = "transcript:later-boundary"
    second["message_refs"][0]["step_index"] = 7
    before = _components(first)[0][0]
    after = _components(second)[0][0]
    assert before["logical_id"] == after["logical_id"]
    assert before["occurrence_id"] != after["occurrence_id"]


def test_semantic_expansion_is_bounded_without_losing_total_or_original_content():
    raw = detail(messages=[{"role": "user", "content": [{"type": "text", "text": str(index)} for index in range(500)]}])
    components, estimate = _components(raw)
    component = components[0]
    assert component["segmentation_partial"] and estimate["partial"]
    assert len(component["segments"]) <= 257
    assert len(component["content"]["content"]) == 500
    assert sum(segment["estimated_tokens"] for segment in component["segments"]) == component["estimated_tokens"]
    assert [category for category, _label, _color in CATEGORIES] == [row["category"] for row in composition_summary(components, estimate)["categories"]]


@pytest.mark.parametrize("name", [{"invalid": "name"}, ["mcp__files__read"]])
def test_malformed_native_tool_names_keep_evidence_without_inventing_identity_or_category(name):
    raw = {"_claude_prompt_records": [
        {"session_id": "session", "agent_id": "", "record": {
            "type": "api-request-shape", "shapeHash": "shape", "shape": {
                "system": [], "tools": [{"name": name}]}}},
        {"session_id": "session", "agent_id": "", "record": {
            "type": "api-request", "id": "request", "shapeHash": "shape", "tail": []}},
    ]}
    selected = inspect_prompt_construction(raw, [])["selected_request"]
    component = selected["components"][0]
    assert component["content"]["name"] == name
    assert component["logical_id"] is None and component["identity_basis"] == "unavailable"
    assert component["category"] == "tools"
    assert component["segments"][0]["category"] == "tools"
    assert sum(segment["estimated_tokens"] for segment in component["segments"]) == component["estimated_tokens"]


@pytest.mark.parametrize("key", ["tool_use_id", "toolCallId", "call_id"])
@pytest.mark.parametrize("identifier", [{"invalid": "id"}, ["call"]])
def test_malformed_result_identifiers_do_not_crash_or_borrow_recorded_call_semantics(key, identifier):
    raw = detail(messages=[
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call", "name": "Skill"}]},
        {"role": "user", "content": [{"type": "tool_result", key: identifier, "content": "Recorded result"}]},
    ])
    components, estimate = _components(raw)
    result = next(segment for segment in components[-1]["segments"] if segment["kind"] == "content")
    assert result["category"] == "tool_outputs"
    assert components[-1]["content"]["content"][0][key] == identifier
    assert composition_summary(components, estimate)["total"] == estimate["total"]
