"""Literal prompt text previews do not infer assembly, rewrites or repetition."""

from __future__ import annotations

import copy
import json

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.insight.prompt_construction import inspect_prompt_construction
from agent_trajviz.insight.prompt_text import build_text_evolution, readable_component_blocks


def component(identifier, content, *, kind="message", role="assistant"):
    return {"id": identifier, "kind": kind, "role": role, "content": content,
            "source_refs": [{"request_id": "r", "field": "messages", "index": 0}]}


def message(*content):
    return {"role": "assistant", "content": list(content)}


def tool(identifier, arguments, *, name="shell"):
    return {"type": "tool_use", "id": identifier, "name": name, "input": arguments}


def evolution(before, after, *, rewrites=()):
    return build_text_evolution({"evidence": "estimated", "scope": "trajectory_estimate", "components": after,
                                 "changes": {"available": True, "evidence": "estimated",
                                             "basis": "adjacent_transcript_candidates",
                                             "added": [part["id"] for part in after], "removed": [],
                                             "rewritten_components": list(rewrites)}}, before)


def step(index, role, text="", **extra):
    return {"index": index, "session_id": "s", "role": role, "tool_calls": [],
            "parts": [{"type": "text", "text": text}] if text else [], "time_created_ms": index * 100, **extra}


def test_readable_plain_text_code_tool_arguments_and_results_keep_literals():
    markup = "<script>literal('not executed')</script> 中文"
    source = component("c", message({"type": "text", "text": markup},
                                    tool("call", {"command": "python -m pytest", "content": "def f():\n    return '<b>'"}),
                                    {"type": "tool_result", "tool_use_id": "call", "content": "TypeError: config_"}))
    original = copy.deepcopy(source)
    blocks = readable_component_blocks(source)
    assert [item["kind"] for item in blocks] == ["text", "tool_arguments", "tool_arguments", "tool_result"]
    assert blocks[0]["text"] == markup and blocks[0]["path"] == "/content/0/text"
    assert blocks[1]["text"] == "python -m pytest" and blocks[1]["tool_name"] == "shell"
    assert blocks[2]["text"] == "def f():\n    return '<b>'"
    assert blocks[3]["text"] == "TypeError: config_"
    assert all(item["source_refs"] == source["source_refs"] for item in blocks)
    assert source == original


@pytest.mark.parametrize("content", [
    {"type": "image", "source": {"type": "base64", "data": "SECRET_BASE64"}},
    {"type": "input_audio", "input_audio": {"data": "SECRET_BASE64"}},
    {"type": "text", "text": "data:image/png;base64,SECRET_BASE64"},
    {"type": "redacted_thinking", "data": "SECRET_BASE64"},
    {"type": "thinking", "thinking": "PRIVATE_REASONING"},
])
def test_opaque_payload_is_not_display_text(content):
    assert readable_component_blocks(component("c", message(content))) == []


def test_tool_fallback_json_omits_opaque_fields_but_keeps_actual_data_argument():
    source = component("c", message(tool("id", {"data": "ordinary argument", "image": {
        "type": "image", "source": {"data": "SECRET_BASE64"}}, "signature": "SECRET_SIGNATURE"})))
    text = readable_component_blocks(source)[0]["text"]
    assert "ordinary argument" in text and "SECRET" not in text


def test_repeat_ignores_only_transport_id_and_counts_earlier_blocks():
    before = [component("old1", message(tool("transport-old", {"command": "pytest", "timeout": 30}))),
              component("old2", message(tool("transport-old2", {"command": "pytest", "timeout": 30})))]
    after = [component("new", message(tool("transport-new", {"command": "pytest", "timeout": 30})))]
    block = evolution(before, after)["blocks"][0]
    assert block["change"] == "added" and block["repeated"] and block["repeat_count"] == 2
    assert block["repeat_kind"] == "same_tool_arguments"
    assert block["repeat_source_component_ids"] == ["old1", "old2"]
    assert not block["repeat_sources_truncated"]
    assert evolution([], after + after)["blocks"][0]["repeat_count"] == 0


@pytest.mark.parametrize("arguments", [
    {"command": "pytest", "timeout": 60}, {"command": "pytest ", "timeout": 30},
    {"command": "pytest", "timeout": 30, "id": "meaningful"},
])
def test_equal_command_is_not_equal_arguments(arguments):
    before = [component("old", message(tool("transport", {"command": "pytest", "timeout": 30})))]
    after = [component("new", message(tool("transport", arguments)))]
    assert not evolution(before, after)["blocks"][0]["repeated"]


def test_repeat_matches_full_text_not_truncated_prefix_and_results_ignore_transport():
    before = [component("old", message({"type": "text", "text": "a" * 7000 + "old"}))]
    after = [component("new", message({"type": "text", "text": "a" * 7000 + "new"}))]
    block = evolution(before, after)["blocks"][0]
    assert len(block["text"]) == 6000 and block["text_truncated"] and not block["repeated"]
    before = [component("old", message({"type": "tool_result", "tool_use_id": "id1", "content": "same result"}))]
    after = [component("new", message({"type": "tool_result", "tool_use_id": "id2", "content": "same result"}))]
    block = evolution(before, after)["blocks"][0]
    assert block["repeat_kind"] == "same_tool_result" and block["repeat_count"] == 1


def test_repeat_reference_cap_is_independent_of_matching_block_count():
    before = [component(f"old{i}", message(*[{"type": "text", "text": "same"}] * 2)) for i in range(10)]
    block = evolution(before, [component("new", message({"type": "text", "text": "same"}))])["blocks"][0]
    assert block["repeat_count"] == 20 and len(block["repeat_source_component_ids"]) == 8
    assert block["repeat_sources_truncated"]
    block = evolution(before[:1], [component("new", message({"type": "text", "text": "same"}))])["blocks"][0]
    assert block["repeat_count"] == 2 and len(block["repeat_source_component_ids"]) == 1
    assert not block["repeat_sources_truncated"]


def test_transcript_checkpoint_added_repeated_command_and_no_synthetic_preview():
    command = {"command": "python -m pytest tests/test_coalesce.py"}
    steps = [step(0, "user", "Implement coalescing."),
             step(1, "assistant", parts=[{"type": "tool", "tool": "shell"}], text_preview="[Tool: shell]",
                  tool_calls=[{"tool_id": "first", "tool_name": "shell", "input": command,
                               "output": "1 failed", "time_end": 150}]),
             step(2, "assistant", tool_calls=[{"tool_id": "second", "tool_name": "shell", "input": command,
                                                "output": "1 failed", "time_end": 250}]),
             step(3, "user", "Wrap-up checkpoint: stop repeating diagnostic checks."),
             step(4, "assistant", "OWN RESPONSE MUST STAY OUT")]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    changes = detail["text_evolution"]
    assert changes["evidence"] == "estimated" and changes["comparison_available"] and not changes["rewrites"]
    assert any(item["text"].startswith("Wrap-up checkpoint") and not item["repeated"] for item in changes["blocks"])
    assert any(item["kind"] == "tool_arguments" and item["repeated"] for item in changes["blocks"])
    assert any(item["kind"] == "tool_result" and item["repeated"] for item in changes["blocks"])
    assert "[Tool: shell]" not in json.dumps(detail) and "OWN RESPONSE MUST STAY OUT" not in json.dumps(detail)
    json.dumps(detail, allow_nan=False)


@pytest.mark.parametrize("foreign", [
    {"is_sub_agent": True, "agent": "reviewer"}, {"is_sub_agent": True}, {"branch_id": "other"},
])
def test_foreign_context_does_not_supply_repetition_or_text(foreign):
    steps = [step(0, "user", "Parent task"), step(1, "assistant", "Parent response"),
             step(2, "user", "SECRET repeated text", **foreign), step(3, "assistant", "foreign", **foreign),
             step(4, "user", "SECRET repeated text"), step(5, "assistant", "current")]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert len(detail["text_evolution"]["blocks"]) == 2
    assert all(not block["repeated"] for block in detail["text_evolution"]["blocks"])


def test_no_comparable_previous_request_means_uncompared_text():
    detail = inspect_prompt_construction({}, [step(0, "user", "Task"), step(1, "assistant", "Response")])["selected_request"]
    assert not detail["text_evolution"]["comparison_available"]
    assert detail["text_evolution"]["blocks"] == [] and detail["text_evolution"]["rewrites"] == []
    assert readable_component_blocks(detail["components"][0])[0]["text"] == "Task"


def test_native_proven_system_rewrite_has_before_after_not_duplicate_body():
    raw = {"_pi_entries": [{"type": "session", "version": 3, "id": "pi-session"},
        {"type": "message", "id": "s", "parentId": None, "message": {
            "role": "system", "content": [], "sections": {"preamble": "Old instruction"}}},
        {"type": "message", "id": "u", "parentId": "s", "message": {"role": "user", "content": "Task"}},
        {"type": "message", "id": "a", "parentId": "u", "message": {"role": "assistant", "content": "Old answer"}},
        {"type": "message", "id": "p", "parentId": "a", "message": {
            "role": "system", "content": [], "sections": {"preamble": "New instruction"}}},
        {"type": "message", "id": "b", "parentId": "p", "message": {"role": "assistant", "content": "Current answer"}}]}
    detail = inspect_prompt_construction(raw, [])["selected_request"]
    evolution = detail["text_evolution"]
    assert evolution["scope"] == "native_prompt_state" and evolution["evidence"] == "reconstructed"
    rewrite = evolution["rewrites"][0]
    assert rewrite["before_text"] == "Old instruction" and rewrite["after_text"] == "New instruction"
    assert not rewrite["before_truncated"] and not rewrite["after_truncated"]
    assert all(block["component_id"] not in {rewrite["before_id"], rewrite["after_id"]} for block in evolution["blocks"])
    assert rewrite["source_refs"] and "Current answer" not in json.dumps(detail)


def test_display_bounds_are_disclosed_and_public_view_schema_roundtrips(tmp_path):
    before = [component("old", message({"type": "text", "text": "Previous"}))]
    after = [component(f"new{i}", message({"type": "text", "text": "x" * 7000})) for i in range(100)]
    bounded = evolution(before, after)
    assert len(bounded["blocks"]) <= 80 and sum(len(block["text"]) for block in bounded["blocks"]) <= 120_000
    assert bounded["blocks_truncated"] and bounded["limitations"]
    run = tv.Trajectory.from_steps([step(0, "user", "First"), step(1, "assistant", "Earlier"),
                                    step(2, "user", "<literal>New</literal>"), step(3, "assistant", "Own")])
    view = run.view("prompt_construction")
    jsonschema.validate(json.loads(view.to_json()), tv.view_schema("prompt_construction"))
    tv.export_json(run, tmp_path, only=["prompt_construction"])
    restored = tv.load(tmp_path / "trajectory.json")
    assert restored.view("prompt_construction").data == view.data
    assert all("text_evolution" not in row for row in view.data["requests"] + view.data["evolution"])
