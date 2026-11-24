"""Public prompt inspection preserves the distinction between requests and candidates."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.insight.prompt_construction import inspect_prompt_construction


def step(index, role, text="", **extra):
    return {"index": index, "role": role, "parts": [{"type": "text", "text": text}] if text else [],
            "session_id": "root", "tool_calls": [], **extra}


def test_transcript_fallback_excludes_answer_current_tools_and_parallel_children():
    steps = [step(0, "system", "Recorded instruction"), step(1, "user", "Current request"),
             step(2, "assistant", "Earlier reasoning", tool_calls=[{
                 "tool_id": "read", "tool_name": "Read", "input": {"path": "file.py"}, "output": "Prior result",
                 "time_end": 9}]),
             step(3, "user", "CHILD PRIVATE INPUT", is_sub_agent=True, agent="reviewer"),
             step(4, "assistant", "CHILD PRIVATE ANSWER", is_sub_agent=True, agent="reviewer"),
             step(5, "assistant", "OWN ANSWER", tool_calls=[{
                 "tool_name": "Read", "input": {}, "output": "OWN FUTURE RESULT"}], time_created_ms=10),
             step(6, "user", "FUTURE REQUEST")]
    original = copy.deepcopy(steps)
    data = inspect_prompt_construction({}, steps)
    detail = data["selected_request"]
    payload = json.dumps(detail)
    assert detail["evidence"] == "estimated" and detail["scope"] == "trajectory_estimate"
    assert "Recorded instruction" in payload and "Current request" in payload and "Prior result" in payload
    assert "CHILD PRIVATE" not in payload and "OWN ANSWER" not in payload and "OWN FUTURE" not in payload
    assert "FUTURE REQUEST" not in payload
    assert detail["tools"] == [] and detail["changes"]["available"]
    assert detail["changes"]["evidence"] == "estimated" and not detail["changes"]["lineage_verified"]
    assert detail["token_estimate"]["evidence"] == "estimated" and detail["token_estimate"]["partial"]
    assert all("messages" not in row for row in data["requests"])
    assert steps == original


def test_observed_compaction_bounds_candidates_and_their_sources():
    steps = [step(0, "system", "Instruction"), step(1, "user", "DROP BEFORE SUMMARY"),
             step(2, "assistant", "Old answer"),
             step(3, "compaction", is_compaction_checkpoint=True,
                  parts=[{"type": "compaction", "summary": "Recorded compact summary"}]),
             step(4, "user", "Keep recent request"), step(5, "assistant", "Final output")]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert "DROP BEFORE SUMMARY" not in json.dumps(detail["messages"] + detail["system"])
    assert "DROP BEFORE SUMMARY" in json.dumps(detail["changes"]["removed_components"])
    assert detail["changes"]["evidence"] == "estimated"
    assert "Recorded compact summary" in json.dumps(detail["messages"])
    assert "Instruction" in json.dumps(detail["system"])
    assert [source["step_index"] for source in detail["sources"]] == [0, 3, 4]
    assert any("compaction" in reason for reason in detail["limitations"])


def test_distinct_child_session_compaction_folds_the_checkpoint_mode_label():
    child = {"session_id": "child", "parent_session_id": "root", "is_sub_agent": True,
             "source_path": "child.jsonl", "agent": "worker"}
    steps = [step(0, "user", "OLD CHILD CONTEXT", **child), step(1, "assistant", "Old response", **child),
             step(2, "compaction", "Recorded child summary", **{**child, "agent": "compaction"},
                  is_compaction_checkpoint=True), step(3, "user", "Current task", **child),
             step(4, "assistant", "Current response", **child)]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert "OLD CHILD CONTEXT" not in json.dumps(detail["messages"])
    assert "Old response" not in json.dumps(detail["messages"])
    assert "Recorded child summary" in json.dumps(detail["messages"])
    assert [source["step_index"] for source in detail["sources"]] == [2, 3]
    assert any("observed compaction" in reason for reason in detail["limitations"])
    # A sibling checkpoint, foreign source or branch cannot reset this context.
    for patch in [{"session_id": "another-child"}, {"parent_session_id": "another-root"},
                  {"source_path": "sibling.jsonl"}, {"branch_id": "sibling-branch"}]:
        sibling = copy.deepcopy(steps)
        sibling[2].update(patch)
        messages = inspect_prompt_construction({}, sibling)["selected_request"]["messages"]
        assert "Recorded child summary" not in json.dumps(messages)
        assert "OLD CHILD CONTEXT" in json.dumps(messages)


def test_shared_session_compaction_is_not_assigned_to_an_unknown_child():
    shared = {"session_id": "root", "parent_session_id": "root", "source_path": "shared.jsonl",
              "is_sub_agent": True}
    steps = [step(0, "user", "Unknown child input", **shared),
             step(1, "compaction", "UNASSIGNED SUMMARY", **{**shared, "agent": "compaction"},
                  is_compaction_checkpoint=True), step(2, "assistant", "Unknown child response", **shared)]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert detail["messages"] == [] and detail["previous_candidate_id"] is None
    assert "UNASSIGNED SUMMARY" not in json.dumps(detail)


def test_delayed_tool_results_and_future_clock_records_are_excluded():
    steps = [step(0, "user", "Initial", time_created_ms=1),
             step(1, "assistant", "Prior", time_created_ms=2, tool_calls=[{
                 "tool_name": "Run", "input": {}, "output": "NOT FINISHED", "time_end": 100}]),
             step(2, "user", "FUTURE CLOCK INPUT", time_created_ms=200),
             step(3, "assistant", "Own answer", time_created_ms=50)]
    messages = inspect_prompt_construction({}, steps)["selected_request"]["messages"]
    assert "Initial" in json.dumps(messages)
    assert "NOT FINISHED" not in json.dumps(messages) and "FUTURE CLOCK INPUT" not in json.dumps(messages)


@pytest.mark.parametrize("end,boundary", [(None, 50), (10, None), (None, None)])
def test_untimed_tool_results_are_omitted_without_inventing_source_order_proof(end, boundary):
    steps = [step(0, "user", "Task"), step(1, "assistant", "Earlier call", tool_calls=[{
        "tool_name": "Read", "input": {}, "output": "UNTIMED FUTURE RESULT", "time_end": end}]),
        step(2, "assistant", "Current response", time_created_ms=boundary)]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert "UNTIMED FUTURE RESULT" not in json.dumps(detail)
    assert any("proven completion" in reason for reason in detail["limitations"])


def test_unknown_child_identity_is_never_pooled_with_parent_or_other_children():
    steps = [step(0, "user", "Parent request", source_path="same.jsonl"),
             step(1, "user", "UNKNOWN CHILD PRIVATE", source_path="same.jsonl", is_sub_agent=True),
             step(2, "assistant", "Parent answer", source_path="same.jsonl"),
             step(3, "assistant", "UNKNOWN CHILD ANSWER", source_path="same.jsonl", is_sub_agent=True)]
    data = inspect_prompt_construction({}, steps)
    parent = inspect_prompt_construction({}, steps, data["requests"][0]["id"])["selected_request"]
    assert "Parent request" in json.dumps(parent) and "UNKNOWN CHILD" not in json.dumps(parent)
    child = data["selected_request"]
    assert child["messages"] == []
    assert any("distinct recorded identity" in reason for reason in child["limitations"])
    assert child["previous_candidate_id"] is None and not child["changes"]["available"]


def test_adjacent_candidate_differences_are_estimated_and_never_cross_known_identities_or_branches():
    steps = [step(0, "user", "Main question", branch_id="a"),
             step(1, "assistant", "Main prior answer", branch_id="a"),
             step(2, "assistant", "OTHER BRANCH PRIVATE", branch_id="b"),
             step(3, "assistant", "CHILD PRIVATE", branch_id="a", agent="worker", is_sub_agent=True),
             step(4, "user", "Follow-up", branch_id="a"),
             step(5, "assistant", "CURRENT ANSWER", branch_id="a")]
    data = inspect_prompt_construction({}, steps)
    detail = data["selected_request"]
    changes = detail["changes"]
    assert changes["available"] and changes["evidence"] == "estimated"
    assert changes["basis"] == "adjacent_transcript_candidates" and changes["lineage_verified"] is False
    assert changes["previous_candidate_id"] == data["requests"][0]["id"]
    assert detail["previous_candidate_id"] == changes["previous_request_id"]
    assert changes["rewritten_components"] == []
    assert "does not verify" in changes["reason"]
    assert "Main prior answer" in json.dumps(changes["added_components"])
    assert "Follow-up" in json.dumps(changes["added_components"])
    assert all(text not in json.dumps(detail) for text in ["OTHER BRANCH PRIVATE", "CHILD PRIVATE", "CURRENT ANSWER"])
    assert data["evolution"][-1]["previous_request_id"] is None
    assert data["evolution"][-1]["previous_candidate_id"] == changes["previous_candidate_id"]


def test_evolution_context_keys_keep_unknown_children_and_explicit_branches_out_of_main_series():
    steps = [step(0, "assistant", tokens={"context_window": 100}),
             step(1, "assistant", tokens={"context_window": 900}, is_sub_agent=True),
             step(2, "assistant", tokens={"context_window": 150}),
             step(3, "assistant", tokens={"context_window": 200}, branch_id="another-branch"),
             step(4, "assistant", tokens={"context_window": 800}, is_sub_agent=True)]
    rows = inspect_prompt_construction({}, steps)["evolution"]
    assert [row["actor_kind"] for row in rows] == ["main", "unknown_child", "main", "main", "unknown_child"]
    assert rows[0]["context_id"] == rows[2]["context_id"]
    assert len({rows[index]["context_id"] for index in [0, 1, 3, 4]}) == 4
    assert rows[1]["actor_name"] == "" and rows[1]["previous_candidate_id"] is None
    assert rows[0]["branch_id"] is None and rows[3]["branch_id"] == "another-branch"
    assert rows[4]["previous_candidate_id"] is None
    assert [row["observed_prompt_tokens"] for row in rows] == [100, 900, 150, 200, 800]
    steps.append(step(5, "assistant", is_sub_agent=True, agent="worker", session_id="child", parent_session_id="root"))
    child = inspect_prompt_construction({}, steps)["evolution"][-1]
    assert child["actor_kind"] == "subagent" and child["actor_name"] == "worker"
    assert child["context_id"] not in {row["context_id"] for row in rows}


def test_native_evolution_context_keys_include_active_branch_identity():
    raw = pi_raw()
    first = inspect_prompt_construction(raw, [])["evolution"]
    raw["_pi_entries"].append({"type": "message", "id": "fork", "parentId": "a",
                               "message": {"role": "assistant", "content": "Fork response"}})
    forked = inspect_prompt_construction(raw, [])["evolution"]
    assert len({row["context_id"] for row in first}) == 1
    assert len({row["context_id"] for row in forked}) == 1
    assert first[0]["context_id"] != forked[0]["context_id"]
    assert first[0]["actor_kind"] == "main" and forked[0]["branch_id"] == "fork"


def test_no_boundary_and_unknown_request_have_explicit_outcomes():
    assert inspect_prompt_construction({}, [step(0, "user", "No response")])["selected_request"] is None
    with pytest.raises(ValueError, match="Unknown prompt request"):
        inspect_prompt_construction({}, [step(0, "user", "No response")], "missing")
    with pytest.raises(TypeError, match="request_id"):
        inspect_prompt_construction({}, [], 2)


def test_telemetry_redaction_does_not_become_prompt_evidence():
    steps = [step(0, "user", "REDACTED PLACEHOLDER", _otel={"fields": {"text": "unavailable"}}),
             step(1, "assistant", "Later")]
    detail = inspect_prompt_construction({}, steps)["selected_request"]
    assert detail["messages"] == []
    assert any("does not mean an empty prompt" in message for message in detail["limitations"])


def pi_raw():
    def entry(eid, parent, role, content="", **extra):
        return {"type": "message", "id": eid, "parentId": parent,
                "message": {"role": role, "content": content, **extra}}

    return {"_pi_entries": [{"type": "session", "version": 3, "id": "pi-session"},
                           entry("s", None, "system", sections={"preamble": "Rules", "skill": "Old skill"},
                                 toolsAdded=[{"name": "read", "parameters": {"type": "object"}}]),
                           entry("u", "s", "user", "User request"),
                           entry("a", "u", "assistant", "Earlier response"),
                           entry("p", "a", "system", sections={"skill": None},
                                 toolsRemoved=[{"name": "read"}], toolsAdded=[{"name": "edit"}]),
                           entry("b", "p", "assistant", "DO NOT LEAK ANSWER")]}


def test_native_state_delta_is_descriptive_and_scoped_to_recorded_lineage():
    raw = pi_raw()
    data = inspect_prompt_construction(raw, [])
    detail = data["selected_request"]
    assert detail["scope"] == "native_prompt_state" and detail["changes"]["available"]
    assert detail["changes"]["previous_request_id"] == data["requests"][0]["id"]
    assert {row["label"] for row in detail["changes"]["removed_components"]} >= {"skill", "read"}
    assert "edit" in {row["label"] for row in detail["changes"]["added_components"]}
    assert "DO NOT LEAK ANSWER" not in json.dumps(detail)
    assert detail["token_estimate"]["total"] == sum(row["estimated_tokens"] for row in detail["components"])
    with pytest.raises(ValueError):
        inspect_prompt_construction(raw, [], "not-a-request")


def test_evolution_catalog_is_compact_and_uses_only_mapped_validated_token_counts():
    raw = pi_raw()
    steps = [step(10, "assistant", id="a", session_id="pi-session", tokens={
        "input": 100, "output": 10, "cache_read": 20, "cache_write": 5, "total": 135}),
        step(20, "assistant", id="b", session_id="pi-session", tokens={"prompt_peak": 250})]
    data = inspect_prompt_construction(raw, steps)
    assert [row["order"] for row in data["evolution"]] == [1, 2]
    assert [row["observed_prompt_tokens"] for row in data["evolution"]] == [125, 250]
    assert data["evolution"][0]["observed_prompt_basis"] == "validated_recorded_input_and_cache_counts"
    assert data["evolution"][1]["observed_prompt_basis"] == "recorded_largest_request_in_step"
    assert any(event["kind"] == "system_section_removed" for event in data["evolution"][1]["prompt_events"])
    assert all("components" not in row and "messages" not in row for row in data["evolution"])
    # Two distinct occupancy equations fit these counts; show no chosen guess.
    steps[0]["tokens"] = {"input": 100, "output": 5, "reasoning": 10, "cache_read": 10, "cache_write": 0, "total": 115}
    assert inspect_prompt_construction(raw, steps)["evolution"][0]["observed_prompt_tokens"] is None
    # A declared context occupancy is authoritative over a step maximum.
    steps[1]["tokens"]["context_window"] = 200
    assert inspect_prompt_construction(raw, steps)["evolution"][1]["observed_prompt_tokens"] == 200
    steps[1]["index"] = 10  # duplicate mappings are never guessed
    assert all(row["observed_prompt_tokens"] is None for row in inspect_prompt_construction(raw, steps)["evolution"])


def test_proven_section_tool_and_message_identities_show_rewrites_with_before_after():
    raw = pi_raw()
    patch = raw["_pi_entries"][4]["message"]
    patch["sections"] = {"skill": "New skill instructions"}
    patch["toolsRemoved"] = []
    patch["toolsAdded"] = [{"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    raw["_pi_entries"].insert(5, {"type": "context_edit", "id": "edit-u", "parentId": "p", "targetId": "u",
                                    "replacement": "Rewritten user context"})
    raw["_pi_entries"][-1]["parentId"] = "edit-u"
    detail = inspect_prompt_construction(raw, [])["selected_request"]
    changes = detail["changes"]
    rewrites = {item["identity"]: item for item in changes["rewritten_components"]}
    assert set(rewrites) == {"pi:system:skill", "pi:tool:read", "pi:message:message:u"}
    assert rewrites["pi:system:skill"]["before"]["text"] == "Old skill"
    assert rewrites["pi:system:skill"]["after"]["text"] == "New skill instructions"
    assert rewrites["pi:message:message:u"]["before"]["content"] == "User request"
    assert rewrites["pi:message:message:u"]["after"]["content"] == "Rewritten user context"
    assert all(item["before_id"] in changes["removed"] and item["after_id"] in changes["added"]
               for item in rewrites.values())
    assert changes["groups_changed"] == {"system": True, "tools": True, "messages": True}
    assert all(len(item["preview"]) <= 500 and item["preview_format"] == "json"
               for item in changes["added_components"] + changes["removed_components"])


def test_evolution_does_not_invent_measurements_or_message_rewrite_identity():
    raw = pi_raw()
    # Replacing a user row with a new native entry is an addition/removal, not a rewrite.
    raw["_pi_entries"].insert(5, {"type": "message", "id": "different-user", "parentId": "p",
                                    "message": {"role": "user", "content": "Another request"}})
    raw["_pi_entries"][-1]["parentId"] = "different-user"
    data = inspect_prompt_construction(raw, [])
    assert all(row["observed_prompt_tokens"] is None for row in data["evolution"])
    assert not any(item["kind"] == "message" for item in data["selected_request"]["changes"]["rewritten_components"])
    run = tv.Trajectory.from_steps([step(0, "user", "Task"), step(1, "assistant", "Done")])
    assert run.view("prompt_construction").data["evolution"][0]["observed_prompt_tokens"] is None


def test_loaded_native_sessions_without_usage_do_not_plot_normalized_zero(tmp_path):
    pi_path = tmp_path / "pi.jsonl"
    pi_path.write_text("\n".join(json.dumps(row) for row in pi_raw()["_pi_entries"]))
    pi = tv.load(pi_path)
    assert pi.format == "pi" and pi.steps
    assert all(row["observed_prompt_tokens"] is None for row in pi.view("prompt_construction").data["evolution"])
    opencode_path = tmp_path / "opencode.json"
    opencode_path.write_text(json.dumps({"info": {"id": "root"}, "messages": [
        {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Question"}]},
        {"info": {"id": "a", "role": "assistant"}, "parts": [{"type": "text", "text": "Answer"}]}]}))
    opencode = tv.load(opencode_path, format="opencode")
    assert opencode.steps[-1]["tokens"]["input"] == 0  # normalization default, not a measured zero
    assert opencode.view("prompt_construction").data["evolution"][0]["observed_prompt_tokens"] is None


def test_transcript_events_follow_response_boundaries_without_claiming_prompt_assembly():
    steps = [step(0, "user", "Task", time_created_ms=1),
             step(1, "assistant", "Earlier answer", time_created_ms=10, tool_calls=[{
                 "tool_name": "Read", "tool_id": "call", "output": "TOOL BODY NOT IN EVENT METADATA",
                 "time_start": 15, "time_end": 20}]),
             step(2, "system", "Recorded system update", time_created_ms=21),
             step(3, "assistant", "Current answer", time_created_ms=25)]
    data = inspect_prompt_construction({}, steps)
    first, second = data["requests"]
    assert [event["kind"] for event in first["prompt_events"]] == ["transcript_user_message"]
    events = {event["kind"]: event for event in second["prompt_events"]}
    assert set(events) == {"transcript_tool_result", "transcript_system_message"}
    assert events["transcript_tool_result"]["source_refs"][0]["step_index"] == 1
    assert events["transcript_tool_result"]["source_refs"][0]["record"] == "normalized_step"
    assert events["transcript_tool_result"]["tool_name"] == "Read"
    assert all(event["evidence"] == "observed_transcript" for event in events.values())
    assert data["selected_request"]["prompt_events"] == data["evolution"][-1]["prompt_events"] == second["prompt_events"]
    assert "TOOL BODY NOT IN EVENT METADATA" not in json.dumps(data["requests"] + data["evolution"])
    assert data["selected_request"]["changes"]["evidence"] == "estimated"


def test_tool_result_event_waits_for_first_proven_subsequent_boundary():
    steps = [step(0, "assistant", time_created_ms=10, tool_calls=[{
        "tool_name": "Run", "tool_id": "late", "output": "Late result", "time_end": 50}]),
        step(1, "assistant", time_created_ms=25), step(2, "assistant"),
        step(3, "assistant", time_created_ms=70), step(4, "assistant", time_created_ms=80)]
    rows = inspect_prompt_construction({}, steps)["evolution"]
    assert [sum(event["count"] for event in row["prompt_events"] if event["kind"] == "transcript_tool_result")
            for row in rows] == [0, 0, 0, 1, 0]


def test_tool_result_event_cannot_precede_its_future_source_message():
    steps = [step(0, "assistant", time_created_ms=1000, tool_calls=[{
        "tool_name": "Read", "output": "Result", "time_end": 900}]),
        step(1, "assistant", time_created_ms=950), step(2, "assistant", time_created_ms=1100)]
    rows = inspect_prompt_construction({}, steps)["requests"]
    assert rows[1]["prompt_events"] == []
    assert rows[2]["prompt_events"][0]["kind"] == "transcript_tool_result"


@pytest.mark.parametrize("fields", [None, [], "invalid"])
def test_malformed_tool_evidence_fields_are_unavailable_not_event_proof(fields):
    steps = [step(0, "assistant", time_created_ms=10, tool_calls=[{
        "tool_name": "Read", "output": "UNVERIFIED RESULT", "time_end": 20,
        "metadata": {"_otel": {"fields": fields}}}]), step(1, "assistant", time_created_ms=30)]
    data = inspect_prompt_construction({}, steps)
    assert data["requests"][-1]["prompt_events"] == []
    assert "UNVERIFIED RESULT" not in json.dumps(data["selected_request"])


def test_transcript_event_contexts_and_compaction_are_source_backed_and_isolated():
    child = {"session_id": "child", "parent_session_id": "root", "is_sub_agent": True, "agent": "worker"}
    steps = [step(0, "user", "Parent task"), step(1, "user", "Child task", **child),
             step(2, "assistant", time_created_ms=10, **child),
             step(3, "compaction", "Child checkpoint", **{**child, "agent": "compaction"},
                  is_compaction_checkpoint=True), step(4, "assistant", time_created_ms=20, **child),
             step(5, "user", "UNKNOWN CHILD", is_sub_agent=True),
             step(6, "assistant", time_created_ms=30, tokens={"context_window": 1000}),
             step(7, "assistant", time_created_ms=40, tokens={"context_window": 5})]
    rows = inspect_prompt_construction({}, steps)["requests"]
    assert rows[0]["prompt_events"][0]["source_refs"][0]["step_index"] == 1
    assert [event["kind"] for event in rows[1]["prompt_events"]] == ["transcript_compaction"]
    assert rows[1]["prompt_events"][0]["source_refs"][0]["step_index"] == 3
    assert rows[2]["prompt_events"][0]["source_refs"][0]["step_index"] == 0
    assert rows[3]["prompt_events"] == []  # a token drop is not compaction evidence
    assert not any(source["step_index"] == 5 for row in rows for event in row["prompt_events"]
                   for source in event["source_refs"])


def test_transcript_event_metadata_is_bounded_and_missing_results_stay_unrecorded():
    calls = [{"tool_name": "Read", "tool_id": f"read-{index}", "output": "Result", "time_end": 20}
             for index in range(50)]
    calls += [{"tool_name": "NoTime", "output": "Pending", "time_end": None},
              {"tool_name": "Unobserved", "output": "Hidden", "time_end": 20, "metadata": {"_output_observed": False}},
              {"tool_name": "DefaultEmpty", "output": "", "time_end": 20},
              {"tool_name": "ObservedEmpty", "output": "", "time_end": 20, "metadata": {"_output_observed": True}}]
    steps = [step(0, "assistant", time_created_ms=10, tool_calls=calls), step(1, "assistant", time_created_ms=30)]
    event = inspect_prompt_construction({}, steps)["requests"][-1]["prompt_events"][0]
    assert event["kind"] == "transcript_tool_result" and event["count"] == 51
    assert len(event["source_refs"]) == 8 and event["source_refs_truncated"]
    assert "tool_name" not in event  # mixed tool names are not mislabelled as one tool
    assert all(source["tool_name"] == "Read" for source in event["source_refs"])


def test_large_opaque_image_content_is_preserved_but_not_counted_as_text_tokens():
    raw = pi_raw()
    raw["_pi_entries"][2]["message"]["content"] = [
        {"type": "image", "data": "A" * 20_000, "mimeType": "image/png"}, {"type": "text", "text": "Inspect image"}]
    detail = inspect_prompt_construction(raw, [])["selected_request"]
    assert detail["messages"][0]["content"][0]["data"] == "A" * 20_000
    assert detail["token_estimate"]["partial"] and detail["token_estimate"]["total"] < 1000


def test_public_view_schema_options_and_normalized_export_roundtrip(tmp_path):
    from dataclasses import replace

    run = replace(tv.Trajectory.from_steps([step(0, "user", "Task"), step(1, "assistant", "Done")]), raw=pi_raw())
    view = run.view("prompt_construction")
    assert set(tv.VIEW_CATALOG["prompt_construction"]["options"]) == {"request_id"}
    jsonschema.validate(json.loads(view.to_json()), tv.view_schema("prompt_construction"))
    request_id = view.data["requests"][0]["id"]
    assert run.view("prompt_construction", request_id=request_id).data["selected_request"]["id"] == request_id
    with pytest.raises(TypeError):
        run.view("prompt_construction", step=1)
    tv.export_json(run, tmp_path, only=["prompt_construction"])
    restored = tv.load(tmp_path / "trajectory.json")
    assert restored.view("prompt_construction").data == view.data
    assert json.loads((tmp_path / "prompt_construction.json").read_text())["data"] == view.data


def test_existing_claude_request_records_have_a_schema_valid_view_without_token_changes():
    path = Path(__file__).parent / "fixtures/claude_current/session-current.jsonl"
    run = tv.load(path)
    before = copy.deepcopy(run.steps), copy.deepcopy(run.metrics)
    view = run.view("prompt_construction")
    assert len(view.data["requests"]) == 2 and view.data["selected_request"]["scope"] == "recorded_request"
    assert all(row["observed_prompt_tokens"] is None for row in view.data["evolution"])
    jsonschema.validate(json.loads(view.to_json()), tv.view_schema("prompt_construction"))
    assert (run.steps, run.metrics) == before
