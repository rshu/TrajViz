"""Pi prompt replay follows native branches without changing transcript data."""

from __future__ import annotations

import copy
import json

import agent_trajviz as tv
import pytest
from agent_trajviz.insight.loaders import _convert_pi_events
from agent_trajviz.insight.parser import parse_steps
from agent_trajviz.insight.prompt_pi import inspect_pi


def header(**extra):
    return {"type": "session", "version": 3, "id": "session", **extra}


def entry(eid, parent, role, content="", **extra):
    return {"type": "message", "id": eid, "parentId": parent,
            "message": {"role": role, "content": content, **extra}}


def inspect(events, request_id=None):
    raw = _convert_pi_events(events)
    raw["_pi_prompt_source"] = "recorded/session.jsonl"
    return inspect_pi(raw, parse_steps(raw), request_id)


def test_prompt_patches_and_request_boundaries_do_not_leak_response():
    events = [header(),
              entry("sys", None, "system", sections={"preamble": "Base", "skills": "Old skill"},
                    toolsAdded=[{"name": "read", "parameters": {"type": "object"}}, {"name": "write"}]),
              entry("u", "sys", "user", "Fix the regression"),
              entry("a", "u", "assistant", [{"type": "toolCall", "id": "c", "name": "read"}], model="m1"),
              entry("r", "a", "toolResult", [{"type": "text", "text": "Earlier tool result"}], toolCallId="c"),
              entry("patch", "r", "system", sections={"skills": None, "cwd": "/project"},
                    toolsRemoved=[{"name": "write"}], toolsAdded=[{"name": "edit", "parameters": {"type": "object"}}]),
              entry("b", "patch", "assistant", "DO NOT INCLUDE OWN ANSWER", model="m2"),
              entry("later", "b", "toolResult", "DO NOT INCLUDE FUTURE RESULT")]
    before = copy.deepcopy(events)
    value = inspect(events)
    assert value["available"] and len(value["requests"]) == 2
    detail = value["selected_request"]
    assert detail["model"] == "m2" and detail["scope"] == "native_prompt_state"
    assert detail["evidence"] == "reconstructed" and detail["missing_references"] == []
    assert {b["section"]: b["text"] for b in detail["system"]} == {"preamble": "Base", "cwd": "/project"}
    assert [t["name"] for t in detail["tools"]] == ["read", "edit"]
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "toolResult"]
    assert "Earlier tool result" in json.dumps(detail["messages"])
    assert "DO NOT INCLUDE" not in json.dumps(detail)
    assert detail["previous_request_id"] == value["requests"][0]["id"]
    recorded_changes = {(event["kind"], event["target_id"]) for event in detail["prompt_events"]}
    assert recorded_changes >= {("system_section_removed", "skills"), ("system_section_added", "cwd"),
                                ("tool_removed", "write"), ("tool_added", "edit")}
    assert next(event["count"] for event in detail["prompt_events"] if event["kind"] == "tool_results") == 1
    assert all("messages" not in row for row in value["requests"])
    assert detail["step_index"] is not None
    assert any(s["record_id"] == "patch" and s["source_path"] == "recorded/session.jsonl" for s in detail["sources"])
    earlier = inspect(events, value["requests"][0]["id"])["selected_request"]
    assert [m["role"] for m in earlier["messages"]] == ["user"]
    assert {b["section"] for b in earlier["system"]} == {"preamble", "skills"}
    assert events == before
    json.dumps(value, allow_nan=False)


def test_active_branch_excludes_sibling_messages_and_edits():
    events = [header(), entry("u", None, "user", "Original task"),
              entry("a", "u", "assistant", "Shared answer"),
              entry("sibling", "a", "user", "SECRET SIBLING"),
              {"type": "context_edit", "id": "sibling-edit", "parentId": "sibling", "targetId": "u", "replacement": None},
              entry("other-answer", "sibling-edit", "assistant", "SECRET SIBLING ANSWER"),
              {"type": "branch_summary", "id": "summary", "parentId": "a", "fromId": "other-answer", "summary": "Useful branch summary"},
              entry("b", "summary", "assistant", "Current answer")]
    value = inspect(events)
    assert len(value["requests"]) == 2
    detail = value["selected_request"]
    assert "SECRET SIBLING" not in json.dumps(detail)
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "branchSummary"]
    assert detail["messages"][0]["content"] == "Original task"
    assert detail["branch_id"] == "b"
    assert [event["kind"] for event in detail["prompt_events"]] == ["branch_summary"]
    assert detail["prompt_events"][0]["source"]["record_id"] == "summary"
    assert [ref["entry_id"] for ref in detail["message_refs"]] == ["u", "a", "summary"]


def test_latest_compaction_checkpoint_and_retained_range():
    events = [header(), entry("sys", None, "system", sections={"preamble": "Old prompt"}, toolsAdded=[{"name": "old"}]),
              entry("old", "sys", "user", "SUMMARIZED EARLIER TEXT"),
              entry("a", "old", "assistant", "Old answer"),
              entry("keep", "a", "user", "Retained question"),
              entry("retained-system", "keep", "system", sections={"obsolete": "must be folded"}),
              {"type": "compaction", "id": "compact", "parentId": "retained-system", "summary": "Recorded summary",
               "firstKeptEntryId": "keep", "systemMessage": {"role": "system", "content": "", "sections": {"preamble": "Checkpoint"},
                                                             "toolsAdded": [{"name": "new"}]}},
              entry("a2", "compact", "assistant", "Final response")]
    detail = inspect(events)["selected_request"]
    assert [m["content"] for m in detail["messages"]] == ["Recorded summary", "Retained question"]
    assert [m["role"] for m in detail["messages"]] == ["compactionSummary", "user"]
    assert detail["system"] == [{"type": "text", "section": "preamble", "text": "Checkpoint"}]
    assert [t["name"] for t in detail["tools"]] == ["new"]
    assert detail["missing_references"] == []
    changes = {(event["kind"], event["target_id"]) for event in detail["prompt_events"]}
    assert changes >= {("compaction", "keep"), ("tool_removed", "old"), ("tool_added", "new"),
                       ("system_section_updated", "preamble")}
    assert detail["system_refs"][0]["record_id"] == "compact"
    assert detail["tools_refs"][0]["record_id"] == "compact"


def test_retain_none_and_missing_retention_fail_closed():
    events = [header(), entry("u", None, "user", "Earlier content"),
              {"type": "compaction", "id": "compact", "parentId": "u", "summary": "Summary", "firstKeptEntryId": "compact"},
              entry("a", "compact", "assistant", "response")]
    detail = inspect(events)["selected_request"]
    assert [m["content"] for m in detail["messages"]] == ["Summary"]
    assert any("checkpoint" in text for text in detail["limitations"])
    events[2]["firstKeptEntryId"] = "unavailable"
    detail = inspect(events)["selected_request"]
    assert detail["missing_references"] == ["unavailable"]
    assert [m["content"] for m in detail["messages"]] == ["Summary"]


def test_context_edits_are_branch_relative_and_latest_wins():
    events = [header(), entry("u", None, "user", "Removed question"),
              entry("a", "u", "assistant", [{"type": "text", "text": "Original answer"}]),
              {"type": "context_edit", "id": "edit1", "parentId": "a", "targetId": "a", "replacement": "First replacement"},
              {"type": "context_edit", "id": "edit2", "parentId": "edit1", "targetId": "a", "replacement": "Latest replacement"},
              {"type": "context_edit", "id": "omit", "parentId": "edit2", "targetId": "u", "replacement": None},
              entry("b", "omit", "assistant", "New output")]
    detail = inspect(events)["selected_request"]
    assert detail["messages"] == [{"role": "assistant", "content": [{"type": "text", "text": "Latest replacement"}]}]
    assert {s["record_id"] for s in detail["sources"]} >= {"edit2", "omit", "a"}
    earlier_id = inspect(events)["requests"][0]["id"]
    earlier = inspect(events, earlier_id)["selected_request"]
    assert earlier["messages"] == [{"role": "user", "content": "Removed question"}]
    assert events[2]["message"]["content"][0]["text"] == "Original answer"
    assert detail["message_refs"][0]["entry_id"] == "a"
    assert detail["message_refs"][0]["context_edit_id"] == "edit2"
    assert detail["message_refs"][0]["origin"] == "message"
    assert detail["message_refs"][0]["source_path"] == "recorded/session.jsonl"
    assert [(event["target_id"], event["action"]) for event in detail["prompt_events"]
            if event["kind"] == "context_edit"] == [("a", "replace"), ("a", "replace"), ("u", "omit")]


def test_custom_context_bash_exclusion_and_hostile_content_are_preserved_as_data():
    hostile = '<script>alert("context")</script>'
    events = [header(), entry("u", None, "user", hostile),
              {"type": "custom", "id": "state", "parentId": "u", "customType": "state", "data": "NOT MODEL CONTEXT"},
              {"type": "custom_message", "id": "custom", "parentId": "state", "content": "Hidden in UI", "display": False},
              entry("bash", "custom", "bashExecution", command="ls", output="EXCLUDED OUTPUT", excludeFromContext=True),
              entry("bash2", "bash", "bashExecution", command="pwd", output="/project", exitCode=0),
              entry("a", "bash2", "assistant", "Answer")]
    detail = inspect(events)["selected_request"]
    assert detail["messages"][0]["content"] == hostile
    assert [m["role"] for m in detail["messages"]] == ["user", "user", "bashExecution"]
    assert detail["messages"][1]["content"] == "Hidden in UI"
    assert "NOT MODEL CONTEXT" not in json.dumps(detail)
    assert "EXCLUDED OUTPUT" not in json.dumps(detail)


def test_broken_parent_chain_never_uses_unrelated_linear_history():
    events = [header(), entry("unrelated", None, "user", "OTHER BRANCH"),
              entry("a", "missing-parent", "assistant", "Answer")]
    value = inspect(events)
    detail = value["selected_request"]
    assert detail["messages"] == []
    assert detail["missing_references"] == ["missing-parent"]
    assert "OTHER BRANCH" not in json.dumps(detail)
    cyclic = [header(), entry("u", "a", "user", "Bad chain"), entry("a", "u", "assistant", "answer")]
    assert inspect(cyclic)["selected_request"]["missing_references"]


def test_old_sessions_do_not_invent_system_or_tool_definitions():
    events = [header(version=1), {"type": "message", "message": {"role": "user", "content": "Task"}},
              {"type": "message", "message": {"role": "assistant", "content": "Result"}}]
    value = inspect(events)
    detail = value["selected_request"]
    assert detail["system"] == [] and detail["tools"] == []
    assert len(detail["limitations"]) == 3
    assert detail["evidence"] == "reconstructed"
    assert "final request" in detail["limitations"][0]
    assert inspect_pi({"_pi_entries": [header(version=99)]}, [])["available"] is False
    assert inspect_pi({}, [])["available"] is False
    with pytest.raises(ValueError, match="Unknown prompt request id"):
        inspect(events, "not-a-request")


def test_progress_entries_are_preferred_over_repeated_completion_notification():
    system = entry("sys", None, "system", sections={"preamble": "Base"}, toolsAdded=[])
    user = entry("u", "sys", "user", "Task")
    answer = entry("a", "u", "assistant", "Answer")
    events = [header(), *({"type": "entry_appended", "entry": e} for e in (system, user, answer)),
              {"type": "message_end", "message": answer["message"]}]
    value = inspect_pi({"_pi_entries": events}, [])
    assert len(value["requests"]) == 1
    assert value["selected_request"]["system"][0]["text"] == "Base"


def test_loading_and_normalized_roundtrip_leave_counts_unchanged(tmp_path):
    events = [header(), entry("sys", None, "system", sections={"preamble": "Base"}, toolsAdded=[]),
              entry("u", "sys", "user", "Task"),
              entry("a", "u", "assistant", "Answer", usage={"input": 10, "output": 2, "totalTokens": 12})]
    path = tmp_path / "pi.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    run = tv.load(path)
    before = copy.deepcopy(run.raw)
    inspect_pi(run.raw, run.steps)
    assert run.metrics["tokens"]["total"] == 12
    assert run.raw == before
    assert run.raw["_input_files"] == [str(path)]
    assert run.raw["_pi_prompt_source"] == str(path)
    assert all(source["source_path"] == str(path) for source in inspect_pi(run.raw, run.steps)["selected_request"]["sources"])
    tv.export_json(run, tmp_path / "out")
    reloaded = tv.load(tmp_path / "out" / "trajectory.json")
    original_prompt = inspect_pi(run.raw, run.steps)
    assert inspect_pi(reloaded.raw, reloaded.steps) == original_prompt
    assert original_prompt["selected_request"]["messages"] == [{"role": "user", "content": "Task"}]


def test_malformed_source_values_are_bounded_json_and_missing_tool_schemas_are_disclosed():
    events = [header(), entry("sys", None, "system", sections={"base": "Recorded prompt"},
                             toolsAdded=[{"name": "partial"}, {"name": "nan", "parameters": {"value": float("nan")}}]),
              entry("u", "sys", "user", [{"type": "image", "other": float("inf")}]),
              entry("a", "u", "assistant", "Response")]
    detail = inspect_pi({"_pi_entries": events}, [])["selected_request"]
    assert any("missing its schema" in line for line in detail["limitations"])
    assert any("safe JSON" in line for line in detail["limitations"])
    assert detail["tools"][1]["parameters"]["value"] is None
    assert detail["messages"][0]["content"][0]["other"] is None
    json.dumps(detail, allow_nan=False)
    assert inspect_pi({"_pi_entries": [header(version=3.0)]}, [])["available"] is False


def test_missing_compaction_boundary_does_not_retain_anonymous_legacy_entries():
    events = [header(version=1), {"type": "message", "message": {"role": "user", "content": "UNKNOWN RETENTION"}},
              {"type": "compaction", "summary": "Summary only"},
              {"type": "message", "message": {"role": "assistant", "content": "Answer"}}]
    detail = inspect_pi({"_pi_entries": events}, [])["selected_request"]
    assert [m["content"] for m in detail["messages"]] == ["Summary only"]
    assert "firstKeptEntryId" in detail["missing_references"]


def test_evolution_records_prompt_rewrites_with_native_identity_and_bounded_catalog():
    events = [header(), entry("s", None, "system", sections={"preamble": "Original instruction"},
                             toolsAdded=[{"name": "read", "description": "Old", "parameters": {"type": "object"}}]),
              entry("u1", "s", "user", "First question"), entry("u2", "u1", "user", "More detail"),
              entry("a", "u2", "assistant", "Response"),
              entry("patch", "a", "system", sections={"preamble": "Rewritten instruction"},
                    toolsAdded=[{"name": "read", "description": "Updated", "parameters": {"type": "object"}}]),
              entry("u3", "patch", "user", "Follow-up"), entry("b", "u3", "assistant", "Other response")]
    result = inspect(events)
    first, last = result["requests"]
    assert next(event for event in first["prompt_events"] if event["kind"] == "user_messages")["count"] == 2
    assert [(event["kind"], event["target_id"]) for event in last["prompt_events"]] == [
        ("system_section_updated", "preamble"), ("tool_updated", "read"), ("user_messages", "")]
    detail = result["selected_request"]
    assert [ref["entry_id"] for ref in detail["message_refs"]] == ["u1", "u2", "a", "u3"]
    assert detail["system_refs"][0]["record_id"] == detail["tools_refs"][0]["record_id"] == "patch"
    # Catalog annotations contain source locations/counts, never old/new prompt
    # snapshots, so listing requests does not duplicate a growing conversation.
    assert "Rewritten instruction" not in json.dumps(result["requests"])
    assert "Original instruction" not in json.dumps(result["requests"])
    assert "Other response" not in json.dumps(detail)


def test_generation_catalog_materializes_only_the_selected_context(monkeypatch):
    import agent_trajviz.insight.prompt_pi as module

    events = [header(), entry("s", None, "system", sections={"preamble": "Rules"}, toolsAdded=[])]
    previous = "s"
    for number in range(600):
        events.append(entry(f"u{number}", previous, "user", f"Question {number}"))
        events.append(entry(f"a{number}", f"u{number}", "assistant", f"Answer {number}"))
        previous = f"a{number}"
    materialized = []
    replay = module._prompt_state

    def selected_only(prefix, source_path):
        materialized.append(len(prefix))
        return replay(prefix, source_path)

    monkeypatch.setattr(module, "_prompt_state", selected_only)
    result = module.inspect_pi({"_pi_entries": events}, [])
    assert len(result["requests"]) == 600
    assert len(materialized) == 1
    assert all("messages" not in request for request in result["requests"])
    assert all(next(event["count"] for event in request["prompt_events"] if event["kind"] == "user_messages") == 1
               for request in result["requests"])
