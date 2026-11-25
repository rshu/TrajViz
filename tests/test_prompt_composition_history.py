"""Composition history describes recorded request inputs without leaking contexts."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import jsonschema

import agent_trajviz as tv
from agent_trajviz.insight.prompt_construction import inspect_prompt_construction


def _entry(identifier, parent, role, content="", **extra):
    return {"type": "message", "id": identifier, "parentId": parent,
            "message": {"role": role, "content": content, **extra}}


def _native():
    return {"_pi_prompt_source": "recorded/session.jsonl", "_pi_entries": [
        {"type": "session", "version": 3, "id": "composition"},
        _entry("system", None, "system", sections={
            "preamble": "Base instruction", "rules": "Run focused tests", "skills": "Testing guide"},
            toolsAdded=[{"name": "read", "parameters": {"type": "object"}},
                        {"name": "mcp__files__read", "parameters": {"type": "object"}}]),
        _entry("user", "system", "user", "Repair the parser"),
        _entry("answer-1", "user", "assistant", [{"type": "toolCall", "id": "read-1", "name": "read"}]),
        _entry("result", "answer-1", "toolResult", "Recorded file contents", toolCallId="read-1", toolName="read"),
        _entry("patch", "result", "system", sections={"rules": "Run tests/test_parser.py and preserve public APIs"}),
        _entry("answer-2", "patch", "assistant", "Selected response is excluded"),
    ]}


def _step(index, role, text, **extra):
    return {"index": index, "role": role, "parts": [{"type": "text", "text": text}],
            "session_id": "root", "source_path": "recorded/root.jsonl", "tool_calls": [], **extra}


def _categories(composition):
    return {row["category"]: row["estimated_tokens"] for row in composition["categories"]}


def _pointer(value, path):
    for part in path.split("/")[1:] if path else []:
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def test_native_history_categorizes_each_boundary_and_keeps_observed_sizes_separate():
    data = inspect_prompt_construction(_native(), [])
    assert data["composition_history"]["complete"]
    assert data["composition_history"]["computed_requests"] == len(data["requests"]) == 2
    assert all(row["observed_prompt_tokens"] is None for row in data["evolution"])
    for row in data["evolution"]:
        summary = row["composition"]
        assert summary["evidence"] == "estimated" and summary["basis"]
        assert sum(_categories(summary).values()) == summary["total"]
        assert summary["component_count"] > 0
        assert "components" not in row and "messages" not in row
    first, last = (row["composition"] for row in data["evolution"])
    assert {"system", "rules", "skills", "tools", "mcp", "conversation"} <= {
        key for key, value in _categories(first).items() if value > 0}
    assert _categories(first).get("tool_outputs", 0) == 0
    assert _categories(last)["tool_outputs"] > 0
    assert data["selected_request"]["composition"] == last
    assert last["total"] == data["selected_request"]["token_estimate"]["total"]


def test_future_content_cannot_change_an_earlier_boundary_composition():
    raw = _native()
    before = inspect_prompt_construction(raw, [])
    first_id = before["requests"][0]["id"]
    earlier = inspect_prompt_construction(raw, [], first_id)["selected_request"]
    altered = copy.deepcopy(raw)
    altered["_pi_entries"][4]["message"]["content"] = "Future tool result " * 1000
    altered["_pi_entries"][5]["message"]["sections"]["rules"] = "Future rules " * 1000
    after = inspect_prompt_construction(altered, [], first_id)
    assert after["selected_request"]["composition"] == earlier["composition"]
    assert after["evolution"][0]["composition"] == before["evolution"][0]["composition"]
    assert "Future tool result" not in json.dumps(after["selected_request"])
    assert "Future rules" not in json.dumps(after["selected_request"])


def test_native_identity_survives_rewrites_and_occurrences_resolve_to_original_content():
    raw = _native()
    data = inspect_prompt_construction(raw, [])
    earlier = inspect_prompt_construction(raw, [], data["requests"][0]["id"])["selected_request"]
    current = data["selected_request"]
    old_rule = next(p for p in earlier["components"] if p["label"] == "rules")
    new_rule = next(p for p in current["components"] if p["label"] == "rules")
    assert new_rule["category"] == "rules" and new_rule["role"] == "system"
    assert new_rule["logical_id"] and old_rule["logical_id"] == new_rule["logical_id"]
    assert old_rule["content_hash"] != new_rule["content_hash"]
    assert old_rule["occurrence_id"] != new_rule["occurrence_id"]
    old_user = next(p for p in earlier["components"] if p["kind"] == "message")
    new_user = next(p for p in current["components"] if p["content"] == old_user["content"])
    assert old_user["logical_id"] == new_user["logical_id"] and new_user["logical_id"]
    assert old_user["content_hash"] == new_user["content_hash"]
    assert old_user["occurrence_id"] != new_user["occurrence_id"]
    assert any(ref.get("source_path") == "recorded/session.jsonl" for ref in new_rule["provenance"])
    assert any(ref.get("record_id") == "patch" for ref in new_rule["provenance"])
    for part in current["components"]:
        assert _pointer(current, part["location"]["json_pointer"]) == part["content"]
        assert sum(segment["estimated_tokens"] for segment in part["segments"]) == part["estimated_tokens"]
        for segment in part["segments"]:
            _pointer(part["content"], segment["json_pointer"])
            assert segment["provenance"]


def test_transcript_history_isolates_child_ownership_and_branch_membership():
    child = {"is_sub_agent": True, "agent": "reviewer", "parent_agent_id": "main"}
    steps = [_step(0, "user", "Main request"), _step(1, "assistant", "Main answer"),
             _step(2, "user", "Child input", **child), _step(3, "assistant", "Child answer", **child),
             _step(4, "user", "Sibling input", branch_id="other"),
             _step(5, "assistant", "Sibling answer", branch_id="other"),
             _step(6, "assistant", "Latest main answer")]
    data = inspect_prompt_construction({}, steps)
    root = [row for row in data["evolution"] if row["actor_kind"] == "main" and not row["branch_id"]]
    assert len(root) == 2
    changed = copy.deepcopy(steps)
    changed[2]["parts"][0]["text"] *= 1000
    changed[4]["parts"][0]["text"] *= 1000
    after = inspect_prompt_construction({}, changed)
    assert [row["composition"] for row in after["evolution"] if row["id"] in {r["id"] for r in root}] == [
        row["composition"] for row in root]
    assert len({row["context_id"] for row in data["evolution"]}) == 3
    assert all(row["composition"]["partial"] for row in data["evolution"])


def test_appending_a_pi_generation_does_not_rename_earlier_native_components():
    raw = _native()
    data = inspect_prompt_construction(raw, [])
    selected = data["selected_request"]
    extended = copy.deepcopy(raw)
    extended["_pi_entries"].extend([
        _entry("next-user", "answer-2", "user", "Another task"),
        _entry("answer-3", "next-user", "assistant", "Another response"),
    ])
    restored = inspect_prompt_construction(extended, [], selected["id"])["selected_request"]
    before = {p["id"]: p["logical_id"] for p in selected["components"]}
    assert before and all(identifier for identifier in before.values())
    assert {p["id"]: p["logical_id"] for p in restored["components"]} == before


def test_transcript_components_keep_recorded_sources_across_retained_occurrences():
    steps = [_step(0, "user", "Recorded task", raw_index=42, message_id="user-uuid"),
             _step(1, "assistant", "Earlier answer", raw_index=43),
             _step(2, "assistant", "Selected answer", raw_index=44)]
    data = inspect_prompt_construction({}, steps)
    first = inspect_prompt_construction({}, steps, data["requests"][0]["id"])["selected_request"]
    original = first["components"][0]
    retained = next(p for p in data["selected_request"]["components"] if p["content"] == original["content"])
    assert original["logical_id"] and retained["logical_id"] == original["logical_id"]
    assert retained["content_hash"] == original["content_hash"]
    assert retained["occurrence_id"] != original["occurrence_id"]
    assert any(ref.get("raw_index") == 42 and ref.get("message_id") == "user-uuid"
               and ref.get("source_path") == "recorded/root.jsonl" for ref in retained["provenance"])


def test_history_reuses_catalog_and_active_path_instead_of_reparsing_per_boundary(monkeypatch):
    from agent_trajviz.insight import prompt_construction, prompt_pi

    calls = {"transcript": 0, "pi": 0}
    original_catalog = prompt_construction._transcript_catalog
    original_path = prompt_pi._active_path

    def catalog(steps):
        calls["transcript"] += 1
        return original_catalog(steps)

    def active_path(entries):
        calls["pi"] += 1
        return original_path(entries)

    monkeypatch.setattr(prompt_construction, "_transcript_catalog", catalog)
    monkeypatch.setattr(prompt_pi, "_active_path", active_path)
    steps = [_step(0, "user", "Task"), *(_step(i + 1, "assistant", f"Answer {i}") for i in range(8))]
    transcript = inspect_prompt_construction({}, steps)
    native = inspect_prompt_construction(_native(), [])
    assert transcript["composition_history"]["computed_requests"] == 8
    assert native["composition_history"]["computed_requests"] == 2
    assert calls == {"transcript": 1, "pi": 1}


def test_missing_native_references_keep_composition_partial_without_inferred_content():
    raw = {"_claude_prompt_records": [{"source_path": "claude.jsonl", "raw_index": 2,
        "session_id": "claude", "agent_id": "", "record": {
            "type": "api-request", "id": "request", "shapeHash": "missing-shape", "tail": ["missing-message"]}}]}
    data = inspect_prompt_construction(raw, [])
    detail = data["selected_request"]
    assert detail["missing_references"]
    assert detail["composition"]["partial"] and detail["composition"]["total"] == 0
    assert data["evolution"][0]["composition"] == detail["composition"]


def test_history_bounds_keep_every_boundary_and_prioritize_selected_request():
    steps = [_step(0, "user", "Task")]
    steps.extend(_step(index + 1, "assistant", f"Response {index}") for index in range(205))
    data = inspect_prompt_construction({}, steps)
    coverage = data["composition_history"]
    assert len(data["requests"]) == len(data["evolution"]) == coverage["total_requests"] == 205
    assert not coverage["complete"] and coverage["limitations"]
    assert 0 < coverage["computed_requests"] <= 200
    assert sum(row["composition"] is not None for row in data["evolution"]) == coverage["computed_requests"]
    assert data["evolution"][-1]["composition"] == data["selected_request"]["composition"]


def test_public_schema_and_normalized_export_preserve_composition_and_provenance(tmp_path):
    run = replace(tv.Trajectory.from_steps([_step(0, "user", "Task"), _step(1, "assistant", "Done")]), raw=_native())
    before = copy.deepcopy(run.raw), copy.deepcopy(run.steps), copy.deepcopy(run.metrics)
    view = run.view("prompt_construction")
    jsonschema.Draft202012Validator(tv.view_schema("prompt_construction")).validate(json.loads(view.to_json()))
    tv.export_json(run, tmp_path, only=["prompt_construction"])
    restored = tv.load(tmp_path / "trajectory.json")
    assert restored.view("prompt_construction").data == view.data
    assert (run.raw, run.steps, run.metrics) == before


def test_no_boundaries_have_explicit_empty_composition_coverage():
    data = inspect_prompt_construction({}, [_step(0, "user", "No assistant yet")])
    assert data["selected_request"] is None and data["evolution"] == []
    assert data["composition_history"]["total_requests"] == 0
    assert data["composition_history"]["computed_requests"] == 0
    assert data["composition_history"]["complete"]
