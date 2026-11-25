"""Native session integration, with opt-in coverage for private local captures.

Only neutral generated records and existing public fixtures are used by default.
Set TRAJVIZ_CLAUDE_SESSION_FIXTURE to inspect a local Claude JSONL family without
copying its prompt content into the repository or printing it in test failures.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.insight.ui import prompt_tab

_FIXTURES = Path(__file__).parent / "fixtures"


def _write(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _assert_summary(summary):
    assert summary["evidence"] == "estimated"
    assert sum(row["estimated_tokens"] for row in summary["categories"]) == summary["total"]


def _assert_public_view(run):
    view = run.view("prompt_construction")
    errors = list(jsonschema.Draft202012Validator(tv.view_schema("prompt_construction")).iter_errors(
        json.loads(view.to_json())))
    # Keep diagnostics structural even when the opt-in capture is private.
    assert not errors, [(list(error.absolute_path), error.validator) for error in errors]
    data = view.data
    assert data["available"] and data["requests"]
    assert len(data["evolution"]) == len(data["requests"])
    for row in data["evolution"]:
        if row["composition"] is not None:
            _assert_summary(row["composition"])
    return data


def test_bundled_claude_family_zip_keeps_native_owners_and_composition(tmp_path):
    source = _FIXTURES / "claude_current" / "session-current.jsonl"
    archive = tmp_path / "session-family.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for member in sorted(source.parent.rglob("*.jsonl")):
            handle.write(member, member.relative_to(source.parent))
    direct = _assert_public_view(tv.load(source))
    zipped = _assert_public_view(tv.load(archive))
    assert [row["id"] for row in zipped["requests"]] == [row["id"] for row in direct["requests"]]
    assert {row["actor_kind"] for row in zipped["evolution"]} == {"main", "subagent"}
    assert [row["composition"] for row in zipped["evolution"]] == [row["composition"] for row in direct["evolution"]]
    assert all(row["scope"] == "recorded_request" for row in zipped["evolution"])


def test_public_redacted_native_run_exposes_transcript_composition_without_inventing_requests():
    data = _assert_public_view(tv.load(_FIXTURES / "native_claude_code" / "django__django-11477.jsonl"))
    assert all(row["scope"] == "trajectory_estimate" for row in data["evolution"])
    assert all(row["composition"] is None or row["composition"]["partial"] for row in data["evolution"])
    assert data["selected_request"]["changes"].get("lineage_verified") is False


def test_native_missing_blob_chain_keeps_absence_explicit_and_preserves_reference_origins(tmp_path):
    records = [
        {"type": "user", "sessionId": "neutral", "uuid": "u", "message": {"role": "user", "content": "Task"}},
        {"type": "api-request-shape", "sessionId": "neutral", "shapeHash": "shape",
         "shape": {"system": [{"type": "text", "text": "Recorded instruction"}], "tools": []}},
        {"type": "api-request-blob", "sessionId": "neutral", "hash": "known",
         "message": {"role": "user", "content": [{"type": "text", "text": "Recorded task"}]}},
        {"type": "api-request", "sessionId": "neutral", "id": "first", "shapeHash": "shape", "keep": 0, "tail": ["known"]},
        {"type": "api-request", "sessionId": "neutral", "id": "second", "shapeHash": "shape", "base": "first",
         "keep": 1, "tail": ["absent-1", "absent-2"]},
        {"type": "api-request", "sessionId": "neutral", "id": "third", "shapeHash": "shape", "base": "second",
         "keep": 3, "tail": ["absent-3"]},
    ]
    run = tv.load(_write(tmp_path / "missing-native-blobs.jsonl", records))
    data = _assert_public_view(run)
    rows = data["requests"]
    first = run.view("prompt_construction", request_id=rows[0]["id"]).data["selected_request"]
    second = run.view("prompt_construction", request_id=rows[1]["id"]).data["selected_request"]
    third = data["selected_request"]
    assert first["evidence_quality"]["status"] == "complete"
    assert second["evidence_quality"]["status"] == third["evidence_quality"]["status"] == "partial"
    missing = {item["reference"]: item for item in third["evidence_quality"]["missing_references"]}
    assert set(missing) == {"absent-1", "absent-2", "absent-3"}
    assert all(item["kind"] == "message" and item["reason"] == "not_recorded" for item in missing.values())
    assert any(ref.get("raw_index") == 4 for ref in missing["absent-1"]["source_refs"])
    assert any(ref.get("raw_index") == 5 for ref in missing["absent-3"]["source_refs"])
    assert [message["content"][0]["text"] for message in third["messages"]] == ["Recorded task"]
    assert third["composition"]["partial"] and not third["changes"]["available"]


def test_generated_native_pi_branch_keeps_context_edits_and_compaction_source_history(tmp_path):
    def message(identifier, parent, role, content="", **extra):
        return {"type": "message", "id": identifier, "parentId": parent,
                "message": {"role": role, "content": content, **extra}}

    records = [
        {"type": "session", "version": 3, "id": "neutral-pi"},
        message("system", None, "system", sections={"rules": "Run focused tests"}, toolsAdded=[]),
        message("user", "system", "user", "Original task"),
        message("answer", "user", "assistant", "Earlier answer"),
        message("sibling", "answer", "user", "Excluded sibling context"),
        message("sibling-answer", "sibling", "assistant", "Excluded sibling response"),
        {"type": "context_edit", "id": "edit", "parentId": "answer", "targetId": "user", "replacement": "Updated task"},
        message("next", "edit", "assistant", "Second answer"),
        {"type": "compaction", "id": "summary", "parentId": "next", "firstKeptEntryId": "summary",
         "summary": "Recorded compact summary", "systemMessage": {
             "role": "system", "sections": {"rules": "Retained rules"}, "toolsAdded": []}},
        message("last", "summary", "assistant", "Selected response"),
    ]
    run = tv.load(_write(tmp_path / "pi-native.jsonl", records))
    data = _assert_public_view(run)
    assert len(data["requests"]) == 3
    edited = run.view("prompt_construction", request_id=data["requests"][1]["id"]).data["selected_request"]
    assert edited["messages"][0]["content"] == "Updated task"
    assert any(ref.get("context_edit_id") == "edit" for ref in edited["message_refs"])
    assert "Excluded sibling" not in json.dumps(edited)
    selected = data["selected_request"]
    assert [row["content"] for row in selected["messages"]] == ["Recorded compact summary"]
    assert any(row["category"] == "summarized" and row["estimated_tokens"] > 0
               for row in selected["composition"]["categories"])
    assert any(ref.get("record_id") == "summary" for part in selected["components"] for ref in part["provenance"])


def test_optional_local_claude_native_family_without_vendoring_private_content():
    configured = os.environ.get("TRAJVIZ_CLAUDE_SESSION_FIXTURE")
    if not configured:
        pytest.skip("set TRAJVIZ_CLAUDE_SESSION_FIXTURE to a local native Claude session JSONL")
    source = Path(configured).expanduser()
    assert source.is_file(), "TRAJVIZ_CLAUDE_SESSION_FIXTURE must name an existing JSONL file"
    run = tv.load(source)
    before = _digest(run.raw), _digest(run.steps), _digest(run.metrics)
    data = _assert_public_view(run)
    assert all(row["scope"] == "recorded_request" for row in data["requests"])
    for row in data["requests"]:
        selected = run.view("prompt_construction", request_id=row["id"]).data["selected_request"]
        quality = selected["evidence_quality"]
        assert quality["status"] in {"complete", "partial"}
        assert quality["content_complete"] == (not selected["missing_references"])
        assert quality["estimation"]["provider_request_verified"] is False
        assert all(item["source_refs"] for item in quality["missing_references"])
        output = prompt_tab.outputs(run, row["id"])
        assert output[-1]["id"] == row["id"]
        assert "could not be completed" not in output[2]
        _assert_summary(selected["composition"])
    assert (_digest(run.raw), _digest(run.steps), _digest(run.metrics)) == before
