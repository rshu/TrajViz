"""Relocating saved evidence preserves verdict identity, while edits invalidate it."""
from __future__ import annotations

import json
import shutil

import pytest

from agent_trajviz.awe._engine import config, judge, provenance


@pytest.fixture
def evidence_roots(tmp_path, monkeypatch):
    first = tmp_path / "first-location"
    trajectory = first / "data" / "trajectory" / "claude_code"
    requirements = first / "data" / "requirements"
    trajectory.mkdir(parents=True)
    requirements.mkdir()
    (trajectory / "case-1.jsonl").write_text('{"type":"assistant","message":{"content":"Public parent evidence"}}\n')
    child = trajectory / "case-1" / "subagents" / "agent-child.jsonl"
    child.parent.mkdir(parents=True)
    child.write_text('{"type":"assistant","message":{"content":"Public child evidence"}}\n')
    (requirements / "case-1.json").write_text(json.dumps({"problem_statement": "Public task", "gold_patch": "reference-one"}))
    second = tmp_path / "relocated-evidence"
    shutil.copytree(first, second)
    monkeypatch.setattr(config, "JUDGE_CACHE_DIR", tmp_path / "judge-cache")
    monkeypatch.setattr(provenance, "_warned_verdicts", set())
    def select(root):
        monkeypatch.setattr(config, "TRAJECTORY_DIR", root / "data" / "trajectory")
        monkeypatch.setattr(config, "REQUIREMENTS_DIR", root / "data" / "requirements")
    return first, second, select


def test_identical_relocated_evidence_reuses_verified_verdict_and_refreshes_normalization_identity(evidence_roots):
    first, second, select = evidence_roots
    select(first)
    key = judge._verdict_key("claude_code", "case-1", "test-model")
    normalized_key = provenance.trajectory_fingerprint("claude_code", "case-1")
    cache = judge.verdict_path("claude_code", "case-1", "test-model")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"input_fingerprint": key, "verdict": {}, "finish_reason": "stop"}))
    select(second)
    assert judge._verdict_key("claude_code", "case-1", "test-model") == key
    record = judge.verdict_cached("claude_code", "case-1", "test-model")
    assert record is not None and record["cache_provenance"]["status"] == "verified"
    assert provenance.trajectory_fingerprint("claude_code", "case-1") != normalized_key


def test_child_edit_invalidates_portable_verdict_without_changing_parent_bytes(evidence_roots):
    first, second, select = evidence_roots
    select(first)
    key = judge._verdict_key("claude_code", "case-1", "test-model")
    cache = judge.verdict_path("claude_code", "case-1", "test-model")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"input_fingerprint": key, "verdict": {}, "finish_reason": "stop"}))
    select(second)
    parent = second / "data" / "trajectory" / "claude_code" / "case-1.jsonl"
    original = parent.read_bytes()
    child = parent.with_suffix("") / "subagents" / "agent-child.jsonl"
    child.write_text('{"type":"assistant","message":{"content":"Changed child evidence"}}\n')
    assert parent.read_bytes() == original
    assert judge._verdict_key("claude_code", "case-1", "test-model") != key
    with pytest.warns(RuntimeWarning, match="stale verdicts"):
        assert judge.verdict_cached("claude_code", "case-1", "test-model") is None


def test_blinded_verdict_identity_keeps_task_model_and_prompt_but_excludes_gold_patch(evidence_roots):
    first, _, select = evidence_roots
    select(first)
    key = judge._verdict_key("claude_code", "case-1", "test-model")
    requirement = first / "data" / "requirements" / "case-1.json"
    requirement.write_text(json.dumps({"problem_statement": "Public task", "gold_patch": "another-private-reference",
                                       "FAIL_TO_PASS": ["private-evaluator-test"]}))
    assert judge._verdict_key("claude_code", "case-1", "test-model") == key
    assert judge._verdict_key("claude_code", "case-1", "another-model") != key
    assert provenance.verdict_fingerprint("claude_code", "case-1", "test-model", judge.SYSTEM_PROMPT + " changed",
                                          (judge.build_evidence, judge._render_user)) != key
    requirement.write_text(json.dumps({"problem_statement": "A different task", "gold_patch": "reference-one"}))
    assert judge._verdict_key("claude_code", "case-1", "test-model") != key
