"""Public attribution certifies native parent/child evidence as one run."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz import attribution as public
from agent_trajviz.insight import attribution as seam

_FIXTURES = Path(tv.__file__).resolve().parents[1] / "tests" / "fixtures"
_PATCH = "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"


@pytest.fixture(autouse=True)
def restore_reference_root():
    seam.configure(seam._DEFAULT_ROOT)
    yield
    seam.configure(seam._DEFAULT_ROOT)


def _benchmark(root: Path, agent: str, instance: str) -> None:
    data = root / "data"
    (data / "requirements").mkdir(parents=True, exist_ok=True)
    (data / "patch" / agent).mkdir(parents=True, exist_ok=True)
    (data / "requirements" / (instance + ".json")).write_text(json.dumps({
        "instance_id": instance, "repo": "fixture/project", "base_commit": "baseline",
        "problem_statement": "Fix the source behavior.", "gold_patch": _PATCH,
        "FAIL_TO_PASS": ["test_fix"], "PASS_TO_PASS": ["test_preserve"],
    }))
    (data / "patch" / agent / (instance + ".diff")).write_text(_PATCH.replace("+value = 1", "+value = 2"))
    (data / ("eval_" + agent + ".json")).write_text(json.dumps({instance: False}))


def _case(root: Path, kind: str, *, child_only: bool = False):
    agent = {"claude": "claude_code", "custom": "custom-coding-model", "dsh_v4": "dsh"}.get(kind, kind)
    directory = root / "data" / "trajectory" / agent
    directory.mkdir(parents=True)
    children = []
    if kind in {"claude", "custom"}:
        instance = "session-current"
        source = directory / (instance + ".jsonl")
        fixture = _FIXTURES / "claude_current"
        if child_only:
            shutil.copyfile(fixture / "session-current/subagents/agent-reviewer-current.jsonl", source)
        else:
            shutil.copyfile(fixture / "session-current.jsonl", source)
            shutil.copytree(fixture / "session-current", directory / "session-current")
            children = list((directory / "session-current").rglob("*.jsonl"))
    elif kind == "codex":
        instance = "root"
        source = directory / "root.jsonl"
        fixture = _FIXTURES / "codex_current"
        if child_only:
            shutil.copyfile(fixture / "root/subagents/agent-child.jsonl", source)
        else:
            shutil.copyfile(fixture / "root.jsonl", source)
            shutil.copytree(fixture / "root", directory / "root")
            children = list((directory / "root").rglob("*.jsonl"))
    elif kind == "dsh_v4":
        instance = "session.v4"
        source = directory / "session.v4.jsonl"
        fixture = _FIXTURES / "dsh_native_v4"
        shutil.copyfile(fixture / ("reviewer-a" if child_only else "root") / "session.v4.jsonl", source)
        if not child_only:
            for name in ("reviewer-a", "reviewer-b"):
                shutil.copytree(fixture / name, directory.parent / name)
                children.append(directory.parent / name / "session.v4.jsonl")
    elif kind == "cursor":
        instance = "cursor-native"
        source = directory / (instance + ".json")
        shutil.copyfile(_FIXTURES / "cursor_minimal.json", source)
    elif kind == "pi":
        instance = "pi-native"
        source = directory / (instance + ".jsonl")
        source.write_text("\n".join(json.dumps(row) for row in [
            {"type": "session", "version": 3, "id": "pi-native", "timestamp": "2026-10-26T10:00:00Z", "cwd": "/workspace"},
            {"type": "message", "id": "u1", "message": {"role": "user", "content": [{"type": "text", "text": "Fix the source behavior."}]}},
            {"type": "message", "id": "a1", "message": {"role": "assistant", "content": [{"type": "text", "text": "I inspected the implementation."}], "stopReason": "stop"}},
        ]) + "\n")
    else:
        raise AssertionError(kind)
    _benchmark(root, agent, instance)
    return agent, instance, source, children


@pytest.mark.parametrize("kind", ["claude", "codex", "dsh_v4", "cursor", "pi", "custom"])
def test_public_api_certifies_current_native_sessions_and_declared_custom_agents(tmp_path, kind):
    agent, instance, source, children = _case(tmp_path, kind)
    run = tv.load(source)
    assert run.steps
    if children:
        assert run.raw["_source_merged_count"] == len(children)
    result = public.diagnose(run, agent=agent, instance_id=instance, reference_root=tmp_path)
    assert result["available"], result["reason"]
    assert result["dossier"]["thinking_process"]
    assert result["dossier"]["event_ledger"]["available"] is True
    assert result["used_judge"] is False
    proof = result["dossier"]["source_certification"]
    assert proof["trajectory_sha256"] == run.sha256
    assert proof["trajectory"]["path"] == str(source.resolve())
    assert proof["trajectory"]["adapter_sha256"]
    if children:
        assert len(proof["trajectory"]["files"]) >= 1 + len(children)
    if children:
        assert any(row.get("source_ref", {}).get("thread")
                   for row in result["dossier"]["thinking_process"])


@pytest.mark.parametrize("kind", ["claude", "codex", "dsh_v4"])
def test_native_child_session_can_be_diagnosed_as_its_own_official_run(tmp_path, kind):
    agent, instance, source, _ = _case(tmp_path, kind, child_only=True)
    result = public.diagnose(tv.load(source), agent=agent, instance_id=instance, reference_root=tmp_path)
    assert result["available"], result["reason"]


@pytest.mark.parametrize("kind", ["claude", "codex", "dsh_v4"])
def test_changed_child_after_display_load_is_refused(tmp_path, kind):
    agent, instance, source, children = _case(tmp_path, kind)
    run = tv.load(source)
    children[0].write_text(children[0].read_text() + "\n")
    result = public.diagnose(run, agent=agent, instance_id=instance, reference_root=tmp_path)
    assert result["available"] is False
    assert "does not match the canonical" in result["reason"]
    assert result["primary"] is None and result["faults"] == []


@pytest.mark.parametrize("kind", ["claude", "codex", "dsh_v4"])
def test_child_mutation_during_diagnosis_is_refused(tmp_path, monkeypatch, kind):
    from agent_trajviz.awe._engine import detect

    agent, instance, source, children = _case(tmp_path, kind)
    run = tv.load(source)
    original = detect.detect
    def racing(*args, **kwargs):
        result = original(*args, **kwargs)
        children[0].write_text(children[0].read_text() + "\n")
        return result
    monkeypatch.setattr(detect, "detect", racing)
    result = public.diagnose(run, agent=agent, instance_id=instance, reference_root=tmp_path)
    assert result["available"] is False
    assert "changed while the diagnosis was running" in result["reason"]
    assert result["primary"] is None and result["faults"] == []


def test_parent_only_hash_never_certifies_native_children(tmp_path):
    agent, instance, source, _ = _case(tmp_path, "claude")
    parent_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = seam.diagnose(agent=agent, instance_id=instance, source_path=source, expected_sha=parent_sha,
                           merged_sources=0, reference_root=tmp_path)
    assert result.available is False and "does not match the canonical" in result.reason


def test_relocated_complete_native_tree_passes_without_a_captured_sha(tmp_path):
    root = tmp_path / "reference"
    agent, instance, source, _ = _case(root, "claude")
    copy = tmp_path / "downloaded-copy"
    shutil.copytree(source.parent, copy)
    result = seam.diagnose(agent=agent, instance_id=instance, source_path=copy / source.name, reference_root=root)
    assert result.available, result.reason


def test_archive_identity_has_an_explicit_refusal_instead_of_parent_only_certification(tmp_path):
    agent, instance, source, _ = _case(tmp_path / "reference", "claude")
    archive = tmp_path / "session.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for member in source.parent.rglob("*"):
            if member.is_file():
                stream.write(member, member.relative_to(source.parent))
    run = tv.load(archive)
    result = public.diagnose(run, agent=agent, instance_id=instance, reference_root=tmp_path / "reference")
    assert result["available"] is False
    assert "ZIP archive" in result["reason"] and "matching native session files" in result["reason"]


@pytest.mark.parametrize("agent", ["../escape", "/etc/passwd", "a/b", ".hidden"])
def test_agent_path_validation_precedes_reference_lookups(tmp_path, monkeypatch, agent):
    monkeypatch.setattr(seam._cfg, "requirements_path", lambda *args: pytest.fail("Unsafe agent reached filesystem lookup"))
    result = seam.diagnose(agent=agent, instance_id="case", reference_root=tmp_path)
    assert result.available is False and "invalid agent id" in result.reason
