"""Public Pi loading follows native trees and JSON-stream completion rules."""

import hashlib
import json
import stat
import zipfile

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.loaders import load_trajectory


def _header(**fields):
    return {"type": "session", "version": 3, "id": "pi-test", "timestamp": "2026-11-18T02:00:00Z",
            "cwd": "/workspace", **fields}


def _message(entry_id, parent, text, *, total=0):
    message = {"role": "assistant" if total else "user", "content": [{"type": "text", "text": text}],
               "timestamp": 1794967200000}
    if total:
        message.update(usage={"input": total - 2, "output": 2, "cacheRead": 0, "cacheWrite": 0,
                              "totalTokens": total, "cost": {"total": total / 1000}}, stopReason="stop")
    return {"type": "message", "id": entry_id, "parentId": parent, "timestamp": "2026-11-18T02:00:01Z", "message": message}


def _write(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def _branched():
    return [_header(), _message("user", None, "Compare implementations"),
            _message("old", "user", "Abandoned answer", total=100),
            {"type": "usage", "id": "warm", "parentId": "old", "usage": {"cacheRead": 20, "cost": {"total": 0.02}}},
            _message("new", "user", "Current answer", total=30),
            {"type": "label", "id": "bookmark", "parentId": "new", "targetId": "new", "label": "chosen"}]


def test_public_load_selects_active_branch_and_retains_full_session_usage(tmp_path):
    events = _branched()
    source = _write(tmp_path / "session.jsonl", events)
    run = tv.load(source)
    assert [step["text_preview"] for step in run.steps] == ["Compare implementations", "Current answer"]
    assert run.metrics["tokens"]["total"] == 30
    assert run.raw["metadata"]["history_scope"] == "active_branch"
    assert run.raw["metadata"]["pi_active_leaf_id"] == "bookmark"
    assert run.raw["metadata"]["pi_excluded_entry_count"] == 2
    usage = run.raw["metadata"]["pi_session_usage"]
    assert usage["tokens"] == {"input": 126, "output": 4, "cacheRead": 20, "cacheWrite": 0, "total": 150}
    assert usage["cost"] == pytest.approx(0.15)
    assert run.raw["_pi_entries"] == events
    assert any("other branches" in warning for warning in run.warnings)
    assert run.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    exported = tv.export_json(run, tmp_path / "export")
    assert exported
    reloaded = tv.load(tmp_path / "export" / "trajectory.json")
    assert reloaded.metrics["tokens"] == run.metrics["tokens"]
    assert reloaded.raw["_pi_entries"] == events


def test_legacy_and_mixed_link_exports_keep_their_linear_history(tmp_path):
    events = _branched()
    for event in events:
        event.pop("parentId", None)
    legacy = tv.load(_write(tmp_path / "legacy.jsonl", events))
    assert len(legacy.steps) == 4  # user, two assistant calls and auxiliary usage
    assert legacy.metrics["tokens"]["total"] == 150
    events[-1]["parentId"] = "new"
    mixed = tv.load(_write(tmp_path / "mixed.jsonl", events))
    assert mixed.metrics["tokens"]["total"] == 150
    assert any("links are incomplete" in warning for warning in mixed.warnings)


def test_full_session_usage_includes_native_auxiliary_and_nested_usage(tmp_path):
    events = [_header(), _message("answer", None, "Done", total=10)]
    for index, kind in enumerate(("usage", "compaction", "branch_summary", "message")):
        usage = {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "cost": {"total": 0.01}}
        event = {"type": kind, "id": f"extra-{index}", "parentId": "answer", "usage": usage}
        if kind == "message":
            event = {**event, "message": {"role": "toolResult", "toolCallId": "nested", "toolName": "task",
                                          "content": [{"type": "text", "text": "Nested call complete"}], "usage": usage}}
        events.append(event)
    run = tv.load(_write(tmp_path / "native-usage.jsonl", events))
    usage = run.raw["metadata"]["pi_session_usage"]
    assert usage["tokens"] == {"input": 12, "output": 10, "cacheRead": 12, "cacheWrite": 16, "total": 50}
    assert usage["cost"] == pytest.approx(0.05)
    assert run.metrics["tokens"]["total"] == 20  # active answer and nested usage; siblings are historical


def test_partial_parent_and_cycles_stop_without_reading_unrelated_history(tmp_path):
    missing = [_header(), _message("unrelated", None, "Other root", total=100),
               _message("answer", "unavailable", "Partial answer", total=30)]
    run = tv.load(_write(tmp_path / "partial.jsonl", missing))
    assert run.metrics["tokens"]["total"] == 30
    assert any("unavailable parent" in warning for warning in run.warnings)
    cycle = [_header(), _message("one", "two", "One", total=10), _message("two", "one", "Two", total=20)]
    run = tv.load(_write(tmp_path / "cycle.jsonl", cycle))
    assert run.metrics["tokens"]["total"] == 30
    assert any("cycle" in warning for warning in run.warnings)


def test_fork_path_is_provenance_without_external_file_discovery(tmp_path):
    parent = _write(tmp_path / "parent.jsonl", [_header(), _message("parent", None, "Private parent content", total=100)])
    child = _write(tmp_path / "fork.jsonl", [_header(id="fork", parentSession=str(parent)),
                                           _message("child", None, "Fork content", total=20)])
    run = tv.load(child)
    assert run.metrics["tokens"]["total"] == 20
    assert [step["text_preview"] for step in run.steps] == ["Fork content"]
    assert run.raw["_pi_entries"][0]["parentSession"] == str(parent)
    assert run.raw["_input_files"] == [str(child.resolve())]


def test_zip_loads_a_timestamp_named_pi_session_and_rejects_ambiguous_sessions(tmp_path):
    events = _branched()
    encoded = "\n".join(json.dumps(event) for event in events) + "\n"
    archive = tmp_path / "session.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sessions/2026-11-18T02-00-00_pi-test.jsonl", encoded)
        zf.writestr("README.txt", "One native Pi session")
    run = tv.load(archive)
    assert run.format == "pi" and run.metrics["tokens"]["total"] == 30
    assert run.raw["_input_files"] == [str(archive.resolve())]
    assert run.raw["_pi_entries"] == events
    assert load_trajectory(str(archive), format_hint="codex")["_error_code"] == "mismatch"
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("second.jsonl", encoded)
    assert "multiple Pi sessions" in load_trajectory(str(archive))["_error"]


def test_zip_never_extracts_or_follows_unsafe_member_paths(tmp_path):
    encoded = json.dumps(_header()) + "\n" + json.dumps(_message("answer", None, "Answer", total=10)) + "\n"
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.jsonl", encoded)
        zf.writestr("/absolute.jsonl", encoded)
        link = zipfile.ZipInfo("symlink.jsonl")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, encoded)
        zf.writestr("native.jsonl", encoded)
    run = tv.load(archive)
    assert run.metrics["tokens"]["total"] == 10
    assert not (tmp_path.parent / "escape.jsonl").exists()


def test_json_progress_stream_uses_final_messages_once(tmp_path):
    user = _message("user", None, "Inspect source")["message"]
    answer = _message("answer", "user", "Done", total=25)["message"]
    answer["durationMs"] = 350
    stream = [_header(), {"type": "agent_start"}, {"type": "message_start", "message": user},
              {"type": "message_end", "message": user}, {"type": "message_start", "message": answer},
              {"type": "message_update", "usage": answer["usage"], "assistantMessageEvent": {"type": "text_delta", "delta": "D"}},
              {"type": "message_end", "message": answer}, {"type": "turn_end", "message": answer, "toolResults": []},
              {"type": "agent_end", "messages": [answer]}, {"type": "agent_settled", "aborted": False}]
    run = tv.load(_write(tmp_path / "stream.jsonl", stream))
    assert len(run.steps) == 2
    assert run.metrics["tokens"]["total"] == 25
    assert run.steps[1]["duration"] == 0.35
    assert run.raw["metadata"]["history_scope"] == "event_stream"
    assert run.raw["metadata"]["pi_session_usage"]["tokens"]["total"] == 25
    assert run.raw["_pi_entries"] == stream


def test_unfinished_json_stream_reports_completed_scope(tmp_path):
    answer = _message("answer", None, "Incomplete", total=25)["message"]
    stream = [_header(), {"type": "message_start", "message": answer}]
    assert "no completed messages" in load_trajectory(str(_write(tmp_path / "empty.jsonl", stream)))["_error"]
    stream = [_header(), {"type": "message_start", "message": answer}, {"type": "message_end", "message": answer},
              {"type": "message_start", "message": {**answer, "stopReason": "pending"}}]
    run = tv.load(_write(tmp_path / "partial-stream.jsonl", stream))
    assert run.metrics["tokens"]["total"] == 25
    assert any("unfinished messages" in warning for warning in run.warnings)
