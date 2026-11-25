"""Native request history must not confuse grouped streams with chronology."""

from __future__ import annotations

import copy

from agent_trajviz.insight.prompt_construction import inspect_prompt_construction


def capture():
    rows = []
    for agent, requests in (("", [("parent-one", 10, None), ("parent-two", 40, "parent-one")]),
                            ("child", [("child-one", 20, None), ("child-two", 30, "child-one")])):
        records = [
            {"type": "api-request-shape", "shapeHash": "shape", "shape": {"system": "Recorded rules", "tools": []}},
            {"type": "api-request-blob", "hash": "task", "message": {"role": "user", "content": "Task"}},
        ]
        for identifier, seconds, base in requests:
            records.append({"type": "api-request", "id": identifier, "shapeHash": "shape",
                            "timestamp": f"2026-11-25T00:00:{seconds:02d}Z",
                            "tail": [] if base else ["task"], "keep": 1 if base else 0,
                            **({"base": base} if base else {})})
        rows.extend({"record": record, "session_id": "session", "agent_id": agent,
                     "source_path": f"{agent or 'main'}.jsonl", "raw_index": index}
                    for index, record in enumerate(records))
    return {"_claude_prompt_records": rows}


def test_interleaved_native_agents_follow_recorded_time_and_default_to_latest_request():
    data = inspect_prompt_construction(capture(), [])
    assert [row["id"].rsplit(":", 1)[-1] for row in data["requests"]] == [
        "parent-one", "child-one", "child-two", "parent-two"]
    assert data["request_order"] == {"basis": "recorded_timestamp", "chronological": True, "limitations": []}
    assert data["selected_request"]["id"] == data["requests"][-1]["id"]
    assert data["selected_request"]["changes"]["previous_request_id"] == data["requests"][0]["id"]
    assert [row["id"] for row in data["evolution"]] == [row["id"] for row in data["requests"]]
    child = inspect_prompt_construction(capture(), [], data["requests"][2]["id"])["selected_request"]
    assert child["changes"]["previous_request_id"] == data["requests"][1]["id"]


def test_missing_or_conflicting_clocks_preserve_source_order_and_explain_why():
    for timestamp in (None, "2026-11-25T00:00:01Z"):
        raw = capture()
        raw["_claude_prompt_records"][3]["record"]["timestamp"] = timestamp
        data = inspect_prompt_construction(raw, [])
        assert not data["request_order"]["chronological"] and data["request_order"]["limitations"]
        assert data["request_order"]["basis"] == "source_order"
        assert [row["id"].rsplit(":", 1)[-1] for row in data["requests"]] == [
            "parent-one", "parent-two", "child-one", "child-two"]


def test_presentation_sort_never_uses_a_later_blob_to_repair_earlier_request():
    raw = capture()
    raw["_claude_prompt_records"][2]["record"]["tail"] = ["future"]
    raw["_claude_prompt_records"].append({**copy.deepcopy(raw["_claude_prompt_records"][1]),
        "raw_index": 4, "record": {"type": "api-request-blob", "hash": "future",
                                    "message": {"role": "user", "content": "Later recorded text"}}})
    data = inspect_prompt_construction(raw, [])
    earlier = inspect_prompt_construction(raw, [], data["requests"][0]["id"])["selected_request"]
    assert "blob:future" in earlier["missing_references"] and earlier["messages"] == []


def test_equal_timestamps_keep_source_order_within_each_context():
    raw = capture()
    for row in raw["_claude_prompt_records"]:
        if row["record"]["type"] == "api-request":
            row["record"]["timestamp"] = "2026-11-25T00:00:10Z"
    data = inspect_prompt_construction(raw, [])
    assert data["request_order"]["chronological"]
    assert [row["id"].rsplit(":", 1)[-1] for row in data["requests"]] == [
        "parent-one", "parent-two", "child-one", "child-two"]
