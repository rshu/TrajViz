"""CodeAgent session-export loader."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.insight.context_usage import detect_compaction_events
from agent_trajviz.insight.formats.codeagent import _convert_codeagent_to_internal
from agent_trajviz.insight.loaders import detect_format, load_trajectory
from agent_trajviz.insight.parser import parse_steps
from agent_trajviz.insight.presenters import trajectory_format_label
from agent_trajviz.insight.tool_failure import tool_call_failed


FIXTURE = Path(__file__).parent / "fixtures" / "codeagent_minimal.json"


class CodeAgentLoaderTests(unittest.TestCase):
    def test_format_is_detected_before_generic_opencode(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(detect_format(raw), "codeagent")
        self.assertEqual(detect_format({"info": {"id": "ses_open"}, "messages": []}), "opencode")

    def test_exported_at_alone_is_not_codeagent(self):
        raw = {
            "exported_at": "2026-01-01T00:00:00",
            "session": {
                "id": "ses_x",
                "messages": [
                    {"id": "m1", "data": {"role": "user", "time": {"created": 1}}, "parts": []},
                ],
            },
        }
        self.assertEqual(detect_format(raw), "unknown")

    def test_empty_child_sessions_still_detects_with_typed_parts(self):
        raw = {
            "session": {
                "id": "ses_x",
                "messages": [
                    {
                        "id": "m1",
                        "data": {"role": "user", "time": {"created": 1}},
                        "parts": [{"id": "p1", "data": {"type": "text", "text": "hi"}}],
                    },
                ],
            },
            "child_sessions": [],
        }
        self.assertEqual(detect_format(raw), "codeagent")

    def test_typed_part_late_in_transcript_still_detects(self):
        early = [
            {"id": f"m{i}", "data": {"role": "user"}, "parts": []}
            for i in range(15)
        ]
        early.append({
            "id": "m_late",
            "data": {"role": "assistant"},
            "parts": [{"id": "p", "data": {"type": "text", "text": "late"}}],
        })
        raw = {"session": {"id": "ses_x", "messages": early}, "child_sessions": []}
        self.assertEqual(detect_format(raw), "codeagent")

    def test_loader_preserves_product_identity_and_hierarchy(self):
        loaded = load_trajectory(str(FIXTURE))

        self.assertNotIn("_error", loaded)
        self.assertEqual(detect_format(loaded), "codeagent")
        self.assertTrue(loaded.get("_codeagent_format"))
        self.assertEqual(loaded["_source_format"], "codeagent")
        self.assertEqual(trajectory_format_label("codeagent"), "CodeAgent")
        self.assertEqual(loaded["metadata"]["agent"], "codeagent")
        self.assertEqual(loaded["metadata"]["generator_name"], "codeagent")
        self.assertEqual(loaded["metadata"]["session_id"], "ses_root")
        self.assertEqual(loaded["metadata"]["directory"], "/anonymized/workdir")
        self.assertEqual(loaded["metadata"]["model"], "vibe-coding")
        self.assertEqual(loaded["metadata"]["session_count"], 2)
        self.assertEqual(loaded["metadata"]["sub_agent_count"], 1)
        self.assertEqual(loaded["token_usage"]["total_tokens"], 150)
        self.assertEqual(loaded["token_usage"]["prompt_tokens"], 60)
        self.assertEqual(loaded["token_usage"]["completion_tokens"], 30)
        self.assertEqual(loaded["token_usage"]["reasoning_tokens"], 15)
        self.assertEqual(loaded["token_usage"]["cache_read_tokens"], 45)
        self.assertTrue(loaded["_capabilities"]["has_runtime_token_usage"])
        self.assertEqual(loaded["stats"]["failed_tool_calls"], 1)

    def test_steps_include_parent_tools_child_session_and_failures(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        roles = [step["role"] for step in steps]
        self.assertGreaterEqual(roles.count("user"), 2)
        self.assertGreaterEqual(roles.count("assistant"), 2)

        task = next(
            call
            for step in steps
            for call in step["tool_calls"]
            if call["tool_id"] == "call_task"
        )
        self.assertEqual(task["tool_name"], "task")
        self.assertEqual(task["status"], "completed")

        failed = next(
            call
            for step in steps
            for call in step["tool_calls"]
            if call["tool_id"] == "call_skill_fail"
        )
        self.assertEqual(failed["tool_name"], "skill")
        self.assertTrue(tool_call_failed(failed))
        self.assertIn("Skill not found", str(failed.get("error") or ""))

        child = next(step for step in steps if step.get("session_id") == "ses_child")
        self.assertTrue(child.get("is_sub_agent"))
        self.assertEqual(child.get("parent_session_id"), "ses_root")
        self.assertEqual(child.get("session_depth"), 1)

        read = next(
            call
            for step in steps
            for call in step["tool_calls"]
            if call["tool_id"] == "call_read"
        )
        self.assertEqual(read["tool_name"], "read")
        self.assertIn("guard", read["output"])

    def test_compaction_and_patch_parts_load(self):
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        compact_step = next(
            step
            for step in steps
            if any(part.get("type") == "compaction" for part in step.get("parts") or [])
        )
        self.assertEqual(compact_step["role"], "user")
        self.assertTrue(
            any(part.get("type") == "patch" for part in compact_step.get("parts") or [])
        )
        events = detect_compaction_events(steps)
        self.assertTrue(any(event.get("kind") == "compaction_part" for event in events))

    def test_explicit_format_codeagent_loads(self):
        traj = tv.load(str(FIXTURE), format="codeagent")
        self.assertEqual(traj.format, "codeagent")
        self.assertGreaterEqual(len(traj.steps), 4)

    def test_malformed_messages_without_data_are_skipped(self):
        converted = _convert_codeagent_to_internal({
            "session": {
                "id": "ses_x",
                "time_created": 1,
                "time_updated": 2,
                "messages": [
                    {"id": "bad", "parts": []},
                    {
                        "id": "ok",
                        "data": {"role": "user", "time": {"created": 1}},
                        "parts": [{"id": "p", "data": {"type": "text", "text": "hi"}}],
                    },
                ],
            },
            "child_sessions": [],
        })
        self.assertNotIn("_error", converted)
        roles = [message["info"]["role"] for message in converted["messages"]]
        self.assertEqual(roles, ["user"])
        self.assertNotIn("?", roles)

    def test_grandchild_session_depth(self):
        converted = _convert_codeagent_to_internal({
            "session": {
                "id": "ses_root",
                "time_created": 1,
                "time_updated": 9,
                "messages": [
                    {
                        "id": "m0",
                        "data": {"role": "user", "time": {"created": 1}},
                        "parts": [{"id": "p0", "data": {"type": "text", "text": "root"}}],
                    },
                ],
            },
            "child_sessions": [
                {
                    "id": "ses_child",
                    "parent_id": "ses_root",
                    "title": "child",
                    "messages": [
                        {
                            "id": "m1",
                            "data": {"role": "assistant", "time": {"created": 2}},
                            "parts": [{"id": "p1", "data": {"type": "text", "text": "child"}}],
                        },
                    ],
                },
                {
                    "id": "ses_grand",
                    "parent_id": "ses_child",
                    "title": "grand",
                    "messages": [
                        {
                            "id": "m2",
                            "data": {"role": "assistant", "time": {"created": 3}},
                            "parts": [{"id": "p2", "data": {"type": "text", "text": "grand"}}],
                        },
                    ],
                },
            ],
        })
        by_session = {
            message["info"]["sessionID"]: message["info"]
            for message in converted["messages"]
            if message["info"].get("role") == "assistant"
        }
        self.assertEqual(by_session["ses_child"]["sessionDepth"], 1)
        self.assertEqual(by_session["ses_grand"]["sessionDepth"], 2)
        self.assertEqual(by_session["ses_grand"]["parentSessionID"], "ses_child")
        depths = {entry["info"]["id"]: entry["depth"] for entry in converted["session_manifest"]}
        self.assertEqual(depths["ses_grand"], 2)


if __name__ == "__main__":
    unittest.main()
