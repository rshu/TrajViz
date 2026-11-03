"""OpenCode desktop / Windows session exports use a flat message schema.

The SQLite consolidator writes ``info`` + ``parts``. The desktop export writes
``type`` plus ``text`` or ``content``, and keeps the working directory under
``info.location``. Those turns used to load as role ``?`` with empty parts, so
the workflow cards had no text.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent_trajviz.insight.loaders import load_trajectory
from agent_trajviz.insight.parser import parse_steps


def _write(doc: dict) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(doc, handle)
        return handle.name


class FlatOpenCodeExportTests(unittest.TestCase):
    def test_windows_session_export_fills_workflow_steps(self):
        path = _write({
            "info": {
                "id": "ses_win",
                "location": {"directory": "C:\\Users\\user\\Documents\\Default Project"},
                "model": {"id": "big-pickle", "providerID": "opencode"},
                "time": {"created": 1_000, "updated": 5_000},
            },
            "messages": [
                {
                    "id": "u1", "type": "user", "text": "hello",
                    "time": {"created": 1_000},
                    "metadata": {"agent": "build", "model": {"modelID": "big-pickle", "providerID": "opencode"}},
                },
                {
                    "id": "a1", "type": "assistant", "agent": "build",
                    "model": {"id": "big-pickle", "providerID": "opencode"},
                    "time": {"created": 1_100, "completed": 5_034},
                    "finish": "stop",
                    "tokens": {"input": 3162, "output": 40, "reasoning": 43, "cache": {"read": 1017, "write": 0}},
                    "content": [{"type": "text", "text": "Yes, this is Windows."}],
                },
                {"id": "i1", "type": "idle", "outcome": "succeeded", "time": {"created": 5_034}},
                {"id": "s1", "type": "system", "text": "tool catalog", "time": {"created": 6_000}},
                {
                    "id": "u2", "type": "user", "text": "meaning",
                    "time": {"created": 7_000},
                },
                {
                    "id": "a2", "type": "assistant", "agent": "build",
                    "time": {"created": 7_100, "completed": 8_000},
                    "finish": "stop",
                    "tokens": {"input": 39, "output": 223, "reasoning": 466, "cache": {"read": 6582, "write": 0}},
                    "content": [{"type": "text", "text": "It depends \u2014 on you."}],
                },
            ],
        })
        self.addCleanup(os.remove, path)
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw)
        steps = parse_steps(raw)

        self.assertEqual([step["role"] for step in steps], ["user", "assistant", "system", "user", "assistant"])
        self.assertEqual(steps[0]["text_preview"], "hello")
        self.assertIn("Windows", steps[1]["text_preview"])
        self.assertEqual(steps[1]["model_id"], "big-pickle")
        self.assertEqual(steps[1]["provider_id"], "opencode")
        self.assertEqual(steps[1]["finish"], "stop")
        self.assertEqual(steps[1]["agent"], "build")
        self.assertEqual(steps[1]["tokens"]["input"], 3162)
        self.assertEqual(steps[1]["tokens"]["output"], 40)
        self.assertEqual(steps[1]["tokens"]["reasoning"], 43)
        self.assertEqual(steps[1]["tokens"]["cache_read"], 1017)
        self.assertEqual(steps[1]["tokens"]["total"], 3162 + 40 + 1017)
        self.assertAlmostEqual(steps[1]["duration"], 3.93)
        self.assertIn("\u2014", steps[-1]["text_preview"])

        self.assertEqual(raw["metadata"]["session_id"], "ses_win")
        self.assertEqual(raw["metadata"]["directory"], r"C:\Users\user\Documents\Default Project")
        self.assertEqual(raw["metadata"]["directory_name"], "Default Project")
        self.assertEqual(raw["metadata"]["model"], "big-pickle")
        self.assertEqual(raw["stats"]["user_messages"], 2)
        self.assertEqual(raw["stats"]["assistant_messages"], 2)
        self.assertEqual(raw["token_usage"]["prompt_tokens"], 3201)
        self.assertEqual(raw["token_usage"]["completion_tokens"], 263)
        self.assertNotIn("idle", [step["role"] for step in steps])

    def test_flat_tool_content_and_idle_rows(self):
        path = _write({
            "info": {
                "id": "ses_x",
                "location": {"directory": "C:\\work\\proj"},
                "model": {"id": "big-pickle", "providerID": "opencode"},
                "time": {"created": 1000, "updated": 5000},
            },
            "messages": [
                {
                    "id": "u1", "type": "user", "text": "run tests",
                    "time": {"created": 1000},
                    "metadata": {
                        "agent": "build",
                        "model": {"modelID": "big-pickle", "providerID": "opencode"},
                    },
                },
                {
                    "id": "a1", "type": "assistant", "agent": "build",
                    "model": {"id": "big-pickle", "providerID": "opencode"},
                    "time": {"created": 1100, "completed": 2000},
                    "finish": "tool-calls",
                    "tokens": {
                        "input": 10, "output": 4, "reasoning": 1,
                        "cache": {"read": 6, "write": 0},
                    },
                    "content": [
                        {"type": "text", "text": "running"},
                        {
                            "type": "tool", "tool": "bash", "callID": "c1",
                            "state": {
                                "status": "completed",
                                "input": {"command": "pytest"},
                                "output": "ok",
                                "title": "pytest",
                            },
                        },
                    ],
                },
                {"id": "i1", "type": "idle", "outcome": "succeeded", "time": {"created": 2000}},
            ],
        })
        self.addCleanup(os.remove, path)
        raw = load_trajectory(path)
        steps = parse_steps(raw)
        self.assertEqual([step["role"] for step in steps], ["user", "assistant"])
        self.assertEqual(steps[1]["text_preview"], "running")
        self.assertEqual(steps[1]["tool_call_count"], 1)
        self.assertEqual(steps[1]["tool_calls"][0]["tool_name"], "bash")
        self.assertEqual(steps[1]["tool_calls"][0]["status"], "completed")
        self.assertEqual(steps[1]["tokens"]["total"], 20)
        self.assertEqual(raw["metadata"]["directory_name"], "proj")
        self.assertEqual(raw["metadata"]["model"], "big-pickle")
        self.assertEqual(raw["stats"]["total_tool_calls"], 1)

    def test_consolidator_info_parts_shape_is_unchanged(self):
        path = _write({
            "info": {
                "id": "s1",
                "directory": "/work/proj",
                "time": {"created": 0, "updated": 10_000},
            },
            "messages": [{
                "role": "assistant",
                "info": {
                    "id": "m0",
                    "role": "assistant",
                    "agent": "build",
                    "model": {"modelID": "gpt"},
                    "tokens": {"input": 3, "output": 4, "total": 7, "cache": {"read": 0, "write": 0}},
                    "time": {"created": 0, "completed": 1000},
                },
                "parts": [{"type": "text", "text": "hello"}],
            }],
        })
        self.addCleanup(os.remove, path)
        raw = load_trajectory(path)
        steps = parse_steps(raw)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["role"], "assistant")
        self.assertEqual(steps[0]["text_preview"], "hello")
        self.assertEqual(steps[0]["tokens"]["total"], 7)
        self.assertEqual(steps[0]["model_id"], "")
        self.assertEqual(raw["metadata"]["directory"], "/work/proj")
        self.assertEqual(raw["metadata"]["directory_name"], "proj")
        self.assertEqual(raw["metadata"]["model"], "gpt")
        self.assertEqual(raw["stats"]["assistant_messages"], 1)


if __name__ == "__main__":
    unittest.main()
