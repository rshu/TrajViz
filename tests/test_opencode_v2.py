"""OpenCode 2.1.4 native contracts verified against its released schema."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import agent_trajviz as tv
from agent_trajviz.collect.opencode import export_session_and_collect_children
from agent_trajviz.insight.formats.opencode import _convert_opencode_metadata


def capture() -> dict:
    # https://github.com/anomalyco/opencode/blob/v2.1.4/packages/schema/src/session-message.ts
    return {
        "info": {"id": "ses_root", "title": "Controlled native v2 example", "version": "2.1.4",
                 "location": {"directory": "/workspace"}, "model": {"id": "example-model", "providerID": "example"},
                 "tokens": {"input": 999999}, "time": {"created": 1000, "updated": 5000}},
        "messages": [
            {"id": "msg_user", "type": "user", "time": {"created": 1000}, "text": "Review the parser.", "files": []},
            {"id": "msg_assistant", "type": "assistant", "agent": "build", "time": {"created": 1100, "streamed": 1900, "completed": 4500},
             "model": {"id": "example-model", "providerID": "example"}, "finish": "stop",
             "tokens": {"input": 10, "output": 4, "reasoning": 2, "cache": {"read": 20, "write": 3}},
             "content": [
                 {"type": "reasoning", "text": "Check the evidence.", "time": {"created": 1100, "completed": 1500}},
                 {"type": "tool", "id": "call_shell", "name": "shell", "time": {"created": 1500, "ran": 2000, "completed": 4000},
                  "state": {"status": "completed", "input": {"command": "pytest"}, "content": [{"type": "text", "text": "3 passed"}]}},
                 {"type": "text", "text": "The focused tests pass."},
             ]},
            {"id": "msg_idle", "type": "idle", "time": {"created": 5000}, "outcome": "succeeded"},
        ],
    }


class NativeOpenCodeV2Tests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.json"
            path.write_text(json.dumps(data))
            return tv.load(path)

    def test_typed_records_keep_prompt_model_usage_and_tool_timing(self):
        run = self.load(capture())
        self.assertEqual(run.format, "opencode")
        self.assertEqual([s["role"] for s in run.steps], ["user", "assistant"])
        self.assertIn("Review the parser.", run.steps[0]["text_preview"])
        self.assertEqual(run.steps[1]["model_id"], "example-model")
        self.assertEqual(run.raw["metadata"]["directory"], "/workspace")
        self.assertEqual(run.raw["metadata"]["model"], "example-model")
        self.assertEqual(run.metrics["tokens"]["total"], 39)
        self.assertEqual(run.raw["token_usage"]["total_tokens"], 39)
        tool = run.steps[1]["tool_calls"][0]
        self.assertEqual(tool["tool_name"], "shell")
        self.assertEqual(tool["output"], "3 passed")
        self.assertEqual(tool["time_end"] - tool["time_start"], 2000)
        self.assertEqual(run.steps[1]["tokens"]["reasoning"], 2)
        self.assertEqual(run.steps[1]["duration"], 0.8)
        self.assertEqual(run.steps[1]["timing_source"], "model_request")
        self.assertEqual(run.steps[1]["time_completed_ms"], 1900)
        from agent_trajviz.insight.metrics import generation_seconds
        self.assertEqual(generation_seconds(run.steps[1]), (0.8, 0.0))
        self.assertEqual(run.raw["_opencode_v2_messages"][1]["time"]["completed"], 4500)
        self.assertEqual(run.raw["_opencode_v2_messages"][-1]["type"], "idle")

    def test_failed_model_is_visible_without_fake_lifecycle_turn(self):
        data = capture()
        data["messages"][1].update({"content": [], "finish": "error", "error": {"type": "provider.auth", "message": "Controlled provider failure", "status": 401}})
        data["messages"][1].pop("tokens")
        data["messages"][-1].update({"outcome": "failed", "error": data["messages"][1]["error"]})
        run = self.load(data)
        self.assertEqual(len(run.steps), 2)
        self.assertEqual(run.steps[-1]["finish"], "error")
        self.assertEqual(run.metrics["tool_call_count"], 0)
        self.assertEqual(len(run.warnings), 1)
        self.assertIn("Controlled provider failure", run.warnings[0])
        self.assertIn("Controlled provider failure", run.raw["output"]["error"])

    def test_versionless_failure_preserves_attachments_and_error_evidence(self):
        data = capture()
        del data["info"]["version"]
        user, assistant, idle = data["messages"]
        # Some desktop exports nest model information in user metadata. A
        # native-only attachment and provider error still take precedence.
        user["metadata"] = {"model": {"modelID": "example-model"}}
        user["files"] = [{"path": "input.txt", "mime": "text/plain"}]
        del assistant["time"]["streamed"]
        error = {"type": "provider.auth", "message": "Controlled provider failure", "status": 401}
        assistant.update(content=[], finish="error", error=error)
        idle.update(outcome="failed", error=error)
        run = self.load(data)
        self.assertEqual([step["role"] for step in run.steps], ["user", "assistant"])
        self.assertEqual(run.steps[0]["parts"][-1]["raw"]["data"]["path"], "input.txt")
        self.assertEqual(run.raw["output"]["error"], error["message"])
        self.assertIn(error["message"], run.warnings[0])
        self.assertEqual(run.raw["_opencode_v2_messages"], data["messages"])

    def test_versionless_text_turn_keeps_native_token_accounting(self):
        data = capture()
        del data["info"]["version"]
        user, assistant, _idle = data["messages"]
        del user["files"]
        del assistant["time"]["streamed"]
        assistant["content"] = [{"type": "text", "text": "The focused tests pass."}]
        run = self.load(data)
        self.assertEqual(run.steps[1]["tokens"]["total"], 39)
        self.assertEqual(run.raw["_opencode_v2_messages"], data["messages"])

    def test_resumed_success_keeps_error_history_without_terminal_failure(self):
        data = capture()
        data["messages"][1]["content"][1]["state"]["status"] = "error"
        data["messages"][1]["content"][1]["state"]["error"] = {"message": "Controlled old tool failure"}
        data["messages"][-1].update({"outcome": "interrupted", "error": {"message": "Earlier interruption"}})
        data["messages"].extend([
            {"id": "resumed_user", "type": "user", "text": "Continue.", "time": {"created": 5100}},
            {"id": "resumed_assistant", "type": "assistant", "agent": "build", "finish": "stop", "content": [{"type": "text", "text": "Finished."}],
             "time": {"created": 5200, "streamed": 5900, "completed": 6000}},
            {"id": "resumed_idle", "type": "idle", "outcome": "succeeded", "time": {"created": 6100}},
        ])
        run = self.load(data)
        self.assertIsNone(run.raw["output"]["error"])
        self.assertIsNone(run.metrics["patch_error"])
        self.assertIn("Earlier interruption", " ".join(run.warnings))
        self.assertEqual(run.metrics["tool_fail"], 1)
        self.assertEqual(run.raw["_opencode_v2_messages"][2]["outcome"], "interrupted")
        # A different session's terminal outcome cannot replace the root's.
        data["messages"].append({"id": "child_idle", "type": "idle", "sessionID": "child", "outcome": "failed",
                                 "error": {"message": "Child failure"}, "time": {"created": 6200}})
        run = self.load(data)
        self.assertIsNone(run.metrics["patch_error"])
        self.assertIn("Child failure", " ".join(run.warnings))
        # The current root still reports a genuinely failed terminal outcome.
        data["messages"].append({"id": "root_failed", "type": "idle", "outcome": "failed",
                                 "error": {"message": "Final root failure"}, "time": {"created": 6300}})
        run = self.load(data)
        self.assertEqual(run.metrics["patch_error"], "Final root failure")
        self.assertIn("Earlier interruption", " ".join(run.warnings))

    def test_resumed_in_progress_turn_does_not_inherit_previous_terminal_error(self):
        data = capture()
        data["messages"][-1]["outcome"] = "interrupted"
        data["messages"].extend([
            {"id": "resumed_user", "type": "user", "text": "Continue.", "time": {"created": 5100}},
            {"id": "resumed_assistant", "type": "assistant", "agent": "build", "content": [], "time": {"created": 5200}},
        ])
        run = self.load(data)
        self.assertIsNone(run.metrics["patch_error"])
        self.assertTrue(any("incomplete" in warning for warning in run.warnings))
        self.assertIn("Session turn interrupted.", run.raw["_opencode_v2_history_errors"])
        self.assertFalse(any("Session turn interrupted." in warning for warning in run.warnings))

    def test_tool_errors_keep_text_and_native_error_evidence(self):
        data = capture()
        tool = data["messages"][1]["content"][1]
        tool["state"] = {"status": "error", "input": {"command": "pytest"}, "content": [{"type": "text", "text": "failure output"}],
                         "error": {"type": "tool.execution", "message": "exit 1"}}
        run = self.load(data)
        call = run.steps[1]["tool_calls"][0]
        self.assertEqual(call["status"], "error")
        self.assertEqual(call["error"], "exit 1")
        self.assertEqual(call["output"], "failure output")
        self.assertEqual(call["metadata"]["opencode_v2_tool"]["state"]["error"]["type"], "tool.execution")

    def test_unfinished_tools_do_not_invent_success(self):
        for status in ["streaming", "running"]:
            with self.subTest(status=status):
                data = capture()
                data["messages"][1]["content"][1]["state"] = {"status": status, "input": {"command": "pytest"}}
                run = self.load(data)
                self.assertEqual(run.metrics["tool_call_count"], 1)
                self.assertEqual(run.metrics["tool_success"], 0)
                self.assertIsNone(run.metrics["tool_success_rate"])

    def test_idle_failure_or_interruption_is_visible_without_an_error_payload(self):
        for outcome in ["failed", "interrupted"]:
            with self.subTest(outcome=outcome):
                data = capture()
                data["messages"][-1]["outcome"] = outcome
                run = self.load(data)
                self.assertEqual(len(run.steps), 2)
                if outcome == "failed":
                    self.assertIn(outcome, " ".join(run.warnings))
                else:
                    self.assertFalse(run.warnings)
                self.assertIn(outcome, run.raw["output"]["error"])
                self.assertIn(outcome, " ".join(run.raw["_opencode_v2_history_errors"]))

    def test_shell_native_lifecycle_and_paged_output(self):
        for status, exit_code, successes, failures in [
            ("exited", 0, 1, 0), ("exited", 2, 0, 1), ("timeout", None, 0, 1),
            ("killed", None, 0, 1), ("running", None, 0, 0), ("exited", None, 0, 0),
        ]:
            with self.subTest(status=status, exit_code=exit_code):
                data = capture()
                data["messages"][1] = {"id": "shell", "type": "shell", "shellID": "shell_1", "command": "pytest",
                                       "status": status, "exit": exit_code, "time": {"created": 1100, "completed": 2200},
                                       "output": {"output": "local output", "cursor": 20, "size": 40, "truncated": True}}
                run = self.load(data)
                self.assertEqual(run.metrics["tool_success"], successes)
                self.assertEqual(run.metrics["tool_fail"], failures)
                self.assertEqual(run.steps[1]["tool_calls"][0]["output"], "local output")
                self.assertTrue(run.steps[1]["tool_calls"][0]["metadata"]["opencode_v2_shell"]["output"]["truncated"])
                if not successes and not failures:
                    self.assertIsNone(run.metrics["tool_success_rate"])

    def test_compaction_usage_is_not_a_conversation_turn(self):
        data = capture()
        data["messages"].insert(1, {"id": "compact", "type": "compaction", "status": "completed", "reason": "auto",
                                    "summary": "Preserve the task context.", "time": {"created": 1050},
                                    "tokens": {"input": 2, "output": 1, "reasoning": 0, "cache": {"read": 0, "write": 0}}})
        run = self.load(data)
        self.assertEqual(run.steps[1]["role"], "compaction")
        self.assertTrue(run.steps[1]["is_compaction_checkpoint"])
        self.assertEqual(run.steps[1]["compaction_reason"], "auto")
        self.assertEqual(run.metrics["assistant_turns"], 1)
        self.assertEqual(run.metrics["tokens"]["total"], 42)

    def test_pending_or_failed_compaction_is_not_a_completed_checkpoint(self):
        for status in ["running", "failed"]:
            with self.subTest(status=status):
                data = capture()
                record = {"id": "compact", "type": "compaction", "status": status, "reason": "auto",
                          "summary": "Pending context work.", "time": {"created": 1050}}
                if status == "failed":
                    record["error"] = {"type": "provider.error", "message": "Controlled compaction failure"}
                data["messages"].insert(1, record)
                run = self.load(data)
                self.assertEqual(run.steps[1]["role"], "system")
                self.assertFalse(run.steps[1]["is_compaction_checkpoint"])
                view = json.loads(run.views()["context_pressure"].to_json())
                self.assertEqual(view["data"]["stats"]["compaction_count"], 0)
                if status == "failed":
                    self.assertIn("Controlled compaction failure", " ".join(run.warnings))

    def test_file_attachments_snapshots_and_malformed_optional_content(self):
        data = capture()
        data["messages"][0]["files"] = [{"path": "input.txt", "mime": "text/plain"}]
        data["messages"][1]["snapshot"] = {"start": "before", "end": "after", "files": ["parser.py"]}
        run = self.load(data)
        self.assertEqual(run.steps[0]["parts"][-1]["raw"]["data"]["path"], "input.txt")
        self.assertEqual(run.steps[1]["parts"][-1]["data"]["files"], ["parser.py"])
        for content in [None, "bad", {}]:
            data["messages"][1]["content"] = content
            self.assertEqual(len(self.load(data).steps), 2)

    def test_normalization_is_idempotent_and_keeps_raw_capture(self):
        raw = capture()
        original = copy.deepcopy(raw["messages"])
        _convert_opencode_metadata(raw)
        first = copy.deepcopy(raw)
        _convert_opencode_metadata(raw)
        self.assertEqual(raw, first)
        self.assertEqual(raw["_opencode_v2_messages"], original)

    def test_full_and_light_roundtrip_and_all_view_schemas(self):
        import jsonschema
        from agent_trajviz.figures import figures

        run = self.load(capture())
        views = run.views()
        self.assertEqual(len(views), 36)
        for name, view in views.items():
            jsonschema.Draft202012Validator(tv.view_schema(name)).validate(json.loads(view.to_json()))
        self.assertEqual(len(figures(views)), 12)
        for raw in [False, True]:
            restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=raw), allow_nan=False)))
            self.assertEqual(restored.steps, run.steps)
            if raw:
                self.assertEqual(restored.metrics, run.metrics)


V2_SCHEMA = """
CREATE TABLE session_v2 (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, title TEXT, version TEXT,
 directory TEXT, time_created INTEGER, time_updated INTEGER, model TEXT);
CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER,
 time_created INTEGER, time_updated INTEGER, data TEXT);
"""


class NativeOpenCodeV2CollectionTests(unittest.TestCase):
    def database(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(V2_SCHEMA)
        for identity, parent, created in [("ses_root", None, 1000), ("ses_child", "ses_root", 2000),
                                          ("ses_grandchild", "ses_child", 3000), ("ses_unrelated", None, 4000)]:
            conn.execute("INSERT INTO session_v2 VALUES (?,?,?,?,?,?,?,?,?)",
                         (identity, "project", parent, identity, "2.1.4", "/workspace", created, 5000,
                          json.dumps({"id": "example-model", "providerID": "example"})))
            data = {"time": {"created": created}, "text": identity, "files": []}
            conn.execute("INSERT INTO session_message VALUES (?,?,?,?,?,?,?)",
                         ("msg_" + identity, identity, "user", 1, created, created, json.dumps(data)))
        return conn

    def test_collects_selected_family_only_with_native_ids_and_types(self):
        with self.database() as conn:
            result = export_session_and_collect_children(conn, "ses_root", set())
        self.assertEqual([m["sessionID"] for m in result["messages"]], ["ses_root", "ses_child", "ses_grandchild"])
        self.assertEqual([m["type"] for m in result["messages"]], ["user"] * 3)
        self.assertEqual(result["statistics"]["subagent_sessions"], 2)
        run = NativeOpenCodeV2Tests().load(result)
        self.assertEqual([s["is_sub_agent"] for s in run.steps], [False, True, True])
        self.assertEqual(run.raw["metadata"]["sub_agent_count"], 2)
        self.assertEqual([s["parent_session_id"] for s in run.steps], ["", "ses_root", "ses_child"])
        self.assertEqual([s["session_depth"] for s in run.steps], [0, 1, 2])

    def test_corrupt_projected_record_warns_without_losing_other_sessions(self):
        with self.database() as conn:
            conn.execute("INSERT INTO session_message VALUES (?,?,?,?,?,?,?)", ("msg_bad", "ses_root", "assistant", 2, 1500, 1500, "[]"))
            result = export_session_and_collect_children(conn, "ses_root", set())
        self.assertEqual(len(result["messages"]), 3)
        self.assertIn("msg_bad", result["_load_warnings"][0])

    def test_missing_change_counts_remain_unknown(self):
        with self.database() as conn:
            result = export_session_and_collect_children(conn, "ses_root", set())
        self.assertEqual(result["info"]["summary"], {})
        run = NativeOpenCodeV2Tests().load(result)
        self.assertIsNone(run.metrics["files_changed"])
        self.assertIsNone(run.metrics["churn"])

    def test_depth_limit_and_unknown_session(self):
        with self.database() as conn:
            result = export_session_and_collect_children(conn, "ses_root", set(), max_depth=1)
            self.assertEqual(len(result["messages"]), 1)
            self.assertIn("depth limit", result["_load_warnings"][0])
            with self.assertRaisesRegex(ValueError, "Session not found"):
                export_session_and_collect_children(conn, "missing", set())
