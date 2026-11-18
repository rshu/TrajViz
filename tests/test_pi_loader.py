"""Pi coding-agent JSONL loader."""

import json
import os
import tempfile
import unittest

from agent_trajviz.insight.loaders import detect_format, load_trajectory
from agent_trajviz.insight.metrics import extract_agent_info
from agent_trajviz.insight.parser import parse_steps


def _write(tmp: str, name: str, events: list) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


def _session(**extra):
    event = {
        "type": "session",
        "version": 3,
        "id": "sess-1",
        "timestamp": "2026-08-24T01:29:43.221Z",
        "cwd": "/home/user/proj",
    }
    event.update(extra)
    return event


def _msg(role, ts_iso, content, **msg_extra):
    message = {"role": role, "content": content, "timestamp": 1787535033022}
    message.update(msg_extra)
    return {
        "type": "message",
        "id": msg_extra.get("id", f"id-{role}"),
        "timestamp": ts_iso,
        "message": message,
    }


class PiLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.events = [
            _session(),
            {"type": "model_change", "id": "m1", "timestamp": "2026-08-24T01:30:18.190Z",
             "provider": "zenmux", "modelId": "z-ai/glm-5.3-free"},
            {"type": "thinking_level_change", "id": "t1",
             "timestamp": "2026-08-24T01:30:18.191Z", "thinkingLevel": "medium"},
            _msg("user", "2026-08-24T01:30:33.026Z",
                 [{"type": "text", "text": "explore the repo"}]),
            _msg("assistant", "2026-08-24T01:30:33.065Z", [],
                 stopReason="error",
                 errorMessage="No API key for provider: zenmux",
                 provider="zenmux", model="z-ai/glm-5.3-free",
                 usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                        "totalTokens": 0}),
            {"type": "model_change", "id": "m2", "timestamp": "2026-08-24T01:32:56.590Z",
             "provider": "cline", "modelId": "nvidia/nemotron-3.5-lightning:free"},
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                    {"type": "thinking", "thinking": "I should list the directory."},
                    {"type": "toolCall", "id": "call-1", "name": "bash",
                     "arguments": {"command": "ls /home/user/proj"}},
                 ],
                 stopReason="toolUse",
                 provider="cline", model="nvidia/nemotron-3.5-lightning:free",
                 usage={"input": 1728, "output": 87, "cacheRead": 0, "cacheWrite": 0,
                        "reasoning": 49, "totalTokens": 1815}),
            _msg("toolResult", "2026-08-24T01:33:03.369Z",
                 [{"type": "text", "text": "trajviz\nREADME.md"}],
                 toolCallId="call-1", toolName="bash", isError=False),
            _msg("assistant", "2026-08-24T01:33:10.013Z", [
                    {"type": "toolCall", "id": "call-2", "name": "read",
                     "arguments": {"path": "/home/user/proj/README.md"}},
                 ],
                 stopReason="toolUse",
                 usage={"input": 2000, "output": 40, "totalTokens": 2040}),
            _msg("toolResult", "2026-08-24T01:33:10.029Z",
                 [{"type": "text", "text": "# TrajViz"}],
                 toolCallId="call-2", toolName="read", isError=False),
            _msg("assistant", "2026-08-24T01:33:12.379Z", [
                    {"type": "toolCall", "id": "call-3", "name": "write",
                     "arguments": {"path": "/home/user/proj/DESIGN.md",
                                   "content": "# Design"}},
                 ],
                 stopReason="toolUse",
                 usage={"input": 2100, "output": 55, "totalTokens": 2155}),
            _msg("toolResult", "2026-08-24T01:33:12.390Z",
                 [{"type": "text", "text": "written"}],
                 toolCallId="call-3", toolName="write", isError=False),
            _msg("assistant", "2026-08-24T01:33:36.288Z", [
                    {"type": "thinking", "thinking": "Summarize for the user."},
                    {"type": "text", "text": "Here is an overview of the repo."},
                 ],
                 stopReason="stop",
                 usage={"input": 8962, "output": 782, "reasoning": 30,
                        "totalTokens": 9744}),
        ]

    def _load(self, events=None, name="session.jsonl"):
        return load_trajectory(_write(self.tmp, name, events or self.events))

    def test_detects_as_pi(self):
        raw = self._load()
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "pi")
        self.assertTrue(raw.get("_pi_format"))

    def test_does_not_steal_codex_jsonl(self):
        events = [
            {"type": "session_meta", "timestamp": "2026-01-05T12:00:00.000Z",
             "payload": {"id": "s1", "cwd": "/p", "model": "gpt-5"}},
        ]
        raw = self._load(events, name="codex.jsonl")
        self.assertEqual(detect_format(raw), "codex")

    def test_session_metadata(self):
        raw = self._load()
        self.assertEqual(raw["metadata"]["session_id"], "sess-1")
        self.assertEqual(raw["metadata"]["directory"], "/home/user/proj")
        self.assertEqual(raw["metadata"]["directory_name"], "proj")
        self.assertEqual(raw["metadata"]["agent"], "pi")
        self.assertEqual(raw["metadata"]["model"], "nvidia/nemotron-3.5-lightning:free")

    def test_user_and_assistant_steps(self):
        steps = parse_steps(self._load())
        roles = [s["role"] for s in steps]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        user = next(s for s in steps if s["role"] == "user")
        self.assertIn("explore the repo", user["text_preview"])

    def test_thinking_becomes_reasoning(self):
        steps = parse_steps(self._load())
        reasoned = [s for s in steps if s.get("has_reasoning")]
        self.assertTrue(reasoned)
        self.assertTrue(any(
            any(p.get("type") == "reasoning" for p in s["parts"])
            for s in reasoned
        ))

    def test_tool_call_paired_with_result(self):
        steps = parse_steps(self._load())
        bash = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-1"
        )
        self.assertEqual(bash["tool_name"], "Bash")
        self.assertEqual(bash["status"], "success")
        self.assertIn("command", bash["input"])
        self.assertIn("trajviz", bash["output"])

    def test_read_write_normalized_to_file_path(self):
        steps = parse_steps(self._load())
        read = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-2"
        )
        write = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-3"
        )
        self.assertEqual(read["tool_name"], "Read")
        self.assertEqual(read["input"]["file_path"], "/home/user/proj/README.md")
        self.assertEqual(write["tool_name"], "Write")
        self.assertEqual(write["input"]["file_path"], "/home/user/proj/DESIGN.md")

    def test_error_assistant_surfaces_message(self):
        steps = parse_steps(self._load())
        error_steps = [
            s for s in steps
            if s["role"] == "assistant" and "No API key" in (s.get("text_preview") or "")
        ]
        self.assertTrue(error_steps)
        self.assertEqual(error_steps[0].get("finish"), "error")

    def test_token_usage_ingested(self):
        steps = parse_steps(self._load())
        # Native Pi reasoning is included in output, not added again.
        assistant = [
            s for s in steps
            if s["role"] == "assistant" and s["tokens"]["input"] == 1728
        ]
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0]["tokens"]["output"], 87)
        self.assertEqual(assistant[0]["tokens"]["reasoning"], 49)
        self.assertEqual(assistant[0]["tokens"]["total"], 1728 + 87)

    def test_token_total_includes_cache(self):
        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z",
                 [{"type": "text", "text": "ok"}],
                 stopReason="stop",
                 usage={"input": 100, "output": 20, "cacheRead": 50,
                        "cacheWrite": 10, "reasoning": 5, "totalTokens": 120}),
        ]
        steps = parse_steps(self._load(events, name="cache.jsonl"))
        asst = next(s for s in steps if s["role"] == "assistant")
        self.assertEqual(asst["tokens"]["cache_read"], 50)
        self.assertEqual(asst["tokens"]["cache_write"], 10)
        self.assertEqual(asst["tokens"]["total"], 100 + 20 + 50 + 10)

    def test_model_request_bounds_and_native_tool_duration_are_separate(self):
        import agent_trajviz as tv

        events = [
            _session(),
            {"type": "message", "id": "request", "parentId": None,
             "timestamp": "2026-08-24T01:33:02.000Z", "message": {
                 "role": "assistant", "timestamp": 1787535180000,
                 "content": [{"type": "toolCall", "id": "call", "name": "bash", "arguments": {"command": "pytest"}}],
                 "usage": {"input": 100, "output": 20, "reasoning": 10, "totalTokens": 120}}},
            {"type": "message", "id": "result", "parentId": "request",
             "timestamp": "2026-08-24T01:33:07.000Z", "message": {
                 "role": "toolResult", "timestamp": 1787535187000, "durationMs": 5000,
                 "toolCallId": "call", "toolName": "bash", "content": [{"type": "text", "text": "passed"}]}},
            _msg("user", "2026-08-24T02:00:00.000Z", "next task", timestamp=1787536800000),
        ]
        path = _write(self.tmp, "request-timing.jsonl", events)
        run = tv.load(path)
        step = run.steps[0]
        self.assertEqual(step["duration"], 2)
        self.assertEqual(step["tool_calls"][0]["duration_ms"], 5000)
        self.assertEqual(run.metrics["tokens"]["total"], 120)
        self.assertEqual(run.metrics["output_tokens_per_sec"], 10)
        self.assertIsNone(run.view("steps").data[0]["tool_time_share"])
        self.assertEqual(run.view("step_durations").data["steps"][0]["duration_s"], 2)

    def test_missing_request_start_is_unknown_not_backfilled_from_later_entries(self):
        events = [_session(), {"type": "message", "id": "request", "timestamp": "2026-08-24T01:33:02.000Z",
                              "message": {"role": "assistant", "content": "done"}},
                  {"type": "session_info", "timestamp": "2026-08-24T01:34:02.000Z", "name": "renamed"}]
        step = parse_steps(self._load(events))[0]
        self.assertIsNone(step["duration"])
        self.assertIsNone(step["time_completed_ms"])

    def test_non_message_usage_compaction_and_branch_provenance_survive(self):
        events = [_session(),
                  _msg("assistant", "2026-08-24T01:33:02.000Z", "original", id="answer",
                       usage={"input": 10, "output": 2, "totalTokens": 12}),
                  {"type": "usage", "id": "warm", "parentId": "answer", "timestamp": "2026-08-24T01:33:03.000Z",
                   "kind": "future_kind", "usage": {"input": 0, "output": 0, "cacheRead": 50, "totalTokens": 50}},
                  {"type": "compaction", "id": "compact", "parentId": "warm", "timestamp": "2026-08-24T01:33:04.000Z",
                   "summary": "compacted context", "tokensBefore": 50000,
                   "usage": {"input": 4, "output": 1, "totalTokens": 5}},
                  {"type": "branch_summary", "id": "branch", "parentId": "answer", "fromId": "compact",
                   "timestamp": "2026-08-24T01:33:05.000Z", "summary": "other branch",
                   "usage": {"input": 2, "output": 1, "totalTokens": 3}},
                  {"type": "context_edit", "id": "context", "parentId": "branch", "targetId": "answer", "replacement": None},
                  {"type": "session_info", "id": "title", "parentId": "context", "name": "Audit task"}]
        raw = self._load(events)
        steps = parse_steps(raw)
        self.assertEqual(sum(step["tokens"]["total"] for step in steps), 70)
        self.assertEqual(raw["metadata"]["title"], "Audit task")
        self.assertEqual(raw["_pi_entries"], events)
        self.assertEqual(steps[0]["text_preview"], "original")
        compact = next(step for step in steps if step["role"] == "compaction")
        self.assertEqual(compact["compaction_metadata"]["preTokens"], 50000)
        branch = next(step for step in steps if step["id"] == "branch")
        self.assertEqual(branch["parent_id"], "answer")

    def test_system_prompt_sections_and_tool_changes_remain_inspectable(self):
        events = [_session(parentSession="/unreadable/parent.jsonl"),
                  _msg("system", "2026-08-24T01:30:33.026Z", "Base prompt", id="system",
                       sections={"cwd": "/project", "skills": None},
                       toolsAdded=[{"name": "read", "description": "Read a file", "parameters": {"type": "object"}}],
                       toolsRemoved=[{"name": "write"}])]
        raw = self._load(events)
        step = parse_steps(raw)[0]
        self.assertEqual(step["role"], "system")
        text = "\n".join(part.get("text", "") for part in step["parts"])
        for expected in ("Base prompt", "[cwd]\n/project", "[Prompt section removed: skills]",
                         "[Tools added: read]", "[Tools removed: write]"):
            self.assertIn(expected, text)
        self.assertEqual(raw["metadata"]["parent_session_path"], "/unreadable/parent.jsonl")
        self.assertEqual(raw["_pi_entries"], events)

    def test_custom_context_and_images_are_visible_without_image_payload(self):
        from agent_trajviz.insight.workflow_role import workflow_role

        image = {"type": "image", "mimeType": "image/png", "data": "private-base64-payload"}
        events = [_session(),
                  _msg("user", "2026-08-24T01:30:33.026Z", [image], id="image"),
                  {"type": "custom_message", "id": "injected", "parentId": "image",
                   "timestamp": "2026-08-24T01:30:34.000Z", "customType": "review-context",
                   "content": [{"type": "text", "text": "Injected context"}, image], "display": False},
                  _msg("custom", "2026-08-24T01:30:35.000Z", "Legacy custom context", id="custom")]
        raw = self._load(events)
        steps = parse_steps(raw)
        self.assertEqual(len(steps), 3)
        self.assertEqual([workflow_role(step) for step in steps], ["user", "system", "system"])
        self.assertEqual(steps[0]["text_preview"], "[Image: image/png]")
        self.assertIn("Injected context", steps[1]["text_preview"])
        self.assertNotIn("private-base64-payload", json.dumps(steps))
        self.assertIn("private-base64-payload", json.dumps(raw["_pi_entries"]))

    def test_user_bash_execution_retains_outcome_and_context_exclusion(self):
        events = [_session()]
        for index, extra in enumerate(({"exitCode": 0}, {"exitCode": 7}, {"cancelled": True}, {})):
            events.append(_msg("bashExecution", "2026-08-24T01:30:33.026Z", None, id=f"bash-{index}",
                               command="pytest -q", output="recorded output", truncated=True,
                               fullOutputPath="/tmp/output.log", excludeFromContext=True, **extra))
        steps = parse_steps(self._load(events))
        self.assertEqual([step["role"] for step in steps], ["system"] * 4)
        self.assertEqual([step["tool_calls"][0]["status"] for step in steps], ["success", "error", "error", "unknown"])
        self.assertEqual([step["error_count"] for step in steps], [0, 1, 1, 0])
        self.assertEqual(sum(step["tokens"]["total"] for step in steps), 0)
        for step in steps:
            tool = step["tool_calls"][0]
            self.assertEqual(tool["input"], {"command": "pytest -q"})
            self.assertTrue(tool["metadata"]["excludeFromContext"])
            self.assertTrue(tool["metadata"]["truncated"])
            self.assertIsNone(tool["duration_ms"])
        self.assertFalse(steps[-1]["tool_calls"][0]["metadata"]["_outcome_observed"])

    def test_native_monotonic_request_duration_overrides_persistence_delay(self):
        events = [_session(),
                  _msg("assistant", "2026-08-24T01:34:00.000Z", "done", id="request",
                       timestamp=1787535180000, durationMs=1200,
                       usage={"input": 100, "output": 20, "totalTokens": 120})]
        step = parse_steps(self._load(events))[0]
        self.assertEqual(step["duration"], 1.2)
        self.assertEqual(step["time_completed_ms"], 1787535181200)

    def test_empty_aborted_and_deferred_responses_preserve_finish(self):
        events = [_session(),
                  _msg("assistant", "2026-08-24T01:30:33.026Z", [], stopReason="aborted", id="aborted"),
                  _msg("assistant", "2026-08-24T01:30:34.000Z", [], stopReason="deferred", id="deferred")]
        steps = parse_steps(self._load(events))
        self.assertEqual([step["finish"] for step in steps], ["aborted", "deferred"])
        self.assertEqual([step["text_preview"] for step in steps], ["[Response aborted]", "[Response deferred]"])

    def test_tool_usage_and_nested_call_metadata_are_preserved_once(self):
        events = [_session(),
                  _msg("assistant", "2026-08-24T01:30:33.026Z",
                       [{"type": "toolCall", "id": "code-call", "name": "code", "arguments": {"code": "read()"}}],
                       id="request", usage={"input": 10, "output": 2, "totalTokens": 12}),
                  _msg("toolResult", "2026-08-24T01:30:34.000Z", [{"type": "text", "text": "done"}],
                       id="result", toolCallId="code-call", toolName="code", durationMs=500, isError=False,
                       usage={"input": 20, "output": 3, "totalTokens": 23},
                       nestedCalls={"calls": [{"id": "inner", "name": "read", "status": "ok", "durationMs": 15}],
                                    "complete": True})]
        raw = self._load(events)
        steps = parse_steps(raw)
        self.assertEqual(sum(step["tokens"]["total"] for step in steps), 35)
        self.assertEqual(sum(len(step["tool_calls"]) for step in steps), 2)
        self.assertEqual(steps[1]["role"], "tool")
        tool = steps[0]["tool_calls"][0]
        self.assertEqual(tool["duration_ms"], 500)
        self.assertEqual(tool["metadata"]["pi_result"]["nestedCalls"]["calls"][0]["name"], "read")
        nested = steps[0]["tool_calls"][1]
        self.assertEqual(nested["tool_name"], "Read")
        self.assertEqual(nested["metadata"]["pi_nested_parent_id"], "code-call")
        self.assertEqual(nested["duration_ms"], 15)
        self.assertIsNone(nested["time_start"])
        self.assertIsNone(nested["time_end"])

    def test_nested_calls_keep_arguments_failures_and_unfinished_outcomes(self):
        events = [_session(),
                  _msg("assistant", "2026-08-24T01:30:33.026Z",
                       [{"type": "toolCall", "id": "code-call", "name": "codemode", "arguments": {"code": "write()"}}],
                       id="request"),
                  _msg("toolResult", "2026-08-24T01:30:34.000Z", "done", id="result",
                       toolCallId="code-call", toolName="codemode", isError=False,
                       nestedCalls={"calls": [
                           {"id": "write-1", "name": "write", "arguments": {"path": "result.py", "content": "ok"},
                            "status": "error", "error": "permission denied", "durationMs": 5},
                           {"id": "unfinished", "name": "bash", "argumentsBytes": 250, "status": "unfinished"}],
                           "complete": False})]
        step = parse_steps(self._load(events))[0]
        self.assertEqual(step["error_count"], 1)
        write, unfinished = step["tool_calls"][1:]
        self.assertEqual(write["input"]["file_path"], "result.py")
        self.assertEqual(write["error"], "permission denied")
        self.assertTrue(write["metadata"]["_input_observed"])
        self.assertEqual(unfinished["status"], "unknown")
        self.assertFalse(unfinished["metadata"]["_outcome_observed"])
        self.assertFalse(unfinished["metadata"]["_input_observed"])
        self.assertEqual(unfinished["metadata"]["pi_nested_call"]["argumentsBytes"], 250)

    def test_message_without_persistence_time_keeps_missing_duration_unknown(self):
        events = [_session(), {"type": "message", "id": "streamed", "message": {
            "role": "assistant", "timestamp": 1787535180000,
            "content": [{"type": "text", "text": "done"}], "stopReason": "stop"}}]
        step = parse_steps(self._load(events))[0]
        self.assertIsNone(step["duration"])
        self.assertIsNone(step["time_completed_ms"])

    def test_tool_results_follow_parent_links_when_call_ids_repeat(self):
        from agent_trajviz.insight.formats.pi import _convert_pi_to_internal

        def call(ident, parent):
            event = _msg("assistant", "2026-08-24T01:30:33.026Z",
                         [{"type": "toolCall", "id": "same-call", "name": "read", "arguments": {"path": ident}}],
                         id=ident)
            event["parentId"] = parent
            return event

        first_result = _msg("toolResult", "2026-08-24T01:30:34.000Z", "first output",
                            id="r1", toolCallId="same-call", toolName="read", isError=False)
        first_result["parentId"] = "a1"
        second_result = _msg("toolResult", "2026-08-24T01:30:35.000Z", "second output",
                             id="r2", toolCallId="same-call", toolName="read", isError=False)
        second_result["parentId"] = "a2"
        raw = _convert_pi_to_internal([_session(), call("a1", None), call("a2", None), first_result, second_result])
        tools = [tool for step in parse_steps(raw) for tool in step["tool_calls"]]
        self.assertEqual([tool["output"] for tool in tools], ["first output", "second output"])
        self.assertEqual([tool["status"] for tool in tools], ["success", "success"])

    def test_string_content_is_one_text_part(self):
        events = [
            _session(),
            _msg("user", "2026-08-24T01:30:33.026Z", "explore the repo"),
            _msg("assistant", "2026-08-24T01:33:36.288Z",
                 "Here is an overview of the repo.",
                 stopReason="stop"),
        ]
        steps = parse_steps(self._load(events, name="string-content.jsonl"))
        user = next(s for s in steps if s["role"] == "user")
        asst = next(s for s in steps if s["role"] == "assistant")
        self.assertEqual(user["text_preview"], "explore the repo")
        self.assertEqual(len(user["parts"]), 1)
        self.assertEqual(asst["text_preview"], "Here is an overview of the repo.")
        self.assertEqual(len(asst["parts"]), 1)

    def test_session_header_uses_last_model(self):
        steps = parse_steps(self._load())
        model_id, provider_id, _ = extract_agent_info(steps)
        self.assertEqual(model_id, "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(provider_id, "cline")

    def test_timestamps_are_epoch_ms(self):
        steps = parse_steps(self._load())
        stamped = [s for s in steps if s.get("time_created_ms")]
        self.assertTrue(stamped)
        for s in stamped:
            self.assertIsInstance(s["time_created_ms"], int)
            self.assertGreater(s["time_created_ms"], 10**12)

    def test_dangling_tool_call_keeps_outcome_unknown(self):
        import agent_trajviz as tv

        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                {"type": "toolCall", "id": "call-x", "name": "bash",
                 "arguments": {"command": "pwd"}},
            ]),
        ]
        run = tv.load(_write(self.tmp, "dangling.jsonl", events))
        tool = next(tc for s in run.steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "unknown")
        self.assertEqual(tool["tool_id"], "call-x")
        self.assertEqual(tool["input"], {"command": "pwd"})
        self.assertFalse(tool["metadata"]["_outcome_observed"])
        self.assertFalse(tool["metadata"]["_output_observed"])
        self.assertEqual(run.steps[0]["error_count"], 0)

    def test_partial_stream_with_unfinished_tool_result_has_no_false_failure(self):
        import agent_trajviz as tv
        from agent_trajviz.insight.telemetry_quality import tool_field_observed
        from agent_trajviz.insight.tool_failure import tool_call_failed

        assistant = {"role": "assistant", "timestamp": 1787535180000, "durationMs": 250,
                     "content": [{"type": "toolCall", "id": "running", "name": "bash",
                                  "arguments": {"command": "pytest -q"}}], "stopReason": "toolUse",
                     "usage": {"input": 10, "output": 2, "totalTokens": 12}}
        events = [_session(), {"type": "message_start", "message": assistant},
                  {"type": "message_end", "message": assistant},
                  {"type": "message_start", "message": {"role": "toolResult", "toolCallId": "running",
                   "toolName": "bash", "content": [], "isError": False, "timestamp": 1787535180250}}]
        run = tv.load(_write(self.tmp, "partial-stream.jsonl", events))
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.metrics["tokens"]["total"], 12)
        tool = run.steps[0]["tool_calls"][0]
        self.assertEqual(tool["status"], "unknown")
        self.assertFalse(tool_call_failed(tool))
        self.assertFalse(tool_field_observed(tool, "tool_outcomes"))
        self.assertFalse(tool_field_observed(tool, "tool_outputs"))
        self.assertEqual(run.steps[0]["error_count"], 0)
        self.assertTrue(any("unfinished messages" in warning for warning in run.warnings))

    def test_orphan_tool_result_preserves_native_execution_duration(self):
        import agent_trajviz as tv
        from agent_trajviz.insight.metrics import tool_call_duration_ms

        events = [_session(),
                  _msg("toolResult", "2026-08-24T01:33:03.000Z", "completed", id="orphan",
                       timestamp=1787535183000, toolCallId="missing-call", toolName="read", isError=False,
                       durationMs=5)]
        run = tv.load(_write(self.tmp, "orphan-result.jsonl", events))
        self.assertEqual(run.steps[0]["id"], "orphan")
        self.assertIsNone(run.steps[0]["duration"])
        tool = run.steps[0]["tool_calls"][0]
        self.assertEqual(tool["duration_ms"], 5)
        self.assertEqual(tool_call_duration_ms(tool), 5)
        self.assertEqual(tool["time_start"], 1787535182995)
        self.assertEqual(tool["time_end"], 1787535183000)
        self.assertEqual(tool["tool_name"], "Read")
        self.assertEqual(tool["status"], "success")

    def test_failed_tool_result_status(self):
        events = [
            _session(),
            _msg("assistant", "2026-08-24T01:33:02.137Z", [
                {"type": "toolCall", "id": "call-f", "name": "bash",
                 "arguments": {"command": "false"}},
            ]),
            _msg("toolResult", "2026-08-24T01:33:03.000Z",
                 [{"type": "text", "text": "exit 1"}],
                 toolCallId="call-f", toolName="bash", isError=True),
        ]
        steps = parse_steps(self._load(events, name="fail.jsonl"))
        tool = next(tc for s in steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "error")
        self.assertEqual(tool["output"], "exit 1")

    def test_truncated_final_line_does_not_reject_file(self):
        path = os.path.join(self.tmp, "trunc.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events[:4]:
                f.write(json.dumps(event) + "\n")
            f.write('{"type":"message","timestamp":"2026-')
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "pi")

    def test_unknown_jsonl_still_rejected(self):
        path = _write(self.tmp, "other.jsonl", [{"type": "not-a-session", "id": "x"}])
        raw = load_trajectory(path)
        self.assertIn("_error", raw)
        self.assertIn("Pi", raw["_error"])


if __name__ == "__main__":
    unittest.main()
