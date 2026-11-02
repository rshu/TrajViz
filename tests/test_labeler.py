import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from agent_trajviz import _fileio
from agent_trajviz.labeling import labeler, llm
from agent_trajviz.labeling.taxonomy import DEFAULT_TAXONOMY
from agent_trajviz.insight.labels import aggregate_labels


def _http_response(status, payload):
    """A real ``requests`` response: ``raise_for_status`` raises the real HTTPError, whose
    message carries the URL, and ``json()`` parses the body."""
    response = requests.models.Response()
    response.status_code = status
    response.url = "https://example.invalid/v1/chat/completions?key=synthetic-query"
    response._content = json.dumps(payload).encode()
    return response


def _openai_reply(content, finish_reason="stop"):
    choice = {"message": {"content": content}, "finish_reason": finish_reason}
    return _http_response(200, {"choices": [choice]})


def _anthropic_reply(content, stop_reason="end_turn"):
    return _http_response(200, {"content": content, "stop_reason": stop_reason})


def _error_reply(status):
    return _http_response(status, {"error": {"message": "synthetic-secret-body"}})


VALID_ANSWER = '{"phase":"understand","action":"code_reading"}'
# Request details that must never reach a log line or an error message.
PRIVATE = ("example.invalid", "synthetic-query", "synthetic-secret-body", "synthetic-key")


def _step(index, role="assistant", text="Reading the code"):
    return {"index": index, "raw_index": index, "role": role, "text_preview": text,
            "tokens": {}, "parts": [], "tool_calls": []}


def _steps(assistant_count):
    """A user prompt (index 0), then *assistant_count* assistant steps (indices 1..n)."""
    assistant = [_step(i) for i in range(1, assistant_count + 1)]
    return [_step(0, "user", "Please fix the bug"), *assistant]


@contextlib.contextmanager
def _model_answers(steps, replies):
    """Load *steps* as the trajectory; ``requests.post`` answers with *replies*: one response
    for every call, or a list with one response per call."""
    answer = {"side_effect": replies} if isinstance(replies, list) else {"return_value": replies}
    with (
        patch.object(labeler, "load_all_steps", return_value=steps),
        patch.object(requests, "post", **answer) as post,
        patch.object(labeler.time, "sleep"),
    ):
        yield post


def _label_to(output_path, **kwargs):
    return labeler.label_trajectory(
        "input.json", str(output_path), base_url="https://example.invalid/v1",
        api_key="synthetic-key", model="test-model", **kwargs,
    )


def _trajectory_file(directory):
    path = Path(directory) / "trajectory.json"
    path.write_text('{"messages": []}\n', encoding="utf-8")
    return path


def _run_command(*args):
    """``main()`` with a clean ``LABEL_*`` environment and no ``.env``; returns (code, stderr)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LABEL_", "ANALYZE_"))}
    connection = ["--base-url", "https://example.invalid/v1", "--api-key", "synthetic-key",
                  "--model", "test-model", "--provider", "openai"]
    stderr = io.StringIO()
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(llm, "_load_dotenv", lambda *a, **k: None),
        contextlib.redirect_stderr(stderr),
    ):
        code = labeler.main([*args, *connection])
    return code, stderr.getvalue()


class LabelerTests(unittest.TestCase):
    def test_dotenv_only_loads_llm_settings_and_preserves_process_values(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LABEL_MODEL": "process-model"}, clear=True):
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "LABEL_MODEL=file-model\n"
                "LABEL_API_KEY='synthetic-key' # test only\n"
                "ANALYZE_MODEL=analysis-model\n"
                "HTTPS_PROXY=https://example.invalid\n"
                "AWE_DECAF_PATH=/synthetic/untrusted\n"
                "PYTHONPATH=/synthetic/untrusted\n"
                "LABEL_INVALID.KEY=ignored\n",
                encoding="utf-8",
            )
            llm._load_dotenv(str(env_file))
            self.assertEqual(dict(os.environ), {
                "LABEL_MODEL": "process-model",
                "LABEL_API_KEY": "synthetic-key",
                "ANALYZE_MODEL": "analysis-model",
            })

    def test_retry_logs_never_include_provider_error_details(self):
        sensitive_message = "https://viewer:synthetic-password@example.invalid/v1?key=synthetic-query"
        with (
            patch.object(llm, "call_llm", side_effect=RuntimeError(sensitive_message)) as call,
            patch.object(labeler.time, "sleep"),
            self.assertLogs(labeler.logger, level="WARNING") as logs,
        ):
            label, failure = labeler._classify(
                {"index": 1, "text_preview": "synthetic prompt"},
                base_url="https://example.invalid/v1", api_key="synthetic-key", model="test-model",
                system_prompt="test", valid=(set(), set(), {}), provider="openai", temperature=None,
                max_tokens=None, max_content_chars=8000,
            )
        self.assertIsNone(label)
        self.assertEqual(failure, "RuntimeError")
        self.assertEqual(call.call_count, 2)
        output = "\n".join(logs.output)
        self.assertIn("RuntimeError", output)
        self.assertIn("retrying", output)
        self.assertIn("labeled unknown", output)
        self.assertNotIn("synthetic-password", output)
        self.assertNotIn("synthetic-query", output)
        self.assertNotIn("example.invalid", output)

    def test_refuses_to_overwrite_input_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.json"
            original = '{"messages": []}\n'
            trajectory_path.write_text(original, encoding="utf-8")

            with (
                patch.object(labeler, "load_all_steps") as load_steps,
                self.assertRaises(labeler.OutputSafetyError),
            ):
                labeler.label_trajectory(
                    str(trajectory_path),
                    str(trajectory_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                )

            load_steps.assert_not_called()
            self.assertEqual(
                trajectory_path.read_text(encoding="utf-8"), original
            )

    def test_atomic_write_preserves_existing_output_on_serialization_error(self):
        steps = [
            {
                "index": 0,
                "role": "user",
                "text_preview": "Please inspect this project",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            }
        ]

        def fail_after_partial_write(_data, handle, **_kwargs):
            handle.write('{"partial":')
            raise RuntimeError("serialization failed")

        with tempfile.TemporaryDirectory() as tmp:
            trajectory_path = Path(tmp) / "trajectory.json"
            trajectory_path.write_text('{"messages": []}\n', encoding="utf-8")
            output_path = Path(tmp) / "labeled.json"
            original_output = '{"existing": true}\n'
            output_path.write_text(original_output, encoding="utf-8")
            taxonomy_path = str(DEFAULT_TAXONOMY)

            with (
                patch.object(labeler, "load_all_steps", return_value=steps),
                patch.object(
                    _fileio.json,
                    "dump",
                    side_effect=fail_after_partial_write,
                ),
                self.assertRaisesRegex(RuntimeError, "serialization failed"),
            ):
                labeler.label_trajectory(
                    str(trajectory_path),
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"), original_output
            )
            self.assertEqual(list(Path(tmp).glob(".labeled.json.*.tmp")), [])

    def test_emits_every_index_and_defaults_user(self):
        steps = [
            {
                "index": 4,
                "raw_index": 0,
                "role": "user",
                "text_preview": "Please inspect this project",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            },
            {
                "index": 9,
                "raw_index": 1,
                "role": "assistant",
                "text_preview": "I will inspect the files",
                "tokens": {"total": 10},
                "parts": [{"type": "reasoning", "text": "Need inspect files"}],
                "tool_calls": [],
            },
            {
                "index": 12,
                "raw_index": 2,
                "role": "system",
                "text_preview": "system context",
                "tokens": {},
                "parts": [],
                "tool_calls": [],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "labeled.json"
            taxonomy_path = str(DEFAULT_TAXONOMY)
            with (
                patch.object(labeler, "load_all_steps", return_value=steps),
                patch.object(
                    llm,
                    "call_llm",
                    return_value='{"phase":"understand","action":"code_reading"}',
                ) as call_llm,
            ):
                labeler.label_trajectory(
                    "input.json",
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "trajectory_labels.v2")
        self.assertEqual([s["index"] for s in data["steps"]], [4, 9, 12])
        self.assertEqual([s["raw_index"] for s in data["steps"]], [0, 1, 2])
        self.assertEqual(len(data["steps"]), len(steps))
        self.assertEqual(
            (data["steps"][0]["phase"], data["steps"][0]["action"]),
            ("user", "user_prompt"),
        )
        self.assertEqual(data["steps"][0]["label_source"], "default")
        self.assertEqual(
            (data["steps"][1]["phase"], data["steps"][1]["action"]),
            ("understand", "code_reading"),
        )
        self.assertEqual(data["steps"][1]["label_source"], "llm")
        self.assertEqual(
            (data["steps"][2]["phase"], data["steps"][2]["action"]),
            ("unknown", "unknown"),
        )
        call_llm.assert_called_once()

    def test_empty_assistant_is_retained_without_llm_call(self):
        steps = [
            {
                "index": 0,
                "role": "assistant",
                "text_preview": "",
                "parts": [],
                "tool_calls": [],
                "tokens": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "labeled.json"
            taxonomy_path = str(DEFAULT_TAXONOMY)
            with (
                patch.object(labeler, "load_all_steps", return_value=steps),
                patch.object(llm, "call_llm") as call_llm,
            ):
                labeler.label_trajectory(
                    "input.json",
                    str(output_path),
                    base_url="https://example.invalid/v1",
                    api_key="test",
                    model="test-model",
                    taxonomy_path=str(taxonomy_path),
                )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["steps"][0]["index"], 0)
        self.assertEqual(data["steps"][0]["label_source"], "fallback")
        call_llm.assert_not_called()

    def test_user_steps_do_not_distort_assistant_metrics_or_duplicate_timeline(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as trajectory:
            data = {
                "schema_version": "trajectory_labels.v2",
                "taxonomy_version": "v1",
                "model": "test-model",
                "trajectory_file": trajectory.name,
                "steps": [
                    {
                        "index": 0,
                        "role": "user",
                        "phase": "user",
                        "action": "user_prompt",
                        "duration_s": 0,
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "phase": "understand",
                        "action": "code_reading",
                        "duration_s": 1,
                    },
                ],
            }
            source_steps = [{"index": 0, "role": "user"}]
            with (
                patch(
                    "agent_trajviz.insight.loaders.load_trajectory",
                    return_value={},
                ),
                patch(
                    "agent_trajviz.insight.parser.parse_steps",
                    return_value=source_steps,
                ),
            ):
                result = aggregate_labels(data)

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["classified"], 1)
        self.assertEqual(result["classification_rate"], 100.0)
        self.assertNotIn("user", result["phase_counts"])
        self.assertEqual([s["index"] for s in result["steps"]], [0, 1])


class AnthropicEndpointTests(unittest.TestCase):
    def test_messages_url_gains_v1_only_when_the_base_lacks_it(self):
        # The SDK convention (no /v1) 404s on <base>/messages at api.anthropic.com.
        for base, expected in (
            ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
            ("https://api.deepseek.com/anthropic",
             "https://api.deepseek.com/anthropic/v1/messages"),
            ("https://api.deepseek.com/anthropic/v1/",
             "https://api.deepseek.com/anthropic/v1/messages"),
        ):
            with self.subTest(base=base):
                reply = _anthropic_reply([{"type": "text", "text": "ok"}])
                with patch.object(requests, "post", return_value=reply) as post:
                    llm.call_llm(base, "synthetic-key", "m", "system", "user", provider="anthropic")
                self.assertEqual(post.call_args.args[0], expected)


class BudgetExhaustionTests(unittest.TestCase):
    """Reasoning models can spend the whole output budget on hidden thinking."""

    def _call(self, reply, provider="openai"):
        with patch.object(requests, "post", return_value=reply):
            return llm.call_llm("https://example.invalid/v1", "synthetic-key", "m", "system",
                                "user", provider=provider, max_tokens=64)

    def test_an_empty_reply_at_the_token_limit_raises(self):
        thinking = {"type": "thinking", "thinking": "..."}
        for provider, reply in (
            ("openai", _openai_reply("", "length")),
            ("openai", _openai_reply(None, "length")),
            ("openai", _openai_reply(" \n", "length")),
            ("anthropic", _anthropic_reply([thinking], "max_tokens")),
            ("anthropic", _anthropic_reply([], "max_tokens")),
            ("anthropic", _anthropic_reply([{"type": "text", "text": " "}], "max_tokens")),
        ):
            with self.subTest(provider=provider, body=reply.json()):
                with self.assertRaises(llm.LabelBudgetExhausted) as raised:
                    self._call(reply, provider)
                self.assertIsInstance(raised.exception, RuntimeError)
                self.assertEqual(raised.exception.limit, 64)

    def test_truncated_text_and_other_replies_are_returned_as_before(self):
        # A cut-off answer may still parse; the parser decides.
        cut_off = '{"phase": "debug"'
        self.assertEqual(self._call(_openai_reply(cut_off, "length")), cut_off)
        self.assertEqual(self._call(_openai_reply("", "stop")), "")
        reply = _anthropic_reply([{"type": "text", "text": cut_off}], "max_tokens")
        self.assertEqual(self._call(reply, "anthropic"), cut_off)
        with self.assertRaisesRegex(ValueError, "no text blocks"):
            self._call(_anthropic_reply([{"type": "thinking", "thinking": "..."}]), "anthropic")


class FailureReportingTests(unittest.TestCase):
    """Model-call failures are retried once, then named in the log by class and HTTP status."""

    def test_an_empty_reply_at_the_token_limit_is_retried_once_then_labeled_unknown(self):
        empty = _openai_reply("", "length")
        with (
            tempfile.TemporaryDirectory() as tmp,
            _model_answers(_steps(2), [empty, empty, _openai_reply(VALID_ANSWER)]) as post,
            self.assertLogs(labeler.logger, level="WARNING") as logs,
        ):
            result = _label_to(Path(tmp) / "labeled.json", max_tokens=64)
        self.assertEqual(post.call_count, 3)
        labels = {s["index"]: (s["phase"], s["action"], s["label_source"]) for s in result["steps"]}
        self.assertEqual(labels[1], ("unknown", "unknown", "fallback"))
        self.assertEqual(labels[2], ("understand", "code_reading", "llm"))
        output = "\n".join(logs.output)
        for line in ("labeling index 1 failed (LabelBudgetExhausted); retrying",
                     "labeling index 1 failed again (LabelBudgetExhausted); labeled unknown",
                     "1 of 2 assistant steps could not be classified (LabelBudgetExhausted ×1)"):
            self.assertIn(line, output)

    def test_http_failures_log_the_status_but_never_the_url_or_body(self):
        with (
            patch.object(requests, "post", return_value=_error_reply(401)) as post,
            patch.object(labeler.time, "sleep"),
            self.assertLogs(labeler.logger, level="WARNING") as logs,
        ):
            labeler._classify(
                _step(1), system_prompt="test", valid=(set(), set(), {}),
                base_url="https://example.invalid/v1", api_key="synthetic-key", model="test-model",
                provider="openai", temperature=None, max_tokens=None, max_content_chars=8000,
            )
        self.assertEqual(post.call_count, 2)
        output = "\n".join(logs.output)
        for line in ("labeling index 1 failed (HTTPError 401); retrying",
                     "labeling index 1 failed again (HTTPError 401); labeled unknown"):
            self.assertIn(line, output)
        for private in PRIVATE:
            self.assertNotIn(private, output)


class RejectedCredentialsTests(unittest.TestCase):
    """A rejected key or a wrong URL stops the run instead of writing an all-unknown sidecar."""

    KEY_REMEDY = "check LABEL_API_KEY / --api-key and LABEL_BASE_URL / --base-url"
    URL_REMEDY = "check LABEL_BASE_URL / --base-url and LABEL_MODEL / --model"

    def test_three_rejected_steps_abort_the_run_before_anything_is_written(self):
        cases = ((401, self.KEY_REMEDY), (403, self.KEY_REMEDY), (404, self.URL_REMEDY))
        for status, remedy in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                # The empty assistant step (index 1) never reaches the model; it does not count.
                steps = [_step(0, "user"), _step(1, text=""), *[_step(i) for i in range(2, 7)]]
                with (
                    _model_answers(steps, _error_reply(status)) as post,
                    self.assertRaises(labeler.LabelingAborted) as raised,
                ):
                    _label_to(Path(tmp) / "labeled.json")
                self.assertEqual(str(raised.exception),
                                 "labeling aborted: the API rejected the first 3 requests "
                                 f"(HTTP {status}); {remedy}")
                self.assertIsInstance(raised.exception, RuntimeError)
                # Steps 2-4, each retried once; steps 5 and 6 are never sent.
                self.assertEqual(post.call_count, 6)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_an_aborted_run_leaves_an_existing_sidecar_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labeled.json"
            output.write_text('{"existing": true}\n', encoding="utf-8")
            with (
                _model_answers(_steps(3), _error_reply(401)),
                self.assertRaises(labeler.LabelingAborted),
            ):
                _label_to(output)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"existing": true}\n')
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["labeled.json"])

    def test_a_success_among_the_first_three_steps_keeps_the_run_going(self):
        replies = [*[_error_reply(401)] * 2, _openai_reply(VALID_ANSWER), *[_error_reply(401)] * 4]
        with tempfile.TemporaryDirectory() as tmp, _model_answers(_steps(4), replies) as post:
            result = _label_to(Path(tmp) / "labeled.json")
            self.assertTrue((Path(tmp) / "labeled.json").is_file())
        self.assertEqual(post.call_count, 7)
        sources = [s["label_source"] for s in result["steps"]]
        self.assertEqual(sources, ["default", "fallback", "llm", "fallback", "fallback"])

    def test_the_command_exits_2_and_names_the_status_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "trajectory_labeled.json"
            with _model_answers(_steps(4), _error_reply(401)):
                code, stderr = _run_command(str(_trajectory_file(tmp)), "-o", str(output))
            self.assertFalse(output.exists())
        self.assertEqual(code, 2)
        self.assertIn("HTTP 401", stderr)
        self.assertIn("agent-trajviz-label: labeling aborted: the API rejected the first 3 "
                      f"requests (HTTP 401); {self.KEY_REMEDY}\n", stderr)
        for private in PRIVATE:
            self.assertNotIn(private, stderr)


class FailedRunTests(unittest.TestCase):
    """A run whose model calls all fail writes nothing; a partly failed run says how much failed."""

    def test_a_run_whose_every_call_fails_writes_no_sidecar(self):
        outage = [_error_reply(503)] * 4 + [requests.ConnectionError("synthetic-secret-body")] * 2
        cases = (
            (_steps(3), _error_reply(503), "all 3 assistant steps failed (HTTPError 503 ×3)"),
            # Too short for the three-step rejection check; the end-of-run check still applies.
            (_steps(2), _error_reply(401), "all 2 assistant steps failed (HTTPError 401 ×2)"),
            (_steps(3), outage,
             "all 3 assistant steps failed (HTTPError 503 ×2, ConnectionError ×1)"),
        )
        for steps, replies, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                with (
                    _model_answers(steps, replies),
                    self.assertRaises(labeler.LabelingAborted) as raised,
                ):
                    _label_to(Path(tmp) / "labeled.json")
                self.assertEqual(str(raised.exception),
                                 f"labeling aborted: {expected}; no sidecar written")
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_the_command_exits_2_when_every_call_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labeled.json"
            output.write_text('{"existing": true}\n', encoding="utf-8")
            with _model_answers(_steps(3), _error_reply(503)):
                code, stderr = _run_command(str(_trajectory_file(tmp)), "-o", str(output))
            self.assertEqual(output.read_text(encoding="utf-8"), '{"existing": true}\n')
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["labeled.json", "trajectory.json"])
        self.assertEqual(code, 2)
        self.assertIn("agent-trajviz-label: labeling aborted: all 3 assistant steps failed "
                      "(HTTPError 503 ×3); no sidecar written\n", stderr)

    def test_one_failure_among_successes_is_written_with_a_warning(self):
        replies = [_error_reply(503), _error_reply(503), *[_openai_reply(VALID_ANSWER)] * 2]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labeled.json"
            with (
                _model_answers(_steps(3), replies),
                self.assertLogs(labeler.logger, level="WARNING") as logs,
            ):
                code, _stderr = _run_command(str(_trajectory_file(tmp)), "-o", str(output))
            data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        sources = [s["label_source"] for s in data["steps"]]
        self.assertEqual(sources, ["default", "fallback", "llm", "llm"])
        warning = "1 of 3 assistant steps could not be classified (HTTPError 503 ×1)"
        self.assertIn(f"WARNING:agent_trajviz.labeling.labeler:{warning}", logs.output)

    def test_unusable_answers_are_not_failures(self):
        # The model answered every time; its answers just parse to unknown.
        with (
            tempfile.TemporaryDirectory() as tmp,
            _model_answers(_steps(3), _openai_reply("not sure")),
            self.assertNoLogs(labeler.logger, level="WARNING"),
        ):
            result = _label_to(Path(tmp) / "labeled.json")
        sources = [s["label_source"] for s in result["steps"]]
        self.assertEqual(sources, ["default", "fallback", "fallback", "fallback"])


class CountsTests(unittest.TestCase):
    """The sidecar's ``counts`` describe the records it holds."""

    # A user prompt, two classified assistant steps, an empty assistant step, a system step.
    STEPS = [_step(0, "user"), _step(1), _step(2, text=""), _step(3, "system", "context"), _step(4)]

    def _label(self, **kwargs):
        with (
            tempfile.TemporaryDirectory() as tmp,
            _model_answers(self.STEPS, _openai_reply(VALID_ANSWER)),
        ):
            _label_to(Path(tmp) / "labeled.json", **kwargs)
            sidecar = json.loads((Path(tmp) / "labeled.json").read_text(encoding="utf-8"))
        return sidecar["counts"], sidecar["steps"]

    def _assert_describes(self, counts, records):
        self.assertEqual(counts["total"], len(records))
        self.assertEqual(counts["user"], sum(r["role"] == "user" for r in records))
        not_classified = sum(r["label_source"] != "llm" for r in records)
        self.assertEqual(counts["default_or_fallback"], not_classified)

    def test_assistant_only_counts_describe_the_written_records(self):
        counts, records = self._label(assistant_only=True)
        self.assertEqual(counts, {"total": 3, "assistant": 3, "user": 0, "default_or_fallback": 1})
        self._assert_describes(counts, records)

    def test_full_counts_are_unchanged(self):
        counts, records = self._label()
        self.assertEqual(counts, {"total": 5, "assistant": 3, "user": 1, "default_or_fallback": 3})
        self._assert_describes(counts, records)


class DefaultBudgetTests(unittest.TestCase):
    """Reasoning models think before they answer, so the default output budget is 4096 tokens."""

    def test_the_command_sends_4096_when_no_limit_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labeled.json"
            with (
                _model_answers(_steps(1), _openai_reply(VALID_ANSWER)) as post,
                patch.object(llm, "call_llm", wraps=llm.call_llm) as call,
            ):
                code, _stderr = _run_command(str(_trajectory_file(tmp)), "-o", str(output))
        self.assertEqual(code, 0)
        self.assertEqual(call.call_args.kwargs["max_tokens"], 4096)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 4096)

    def test_an_anthropic_request_without_a_limit_asks_for_4096(self):
        reply = _anthropic_reply([{"type": "text", "text": "ok"}])
        with patch.object(requests, "post", return_value=reply) as post:
            llm.call_llm("https://example.invalid/v1", "synthetic-key", "m", "system", "user",
                         provider="anthropic")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 4096)

    def test_the_help_states_the_default(self):
        help_text = " ".join(labeler._build_parser().format_help().split())
        self.assertIn("Max response tokens (default: 4096)", help_text)


if __name__ == "__main__":
    unittest.main()
