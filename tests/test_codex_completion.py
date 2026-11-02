"""Codex runs that complete normally must say so.

The Codex converter never set a finish reason, so every assistant step carried
``finish=""``, even in rollouts that end in ``task_complete``.
Everything that asks "did this run stop normally?" (the converge outcome, the
batch report's per-task outcome, the multi-run "finished" column) reads the
last assistant step's finish, so every Codex run read as not finished,
including the ones that resolved their task. ``task_complete`` is Codex's own
statement that a turn ended normally; it now marks that turn's final assistant
message ``stop``, the value the other formats use. A rollout cut off before
``task_complete`` keeps an empty finish and still reads as not finished.

``task_complete`` is also the only record of when a turn ended. A turn's final
assistant message used to be completed at the next turn's start instead, so the
idle time between two turns read as that step's LLM inference.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent_trajviz.converge.alignment import _detect_success
from agent_trajviz.insight.metrics import _compute_timing_metrics
from agent_trajviz.insight.session import LoadedSession, load_session


def _ev(ts: int, etype: str, payload: dict) -> dict:
    # ``ts`` counts seconds from midnight, so a turn can start minutes after the last.
    stamp = f"2026-01-01T{ts // 3600:02d}:{ts // 60 % 60:02d}:{ts % 60:02d}Z"
    return {"timestamp": stamp, "type": etype, "payload": payload}


def _user(ts: int, text: str) -> dict:
    return _ev(ts, "response_item", {"type": "message", "role": "user",
                                     "content": [{"type": "input_text", "text": text}]})


def _assistant(ts: int, text: str) -> dict:
    return _ev(ts, "response_item", {"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": text}]})


def _call(ts: int, call_id: str) -> list[dict]:
    return [
        _ev(ts, "response_item", {"type": "function_call", "name": "shell", "call_id": call_id,
                                  "arguments": json.dumps({"command": ["bash", "-lc", "ls"]})}),
        _ev(ts, "response_item", {"type": "function_call_output", "call_id": call_id,
                                  "output": json.dumps({"output": "a.py", "metadata": {"exit_code": 0}})}),
    ]


def _session(*events: dict) -> list[dict]:
    meta = _ev(0, "session_meta", {"id": "s", "cwd": "/w", "originator": "codex_cli_rs", "cli_version": "0.1"})
    return [meta, *events]


def _load(events: list[dict]) -> LoadedSession:
    path = os.path.join(tempfile.mkdtemp(), "rollout.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(e) for e in events))
    result = load_session(path)
    assert isinstance(result, LoadedSession), result
    return result


def _assistant_steps(session: LoadedSession) -> list[dict]:
    return [s for s in session.steps if s.get("role") == "assistant"]


def _two_turns_ten_minutes_apart() -> LoadedSession:
    return _load(_session(
        _user(0, "first task"), _assistant(5, "first done"),
        _ev(8, "event_msg", {"type": "task_complete"}),
        _user(608, "second task"), _assistant(612, "second done"),
        _ev(615, "event_msg", {"type": "task_complete"}),
    ))


def _timed(role: str, created_s: int, completed_s: int) -> dict:
    """A parsed step reduced to the fields the response latencies read."""
    return {"role": role, "time_created_ms": created_s * 1000,
            "time_completed_ms": completed_s * 1000}


class CodexCompletionTests(unittest.TestCase):
    def test_task_complete_marks_the_turns_last_assistant_step_stop(self):
        s = _load(_session(
            _user(1, "fix it"),
            _ev(2, "event_msg", {"type": "task_started"}),
            *_call(3, "c1"),
            _assistant(4, "done"),
            _ev(5, "event_msg", {"type": "task_complete"}),
        ))
        steps = _assistant_steps(s)
        self.assertEqual(steps[-1]["finish"], "stop")
        self.assertTrue(_detect_success(s.steps))

    def test_a_rollout_cut_off_before_task_complete_is_not_finished(self):
        s = _load(_session(_user(1, "fix it"), *_call(2, "c1"), _assistant(3, "working")))
        self.assertEqual(_assistant_steps(s)[-1]["finish"], "")
        self.assertFalse(_detect_success(s.steps))

    def test_each_completed_turn_ends_in_stop_and_mid_turn_steps_are_untouched(self):
        s = _load(_session(
            _user(1, "first"), *_call(2, "c1"), _assistant(3, "first done"),
            _ev(4, "event_msg", {"type": "task_complete"}),
            _user(5, "second"), *_call(6, "c2"), _assistant(7, "second done"),
            _ev(8, "event_msg", {"type": "task_complete"}),
        ))
        finishes = [step["finish"] for step in _assistant_steps(s)]
        self.assertEqual(finishes.count("stop"), 2, finishes)
        self.assertEqual(finishes[-1], "stop")

    def test_classic_shell_argv_becomes_a_command_string(self):
        # Older rollouts record the `shell` tool's command as argv; it used to
        # reach file-path extraction as a list and fail the whole analysis.
        s = _load(_session(_user(1, "fix it"), *_call(2, "c1"), _assistant(3, "done")))
        calls = [tc for step in s.steps for tc in step.get("tool_calls") or []]
        self.assertTrue(calls)
        self.assertEqual(calls[0]["input"].get("command"), "bash -lc ls")

    def test_task_complete_without_an_assistant_message_is_harmless(self):
        s = _load(_session(_user(1, "hello"), _ev(2, "event_msg", {"type": "task_complete"}),
                           _assistant(3, "late reply")))
        self.assertEqual(_assistant_steps(s)[-1]["finish"], "")


class CodexTurnTimingTests(unittest.TestCase):
    """A turn's final assistant step ends at that turn's ``task_complete``.

    Ending it at the next turn's start booked the idle time between turns as
    the step's own generation time: ten idle minutes made a 3-second step read
    as 603 seconds of LLM inference, and pushed the first response out to the
    start of the second turn.
    """

    def test_a_turns_final_step_ends_at_task_complete_not_at_the_next_turns_start(self):
        first, last = _assistant_steps(_two_turns_ten_minutes_apart())
        self.assertEqual(first["duration"], 3.0)  # 5 s -> 8 s, not 5 s -> 608 s
        self.assertEqual(last["duration"], 3.0)

    def test_the_last_turn_ends_at_its_task_complete_not_at_the_end_of_the_file(self):
        # A next turn that has started but recorded nothing yet is not part of
        # the last step; without the stamp the end-of-file backfill booked it.
        s = _load(_session(
            _user(0, "task"), _assistant(5, "done"),
            _ev(8, "event_msg", {"type": "task_complete"}),
            _ev(608, "event_msg", {"type": "task_started"}),
        ))
        self.assertEqual(_assistant_steps(s)[-1]["duration"], 3.0)

    def test_a_later_turn_without_a_reply_does_not_move_this_turns_end(self):
        # The second task_complete finds the same final message; its recorded
        # end stays put.
        s = _load(_session(
            _user(0, "task"), _assistant(5, "done"),
            _ev(8, "event_msg", {"type": "task_complete"}),
            _ev(600, "event_msg", {"type": "task_started"}),
            _ev(608, "event_msg", {"type": "task_complete"}),
        ))
        self.assertEqual(_assistant_steps(s)[-1]["duration"], 3.0)

    def test_a_task_complete_stamped_before_its_message_does_not_end_it(self):
        # Out-of-order stamps (a hand-edited file) give no valid end, not a
        # negative duration.
        s = _load(_session(_user(0, "task"), _assistant(5, "done"),
                           _ev(3, "event_msg", {"type": "task_complete"})))
        self.assertIsNone(_assistant_steps(s)[-1]["duration"])

    def test_first_response_latency_ends_at_the_first_turns_task_complete(self):
        self.assertEqual(_two_turns_ten_minutes_apart().metrics["time_to_first_token"], 8.0)

    def test_a_cut_off_turn_ahead_of_the_first_user_step_is_not_its_response(self):
        # A sub-agent's first task arrives as a developer message. Its turn is
        # cut off before task_complete, so it is taken to end where the next
        # message begins: the user's question at 10 s.
        s = _load(_session(
            _ev(1, "response_item", {"type": "message", "role": "developer",
                                     "content": [{"type": "input_text", "text": "sub-task"}]}),
            _assistant(5, "early"),
            _user(10, "question"), _assistant(15, "answer"),
            _ev(18, "event_msg", {"type": "task_complete"}),
        ))
        self.assertEqual(s.metrics["time_to_first_token"], 8.0)  # 10 s -> 18 s, not 0.0


class FirstResponseLatencyTests(unittest.TestCase):
    """The first response began and completed at or after the first user step.

    A Codex sub-agent's first task arrives as a non-user message, so a turn
    can run before the first user step exists. Measured from that later user
    step, its completion gave a negative latency, or 0.0 when the turn was cut
    off and taken to end where the user step begins.
    """

    def test_a_step_completed_before_the_first_user_step_is_not_its_response(self):
        steps = [_timed("assistant", 0, 1), _timed("user", 5, 5)]
        self.assertIsNone(_compute_timing_metrics(steps)["time_to_first_token"])

    def test_latency_runs_to_the_first_step_after_the_first_user_step(self):
        steps = [_timed("assistant", 0, 1), _timed("user", 5, 5), _timed("assistant", 6, 9)]
        self.assertEqual(_compute_timing_metrics(steps)["time_to_first_token"], 4.0)

    def test_a_step_that_began_before_the_first_user_step_is_not_its_response(self):
        # It completes exactly as the user step begins.
        steps = [_timed("assistant", 5, 10), _timed("user", 10, 10), _timed("assistant", 15, 18)]
        self.assertEqual(_compute_timing_metrics(steps)["time_to_first_token"], 8.0)

    def test_a_reply_stamped_in_its_questions_second_is_the_first_response(self):
        # Second-resolution stamps can put a fast reply in the same second.
        steps = [_timed("user", 5, 5), _timed("assistant", 5, 5), _timed("assistant", 6, 9)]
        self.assertEqual(_compute_timing_metrics(steps)["time_to_first_token"], 0.0)

    def test_a_step_stamped_completed_before_it_began_gives_no_negative_latency(self):
        steps = [_timed("user", 5, 5), _timed("assistant", 6, 4)]
        self.assertIsNone(_compute_timing_metrics(steps)["time_to_first_token"])

    def test_a_step_without_a_creation_stamp_is_judged_by_its_completion(self):
        # Nothing shows it began before the user step.
        steps = [_timed("user", 5, 5),
                 {"role": "assistant", "time_created_ms": None, "time_completed_ms": 9_000}]
        self.assertEqual(_compute_timing_metrics(steps)["time_to_first_token"], 4.0)


if __name__ == "__main__":
    unittest.main()
