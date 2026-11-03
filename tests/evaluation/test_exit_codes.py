"""Schema 19 (DESIGN §0ax): the adapters keep the recorded exit code of every
shell result that has one, Codex write_stdin continuation chunks fold back
into the originating exec_command result, and unittest's bare ``OK`` is a
clean verdict. The run-outcome classifiers already preferred a recorded exit
code; these tests pin the adapters actually delivering it per scaffold.

Run:  python3 -m unittest tests.test_exit_codes -v
"""

from __future__ import annotations

import json
import unittest


from agent_trajviz.awe._engine.adapters import adapt_claude_code, adapt_opencode, adapt_codex
from agent_trajviz.awe._engine.audit import _result_failed
from agent_trajviz.awe._engine.trajectory import codex_result_header, claude_code_exit_code, looks_like_test_failure
from agent_trajviz.awe._engine.trajsignals import _run_failed

ROOT = "/Users/rshu/Documents/GitHub/ARGUS/experiments/collect/workdirs/django__django-1/codex"


def _results(traj):
    return {s.call_id: s for s in traj.steps if s.type == "tool_result"}


class TestVerdictText(unittest.TestCase):
    def test_bare_ok_is_clean(self):
        out = "Creating test database...\n....\nRan 4 tests in 0.1s\n\nOK\n"
        self.assertFalse(looks_like_test_failure(out))
        self.assertFalse(looks_like_test_failure("Ran 4 tests\n\nOK (skipped=1)\n"))
        self.assertTrue(looks_like_test_failure("Ran 4 tests\n\nFAILED (failures=1)\n"))

    def test_codex_header_parse(self):
        done = (
            "Command: /bin/zsh -lc 'pytest -q'\nChunk ID: ab\nWall time: 0.0 seconds\n"
            "Process exited with code 1\nOriginal token count: 9\nOutput:\n1 failed\n"
        )
        self.assertEqual(codex_result_header(done), (1, None, "1 failed\n"))
        running = (
            "Command: /bin/zsh -lc 'pytest -q'\nChunk ID: ab\nWall time: 1.0 seconds\n"
            "Process running with session ID 4242\nOriginal token count: 9\nOutput:\n...."
        )
        self.assertEqual(codex_result_header(running), (None, 4242, "...."))
        self.assertEqual(codex_result_header("plain text"), (None, None, "plain text"))

    def test_claude_code_exit_text(self):
        self.assertEqual(claude_code_exit_code("Exit code 1\nTraceback ..."), 1)
        self.assertIsNone(claude_code_exit_code("error: command failed with exit code 1"))
        self.assertIsNone(claude_code_exit_code("all good"))


class TestClaudeCodeAdapter(unittest.TestCase):
    def test_exit_code_text_recorded(self):
        raw = {
            "trajectory": [
                {
                    "role": "assistant",
                    "index": 0,
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "c1",
                            "name": "Bash",
                            "input": {"command": "pytest tests/test_a.py -q"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "index": 1,
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "is_error": True,
                            "output": "Exit code 1\n1 failed, 2 passed",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "index": 2,
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "c2",
                            "name": "Bash",
                            "input": {"command": "pytest tests/test_a.py -q"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "index": 3,
                    "content": [{"type": "tool_result", "tool_call_id": "c2", "is_error": False, "output": "3 passed"}],
                },
            ]
        }
        t = adapt_claude_code(raw, "claude_code", "i")
        r = _results(t)
        self.assertEqual(r["c1"].args, {"exit_code": 1})
        self.assertEqual(r["c2"].args, {})
        self.assertTrue(_run_failed(r["c1"]))
        self.assertFalse(_run_failed(r["c2"]))


class TestOpenCodeAdapter(unittest.TestCase):
    def _raw(self, exit_code, output, status="completed"):
        return {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "k1",
                            "state": {
                                "status": status,
                                "input": {"command": "python -m pytest tests -q"},
                                "output": output,
                                "metadata": {"exit": exit_code, "truncated": False},
                            },
                        }
                    ],
                }
            ]
        }

    def test_metadata_exit_wins_over_text(self):
        # a verbose failing run whose excerpt shows no marker: exit code catches it
        t = adapt_opencode(self._raw(1, "collecting ...\n" + "x" * 2000), "opencode_glm", "i")
        r = _results(t)["k1"]
        self.assertEqual(r.args, {"exit_code": 1})
        self.assertTrue(_run_failed(r))
        self.assertTrue(_result_failed(r))  # audit mirror agrees
        # echoed failure text in a passing run: exit 0 vetoes it
        t = adapt_opencode(self._raw(0, "grep -n 'failed' tests/x.py\n12: assert not failed"), "opencode_glm", "i")
        r = _results(t)["k1"]
        self.assertEqual(r.args, {"exit_code": 0})
        self.assertFalse(_run_failed(r))
        self.assertFalse(_result_failed(r))

    def test_missing_metadata_falls_back_to_text(self):
        raw = self._raw(None, "1 failed")
        raw["messages"][0]["parts"][0]["state"]["metadata"] = {}
        r = _results(adapt_opencode(raw, "opencode_glm", "i"))["k1"]
        self.assertEqual(r.args, {})
        self.assertTrue(_run_failed(r))


def _codex_lines(*payloads):
    return [{"type": "response_item", "payload": p} for p in payloads]


def _call(cid, name, args):
    return {"type": "function_call", "name": name, "call_id": cid, "arguments": json.dumps(args)}


def _out(cid, text):
    return {"type": "function_call_output", "call_id": cid, "output": text}


def _hdr(cmd, *, exit_code=None, session=None, body=""):
    state = (
        f"Process exited with code {exit_code}"
        if exit_code is not None
        else f"Process running with session ID {session}"
    )
    return (
        f"Command: /bin/zsh -lc '{cmd}'\nChunk ID: 1a\nWall time: 1.0 seconds\n"
        f"{state}\nOriginal token count: 9\nOutput:\n{body}"
    )


class TestCodexAdapter(unittest.TestCase):
    def test_header_exit_code_recorded(self):
        lines = _codex_lines(
            _call("c1", "exec_command", {"cmd": "pytest tests/test_a.py -q"}),
            _out("c1", _hdr("pytest tests/test_a.py -q", exit_code=1, body="F\n1 failed\n")),
        )
        r = _results(adapt_codex(lines, "codex", "i"))["c1"]
        self.assertEqual(r.args, {"exit_code": 1})
        self.assertTrue(_run_failed(r))

    def test_continuation_folds_into_origin(self):
        lines = _codex_lines(
            _call("c1", "exec_command", {"cmd": "pytest tests/test_a.py -q"}),
            _out("c1", _hdr("pytest tests/test_a.py -q", session=77, body="collecting ...\n")),
            _call("c2", "write_stdin", {"session_id": 77, "chars": "", "yield_time_ms": 1000}),
            _out("c2", _hdr("pytest tests/test_a.py -q", session=77, body="...F\n")),
            _call("c3", "write_stdin", {"session_id": 77, "chars": "", "yield_time_ms": 1000}),
            _out(
                "c3",
                _hdr(
                    "pytest tests/test_a.py -q",
                    exit_code=1,
                    body="FAILED tests/test_a.py::test_x\n1 failed, 3 passed\n",
                ),
            ),
        )
        t = adapt_codex(lines, "codex", "i")
        r = _results(t)
        origin = r["c1"]
        self.assertEqual(origin.args["exit_code"], 1)
        self.assertEqual(origin.args["output_aggregation"]["source_steps"], [1, 3, 5])
        self.assertEqual(origin.args["output_aggregation"]["completion_step"], 5)
        self.assertIn("1 failed, 3 passed", origin.output_excerpt)
        self.assertIn("collecting", origin.output_excerpt)
        self.assertTrue(_run_failed(origin))
        # the chunks remain their own steps, linked back to the origin
        self.assertEqual(r["c2"].args.get("continuation_of"), "c1")
        self.assertEqual(r["c3"].args.get("continuation_of"), "c1")
        self.assertEqual(r["c3"].args.get("exit_code"), 1)
        # only the exec_command is a test run; the write_stdin calls are not
        tests = [s for s in t.steps if s.type == "tool_call" and s.test_run]
        self.assertEqual([s.call_id for s in tests], ["c1"])

    def test_unrelated_write_stdin_is_left_alone(self):
        lines = _codex_lines(
            _call("c1", "exec_command", {"cmd": "pytest -q"}),
            _out("c1", _hdr("pytest -q", exit_code=0, body="3 passed\n")),
            _call("c2", "write_stdin", {"session_id": 5, "chars": "y\n"}),
            _out("c2", _hdr("python repl", exit_code=0, body="ok\n")),
        )
        r = _results(adapt_codex(lines, "codex", "i"))
        self.assertNotIn("continuation_of", r["c2"].args)
        self.assertEqual(r["c1"].args, {"exit_code": 0})


if __name__ == "__main__":
    unittest.main()
