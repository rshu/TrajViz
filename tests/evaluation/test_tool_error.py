"""Tests for mechanical tool-call error detection (the behavioral tool-use signal).

Covers the high-precision text detector, the effective-error decision (explicit flag
OR result-text signature, scoped to avoid false positives on file-read content and to
exclude test failures), and that the fix makes a flagless scaffold (Codex-shaped log)
register tool errors instead of being structurally zero.

Run:  python3 -m unittest tests.test_tool_error -v
"""

from __future__ import annotations

import json
import unittest


from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    classify_tool_error,
    looks_like_tool_error,
    READ,
    RUN_COMMAND,
    RUN_TEST,
)
from agent_trajviz.awe._engine.trajsignals import _repair_and_tool_signals, compute
from agent_trajviz.awe._engine.gold import GoldReference


def _step(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _call(i, cid, action=RUN_COMMAND, native="exec_command", test=False, files=None):
    return _step(
        i,
        actor="agent",
        type="tool_call",
        canonical_action=action,
        native_tool=native,
        test_run=test,
        files=files or [],
        call_id=cid,
        args={"command": "cmd"},
    )


def _result(i, cid, output="", error=False):
    return _step(i, actor="tool", type="tool_result", output_excerpt=output, tool_error=error, call_id=cid)


def _traj(steps):
    return NormalizedTrajectory("a", "i", steps)


def _gold():
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=[],
        fail_to_pass=["pkg/tests/test_x.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
    )


class TestTextDetector(unittest.TestCase):
    """The detector must fire on high-precision shell/tool failure signatures and stay
    silent on benign output (including source/file content that mentions 'error')."""

    def test_invalid_target_hallucinated_path(self):
        self.assertEqual(classify_tool_error("cat: pkg/x.py: No such file or directory"), "invalid_target")
        self.assertTrue(looks_like_tool_error("fatal: pathspec 'a/b.py' did not match any file"))

    def test_unfindable_edit_target_is_invalid_target(self):
        self.assertEqual(classify_tool_error("Error: String not found in file"), "invalid_target")

    def test_command_not_found(self):
        self.assertEqual(classify_tool_error("bash: frobnicate: command not found"), "command_not_found")

    def test_bad_arguments_and_failed_patch(self):
        self.assertEqual(classify_tool_error("error: patch does not apply"), "bad_arguments")
        self.assertEqual(classify_tool_error("usage: prog\nprog: error: unrecognized arguments: --x"), "bad_arguments")

    def test_permission_denied(self):
        self.assertEqual(classify_tool_error("bash: ./run.sh: Permission denied"), "permission_denied")

    def test_benign_output_is_not_an_error(self):
        for s in [
            "",
            "All tests passed",
            "def foo():\n    return 1",
            "Found 3 matches for 'error'",
            "raise ValueError('bad value')",
            "3 passed in 0.12s",
        ]:
            self.assertIsNone(classify_tool_error(s), f"false positive on: {s!r}")
            self.assertFalse(looks_like_tool_error(s))


class TestEffectiveToolError(unittest.TestCase):
    """_repair_and_tool_signals must count mechanical tool errors via flag-or-text,
    recover flagless (Codex-shaped) errors, exclude test failures and unscanned read
    content, and only blame 'gave up' (no retry) separately from the behavioral count."""

    def test_flagless_text_error_is_recovered_and_blamed_when_not_retried(self):
        # Codex shape: no explicit error flag, error only in the output text, no retry.
        t = _traj([_call(0, "c1"), _result(1, "c1", "cat: pkg/x.py: No such file or directory")])
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_call_errors"], 1)
        self.assertEqual(r["tool_error_subtypes"], {"invalid_target": 1})
        self.assertTrue(r["encountered_tool_error"])
        self.assertTrue(r["tool_failure_not_retried"])

    def test_recovered_error_is_counted_but_not_blamed(self):
        # Same error, but the agent issues another tool call right after -> recovered.
        t = _traj(
            [_call(0, "c1"), _result(1, "c1", "No such file or directory"), _call(2, "c2"), _result(3, "c2", "ok")]
        )
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_call_errors"], 1)  # behavioral: still counted
        self.assertFalse(r["tool_failure_not_retried"])  # blame: not charged

    def test_read_content_is_not_text_scanned(self):
        # A SUCCESSFUL read whose content mentions an error string must NOT be flagged.
        t = _traj(
            [
                _call(0, "c1", action=READ, native="Read"),
                _result(1, "c1", "log line: No such file or directory\n... more"),
            ]
        )
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_call_errors"], 0)

    def test_read_with_explicit_error_flag_counts(self):
        # A genuinely failed read (scaffold set the flag) — e.g. a hallucinated path.
        t = _traj([_call(0, "c1", action=READ, native="Read"), _result(1, "c1", "opaque", error=True)])
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_call_errors"], 1)
        self.assertEqual(r["tool_error_subtypes"], {"other": 1})  # flag set, no text signature

    def test_test_failures_are_not_tool_errors(self):
        # A failing test (flagged) is a test failure, not a tool-call failure.
        t = _traj([_call(0, "c1", action=RUN_TEST, test=True), _result(1, "c1", "1 failed", error=True)])
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_call_errors"], 0)
        self.assertEqual(r["n_tool_calls"], 0)  # test calls excluded from the denominator

    def test_denominator_excludes_only_tests(self):
        t = _traj(
            [
                _call(0, "c1"),
                _result(1, "c1", "ok"),
                _call(2, "c2", action=RUN_TEST, test=True),
                _result(3, "c2", "ok"),
                _call(4, "c3"),
                _result(5, "c3", "ok"),
            ]
        )
        r = _repair_and_tool_signals(t)
        self.assertEqual(r["n_tool_calls"], 2)
        self.assertEqual(r["n_tool_call_errors"], 0)


class TestComputeIntegration(unittest.TestCase):
    """The new fields must flow through compute() and survive the cache round-trip."""

    def test_signals_carry_tool_error_fields_and_serialize(self):
        t = _traj([_call(0, "c1"), _result(1, "c1", "bash: foo: command not found")])
        sig = compute(t, _gold())
        self.assertEqual(sig.n_tool_call_errors, 1)
        self.assertEqual(sig.tool_error_subtypes, {"command_not_found": 1})
        self.assertEqual(sig.n_tool_calls, 1)
        # cache round-trip (signals_cached reconstructs via TrajectorySignals(**json)).
        from agent_trajviz.awe._engine.trajsignals import TrajectorySignals

        again = TrajectorySignals(**json.loads(json.dumps(sig.to_dict())))
        self.assertEqual(again.tool_error_subtypes, {"command_not_found": 1})


if __name__ == "__main__":
    unittest.main()
