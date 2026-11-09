"""Blind retries: a failed call resubmitted verbatim and failing again.

Only consecutive identical calls count (same tool, same input, nothing in
between). A retry that succeeds is exempt — transient errors deserve one
verbatim shot. Flagged is the wasted pattern: unchanged input in an unchanged
environment fails the same way twice.
"""

import unittest

from agent_trajviz.insight.patterns import detect_blind_retries
from agent_trajviz.insight.issues import prepare_overview_issues
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from tests.overview_session_stub import overview_session as _session

def _step(index, calls=None):
    return {"index": index, "role": "assistant", "tool_calls": calls or [], "tokens": {}}

def _call(tool="bash", command="ls /tmp", status="completed", title=None):
    return {
        "tool_name": tool,
        "status": status,
        "input": {"command": command},
        "title": title or command,
        "output": "ok",
    }

def _failed(tool="bash", command="ls /tmp", title=None):
    return _call(tool=tool, command=command, status="error", title=title)

class DetectBlindRetriesTests(unittest.TestCase):
    def test_failed_then_identical_failed_retry_is_flagged(self):
        steps = [
            _step(0, [_failed(command="cat missing.txt")]),
            _step(1, [_failed(command="cat missing.txt")]),
        ]
        retries = detect_blind_retries(steps)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["tool"], "bash")
        self.assertEqual(retries[0]["attempts"], 2)
        self.assertEqual(retries[0]["failed"], 2)
        self.assertEqual(retries[0]["steps"], [0, 1])

    def test_retry_that_succeeds_is_exempt(self):
        # Transient error: fail once, verbatim retry succeeds — not flagged.
        steps = [
            _step(0, [_failed(command="curl api")]),
            _step(1, [_call(command="curl api")]),
        ]
        self.assertEqual(detect_blind_retries(steps), [])

    def test_changed_input_is_not_a_blind_retry(self):
        steps = [
            _step(0, [_failed(command="grep foo src/")]),
            _step(1, [_failed(command="grep foo src/ --include=*.py")]),
        ]
        self.assertEqual(detect_blind_retries(steps), [])

    def test_other_call_in_between_breaks_the_run(self):
        steps = [
            _step(0, [_failed(command="cat a")]),
            _step(1, [_call(tool="read", command="b")]),
            _step(2, [_failed(command="cat a")]),
        ]
        self.assertEqual(detect_blind_retries(steps), [])

    def test_three_in_a_row_across_steps_counts_one_run(self):
        steps = [
            _step(0, [_failed(command="make")]),
            _step(1, [_failed(command="make"), _failed(command="make")]),
        ]
        retries = detect_blind_retries(steps)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["attempts"], 3)
        self.assertEqual(retries[0]["steps"], [0, 1])

    def test_success_first_then_failures_is_not_flagged(self):
        # The run must *start* with the failure; a later hiccup after success
        # is a different story (the environment changed under the agent).
        steps = [
            _step(0, [_call(command="make")]),
            _step(1, [_failed(command="make")]),
        ]
        self.assertEqual(detect_blind_retries(steps), [])

    def test_dict_inputs_are_compared_key_order_independent(self):
        steps = [
            _step(0, [{"tool_name": "edit", "status": "error",
                       "input": {"filePath": "a.py", "old_string": "x"}}]),
            _step(1, [{"tool_name": "edit", "status": "error",
                       "input": {"old_string": "x", "filePath": "a.py"}}]),
        ]
        retries = detect_blind_retries(steps)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["tool"], "edit")

class BlindRetryCardTests(unittest.TestCase):
    def _retries(self):
        return [{
            "tool": "bash", "attempts": 3, "failed": 3,
            "steps": [4, 5], "display": "pytest tests/",
        }]

    def test_card_is_built_from_the_detector_output(self):
        issues = prepare_overview_issues(_session(blind_retries=self._retries()))
        card = [i for i in issues if i.source_id == "antipattern:blind_retry:0"]
        self.assertEqual(len(card), 1)
        self.assertEqual(card[0].kind, "antipattern")
        self.assertEqual(card[0].title, "Blind retry of failed bash call (3×)")
        self.assertIn("3 of 3 identical attempts failed — pytest tests/", card[0].detail)
        self.assertEqual(card[0].steps, (4, 5))

    def test_no_card_when_nothing_was_flagged(self):
        issues = prepare_overview_issues(_session())
        self.assertFalse([i for i in issues if "blind_retry" in i.source_id])

    def test_card_is_localized_to_chinese(self):
        html = build_overview_issues_html(_session(blind_retries=self._retries()))
        self.assertIn("失败后原样重试 bash（3×）", html)
        self.assertIn("3 次完全相同的调用中 3 次失败 — pytest tests/", html)
        self.assertIn("同样的输入在同样的环境下只会得到同样的结果", html)
        self.assertIn("失败后先读错误信息", html)

if __name__ == "__main__":
    unittest.main()
