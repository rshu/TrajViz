"""User repeated messages: the exact same user text sent two or more times.

Only whitespace is normalized; anything else ("继续" vs "请继续") is a
different message and stays out, keeping the rule explainable in one
sentence.
"""

import unittest

from agent_trajviz.insight.patterns import detect_user_repeats
from agent_trajviz.insight.issues import prepare_overview_issues
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from tests.overview_session_stub import overview_session as _session

def _user(index, text):
    return {"index": index, "role": "user", "text_preview": text, "tokens": {}}

def _assistant(index):
    return {"index": index, "role": "assistant", "text_preview": "done", "tokens": {}}

class DetectUserRepeatsTests(unittest.TestCase):
    def test_verbatim_repeat_is_flagged(self):
        steps = [
            _user(0, "把报告改成中文"),
            _assistant(1),
            _user(2, "把报告改成中文"),
        ]
        reps = detect_user_repeats(steps)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["count"], 2)
        self.assertEqual(reps[0]["steps"], [0, 2])

    def test_whitespace_differences_do_not_hide_a_repeat(self):
        steps = [
            _user(0, "fix the bug\n   in parser.py"),
            _user(4, "fix the bug in parser.py"),
        ]
        reps = detect_user_repeats(steps)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["message"], "fix the bug in parser.py")

    def test_different_messages_are_not_flagged(self):
        steps = [
            _user(0, "继续"),
            _user(2, "请继续"),
            _user(4, "继续吧"),
        ]
        self.assertEqual(detect_user_repeats(steps), [])

    def test_assistant_text_is_ignored(self):
        steps = [
            _user(0, "done"),
            _assistant(1),  # same text, wrong role
        ]
        self.assertEqual(detect_user_repeats(steps), [])

    def test_three_repeats_count_and_sort_first(self):
        steps = [
            _user(0, "继续"),
            _assistant(1),
            _user(2, "继续"),
            _assistant(3),
            _user(4, "继续"),
            _user(6, "别的消息"),
            _user(8, "别的消息"),
        ]
        reps = detect_user_repeats(steps)
        self.assertEqual([r["count"] for r in reps], [3, 2])
        self.assertEqual(reps[0]["steps"], [0, 2, 4])

    def test_empty_messages_are_skipped(self):
        steps = [_user(0, ""), _user(2, "   ")]
        self.assertEqual(detect_user_repeats(steps), [])

class UserRepeatCardTests(unittest.TestCase):
    def _repeats(self):
        return [{"message": "把报告改成中文", "count": 2, "steps": [0, 2]}]

    def test_synthetic_continuation_messages_are_not_user_repeats(self):
        # OpenCode injects "Continue if you have next steps..." after each
        # compaction. It repeats verbatim across compactions but the framework
        # typed it, not the user.
        def _synthetic(index, text="Continue if you have next steps, or stop."):
            return {
                "index": index,
                "role": "user",
                "tool_calls": [],
                "tokens": {},
                "parts": [{"type": "text", "text": text, "synthetic": True}],
            }

        steps = [_synthetic(24), _synthetic(33), _synthetic(39)]
        self.assertEqual(detect_user_repeats(steps), [])

    def test_mixed_synthetic_and_typed_text_uses_the_typed_one(self):
        step = {
            "index": 1,
            "role": "user",
            "tool_calls": [],
            "tokens": {},
            "parts": [
                {"type": "text", "text": "Continue if you have next steps.", "synthetic": True},
                {"type": "text", "text": "请继续", "synthetic": False},
            ],
        }
        hits = detect_user_repeats([step, dict(step, index=5)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["message"], "请继续")

    def test_empty_leading_text_part_does_not_hide_later_text(self):
        step = {
            "index": 1,
            "role": "user",
            "tool_calls": [],
            "tokens": {},
            "parts": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "真实消息"},
            ],
        }
        hits = detect_user_repeats([step, dict(step, index=4)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["message"], "真实消息")

    def test_typed_text_parts_still_count(self):
        def _typed(index, text="把报告发我"):
            return {
                "index": index,
                "role": "user",
                "tool_calls": [],
                "tokens": {},
                "parts": [{"type": "text", "text": text}],
            }

        hits = detect_user_repeats([_typed(1), _typed(4)])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["count"], 2)

    def test_card_is_built_from_the_detector_output(self):
        issues = prepare_overview_issues(_session(user_repeats=self._repeats()))
        card = [i for i in issues if i.source_id == "antipattern:user_repeat:0"]
        self.assertEqual(len(card), 1)
        self.assertEqual(card[0].kind, "antipattern")
        self.assertEqual(card[0].title, "User repeated a message (2×)")
        self.assertEqual(card[0].detail, "把报告改成中文")
        self.assertEqual(card[0].steps, (0, 2))

    def test_no_card_when_nothing_was_flagged(self):
        issues = prepare_overview_issues(_session())
        self.assertFalse([i for i in issues if "user_repeat" in i.source_id])

    def test_card_is_localized_to_chinese(self):
        html = build_overview_issues_html(_session(user_repeats=self._repeats()))
        self.assertIn("用户重复发送同一消息 (2×)", html)
        self.assertIn("把报告改成中文", html)
        self.assertIn("最直接的证据", html)
        self.assertIn("查看第一次发送该消息后智能体的响应", html)

if __name__ == "__main__":
    unittest.main()
