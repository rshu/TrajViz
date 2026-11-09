"""Oversized single tool outputs: detection, issue card, and localization.

A tool result stays in the conversation history, so one oversized output is
paid for again on every later step until a compaction drops it. The detector
flags an output when it estimates to ≥5k tokens, or ≥2k tokens while covering
≥15% of that step's context occupancy.
"""

import unittest

from agent_trajviz.insight.patterns import detect_oversized_outputs
from agent_trajviz.insight.issues import prepare_overview_issues
from agent_trajviz.insight.presenters.issues import build_overview_issues_html
from tests.overview_session_stub import overview_session as _session

def _step(index, calls=None, *, input_tokens=0, cache_read=0):
    return {
        "index": index,
        "role": "assistant",
        "tool_calls": calls or [],
        "tokens": {
            "total": 0,
            "input": input_tokens,
            "output": 0,
            "reasoning": 0,
            "cache_read": cache_read,
            "cache_write": 0,
        },
    }

def _call(tool="bash", output="", title="cmd"):
    return {"tool_name": tool, "status": "success", "output": output, "title": title}

class DetectOversizedOutputsTests(unittest.TestCase):
    def test_absolute_threshold_flags_huge_output(self):
        # 15,000 ASCII chars ≈ 6,000 tokens — over the 5k absolute floor.
        steps = [_step(0, [_call(output="x" * 15_000)], input_tokens=50_000)]
        flagged = detect_oversized_outputs(steps)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["step"], 0)
        self.assertEqual(flagged[0]["tool"], "bash")
        self.assertGreaterEqual(flagged[0]["tokens"], 5_000)

    def test_relative_threshold_flags_output_filling_the_window(self):
        # 6,000 chars ≈ 2,400 tokens landing in a next-step context of 15k ≈ 16%.
        steps = [
            _step(4, [_call(tool="read", output="y" * 6_000)]),
            _step(5, input_tokens=15_000),
        ]
        flagged = detect_oversized_outputs(steps)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["share"], 16)

    def test_moderate_output_in_a_large_context_is_not_flagged(self):
        # Same 2,400 tokens but the next context is 50k → ~5%, below the bar.
        steps = [
            _step(4, [_call(output="y" * 6_000)]),
            _step(5, input_tokens=50_000),
        ]
        self.assertEqual(detect_oversized_outputs(steps), [])

    def test_small_outputs_are_never_flagged(self):
        steps = [_step(0, [_call(output="ok")], input_tokens=100)]
        self.assertEqual(detect_oversized_outputs(steps), [])

    def test_no_token_data_falls_back_to_the_absolute_threshold_only(self):
        # Occupancy unknown → share is None; only the 5k floor can fire.
        small = detect_oversized_outputs([_step(0, [_call(output="z" * 6_000)])])
        self.assertEqual(small, [])
        big = detect_oversized_outputs([_step(0, [_call(output="z" * 15_000)])])
        self.assertEqual(len(big), 1)
        self.assertIsNone(big[0]["share"])

    def test_no_output_placeholder_is_skipped(self):
        steps = [_step(0, [_call(output="(no output)")], input_tokens=100)]
        self.assertEqual(detect_oversized_outputs(steps), [])

    def test_results_are_capped_and_sorted_largest_first(self):
        calls = [_call(output="a" * (2 + i) * 7_500) for i in range(7)]
        steps = [_step(i, [c]) for i, c in enumerate(calls)]
        flagged = detect_oversized_outputs(steps)
        self.assertEqual(len(flagged), 5)
        self.assertEqual(
            [f["tokens"] for f in flagged],
            sorted([f["tokens"] for f in flagged], reverse=True),
        )

class OversizedOutputCardTests(unittest.TestCase):
    def _oversized(self):
        return [
            {"step": 3, "tool": "bash", "tokens": 8203, "share": 26, "title": "cat big.log"},
            {"step": 9, "tool": "read", "tokens": 5120, "share": None, "title": "read huge.py"},
        ]

    def test_card_is_built_from_the_detector_output(self):
        issues = prepare_overview_issues(_session(oversized_outputs=self._oversized()))
        card = [i for i in issues if i.source_id == "antipattern:oversized_output"]
        self.assertEqual(len(card), 1)
        self.assertEqual(card[0].kind, "antipattern")
        self.assertEqual(card[0].title, "Oversized tool outputs (2×)")
        self.assertIn("step 3 bash ≈8,203 tokens (26% of context)", card[0].detail)
        self.assertIn("step 9 read ≈5,120 tokens", card[0].detail)
        self.assertEqual(card[0].steps, (3, 9))

    def test_no_card_when_nothing_was_flagged(self):
        issues = prepare_overview_issues(_session())
        self.assertFalse([i for i in issues if i.source_id == "antipattern:oversized_output"])

    def test_card_is_localized_to_chinese(self):
        html = build_overview_issues_html(_session(oversized_outputs=self._oversized()))
        self.assertIn("单条工具输出过大 (2×)", html)
        self.assertIn("第3步 bash ≈8,203 token（占上下文 26%）", html)
        self.assertIn("第9步 read ≈5,120 token", html)
        self.assertIn("一直留在对话历史里", html)
        self.assertIn("带条数限制的读取", html)

if __name__ == "__main__":
    unittest.main()
