"""Tests for the Deep Dive "top context-growth steps" hotspot table."""

import re
import unittest

from agent_trajviz.insight.analysis import hotspots
from agent_trajviz.insight.formatting import _build_hotspots_md


def _row(
    idx,
    role="assistant",
    agent="main",
    tokens_input=0,
    cache_read=0,
    tokens_total=0,
    tool_calls=0,
):
    prompt = tokens_input + cache_read
    return {
        "index": idx,
        "role": role,
        "agent": agent,
        "model_id": "",
        "finish": "",
        "duration": 1.0,
        "tokens_total": tokens_total,
        "tokens_input": tokens_input,
        "tokens_output": 0,
        "cache_read": cache_read,
        "non_cache_tokens": tokens_input,
        "cache_ratio": (cache_read / tokens_total) if tokens_total else None,
        "tokens_per_sec": None,
        "non_cache_per_sec": None,
        "output_input_ratio": None,
        "tool_calls": tool_calls,
        "errors": 0,
        "tool_time_sum": 0.0,
        "tool_time_share": 0.0,
        "reasoning_parts": 0,
        "text_parts": 0,
        "patch_parts": 0,
        "prompt_tokens": prompt,
    }


class TopGrowthRowsTests(unittest.TestCase):
    def test_growth_splits_into_completion_and_new_content(self):
        rows = [
            _row(1, tokens_input=500, cache_read=7500, tokens_total=9000),
            _row(2, tokens_input=300, cache_read=8700, tokens_total=9500),
            _row(3, tokens_input=100, cache_read=11900, tokens_total=13000),
        ]
        (bigger, smaller) = hotspots(rows)["top_growth"]
        # Step 2 -> 3: prompt 9000 -> 12000; completion 500, new content 2500.
        self.assertEqual(
            (bigger["index"], bigger["growth"], bigger["own_completion"], bigger["new_content"]),
            (2, 3000, 500, 2500),
        )
        # Step 1 -> 2: prompt 8000 -> 9000; completion 1000, no new content.
        self.assertEqual(
            (smaller["index"], smaller["growth"], smaller["own_completion"], smaller["new_content"]),
            (1, 1000, 1000, 0),
        )

    def test_ranking_is_by_growth_desc_limited_to_five(self):
        rows = []
        for i in range(1, 8):
            rows.append(_row(i, tokens_input=100, cache_read=1000 + 100 * i, tokens_total=1100))
        # Each step grows the prompt by 100 (a tie: stable order keeps step
        # order); the last step has no successor.
        top = hotspots(rows)["top_growth"]
        self.assertEqual([r["index"] for r in top], [1, 2, 3, 4, 5])

    def test_user_rows_between_calls_do_not_break_pairing(self):
        rows = [
            _row(1, tokens_input=100, cache_read=1000, tokens_total=1200),
            _row(2, role="user"),
            _row(3, tokens_input=100, cache_read=2100, tokens_total=2200),
        ]
        (only,) = hotspots(rows)["top_growth"]
        # The user message lands in the next prompt: growth 1100, of which
        # the assistant's own completion is 100.
        self.assertEqual(
            (only["index"], only["growth"], only["own_completion"], only["new_content"]),
            (1, 1100, 100, 1000),
        )

    def test_subagent_rows_never_count_as_parent_growth(self):
        rows = [
            _row(1, agent="main", tokens_input=100, cache_read=1000, tokens_total=1200),
            _row(2, agent="sub", tokens_input=900, cache_read=0, tokens_total=1000),
            _row(3, agent="sub", tokens_input=100, cache_read=1800, tokens_total=2000),
            _row(4, agent="main", tokens_input=100, cache_read=1100, tokens_total=1300),
        ]
        top = hotspots(rows)["top_growth"]
        by_index = {r["index"]: r for r in top}
        # main step 1 pairs with main step 4 (sub traffic skipped): +100.
        self.assertEqual(by_index[1]["growth"], 100)
        # sub step 2 pairs with sub step 3: 900 -> 1900, +1000.
        self.assertEqual(by_index[2]["growth"], 1000)
        # The last calls of each agent have no successor.
        self.assertNotIn(3, by_index)
        self.assertNotIn(4, by_index)

    def test_shrinking_context_is_not_growth(self):
        rows = [
            _row(1, tokens_input=100, cache_read=5000, tokens_total=5200),
            _row(2, tokens_input=100, cache_read=1000, tokens_total=1200),
        ]
        self.assertEqual(hotspots(rows)["top_growth"], [])

    def test_single_call_has_no_successor(self):
        rows = [_row(1, tokens_input=100, cache_read=1000, tokens_total=1200)]
        self.assertEqual(hotspots(rows)["top_growth"], [])

    def test_step_token_breakdown_keeps_nested_cache_reads_disjoint(self):
        rows = [
            _row(1, tokens_input=100, cache_read=40, tokens_total=110),
            _row(2, tokens_input=110, cache_read=90, tokens_total=118),
        ]
        steps = [
            {"tokens": {"total": 110, "input": 100, "output": 10, "cache_read": 40}},
            {"tokens": {"total": 118, "input": 110, "output": 8, "cache_read": 90}},
        ]
        (growth,) = hotspots(rows, steps=steps)["top_growth"]
        self.assertEqual((growth["growth"], growth["own_completion"], growth["new_content"]), (10, 10, 0))


class HotspotsMarkdownTests(unittest.TestCase):
    def test_growth_table_renders_with_footnote(self):
        rows = [
            _row(1, tokens_input=500, cache_read=7500, tokens_total=9000, tool_calls=3),
            _row(2, tokens_input=300, cache_read=8700, tokens_total=9500),
            _row(3, tokens_input=100, cache_read=11900, tokens_total=13000),
        ]
        md = _build_hotspots_md(rows)
        self.assertIn("**Top context-growth steps**", md)
        self.assertIn("+3,000 | 500 | 2,500 | 0 |", md)
        self.assertIn("Own Completion = the step's output + reasoning", md)

    def test_growth_table_omitted_without_successors(self):
        rows = [_row(1, tokens_input=100, cache_read=1000, tokens_total=1200)]
        md = _build_hotspots_md(rows)
        self.assertNotIn("Top context-growth steps", md)

    def test_step_numbers_are_workflow_jump_links(self):
        rows = [
            _row(1, tokens_input=500, cache_read=7500, tokens_total=9000, tool_calls=3),
            _row(2, tokens_input=300, cache_read=8700, tokens_total=9500),
            _row(3, tokens_input=100, cache_read=11900, tokens_total=13000),
        ]
        md = _build_hotspots_md(rows)
        # Every table's Step column links to the Workflow card of that step.
        # The link is a fragment anchor (not onclick: the client-side markdown
        # sanitizer strips event-handler attributes); insight.py's delegated
        # click handler turns #tv-step-N into the jump.
        for table in ("Top latency steps", "Top token-load steps", "Top context-growth steps"):
            self.assertIn(table, md)
        for idx in (1, 2, 3):
            self.assertIn(
                f"<a href='#tv-step-{idx}' class='insight-step-link'>{idx}</a>",
                md,
            )
        self.assertNotIn("onclick", md)
        # No bare (unlinkable) step cells remain in the Step column.
        self.assertIsNone(re.search(r"(?m)^\| \d+ \|", md))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
