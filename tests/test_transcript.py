"""Tests for the expanded-step body (full text, collapsed reasoning and tools)."""

import unittest

from agent_trajviz.insight.parser import parse_steps
from agent_trajviz.insight.presenters.transcript import TRANSCRIPT_CSS, build_step_parts_html


def _step(role="assistant", index=0, parts=None, text_preview="", tokens=None,
          duration=None, time_created_ms=None, agent="", is_sub_agent=False,
          session_id=""):
    return {
        "index": index, "raw_index": index, "role": role, "agent": agent,
        "session_id": session_id, "is_sub_agent": is_sub_agent,
        "parts": parts or [], "tool_calls": [], "tool_call_count": 0,
        "error_count": 0, "has_reasoning": False, "text_preview": text_preview,
        "tokens": tokens if tokens is not None else {
            "total": 0, "input": 0, "output": 0,
            "cache_read": 0, "cache_write": 0,
        },
        "duration": duration, "time_created_ms": time_created_ms,
        "finish": "", "model_id": "", "provider_id": "",
    }


def _html(steps) -> str:
    return "\n".join(build_step_parts_html(step) for step in steps)


class StepBodyTests(unittest.TestCase):
    def test_full_text_is_not_the_preview(self):
        long_text = "HEAD " + "x" * 3000
        html_out = _html([
            _step(text_preview=long_text[:80], parts=[{"type": "text", "text": long_text}]),
            _step(index=1, text_preview="short preview", parts=[{"type": "text", "text": "middle message"}]),
        ])
        self.assertIn("x" * 3000, html_out)
        self.assertIn("middle message", html_out)
        self.assertIn("white-space: pre-wrap", TRANSCRIPT_CSS)

    def test_reasoning_is_collapsed_with_char_count(self):
        reasoning = "thinking " * 100  # 900 chars
        steps = [_step(index=0, parts=[
            {"type": "reasoning", "text": reasoning},
        ])]
        html_out = _html(steps)
        self.assertIn("<details", html_out)
        self.assertIn(f"Reasoning ({len(reasoning)} chars)", html_out)
        # Default collapsed: no `open` attribute on the reasoning block.
        details_start = html_out.index("Reasoning (")
        block = html_out[html_out.rindex("<details", 0, details_start):]
        self.assertNotIn("<details open", block[: block.index(">") + 1])
        # Full reasoning text is inside the details body.
        self.assertIn(reasoning, html_out)

    def test_reasoning_label_uses_the_real_token_metric(self):
        reasoning = "thinking " * 100
        steps = [_step(index=0, parts=[
            {"type": "reasoning", "text": reasoning},
        ], tokens={"total": 9100, "input": 0, "output": 100,
                   "reasoning": 9000, "cache_read": 0, "cache_write": 0})]
        html_out = _html(steps)
        self.assertIn("Reasoning (9,000 tok)", html_out)
        self.assertNotIn("chars)", html_out)

    def test_reasoning_label_marks_parts_when_a_step_has_several(self):
        # Step-level tokens cannot be split per part; every label carries
        # the total plus the part ordinal.
        steps = [_step(index=0, parts=[
            {"type": "reasoning", "text": "first thought"},
            {"type": "text", "text": "answer"},
            {"type": "reasoning", "text": "second thought"},
        ], tokens={"total": 500, "input": 0, "output": 100,
                   "reasoning": 400, "cache_read": 0, "cache_write": 0})]
        html_out = _html(steps)
        self.assertIn("Reasoning (400 tok · part 1/2)", html_out)
        self.assertIn("Reasoning (400 tok · part 2/2)", html_out)

    def test_tool_call_is_collapsed_with_name_title_and_duration(self):
        steps = [_step(index=0, parts=[{
            "type": "tool_call", "tool_name": "bash", "title": "run tests",
            "status": "completed",
            "input": {"command": "pytest -q"},
            "output": "3 passed",
            "time_start": 1_000_000, "time_end": 1_000_750,
        }])]
        html_out = _html(steps)
        self.assertIn("<details", html_out)
        self.assertIn("bash", html_out)
        self.assertIn("run tests", html_out)
        self.assertIn("0.75s", html_out)
        self.assertIn("pytest -q", html_out)
        self.assertIn("3 passed", html_out)

    def test_tool_call_output_is_never_truncated(self):
        # The Workflow detail panel caps output at 2000 chars; the transcript
        # is the full-fidelity view, so a 5000-char output must survive whole.
        long_output = "y" * 5000
        steps = [_step(index=0, parts=[{
            "type": "tool_call", "tool_name": "read", "title": "big file",
            "status": "completed", "input": {"file_path": "/tmp/big.txt"},
            "output": long_output,
        }])]
        html_out = _html(steps)
        self.assertIn(long_output, html_out)
        self.assertNotIn("truncated", html_out)
        self.assertNotIn("...", html_out[html_out.index("big file"):])

    def test_tool_call_error_is_shown(self):
        steps = [_step(index=0, parts=[{
            "type": "tool_call", "tool_name": "bash", "title": "fails",
            "status": "error", "input": {"command": "ls /nope"},
            "output": "", "error": "No such file or directory",
        }])]
        html_out = _html(steps)
        self.assertIn("error", html_out)
        self.assertIn("No such file or directory", html_out)

    def test_tool_call_without_title_falls_back_to_key_argument(self):
        steps = [_step(index=0, parts=[{
            "type": "tool_call", "tool_name": "bash", "title": "",
            "status": "completed", "input": {"command": "echo hi"},
            "output": "hi",
        }])]
        self.assertIn("echo hi", _html(steps))


class TranscriptEscapingTests(unittest.TestCase):
    def test_script_tags_and_quotes_are_escaped(self):
        # html.escape(quote=True) is the project bar (test_render_injection):
        # both element-content and attribute-context breakouts must be inert.
        hostile_text = '<script>alert("x")</script> " onmouseover="evil \' onclick=\'evil'
        html_out = _html([
            _step(parts=[{"type": "text", "text": hostile_text}]),
            _step(index=1, parts=[{
                "type": "tool_call", "tool_name": "bash",
                "title": '" onmouseover="evil',
                "status": "completed",
                "input": {"command": "' onclick='evil"},
                "output": hostile_text,
            }]),
        ])
        self.assertNotIn("<script>", html_out)
        self.assertNotIn('" onmouseover="', html_out)
        self.assertNotIn("' onclick='", html_out)
        self.assertIn("&lt;script&gt;", html_out)
        self.assertIn("&quot; onmouseover=&quot;", html_out)
        self.assertIn("&#x27; onclick=&#x27;", html_out)


class TranscriptPlaceholderTests(unittest.TestCase):
    def test_message_without_parts_shows_placeholder(self):
        steps = [_step(role="assistant", index=0, parts=[], text_preview="")]
        html_out = _html(steps)
        self.assertIn("(no content)", html_out)

    def test_message_with_only_step_markers_shows_placeholder(self):
        # step_start / step_finish are bookkeeping, not conversation content.
        steps = [_step(role="assistant", index=0, parts=[
            {"type": "step_start", "name": "main"},
            {"type": "step_finish", "name": "main", "reason": "stop"},
        ])]
        self.assertIn("(no content)", _html(steps))


class TranscriptFromParserTests(unittest.TestCase):
    """End-to-end: a normalized {info, messages} payload through parse_steps."""

    RAW = {
        "info": {"id": "ses_root"},
        "messages": [
            {
                "info": {"role": "user", "time": {"created": 1_000_000},
                         "sessionID": "ses_root"},
                "parts": [{"type": "text", "text": "please run the tests"}],
            },
            {
                "info": {
                    "role": "assistant",
                    "time": {"created": 1_001_000, "completed": 1_003_000},
                    "agent": "build", "sessionID": "ses_root",
                    "tokens": {"total": 500, "input": 400, "output": 100},
                },
                "parts": [
                    {"type": "reasoning", "text": "I should run pytest first."},
                    {"type": "text", "text": "Running the suite now."},
                    {
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "title": "pytest",
                            "input": {"command": "pytest -q"},
                            "output": "2 passed in 0.5s",
                            "time": {"start": 1_001_500, "end": 1_002_500},
                        },
                    },
                ],
            },
        ],
    }

    def test_parsed_steps_render_full_conversation(self):
        steps = parse_steps(self.RAW)
        html_out = _html(steps)
        reasoning_text = "I should run pytest first."
        self.assertIn("please run the tests", html_out)
        self.assertIn("Running the suite now.", html_out)
        self.assertIn(f"Reasoning ({len(reasoning_text)} chars)", html_out)
        self.assertIn(reasoning_text, html_out)
        self.assertIn("pytest -q", html_out)
        self.assertIn("2 passed in 0.5s", html_out)
        self.assertIn("1.00s", html_out)


class WorkflowExpandedViewTests(unittest.TestCase):
    """The former Transcript tab, embedded in Workflow as the Expanded view."""

    def _steps(self):
        return [
            _step(role="user", index=0, text_preview="user preview line",
                  parts=[{"type": "text", "text": "full user message"}],
                  time_created_ms=1_000_000),
            _step(role="assistant", index=1, text_preview="assistant preview line",
                  parts=[
                      {"type": "reasoning", "text": "thinking hard"},
                      {"type": "text", "text": "the whole answer"},
                      {"type": "tool_call", "tool": "bash",
                       "input": {"command": "echo hi"}, "output": "hi",
                       "state": {"status": "completed"}},
                  ],
                  time_created_ms=1_001_000),
        ]

    def test_expanded_cards_carry_full_content_and_stable_ids(self):
        from agent_trajviz.insight.rendering import render_workflow_html

        out = render_workflow_html(self._steps(), expanded=True)
        # Full content inline, not the one-line preview.
        self.assertIn("wf-expand-body", out)
        self.assertIn("full user message", out)
        self.assertIn("the whole answer", out)
        self.assertIn("echo hi", out)
        # Reasoning and tool I/O stay collapsed so layout stays cheap.
        self.assertIn("<details", out)
        # Card ids/classes survive — selection, j/k, deep links, detail store.
        self.assertIn('id="wf-card-0"', out)
        self.assertIn('id="wf-card-1"', out)
        self.assertIn("wf-header", out)
        # Transcript CSS ships with the expanded view.
        self.assertIn("tv-text", out)

    def test_expanded_text_parts_carry_a_text_label(self):
        # 裸段落夹在 Reasoning / Tool 折叠块之间时无法辨认是哪种内容，
        # 需要与 Input/Output 同款的小标识。
        from agent_trajviz.insight.rendering import render_workflow_html

        out = render_workflow_html(self._steps(), expanded=True)
        self.assertIn("<div class='tv-io-label'>Text</div>", out)
        self.assertEqual(2, out.count("<div class='tv-io-label'>Text</div>"))

        # 空 text part 不产生标签（与详情面板口径一致）。
        empty = [_step(role="assistant", index=0,
                       parts=[{"type": "text", "text": ""}])]
        out2 = render_workflow_html(empty, expanded=True)
        self.assertNotIn("tv-io-label'>Text", out2)

    def test_cards_mode_keeps_the_compact_preview(self):
        from agent_trajviz.insight.rendering import render_workflow_html

        out = render_workflow_html(self._steps())
        self.assertIn("wf-preview", out)
        self.assertNotIn("<div class='wf-expand-body'>", out)
        self.assertNotIn("the whole answer", out)

    def test_filtered_outputs_respect_the_view_mode(self):
        from agent_trajviz.insight.presenters.workflow import build_filtered_workflow_outputs

        steps = self._steps()
        cards, _, _ = build_filtered_workflow_outputs(
            steps, "Assistant,User,All", "")
        expanded, count, _ = build_filtered_workflow_outputs(
            steps, "Assistant,User,All", "", expanded=True)
        self.assertNotIn("<div class='wf-expand-body'>", cards)
        self.assertIn("wf-expand-body", expanded)
        self.assertIn("the whole answer", expanded)
        self.assertIn("Showing 2 of 2 steps", count)
        # Keyword filtering applies in expanded mode too.
        narrowed, _, _ = build_filtered_workflow_outputs(
            steps, "Assistant,User,All", "assistant", expanded=True)
        self.assertIn("wf-expand-body", narrowed)
        self.assertIn("the whole answer", narrowed)
        self.assertNotIn("full user message", narrowed)
