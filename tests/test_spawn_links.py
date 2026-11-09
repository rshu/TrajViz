"""Workflow spawn-link badge, compaction summary, and full tool output."""

import unittest

from i18n_test_helpers import unmark_html

from agent_trajviz.insight.parser import parse_steps
from agent_trajviz.insight.rendering import (
    _format_tool_call_detail,
    _spawn_step_map,
    format_step_detail,
    render_workflow_html,
)

ROOT_SID = "ses_root"
CHILD_SID = "ses_child"


def _msg(mid, sid, role="assistant", agent="build", parts=None,
         is_sub_agent=False, created=1000, completed=2000):
    return {
        "info": {
            "id": mid, "sessionID": sid, "role": role, "agent": agent,
            "isSubAgent": is_sub_agent,
            "tokens": {"input": 10, "output": 5, "reasoning": 0,
                       "cache": {"read": 0, "write": 0}},
            "time": {"created": created, "completed": completed},
        },
        "parts": parts or [],
    }


def _text_part(text):
    return {"type": "text", "text": text}


def _task_part(mid, sid, child_sid=None):
    state = {
        "status": "completed", "title": "spawn helper",
        "input": {"prompt": "go investigate"}, "output": "child finished",
        "time": {"start": 1100, "end": 1900},
    }
    if child_sid:
        state["metadata"] = {"sessionId": child_sid, "parentSessionId": sid}
    return {
        "type": "tool", "tool": "task", "callID": f"call-{mid}",
        "id": f"part-{mid}", "sessionID": sid, "messageID": mid,
        "state": state,
    }


def _raw(with_metadata=True):
    child_sid = CHILD_SID if with_metadata else None
    return {
        "info": {"id": ROOT_SID, "sessionID": ROOT_SID},
        "messages": [
            _msg("m0", ROOT_SID, role="user", parts=[_text_part("main task")]),
            _msg("m1", ROOT_SID, parts=[
                _text_part("spawning a helper"),
                _task_part("m1", ROOT_SID, child_sid),
            ]),
            _msg("m2", CHILD_SID, role="user", agent="explore",
                 is_sub_agent=not with_metadata,
                 parts=[_text_part("child prompt")]),
            _msg("m3", CHILD_SID, agent="explore",
                 is_sub_agent=not with_metadata,
                 parts=[_text_part("child answer")]),
        ],
    }


class SpawnStepMapTests(unittest.TestCase):
    def test_metadata_link_maps_child_to_spawning_step(self):
        steps = parse_steps(_raw(with_metadata=True))
        self.assertEqual({CHILD_SID: 1}, _spawn_step_map(steps))

    def test_fallback_nearest_preceding_spawn_step(self):
        steps = parse_steps(_raw(with_metadata=False))
        self.assertEqual({CHILD_SID: 1}, _spawn_step_map(steps))

    def test_no_spawn_steps_gives_empty_map(self):
        raw = _raw(with_metadata=False)
        raw["messages"][1]["parts"] = [_text_part("no tools here")]
        self.assertEqual({}, _spawn_step_map(parse_steps(raw)))


class SpawnBadgeTests(unittest.TestCase):
    def test_badge_shown_once_on_first_child_card(self):
        html_out = render_workflow_html(parse_steps(_raw(with_metadata=True)))
        self.assertEqual(1, html_out.count(">spawned by #1</span>"))
        self.assertIn("tvGotoWorkflowStep(1)", html_out)

    def test_no_badge_without_spawn_link(self):
        raw = _raw(with_metadata=False)
        raw["messages"][1]["parts"] = [_text_part("no tools here")]
        self.assertNotIn("spawned by", render_workflow_html(parse_steps(raw)))


class CompactionDetailTests(unittest.TestCase):
    def test_summary_text_is_rendered(self):
        raw = {
            "info": {"id": ROOT_SID},
            "messages": [
                _msg("m0", ROOT_SID, role="user", parts=[_text_part("hi")]),
                _msg("m1", ROOT_SID, parts=[{
                    "type": "compaction",
                    "summary": "用户要求修复登录页。已完成：定位 bug 于 auth.py。",
                    "reason": "auto",
                }]),
            ],
        }
        steps = parse_steps(raw)
        detail = format_step_detail(steps[1])
        self.assertIn("Compaction", detail)
        self.assertIn("定位 bug 于 auth.py", detail)
        self.assertIn("Reason: auto", detail)

    def test_missing_summary_shows_placeholder(self):
        raw = {
            "info": {"id": ROOT_SID},
            "messages": [
                _msg("m0", ROOT_SID, role="user", parts=[_text_part("hi")]),
                _msg("m1", ROOT_SID, parts=[{"type": "compaction"}]),
            ],
        }
        detail = format_step_detail(parse_steps(raw)[1])
        self.assertIn("no summary recorded", detail)


class ToolOutputFullTextTests(unittest.TestCase):
    def _tool_step(self, output):
        raw = {
            "info": {"id": ROOT_SID},
            "messages": [
                _msg("m0", ROOT_SID, role="user", parts=[_text_part("hi")]),
                _msg("m1", ROOT_SID, parts=[{
                    "type": "tool", "tool": "bash", "callID": "c1",
                    "id": "p1", "sessionID": ROOT_SID, "messageID": "m1",
                    "state": {"status": "completed", "input": {"command": "ls"},
                              "output": output},
                }]),
            ],
        }
        return parse_steps(raw)[1]

    def test_long_output_is_not_truncated_at_2000_chars(self):
        output = "x" * 5000
        detail = _format_tool_call_detail(self._tool_step(output)["parts"][0])
        self.assertNotIn("(truncated)", detail)
        self.assertIn("5,000 chars", detail)
        # The full text is present (escaped) inside the details body.
        self.assertIn("x" * 5000, detail)

    def test_short_output_keeps_char_count_summary(self):
        detail = unmark_html(_format_tool_call_detail(self._tool_step("ok")["parts"][0]))
        self.assertIn("Output (2 chars)", detail)
        self.assertIn("ok", detail)

    def test_output_is_json_serialized(self):
        detail = _format_tool_call_detail(
            self._tool_step({"files": ["a.py"]})["parts"][0]
        )
        self.assertIn("&quot;files&quot;", detail)


if __name__ == "__main__":
    unittest.main()
