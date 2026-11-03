"""Regressions for signals lost or invented during normalization."""

import unittest
from agent_trajviz.awe._engine.adapters import adapt_claude_code, adapt_codex, adapt_opencode
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep as S,
    NormalizedTrajectory as T,
    edit_reversals,
    read_observations,
)


class EvidencePreservation(unittest.TestCase):
    def test_middle_output_and_native_input_roundtrip(self):
        output = "a" * 900 + "\nMIDDLE_SIGNAL\n" + "b" * 900
        raw = {
            "trajectory": [
                {
                    "role": "assistant",
                    "index": 0,
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "Read",
                            "id": "r",
                            "input": {"file_path": "a.py", "limit": 200, "extra": "keep"},
                        },
                        {"type": "tool_result", "tool_call_id": "r", "output": output},
                    ],
                }
            ]
        }
        traj = T.from_dict(adapt_claude_code(raw, "a", "i").to_dict())
        self.assertEqual(traj.steps[1].output_excerpt, output)
        self.assertEqual(traj.steps[0].native_input["extra"], "keep")
        self.assertEqual(traj.steps[1].source_ref["pointer"], "/trajectory/0/content/1")

    def test_pending_read_is_unanswered(self):
        traj = adapt_opencode(
            {
                "messages": [
                    {
                        "info": {"role": "assistant"},
                        "parts": [
                            {
                                "type": "tool",
                                "tool": "read",
                                "callID": "r",
                                "state": {"status": "pending", "input": {"filePath": "a.py"}},
                            }
                        ],
                    }
                ]
            },
            "a",
            "i",
        )
        self.assertEqual(len(traj.steps), 1)
        self.assertEqual(read_observations(traj)[2], {"a.py"})

    def test_workdir_and_visible_summary(self):
        traj = adapt_codex(
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": "r",
                        "arguments": '{"cmd":"cat core.py","workdir":"pkg"}',
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "reasoning", "summary": [{"text": "Visible constraint"}]},
                },
            ],
            "codex",
            "i",
        )
        self.assertEqual(traj.steps[0].files, ["pkg/core.py"])
        self.assertEqual(traj.steps[1].text, "Visible constraint")
        self.assertEqual(traj.steps[1].observability, "observed")

    def _edits(self, transitions, failed=False, results=True):
        steps = []
        for i, (old, new) in enumerate(transitions):
            steps.append(
                S(
                    len(steps),
                    i,
                    "agent",
                    "tool_call",
                    canonical_action="edit",
                    files=["a.py"],
                    call_id=str(i),
                    args={"old_content": old, "new_content": new},
                )
            )
            if results:
                steps.append(S(len(steps), i, "tool", "tool_result", call_id=str(i), tool_error=failed))
        return T("a", "i", steps)

    def test_only_observed_directed_inverse_counts(self):
        self.assertTrue(edit_reversals(self._edits([("x=1", "x=2"), ("x=2", "x=1")])))
        for transitions in (
            [("\n\nimport copy", "\nimport copy")] * 3,
            [("x=1", "x=2"), ("x=2", "x=3")],
            [("x=1", "x=2"), ("other=1", "other=2"), ("x=2", "x=1")],
        ):
            self.assertFalse(edit_reversals(self._edits(transitions)))
        self.assertFalse(edit_reversals(self._edits([("a", "b"), ("b", "a")], failed=True)))
        self.assertFalse(edit_reversals(self._edits([("a", "b"), ("b", "a")], results=False)))

    def test_long_shared_prefix_does_not_hide_forward_edit(self):
        prefix = "\n".join(f"line{i}" for i in range(65))
        traj = adapt_claude_code(
            {
                "trajectory": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "name": "Edit",
                                "id": "e",
                                "input": {
                                    "file_path": "a.py",
                                    "old_string": prefix + "\nx=1",
                                    "new_string": prefix + "\nx=2",
                                },
                            }
                        ],
                    }
                ]
            },
            "a",
            "i",
        )
        self.assertEqual(len(traj.steps[0].args["new_lines"]), 66)
        self.assertTrue(traj.steps[0].args["new_content"].endswith("x=2"))

    def test_delete_content_is_not_missing(self):
        traj = adapt_claude_code(
            {
                "trajectory": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "name": "Edit",
                                "id": "e",
                                "input": {"file_path": "a.py", "old_string": "x", "new_string": ""},
                            }
                        ],
                    }
                ]
            },
            "a",
            "i",
        )
        self.assertEqual(traj.steps[0].args["new_content"], "")
