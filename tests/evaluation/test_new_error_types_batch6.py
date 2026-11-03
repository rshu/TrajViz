"""Tests for v0.27 batch 6 — edit-content capture, the calibrated
reverted_own_change promotion, the rejected (protective) walked-back anchor,
codex cwd tracking, and the unattributed-case dossier visibility.

Run:  python3 -m unittest tests.test_new_error_types_batch6 -v
"""

from __future__ import annotations

import unittest


from agent_trajviz.awe._engine.adapters import (
    adapt_claude_code,
    adapt_codex,
    adapt_opencode,
    _apply_patch_content,
    _command_cwd,
    _Builder,
)
from agent_trajviz.awe._engine.audit import audit_fault, RE_DERIVED, MISMATCH
from agent_trajviz.awe._engine.detect import objective_candidates, produced_error_types
from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.trajectory import NormalizedStep, NormalizedTrajectory, EDIT
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine import config as awe_config


def _gold(changed=None):
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=["pkg/a.py"],
        gold_symbols=[],
        gold_patch_loc=10,
        test_patch_files=["tests/test_a.py"],
        fail_to_pass=["tests/test_a.py::test_x"],
        pass_to_pass=[],
        is_multi_file=False,
        is_cross_module=False,
        modules=["pkg"],
        opportunities={"code_verification": True, "task_planning": False},
        gold_changed_lines=changed or {},
    )


def _edit(i, old, new, f="pkg/a.py"):
    return NormalizedStep(
        step_id=i,
        turn_id=i,
        actor="agent",
        type="tool_call",
        canonical_action=EDIT,
        files=[f],
        args={"old_lines": old, "new_lines": new},
    )


class TestContentCapture(unittest.TestCase):
    def test_claude_code_edit_content(self):
        raw = {
            "trajectory": [
                {
                    "index": 0,
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "Edit",
                            "id": "c1",
                            "input": {"file_path": "/w/pkg/a.py", "old_string": "x = 1\n", "new_string": "x = 2\n"},
                        }
                    ],
                }
            ]
        }
        traj = adapt_claude_code(raw, "a", "i")
        step = traj.steps[0]
        self.assertEqual(step.args.get("old_lines"), ["0|x = 1"])
        self.assertEqual(step.args.get("new_lines"), ["0|x = 2"])

    def test_opencode_apply_patch_tool(self):
        body = "*** Begin Patch\n*** Update File: pkg/a.py\n@@\n-x = 1\n+x = 2\n*** End Patch"
        raw = {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "apply_patch",
                            "callID": "c1",
                            "state": {"status": "completed", "input": {"patchText": body}, "output": "ok"},
                        }
                    ],
                }
            ]
        }
        traj = adapt_opencode(raw, "a", "i")
        call = traj.steps[0]
        self.assertEqual(call.canonical_action, EDIT)
        self.assertEqual(call.files, ["pkg/a.py"])
        self.assertEqual(call.args.get("new_lines"), ["0|x = 2"])

    def test_codex_custom_tool_call_input_field(self):
        body = "*** Begin Patch\n*** Update File: pkg/a.py\n@@\n-x = 1\n+x = 2\n*** End Patch"
        lines = [
            {
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "apply_patch", "input": body, "call_id": "c1"},
            }
        ]
        traj = adapt_codex(lines, "a", "i")
        call = traj.steps[0]
        self.assertEqual(call.canonical_action, EDIT)
        self.assertEqual(call.files, ["pkg/a.py"])
        self.assertEqual(call.args.get("new_lines"), ["0|x = 2"])

    def test_apply_patch_content_skips_markers(self):
        old, new = _apply_patch_content("*** Update File: x\n@@\n-a\n+b\n--- junk\n+++ junk")
        self.assertEqual(old, "a")
        self.assertEqual(new, "b")


class TestCodexCwd(unittest.TestCase):
    def test_relative_cd_within_one_command(self):
        b = _Builder("codex", "i")
        self.assertEqual(_command_cwd("cd sklearn && cat metrics/foo.py", b), "sklearn")
        self.assertEqual(_command_cwd("cd sklearn && cd ..", b), "")

    def test_codex_read_set_resolves_through_cwd(self):
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cd pkg && cat sub/a.py"}',
                    "call_id": "c1",
                },
            },
        ]
        traj = adapt_codex(lines, "codex", "i")
        self.assertEqual(traj.steps[0].files, ["pkg/sub/a.py"])


class TestRevertedOwnChange(unittest.TestCase):
    def _traj(self, revert):
        steps = [
            NormalizedStep(
                step_id=0,
                turn_id=0,
                actor="agent",
                type="tool_call",
                canonical_action="run_test",
                test_run=True,
                call_id="t1",
                args={"command": "pytest tests/test_a.py"},
            ),
            NormalizedStep(
                step_id=1, turn_id=1, actor="tool", type="tool_result", output_excerpt="1 failed", call_id="t1"
            ),
            _edit(2, ["0|x = 1"], ["0|x = 2"]),
        ]
        if revert:
            steps.append(_edit(3, ["0|x = 2"], ["0|x = 1"]))  # exact undo
        else:
            steps.append(_edit(3, ["0|x = 2"], ["0|x = 3"]))  # forward progress
        for step in steps:
            if step.canonical_action == EDIT:
                step.call_id = f"e{step.step_id}"
                step.args.update(
                    old_content="\n".join(x.split("|", 1)[-1] for x in step.args["old_lines"]),
                    new_content="\n".join(x.split("|", 1)[-1] for x in step.args["new_lines"]),
                )
        observed = []
        for step in steps:
            step.step_id = len(observed)
            observed.append(step)
            if step.canonical_action == EDIT:
                observed.append(
                    NormalizedStep(
                        step_id=len(observed),
                        turn_id=step.turn_id,
                        actor="tool",
                        type="tool_result",
                        call_id=step.call_id,
                        output_excerpt="Edited successfully",
                    )
                )
        steps = observed
        steps.append(NormalizedStep(step_id=len(steps), turn_id=4, actor="agent", type="message", text="hm"))
        return NormalizedTrajectory("a", "i", steps)

    def test_signal_and_candidate(self):
        sig = compute(self._traj(True), _gold())
        self.assertTrue(sig.edit_reverted_own_change)
        ets = {
            c.error_type
            for c in objective_candidates(
                _gold(),
                {
                    "flag": "right_place_wrong_change",
                    "hit": ["pkg/a.py"],
                    "missed_unreached": [],
                    "missed_reached": [],
                    "extra_files": [],
                    "is_cross_module": False,
                    "hit_overlap": None,
                    "hit_region": None,
                    "recall": 1.0,
                },
                sig,
                False,
            )
        }
        self.assertIn("reverted_own_change", ets)

    def test_forward_progress_is_not_a_revert(self):
        self.assertFalse(compute(self._traj(False), _gold()).edit_reverted_own_change)

    def test_audit_re_derives_and_tampers(self):
        fault = {
            "capability": "self_repair_loop",
            "error_type": "reverted_own_change",
            "confidence": "medium",
            "detector_evidence": {"edit_reverted_own_change": True},
        }
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(True))
        self.assertEqual(a["verdict"], RE_DERIVED)
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=self._traj(False))
        self.assertEqual(a["verdict"], MISMATCH)


class TestRejectedWalkedBack(unittest.TestCase):
    def test_traj_matched_gold_lines_computed_but_not_an_error_type(self):
        g = _gold(changed={"pkg/a.py": {"added": {"0|x = 2": 1}, "removed": {}}})
        traj = NormalizedTrajectory("a", "i", [_edit(0, ["0|x = 1"], ["0|x = 2"])])
        sig = compute(traj, g)
        self.assertEqual(sig.traj_matched_gold_lines, {"pkg/a.py": 1})
        self.assertNotIn("reverted_correct_fix", produced_error_types())


class TestUnattributedVisibility(unittest.TestCase):
    def test_slug_injective(self):
        self.assertNotEqual(awe_config.model_slug("org/model:free"), awe_config.model_slug("org/model_free"))


if __name__ == "__main__":
    unittest.main()


class TestReviewFindings(unittest.TestCase):
    """Pins the CONFIRMED findings from the pre-push workflow review."""

    def test_cd_to_repo_root_yields_clean_paths(self):
        # finding 1: '/x' sentinel leaked a phantom 'x/' prefix after cd <root>
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cd /w/workdirs/i/codex && cat pkg/a.py"}',
                    "call_id": "c1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cat setup.py"}',
                    "call_id": "c2",
                },
            },
        ]
        traj = adapt_codex(lines, "codex", "i")
        self.assertEqual(traj.steps[0].files, ["pkg/a.py"])
        self.assertEqual(traj.steps[1].files, ["setup.py"])

    def test_cwd_does_not_persist_across_commands(self):
        # finding 7: each exec runs in a fresh shell — cd must be per-command
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cd pkg && ls"}',
                    "call_id": "c1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cat sub/a.py"}',
                    "call_id": "c2",
                },
            },
        ]
        traj = adapt_codex(lines, "codex", "i")
        self.assertEqual(traj.steps[1].files, ["sub/a.py"])  # not pkg/sub/a.py

    def test_absolute_path_not_double_prefixed(self):
        # finding 3
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cd pkg && cat /w/workdirs/i/codex/pkg/a.py"}',
                    "call_id": "c1",
                },
            }
        ]
        traj = adapt_codex(lines, "codex", "i")
        self.assertEqual(traj.steps[0].files, ["pkg/a.py"])

    def test_cd_flags_and_dash_do_not_corrupt_cwd(self):
        # finding 6
        b = _Builder("codex", "i")
        from agent_trajviz.awe._engine.adapters import _command_cwd

        self.assertEqual(_command_cwd("cd -P pkg && cat a.py", b), "pkg")
        self.assertEqual(_command_cwd("cd - && cat a.py", b), "")
        self.assertEqual(_command_cwd('cd "my dir" && cat a.py', b), "my dir")

    def test_multifile_patch_per_file_content(self):
        # finding 8: union content must not be attributed to every file
        body = (
            "*** Begin Patch\n*** Update File: pkg/a.py\n@@\n-a1\n+a2\n"
            "*** Update File: pkg/b.py\n@@\n-b1\n+b2\n*** End Patch"
        )
        lines = [
            {
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "apply_patch", "input": body, "call_id": "c1"},
            }
        ]
        traj = adapt_codex(lines, "codex", "i")
        per = traj.steps[0].args.get("per_file_lines")
        self.assertEqual(per["pkg/a.py"]["new_lines"], ["0|a2"])
        self.assertEqual(per["pkg/b.py"]["new_lines"], ["0|b2"])
        self.assertNotIn("0|b2", per["pkg/a.py"]["new_lines"])

    def test_audit_revert_requires_failure_event(self):
        # finding 5: a revert on a failure-free run must MISMATCH, not re-derive
        steps = [_edit(0, ["0|x = 1"], ["0|x = 2"]), _edit(1, ["0|x = 2"], ["0|x = 1"])]
        traj = NormalizedTrajectory("a", "i", steps)
        fault = {
            "capability": "self_repair_loop",
            "error_type": "reverted_own_change",
            "confidence": "medium",
            "detector_evidence": {"edit_reverted_own_change": True},
        }
        a = audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=traj)
        self.assertEqual(a["verdict"], MISMATCH)

    def test_unattributed_record_renders(self):
        # finding 2: the record must carry every key to_markdown reads
        from agent_trajviz.awe._engine.dossier import to_markdown, blame_status_note

        rec = {
            "case_id": "a__i",
            "agent": "a",
            "instance_id": "i",
            "repo": "r",
            "blame_status": "unattributed",
            "blame_status_note": blame_status_note("unattributed"),
            "task": {"problem_statement": "p", "gold_files": ["pkg/a.py"], "multi_file": False, "cross_module": False},
            "reasoning_observability": "n/a (no fault diagnosed)",
            "thinking_process": [],
            "faults": [],
        }
        md = to_markdown([rec])
        self.assertIn("unattributed", md)
