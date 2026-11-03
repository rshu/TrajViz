import unittest
from agent_trajviz.awe._engine.adapters import adapt_codex
from agent_trajviz.awe._engine.evidenceview import build_view, render_rows
from agent_trajviz.awe._engine.verification import verification_observations
from agent_trajviz.awe._engine.trajectory import read_observations
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine.audit import _failure_event_steps, _redrv_retry_without_change
from .test_exit_codes import _call, _out, _hdr, _codex_lines
from .test_new_error_types_batch6 import _gold


class AsyncEvidenceProvenance(unittest.TestCase):
    def _traj(self, between=(), complete=True, body="1 failed in 0.1s"):
        cmd = "pytest tests/test_a.py"
        items = [_call("t", "exec_command", {"cmd": cmd}), _out("t", _hdr(cmd, session=77, body="collecting\n"))]
        items.extend(between)
        if complete:
            items.extend(
                [_call("poll", "write_stdin", {"session_id": 77}), _out("poll", _hdr(cmd, exit_code=1, body=body))]
            )
        return adapt_codex(_codex_lines(*items), "codex", "i")

    def test_unfinished_session_is_unknown_even_with_summary(self):
        cmd = "pytest tests/test_a.py"
        traj = adapt_codex(
            _codex_lines(
                _call("t", "exec_command", {"cmd": cmd}), _out("t", _hdr(cmd, session=77, body="1 passed in 0.1s"))
            ),
            "codex",
            "i",
        )
        run = verification_observations(traj)["runs"][0]
        self.assertEqual(run["summary"], {"status": "unknown", "reason": "running_session"})
        self.assertIsNone(run["result"])
        self.assertEqual(_failure_event_steps(traj), [])

    def test_completion_and_sources_are_visible(self):
        traj = self._traj()
        run = verification_observations(traj)["runs"][0]
        self.assertEqual(run["result"]["step_id"], 3)
        self.assertEqual(run["output_aggregation"]["source_steps"], [1, 3])
        text = "\n".join(render_rows(build_view("task", traj)))
        self.assertIn("available at step 3", text)
        self.assertEqual(_failure_event_steps(traj), [3])

    def test_edit_before_late_failure_is_not_a_reaction(self):
        edit = {
            "type": "custom_tool_call",
            "name": "apply_patch",
            "call_id": "e",
            "input": "*** Begin Patch\n*** Update File: pkg/a.py\n@@\n-x=1\n+x=2\n*** End Patch",
        }
        traj = self._traj([edit])
        signal = compute(traj, _gold())
        self.assertTrue(signal.encountered_failure)
        self.assertFalse(signal.repair_not_reverified)
        self.assertEqual(signal.test_files_edited_after_failure, [])

    def test_retry_after_completion_is_still_observed(self):
        traj = self._traj()
        tail = adapt_codex(
            _codex_lines(
                _call("again", "exec_command", {"cmd": "pytest tests/test_a.py"}),
                _out("again", _hdr("pytest tests/test_a.py", exit_code=1, body="1 failed")),
            ),
            "codex",
            "i",
        )
        for step in tail.steps:
            step.step_id += len(traj.steps)
        traj.steps.extend(tail.steps)
        self.assertTrue(compute(traj, _gold()).retry_without_change)
        self.assertTrue(_redrv_retry_without_change(traj))

    def test_unknown_absolute_workdir_is_not_repo_root(self):
        traj = adapt_codex(
            _codex_lines(_call("r", "exec_command", {"cmd": "cat core.py", "workdir": "/external/project"})),
            "codex",
            "i",
        )
        self.assertEqual(traj.steps[0].files, ["/external/project/core.py"])
        self.assertEqual(read_observations(traj)[2], {"/external/project/core.py"})

    def test_completion_metadata_is_used_for_recorded_state(self):
        import json

        cmd = "go test ./..."
        final = json.dumps(
            {
                "output": "done",
                "metadata": {
                    "exit_code": 0,
                    "verification_state": {"revision": "b"},
                    "verification_tests": {"complete": True, "results": [{"test_id": "TestClamp", "status": "passed"}]},
                },
            }
        )
        traj = adapt_codex(
            _codex_lines(
                _call("t", "exec_command", {"cmd": cmd, "verification_state": {"revision": "a"}}),
                _out("t", _hdr(cmd, session=77)),
                _call("poll", "write_stdin", {"session_id": 77}),
                _out("poll", final),
            ),
            "codex",
            "i",
        )
        run = verification_observations(traj)["runs"][0]
        self.assertEqual(run["state"]["records"][-1]["fields"]["revision"], "b")
        self.assertEqual(run["summary"]["counts"]["executed"], 1)
        self.assertEqual(run["linkage"], "session_chunks")
