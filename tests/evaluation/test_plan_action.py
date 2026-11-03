"""Schema 20 (DESIGN §0ay): the scaffolds' plan tools become PLAN steps that
keep their items and statuses; trajsignals carries the plan_* cohort
features; the judge's evidence view and the dossier's thinking excerpt show
the plan. No blame path — these are behavioral features pending calibration.

Run:  python3 -m unittest tests.test_plan_action -v
"""

from __future__ import annotations

import json
import unittest


from agent_trajviz.awe._engine.adapters import adapt_claude_code, adapt_opencode, adapt_codex
from agent_trajviz.awe._engine.cohort import features_of
from agent_trajviz.awe._engine.trajectory import (
    NormalizedStep,
    NormalizedTrajectory,
    PLAN,
    EDIT,
    plan_items,
    SCHEMA_VERSION,
)
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine.gold import GoldReference


def _gold():
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
        opportunities={"code_verification": True, "task_planning": True},
        gold_changed_lines={},
    )


TODOS = [
    {"content": "Read the code", "status": "completed", "activeForm": "Reading"},
    {"content": "Patch it", "status": "in_progress", "activeForm": "Patching"},
    {"content": "Run tests", "status": "pending", "activeForm": "Running"},
]


class TestNormalizer(unittest.TestCase):
    def test_schema_bumped(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 20)

    def test_plan_items_forms(self):
        self.assertEqual(plan_items(TODOS)[1], {"content": "Patch it", "status": "in_progress"})
        self.assertEqual(
            plan_items([{"step": "Do X", "status": "Completed"}]), [{"content": "Do X", "status": "completed"}]
        )
        self.assertEqual(plan_items(["bare"]), [{"content": "bare", "status": "unknown"}])
        self.assertEqual(plan_items(None), [])


class TestAdapters(unittest.TestCase):
    def _plan_steps(self, traj):
        return [s for s in traj.steps if s.canonical_action == PLAN]

    def test_claude_code_todowrite(self):
        raw = {
            "trajectory": [
                {
                    "role": "assistant",
                    "index": 0,
                    "content": [{"type": "tool_call", "id": "c1", "name": "TodoWrite", "input": {"todos": TODOS}}],
                }
            ]
        }
        p = self._plan_steps(adapt_claude_code(raw, "claude_code", "i"))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].native_tool, "TodoWrite")
        self.assertEqual([it["status"] for it in p[0].args["items"]], ["completed", "in_progress", "pending"])

    def test_opencode_todowrite(self):
        raw = {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "todowrite",
                            "callID": "k1",
                            "state": {
                                "status": "completed",
                                "input": {"todos": TODOS[:2]},
                                "output": "ok",
                                "metadata": {},
                            },
                        }
                    ],
                }
            ]
        }
        p = self._plan_steps(adapt_opencode(raw, "opencode_opus", "i"))
        self.assertEqual(len(p), 1)
        self.assertEqual(len(p[0].args["items"]), 2)

    def test_codex_update_plan(self):
        args = {
            "explanation": "root cause found",
            "plan": [{"step": "Add test", "status": "completed"}, {"step": "Patch", "status": "in_progress"}],
        }
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "update_plan",
                    "call_id": "c1",
                    "arguments": json.dumps(args),
                },
            }
        ]
        p = self._plan_steps(adapt_codex(lines, "codex", "i"))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].args["explanation"], "root cause found")
        self.assertEqual(p[0].args["items"][1], {"content": "Patch", "status": "in_progress"})


def _step(i, action, **kw):
    base = dict(step_id=i, turn_id=i, actor="agent", type="tool_call", canonical_action=action, call_id=f"c{i}")
    base.update(kw)
    return NormalizedStep(**base)


class TestSignals(unittest.TestCase):
    def test_plan_fields(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(0, PLAN, native_tool="TodoWrite", args={"items": plan_items(TODOS)}),
                _step(1, EDIT, files=["pkg/a.py"], args={"path": "pkg/a.py"}),
                _step(
                    2,
                    PLAN,
                    native_tool="TodoWrite",
                    args={
                        "items": plan_items(
                            [
                                {"content": "Patch it", "status": "completed"},
                                {"content": "Run tests", "status": "pending"},
                            ]
                        )
                    },
                ),
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.n_plan_updates, 2)
        self.assertTrue(sig.plan_declared)
        self.assertTrue(sig.plan_before_first_edit)
        self.assertEqual(sig.plan_items_total, 2)
        self.assertEqual(sig.plan_items_open_at_end, 1)
        f = features_of(sig)
        self.assertEqual(f["plan_declared"], 1.0)
        self.assertEqual(f["plan_left_open"], 1.0)

    def test_plan_after_edit_and_subagent_plans_ignored(self):
        traj = NormalizedTrajectory(
            "a",
            "i",
            [
                _step(0, EDIT, files=["pkg/a.py"], args={"path": "pkg/a.py"}),
                _step(1, PLAN, args={"items": plan_items(TODOS)}, subagent_id="sub-1"),
                _step(2, PLAN, args={"items": plan_items(TODOS[:1])}),
            ],
        )
        sig = compute(traj, _gold())
        self.assertEqual(sig.n_plan_updates, 1)  # the sub-agent plan is not the run's
        self.assertFalse(sig.plan_before_first_edit)
        self.assertEqual(sig.plan_items_open_at_end, 0)

    def test_no_plan(self):
        traj = NormalizedTrajectory("a", "i", [_step(0, EDIT, files=["pkg/a.py"], args={})])
        sig = compute(traj, _gold())
        self.assertFalse(sig.plan_declared)
        self.assertEqual(features_of(sig)["plan_declared"], 0.0)


class TestEvidenceViews(unittest.TestCase):
    def test_judge_and_dossier_show_the_plan(self):
        from agent_trajviz.awe._engine.judge import build_evidence
        from agent_trajviz.awe._engine.dossier import thinking_excerpt

        traj = NormalizedTrajectory(
            "a", "i", [_step(0, PLAN, native_tool="TodoWrite", args={"items": plan_items(TODOS)})]
        )
        rows = build_evidence(_gold(), traj)["rows"]
        self.assertEqual(rows[0]["kind"], "plans")
        self.assertIn("[in_progress] Patch it", rows[0]["text"])
        ex = thinking_excerpt(traj)
        self.assertEqual(ex[0]["kind"], "plans")
        self.assertIn("Run tests", ex[0]["text"])


if __name__ == "__main__":
    unittest.main()
