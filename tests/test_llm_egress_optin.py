"""Model requests require an explicit action in the dashboard.

TrajViz is offline-by-default analytics over traces that routinely contain
source code, absolute local paths, shell commands and customer data. These
tests inspect the dashboard's actual Gradio callbacks, including those chained
to trajectory loads.
"""

import unittest

import gradio as gr

from agent_trajviz.insight.insight import build_ui


def _deps_for(demo, fn_name):
    """Dependencies in the built UI whose handler is *fn_name*."""
    out = []
    for bf in demo.fns.values():
        fn = getattr(bf, "fn", None)
        if getattr(fn, "__name__", None) == fn_name:
            out.append(bf)
    return out


class IssuesJudgeOptInTests(unittest.TestCase):
    """Loading a trajectory leaves the Overview judge idle."""

    @classmethod
    def setUpClass(cls):
        cls.demo = build_ui()

    def test_judge_is_only_bound_to_an_explicit_click(self):
        deps = _deps_for(self.demo, "on_suggest_fixes")
        events = [event for bf in deps for _block_id, event in bf.targets]
        self.assertEqual(events, ["click"])

    def test_judge_button_names_the_egress(self):
        blocks = self.demo.blocks
        click_triggers = []
        for bf in _deps_for(self.demo, "on_suggest_fixes"):
            for block_id, event in bf.targets:
                if event == "click":
                    click_triggers.append(blocks[block_id])
        self.assertTrue(click_triggers, "expected a click binding for re-run")
        for trigger in click_triggers:
            self.assertIsInstance(trigger, gr.Button)
            label = (trigger.value or "").lower()
            self.assertIn("llm", label)
            self.assertIn("sends", label)


class SidebarAnalysisOptInTests(unittest.TestCase):
    """Loading a trajectory does not call out, even while Settings is open."""

    def test_load_with_closed_sidebar_does_not_call_the_provider(self):
        from agent_trajviz.insight import assistant
        from agent_trajviz.insight.ui import sidebar

        calls = []

        def _boom(*a, **k):
            calls.append(1)
            raise AssertionError("provider must not be called on load")

        original = assistant.analyze_loaded_trajectory
        sidebar.analyze_loaded_trajectory = _boom
        try:
            demo = build_ui()
            handlers = _deps_for(demo, "on_trajectory_loaded")
            self.assertTrue(handlers, "sidebar load handler not found")
            steps = [{"index": 0, "role": "user", "parts": [], "tool_calls": []}]
            for bf in handlers:
                bf.fn(steps, "")
        finally:
            sidebar.analyze_loaded_trajectory = original
            assistant.analyze_loaded_trajectory = original
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
