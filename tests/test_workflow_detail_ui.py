import inspect
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from i18n_test_helpers import unmark_html

class WorkflowDetailUiTests(unittest.TestCase):
    @staticmethod
    def _metric_step(**overrides):
        step = {
            "index": 1,
            "role": "assistant",
            "parts": [],
            "tokens": {
                "total": 100,
                "input": 100,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            "duration": 2.0,
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": False,
        }
        step.update(overrides)
        return step

    def test_selecting_workflow_card_resets_detail_panel_scroll(self):
        from agent_trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)
        self.assertIn("detailPanel.scrollTop = 0", source)

    def test_duration_chart_click_jumps_to_workflow(self):
        from agent_trajviz.insight import insight as insight_mod
        from agent_trajviz.insight.ui import overview_tab

        source = inspect.getsource(insight_mod.build_ui)
        self.assertIn("tvBindChartWorkflowJumps", source)
        self.assertIn("plotly_click", source)
        self.assertIn("tvGotoWorkflowStep", source)
        self.assertIn("tvFocusWorkflowCard", source)
        self.assertIn("wf-flash", source)
        self.assertIn("popstate", source)
        self.assertIn("tvTab", source)
        self.assertIn("tvPushTabReturnPoint", source)
        self.assertIn("tvSelectWorkflowCard", source)
        self.assertIn("restore", source)
        self.assertIn("tvWorkflowStepUrl", source)
        self.assertNotIn("__tvJumpBound", source)
        self.assertNotIn("tvBindDurationJump", source)
        self.assertIn('elem_id="duration-chart"', inspect.getsource(overview_tab.layout))
        self.assertIn('elem_id="tool-outcome-chart"', inspect.getsource(overview_tab.layout))
        self.assertIn('elem_id="tool-duration-chart"', inspect.getsource(overview_tab.layout))
        self.assertIn('elem_id="agent-swimlane-chart"', inspect.getsource(overview_tab.layout))
        self.assertIn("tool-outcome-chart", source)
        self.assertIn("tool-duration-chart", source)
        self.assertIn("agent-swimlane-chart", source)

    def test_workflow_jump_flash_style_exists(self):
        styles = Path(__file__).resolve().parents[1].joinpath("agent_trajviz", "insight", "styles.py").read_text()
        self.assertIn(".wf-card.wf-flash", styles)
        self.assertIn("@keyframes wf-flash-pulse", styles)

    def test_detail_tabs_stay_visible_inside_scrollable_detail_panel(self):
        styles = Path(__file__).resolve().parents[1].joinpath("agent_trajviz", "insight", "styles.py").read_text()

        self.assertIn("position: sticky", styles[styles.index(".dp-tabs"):styles.index(".dp-tab {")])
        self.assertIn("top: 0", styles[styles.index(".dp-tabs"):styles.index(".dp-tab {")])

    def test_detail_tabs_do_not_use_fragile_inline_click_handler(self):
        from agent_trajviz.insight.rendering import format_step_detail

        html = format_step_detail({
            "index": 1,
            "role": "assistant",
            "parts": [{"type": "text", "content": "hello"}],
            "tokens": {
                "total": 0,
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": False,
        })

        self.assertIn("class='dp-tabs'", html)
        self.assertNotIn("onclick=", html[html.index("class='dp-tabs'"):html.index("data-tab-content='content'")])

    def test_detail_panel_reasoning_collapses_with_token_label(self):
        from agent_trajviz.insight.rendering import format_step_detail

        html = format_step_detail({
            "index": 1,
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "text": "long chain of thought"},
                {"type": "text", "text": "the answer"},
            ],
            "tokens": {
                "total": 4100,
                "input": 0,
                "output": 100,
                "reasoning": 4000,
                "cache_read": 0,
                "cache_write": 0,
            },
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": True,
        })

        # Reasoning collapses behind a summary labelled with the real metric.
        self.assertIn("<summary>Reasoning (4,000 tok)</summary>", html)
        details_start = html.index("Reasoning (4,000 tok)")
        block = html[html.rindex("<details", 0, details_start):]
        self.assertNotIn("<details open", block[: block.index(">") + 1])
        self.assertIn("long chain of thought", html)
        # Text stays an open section, not collapsed.
        text_start = html.index("the answer")
        self.assertNotIn(
            "<details", html[html.rindex("<div", 0, text_start):text_start]
        )

    def test_detail_panel_drops_text_label_when_part_is_empty(self):
        from agent_trajviz.insight.rendering import format_step_detail

        html = format_step_detail({
            "index": 1,
            "role": "assistant",
            "parts": [
                {"type": "text", "text": ""},  # 纯工具调用步常见：空 text part
                {"type": "text", "text": "the answer"},
            ],
            "tokens": {
                "total": 100,
                "input": 0,
                "output": 100,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": False,
        })

        # 空 text part 不再渲染孤零零的 "Text" 标签。
        self.assertIn("the answer", html)
        self.assertEqual(1, html.count(">Text<"))

    def test_workflow_registers_delegated_detail_tab_handler(self):
        from agent_trajviz.insight.ui import workflow_tab

        source = inspect.getsource(workflow_tab)

        self.assertIn("__dpTabHandlerAttached", source)
        self.assertIn("document.addEventListener('click', function(e) {", source)
        self.assertIn("e.target.closest('.dp-tab')", source)

    def test_missing_reasoning_keeps_table_with_na_row(self):
        from agent_trajviz.insight.rendering import _format_metrics_tab

        tokens = dict(self._metric_step()["tokens"])
        del tokens["reasoning"]
        html = unmark_html(_format_metrics_tab(self._metric_step(tokens=tokens)))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertNotIn("Metrics unavailable", html)
        self.assertIn("<td>Reasoning Tokens</td><td>n/a</td>", html)
        self.assertIn("<td>Total Tokens</td><td>100</td>", html)

    def test_complete_metrics_render_the_whole_table_and_keep_real_zeroes(self):
        from agent_trajviz.insight.rendering import _format_metrics_tab

        html = unmark_html(_format_metrics_tab(self._metric_step()))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertIn("<td>Output Tokens</td><td>0</td>", html)
        self.assertIn("<td>Reasoning Tokens</td><td>0</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>0.0%</td>", html)
        self.assertNotIn("Metrics unavailable", html)

    def test_one_missing_metric_replaces_the_entire_table_with_a_message(self):
        from agent_trajviz.insight.rendering import _format_metrics_tab

        tokens = dict(self._metric_step()["tokens"])
        del tokens["cache_write"]
        html = unmark_html(_format_metrics_tab(self._metric_step(tokens=tokens)))

        self.assertNotIn("<table", html)
        self.assertIn("Metrics unavailable", html)
        self.assertIn("Missing or unavailable: Cache Write", html)

    def test_missing_duration_keeps_token_table_with_na_derived_rows(self):
        # The final step of a Claude Code trajectory has genuine token data
        # but no recorded duration; the table must not be hidden for it.
        from agent_trajviz.insight.rendering import _format_metrics_tab

        html = unmark_html(_format_metrics_tab(self._metric_step(duration=None)))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertNotIn("Metrics unavailable", html)
        self.assertIn("<td>Total Tokens</td><td>100</td>", html)
        self.assertIn("<td>Duration</td><td>n/a</td>", html)
        self.assertIn("<td>Throughput</td><td>n/a</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>0.0%</td>", html)

    def test_real_zero_duration_renders_as_zero_not_missing(self):
        from agent_trajviz.insight.rendering import _format_metrics_tab

        html = unmark_html(_format_metrics_tab(self._metric_step(duration=0.0)))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertNotIn("Metrics unavailable", html)
        self.assertIn("<td>Duration</td><td>0.0s</td>", html)
        self.assertIn("<td>Throughput</td><td>n/a</td>", html)

    def test_zero_total_tokens_shows_na_derived_rows_without_dividing(self):
        from agent_trajviz.insight.rendering import _format_metrics_tab

        tokens = {
            "total": 0, "input": 0, "output": 0,
            "reasoning": 0, "cache_read": 0, "cache_write": 0,
        }
        html = unmark_html(_format_metrics_tab(self._metric_step(tokens=tokens)))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertIn("<td>Total Tokens</td><td>0</td>", html)
        # 0 tokens over a real duration is a genuine 0 tok/s, but 0/0 cache
        # ratio is undefined and must degrade to n/a instead of dividing.
        self.assertIn("<td>Throughput</td><td>0 tok/s</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>n/a</td>", html)

    def test_subagent_task_prompt_is_not_badged_as_user(self):
        from agent_trajviz.insight.rendering import (
            format_step_detail,
            render_toc_sidebar,
            render_workflow_html,
            workflow_role,
            workflow_role_label,
        )

        human = self._metric_step(
            index=0,
            role="user",
            is_sub_agent=False,
            parts=[{"type": "text", "text": "please explore"}],
            text_preview="please explore",
        )
        spawn = self._metric_step(
            index=1,
            role="user",
            is_sub_agent=True,
            agent="explore",
            session_id="ses_child",
            parts=[{"type": "text", "text": "Explore the repo"}],
            text_preview="Explore the repo",
        )
        compact = self._metric_step(
            index=2,
            role="user",
            is_sub_agent=True,
            agent="explore",
            parts=[{"type": "compaction", "summary": "prior work"}],
            text_preview="",
        )
        synthetic = self._metric_step(
            index=3,
            role="user",
            is_sub_agent=True,
            agent="explore",
            parts=[{
                "type": "text",
                "text": "Continue if you have next steps",
                "synthetic": True,
            }],
            text_preview="Continue if you have next steps",
        )

        self.assertEqual(workflow_role(human), "user")
        self.assertEqual(workflow_role(spawn), "task")
        self.assertEqual(workflow_role(compact), "compaction")
        self.assertEqual(workflow_role(synthetic), "system")
        self.assertEqual(workflow_role_label(spawn), "Task")

        cards = render_workflow_html([human, spawn, compact, synthetic])
        self.assertIn(">User</span>", cards)
        self.assertIn(">Task</span>", cards)
        self.assertIn(">Compaction</span>", cards)
        self.assertIn(">System</span>", cards)
        spawn_card = cards[cards.index('id="wf-card-1"'):cards.index('id="wf-card-2"')]
        self.assertNotIn(">User</span>", spawn_card)
        self.assertIn(">Task</span>", spawn_card)

        toc = render_toc_sidebar([spawn])
        self.assertIn(">Task</span>", toc)
        self.assertNotIn(">User</span>", toc)

        detail = unmark_html(format_step_detail(spawn))
        self.assertIn(">Task</span>", detail)
        self.assertIn("<td>Role</td>", detail)
        self.assertIn("Task", detail)
        self.assertNotIn(">User</span>", detail)

    def test_system_error_cards_use_amber_not_red(self):
        from agent_trajviz.insight.rendering import _card_style, render_workflow_html

        system = self._metric_step(
            index=5,
            tool_calls=[{
                "tool_name": "Grep",
                "status": "error",
                "metadata": {},
            }],
            error_count=1,
            text_preview="pattern failed",
        )
        tool = self._metric_step(
            index=6,
            tool_calls=[{
                "tool_name": "Bash",
                "status": "error",
                "metadata": {},
            }],
            error_count=1,
            text_preview="script failed",
        )
        bg_s, border_s, label_s, kind_s = _card_style(system)
        bg_t, border_t, label_t, kind_t = _card_style(tool)
        self.assertEqual(label_s, "System Error")
        self.assertEqual(kind_s, "system")
        self.assertEqual(border_s, "var(--wf-border-system-error)")
        self.assertEqual(bg_s, "var(--wf-bg-system-error)")
        self.assertEqual(label_t, "Tool Error")
        self.assertEqual(kind_t, "tool")
        self.assertEqual(border_t, "var(--wf-border-error)")

        html = render_workflow_html([system, tool])
        self.assertIn("System Error", html)
        self.assertIn("Tool Error", html)
        self.assertIn("var(--wf-border-system-error)", html)
        self.assertIn("var(--wf-bg-system-error)", html)


def _bind_jumps_source():
    from agent_trajviz.insight import insight as insight_mod

    source = inspect.getsource(insight_mod.build_ui)
    begin = source.index("__TV_BIND_JUMPS_BEGIN__")
    end = source.index("__TV_BIND_JUMPS_END__")
    block = source[begin:end]
    block = block[block.index("window.tvBindChartWorkflowJumps"):]
    return block[:block.rindex("};") + 2]


@unittest.skipUnless(shutil.which("node"), "requires Node.js to execute the jump binder")
class ChartWorkflowJumpBinderTests(unittest.TestCase):
    """Run the embedded Plotly rebind helper against a fake graph div."""

    def test_rebind_after_plotly_wipes_listeners_and_reads_step_index(self):
        script = r"""
        var jumped = [];
        var window = {
            tvGotoWorkflowStep: function (idx) { jumped.push(idx); }
        };
        function makePlot() {
            var listeners = [];
            var gd = {
                style: {},
                on: function (ev, fn) { if (ev === 'plotly_click') listeners.push(fn); },
                removeListener: function (ev, fn) {
                    if (ev !== 'plotly_click') return;
                    listeners = listeners.filter(function (x) { return x !== fn; });
                },
                querySelector: function () { return null; },
                wipe: function () { listeners = []; },
                fire: function (data) { listeners.slice().forEach(function (fn) { fn(data); }); },
                count: function () { return listeners.length; }
            };
            return gd;
        }
        var plot = makePlot();
        var roots = {
            'tool-duration-chart': {
                querySelector: function () { return plot; }
            }
        };
        var document = {
            getElementById: function (id) { return roots[id] || null; }
        };
        """ + _bind_jumps_source() + r"""
        window.tvBindChartWorkflowJumps();
        window.tvBindChartWorkflowJumps();
        if (plot.count() !== 1) throw new Error('expected 1 listener after rebind, got ' + plot.count());
        plot.fire({ points: [{ customdata: [7] }] });
        plot.wipe();
        if (plot.count() !== 0) throw new Error('wipe failed');
        window.tvBindChartWorkflowJumps();
        plot.fire({ points: [{ customdata: 9 }] });
        console.log(JSON.stringify({ jumped: jumped, listeners: plot.count() }));
        """
        proc = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            raise AssertionError(f"node failed: {proc.stderr}\n{proc.stdout}")
        result = json.loads(proc.stdout)
        self.assertEqual(result["jumped"], [7, 9])
        self.assertEqual(result["listeners"], 1)


if __name__ == "__main__":
    unittest.main()


class ReasoningDetailContentTests(unittest.TestCase):
    @staticmethod
    def render(parts, has_reasoning):
        from agent_trajviz.insight.rendering import format_step_detail
        return unmark_html(format_step_detail(WorkflowDetailUiTests._metric_step(parts=parts, has_reasoning=has_reasoning)))

    def test_reasoning_record_without_text_has_an_honest_note(self):
        out = self.render([{"type": "reasoning", "text": ""}, {"type": "text", "text": "Final response."}], True)
        self.assertIn("Reasoning text unavailable", out)
        self.assertIn("No readable reasoning text was included in this export.", out)
        self.assertIn("Final response.", out)
        self.assertNotIn("<div class='dp-section-title'>Reasoning</div>", out)

    def test_readable_reasoning_is_preserved_while_empty_chunks_are_omitted(self):
        out = self.render([{"type": "reasoning", "text": "   \n"},
                           {"type": "reasoning", "text": "A recorded reasoning summary."},
                           {"type": "text", "text": "Final response."}], True)
        self.assertIn("A recorded reasoning summary.", out)
        # Readable reasoning is collapsed into a details block (not a titled section card).
        self.assertEqual(out.count("<details class='dp-details'>"), 1)
        self.assertIn("<summary>Reasoning (", out)
        self.assertNotIn("<div class='dp-section-title'>Reasoning</div>", out)
        self.assertNotIn("Reasoning text unavailable", out)

    def test_truly_blank_content_does_not_leave_empty_section_cards(self):
        out = self.render([{"type": "reasoning", "text": "  "}, {"type": "text", "text": "\n"}], False)
        self.assertNotIn("dp-section-reasoning", out)
        self.assertNotIn("dp-section-text", out)
        self.assertIn("<em>No content</em>", out)

    def test_reasoning_flag_without_a_readable_part_is_not_silently_lost(self):
        out = self.render([{"type": "text", "text": "Final response."}], True)
        self.assertIn("Reasoning text unavailable", out)
        self.assertIn("Final response.", out)

    def test_several_empty_reasoning_parts_produce_only_one_notice(self):
        out = self.render([{"type": "reasoning", "text": ""}, {"type": "reasoning", "text": None}], True)
        self.assertEqual(out.count("No readable reasoning text was included in this export."), 1)
