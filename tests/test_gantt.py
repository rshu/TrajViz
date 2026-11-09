"""Wall-clock execution Gantt: lanes, parallel overlap, and tool overlays.

The Gantt is the only chart whose x-axis is real time rather than step
index, so it is the only place parallel sub-agent execution is visible.
These tests assert the quantities it draws: relative-second offsets from
the earliest step, one lane per timeline agent, overlapping bars across
lanes when two agents ran concurrently, and tool windows overlaid on the
step body.
"""

from __future__ import annotations

import inspect
import unittest

from agent_trajviz.insight.charts import build_execution_gantt_chart
from agent_trajviz.insight.charts.gantt import (
    GANTT_AXIS_WALLCLOCK,
    STEP_BAR_COLOR,
    TOOL_BAR_COLOR,
    WAIT_BAND_COLOR,
)


def tc(
    name: str,
    *,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> dict:
    """One tool call with an optional wall-clock execution window."""
    call: dict = {"tool_name": name, "status": "completed"}
    if start_ms is not None:
        call["time_start"] = start_ms
    if end_ms is not None:
        call["time_end"] = end_ms
    return call


def step(
    index: int,
    *,
    role: str = "assistant",
    agent: str = "",
    session_id: str = "s-main",
    created_ms: float | None = None,
    completed_ms: float | None = None,
    duration: float | None = None,
    total: int = 0,
    tool_calls: list[dict] | None = None,
) -> dict:
    """A parsed step with timing fields, as ``parse_steps`` emits them."""
    s = {
        "index": index,
        "role": role,
        "agent": agent,
        "session_id": session_id,
        "tokens": {
            "total": total,
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 0,
        },
        "tool_calls": tool_calls or [],
        "tool_call_count": len(tool_calls or []),
        "duration": duration,
        "parts": [],
        "error_count": 0,
        "finish": "stop",
    }
    if created_ms is not None:
        s["time_created_ms"] = created_ms
    if completed_ms is not None:
        s["time_completed_ms"] = completed_ms
    return s


def _bars(fig) -> list[dict]:
    """Flatten the figure's bars to (lane, base, width) interval records."""
    out = []
    for tr in fig.data:
        if tr.type != "bar":
            continue
        hover = tr.hovertext
        hovers = [hover] * len(tr.y) if isinstance(hover, str) or hover is None else list(hover)
        for y, base, width, text in zip(tr.y, tr.base, tr.x, hovers, strict=True):
            out.append(
                {
                    "lane": y,
                    "start": float(base),
                    "end": float(base) + float(width),
                    "color": tr.marker.color,
                    "hovertext": text,
                }
            )
    return out


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


class ExecutionGanttTests(unittest.TestCase):
    def test_no_time_data_returns_none(self):
        steps = [
            step(0, duration=1.0),
            step(1, duration=2.0, tool_calls=[tc("Read")]),
        ]
        self.assertIsNone(build_execution_gantt_chart(steps))

    def test_empty_steps_returns_none(self):
        self.assertIsNone(build_execution_gantt_chart([]))

    def test_single_agent_sequential_steps_do_not_overlap(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000, total=100),
            step(1, created_ms=12_000, completed_ms=15_000, total=200),
        ]
        fig = build_execution_gantt_chart(steps)
        self.assertIsNotNone(fig)
        bars = _bars(fig)
        self.assertEqual(len(bars), 2)
        self.assertEqual({b["lane"] for b in bars}, {"main"})
        self.assertFalse(_overlaps(bars[0], bars[1]))
        # x values are seconds relative to the earliest created timestamp.
        self.assertAlmostEqual(bars[0]["start"], 0.0)
        self.assertAlmostEqual(bars[0]["end"], 2.0)
        self.assertAlmostEqual(bars[1]["start"], 2.0)
        self.assertAlmostEqual(bars[1]["end"], 5.0)
        self.assertEqual(fig.layout.xaxis.title.text, "Time (s) — relative to the first message")
        self.assertEqual(fig.layout.title.text, "Execution Timeline (wall-clock)")

    def test_parallel_agents_overlap_across_lanes(self):
        steps = [
            step(0, agent="build", session_id="s-main",
                 created_ms=10_000, completed_ms=20_000),
            step(1, agent="explore", session_id="s-sub",
                 created_ms=12_000, completed_ms=18_000),
        ]
        fig = build_execution_gantt_chart(steps)
        bars = _bars(fig)
        lanes = {b["lane"] for b in bars}
        self.assertEqual(lanes, {"build", "explore"})
        main_bar = next(b for b in bars if b["lane"] == "build")
        sub_bar = next(b for b in bars if b["lane"] == "explore")
        # The sub-agent ran inside the main agent's window: parallel work.
        self.assertTrue(_overlaps(main_bar, sub_bar))
        self.assertAlmostEqual(sub_bar["start"], 2.0)
        self.assertAlmostEqual(sub_bar["end"], 8.0)

    def test_sequential_agents_do_not_falsely_overlap(self):
        steps = [
            step(0, agent="build", session_id="s-main",
                 created_ms=10_000, completed_ms=12_000),
            step(1, agent="explore", session_id="s-sub",
                 created_ms=20_000, completed_ms=25_000),
        ]
        fig = build_execution_gantt_chart(steps)
        bars = _bars(fig)
        main_bar = next(b for b in bars if b["lane"] == "build")
        sub_bar = next(b for b in bars if b["lane"] == "explore")
        self.assertFalse(_overlaps(main_bar, sub_bar))

    def test_timed_tool_calls_are_overlaid_in_the_tool_color(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=20_000,
                 tool_calls=[tc("Read", start_ms=11_000, end_ms=13_000)]),
        ]
        fig = build_execution_gantt_chart(steps)
        bars = _bars(fig)
        step_bar = next(b for b in bars if b["color"] == STEP_BAR_COLOR)
        tool_bar = next(b for b in bars if b["color"] == TOOL_BAR_COLOR)
        self.assertAlmostEqual(step_bar["start"], 0.0)
        self.assertAlmostEqual(step_bar["end"], 10.0)
        # The tool window sits inside the step, in relative seconds.
        self.assertAlmostEqual(tool_bar["start"], 1.0)
        self.assertAlmostEqual(tool_bar["end"], 3.0)
        self.assertTrue(_overlaps(step_bar, tool_bar))
        self.assertEqual(step_bar["lane"], tool_bar["lane"])

    def test_untimed_tool_calls_draw_no_tool_bar(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=20_000,
                 tool_calls=[tc("Read")]),
        ]
        fig = build_execution_gantt_chart(steps)
        self.assertEqual(
            [b["color"] for b in _bars(fig)],
            [STEP_BAR_COLOR],
        )

    def test_hover_text_carries_the_step_number(self):
        steps = [
            step(7, created_ms=10_000, completed_ms=12_000, total=1234,
                 tool_calls=[tc("Read", start_ms=11_000, end_ms=11_500)]),
        ]
        fig = build_execution_gantt_chart(steps)
        hovers = [b["hovertext"] for b in _bars(fig)]
        self.assertEqual(len(hovers), 2)
        for hover in hovers:
            self.assertIn("Step 7", hover)
        step_hover = next(h for h in hovers if "tokens" in h)
        self.assertIn("1,234 tokens", step_hover)
        self.assertIn("Read", step_hover)

    def test_user_steps_do_not_occupy_a_lane(self):
        steps = [
            step(0, role="user", created_ms=10_000),
            step(1, created_ms=10_000, completed_ms=12_000),
        ]
        fig = build_execution_gantt_chart(steps)
        self.assertEqual({b["lane"] for b in _bars(fig)}, {"main"})

    def test_user_messages_become_markers_on_a_top_lane(self):
        # User prompts draw as green markers on a dedicated lane so idle
        # gaps between steps are explained (the human was away).
        steps = [
            step(0, role="user", created_ms=10_000),
            step(1, created_ms=10_000, completed_ms=12_000),
            step(2, role="user", created_ms=600_000),
            step(3, created_ms=601_000, completed_ms=602_000),
        ]
        for s in steps:
            if s["role"] == "user":
                s["text_preview"] = "用户消息"
        fig = build_execution_gantt_chart(steps)
        # Step bars live on agent lanes; the idle band gets its own lane.
        wait_bars = [b for b in _bars(fig) if b["lane"] == "user wait"]
        self.assertEqual(1, len(wait_bars))
        self.assertEqual({"main"}, {b["lane"] for b in _bars(fig)} - {"user wait"})
        scatters = [t for t in fig.data if t.type == "scatter"]
        self.assertEqual(1, len(scatters))
        sc = scatters[0]
        self.assertEqual({"user"}, set(sc.y))
        self.assertEqual(2, len(sc.x))
        self.assertIn("User message", sc.name)
        # Lane order bottom → top: main, user, user wait on top.
        self.assertEqual(["user", "user wait"], list(fig.layout.yaxis.categoryarray[-2:]))

    def test_missing_completion_falls_back_to_duration(self):
        steps = [step(0, created_ms=10_000, duration=3.0)]
        fig = build_execution_gantt_chart(steps)
        (bar,) = _bars(fig)
        self.assertAlmostEqual(bar["start"], 0.0)
        self.assertAlmostEqual(bar["end"], 3.0)

    def test_dark_mode_still_builds(self):
        steps = [step(0, created_ms=10_000, completed_ms=12_000)]
        fig = build_execution_gantt_chart(steps, dark=True)
        self.assertIsNotNone(fig)


class OverviewGanttWiringTests(unittest.TestCase):
    """The new component must appear at every OverviewRefs hook point."""

    def test_gantt_chart_is_wired_into_all_four_hook_points(self):
        from agent_trajviz.insight.ui import overview_tab

        for part in (
            overview_tab.OverviewRefs,
            overview_tab.load_slots,
            overview_tab.pack_load,
            overview_tab.layout,
        ):
            source = inspect.getsource(part)
            self.assertIn(
                "gantt_chart",
                source,
                f"{part.__name__ if hasattr(part, '__name__') else part} "
                "is missing the gantt_chart wiring",
            )

    def test_gantt_chart_is_a_registered_load_slot(self):
        from agent_trajviz.insight.ui.load import load_slot_keys
        from agent_trajviz.insight.ui.overview_tab import pack_load

        self.assertIn("gantt_chart", load_slot_keys())
        empty = pack_load()
        self.assertIn("gantt_chart", empty)

    def test_build_chart_outputs_falls_back_to_an_empty_figure(self):
        from agent_trajviz.insight.presenters.overview import (
            build_chart_outputs,
            empty_plotly_fig,
        )

        class _Session:
            steps = [step(0, duration=1.0)]  # no timestamps → gantt is None
            agent_summaries = []
            format = ""
            plan_history = []
            plan_metrics = {}
            shown = None
            path = ""
            raw = {}

        ch = build_chart_outputs(_Session())
        self.assertEqual(ch["gantt_fig"].to_json(), empty_plotly_fig().to_json())


if __name__ == "__main__":
    unittest.main()


class TaskToolHoverNoteTest(unittest.TestCase):
    """task 工具的悬停必须注明父级阻塞、子代理在工作。

    task 调用期间父级与子级泳道的时钟重叠是"等待"而非"并行计算"，
    悬停文案需要把这层意思说清楚，避免误读为两个 agent 同时思考。
    """

    def test_task_tool_hover_says_parent_blocked(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=30_000, tool_calls=[
                tc("task", start_ms=10_100, end_ms=29_900),
            ]),
        ]
        fig = build_execution_gantt_chart(steps)
        task_bar = next(b for b in _bars(fig) if b["color"] == TOOL_BAR_COLOR)
        self.assertIn("parent blocked, sub-agent working", task_bar["hovertext"])

    def test_ordinary_tool_hover_has_no_wait_note(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=30_000, tool_calls=[
                tc("bash", start_ms=10_100, end_ms=10_200),
            ]),
        ]
        fig = build_execution_gantt_chart(steps)
        tool_bar = next(b for b in _bars(fig) if b["color"] == TOOL_BAR_COLOR)
        self.assertNotIn("parent blocked", tool_bar["hovertext"])


class IdleGapShadingTest(unittest.TestCase):
    """长空闲区间要加底纹和标注，否则图中空白会被误读为模型卡住。"""

    def test_long_gap_is_shaded_labeled_and_compressed(self):
        steps = [
            step(0, role="user", created_ms=10_000),
            step(1, created_ms=10_000, completed_ms=12_000),
            # 340s 空闲，用户回来发第二条消息
            step(2, role="user", created_ms=352_000),
            step(3, created_ms=352_100, completed_ms=354_000),
        ]
        for s in steps:
            if s["role"] == "user":
                s["text_preview"] = "继续"
        fig = build_execution_gantt_chart(steps)
        wait_bars = [b for b in _bars(fig) if b["lane"] == "user wait"]
        self.assertEqual(1, len(wait_bars))
        # 空闲带被压缩到固定宽度（busy 3.9s → max(15, 0.08*3.9) = 15s）。
        self.assertAlmostEqual(2.0, wait_bars[0]["start"], places=1)
        self.assertAlmostEqual(17.0, wait_bars[0]["end"], places=1)
        # 空闲带用醒目的玫瑰色，且只在自己的泳道上，不越过 agent 泳道。
        self.assertEqual(WAIT_BAND_COLOR, wait_bars[0]["color"])
        self.assertEqual({"main"}, {b["lane"] for b in _bars(fig)} - {"user wait"})
        labels = [a.text for a in fig.layout.annotations]
        self.assertTrue(
            any("idle 5m 40s" in t and "waiting for user" in t for t in labels),
            labels,
        )
        self.assertTrue(any("axis compressed" in t for t in labels), labels)

    def test_compressed_axis_shifts_later_bars_but_hover_keeps_real_times(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=352_100, completed_ms=354_000),
        ]
        fig = build_execution_gantt_chart(steps)
        bars = _bars(fig)
        # 第二根杆在图上紧跟着压缩带（2.0 + 15.0 = 17.0），不再被推到 342s。
        self.assertAlmostEqual(17.0, bars[1]["start"], places=1)
        # 悬停文本仍报告真实时间。
        self.assertIn("342.1s → 344.0s", bars[1]["hovertext"])

    def test_wallclock_mode_keeps_true_scale(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=352_100, completed_ms=354_000),
        ]
        fig = build_execution_gantt_chart(steps, axis_mode=GANTT_AXIS_WALLCLOCK)
        bars = _bars(fig)
        self.assertAlmostEqual(342.1, bars[1]["start"], places=1)
        labels = [a.text for a in fig.layout.annotations]
        self.assertTrue(labels and "axis compressed" not in labels[0], labels)

    def test_weekend_scale_gap_gets_day_label(self):
        two_days_ms = 2 * 24 * 3600 * 1000
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            # 空闲恰好 2 天：12s → 172_802s。
            step(1, created_ms=12_000 + two_days_ms, completed_ms=12_000 + two_days_ms + 2000),
        ]
        fig = build_execution_gantt_chart(steps)
        labels = [a.text for a in fig.layout.annotations]
        self.assertTrue(any("idle 2d" in t for t in labels), labels)
        # 压缩后总轴长仍在分钟级，工作杆不会被压扁。
        bars = _bars(fig)
        self.assertLess(bars[1]["start"], 60.0)

    def test_short_gaps_are_not_shaded(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=12_500, completed_ms=14_000),
        ]
        fig = build_execution_gantt_chart(steps)
        self.assertEqual(0, len(fig.layout.annotations))
        self.assertNotIn("user wait", list(fig.layout.yaxis.categoryarray))


class CompressedAxisTickTest(unittest.TestCase):
    """压缩轴的刻度必须标真实时间，否则轴上数字会撒谎。"""

    def test_compressed_axis_ticks_carry_real_times(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=352_100, completed_ms=354_000),
        ]
        fig = build_execution_gantt_chart(steps)
        xa = fig.layout.xaxis
        # 刻度位置：0、压缩带起点(2.0)、压缩带终点(17.0)。
        self.assertEqual([0.0, 2.0, 17.0], [float(v) for v in xa.tickvals])
        # 标签是这些位置对应的真实时间。
        self.assertEqual(["0s", "2s", "5m 42s"], list(xa.ticktext))

    def test_wallclock_axis_keeps_default_ticks(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=352_100, completed_ms=354_000),
        ]
        fig = build_execution_gantt_chart(steps, axis_mode=GANTT_AXIS_WALLCLOCK)
        self.assertIsNone(fig.layout.xaxis.tickvals)

    def test_no_gap_axis_keeps_default_ticks(self):
        steps = [
            step(0, created_ms=10_000, completed_ms=12_000),
            step(1, created_ms=12_500, completed_ms=14_000),
        ]
        fig = build_execution_gantt_chart(steps)
        self.assertIsNone(fig.layout.xaxis.tickvals)
