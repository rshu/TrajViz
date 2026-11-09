"""Overview Issues triage panel — collect, rank, render."""

import unittest

from agent_trajviz.insight.issues import dedupe_overlapping_issues
from agent_trajviz.insight.presenters.issues import (
    IssueJudgment,
    OverviewIssue,
    build_overview_issues_html,
    build_run_summary_html,
    collect_overview_issues,
    group_issue_families,
    issue_analysis_question_from_payload,
    overview_issues_payload,
    prepare_overview_issues,
    rank_issues,
    render_overview_issues_html,
)
from agent_trajviz.insight.presenters.overview import build_token_summary_html
from tests.overview_session_stub import overview_session as _session


class OverviewIssuesTests(unittest.TestCase):
    def test_healthy_empty_state(self):
        html = build_overview_issues_html(_session())
        self.assertIn("overview-issues", html)
        self.assertIn("<details", html)
        self.assertIn("overview-issues-summary", html)
        # Verdict stays Chinese (CodeAgent); panel chrome uses upstream i18n markers.
        self.assertIn("未检测到明显问题", html)
        self.assertIn("none detected", html)
        self.assertIn("No major workflow issues detected", html)
        self.assertNotIn("insight-step-link", html)
        self.assertNotIn("<details class='overview-issues-panel' id='overview-issues' open", html)

    def test_issues_panel_starts_expanded(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                steps=(1,),
            )
        ])
        self.assertIn("<details class='overview-issues-panel' id='overview-issues' open>", html)
        self.assertIn("1 issue", html)
        self.assertIn("overview-issue-index", html)
        self.assertIn("#1", html)
        self.assertIn("错误", html)
        self.assertNotIn("修改位置", html)
        self.assertNotIn("修复建议", html)

    def test_errors_rank_above_antipatterns(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "npm: exit code 1",
                    "count": 2,
                    "example_error": "command failed",
                    "recovery_path": ["Read", "Edit"],
                    "steps": [5, 9],
                    "error_class": "tool",
                }],
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
                performance_bottlenecks=[{
                    "step_idx": 8,
                    "duration": 42.5,
                    "cause": "tool",
                    "title": "Tool bottleneck: Bash: npm test (40.0s)",
                    "detail": "Step 8: 42.5s — 40s executing tools",
                    "why": "tool wait",
                }],
            )
        )
        shown = rank_issues(issues)
        self.assertEqual(shown[0].kind, "error")
        self.assertEqual(shown[1].kind, "antipattern")
        self.assertEqual(shown[2].kind, "bottleneck")

    def test_system_scaffold_miss_is_low_risk_antipattern(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Grep: No matches found",
                    "count": 2,
                    "example_error": "No matches found",
                    "recovery_path": None,
                    "steps": [3, 4],
                    "error_class": "system",
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn("System error", issues[0].title)
        self.assertIn("low risk", issues[0].why)

    def test_frequent_system_errors_call_out_volume(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Read: ENOENT",
                    "count": 6,
                    "example_error": "ENOENT",
                    "recovery_path": None,
                    "steps": [1, 2, 3, 4, 5, 6],
                    "error_class": "system",
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn("Frequent system errors", issues[0].title)
        self.assertIn("volume", issues[0].why)

    def test_bottleneck_issue_from_performance_bottlenecks(self):
        issues = collect_overview_issues(
            _session(
                performance_bottlenecks=[{
                    "step_idx": 12,
                    "duration": 18.2,
                    "cause": "tool",
                    "title": "Tool bottleneck: Bash: npm test (14.0s)",
                    "detail": "Step 12: 18.2s — 15s executing tools (Bash: npm test 14s)",
                    "why": "Most of this outlier step was spent in a tool.",
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "bottleneck")
        self.assertEqual(issues[0].steps, (12,))
        self.assertIn("Tool bottleneck", issues[0].title)
        self.assertIn("npm test", issues[0].detail)

    def test_progress_banner_visible_while_judging(self):
        html = render_overview_issues_html(
            [
                OverviewIssue(
                    kind="error",
                    title="Bash: exit 1 (1×)",
                    detail="boom",
                    why="x",
                    steps=(3,),
                )
            ],
            progress="Suggesting fixes 1/3 — Bash: exit 1 (1×)",
        )
        self.assertIn("overview-issues-progress", html)
        self.assertIn("Thinking…", html)
        self.assertIn("overview-issues-progress-spinner", html)
        self.assertIn("animateTransform", html)
        self.assertIn("currentColor", html)
        self.assertIn("Suggesting fixes 1/3", html)
        self.assertIn("suggesting fixes…", html)
        self.assertNotIn("修改位置", html)

    def test_step_chips_without_fix_or_change(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="Bash: exit 1 (1×)",
                detail="boom",
                why="Typical recovery: Read",
                steps=(3, 7),
            )
        ])
        self.assertIn("tvGotoWorkflowStep(3)", html)
        self.assertIn("tvGotoWorkflowStep(7)", html)
        self.assertIn(">第3步<", html)
        self.assertIn(">第7步<", html)
        self.assertIn("overview-issue-why", html)
        self.assertIn("overview-issue-action", html)
        self.assertIn("overview-issue-tech", html)
        self.assertIn(">影响<", html)
        self.assertIn(">建议<", html)
        self.assertIn(">技术细节<", html)
        self.assertIn("失败后智能体调用 Read，自行恢复了", html)
        self.assertIn("overview-issue-analyze", html)
        self.assertIn("AI 深入分析", html)
        self.assertIn("tvAnalyzeIssue", html)
        self.assertIn('data-issue-index="1"', html)
        self.assertNotIn("Analyze this specific Overview issue", html)
        self.assertNotIn("overview-issue-more", html)
        self.assertNotIn("修改位置", html)
        self.assertNotIn("修复建议", html)
        self.assertNotIn("补充：", html)

    def test_failure_shows_recovery_as_why_not_fix(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Bash: exit 1",
                    "count": 2,
                    "example_error": "command failed",
                    "recovery_path": ["Read", "Edit"],
                    "steps": [5, 9],
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("Read → Edit", issues[0].why)
        self.assertIn("Bash: exit 1", issues[0].title)  # English data preserved
        html = render_overview_issues_html(issues)
        self.assertNotIn("修复建议", html)
        self.assertNotIn("修改位置", html)

    def test_issue_ask_payload_rebuilds_focused_question(self):
        issue = OverviewIssue(
            kind="error",
            title="Bash: exit 1 (1×)",
            detail="boom",
            why="Typical recovery: Read",
            steps=(3, 7),
            source_id="fail:Bash: exit 1",
            judgment=IssueJudgment(where="CLAUDE.md", fix="Check the path", also="", confidence="high"),
        )
        payload = overview_issues_payload([issue])
        question = issue_analysis_question_from_payload(payload, 1)
        self.assertIsNotNone(question)
        self.assertIn("Analyze this specific Overview issue directly (issue #1)", question)
        self.assertIn("Bash: exit 1", question)
        self.assertIn("Current suggested fix location: CLAUDE.md", question)
        self.assertIsNone(issue_analysis_question_from_payload(payload, 2))

    def test_shows_all_ranked_issues(self):
        issues = [
            OverviewIssue(
                kind="antipattern",
                title=f"Waste pattern #{i}",
                detail="x",
                why="y",
                steps=(i,),
            )
            for i in range(12)
        ]
        shown = rank_issues(issues)
        self.assertEqual(len(shown), 12)
        html = render_overview_issues_html(shown)
        self.assertIn("12 issues", html)
        self.assertNotIn("Show all", html)
        self.assertNotIn("overview-issues-remainder", html)
        for i in range(12):
            self.assertIn(f"Waste pattern #{i}", html)
            self.assertIn(f"#{i + 1}", html)
        self.assertIn("行为问题", html)

    def test_skips_tool_error_rollup_when_failure_patterns_exist(self):
        session = _session(
            failure_patterns=[{
                "cluster_label": "Bash: exit 1",
                "count": 1,
                "example_error": "fail",
                "recovery_path": None,
                "steps": [2],
            }],
            steps=[{
                "index": 2,
                "tool_calls": [{"error_type": "platform"}],
            }],
        )
        issues = collect_overview_issues(session)
        kinds_titles = [(i.kind, i.title) for i in issues]
        self.assertEqual(len([t for k, t in kinds_titles if k == "error"]), 1)
        self.assertNotIn("tool error(s)", "".join(t for _, t in kinds_titles))

    def test_fruitless_streak_becomes_antipattern_issue(self):
        issues = collect_overview_issues(
            _session(
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn(4, issues[0].steps)
        self.assertIn(6, issues[0].steps)

    def test_premature_compactions_become_antipattern_issues(self):
        issues = collect_overview_issues(
            _session(
                premature_compactions=[
                    {"step": 139, "occupancy_before": 11_941, "occupancy_after": 12_766,
                     "before_pct": 9.3, "window_limit": 128_000, "grew": True,
                     "reason": "window_grew"},
                    {"step": 245, "occupancy_before": 41_935, "occupancy_after": 12_130,
                     "before_pct": 32.8, "window_limit": 128_000, "grew": False,
                     "reason": "low_occupancy"},
                    {"step": 259, "occupancy_before": 29_140, "occupancy_after": 12_162,
                     "before_pct": 22.8, "window_limit": 128_000, "grew": False,
                     "reason": "low_occupancy"},
                ],
            )
        )
        titles = [i.title for i in issues]
        self.assertTrue(any("grew the context window" in t for t in titles))
        self.assertTrue(any("premature compaction" in t.lower() for t in titles))
        grew = next(i for i in issues if "grew the context window" in i.title)
        premature = next(i for i in issues if "premature compaction" in i.title.lower())
        self.assertEqual(grew.steps, (139,))
        self.assertEqual(premature.steps, (245, 259))
        self.assertIn("23–33%", premature.detail)
        self.assertIn("128k", premature.detail)
        html = render_overview_issues_html(rank_issues(issues))
        self.assertIn("上下文被过早压缩", html)
        self.assertIn("压缩未起效：上下文不降反增", html)
        self.assertIn("压缩时窗口占用：假定 128k token 窗口的", html)
        # Visible chrome is Chinese; the AI ask prompt is rebuilt server-side.
        self.assertNotIn("overview-issue-title'>Premature compactions", html)
        self.assertNotIn("overview-issue-detail'>window occupancy at compaction time", html)
        self.assertNotIn("Title: Premature compactions", html)
        self.assertIn('data-issue-index="', html)

    def test_failure_cascade_skips_single_step_chains(self):
        issues = collect_overview_issues(
            _session(
                failure_chains=[
                    {"start": 1, "end": 1, "steps": [1]},
                    {"start": 3, "end": 5, "steps": [3, 4, 5]},
                ],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "error")
        self.assertIn("cascade", issues[0].title.lower())
        self.assertEqual(issues[0].steps, (3, 4, 5))

    def test_dedupe_drops_cascade_covered_by_failure_pattern(self):
        session = _session(
            failure_patterns=[{
                "cluster_label": "Bash: exit 1",
                "count": 3,
                "example_error": "fail",
                "recovery_path": None,
                "steps": [3, 4, 5],
            }],
            failure_chains=[
                {"start": 3, "end": 5, "steps": [3, 4, 5]},
            ],
        )
        raw = collect_overview_issues(session)
        self.assertEqual(len(raw), 2)  # raw collect still has both
        self.assertEqual(len(dedupe_overlapping_issues(raw)), 1)
        prepared = prepare_overview_issues(session)
        self.assertEqual(len(prepared), 1)
        self.assertTrue(prepared[0].source_id.startswith("fail:"))

    def test_edit_thrash_group_and_search_waste_group(self):
        issues = rank_issues(collect_overview_issues(
            _session(
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
                edit_thrash=[
                    {
                        "path": "src/a.py",
                        "count": 3,
                        "fail_count": 2,
                        "steps": [5, 6],
                        "start_step": 5,
                        "end_step": 6,
                    },
                    {
                        "path": "src/b.py",
                        "count": 2,
                        "fail_count": 1,
                        "steps": [8, 9],
                        "start_step": 8,
                        "end_step": 9,
                    },
                ],
                repeated_searches=[{
                    "signature": "grep:foo|",
                    "display": "foo|",
                    "count": 3,
                    "steps": [1, 4, 9],
                }],
            )
        ))
        views = group_issue_families(issues)
        thrash_groups = [v for v in views if v.parent_title and "失败编辑重试" in v.parent_title]
        search_groups = [v for v in views if v.parent_title == "搜索浪费"]
        self.assertEqual(len(thrash_groups), 1)
        self.assertEqual(len(thrash_groups[0].leaves), 2)
        self.assertEqual(len(search_groups), 1)
        self.assertEqual(len(search_groups[0].leaves), 2)

        html = render_overview_issues_html(issues)
        self.assertIn("overview-issue-group", html)
        self.assertIn("overview-issue-group-head", html)
        self.assertIn("失败编辑重试（2 个文件）", html)
        self.assertIn("搜索浪费", html)
        for i in range(1, len(issues) + 1):
            self.assertIn(f'data-issue-index="{i}"', html)
        self.assertIn(f"{len(issues)} issues", html)
        payload = overview_issues_payload(issues)
        self.assertEqual(len(payload), len(issues))
        self.assertIsNotNone(issue_analysis_question_from_payload(payload, 1))

    def test_singleton_family_stays_flat(self):
        issues = rank_issues(collect_overview_issues(
            _session(
                edit_thrash=[{
                    "path": "src/only.py",
                    "count": 3,
                    "fail_count": 2,
                    "steps": [5, 6],
                    "start_step": 5,
                    "end_step": 6,
                }],
            )
        ))
        views = group_issue_families(issues)
        self.assertEqual(len(views), 1)
        self.assertIsNone(views[0].parent_title)
        html = render_overview_issues_html(issues)
        self.assertNotIn("overview-issue-group", html)
        self.assertIn('data-issue-index="1"', html)

    def test_bottleneck_same_cause_groups(self):
        issues = rank_issues(collect_overview_issues(
            _session(
                performance_bottlenecks=[
                    {
                        "step_idx": 2,
                        "cause": "idle",
                        "title": "Idle bottleneck at #2",
                        "detail": "wait",
                    },
                    {
                        "step_idx": 7,
                        "cause": "idle",
                        "title": "Idle bottleneck at #7",
                        "detail": "wait again",
                    },
                    {
                        "step_idx": 4,
                        "cause": "tool",
                        "title": "Tool bottleneck at #4",
                        "detail": "slow tool",
                    },
                ],
            )
        ))
        views = group_issue_families(issues)
        idle_groups = [v for v in views if v.parent_title and "空闲" in v.parent_title]
        self.assertEqual(len(idle_groups), 1)
        self.assertEqual(len(idle_groups[0].leaves), 2)
        flat_tool = [v for v in views if v.parent_title is None]
        self.assertEqual(len(flat_tool), 1)
        self.assertTrue(flat_tool[0].leaves[0][1].source_id.startswith("bottleneck:tool:"))

    def test_group_preview_hides_beyond_three(self):
        thrash = [
            {
                "path": f"src/f{i}.py",
                "count": 3,
                "fail_count": 2,
                "steps": [i * 2, i * 2 + 1],
                "start_step": i * 2,
                "end_step": i * 2 + 1,
            }
            for i in range(1, 6)
        ]
        issues = rank_issues(collect_overview_issues(_session(edit_thrash=thrash)))
        self.assertEqual(len(issues), 5)
        html = render_overview_issues_html(issues)
        self.assertIn("overview-issue-more", html)
        self.assertIn("显示其余 2 项", html)
        for i in range(1, 6):
            self.assertIn(f'data-issue-index="{i}"', html)
        html3 = render_overview_issues_html(issues[:3])
        self.assertNotIn("overview-issue-more", html3)

    def test_plan_resets_edit_thrash_repeated_search(self):
        issues = collect_overview_issues(
            _session(
                plan_metrics={"plan_resets": 2, "stalled": []},
                plan_history=[{"step": 2, "items": []}, {"step": 10, "items": []}],
                edit_thrash=[{
                    "path": "src/app.py",
                    "count": 4,
                    "fail_count": 2,
                    "steps": [5, 6, 7, 8],
                    "start_step": 5,
                    "end_step": 8,
                }],
                repeated_searches=[{
                    "signature": "grep:foo|",
                    "display": "foo|",
                    "count": 3,
                    "steps": [1, 4, 9],
                }],
            )
        )
        titles = [i.title for i in issues]
        self.assertTrue(any("plan reset" in t.lower() for t in titles))
        self.assertTrue(any("Failed edit retries" in t for t in titles))
        self.assertTrue(any("Repeated empty search" in t for t in titles))
        thrash = next(i for i in issues if "Failed edit retries" in i.title)
        self.assertEqual(thrash.steps, (5, 6, 7, 8))
        for title in titles:
            self.assertFalse(
                title[:1].isdigit(),
                f"issue title should not start with a digit: {title!r}",
            )


class RunSummaryTests(unittest.TestCase):
    """The three-line run story under the verdict banner."""

    @staticmethod
    def _step(role, text="", *, tokens=None, tool_calls=None):
        return {
            "index": 0,
            "role": role,
            "agent": "",
            "is_sub_agent": False,
            "session_id": "s1",
            "tokens": tokens or {},
            "parts": [{"type": "text", "text": text}] if text else [],
            "tool_calls": tool_calls or [],
            "model_id": "m",
        }

    def _session(self, steps, metrics=None):
        session = _session()
        session.steps = steps
        if metrics is not None:
            session.metrics = metrics
        return session

    def test_ai_rows_and_numbers_rows(self):
        session = self._session(
            [
                self._step("user", "修复登录页在 Safari 下崩溃的问题"),
                self._step(
                    "assistant",
                    "已修复：给 flex 容器补了 min-width，回归测试全绿。",
                    tokens={"total": 12345, "input": 1000, "cache_read": 9000},
                ),
            ],
            metrics={
                "wall_clock": 125,
                "tokens": {"total": 12345, "input": 1000, "cache_read": 9000},
                "tool_call_count": 7,
                "tool_breakdown": {"bash": 5, "read": 2},
            },
        )
        html = build_run_summary_html(
            session, ai_summary=("修复登录页崩溃", "已修复，测试全绿")
        )
        self.assertIn("ov-run-summary", html)
        self.assertIn("ov-run-label'>任务", html)
        self.assertIn("修复登录页崩溃", html)
        self.assertIn("ov-run-label'>结局", html)
        self.assertIn("已修复，测试全绿", html)
        self.assertIn("ov-run-label'>数字", html)
        self.assertIn("2 步", html)
        self.assertIn("2m 5s", html)
        self.assertIn("12.3k token", html)
        self.assertIn("缓存命中 90%", html)
        self.assertIn("工具 7 次（bash 最多，5 次）", html)

    def test_default_mode_hides_task_and_outcome_rows(self):
        # Without AI the 任务/结局 rows are omitted — only 数字 remains.
        session = self._session(
            [
                self._step("user", "修复登录页在 Safari 下崩溃的问题"),
                self._step("assistant", "已修复：回归测试全绿。"),
            ],
        )
        html = build_run_summary_html(session)
        self.assertIn("ov-run-summary", html)
        self.assertIn("ov-run-label'>数字", html)
        self.assertIn("2 步", html)
        self.assertNotIn("ov-run-label'>任务", html)
        self.assertNotIn("ov-run-label'>结局", html)
        self.assertNotIn("修复登录页在 Safari 下崩溃的问题", html)
        self.assertNotIn("已修复：回归测试全绿", html)

    def test_summary_sits_between_banner_and_panel(self):
        session = self._session(
            [self._step("user", "hi"), self._step("assistant", "done")],
        )
        html = build_overview_issues_html(session)
        self.assertLess(html.index("ov-verdict"), html.index("ov-run-summary"))
        self.assertLess(html.index("ov-run-summary"), html.index("overview-issues-panel"))

    def test_raw_user_text_is_never_rendered(self):
        # Raw first/last messages are not a display path any more — the rows
        # are AI-only, so untrusted user text never reaches the summary.
        session = self._session(
            [self._step("user", "<img src=x onerror=alert(1)>")],
        )
        for html in (
            build_run_summary_html(session),
            build_run_summary_html(session, ai_pending=True),
        ):
            self.assertNotIn("<img src=x", html)
            self.assertNotIn("onerror", html)

    def test_all_failed_ending_is_flagged(self):
        session = self._session(
            [
                self._step("user", "跑一下部署"),
                self._step(
                    "assistant",
                    "尝试部署",
                    tool_calls=[
                        {"tool_name": "bash", "status": "error"},
                        {"tool_name": "bash", "status": "error"},
                    ],
                ),
            ],
        )
        html = build_run_summary_html(session, ai_summary=("部署", "尝试部署"))
        self.assertIn("ov-run-note", html)
        self.assertIn("该步工具调用全部失败", html)

    def test_recovered_ending_is_not_flagged(self):
        session = self._session(
            [
                self._step("user", "跑一下部署"),
                self._step(
                    "assistant",
                    "部署完成",
                    tool_calls=[
                        {"tool_name": "bash", "status": "error"},
                        {"tool_name": "bash", "status": "completed"},
                    ],
                ),
            ],
        )
        html = build_run_summary_html(session, ai_summary=("部署", "部署完成"))
        self.assertNotIn("ov-run-note", html)

    def test_no_steps_means_no_summary(self):
        self.assertEqual(build_run_summary_html(_session()), "")
        self.assertNotIn("ov-run-summary", build_overview_issues_html(_session()))

    def test_pending_mode_shows_spinner_placeholder(self):
        session = self._session(
            [self._step("user", "很长的原始任务描述"), self._step("assistant", "完成")],
        )
        html = build_run_summary_html(session, ai_pending=True)
        self.assertIn("ov-run-label'>任务与结局", html)
        self.assertIn("正在等待 AI 总结任务与结局…", html)
        self.assertIn("overview-issues-progress-spinner", html)
        self.assertIn("animateTransform", html)
        self.assertIn("ov-run-label'>数字", html)
        self.assertNotIn("很长的原始任务描述", html)
        self.assertNotIn("ov-run-label'>任务</span>", html)

    def test_ai_summary_replaces_raw_text_with_badge(self):
        session = self._session(
            [
                self._step("user", "很长的原始任务描述，不应该出现在 AI 总结模式下"),
                self._step("assistant", "很长的原始结局描述"),
            ],
        )
        html = build_run_summary_html(session, ai_summary=("AI 总结的任务", "AI 总结的结局"))
        self.assertIn("overview-issue-label-ai", html)
        self.assertIn("AI 总结的任务", html)
        self.assertIn("AI 总结的结局", html)
        self.assertNotIn("不应该出现在 AI 总结模式下", html)
        self.assertNotIn("很长的原始结局描述", html)

    def test_ai_summary_escapes_model_text(self):
        session = self._session([self._step("user", "hi")])
        html = build_run_summary_html(
            session, ai_summary=("<img src=x onerror=alert(1)>", "ok")
        )
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;img src=x onerror=", html)

    def test_partial_ai_summary_shows_placeholder_per_row(self):
        session = self._session(
            [self._step("user", "原始任务"), self._step("assistant", "原始结局")],
        )
        html = build_run_summary_html(session, ai_summary=("", "AI 结局"))
        self.assertIn("（AI 未总结任务）", html)
        self.assertIn("AI 结局", html)
        self.assertNotIn("原始任务", html)
        self.assertNotIn("原始结局", html)

    def test_build_overview_issues_html_passes_ai_state_through(self):
        session = self._session(
            [self._step("user", "hi"), self._step("assistant", "done")],
        )
        html = build_overview_issues_html(session, ai_summary=("任务A", "结局B"))
        self.assertIn("任务A", html)
        self.assertIn("结局B", html)
        html = build_overview_issues_html(session, ai_pending=True)
        self.assertIn("正在等待 AI 总结任务与结局…", html)


class TokenSummaryLineTests(unittest.TestCase):
    """The always-visible takeaway line above the Token Usage chart."""

    @staticmethod
    def _step(role, *, tokens=None, parts=None):
        return {
            "index": 0,
            "role": role,
            "agent": "",
            "is_sub_agent": False,
            "session_id": "s1",
            "tokens": tokens or {},
            "parts": parts or [],
            "tool_calls": [],
            "model_id": "m",
        }

    def _session(self, steps):
        session = _session()
        session.steps = steps
        return session

    def test_cache_hit_max_step_and_peak(self):
        html = build_token_summary_html(self._session([
            self._step("user"),
            self._step("assistant", tokens={"total": 8000, "input": 500, "cache_read": 7000}),
            self._step("assistant", tokens={"total": 20000, "input": 300, "cache_read": 19000}),
        ]))
        self.assertIn("缓存命中 97%（命中 26k / 新算 800）", html)
        self.assertIn("最大单步 20k（第 2 步）", html)
        self.assertIn("上下文峰值 19.3k", html)
        self.assertNotIn("压缩", html)

    def test_compaction_drop_is_summarised(self):
        html = build_token_summary_html(self._session([
            self._step("assistant", tokens={"total": 10000, "input": 10000}),
            self._step("user", parts=[{"type": "compaction", "summary": "s"}]),
            self._step("assistant", tokens={"total": 4000, "input": 4000}),
        ]))
        self.assertIn("压缩丢弃 6k token（1 次）", html)

    def test_empty_session_renders_nothing(self):
        self.assertEqual(build_token_summary_html(self._session([])), "")
        session = _session()
        session.steps = [{"role": "user", "parts": [], "tool_calls": [], "tokens": {}}]
        self.assertEqual(build_token_summary_html(session), "")


if __name__ == "__main__":
    unittest.main()
