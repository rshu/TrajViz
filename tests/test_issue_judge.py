"""LLM Issues fix judge — pack, parse, render (mocked chat)."""

import os
import unittest
from unittest.mock import patch

import requests

from agent_trajviz.insight import assistant
from agent_trajviz.insight.issue_judge import (
    judge_issue_fix,
    judge_overview_issues,
    judge_run_summary,
    iter_judge_overview_issues,
    pack_issue_judge_context,
    parse_issue_judgment,
    parse_run_summary,
)
from agent_trajviz.insight.llm_config import AnalysisLLMConfig
from agent_trajviz.insight.presenters.issues import (
    OverviewIssue,
    collect_overview_issues,
    rank_issues,
    render_overview_issues_html,
)
from tests.overview_session_stub import overview_session


def _cfg(*, source: str = "analyze") -> AnalysisLLMConfig:
    return AnalysisLLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        provider="openai",
        temperature=0.0,
        max_tokens=512,
        timeout=30,
        source=source,
    )


def _issue() -> OverviewIssue:
    return OverviewIssue(
        kind="error",
        title="Bash: exit 1 (1×)",
        detail="fail",
        why="no recovery",
        steps=(1,),
        source_id="fail:0",
    )


def _session(**kwargs):
    kwargs.setdefault("format", "claude_code")
    return overview_session(**kwargs)


class ParseJudgmentTests(unittest.TestCase):
    def test_raw_json(self):
        j = parse_issue_judgment(
            '{"where":"CLAUDE.md","fix":"Stop Bash reads","also":"","confidence":"high"}'
        )
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertEqual(j.fix, "Stop Bash reads")
        self.assertEqual(j.confidence, "high")

    def test_fenced_json(self):
        j = parse_issue_judgment(
            """Here you go:
```json
{"where": "Environment", "fix": "Fix permissions", "also": "not a prompt", "confidence": "medium"}
```
"""
        )
        self.assertEqual(j.where, "Environment")
        self.assertIn("permissions", j.fix)

    def test_rejects_missing_fix(self):
        with self.assertRaises(ValueError):
            parse_issue_judgment('{"where":"x","fix":"","also":"","confidence":"low"}')


class PackContextTests(unittest.TestCase):
    def test_includes_workflow_window_and_skills(self):
        session = _session(
            steps=[
                {
                    "index": 0,
                    "role": "user",
                    "text_preview": "Fix the auth bug in login.py",
                    "tool_calls": [],
                },
                {
                    "index": 4,
                    "role": "assistant",
                    "text_preview": "searching",
                    "tool_calls": [{
                        "tool_name": "Bash",
                        "input": {"command": "cat foo.py"},
                        "error_type": "exit 1",
                    }],
                },
                {
                    "index": 5,
                    "role": "assistant",
                    "text_preview": "retry",
                    "tool_calls": [{"tool_name": "Read", "input": {"file_path": "foo.py"}}],
                },
            ],
            file_interactions=[{
                "step": 3,
                "tool": "Skill",
                "path": "skill:search",
                "type": "skill",
            }],
        )
        issue = OverviewIssue(
            kind="antipattern",
            title="Bash-for-reading (1×)",
            detail="sed/cat",
            why="shell reads",
            steps=(4,),
            source_id="antipattern:bash_read",
        )
        packed = pack_issue_judge_context(session, issue)
        self.assertIn("Bash-for-reading", packed)
        self.assertIn('"index": 4', packed)
        self.assertIn("cat foo.py", packed)
        self.assertIn("skill:search", packed)
        self.assertIn("claude_code", packed)
        self.assertIn("user_task", packed)
        self.assertIn("Fix the auth bug in login.py", packed)


class JudgeMockTests(unittest.TestCase):
    def test_judge_issue_fix_uses_chat_fn(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        issue = OverviewIssue(
            kind="error",
            title="Bash: exit 1 (1×)",
            detail="fail",
            why="no recovery",
            steps=(1,),
            source_id="fail:0",
        )

        def chat_fn(config, system, messages):
            self.assertIn("fix judge", system.lower())
            self.assertIn("Write the values of \"where\", \"fix\", and \"also\" in English", system)
            self.assertIn("English", messages[0]["content"])
            self.assertEqual(messages[0]["role"], "user")
            return (
                '{"where":"CLAUDE.md","fix":"Bash: check the path exists before reading it",'
                '"also":"","confidence":"high"}'
            )

        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "ANALYZE_LANGUAGE"}, clear=True):
            j = judge_issue_fix(session, issue, config=_cfg(), chat_fn=chat_fn)
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertIn("Bash", j.fix)

    def test_judge_overview_attaches_judgment_and_html(self):
        session = _session(
            fruitless_streaks=[{
                "start_step": 4,
                "end_step": 6,
                "length": 3,
            }],
            steps=[
                {"index": i, "role": "assistant", "tool_calls": []}
                for i in range(4, 7)
            ],
        )
        issues = rank_issues(collect_overview_issues(session))
        self.assertEqual(len(issues), 1)

        def chat_fn(config, system, messages):
            return (
                '{"where":"CLAUDE.md","fix":"空搜索超过 N 次后停止",'
                '"also":"预先给出路径","confidence":"medium"}'
            )

        judged, errors = judge_overview_issues(
            session, issues, config=_cfg(), chat_fn=chat_fn,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(judged[0].judgment)
        html = render_overview_issues_html(judged)
        # Option A: the LLM judgment owns the single action slot.
        self.assertIn("修改位置", html)
        self.assertIn("CLAUDE.md", html)
        self.assertIn("空搜索超过 N 次后停止", html)
        self.assertIn("AI · 置信度：中", html)
        self.assertIn(">建议<", html)
        # The rule-based playbook folds away under 技术细节 as 常规做法.
        self.assertIn("常规做法", html)
        self.assertIn("更换搜索关键词或路径", html)
        # Upstream i18n marker (English fallback); zh catalog: "{count} 项附有模型修复建议".
        self.assertIn("1 with LLM fix", html)

    def test_iter_judge_yields_before_each_and_finished(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        issues = [
            OverviewIssue(
                kind="error",
                title=f"Issue {i}",
                detail="d",
                why="w",
                steps=(1,),
                source_id=f"t:{i}",
            )
            for i in range(2)
        ]
        calls = {"n": 0}

        def chat_fn(config, system, messages):
            calls["n"] += 1
            return (
                '{"where":"CLAUDE.md","fix":"修复",'
                '"also":"","confidence":"high"}'
            )

        snaps = list(iter_judge_overview_issues(
            session, issues, config=_cfg(), chat_fn=chat_fn,
        ))
        self.assertEqual(len(snaps), 3)  # before #1, before #2, finished
        self.assertFalse(snaps[0].finished)
        self.assertEqual(snaps[0].current, 1)
        self.assertEqual(snaps[0].current_title, "Issue 0")
        self.assertIsNone(snaps[0].issues[0].judgment)
        self.assertFalse(snaps[1].finished)
        self.assertEqual(snaps[1].current, 2)
        self.assertIsNotNone(snaps[1].issues[0].judgment)  # first done
        self.assertTrue(snaps[2].finished)
        self.assertEqual(calls["n"], 2)
        self.assertTrue(all(i.judgment for i in snaps[2].issues))

    def test_judge_error_names_the_settings_field_for_a_settings_connection(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        response = requests.models.Response()
        response.status_code = 401

        def chat_fn(config, system, messages):
            raise requests.HTTPError("401 Client Error", response=response)

        _, errors = judge_overview_issues(
            session, [_issue()], config=_cfg(source="session"), chat_fn=chat_fn,
        )
        self.assertEqual(errors, [
            "Bash: exit 1 (1×): "
            "The analysis API rejected the key (HTTP 401). Check the API key in Settings.",
        ])

    def test_judge_reports_an_exhausted_output_budget(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])

        def chat_fn(config, system, messages):
            raise assistant.OutputBudgetExhausted(2048)

        _, errors = judge_overview_issues(session, [_issue()], config=_cfg(), chat_fn=chat_fn)
        self.assertEqual(len(errors), 1)
        self.assertIn("entire output budget of 2048 tokens", errors[0])

    def test_render_without_judgment_has_no_change_fix(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                steps=(1,),
            )
        ])
        self.assertNotIn(">Change<", html)
        self.assertNotIn(">Fix", html)


class RunSummaryJudgeTests(unittest.TestCase):
    """The one-shot (task, outcome) LLM summary of the whole run."""

    def _session(self):
        return _session(
            steps=[
                {
                    "index": 0,
                    "role": "user",
                    "parts": [{"type": "text", "text": "把 SO_LOAD_PATH 改成新路径"}],
                    "tool_calls": [],
                },
                {
                    "index": 1,
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "已改完 11 个文件"}],
                    "tool_calls": [],
                },
            ],
            metrics={"tool_call_count": 12, "tokens": {"total": 40000}},
        )

    def test_parse_raw_and_fenced_json(self):
        self.assertEqual(
            parse_run_summary('{"task":"改路径","outcome":"完成"}'),
            ("改路径", "完成"),
        )
        task, outcome = parse_run_summary(
            '```json\n{"task": "改路径", "outcome": "完成"}\n```'
        )
        self.assertEqual((task, outcome), ("改路径", "完成"))

    def test_parse_rejects_missing_task(self):
        with self.assertRaises(ValueError):
            parse_run_summary('{"task":"","outcome":"完成"}')
        with self.assertRaises(ValueError):
            parse_run_summary("no json here")

    def test_judge_run_summary_sends_context_and_language(self):
        seen = {}

        def chat_fn(config, system, messages):
            seen["system"] = system
            seen["user"] = messages[0]["content"]
            return '{"task":"把加载路径迁移到新目录","outcome":"11 个文件全部改完"}'

        task, outcome = judge_run_summary(
            self._session(), config=_cfg(), chat_fn=chat_fn, language="zh",
        )
        self.assertEqual(task, "把加载路径迁移到新目录")
        self.assertEqual(outcome, "11 个文件全部改完")
        self.assertIn("run summarizer", seen["system"])
        self.assertIn("把 SO_LOAD_PATH 改成新路径", seen["user"])
        self.assertIn("已改完 11 个文件", seen["user"])
        self.assertIn('"steps": 2', seen["user"])
        self.assertIn("Simplified Chinese (Mandarin)", seen["user"])

    def test_judge_run_summary_requires_config(self):
        empty = AnalysisLLMConfig(
            base_url="", api_key="", model="", provider="openai",
            temperature=0.0, max_tokens=512, timeout=30, source="analyze",
        )
        with self.assertRaises(ValueError):
            judge_run_summary(self._session(), config=empty, require_config=True)


if __name__ == "__main__":
    unittest.main()
