"""AI Trajectory Analysis sidebar: config, brief packing, chat (no live HTTP)."""

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from agent_trajviz.insight import assistant, llm_config
from agent_trajviz.insight.llm_config import (
    AnalysisLLMConfig,
    config_status_html,
    load_env_files,
    resolve_analysis_config,
    setup_help_text,
)


def _cfg(**overrides) -> AnalysisLLMConfig:
    base = dict(
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        model="demo-model",
        provider="openai",
        temperature=0.2,
        max_tokens=2048,
        timeout=120,
        source="analyze",
    )
    base.update(overrides)
    return AnalysisLLMConfig(**base)


class _FakeResponse:
    """A 200 response carrying *payload*, for a patched ``requests.post``."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _openai_reply(content, finish_reason: str) -> dict:
    return {"choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content}}]}


def _anthropic_reply(blocks: list[dict], stop_reason: str) -> dict:
    return {"type": "message", "role": "assistant", "content": blocks, "stop_reason": stop_reason}


def _http_error(status: int) -> requests.HTTPError:
    response = requests.models.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=response)


def _raising(exc: BaseException):
    """A chat_fn whose request fails with *exc*."""

    def chat_fn(*_args, **_kwargs):
        raise exc

    return chat_fn


def _step(index, role="assistant", *, duration=1.0, tokens=10, tools=None, preview=""):
    tool_calls = tools or []
    return {
        "index": index,
        "role": role,
        "duration": duration,
        "finish": "",
        "text_preview": preview,
        "error_count": sum(1 for tc in tool_calls if tc.get("status") == "error" or tc.get("error_type")),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "tokens": {
            "total": tokens,
            "input": tokens,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
        "parts": [],
    }


class AnalysisConfigTests(unittest.TestCase):
    def test_analyze_vars_win_over_label_vars(self):
        env = {
            "ANALYZE_BASE_URL": "https://analyze.example/v1",
            "ANALYZE_API_KEY": "sk-analyze",
            "ANALYZE_MODEL": "analyze-model",
            "ANALYZE_PROVIDER": "openai",
            "LABEL_BASE_URL": "https://label.example/v1",
            "LABEL_API_KEY": "sk-label",
            "LABEL_MODEL": "label-model",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
        self.assertTrue(cfg.ready)
        self.assertEqual(cfg.base_url, "https://analyze.example/v1")
        self.assertEqual(cfg.api_key, "sk-analyze")
        self.assertEqual(cfg.model, "analyze-model")
        self.assertEqual(cfg.source, "analyze")

    def test_falls_back_to_label_vars(self):
        env = {
            "LABEL_BASE_URL": "https://label.example/v1",
            "LABEL_API_KEY": "sk-label",
            "LABEL_MODEL": "label-model",
            "LABEL_PROVIDER": "anthropic",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
        self.assertTrue(cfg.ready)
        self.assertEqual(cfg.model, "label-model")
        self.assertEqual(cfg.provider, "anthropic")
        self.assertEqual(cfg.source, "label")

    def test_default_output_budget_leaves_room_for_reasoning(self):
        # Reasoning models spend part of max_tokens on hidden thinking before the visible reply.
        for env in ({}, {"LABEL_MAX_TOKENS": "0"}, {"ANALYZE_MAX_TOKENS": "-5"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_analysis_config().max_tokens, 8192)
        with patch.dict(os.environ, {"ANALYZE_MAX_TOKENS": "4096"}, clear=True):
            self.assertEqual(resolve_analysis_config().max_tokens, 4096)

    def test_missing_key_is_not_ready(self):
        env = {
            "ANALYZE_BASE_URL": "https://api.example.com/v1",
            "ANALYZE_MODEL": "demo",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = resolve_analysis_config()
            html = config_status_html(loaded_steps=0)
        self.assertFalse(cfg.ready)
        self.assertTrue(any("API_KEY" in name for name in cfg.missing))
        self.assertIn("Not configured", html)
        self.assertNotIn("sk-", html)
        self.assertIn("ANALYZE_API_KEY", html)

    def test_dotenv_is_fill_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ANALYZE_MODEL=from-file\nANALYZE_API_KEY=sk-file\n", encoding="utf-8")
            previous = os.getcwd()
            env = os.environ.copy()
            env["ANALYZE_MODEL"] = "from-process"
            env.pop("ANALYZE_API_KEY", None)
            try:
                os.chdir(tmp)
                with (
                    patch.object(llm_config, "_REPO_ROOT", Path(tmp)),
                    patch.dict(os.environ, env, clear=True),
                ):
                    load_env_files()
                    self.assertEqual(os.environ["ANALYZE_MODEL"], "from-process")
                    self.assertEqual(os.environ["ANALYZE_API_KEY"], "sk-file")
            finally:
                os.chdir(previous)

    def test_endpoint_status_displays_only_host_and_port(self):
        for url, expected in (
            ("https://viewer:synthetic-password@example.invalid:8443/private?token=synthetic-query", "example.invalid:8443"),
            ("http://viewer:synthetic-password@[::1]:8080/v1", "[::1]:8080"),
            ("synthetic-password", "(invalid endpoint)"),
            ("https://viewer:synthetic-password@[broken", "(invalid endpoint)"),
            ("https://example.invalid:synthetic-password", "(invalid endpoint)"),
        ):
            with self.subTest(url=url), patch.object(llm_config, "resolve_analysis_config", return_value=_cfg(base_url=url)):
                status = config_status_html()
                self.assertIn(expected, status)
                self.assertNotIn("synthetic-password", status)
                self.assertNotIn("synthetic-query", status)
                self.assertNotIn("viewer", status)


#: The brief's sections, in the order ``_compose_brief`` writes them.
_BRIEF_SECTIONS = (
    "SESSION", "HEALTH", "PERFORMANCE", "AGENTS", "FILES", "TOOL_SEQUENCES",
    "BOTTLENECKS", "FAILURES", "WASTE", "ERROR_STEPS", "STEP_INDEX",
)


def _agent_summary(label, *, steps=5, tokens=10_000, duration=10.0, tools=4, errors=0,
                   cache=80.0, gen=40.0, cache_read=8_000):
    """A ``compute_agent_summary`` row with every field the brief and the insights read."""
    return {
        "agent_id": label,
        "label": label,
        "step_count": steps,
        "total_tokens": tokens,
        "total_duration_s": duration,
        "tool_call_count": tools,
        "error_count": errors,
        "cache_efficiency_pct": cache,
        "cache_read_tokens": cache_read,
        "output_tokens_per_sec": gen,
    }


def _brief_with(steps, agents):
    """``_compose_brief`` over *steps* and *agents*, every other input empty."""
    return assistant._compose_brief(
        raw={}, steps=steps, metrics={}, wall_fmt="10m 0s", verdicts=[], agents=agents,
        bottlenecks=[], fail_pats=[], chains=[], chain_metrics={}, streaks=[],
        bash_flags=[], tool_seqs=[], file_interactions=[],
    )


class AnalysisBriefTests(unittest.TestCase):
    def test_brief_includes_task_health_bottlenecks_and_errors(self):
        steps = [
            _step(0, "user", duration=None, tokens=0, preview="Fix the flaky login test"),
            _step(1, duration=40.0, tokens=9000, tools=[
                {"tool_name": "Bash", "status": "error", "error_type": "missing_file",
                 "input": {"command": "cat /nope"}, "output": "No such file or directory"},
            ]),
            _step(2, duration=1.0, tokens=20, tools=[
                {"tool_name": "Read", "status": "success", "input": {"file_path": "app.py"}, "output": "ok"},
            ]),
        ]
        brief = assistant.build_analysis_brief(steps, {"metadata": {"title": "login-flake"}})
        self.assertIn("Fix the flaky login test", brief)
        self.assertIn("HEALTH", brief)
        self.assertIn("BOTTLENECKS", brief)
        self.assertIn("p95_tool_duration", brief)
        self.assertIn("tool_wait_pct", brief)
        self.assertIn("FILES", brief)
        self.assertIn("step 1", brief)
        self.assertIn("missing_file", brief)
        self.assertIn("ERROR_STEPS", brief)
        self.assertIn("login-flake", brief)

    def test_brief_clips_huge_tool_output(self):
        huge = "X" * 5000
        steps = [
            _step(0, tools=[{
                "tool_name": "Bash", "status": "error", "error_type": "tool_error",
                "input": {"command": "yes"}, "output": huge,
            }]),
        ]
        brief = assistant.build_analysis_brief(steps, {})
        self.assertNotIn(huge, brief)
        self.assertLess(len(brief), assistant._BRIEF_CHAR_LIMIT)

    def test_many_agents_leave_room_for_every_section(self):
        # B3: one line per sub-agent let a 257-agent session fill the brief, and
        # the tail cut then dropped every step-level section the model must cite.
        steps = [_step(0, "user", duration=None, tokens=0, preview="Port the parser")]
        steps += [_step(i, duration=1.0, tokens=10) for i in range(1, 80)]
        # The parent ran briefly, so only its being the parent keeps it listed.
        main = _agent_summary("main", steps=40, tokens=500_000, duration=0.5, tools=50,
                              cache_read=450_000)
        subs = [_agent_summary(f"a{i:016x}", duration=float(i)) for i in range(1, 300)]
        subs[0]["error_count"] = 3  # the shortest run: omitted, yet still in the insights
        brief = _brief_with(steps, [main, *subs])

        self.assertEqual([title for title in _BRIEF_SECTIONS if f"\n## {title}\n" not in brief], [])
        section = brief.split("\n## AGENTS\n", 1)[1].split("\n\n## FILES\n", 1)[0].splitlines()
        omitted = [line for line in section if line.startswith("- … ")]
        insights = [line for line in section if line.startswith("- insight: ")]
        listed = [line for line in section if line not in omitted and line not in insights]
        self.assertLessEqual(len(listed), 24)
        self.assertEqual(
            [line.split(":", 1)[0] for line in listed],
            ["- main", *(f"- a{i:016x}" for i in range(299, 276, -1))],
        )
        self.assertEqual(omitted, [
            "- … 276 more sub-agents not listed (showing the 23 longest-running): "
            "1380 assistant steps, 2760000 tok, 1104 tools, 3 errors in total",
        ])
        self.assertEqual(insights, [
            "- insight: Agent a0000000000000001 has 60% error rate (3 errors in 5 steps)",
        ])

    def test_agent_lines_at_or_below_the_cap_are_unchanged(self):
        # Not in duration order, so a cap that re-sorted short lists would show.
        summaries = [
            _agent_summary("main", steps=12, tokens=90_000, duration=120.5, tools=20,
                           cache=85.0, gen=42.5, cache_read=76_500),
            _agent_summary("explore", steps=3, tokens=4_000, duration=15.25, tools=6, errors=1,
                           cache=None, gen=None, cache_read=0),
            _agent_summary("plan", steps=2, tokens=3_000, duration=9.0, tools=2, cache=70.0,
                           gen=12.0, cache_read=2_100),
            _agent_summary("review", steps=4, tokens=2_000, duration=30.0, tools=4, cache=50.0,
                           gen=8.5, cache_read=1_000),
            _agent_summary("fix", steps=1, tokens=1_000, duration=2.0, tools=3, cache=60.0,
                           gen=20.0, cache_read=600),
        ]
        self.assertEqual(assistant._agent_lines(summaries), [
            "- main: 12 assistant steps, 90000 tok, 120.5s, 20 tools, 0 errors, "
            "cache 85.0%, gen 42.5 tok/s",
            "- explore: 3 assistant steps, 4000 tok, 15.25s, 6 tools, 1 errors, "
            "cache n/a, gen n/a tok/s",
            "- plan: 2 assistant steps, 3000 tok, 9.0s, 2 tools, 0 errors, "
            "cache 70.0%, gen 12.0 tok/s",
            "- review: 4 assistant steps, 2000 tok, 30.0s, 4 tools, 0 errors, "
            "cache 50.0%, gen 8.5 tok/s",
            "- fix: 1 assistant steps, 1000 tok, 2.0s, 3 tools, 0 errors, "
            "cache 60.0%, gen 20.0 tok/s",
            "- insight: Agent main consumed 90% of total tokens",
            "- insight: Agent explore has 33% error rate (1 errors in 3 steps)",
        ])

    def test_agents_over_the_cap_list_main_then_the_longest_running(self):
        # OTLP rows: the insights skip them, and they may lack a duration or a
        # count. Unrecorded values rank last and add nothing to the totals.
        long_runs = [_agent_summary(f"long-{i:02d}", duration=100.0 + i) for i in range(22)]
        summaries = [
            _agent_summary("main", duration=0.5),
            _agent_summary("untimed", steps=4, tokens=None, duration=None, tools=3, errors=None),
            _agent_summary("tie-fewer-tokens", steps=2, tokens=200, duration=10.0, tools=1),
            *long_runs,
            _agent_summary("tie-more-tokens", tokens=300, duration=10.0),
            _agent_summary("short", steps=3, tokens=50, duration=1.0, tools=2, errors=1),
        ]
        for row in summaries:
            row["data_quality"] = {"source": "otlp"}

        # Exactly at the cap nothing is re-ordered or folded.
        at_cap = assistant._agent_lines(summaries[:24])
        self.assertEqual([line.split(":", 1)[0] for line in at_cap],
                         [f"- {row['label']}" for row in summaries[:24]])

        lines = assistant._agent_lines(summaries)
        self.assertEqual(
            [line.split(":", 1)[0] for line in lines[:24]],
            ["- main", *(f"- long-{i:02d}" for i in range(21, -1, -1)), "- tie-more-tokens"],
        )
        self.assertEqual(lines[24:], [
            "- … 3 more sub-agents not listed (showing the 23 longest-running): "
            "9 assistant steps, 250 tok, 6 tools, 1 errors in total",
        ])

    def test_flagged_agents_leave_room_for_every_section(self):
        # B3 again, through the insights: each sub-agent here caches nothing and
        # errs on one step in five, so it raises two insight lines. With only the
        # agent lines capped, 200 such sub-agents still cut FILES..STEP_INDEX.
        steps = [_step(0, "user", duration=None, tokens=0, preview="Port the parser")]
        steps += [_step(i, duration=1.0, tokens=10) for i in range(1, 80)]
        main = _agent_summary("main", steps=40, tokens=500_000, duration=0.5, tools=50,
                              cache_read=450_000)
        subs = [_agent_summary(f"a{i:016x}", duration=float(i), errors=1, cache=0.0, cache_read=0)
                for i in range(1, 300)]
        brief = _brief_with(steps, [main, *subs])

        self.assertEqual([title for title in _BRIEF_SECTIONS if f"\n## {title}\n" not in brief], [])
        self.assertNotIn("…[truncated", brief)
        section = brief.split("\n## AGENTS\n", 1)[1].split("\n\n## FILES\n", 1)[0].splitlines()
        insights = [line for line in section if line.startswith("- insight: ")]
        self.assertEqual(len(insights), 50)
        self.assertEqual(section[-1], "- … 548 more insights not listed")

    def test_insight_lines_stop_at_the_cap_and_count_the_rest(self):
        # An agent raises at most two insights (no prompt caching, error rate)
        # and only one agent can take over 60% of the tokens or 70% of the tool
        # calls, so 24 agents raise at most 2 * 24 + 2 = 50. Here main raises
        # all four and every sub-agent two: at the agent cap all 50 are listed.
        def flagged(label, **fields):
            return _agent_summary(label, errors=1, cache=0.0, cache_read=0, **fields)

        summaries = [flagged("main", tokens=1_000_000, tools=1_000, duration=500.0)]
        summaries += [flagged(f"sub-{i:02d}", duration=float(i)) for i in range(1, 25)]

        every = [f"- insight: {text}" for text in assistant.generate_agent_insights(summaries[:24])]
        self.assertEqual(len(every), 50)
        self.assertEqual(assistant._agent_lines(summaries[:24])[24:], every)

        # One more agent: the first 50 stay in order and one line counts the rest.
        every = [f"- insight: {text}" for text in assistant.generate_agent_insights(summaries)]
        self.assertEqual(len(every), 52)
        self.assertEqual(assistant._agent_lines(summaries)[25:],
                         [*every[:50], "- … 2 more insights not listed"])

    def test_truncation_marker_names_every_section_it_cut(self):
        steps = [_step(0, "user", duration=None, tokens=0, preview="Port the parser"),
                 _step(1, duration=2.0, tokens=10)]
        with patch.object(assistant, "_BRIEF_CHAR_LIMIT", 600):
            brief = _brief_with(steps, [])
        self.assertLessEqual(len(brief), 600)
        # SESSION and HEALTH survive whole; PERFORMANCE keeps its header but
        # loses lines, so it is named along with every section after it.
        self.assertIn("\n## HEALTH\n", brief)
        self.assertIn("\n## PERFORMANCE\n", brief)
        marker = (
            "\n…[truncated; sections cut: PERFORMANCE, AGENTS, FILES, TOOL_SEQUENCES, "
            "BOTTLENECKS, FAILURES, WASTE, ERROR_STEPS, STEP_INDEX]\n"
        )
        self.assertEqual(brief[-len(marker):], marker)


class AnalysisChatTests(unittest.TestCase):
    def test_no_trajectory_does_not_call_http(self):
        called = []

        def boom(*_a, **_k):
            called.append(1)
            raise AssertionError("should not call")

        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "ANALYZE_LANGUAGE"}, clear=True):
            history = assistant.answer_question("Where did it go wrong?", [], "", chat_fn=boom)
        self.assertEqual(called, [])
        self.assertEqual(history[0]["role"], "user")
        self.assertIn("Load a trajectory first", history[1]["content"])

    def test_missing_config_does_not_call_http(self):
        called = []
        cfg = _cfg(api_key="", source="missing")
        history = assistant.answer_question(
            "Bottlenecks?", [], "brief here", config=cfg,
            chat_fn=lambda *_a, **_k: called.append(1) or "nope",
        )
        self.assertEqual(called, [])
        self.assertIn(".env", history[-1]["content"])
        self.assertIn("ANALYZE_", history[-1]["content"])

    def test_chat_sends_brief_in_system_and_question_in_messages(self):
        captured = {}

        def fake_chat(config, system, messages):
            captured["config"] = config
            captured["system"] = system
            captured["messages"] = messages
            return "Step 4 is the bottleneck."

        cfg = _cfg()
        history = assistant.answer_question(
            "What is slow?",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "## BOTTLENECKS\n- step 4: 40s",
            config=cfg,
            chat_fn=fake_chat,
        )
        self.assertIn("BOTTLENECKS", captured["system"])
        self.assertEqual(captured["messages"][-1], {"role": "user", "content": "What is slow?"})
        self.assertEqual(history[-1]["content"], "Step 4 is the bottleneck.")
        self.assertEqual(captured["config"].model, "demo-model")

    def test_http_error_does_not_leak_response_body(self):
        response = requests.models.Response()
        response.status_code = 401
        response._content = b'{"error":"sk-secret-should-not-appear"}'

        def boom(*_a, **_k):
            raise requests.HTTPError("401", response=response)

        history = assistant.answer_question(
            "Why?", [], "brief", config=_cfg(), chat_fn=boom,
        )
        self.assertIn("HTTP 401", history[-1]["content"])
        self.assertNotIn("sk-secret", history[-1]["content"])

    def test_setup_help_lists_missing_vars(self):
        text = setup_help_text(_cfg(api_key="", model="", base_url="", source="missing"))
        self.assertIn("ANALYZE_API_KEY", text)
        self.assertIn("LABEL_API_KEY", text)

    def test_openai_payload_uses_bearer_and_system(self):
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(assistant.requests, "post", return_value=FakeResp()) as post:
            text = assistant.complete_chat(
                _cfg(), "SYS", [{"role": "user", "content": "q"}],
            )
        self.assertEqual(text, "ok")
        url = post.call_args[0][0]
        headers = post.call_args.kwargs["headers"]
        body = post.call_args.kwargs["json"]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(body["model"], "demo-model")

    def test_anthropic_request_goes_to_the_v1_messages_endpoint(self):
        # The base without /v1, as the official SDKs take it, must not 404 (DeepSeek serves both).
        cfg = _cfg(provider="anthropic", base_url="https://api.deepseek.com/anthropic")
        reply = _anthropic_reply([{"type": "text", "text": "ok"}], "end_turn")
        with patch.object(assistant.requests, "post", return_value=_FakeResponse(reply)) as post:
            text = assistant._call_anthropic(cfg, "SYS", [{"role": "user", "content": "q"}])
        self.assertEqual(text, "ok")
        self.assertEqual(post.call_args[0][0], "https://api.deepseek.com/anthropic/v1/messages")

    def test_every_prompt_names_the_metrics_to_cite(self):
        for prompt in (assistant.SYSTEM_PROMPT, assistant.system_prompt("Simplified Chinese"),
                       assistant.system_prompt("German")):
            self.assertIn("p95_tool_duration", prompt)
            self.assertIn("tool_wait_pct", prompt)
        self.assertIn("## Main performance bottlenecks", assistant.SYSTEM_PROMPT)
        self.assertIn("## 主要性能瓶颈", assistant.system_prompt("Simplified Chinese"))

    def test_auto_analysis_on_load_sends_the_first_question(self):
        captured = {}

        def fake_chat(_config, system, messages):
            captured["system"] = system
            captured["messages"] = messages
            return "Step 1 is the bottleneck."

        steps = [
            _step(0, "user", duration=None, tokens=0, preview="Fix login"),
            _step(1, duration=12.0, tokens=100),
        ]
        with patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "ANALYZE_LANGUAGE"}, clear=True):
            brief, history = assistant.analyze_loaded_trajectory(
                steps, {"metadata": {"title": "run"}}, config=_cfg(), chat_fn=fake_chat,
            )
        self.assertIn("HEALTH", brief)
        self.assertEqual(captured["messages"][-1]["content"], assistant.AUTO_ANALYSIS_QUESTION)
        self.assertIn("Write the entire reply in English", captured["system"])
        self.assertIn("## Main performance bottlenecks", captured["system"])
        self.assertEqual(history[-1]["content"], "Step 1 is the bottleneck.")
        self.assertTrue(history[-1]["content"])


_PARTIAL = "## Main performance bottlenecks\n\n1. **Step 10** is slow because the model re-reads"
_THINKING = {"type": "thinking", "thinking": "Let me look at the bottlenecks.", "signature": "s"}
_BUDGET_SPENT_ENV = (
    "The model used its entire output budget of 2048 tokens before writing a reply (reasoning "
    "models spend it on hidden thinking). Raise ANALYZE_MAX_TOKENS (or Max output tokens in "
    "Settings), or choose a non-reasoning model."
)
_BUDGET_SPENT_SETTINGS = (
    "The model used its entire output budget of 2048 tokens before writing a reply (reasoning "
    "models spend it on hidden thinking). Raise Max output tokens in Settings › Request options, "
    "or choose a non-reasoning model."
)
_CUT_OFF_ENV = ("⚠ Reply cut off at the output limit of 2048 tokens. "
                "Raise ANALYZE_MAX_TOKENS (or Max output tokens in Settings).")
_CUT_OFF_SETTINGS = ("⚠ Reply cut off at the output limit of 2048 tokens. "
                     "Raise Max output tokens in Settings › Request options.")


class OutputBudgetTests(unittest.TestCase):
    """A reply the output limit stopped says so and names where to raise the limit.

    Reasoning models spend the budget on hidden thinking first, so at a small limit the
    visible reply comes back empty or stops mid-sentence.
    """

    @staticmethod
    def _ask(payload, *, language="English", **cfg_overrides):
        with patch.object(assistant.requests, "post", return_value=_FakeResponse(payload)):
            history = assistant.answer_question(
                "Why?", [], "## HEALTH\n- ok", config=_cfg(**cfg_overrides), language=language,
            )
        return history[-1]["content"]

    def test_openai_budget_spent_on_reasoning_names_the_budget(self):
        wordings = (("session", _BUDGET_SPENT_SETTINGS), ("analyze", _BUDGET_SPENT_ENV))
        for content in (None, "", " \n"):
            for source, expected in wordings:
                with self.subTest(content=content, source=source):
                    reply = self._ask(_openai_reply(content, "length"), source=source)
                    self.assertEqual(reply, expected)

    def test_openai_reply_cut_off_keeps_the_text_and_marks_it(self):
        for source, marker in (("analyze", _CUT_OFF_ENV), ("session", _CUT_OFF_SETTINGS)):
            with self.subTest(source=source):
                reply = self._ask(_openai_reply(_PARTIAL + "   \n", "length"), source=source)
                self.assertEqual(reply, _PARTIAL + "\n\n" + marker)

    def test_anthropic_thinking_only_reply_names_the_budget(self):
        reply = self._ask(_anthropic_reply([_THINKING], "max_tokens"), provider="anthropic",
                          source="session")
        self.assertEqual(reply, _BUDGET_SPENT_SETTINGS)
        self.assertNotIn("ValueError", reply)

    def test_anthropic_reply_cut_off_keeps_the_text_and_marks_it(self):
        blocks = [_THINKING, {"type": "text", "text": _PARTIAL}]
        reply = self._ask(_anthropic_reply(blocks, "max_tokens"), provider="anthropic")
        self.assertEqual(reply, _PARTIAL + "\n\n" + _CUT_OFF_ENV)

    def test_cut_off_marker_follows_the_reply_language(self):
        markers = {
            "session": "⚠ 回复在 2048 个输出 token 的上限处被截断。"
                       "请在 Settings › Request options 中提高 Max output tokens。",
            "analyze": "⚠ 回复在 2048 个输出 token 的上限处被截断。"
                       "请提高 ANALYZE_MAX_TOKENS（或 Settings 中的 Max output tokens）。",
        }
        for source, marker in markers.items():
            with self.subTest(source=source):
                reply = self._ask(_openai_reply(_PARTIAL, "length"), language="Simplified Chinese",
                                  source=source)
                self.assertEqual(reply, _PARTIAL + "\n\n" + marker)

    def test_cut_off_inside_a_code_block_closes_it_before_the_marker(self):
        # Left open, the fence would render the marker as code.
        fenced = "Step 10 re-reads the file:\n\n```python\nfor line in open(path):"
        reply = self._ask(_openai_reply(fenced, "length"))
        self.assertEqual(reply, fenced + "\n```" + "\n\n" + _CUT_OFF_ENV)

    def test_cut_off_after_a_closed_code_block_adds_no_fence(self):
        balanced = "Step 10 runs:\n\n```bash\npytest -q\n```\n\nIt is slow because the model"
        reply = self._ask(_openai_reply(balanced, "length"))
        self.assertEqual(reply, balanced + "\n\n" + _CUT_OFF_ENV)

    def test_replies_that_did_not_reach_the_limit_are_unchanged(self):
        self.assertEqual(self._ask(_openai_reply("Step 4.", "stop")), "Step 4.")
        self.assertEqual(self._ask(_openai_reply("", "stop")),
                         "(The model returned an empty reply.)")
        self.assertEqual(self._ask(_anthropic_reply([_THINKING], "end_turn"), provider="anthropic"),
                         "The analysis model request failed: ValueError.")

    def test_public_error_text_for_the_budget_defaults_to_the_variable_names(self):
        exc = assistant.OutputBudgetExhausted(2048)
        self.assertEqual(assistant._public_http_error(exc), _BUDGET_SPENT_ENV)
        self.assertEqual(assistant._public_http_error(exc, _cfg(source="session")),
                         _BUDGET_SPENT_SETTINGS)


class ErrorRemedyTests(unittest.TestCase):
    """A failed request names the remedy where the connection was configured.

    Settings connections ignore the ANALYZE_* variables, so naming them there is inert advice.
    """

    def test_settings_connection_errors_name_the_settings_fields(self):
        cases = [
            ("timeout", requests.Timeout(),
             "The analysis model timed out. "
             "Retry, or raise Timeout (seconds) in Settings › Request options."),
            ("401", _http_error(401),
             "The analysis API rejected the key (HTTP 401). Check the API key in Settings."),
            ("403", _http_error(403),
             "The analysis API rejected the key (HTTP 403). Check the API key in Settings."),
            ("404", _http_error(404),
             "The analysis API returned HTTP 404. Check the Base URL and Model in Settings."),
            ("500", _http_error(500), "The analysis API returned HTTP 500."),
            ("unreachable", requests.ConnectionError("refused"),
             "Could not reach the analysis API. Check the Base URL in Settings and the network."),
        ]
        for case, exc, expected in cases:
            with self.subTest(case=case):
                history = assistant.answer_question(
                    "Why?", [], "brief", config=_cfg(source="session"), chat_fn=_raising(exc),
                )
                self.assertEqual(history[-1]["content"], expected)

    def test_environment_connection_errors_keep_the_variable_names(self):
        cases = [
            ("timeout", requests.Timeout(),
             "The analysis model timed out. Retry, or raise ANALYZE_TIMEOUT."),
            ("401", _http_error(401),
             "The analysis API rejected the key (HTTP 401). Check ANALYZE_API_KEY."),
            ("403", _http_error(403),
             "The analysis API rejected the key (HTTP 403). Check ANALYZE_API_KEY."),
            ("404", _http_error(404),
             "The analysis API returned HTTP 404. Check ANALYZE_BASE_URL and ANALYZE_MODEL."),
            ("500", _http_error(500), "The analysis API returned HTTP 500."),
            ("unreachable", requests.ConnectionError("refused"),
             "Could not reach the analysis API. Check ANALYZE_BASE_URL and the network."),
        ]
        for case, exc, expected in cases:
            with self.subTest(case=case, config="none"):
                # The one-argument form stays the environment wording.
                self.assertEqual(assistant._public_http_error(exc), expected)
            for source in ("analyze", "label", "mixed"):
                with self.subTest(case=case, source=source):
                    text = assistant._public_http_error(exc, _cfg(source=source))
                    self.assertEqual(text, expected)

    def test_body_read_timeout_is_reported_as_a_timeout(self):
        # requests raises ReadTimeout only while waiting for headers. A provider that sends
        # HTTP 200 headers at once and the body when the model finishes times out in the body
        # read, which requests re-raises as ConnectionError(urllib3 ReadTimeoutError).
        def read_timeout():
            return ReadTimeoutError(None, None, "Read timed out.")

        try:
            try:
                raise read_timeout()
            except ReadTimeoutError as inner:
                # As Response.iter_content raises it: chained implicitly, not with `from`.
                raise requests.ConnectionError(inner)  # noqa: B904
        except requests.ConnectionError as raised:
            as_raised = raised
        in_args = requests.ConnectionError(read_timeout())
        as_cause = requests.ConnectionError("Read timed out.")
        as_cause.__cause__ = read_timeout()
        as_context = requests.ConnectionError("Read timed out.")
        as_context.__context__ = read_timeout()
        deeper = requests.ConnectionError("Read timed out.")
        deeper.__context__ = ProtocolError("Connection broken.")
        deeper.__context__.__cause__ = read_timeout()
        for case, exc in [("as raised", as_raised), ("args", in_args), ("cause", as_cause),
                          ("context", as_context), ("deeper", deeper)]:
            with self.subTest(case=case):
                self.assertEqual(assistant._public_http_error(exc),
                                 "The analysis model timed out. Retry, or raise ANALYZE_TIMEOUT.")
                self.assertEqual(assistant._public_http_error(exc, _cfg(source="session")),
                                 "The analysis model timed out. "
                                 "Retry, or raise Timeout (seconds) in Settings › Request options.")

    def test_other_connection_errors_still_read_as_unreachable(self):
        refused = requests.ConnectionError(ProtocolError("Connection aborted."))
        cyclic = requests.ConnectionError("Connection reset.")
        cyclic.__context__ = ProtocolError("Connection broken.")
        cyclic.__context__.__context__ = cyclic
        for case, exc in [("refused", refused), ("cyclic chain", cyclic)]:
            with self.subTest(case=case):
                self.assertEqual(assistant._public_http_error(exc),
                                 "Could not reach the analysis API. "
                                 "Check ANALYZE_BASE_URL and the network.")


class AnalysisUiTests(unittest.TestCase):
    def test_sidebar_is_wired_in_build_ui(self):
        from agent_trajviz.insight.ui import sidebar

        source = inspect.getsource(sidebar)
        self.assertIn("🤖 AI Trajectory Analysis", source)
        self.assertIn("analyze_loaded_trajectory", source)
        self.assertIn("answer_question", source)
        self.assertNotIn("suggest_btns", source)
        self.assertNotIn("Where did it go wrong?", source)
        self.assertNotIn("analysis-panel-sub", source)
        self.assertNotIn("Simplified Chinese", source)
        from agent_trajviz.insight.styles import APP_CSS
        self.assertIn("#analysis-sidebar .analysis-panel-title", APP_CSS)
        self.assertIn("text-align: center", APP_CSS)
        self.assertIn("color-scheme: light", APP_CSS)
        self.assertNotIn("resizable-chart .plotly-graph-div", APP_CSS)
        self.assertIn("max-height: none !important", APP_CSS)
        self.assertIn("position=\"right\"", source)
        self.assertIn("open=False", source)
        self.assertIn('width="min(560px, calc(100vw - 40px))"', source)
        self.assertIn('content: "AI Trajectory Analysis"', APP_CSS)
        self.assertIn("state_analysis_brief", source)
        self.assertIn("on_analysis_ask", source)
        # Opening the combined Analysis/Settings panel never starts model work.
        self.assertIn("on_sidebar_expand", source)
        self.assertIn("analysis_sidebar.expand", source)
        self.assertNotIn("state_sidebar_open", source)
        self.assertIn("on_trajectory_loaded", source)
        self.assertIn("on_analyze_trajectory", source)
        self.assertIn("analysis_run.click", source)
