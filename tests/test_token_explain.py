import unittest

from agent_trajviz.insight.token_explain import (
    RADIX_PAGE_TOKENS,
    build_token_step_explanation_html,
    build_token_step_panel_html,
)


def _step(role="assistant", total=0, inp=0, output=0, cache_read=0, reasoning=0,
          tool_calls=None, parts=None, text_preview="", model_id="test-model",
          session_id="", cache_write=0, provider_id=""):
    return {
        "index": 0, "raw_index": 0, "role": role, "session_id": session_id,
        "tokens": {
            "total": total, "input": inp, "output": output,
            "cache_read": cache_read, "cache_write": cache_write, "reasoning": reasoning,
        },
        "tool_calls": tool_calls or [], "parts": parts or [],
        "text_preview": text_preview, "model_id": model_id, "provider_id": provider_id, "finish": "stop",
    }


class TokenStepExplanationTests(unittest.TestCase):
    def test_user_step_has_no_tokens(self):
        steps = [_step(role="user", text_preview="把 report.md 输出给我")]
        html_out = build_token_step_explanation_html(steps, 0)
        self.assertIn("没有 token 记录", html_out)
        self.assertIn("user", html_out)

    def test_out_of_range_index_is_graceful(self):
        self.assertIn("未找到该步骤", build_token_step_explanation_html([_step()], 5))
        self.assertIn("未找到该步骤", build_token_step_explanation_html([_step()], -1))

    def test_first_call_lists_user_message_and_estimates_other_input(self):
        steps = [
            _step(role="user", text_preview="帮我做六步编码任务"),
            _step(total=19270, inp=8373, output=303, reasoning=10594, cache_read=0),
        ]
        html_out = build_token_step_explanation_html(steps, 1)
        self.assertIn("首次 LLM 调用", html_out)
        self.assertIn("无前缀缓存命中", html_out)
        self.assertIn("8,373", html_out)
        # The user message is estimable (9 CJK chars ≈ 6 tok + 3 wrapper)...
        self.assertIn("用户消息 ≈ 6 tok + 包装 3 tok", html_out)
        self.assertIn("其他输入 ≈ 8,364 tok", html_out)
        self.assertIn("may include system prompts and tool definitions", html_out)

    def test_second_call_decomposes_fresh_input(self):
        # Mirrors the verified sglang numbers: prev total 19270 = 75*256 + 70,
        # so the tail re-prefill is 70 and the new content is 622-70 = 552.
        # The tool result is Chinese prose (770 CJK chars ≈ 550 tok at
        # 1.4 chars/token); the framework is 5 (batch wrapper) + 8 (one
        # response pair) = 13. Estimates sum to 563 vs the actual 552 —
        # listed openly instead of being forced to match.
        prev = _step(
            total=19270, inp=8373, output=303, reasoning=10594, cache_read=0,
            tool_calls=[{
                "tool_name": "read",
                "input": {"path": "notes.md"},
                "output": "中" * 770,
                "status": "ok",
            }],
        )
        cur = _step(
            total=20136, inp=622, output=99, reasoning=215, cache_read=19200,
            provider_id="sglang",
            tool_calls=[{
                "tool_name": "bash",
                "input": {"command": "python3 bisect_demo.py"},
                "output": "ok",
                "status": "ok",
            }],
            parts=[{"type": "text", "text": "测试通过"}],
        )
        html_out = build_token_step_explanation_html([prev, cur], 1)
        self.assertIn("页尾重算 ≈ 70", html_out)
        self.assertIn("新增内容 ≈ 552（估算）", html_out)
        self.assertIn("read(notes.md)", html_out)
        self.assertIn("550 tok", html_out)
        self.assertIn("消息框架 ≈ 13 tok", html_out)
        self.assertIn("估算合计 ≈ 563", html_out)
        self.assertIn("相差 11", html_out)
        self.assertIn("不必相等", html_out)
        # Cache read: 19200 = 75 pages x 256.
        self.assertIn("75 页", html_out)
        self.assertIn("19,200", html_out)
        # Output attribution: text, tool args, and the call syntax tags.
        self.assertIn("正文 4 字符 ≈ 3 tok", html_out)
        self.assertIn("bash(python3 bisect_demo.py) 参数 ≈ 15 tok", html_out)
        self.assertIn("调用语法标签 ≈ 27 tok", html_out)
        # Total summary line keeps the prompt/completion split.
        self.assertIn("prompt 19,822", html_out)
        self.assertIn("completion 314", html_out)

    def test_user_message_between_calls_is_attributed(self):
        prev = _step(total=1000, inp=1000, output=0, reasoning=0, cache_read=0)
        user = _step(role="user", text_preview="y" * 400)
        cur = _step(total=1500, inp=200, output=0, reasoning=0, cache_read=1000)
        html_out = build_token_step_explanation_html([prev, user, cur], 2)
        self.assertIn("用户新消息 ≈ 160 tok", html_out)

    def test_all_items_listed_without_forcing_the_sum(self):
        # A big gap between the estimates (100 + 13 framework) and the actual
        # new content (400) must be shown as an open difference, not hidden.
        prev = _step(
            total=1024, inp=1024, output=0, reasoning=0, cache_read=0,
            tool_calls=[{
                "tool_name": "read",
                "input": {"path": "big.md"},
                "output": "x" * 250,  # ≈ 100 tok
                "status": "ok",
            }],
        )
        cur = _step(total=1424, inp=400, output=0, reasoning=0, cache_read=1024, provider_id="sglang")
        html_out = build_token_step_explanation_html([prev, cur], 1)
        self.assertIn("read(big.md) 的执行结果 ≈ 100 tok", html_out)
        self.assertIn("消息框架 ≈ 13 tok", html_out)
        self.assertIn("估算合计 ≈ 113", html_out)
        self.assertIn("相差 287", html_out)

    def test_no_new_content_between_calls_is_called_out(self):
        prev = _step(total=1024, inp=1024, output=0, reasoning=0, cache_read=0)
        cur = _step(total=1424, inp=400, output=0, reasoning=0, cache_read=1024)
        html_out = build_token_step_explanation_html([prev, cur], 1)
        self.assertIn("没有工具结果或新用户消息", html_out)

    def test_panel_embeds_one_hidden_div_per_step_and_the_method(self):
        steps = [
            _step(role="user", text_preview="hi"),
            _step(total=100, inp=100, output=0, reasoning=0, cache_read=0),
        ]
        panel = build_token_step_panel_html(steps)
        self.assertIn("tv-token-hint", panel)
        self.assertIn("data-step='0'", panel)
        self.assertIn("data-step='1'", panel)
        self.assertIn("display:none", panel)
        self.assertEqual(2, panel.count("tv-token-step-detail"))
        # The estimation method stays visible below whichever step is shown.
        self.assertIn("tv-token-method", panel)
        self.assertIn("估算方法", panel)
        self.assertIn("basic CJK ideographs (U+4E00–U+9FFF) ÷ 1.4", panel)
        self.assertIn("once for an assistant message with tool calls", panel)

    def test_empty_panel_is_just_the_hint(self):
        panel = build_token_step_panel_html([])
        self.assertIn("tv-token-hint", panel)
        self.assertIn("点击上方 Token Usage", panel)
        self.assertIn("Click a bar to see that step", panel)
        self.assertNotIn("data-step", panel)
        self.assertNotIn("tv-token-method", panel)

    def test_indent_levels_match_the_hierarchy(self):
        # 新增内容 and 页尾重算 are siblings (level 1); the per-item details
        # nest one level deeper (level 2). HTML collapses plain spaces, so
        # the indentation must be &nbsp; based.
        from agent_trajviz.insight.token_explain import _IND1, _IND2

        prev = _step(
            total=19270, inp=8373, output=303, reasoning=10594, cache_read=0,
            tool_calls=[{
                "tool_name": "read",
                "input": {"path": "notes.md"},
                "output": "中" * 770,
                "status": "ok",
            }],
        )
        cur = _step(total=20136, inp=622, output=99, reasoning=215, cache_read=19200, provider_id="sglang")
        html_out = build_token_step_explanation_html([prev, cur], 1)
        self.assertIn(f'{_IND1}<span data-tv-i18n="" data-tv-i18n-en="- New content ≈ 552', html_out)
        self.assertIn(f'{_IND1}<span data-tv-i18n="" data-tv-i18n-en="- Partial page recomputation ≈ 70', html_out)
        self.assertIn(f'{_IND2}<span data-tv-i18n="" data-tv-i18n-en="- read(notes.md) result', html_out)
        self.assertIn(f'{_IND2}<span data-tv-i18n="" data-tv-i18n-en="- Message framing ≈ 13 tok', html_out)
        self.assertIn(f'{_IND2}<span data-tv-i18n="" data-tv-i18n-en="(Estimated sum', html_out)

    def test_explanation_embeds_workflow_jump_link(self):
        # The panel links each step to its Workflow card via the shared
        # tvGotoWorkflowStep path (same as Issues/overview badges).
        steps = [
            _step(role="user", text_preview="hi"),
            _step(total=100, inp=100, output=0, reasoning=0, cache_read=0),
        ]
        html_out = build_token_step_explanation_html(steps, 1)
        self.assertIn("insight-step-link", html_out)
        self.assertIn("在 Workflow 中查看该步", html_out)
        self.assertIn("tvGotoWorkflowStep(1)", html_out)
        # Fallback path targets the same card id as the Workflow tab renders.
        self.assertIn("wf-card-1", html_out)
        # User steps without tokens get no panel, hence no link.
        self.assertNotIn("insight-step-link",
                         build_token_step_explanation_html(steps, 0))

    def test_page_size_constant(self):
        # The decomposition math is only valid for whole-page prefix matching.
        self.assertEqual(256, RADIX_PAGE_TOKENS)

    def test_estimator_is_language_aware(self):
        from agent_trajviz.insight.token_explain import _est_tokens

        # Chinese prose packs ~1.4 chars/token, code/ASCII ~2.5 chars/token;
        # a single-rate estimate is off by ~2x on Chinese-heavy output.
        self.assertEqual(100, _est_tokens("中" * 140))
        self.assertEqual(100, _est_tokens("x" * 250))
        # Mixed content sums the two rates.
        self.assertEqual(150, _est_tokens("中" * 140 + "x" * 125))
        # Short strings still count at least one token.
        self.assertEqual(1, _est_tokens("a"))
        self.assertEqual(0, _est_tokens(""))
        self.assertEqual(0, _est_tokens(None))

    def test_subagent_steps_stay_within_their_own_conversation(self):
        # Interleaved root + subagent conversations: the subagent's steps
        # must not treat root steps as their "previous step" — different
        # prefix, different cache lineage.
        steps = [
            _step(role="user", text_preview="主会话用户消息", session_id="s-main"),
            _step(total=19270, inp=8373, output=303, reasoning=10594,
                  cache_read=0, session_id="s-main"),
            _step(total=5000, inp=5000, output=200, reasoning=0,
                  cache_read=0, session_id="s-sub"),
            _step(total=6000, inp=1000, output=300, reasoning=0,
                  cache_read=5000, session_id="s-sub"),
            _step(total=20136, inp=622, output=100, reasoning=0,
                  cache_read=19200, session_id="s-main"),
        ]
        for i, s in enumerate(steps):
            s["index"] = i

        # Subagent first call: first-call treatment, no reference to the
        # root conversation's context or user message.
        sub_first = build_token_step_explanation_html(steps, 2)
        self.assertIn("首次 LLM 调用", sub_first)
        self.assertNotIn("19,270", sub_first)
        self.assertNotIn("主会话用户消息", sub_first)

        # Subagent second call: cache lineage points at its own previous
        # step (5,000), not the root's (19,270).
        sub_second = build_token_step_explanation_html(steps, 3)
        self.assertIn("5,000", sub_second)
        self.assertNotIn("19,270", sub_second)
        self.assertNotIn("20,136", sub_second)

        # Root steps keep referencing root steps.
        root_second = build_token_step_explanation_html(steps, 4)
        self.assertIn("19,270", root_second)

    def test_agent_mode_switch_within_one_session_stays_one_conversation(self):
        # opencode plan -> build is a mode switch inside ONE session: the
        # context is shared, so the build step's cache lineage must point at
        # the plan step even though the agent field differs.
        steps = [
            _step(total=10000, inp=10000, output=0, reasoning=0,
                  cache_read=0, session_id="s1"),
            _step(total=11000, inp=1000, output=500, reasoning=500,
                  cache_read=10000, session_id="s1"),
        ]
        steps[0]["agent"] = "plan"
        steps[1]["agent"] = "build"
        for i, s in enumerate(steps):
            s["index"] = i

        html_out = build_token_step_explanation_html(steps, 1)
        self.assertNotIn("首次 LLM 调用", html_out)
        self.assertIn("10,000", html_out)

    def test_cache_writes_are_prompt_tokens_and_reasoning_is_disjoint(self):
        claude = _step(total=37, inp=10, output=5, cache_read=20, cache_write=2)
        codex = _step(total=13, inp=10, output=3, cache_read=4, reasoning=2)
        claude_html = build_token_step_explanation_html([claude], 0)
        codex_html = build_token_step_explanation_html([codex], 0)
        self.assertIn("Cache Write", claude_html)
        self.assertIn("Total 37 = prompt 32 (10 fresh + 20 read + 2 written) + completion 5", claude_html)
        self.assertIn("Total 13 = prompt 10 (6 fresh + 4 read) + completion 3 (1 output + 2 reasoning)", codex_html)

    def test_unknown_provider_does_not_claim_sglang_page_recomputation(self):
        prev = _step(total=19270, inp=19270)
        cur = _step(total=20136, inp=622, output=314, cache_read=19200, provider_id="anthropic")
        html_out = build_token_step_explanation_html([prev, cur], 1)
        self.assertNotIn("页尾重算", html_out)
        self.assertNotIn("75 页", html_out)
        self.assertIn("19,200", html_out)

    def test_total_only_and_partial_token_telemetry_have_no_invented_breakdown(self):
        total_only = build_token_step_explanation_html([_step(total=50)], 0)
        self.assertIn("仅记录总量 50 token", total_only)
        partial = _step(total=50, inp=30, output=20)
        partial["_otel"] = {"fields": {"tokens.input": "observed", "tokens.output": "observed"}}
        partial_html = build_token_step_explanation_html([partial], 0)
        self.assertIn("没有 token 记录可用于构成分析", partial_html)
        self.assertNotIn("无前缀缓存命中", partial_html)


if __name__ == "__main__":
    unittest.main()
