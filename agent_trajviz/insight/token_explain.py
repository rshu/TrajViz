"""Per-step token breakdown explanations for the Token Usage chart.

Clicking a bar in the Summary section's Token Usage chart renders a
human-readable decomposition of that step's numbers: where the fresh input
came from (page-tail recomputation plus the previous step's tool results),
what the cache read covers, and what output/reasoning represent.

Usage segments follow the same accounting as the chart. Text estimates
are marked ≈. A page-tail estimate is shown only when the recorded SGLang
provider and matching cache counts support that calculation.
"""

from __future__ import annotations

import html
from typing import Any

from ..i18n import paired_text
from .parser import step_token_segments, usable_token_count
from .shell_cmd import tool_call_hint
from .telemetry_quality import field_observed
from .timeline_agents import timeline_agent_id_of

# sglang's radix cache matches prefixes in whole pages; 256 is the page size
# of the deployment these explanations were verified against.
RADIX_PAGE_TOKENS = 256

# Tool outputs are not billed separately by OpenAI-compatible APIs, so their
# token weight is estimated from text length. A single chars/token rate is
# off by ~2x between Chinese prose and code/ASCII, so the estimate mixes two
# rates by character class. Both were least-squares-fitted against a real
# Qwen tokenizer over a full session's tool outputs (aggregate error <5%);
# the project cannot ship per-model tokenizers, so this is as close as it
# gets without one.
CJK_CHARS_PER_TOKEN = 1.4
OTHER_CHARS_PER_TOKEN = 2.5

# Chat-template wrapper overhead, measured with a real tokenizer on the
# qwen3.5 template: consecutive tool results are merged into one user
# message (~5 tok header/footer + ~8 tok per response tag pair), a plain
# user message costs ~3 tok of role tags, and each tool call costs ~24 tok
# of function/parameter syntax (varies with the parameter count) plus ~3
# tok of assistant message header/footer.
TOOL_RESULT_WRAPPER_TOKENS = 8
TOOL_BATCH_WRAPPER_TOKENS = 5
USER_MSG_WRAPPER_TOKENS = 3
TOOL_CALL_SYNTAX_TOKENS = 24
ASSISTANT_MSG_WRAPPER_TOKENS = 3

_HINT_EN = "Click a bar to see that step's token breakdown."
_HINT_ZH = "点击上方 Token Usage 图表的柱子，查看该步 token 构成。"

# HTML collapses runs of plain spaces, so indentation must use &nbsp;.
# Level 1: top-level items (新增内容 / 页尾重算 / 首调构成项).
# Level 2: details nested under a level-1 item.
_IND1 = "&nbsp;&nbsp;"
_IND2 = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

_METHOD_DETAILS_HTML = (
    "<details class='tv-token-method' style='margin-top:6px;font-size:0.85em;"
    "color:var(--ov-muted);'>"
    f"<summary style='cursor:pointer;user-select:none;'>{paired_text('How the estimates are calculated', '估算方法（带 ≈ 的数值如何得出）')}</summary>"
    "<div style='padding:4px 0 2px 14px;line-height:1.7;'>"
    + "<br>".join((
        paired_text(
            "Values marked ≈ are estimates based on the available content. The components can add up to a different count than the provider reports.",
            "带 ≈ 的项根据可见内容估算，各项合计可能与提供商记录的用量不同。",
        ),
        paired_text(
            f"1. Text: basic CJK ideographs (U+4E00–U+9FFF) ÷ {CJK_CHARS_PER_TOKEN} plus other characters ÷ {OTHER_CHARS_PER_TOKEN}. "
            "The result is rounded to a whole number, with a minimum of 1 for nonempty text.",
            f"1. 文本：基本中日韩统一表意文字（U+4E00–U+9FFF）数量 ÷ {CJK_CHARS_PER_TOKEN} + 其他字符数 ÷ {OTHER_CHARS_PER_TOKEN}。"
            "结果取最接近的整数，非空文本至少计 1 个 token。",
        ),
        paired_text(
            f"2. Input framing adds about {TOOL_RESULT_WRAPPER_TOKENS} tokens per listed tool result, "
            f"{TOOL_BATCH_WRAPPER_TOKENS} for the group of results from the previous step, "
            f"and {USER_MSG_WRAPPER_TOKENS} per user message. Output framing adds about {TOOL_CALL_SYNTAX_TOKENS} per tool call "
            f"plus {ASSISTANT_MSG_WRAPPER_TOKENS} once for an assistant message with tool calls.",
            f"2. 输入框架按每条列出的工具结果约 {TOOL_RESULT_WRAPPER_TOKENS} 个 token、上一步的整组结果约 {TOOL_BATCH_WRAPPER_TOKENS} 个、"
            f"每条用户消息约 {USER_MSG_WRAPPER_TOKENS} 个估算。输出框架按每次工具调用约 {TOOL_CALL_SYNTAX_TOKENS} 个估算，"
            f"有工具调用的助手消息另计一次约 {ASSISTANT_MSG_WRAPPER_TOKENS} 个。",
        ),
        paired_text(
            f"3. For recorded SGLang calls, a partial cache page is estimated only when the cache-hit count matches "
            f"the previous call’s full pages at {RADIX_PAGE_TOKENS} tokens per page.",
            f"3. 若提供商记录为 SGLang 且命中量与上一步整页数一致，则用 {RADIX_PAGE_TOKENS} token/页估算缓存页尾重算。",
        ),
    ))
    + "</div></details>"
)


def build_token_step_panel_html(steps: list[dict]) -> str:
    """The full click-target panel: visible hint + one hidden div per step.

    The dashboard binds a client-side plotly_click handler (see
    ``tvBindTokenStepExplain`` in ``insight.py``) that reveals the
    ``[data-step]`` div matching the clicked bar, so no server round-trip
    is needed per click. The estimation-method note sits below the step
    divs so it stays visible whichever step is shown.
    """
    hint = f"<div class='tv-token-hint'>{paired_text(_HINT_EN, _HINT_ZH)}</div>"
    if not steps:
        return hint
    details = [
        f"<div class='tv-token-step-detail' data-step='{i}' style='display:none;'>"
        f"{_render_token_step(step, i, previous, users)}"
        f"</div>"
        for i, (step, (previous, users)) in enumerate(zip(steps, _step_contexts(steps), strict=True))
    ]
    return hint + "".join(details) + _METHOD_DETAILS_HTML


def _est_tokens(text: Any) -> int:
    """Estimate the token count of *text* without the model's tokenizer."""
    if text is None:
        return 0
    s = text if isinstance(text, str) else str(text)
    if not s:
        return 0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    other = len(s) - cjk
    return max(1, round(cjk / CJK_CHARS_PER_TOKEN + other / OTHER_CHARS_PER_TOKEN))


def _fmt(n: int) -> str:
    return f"{n:,}"


def _esc(s: Any) -> str:
    # quote=True: payloads must not survive even inside attribute context.
    return html.escape(str(s), quote=True)


def _tool_hint(tc: dict) -> str:
    """Short human hint for a tool call: name + first argument line."""
    name = tc.get("tool_name") or tc.get("name") or "?"
    hint = tool_call_hint(tc)
    if len(hint) > 48:
        hint = hint[:45] + "…"
    return f"{name}({hint})" if hint else str(name)


def _step_contexts(steps: list[dict]) -> list[tuple[dict | None, list[dict]]]:
    """Index preceding model turns and new user messages once per panel."""
    agent_id = timeline_agent_id_of(steps)
    previous: dict[tuple[str, str], dict] = {}
    pending: dict[tuple[str, str], list[dict]] = {}
    contexts: list[tuple[dict | None, list[dict]]] = []
    for step in steps:
        session_id = str(step.get("session_id") or "")
        # OpenCode can change mode within one session. Claude Code can share
        # a session ID between distinct tagged subagents.
        subagent = str(step.get("agent") or "") if step.get("is_sub_agent") and "parent_agent_id" in step else ""
        key = (session_id, subagent if session_id else agent_id(step))
        role = step.get("role")
        total = usable_token_count((step.get("tokens") or {}).get("total"))
        if role == "user":
            pending.setdefault(key, []).append(step)
        if role != "assistant":
            contexts.append((None, []))
        elif not field_observed(step, "tokens.total"):
            # A partially recorded call breaks the history needed to compare
            # input with the preceding observed model turn.
            previous.pop(key, None)
            pending.pop(key, None)
            contexts.append((None, []))
        elif total:
            contexts.append((previous.get(key), pending.pop(key, [])))
            previous[key] = step
        else:
            contexts.append((None, []))
    return contexts


def _page_tail(step: dict, previous: dict, cache_read: int) -> int | None:
    """Estimate the partial page when both the provider and counts match."""
    if "sglang" not in str(step.get("provider_id") or "").lower():
        return None
    previous_total = previous["tokens"]["total"]
    tail = previous_total % RADIX_PAGE_TOKENS
    return tail if cache_read > 0 and cache_read == previous_total - tail else None


def _sum_note(
    est_total: int,
    reference: int,
    ind: str = _IND2,
    *,
    label: tuple[str, str] = ("recorded count", "记录值"),
) -> str:
    """Compare independent estimates with the recorded or inferred reference."""
    english, chinese = label
    if est_total == reference:
        return ind + paired_text(
            f"(Estimated sum ≈ {_fmt(est_total)}, matching the {english})",
            f"（估算合计 ≈ {_fmt(est_total)}，与{chinese}一致）",
        )
    difference = _fmt(abs(est_total - reference))
    return ind + paired_text(
        f"(Estimated sum ≈ {_fmt(est_total)}. The {english} is {_fmt(reference)}, a difference of {difference}.)",
        f"（估算合计 ≈ {_fmt(est_total)}，{chinese} {_fmt(reference)}，相差 {difference}，各项独立、不必相等）",
    )


def _first_call_lines(users: list[dict], fresh_input: int) -> list[str]:
    lines = [f"<b>{_fmt(fresh_input)}</b> — " + paired_text(
        "Fresh input recorded for the first LLM call. Possible components:",
        "首次 LLM 调用记录中的新输入。可能构成：",
    )]
    est_total = 0
    for s in users:
        est = _est_tokens(s.get("text_preview"))
        if est <= 0:
            continue
        est_total += est + USER_MSG_WRAPPER_TOKENS
        lines.append(
            _IND1 + paired_text(
                f"- User message ≈ {_fmt(est)} tok + {_fmt(USER_MSG_WRAPPER_TOKENS)} tok of framing (estimate)",
                f"- 用户消息 ≈ {_fmt(est)} tok + 包装 {_fmt(USER_MSG_WRAPPER_TOKENS)} tok（估算）",
            )
        )
    rest = max(0, fresh_input - est_total)
    lines.append(
        _IND1 + paired_text(
            f"- Other input ≈ {_fmt(rest)} tok (remainder after estimated user messages and framing, may include system prompts and tool definitions)",
            f"- 其他输入 ≈ {_fmt(rest)} tok（扣除用户消息及其框架的估算后所得，可能包含系统提示和工具定义）",
        )
    )
    return lines


def _fresh_input_lines(previous: dict | None, users: list[dict], fresh_input: int, tail: int | None) -> list[str]:
    if previous is None:
        return _first_call_lines(users, fresh_input)

    prev_total = previous["tokens"]["total"]
    lines = [f"<b>{_fmt(fresh_input)}</b> — " + paired_text(
        "Fresh input tokens processed for this call:",
        "本步需重新计算（prefill）的输入，构成如下：",
    )]

    # Tool results and user messages are candidates for newly added context.
    # Each item carries an estimate independent of the recorded usage.
    new_content = None if tail is None or tail > fresh_input else fresh_input - tail
    if new_content is not None:
        lines.append(_IND1 + paired_text(
            f"- New content ≈ {_fmt(new_content)} (estimate after subtracting the likely partial page):",
            f"- 新增内容 ≈ {_fmt(new_content)}（估算）：扣除可能重算的页尾后，逐项估算：",
        ))
    else:
        lines.append(_IND1 + paired_text(
            "- Estimated content added between calls:", "- 两次调用之间可能进入上下文的内容，逐项估算：",
        ))
    est_total = 0
    n_tools = 0
    for tc in previous.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        est = _est_tokens(tc.get("output"))
        est_total += est
        n_tools += 1
        hint = _tool_hint(tc)
        lines.append(_IND2 + paired_text(
            f"- {hint} result ≈ {_fmt(est)} tok (estimate)", f"- {hint} 的执行结果 ≈ {_fmt(est)} tok（估算）",
        ))
    n_users = 0
    for s in users:
        est = _est_tokens(s.get("text_preview"))
        est_total += est
        n_users += 1
        lines.append(_IND2 + paired_text(
            f"- New user message ≈ {_fmt(est)} tok (estimate)", f"- 用户新消息 ≈ {_fmt(est)} tok（估算）",
        ))
    framework = 0
    if n_tools:
        framework += TOOL_BATCH_WRAPPER_TOKENS + n_tools * TOOL_RESULT_WRAPPER_TOKENS
    if n_users:
        framework += n_users * USER_MSG_WRAPPER_TOKENS
    if framework:
        est_total += framework
        lines.append(
            _IND2 + paired_text(
                f"- Message framing ≈ {_fmt(framework)} tok (role markers and template tags)",
                f"- 消息框架 ≈ {_fmt(framework)} tok（角色标记与模板标签，估算）",
            )
        )
    if (n_tools or n_users) and new_content is not None:
        lines.append(_sum_note(est_total, new_content, label=("new-content estimate", "新增内容估算")))
    elif not (n_tools or n_users):
        lines.append(
            _IND2 + paired_text(
                "- No tool results or new user messages appeared between calls. Framing or context changes may explain the new tokens.",
                "- 两步之间没有工具结果或新用户消息 — 新增内容可能来自消息框架、系统层注入或上下文重组",
            )
        )

    # Tail last: NOT new content. The radix cache reuses whole pages only, so
    # the previous sequence's last incomplete page is re-prefilled even though
    # the model already saw it last turn.
    if tail and new_content is not None:
        pages = prev_total // RADIX_PAGE_TOKENS
        lines.append(
            _IND1 + paired_text(
                f"- Partial page recomputation ≈ {_fmt(tail)} tok. The previous call's {_fmt(prev_total)} tokens cover "
                f"{pages} full {RADIX_PAGE_TOKENS}-token pages.",
                f"- 页尾重算 ≈ {_fmt(tail)}：按 {RADIX_PAGE_TOKENS} token/页估算，"
                f"上一步 {_fmt(prev_total)} token 对应 {pages} 整页（{_fmt(pages * RADIX_PAGE_TOKENS)}）",
            )
        )
    return lines


def _cache_read_lines(previous: dict | None, cache_read: int, tail: int | None) -> list[str]:
    if cache_read <= 0:
        return ["<b>0</b> — " + paired_text("No prefix cache hits", "无前缀缓存命中")]
    if previous is None:
        return [f"<b>{_fmt(cache_read)}</b> — " + paired_text("Prefix cache hits", "前缀缓存命中")]
    prev_total = previous["tokens"]["total"]
    if tail is None:
        return [f"<b>{_fmt(cache_read)}</b> — " + paired_text(
            f"Prefix cache hits. The previous call recorded {_fmt(prev_total)} tokens.",
            f"前缀缓存命中。上一次调用记录 {_fmt(prev_total)} token。",
        )]
    pages = cache_read // RADIX_PAGE_TOKENS
    return [f"<b>{_fmt(cache_read)}</b> — " + paired_text(
        f"The previous call recorded {_fmt(prev_total)} tokens. The hit count matches {pages} pages of {RADIX_PAGE_TOKENS} tokens.",
        f"上一次调用记录 {_fmt(prev_total)} token，命中量对应 {pages} 页 × {RADIX_PAGE_TOKENS}",
    )]


def _output_lines(step: dict, output: int) -> list[str]:
    if output <= 0:
        return ["<b>0</b> — " + paired_text("No visible output", "本步没有可见输出")]
    lines = [f"<b>{_fmt(output)}</b> — " + paired_text(
        "Recorded model output, with estimated components:", "记录的模型输出，逐项估算：",
    )]
    est_total = 0
    text = "".join(
        p.get("text") or ""
        for p in step.get("parts") or []
        if isinstance(p, dict) and p.get("type") == "text"
    )
    if text:
        est = _est_tokens(text)
        est_total += est
        lines.append(_IND1 + paired_text(
            f"- Text of {_fmt(len(text))} characters ≈ {_fmt(est)} tok (estimate)",
            f"- 正文 {_fmt(len(text))} 字符 ≈ {_fmt(est)} tok（估算）",
        ))
    n_calls = 0
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        est = _est_tokens(tc.get("input"))
        est_total += est
        n_calls += 1
        hint = _tool_hint(tc)
        lines.append(_IND1 + paired_text(
            f"- {hint} arguments ≈ {_fmt(est)} tok (estimate)", f"- {hint} 参数 ≈ {_fmt(est)} tok（估算）",
        ))
    if n_calls:
        syntax = n_calls * TOOL_CALL_SYNTAX_TOKENS + ASSISTANT_MSG_WRAPPER_TOKENS
        est_total += syntax
        lines.append(
            _IND1 + paired_text(
                f"- Tool call syntax ≈ {_fmt(syntax)} tok: {_fmt(TOOL_CALL_SYNTAX_TOKENS)} per call "
                f"plus {_fmt(ASSISTANT_MSG_WRAPPER_TOKENS)} for message framing (estimate)",
                f"- 调用语法标签 ≈ {_fmt(syntax)} tok：每次调用 {_fmt(TOOL_CALL_SYNTAX_TOKENS)}"
                f" + 消息头尾 {_fmt(ASSISTANT_MSG_WRAPPER_TOKENS)}（估算）",
            )
        )
    lines.append(_sum_note(est_total, output, ind=_IND1))
    return lines


def build_token_step_explanation_html(steps: list[dict], step_index: int) -> str:
    """Render the per-step token decomposition as HTML for the clicked bar."""
    if not steps or not (0 <= step_index < len(steps)):
        return f"<div style='padding:0.6em;color:var(--ov-muted);'>{paired_text('Step not found.', '未找到该步骤。')}</div>"

    previous, users = _step_contexts(steps)[step_index]
    return _render_token_step(steps[step_index], step_index, previous, users)


def _render_token_step(step: dict, step_index: int, previous: dict | None, users: list[dict]) -> str:
    tokens = step.get("tokens") or {}
    total = usable_token_count(tokens.get("total", 0))
    role = step.get("role") or "?"
    if not total or not all(field_observed(step, f"tokens.{field}")
                            and usable_token_count(tokens.get(field, 0)) is not None
                            for field in ("total", "input", "output", "cache_read", "cache_write")):
        description = paired_text(
            f"Step {step_index} ({role}) has no recorded token breakdown.",
            f"第 {step_index} 步（{role}）没有 token 记录可用于构成分析。",
        )
        return f"<div style='padding:0.6em;color:var(--ov-muted);'>{description}</div>"
    segments = step_token_segments(tokens)
    if segments["reading"] == "total_only":
        description = paired_text(
            f"Step {step_index} records only the total of {_fmt(total)} tokens.",
            f"第 {step_index} 步仅记录总量 {_fmt(total)} token。",
        )
        return f"<div style='padding:0.6em;'>{description}</div>"
    fresh_input = segments["fresh"]
    cache_read = segments["cache_read"]
    cache_write = segments["cache_write"]
    reasoning = segments["reasoning"]
    output = segments["output"]
    prompt = fresh_input + cache_read + cache_write
    completion = output + reasoning
    tail = _page_tail(step, previous, cache_read) if previous is not None else None

    model = step.get("model_id") or ""
    header = paired_text(f"Step {step_index}", f"第 {step_index} 步") + f" · {_esc(role)}"
    if model:
        header += f" · {_esc(model)}"
    # Imported lazily: rendering pulls in pygments, and core analysis modules
    # import this file for the token estimator (they must stay dependency-free).
    from .rendering import _diag_jump_onclick

    jump = (
        f"<span class='insight-step-link' onclick=\"{_diag_jump_onclick(step_index)}\" "
        f"style='margin-left:10px;font-weight:400;'>{paired_text('Open step in Workflow →', '在 Workflow 中查看该步 →')}</span>"
    )

    sections: list[tuple[str, list[str]]] = [
        (paired_text("Cache Read", "缓存命中输入"), _cache_read_lines(previous, cache_read, tail)),
        (paired_text("Fresh Input", "新增输入"), _fresh_input_lines(previous, users, fresh_input, tail)),
    ]
    if cache_write:
        sections.append((paired_text("Cache Write", "缓存写入"), [
            f"<b>{_fmt(cache_write)}</b> — " + paired_text("Input tokens written to cache", "本步写入缓存的输入 token"),
        ]))
    sections.extend([
        (
            paired_text("Reasoning", "思考"),
            [f"<b>{_fmt(reasoning)}</b> — " + paired_text("Recorded reasoning tokens", "本步模型思考 token（think 块，含标签）")]
            if reasoning > 0
            else [paired_text("No reasoning recorded", "本步无思考记录")],
        ),
        (paired_text("Output", "输出"), _output_lines(step, output)),
    ])

    rows = []
    for title, lines in sections:
        body = "<br>".join(lines)
        rows.append(
            f"<tr><td style='padding:4px 10px 4px 0;white-space:nowrap;vertical-align:top;'>{title}</td>"
            f"<td style='padding:4px 0;'>{body}</td></tr>"
        )

    breakdown = f"{_fmt(fresh_input)} fresh + {_fmt(cache_read)} read"
    breakdown_zh = f"{_fmt(fresh_input)} 新增 + {_fmt(cache_read)} 命中"
    if cache_write:
        breakdown += f" + {_fmt(cache_write)} written"
        breakdown_zh += f" + {_fmt(cache_write)} 写入"
    separator, separator_zh = (" = ", " = ") if total == prompt + completion else (". Breakdown: ", "；分项：")
    completion_detail = f" ({_fmt(output)} output + {_fmt(reasoning)} reasoning)" if reasoning else ""
    completion_detail_zh = f"（{_fmt(output)} 输出 + {_fmt(reasoning)} 思考）" if reasoning else ""
    summary = paired_text(
        f"Total {_fmt(total)}{separator}prompt {_fmt(prompt)} ({breakdown}) + completion {_fmt(completion)}{completion_detail}",
        f"总计 {_fmt(total)}{separator_zh}prompt {_fmt(prompt)}（{breakdown_zh}） + completion {_fmt(completion)}{completion_detail_zh}",
    )

    return (
        f"<div style='margin-top:0.4em;'>"
        f"<div style='font-weight:600;margin-bottom:4px;'>{header} — {paired_text('Token breakdown', 'token 构成')}{jump}</div>"
        f"<table style='border-collapse:collapse;font-size:0.92em;'>{''.join(rows)}</table>"
        f"<div style='margin-top:6px;color:var(--body-text-color);'>{summary}</div>"
        f"</div>"
    )
