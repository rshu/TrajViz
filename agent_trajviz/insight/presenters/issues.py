"""Overview Issues panel: render the ranked issues as foldable HTML cards.

The issues themselves are computed by :mod:`agent_trajviz.insight.issues`
(English titles/details for the public API). Dashboard chrome here is Chinese,
including the title / detail / why text shown on each card.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from typing import cast, get_args

from ...i18n import text as ui_text
from ..context_usage import _explicit_compaction_events
from ..issues import (  # noqa: F401  (re-exported: UI and judge import them here)
    Confidence,
    IssueJudgment,
    IssueKind,
    OverviewIssue,
    collect_overview_issues,
    prepare_overview_issues,
    rank_issues,
)
from ..rendering import _step_link_chips
from ..session import LoadedSession
from ..tool_failure import tool_call_failed

_ISSUE_KINDS = frozenset(get_args(IssueKind))
_CONFIDENCES = frozenset(get_args(Confidence))

ISSUE_KIND_COLORS: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "risk": "var(--ov-risk)",
    "antipattern": "var(--ov-warn)",
    "bottleneck": "var(--ov-accent)",
}

_ISSUE_KIND_LABELS_ZH: dict[IssueKind, str] = {
    "error": "错误",
    "risk": "质量风险",
    "antipattern": "行为问题",
    "bottleneck": "耗时过长",
}

_CONFIDENCE_ZH: dict[Confidence, str] = {"high": "高", "medium": "中", "low": "低"}

#: What the reader should do about each issue type, keyed by source_id prefix
#: (first match wins). Falls back to a per-kind action.
_ISSUE_ACTION_ZH: tuple[tuple[str, str], ...] = (
    ("antipattern:compaction_thin_read", "人工复核压缩后那次写修改过的文件，确认内容与文件真实状态一致。"),
    ("antipattern:compaction_premature", "检查压缩摘要是否保留了关键信息；频繁出现时考虑调大上下文窗口。"),
    ("antipattern:compaction_grew", "这次压缩可以忽略；频繁出现时检查摘要是否写得过长。"),
    ("antipattern:fruitless", "更换搜索关键词或路径，先确认目标位置存在。"),
    ("antipattern:repeated_search", "修正查询词或路径，避免重复空搜。"),
    ("antipattern:bash_read", "改用 Read 工具读文件（带行号、更省上下文）。"),
    ("antipattern:edit_thrash", "写入前先 Read 目标文件，确认路径与内容匹配。"),
    ("antipattern:stalled_plan", "检查未完成的待办是否需要收尾或显式关闭。"),
    ("antipattern:plan_resets", "确认计划重置是有意为之；否则让智能体就地修订计划。"),
    ("antipattern:tool_errors", "排查环境问题：路径、权限、依赖是否就绪。"),
    ("antipattern:oversized_output", "让智能体改用带条数限制的读取：Read 指定行区间、head/tail、grep -c，测试命令关闭详细输出。"),
    ("antipattern:blind_retry", "失败后先读错误信息，修改命令或环境（路径/权限/依赖）再重试；偶发超时类错误重试一次属正常。"),
    ("antipattern:user_repeat", "查看第一次发送该消息后智能体的响应，确认它为何没有满足需求。"),
    ("antipattern:unvalidated_writes", "让智能体在改动后运行测试或构建（pytest、make、npm run build 等），至少验证一次。"),
    ("antipattern:redundant_reread", "让智能体依赖已读内容；确需重读时，先确认文件已变更（写入后或压缩后重读是合理的）。"),
    ("fail:system:", "排查运行环境配置（路径、权限）。"),
    ("cascade:", "从第一个失败步骤入手排查根因，避免反复重试放大问题。"),
    ("bottleneck:", "点步骤链接查看该步的耗时构成，针对性优化。"),
)

_ISSUE_ACTION_FALLBACK_ZH: dict[IssueKind, str] = {
    "error": "点步骤链接查看失败详情，排查输入与环境。",
    "risk": "点步骤链接复核相关步骤的输出。",
    "antipattern": "点步骤链接查看相关步骤。",
    "bottleneck": "点步骤链接查看该步骤的耗时构成。",
}

#: Real impact line for tool-failure cards; the recovery path moves to the
#: action line (it is "what to do", not "what it costs").
_FAIL_IMPACT_ZH = "工具调用失败会浪费轮次和 token；连续失败可能拖慢甚至卡住任务。"


def _issue_action_zh(issue: OverviewIssue) -> str:
    """The plain-language "what to do" line for an issue card."""
    sid = issue.source_id or ""
    why = issue.why or ""
    if why.startswith("Typical recovery: "):
        chain = why[len("Typical recovery: ") :]
        tools = [t.strip() for t in chain.split("→")]
        if len(tools) == 1:
            did = f"调用 {tools[0]}"
        else:
            did = "依次调用 " + " → ".join(tools)
        return f"失败后智能体{did}，自行恢复了；这类失败通常无需人工干预。"
    if why == "No recovery path found after this error cluster.":
        return "该错误之后智能体未自行恢复——建议人工排查该步骤的输入与环境。"
    for prefix, action in _ISSUE_ACTION_ZH:
        if sid.startswith(prefix):
            return action
    return _ISSUE_ACTION_FALLBACK_ZH.get(issue.kind, "")


def _issue_title_html(title: str) -> str:
    """Mark authored issue title templates while leaving error/file names intact.

    Used by i18n surface tests and any caller that needs English titles with
    client-side localization. Dashboard cards still go through
    :func:`_localize_issue_text` for the CodeAgent Chinese presentation.
    """
    for pattern, template, names in (
        (r"Context/cache bottleneck at #(\d+) \(([\d.]+)s\)",
         "Context/cache bottleneck at #{step} ({seconds}s)", ("step", "seconds")),
        (r"Inference bottleneck at #(\d+) \(([\d.]+)s\)",
         "Inference bottleneck at #{step} ({seconds}s)", ("step", "seconds")),
        (r"Idle/queue bottleneck before #(\d+) \(([\d.]+)s gap\)",
         "Idle/queue bottleneck before #{step} ({seconds}s gap)", ("step", "seconds")),
        (r"Tool bottleneck: (.+) \(([\d.]+)s\)",
         "Tool bottleneck: {tool} ({seconds}s)", ("tool", "seconds")),
        (r"Failure cascade \((\d+) steps\)",
         "Failure cascade ({count} steps)", ("count",)),
    ):
        if match := re.fullmatch(pattern, title):
            return ui_text(template, **dict(zip(names, match.groups(), strict=True)))
    return html.escape(title)


def _with_count(label_zh: str, title_en: str) -> str:
    """Keep the English ``(N×)`` suffix from ``issues._count_label`` titles."""
    m = re.search(r" \((\d+)×\)$", title_en)
    return label_zh + (m.group(0) if m else "")


def _localize_fruitless_detail(detail: str) -> str:
    m = re.match(r"(\d+) wasted steps — (.+)$", detail)
    if not m:
        return detail
    rest = m.group(2).replace(", ", "，")
    rest = re.sub(r"steps (\d+)-(\d+) \((\d+)\)", r"第\1–\2步（\3）", rest)
    rest = re.sub(r"\+(\d+) more \((\d+)\)", r"另有 \1 段（\2）", rest)
    return f"{m.group(1)} 步无效搜索 — {rest}"


def _localize_bottleneck_title(title: str) -> str:
    m = re.match(r"Idle/queue bottleneck before #(\d+) \((\d+(?:\.\d+)?)s gap\)$", title)
    if m:
        return f"空闲/排队瓶颈：第 {m.group(1)} 步前空等 {m.group(2)} 秒"
    m = re.match(r"Context/cache bottleneck at #(\d+) \((\d+(?:\.\d+)?)s\)$", title)
    if m:
        return f"上下文/缓存瓶颈：第 {m.group(1)} 步耗时 {m.group(2)} 秒"
    m = re.match(r"Inference bottleneck at #(\d+) \((\d+(?:\.\d+)?)s\)$", title)
    if m:
        return f"推理瓶颈：第 {m.group(1)} 步耗时 {m.group(2)} 秒"
    m = re.match(r"Performance bottleneck at #(\d+)$", title)
    if m:
        return f"性能瓶颈：第 {m.group(1)} 步耗时异常"
    return title


def _localize_issue_text(issue: OverviewIssue) -> tuple[str, str, str]:
    """Chinese title / detail / why for the dashboard card; English API unchanged."""
    sid = issue.source_id or ""
    title, detail, why = issue.title, issue.detail, issue.why

    if sid.startswith("antipattern:compaction_premature"):
        m = re.match(
            r"window occupancy at compaction time: "
            r"(.+)% of the assumed (.+)-token window$",
            detail,
        )
        if m:
            detail = f"压缩时窗口占用：假定 {m.group(2)} token 窗口的 {m.group(1)}%"
        elif detail.endswith("well below the window limit"):
            detail = "压缩时窗口占用：远低于窗口上限"
        return (
            _with_count("上下文被过早压缩", title),
            detail,
            "上下文窗口在远未写满时就被压缩——通常是智能体主动触发的 compress/compact，"
            "而非运行环境强制回收。过早压缩会丢失上下文；请检查摘要是否保留了关键信息。",
        )

    if sid.startswith("antipattern:compaction_grew"):
        detail_zh = re.sub(r"step (\d+):", r"第\1步：", detail)
        detail_zh = re.sub(r", \+(\d+) more$", r"，另有 \1 次", detail_zh)
        return (
            _with_count("压缩未起效：上下文不降反增", title),
            detail_zh,
            "压缩结束后窗口占用不小于压缩前——摘要本身比腾出的消息更占 token，"
            "这次压缩没有腾出有效空间。",
        )

    if sid.startswith("antipattern:compaction_thin_read"):
        m = re.match(
            r"Post-compaction write may rest on summarized context \(#(\d+)\)$",
            title,
        )
        if m:
            title = f"压缩后凭摘要改代码（第 {m.group(1)} 步压缩之后）"
        segs = [s.strip() for s in detail.split(";") if s.strip()]
        zh: list[str] = []
        for seg in segs:
            m = re.match(r"compaction dropped (\S+) tokens$", seg)
            if m:
                zh.append(f"压缩丢弃约 {m.group(1)} token")
                continue
            m = re.match(
                r"consecutive writes at steps (\d+) and (\d+) span it: "
                r"~(\S+) tokens of file reads before vs ~(\S+) after \((\d+)% less\)$",
                seg,
            )
            if m:
                zh.append(
                    f"第 {m.group(1)}、{m.group(2)} 步两次连续写之间发生压缩："
                    f"压缩前读取 ≈{m.group(3)} token，压缩后仅 ≈{m.group(4)} token"
                    f"（少 {m.group(5)}%）"
                )
                continue
            zh.append(seg)
        return (
            title,
            "；".join(zh),
            "两次连续写之间发生压缩，且压缩后的文件读取量比压缩前少 20% 以上："
            "第二次写所依据的信息大部分只剩压缩摘要，而非上下文中的新鲜读取，"
            "写出的内容可能偏离文件真实状态。写操作需要读多少因任务而异，"
            "请点步骤链接复核第二次写。",
        )

    if sid.startswith("antipattern:fruitless"):
        return (
            _with_count("连续多次搜索无结果", title),
            _localize_fruitless_detail(detail),
            "连续三次及以上搜索均无匹配；持续空转通常说明搜错了位置或关键词。",
        )

    if sid.startswith("antipattern:bash_read"):
        return (
            _with_count("用 Bash 读文件", title),
            "这些步骤用了 sed/cat/head，而不是 Read 工具",
            "用 shell 读文件会绕过结构化的 Read：没有行号、缓存更差、上下文更大。",
        )

    if sid.startswith("antipattern:stalled_plan"):
        return (
            _with_count("计划项卡住未完成", title),
            detail,
            "待办长期停在进行中却未完成——常见于切换任务后没有收尾。",
        )

    if sid.startswith("antipattern:plan_resets"):
        return (
            _with_count("计划被整表重置", title),
            "待办列表内容被整体替换，新旧项几乎没有重叠",
            "运行中途整表重写计划，通常表示智能体放弃了原上下文，而不是就地完成或修订。",
        )

    if sid.startswith("antipattern:edit_thrash:"):
        m = re.match(r"Failed edit retries on (.+) \((\d+)×\)$", title)
        if m:
            title = f"对 {m.group(1)} 的失败编辑重试（{m.group(2)}×）"
        detail = re.sub(
            r"(\d+) failed write\(s\) in steps (\d+)–(\d+)",
            r"第\2–\3步中有 \1 次写入失败",
            detail,
        )
        return (
            title,
            detail,
            "同一文件在写入失败后反复重写——常见于路径错误、补丁对不上，或缺少前置检查。",
        )

    if sid.startswith("antipattern:repeated_search:"):
        m = re.match(r"Repeated empty search \((\d+)×\)$", title)
        if m:
            title = f"同一搜索多次返回空结果（{m.group(1)}×）"
        return (
            title,
            detail,
            "同一搜索多次返回空结果（不一定连续）——查询词或路径很可能不对。",
        )

    if sid.startswith("antipattern:redundant_reread:"):
        detail_zh = re.sub(
            r"(\S+) re-read (\d+)× with no write or compaction in between "
            r"\(steps ([\d, ]+)\)",
            lambda m: (
                f"{m.group(1)} 重复读取 {m.group(2)} 次——两次读取之间既没有写入"
                f"也没有压缩（第 {m.group(3).replace(', ', '、')} 步）"
            ),
            detail,
        )
        return (
            _with_count("重复读取未变更的文件", title),
            detail_zh,
            "文件既没有被修改，也没有因压缩丢失——内容本来就在上下文里，"
            "重复读取只是白花 token。",
        )

    if sid.startswith("antipattern:unvalidated_writes"):
        detail_zh = re.sub(
            r"(\d+) code file\(s\) written in steps (\d+)–(\d+); "
            r"no test/build/lint command in the whole run",
            r"改了 \1 个代码文件（第 \2–\3 步）；整个运行没有任何测试/构建/检查命令",
            detail,
        )
        return (
            "改了代码但全程未验证",
            detail_zh,
            "所有改动都成功了，但没有任何命令检查过结果——正确性全凭智能体自信，"
            "它自己看不到的回归没有任何机会被发现。",
        )

    if sid.startswith("antipattern:user_repeat:"):
        return (
            _with_count("用户重复发送同一消息", title),
            detail,
            "用户把一字不差的同一条消息又发了一遍——这是“上一次没有满足需求”"
            "最直接的证据，不需要任何推测。",
        )

    if sid.startswith("antipattern:blind_retry:"):
        m = re.match(r"Blind retry of failed (\S+) call \((\d+)×\)$", title)
        if m:
            title = f"失败后原样重试 {m.group(1)}（{m.group(2)}×）"
        detail = re.sub(
            r"(\d+) of (\d+) identical attempts failed — ",
            r"\2 次完全相同的调用中 \1 次失败 — ",
            detail,
        )
        return (
            title,
            detail,
            "失败的调用未做任何修改就被原样重发，且再次失败——同样的输入在同样的"
            "环境下只会得到同样的结果，重试只是白烧 token 和轮次。",
        )

    if sid.startswith("antipattern:oversized_output"):
        detail_zh = re.sub(
            r"step (\d+) (\S+) ≈([\d,]+) tokens \((\d+)% of context\)",
            r"第\1步 \2 ≈\3 token（占上下文 \4%）",
            detail,
        )
        detail_zh = re.sub(
            r"step (\d+) (\S+) ≈([\d,]+) tokens",
            r"第\1步 \2 ≈\3 token",
            detail_zh,
        )
        detail_zh = re.sub(r", \+(\d+) more$", r"，另 \1 次", detail_zh)
        detail_zh = detail_zh.replace(", ", "，")
        return (
            _with_count("单条工具输出过大", title),
            detail_zh,
            "这条工具输出会一直留在对话历史里：在被压缩掉之前，后面每一步都要把它"
            "整条重新计入上下文。大输出通常来自整文件 cat、全量 find 或详细模式测试。",
        )

    if sid.startswith("antipattern:tool_errors"):
        return (
            _with_count("环境类工具错误", title),
            "根据工具输出检测到（平台错误、权限、文件缺失等）",
            "失败的工具调用既费 token 又费轮次，且往往是环境问题而非智能体策略错误。",
        )

    if sid.startswith("fail:system:"):
        if title.startswith("Frequent system errors: "):
            title = f"频繁系统错误：{title[len('Frequent system errors: '):]}"
            why = (
                "单次运行中出现大量 Read/Grep/Edit 类失败——单次风险通常不高，"
                "但数量说明路径、权限或运行环境配置值得排查。"
            )
        elif title.startswith("System error: "):
            title = f"系统错误：{title[len('System error: '):]}"
            why = (
                "脚手架/工具链未命中（搜索、读取或写入）——通常风险较低；"
                "智能体往往无需改作者代码即可恢复。"
            )
        if detail == "system tool failure":
            detail = "系统工具失败"
        return title, detail, why

    if sid.startswith("fail:"):
        # Plain title for the scan; the tool name and exit code live in the
        # folded tech detail, one click away.
        m = re.search(r" \((\d+)×\)$", title)
        plain = title[: m.start()] if m else title
        detail_zh = plain
        if detail:
            detail_zh += f"；示例错误：{detail}"
        return (
            _with_count("工具调用失败", title),
            detail_zh,
            _FAIL_IMPACT_ZH,
        )

    if sid.startswith("cascade:"):
        return (
            f"连续失败（{len(issue.steps)} 步）",
            re.sub(r"steps (\d+)–(\d+)", r"第\1–\2步", detail),
            "连续多个助手步骤失败且中间没有干净恢复；级联往往会把一个根因放大成反复折腾。",
        )

    if sid.startswith("bottleneck:"):
        m = re.match(r"Tool bottleneck: (.+) \((\d+(?:\.\d+)?)s\)$", title)
        if m:
            # Plain title: tool name + duration; the full command/target is in
            # the detail (folded away by the card).
            tool = m.group(1).split(": ", 1)[0]
            title = f"工具瓶颈：{tool} 调用耗时 {m.group(2)} 秒"
        else:
            title = _localize_bottleneck_title(title)
        detail = re.sub(r"Step (\d+):", r"第\1步：", detail)
        detail = detail.replace(" executing tools", " 在执行工具")
        if why.startswith("Large pre-step idle"):
            why = "步骤前有很长空闲，通常是排队或限流——应降低并行压力或调整等待策略，而不是改提示词。"
        elif why.startswith("Most of this outlier step was spent in a tool"):
            why = "该异常耗时步骤主要花在工具上——优先换更轻的工具、缩小范围，或缓存结果避免重复重活。"
        elif why.startswith("Long inference with heavy tokens"):
            why = "推理耗时长且 token 重/缓存命中差——应精简上下文、提高缓存局部性，或避免每轮重载大文件。"
        elif why.startswith("Outlier model-side turn"):
            why = "模型侧耗时异常且工具等待不多——常见于过长推理或提示；收紧指令或拆分任务。"
        elif why.startswith("Session-relative duration outlier"):
            why = "相对本会话的耗时异常点，且有明确的主导工具、空闲/排队或上下文/推理原因。"
        return title, detail, why

    # Synthetic cards in tests (and recovery why without a fail: source id).
    if why.startswith("Typical recovery: ") or (
        why == "No recovery path found after this error cluster."
    ):
        return title, detail, _FAIL_IMPACT_ZH
    return title, detail, why


def overview_issues_payload(issues: list[OverviewIssue]) -> list[dict]:
    """JSON-friendly ranked issues for Gradio State (same shape as ``asdict`` / views)."""
    return [asdict(issue) for issue in issues]


def overview_issue_from_payload(
    row: dict,
    *,
    max_steps: int | None = 12,
) -> OverviewIssue:
    """Rebuild an OverviewIssue from :func:`overview_issues_payload` / ``asdict``.

    *max_steps* caps related step ids for UI prompts (default 12). Pass
    ``None`` for agent export so every related step is kept.
    """
    judgment = None
    raw_j = row.get("judgment")
    if isinstance(raw_j, dict) and raw_j.get("where") and raw_j.get("fix"):
        conf = str(raw_j.get("confidence") or "medium").strip().lower()
        if conf not in _CONFIDENCES:
            conf = "medium"
        judgment = IssueJudgment(
            where=str(raw_j["where"]),
            fix=str(raw_j["fix"]),
            also=str(raw_j.get("also") or ""),
            confidence=cast(Confidence, conf),
        )
    raw_steps = row.get("steps") or ()
    if max_steps is None:
        steps = tuple(int(s) for s in raw_steps)
    else:
        steps = tuple(int(s) for s in raw_steps[:max_steps])
    kind = row.get("kind") if row.get("kind") in _ISSUE_KINDS else "error"
    return OverviewIssue(
        kind=cast(IssueKind, kind),
        title=str(row.get("title") or ""),
        detail=str(row.get("detail") or ""),
        why=str(row.get("why") or ""),
        steps=steps,
        source_id=str(row.get("source_id") or ""),
        judgment=judgment,
    )


def issue_analysis_question(issue: OverviewIssue, number: int) -> str:
    """Focused sidebar question for one ranked Overview issue."""
    step_ids = issue.steps[:12]
    steps = ", ".join(f"#{s}" for s in step_ids)
    lines = [
        f"Analyze this specific Overview issue directly (issue #{number}).",
        "Do not rewrite a full-run report — focus only on this issue.",
        f"Kind: {issue.kind}",
        f"Title: {issue.title}",
    ]
    if issue.detail:
        lines.append(f"Detail: {issue.detail}")
    if issue.why:
        lines.append(f"Why it matters: {issue.why}")
    if steps:
        lines.append(f"Related steps: {steps}")
        if len(issue.steps) > 12:
            lines.append(f"(+{len(issue.steps) - 12} more steps)")
    if issue.judgment is not None:
        j = issue.judgment
        lines.append(f"Current suggested fix location: {j.where}")
        lines.append(f"Current suggested fix: {j.fix}")
        if j.also:
            lines.append(f"Supporting note: {j.also}")
    lines.append(
        "Using the dashboard brief, explain what went wrong at the related steps, "
        "the likely root cause, and concrete fix actions."
    )
    return "\n".join(lines)


def issue_analysis_question_from_payload(rows: list, number: int) -> str | None:
    """Build the focused question for 1-based *number* from a State payload."""
    if number < 1 or number > len(rows):
        return None
    row = rows[number - 1]
    if not isinstance(row, dict):
        return None
    return issue_analysis_question(overview_issue_from_payload(row), number)


def _issue_card(issue: OverviewIssue, number: int, *, workflow_links: bool = True) -> str:
    """Scan row: number + kind badge + plain title, then visible impact and
    action lines; the technical detail folds away below.

    The action slot holds exactly one suggestion: the LLM judgment when it
    exists (with fix location and confidence), otherwise the rule-based
    playbook. When the LLM owns the slot, the playbook folds away under
    技术细节 as 常规做法.
    """
    title_raw, detail_raw, why_raw = _localize_issue_text(issue)
    action_raw = _issue_action_zh(issue)
    title = html.escape(title_raw)
    border = ISSUE_KIND_COLORS[issue.kind]
    steps_html = _step_link_chips(
        list(issue.steps),
        label="第{n}步",
        more="另有 {extra} 步",
        workflow_links=workflow_links,
    )

    judgment = issue.judgment
    why_html = ""
    if why_raw:
        why_html = (
            "<div class='overview-issue-why'>"
            "<span class='overview-issue-label'>影响</span>"
            f"{html.escape(why_raw)}</div>"
        )
    action_html = ""
    if judgment is not None:
        also_html = ""
        if judgment.also:
            also_html = (
                f"<div class='overview-issue-also'>补充：{html.escape(judgment.also)}</div>"
            )
        action_html = (
            "<div class='overview-issue-action'>"
            "<span class='overview-issue-label'>建议</span>"
            f"<span class='overview-issue-label overview-issue-label-ai'>"
            f"AI · 置信度：{_CONFIDENCE_ZH[judgment.confidence]}</span>"
            f"<div class='overview-issue-fix'>{html.escape(judgment.fix)}</div>"
            "<div class='overview-issue-where'>"
            "<span class='overview-issue-where-label'>修改位置</span>"
            f"<span class='overview-issue-where-value'>{html.escape(judgment.where)}</span>"
            f"</div>{also_html}</div>"
        )
    elif action_raw:
        action_html = (
            "<div class='overview-issue-action'>"
            "<span class='overview-issue-label'>建议</span>"
            f"{html.escape(action_raw)}</div>"
        )

    tech_parts: list[str] = []
    if detail_raw:
        tech_parts.append(
            f"<div class='overview-issue-detail'>{html.escape(detail_raw)}</div>"
        )
    if judgment is not None and action_raw:
        tech_parts.append(
            "<div class='overview-issue-detail'>"
            "<span class='overview-issue-label'>常规做法</span>"
            f"{html.escape(action_raw)}</div>"
        )
    tech_html = ""
    if tech_parts:
        tech_html = (
            "<details class='overview-issue-tech'>"
            "<summary>技术细节</summary>"
            f"{''.join(tech_parts)}</details>"
        )
    # Live dashboard only: static/exported reports keep evidence without
    # Gradio onclick handlers (see workflow_links=False / passive reports).
    analyze_html = ""
    if workflow_links:
        analyze_html = (
            f"<button type='button' class='overview-issue-analyze' "
            f"data-issue-index=\"{number}\" "
            f"onclick=\"if(window.tvAnalyzeIssue)"
            f"{{window.tvAnalyzeIssue(this.getAttribute('data-issue-index'));}}\">"
            f"AI 深入分析</button>"
        )
    return (
        f"<div class='overview-issue-card' style='border-left-color:{border};'>"
        f"<div class='overview-issue-head'>"
        f"<span class='overview-issue-index'>#{number}</span>"
        f"<span class='overview-issue-kind' style='color:{border};'>"
        f"{_ISSUE_KIND_LABELS_ZH[issue.kind]}</span>"
        f"<span class='overview-issue-title'>{title}</span>"
        f"</div>"
        f"{why_html}"
        f"{action_html}"
        f"{steps_html}"
        f"{tech_html}"
        f"{analyze_html}"
        f"</div>"
    )


@dataclass(frozen=True)
class _IssueGroupView:
    """One flat card or a parent wrapping same-family leaves (UI only)."""

    parent_title: str | None
    parent_kind: IssueKind
    leaves: tuple[tuple[int, OverviewIssue], ...]  # (leaf_number, issue)


_BOTTLENECK_CAUSE_ZH = {
    "idle": "空闲/排队瓶颈",
    "tool": "工具瓶颈",
    "context": "上下文/缓存瓶颈",
    "inference": "推理瓶颈",
}


def _issue_family_key(issue: OverviewIssue) -> tuple[str, ...] | None:
    """Stable family key for grouping, or None for a singleton flat card."""
    sid = issue.source_id
    if sid.startswith("antipattern:edit_thrash:"):
        return ("edit_thrash",)
    if sid.startswith("antipattern:fruitless") or sid.startswith("antipattern:repeated_search:"):
        return ("search_waste",)
    if sid.startswith("bottleneck:"):
        parts = sid.split(":")
        cause = parts[1] if len(parts) > 1 else "unknown"
        return ("bottleneck", cause)
    return None


def _group_parent_title(key: tuple[str, ...], n: int) -> str:
    kind = key[0]
    if kind == "edit_thrash":
        return f"失败编辑重试（{n} 个文件）"
    if kind == "search_waste":
        return "搜索浪费"
    if kind == "bottleneck":
        label = _BOTTLENECK_CAUSE_ZH.get(key[1] if len(key) > 1 else "unknown", "性能瓶颈")
        return f"{label}（{n}）"
    return f"相关问题（{n}）"


def group_issue_families(ranked: list[OverviewIssue]) -> list[_IssueGroupView]:
    """Nest same-family leaves under parents; singletons stay flat.

    Leaf numbers are the contiguous 1-based indices in *ranked* order (for
    ``data-issue-index`` / State alignment). A group's position is the first
    member's rank; later siblings are pulled into that group.
    """
    numbered = [
        (i, issue, _issue_family_key(issue))
        for i, issue in enumerate(ranked, start=1)
    ]
    used: set[int] = set()
    views: list[_IssueGroupView] = []

    for num, issue, key in numbered:
        if num in used:
            continue
        if key is None:
            used.add(num)
            views.append(_IssueGroupView(None, issue.kind, ((num, issue),)))
            continue
        members = [(n, iss) for n, iss, k in numbered if k == key]
        used.update(n for n, _ in members)
        leaves = tuple(members)
        if len(members) == 1:
            views.append(_IssueGroupView(None, members[0][1].kind, leaves))
        else:
            views.append(
                _IssueGroupView(
                    _group_parent_title(key, len(members)),
                    members[0][1].kind,
                    leaves,
                )
            )
    return views


_GROUP_LEAF_PREVIEW = 3


def _issue_group_children_html(
    leaves: tuple[tuple[int, OverviewIssue], ...],
    *,
    workflow_links: bool = True,
) -> str:
    cards = [
        f"<div class='overview-issue-child'>{_issue_card(issue, number, workflow_links=workflow_links)}</div>"
        for number, issue in leaves
    ]
    if len(cards) <= _GROUP_LEAF_PREVIEW:
        return "".join(cards)
    more = len(cards) - _GROUP_LEAF_PREVIEW
    return (
        f"{''.join(cards[:_GROUP_LEAF_PREVIEW])}"
        f"<details class='overview-issue-more'>"
        f"<summary class='overview-issue-more-summary'>显示其余 {more} 项</summary>"
        f"<div class='overview-issue-more-body'>{''.join(cards[_GROUP_LEAF_PREVIEW:])}</div>"
        f"</details>"
    )


def _issue_group_html(view: _IssueGroupView, *, workflow_links: bool = True) -> str:
    if view.parent_title is None:
        number, issue = view.leaves[0]
        return _issue_card(issue, number, workflow_links=workflow_links)

    border = ISSUE_KIND_COLORS[view.parent_kind]
    return (
        f"<details class='overview-issue-group' open "
        f"style='border-left-color:{border};'>"
        f"<summary class='overview-issue-group-head'>"
        f"<span class='overview-issue-kind' style='color:{border};'>"
        f"{_ISSUE_KIND_LABELS_ZH[view.parent_kind]}</span>"
        f"<span class='overview-issue-title'>{html.escape(view.parent_title)}</span>"
        f"<span class='overview-issue-group-meta'>{len(view.leaves)} 项</span>"
        f"</summary>"
        f"<div class='overview-issue-group-body'>"
        f"{_issue_group_children_html(view.leaves, workflow_links=workflow_links)}</div>"
        f"</details>"
    )


def _issue_cards_html(issues: list[OverviewIssue], *, workflow_links: bool = True) -> str:
    """Render ranked leaf issues, nested under family parents when applicable."""
    return "".join(
        _issue_group_html(view, workflow_links=workflow_links)
        for view in group_issue_families(issues)
    )


def _fmt_k(n: float) -> str:
    """Compact token counts for summary lines: 940 / 12.3k / 1.2M."""
    n = int(n)
    if abs(n) < 1000:
        return f"{n:,}"
    if abs(n) < 1_000_000:
        v = n / 1000
        return f"{v:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.2f}M"


def _run_numbers_zh(steps: list[dict], metrics: dict) -> list[str]:
    """The headline numbers of a run, as display-ready strings.

    Tool names come from the export, so they are escaped here and the list is
    joined WITHOUT a second escape.
    """
    from ..metrics import format_duration_seconds

    nums = [f"{len(steps)} 步"]
    wall = metrics.get("wall_clock")
    if isinstance(wall, (int, float)) and wall > 0:
        nums.append(format_duration_seconds(wall))
    toks = metrics.get("tokens") or {}
    total_tokens = toks.get("total") or 0
    if total_tokens:
        nums.append(f"{_fmt_k(total_tokens)} token")
    cache_read = toks.get("cache_read") or 0
    fresh = toks.get("input") or 0
    if cache_read + fresh > 0:
        pct = round(cache_read / (cache_read + fresh) * 100)
        nums.append(f"缓存命中 {pct}%")
    tool_n = metrics.get("tool_call_count") or 0
    if tool_n:
        breakdown = metrics.get("tool_breakdown") or {}
        if breakdown:
            top_name, top_n = max(breakdown.items(), key=lambda kv: kv[1])
            nums.append(f"工具 {tool_n} 次（{html.escape(str(top_name))} 最多，{top_n} 次）")
        else:
            nums.append(f"工具 {tool_n} 次")
    compactions = _explicit_compaction_events(steps)
    if compactions:
        nums.append(f"压缩 {len(compactions)} 次")
    sub_agents = len(metrics.get("agent_breakdown") or {})
    if sub_agents:
        nums.append(f"{sub_agents + 1} 个 agent")
    return nums


_SPINNER_SVG = (
    "<svg class='overview-issues-progress-spinner' width='14' height='14' "
    "viewBox='0 0 24 24' aria-hidden='true' "
    "style='flex-shrink:0;display:block'>"
    "<circle cx='12' cy='12' r='10' fill='none' "
    "stroke='currentColor' stroke-opacity='0.25' stroke-width='3'/>"
    "<path d='M12 2a10 10 0 0 1 10 10' fill='none' stroke='currentColor' "
    "stroke-width='3' stroke-linecap='round'>"
    "<animateTransform attributeName='transform' type='rotate' "
    "from='0 12 12' to='360 12 12' dur='0.7s' repeatCount='indefinite'/>"
    "</path></svg>"
)


def _issues_spinner_html(label: str, *, inline: bool = False) -> str:
    """SMIL spinner (survives Gradio HTML swaps / prefers-reduced-motion)."""
    tag = "span" if inline else "div"
    style = " style='margin:0;display:inline-flex;'" if inline else ""
    return (
        f"<{tag} class='overview-issues-progress' role='status' "
        f"aria-live='polite'{style}>"
        f"{_SPINNER_SVG}<span>{html.escape(label)}</span></{tag}>"
    )


def _run_pending_value_html() -> str:
    """Spinner placeholder shown while the LLM run summary is being generated."""
    return _issues_spinner_html("正在等待 AI 总结任务与结局…", inline=True)


def build_run_summary_html(
    session: LoadedSession,
    ai_summary: tuple[str, str] | None = None,
    *,
    ai_pending: bool = False,
) -> str:
    """Story of the run shown under the verdict banner.

    任务/结局 come from the LLM (*ai_summary*, shown with an AI badge — see
    ``issue_judge.judge_run_summary``); while it is being generated
    (*ai_pending*) a spinner placeholder is shown instead. Without AI the two
    rows are omitted entirely — only the 数字 row (local metrics) remains.
    """
    steps = session.steps
    if not steps:
        return ""

    # Ending note: the last assistant step that ran tools ended with every
    # call failed — the run likely stopped on broken ground.
    note = ""
    for s in reversed(steps):
        calls = s.get("tool_calls") or []
        if s.get("role") == "assistant" and calls:
            if all(tool_call_failed(c) for c in calls if isinstance(c, dict)):
                note = "该步工具调用全部失败"
            break

    def _row(label: str, value: str) -> str:
        return (
            f"<div class='ov-run-row'>"
            f"<span class='ov-run-label'>{label}</span>"
            f"<span class='ov-run-value'>{value}</span></div>"
        )

    rows = []
    if ai_summary is not None:
        ai_task, ai_outcome = ai_summary
        if ai_task:
            task = (
                "<span class='overview-issue-label overview-issue-label-ai'>AI</span>"
                f"{html.escape(ai_task)}"
            )
        else:
            task = "<span class='ov-run-muted'>（AI 未总结任务）</span>"
        outcome_parts = []
        if ai_outcome:
            outcome_parts.append(
                "<span class='overview-issue-label overview-issue-label-ai'>AI</span>"
                f"{html.escape(ai_outcome)}"
            )
        else:
            outcome_parts.append("<span class='ov-run-muted'>（AI 未总结结局）</span>")
        if note:
            outcome_parts.append(
                f"<span class='ov-run-note'>⚠ {html.escape(note)}</span>"
            )
        rows.append(_row("任务", task))
        rows.append(_row("结局", " ".join(outcome_parts)))
    elif ai_pending:
        rows.append(_row("任务与结局", _run_pending_value_html()))
    metrics = getattr(session, "metrics", None) or {}
    rows.append(_row("数字", " · ".join(_run_numbers_zh(steps, metrics))))
    return "<div class='ov-run-summary'>" + "".join(rows) + "</div>"


def _verdict_banner_html(
    issues: list[OverviewIssue], *, limited_evidence: bool = False
) -> str:
    """One-glance health verdict above the Issues panel: overall status plus
    the single most important issue in plain language.

    Limited-evidence captures (OTLP / incomplete tool outcomes) must not claim
    a healthy empty state — the available checks cannot prove the run is clean.
    """
    if not issues:
        if limited_evidence:
            return (
                "<div class='ov-verdict' style='border-left-color:var(--ov-warn);'>"
                "<div class='ov-verdict-head' style='color:var(--ov-warn);'>"
                "⚠ 证据有限</div>"
                "<div class='ov-verdict-text'>部分工作流检查对此捕获不可用；"
                "现有证据不能证明运行没有问题。</div></div>"
            )
        return (
            "<div class='ov-verdict' style='border-left-color:var(--ov-success);'>"
            "<div class='ov-verdict-head' style='color:var(--ov-success);'>"
            "✅ 未检测到明显问题</div>"
            "<div class='ov-verdict-text'>本次运行的工作流健康；"
            "可在下方各分区查看执行细节。</div></div>"
        )
    errors = sum(1 for i in issues if i.kind == "error")
    risks = sum(1 for i in issues if i.kind == "risk")
    others = len(issues) - errors - risks
    top = issues[0]
    top_title, _, _top_detail = _localize_issue_text(top)
    top_action = _issue_action_zh(top)
    if errors or risks:
        color = "var(--ov-bad)" if errors else "var(--ov-risk)"
        parts = []
        if errors:
            parts.append(f"{errors} 个错误")
        if risks:
            parts.append(f"{risks} 个质量风险")
        head = "❗ 需要关注：" + "、".join(parts)
        text = f"最要紧的一项：「{top_title}」。{top_action}"
    else:
        color = "var(--ov-warn)"
        head = f"⚠ 运行完成，有 {others} 个可优化项（不影响结果正确性）"
        text = f"最主要的一项：「{top_title}」。{top_action}"
    return (
        f"<div class='ov-verdict' style='border-left-color:{color};'>"
        f"<div class='ov-verdict-head' style='color:{color};'>{head}</div>"
        f"<div class='ov-verdict-text'>{html.escape(text)}</div></div>"
    )


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    banner: str = "",
    progress: str = "",
    run_summary: str = "",
    limited_evidence: bool = False,
    workflow_links: bool = True,
) -> str:
    """Render a foldable Issues panel with an evidence-aware empty state.

    *progress* is a live status line shown inside the panel while the LLM judge runs.
    *run_summary* is the three-line run story shown between the verdict banner
    and the panel (see :func:`build_run_summary_html`).
    """
    banner_html = ""
    if banner:
        banner_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{html.escape(banner)}</div>"
        )

    progress_html = ""
    if progress:
        progress_html = (
            "<div class='overview-issues-progress' role='status' aria-live='polite'>"
            f"<span class='overview-issues-progress-label'>{ui_text('Thinking…')}</span>"
            f"{_SPINNER_SVG}<span>{html.escape(progress)}</span>"
            "</div>"
        )

    if not issues:
        status = "limited evidence" if limited_evidence else "none detected"
        detail = (
            "Some workflow checks are unavailable for this capture. "
            "The available evidence does not establish that the run is free of issues."
            if limited_evidence else "No major workflow issues detected."
        )
        return (
            _verdict_banner_html(issues, limited_evidence=limited_evidence)
            + run_summary
            + "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            f"<span>{ui_text('Issues')}</span>"
            f"<span class='overview-issues-summary-meta'>{ui_text(status)}</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            f"{progress_html}"
            f"{banner_html}"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            f"{ui_text(detail)}"
            "</div></div></details>"
        )

    count = len(issues)
    judged = sum(1 for i in issues if i.judgment is not None)
    count_label = ui_text("{count} issue" if count == 1 else "{count} issues", count=count)
    if limited_evidence:
        count_label += " · " + ui_text("limited evidence")
    if progress:
        count_label += " · " + ui_text("suggesting fixes…")
    elif judged:
        count_label += " · " + ui_text("{count} with LLM fix", count=judged)

    if workflow_links:
        hint = (
            "LLM Change/Fix shown below — click a step to open Workflow"
            if judged and not progress
            else "Ranked problems — click a step to open Workflow; expand Why for context"
        )
        # Fork CodeAgent: mention the AI deep-dive control when links are live.
        hint_html = (
            f"<div class='overview-issues-hint'>{ui_text(hint)}"
            " · 「AI 深入分析」打开右侧分析面板继续追问</div>"
        )
    else:
        hint = (
            "LLM Change/Fix shown below — step numbers refer to the recorded trajectory"
            if judged and not progress else
            "Ranked problems — step numbers refer to the recorded trajectory; expand Why for context"
        )
        hint_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{ui_text(hint)}</div>"
        )
    return (
        _verdict_banner_html(issues, limited_evidence=limited_evidence)
        + run_summary
        + "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        f"<span>{ui_text('Issues')}</span>"
        f"<span class='overview-issues-summary-meta'>{count_label}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        f"{progress_html}"
        f"{banner_html}"
        f"{hint_html}"
        + _issue_cards_html(issues, workflow_links=workflow_links)
        + "</div></details>"
    )


def build_overview_issues_html(
    session: LoadedSession,
    *,
    issues: list[OverviewIssue] | None = None,
    banner: str = "",
    progress: str = "",
    ai_summary: tuple[str, str] | None = None,
    ai_pending: bool = False,
    workflow_links: bool = True,
) -> str:
    """Collect (deduped + ranked) and render Overview Issues for *session*."""
    if issues is None:
        ranked = prepare_overview_issues(session)
    else:
        ranked = list(issues)
    limited = (
        session.format == "otlp"
        or bool(getattr(session, "metrics", {}).get("tool_unknown_outcomes"))
    )
    return render_overview_issues_html(
        ranked,
        banner=banner,
        progress=progress,
        run_summary=build_run_summary_html(session, ai_summary, ai_pending=ai_pending),
        limited_evidence=limited,
        workflow_links=workflow_links,
    )
