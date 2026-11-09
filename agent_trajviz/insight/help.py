"""Centralized help text registry for metric tooltips and section explanations."""

from dataclasses import dataclass

# Tooltip text keyed by identifier. Only keys referenced by
# presenters/overview.py for the KPI cards (shown above the main Tabs),
# rendered as data-help tooltips.
# Add an entry here only together with the UI code that renders it.
HELP_TEXT: dict[str, str] = {
    # KPI card metrics
    "steps": "Total conversation turns. Subtitle is assistant vs user; the line below breaks assistant steps down by agent/subagent.",
    "wall_clock": "Elapsed wall-clock time from first to last step, including idle gaps between steps.",
    "tokens": "Total tokens consumed across all steps: the sum of each step's reported total, which counts input, output, reasoning, cache read and cache write once each, however the format nests them (Codex counts cache reads inside input; some models count reasoning inside output). A count that cannot be a token count (a negative input, as some OpenCode exports write) is left out of its field's total, which is then marked partial, so on such trajectories the parts may not add up to the total; the per-step chart draws those steps as their reported total only, hatched. A step reporting more cache-read tokens than its own total is left out of the cache share. Subtitle gen tok/s is output tokens ÷ model generation time (step duration minus spawn wait and timed tool waits).",
    "issues": "Count of ranked Overview Issues (errors, waste patterns, performance bottlenecks). Click to jump to the Issues panel.",
    "tool_success": "Percentage of tool calls that completed without errors. 100% means no tool failures.",
}


@dataclass(frozen=True)
class SectionGuide:
    """Header for one Overview Contents section: a visible one-liner plus a
    hover panel listing what each chart shows and a worked debugging example."""

    summary: str
    charts: tuple[tuple[str, str], ...]
    example: str


# Keyed by the section's OverviewRefs Column name. Chart names keep the
# English titles the charts render with, so readers can match them up.
SECTION_GUIDES: dict[str, SectionGuide] = {
    "performance_section": SectionGuide(
        summary="本次运行的成本和耗时：每一步用了多少 token、花了多长时间。",
        charts=(
            (
                "Token Usage by Step（每步 token 用量）",
                "每根柱子代表一步，按新输入、缓存读取、输出、推理等类型堆叠。"
                "柱子越高，这一步发给模型的内容越多。"
                "有多个智能体/子会话时，每个智能体单独一块图；横轴仍是 Workflow 的全局步骤编号，方便对照。"
                "点击柱子可在下方查看该步 token 构成说明。",
            ),
            (
                "Step Duration（每步耗时）",
                "每根柱子是一步的耗时（已扣除等待子智能体的时间），虚线是该智能体的平均值，"
                "明显偏高的柱子会标出秒数。琥珀色表示系统错误，红色表示工具错误。"
                "有多个智能体时按智能体分图，横轴仍是 Workflow 步骤编号；点击柱子可跳到对应步骤。",
            ),
        ),
        example=(
            "运行明显变慢时，先在 Step Duration 里找最高的柱子并点击，到 Workflow 看这一步在做什么。"
            "再对照 Token Usage 的同一位置：token 也很高，说明是上下文太大拖慢了推理；"
            "token 不高，多半是某个工具执行太久。"
        ),
    ),
    "tools_section": SectionGuide(
        summary="智能体调用了哪些工具、各花了多少时间、在哪里出错。",
        charts=(
            (
                "Behavioral Diagnostics（行为诊断）",
                "一组指标卡，例如平均缓存命中率、工具等待占比、工具耗时 P95。"
                "标红的指标超出了建议范围。",
            ),
            (
                "Tool Call Frequency（工具调用次数）",
                "每个工具被调用了多少次；有多个智能体时按智能体分色。",
            ),
            (
                "Tool Call Duration（工具耗时）",
                "每个工具的累计耗时，横条中的每一段是一次调用。"
                "悬停可看到调用发生在第几步，点击可跳转。",
            ),
            (
                "Tool Outcome Timeline（工具结果时间线）",
                "横轴是步骤，每个点是一次工具调用：圆点表示成功，× 表示失败。",
            ),
            (
                "Skill Calls by Agent（Skill 调用）",
                "哪个智能体调用了哪些 Skill，连线越粗次数越多。",
            ),
        ),
        example=(
            "怀疑智能体在盲目试错时，先看 Tool Outcome Timeline 里有没有一串密集的 ×；"
            "再看 Tool Call Frequency 中 Grep、Read 是否远多于 Edit。"
            "大量搜索却很少修改，通常说明它没找到该改的地方，"
            "可以在 AGENTS.md 或提示词里写清楚相关文件路径。"
        ),
    ),
    "agents_section": SectionGuide(
        summary="主智能体和子智能体如何分工：谁在什么时候工作、各自消耗了多少。",
        charts=(
            (
                "智能体卡片",
                "每个智能体的步数、token、耗时、工具调用、错误数和缓存命中率；"
                "子智能体会注明是在第几步被派生的，点击可跳转。",
            ),
            (
                "Agent Swimlane（智能体泳道图）",
                "每条泳道是一个智能体（最上面是用户消息），色块表示它在哪些步骤连续工作。",
            ),
            (
                "Token Breakdown by Agent（各智能体 token）",
                "每个智能体的 token 按新输入、缓存读取、输出、推理分类。",
            ),
        ),
        example=(
            "总 token 偏高时，先在 Token Breakdown by Agent 中找出占大头的智能体，"
            "再到泳道图上看它的工作区间。如果子智能体在重复主智能体已经做过的搜索，"
            "说明派生任务时交代的背景信息不够。"
        ),
    ),
    "context_section": SectionGuide(
        summary="模型的上下文窗口被哪些内容占用，以及占用量随步骤怎样变化。",
        charts=(
            (
                "占用构成",
                "上下文中各类内容占多少 token：系统提示词、工具定义、规则、Skills、MCP、"
                "对话、工具输出等。",
            ),
            (
                "Context Window Pressure（上下文窗口压力）",
                "横轴是步骤，纵轴是窗口占用量，虚线是窗口上限。只看一个智能体时，"
                "橙色区域是 70%–90%，红色区域是 90% 以上；菱形标记表示一次上下文压缩。",
            ),
            (
                "上方选项",
                "可以只看某个智能体、查看某次压缩前的窗口，或修改窗口上限（默认 128k）。",
            ),
        ),
        example=(
            "智能体在后半段开始忘事、重复做过的事时，看压力曲线是否升进红色区域，"
            "并紧接着出现压缩标记。再看占用构成：如果工具输出占了大头，"
            "说明它读入了太多大文件或长日志，可以让它只读取需要的行。"
        ),
    ),
    "diagnostics_section": SectionGuide(
        summary="文件读写轨迹、错误的根因，以及待办计划的执行情况。",
        charts=(
            (
                "File Interaction Timeline（文件交互时间线）",
                "横轴是步骤，纵轴是文件：圆形是读取，方形是写入，三角是搜索，星形是 Skill；"
                "被修改过的文件有醒目的边框。",
            ),
            (
                "根因候选",
                "按出现次数排列的错误类别，点击可跳到第一次出现的步骤。",
            ),
            (
                "Plan Progress Timeline（计划进度）",
                "每一行是一个待办项，横条从开始延伸到完成；红色表示停滞的待办项。",
            ),
        ),
        example=(
            "最终改动不对时，在文件交互时间线上找到被修改的文件：如果智能体反复读取却迟迟不写，"
            "或者写完又去搜索别的文件，说明它对改哪里没有把握。"
            "再对照计划进度，看是哪个待办项卡住了。"
        ),
    ),
    "deep_dive_section": SectionGuide(
        summary="完整的数字明细：整体指标、最慢和最耗 token 的步骤，以及逐条消息的数据。",
        charts=(
            (
                "Timing / Tokens / Efficiency（指标卡）",
                "整体耗时、token 分布、生成速度等汇总指标。",
            ),
            (
                "Message Hotspots（热点步骤）",
                "耗时最长、token 最多、缓存命中率最低、上下文增量最大的前 5 步"
                "（增量 = 该步输出+思考 与其触发的工具结果/新消息合计让下一次请求变大的部分）。"
                "点击 Step 列的数字可跳转到 Workflow 中该步的卡片查看详情。",
            ),
            (
                "Per-Message Diagnostics（逐条消息）",
                "每一步的耗时、token、生成速度、缓存命中率和工具调用数（最多显示前 80 条）。",
            ),
        ),
        example=(
            "想降低成本时，先看 Hotspots 中缓存命中率最低的几步。"
            "如果它们都紧跟在上下文压缩或切换智能体之后，说明这些地方重新发送了整段上下文，"
            "可以减少压缩次数，或把固定不变的提示词放在开头，方便复用缓存。"
        ),
    ),
    "labels_section": SectionGuide(
        summary="按阶段（Phase）和动作（Action）给每一步分类后的统计，需要先上传 *_labeled.json。",
        charts=(
            (
                "Step Count by Phase / Action（各阶段、各动作的步数）",
                "阶段包括理解（understand）、规划（plan）、实现（implement）、"
                "调试（debug）、验证（validate）和汇报（report）。",
            ),
            (
                "Duration by Phase / Action（各阶段、各动作的耗时）",
                "每个阶段、每种动作累计花了多长时间。",
            ),
            (
                "Step Timeline（步骤时间线）",
                "每一行是一步，横条长度是耗时，颜色表示阶段，条上标注动作。",
            ),
        ),
        example=(
            "运行耗时很长时，先比较各阶段的耗时：理解阶段远长于实现阶段，说明大部分时间花在找代码上。"
            "再在 Step Timeline 上看它是否在实现之后反复回到调试，"
            "这通常意味着第一次的修改方向错了。"
        ),
    ),
}

SECTION_GUIDES_EN: dict[str, SectionGuide] = {
    "performance_section": SectionGuide(
        summary="Review how many tokens each step used and how long it took.",
        charts=(
            ("Token Usage by Step", "Bars separate fresh input, cache reads, output, and reasoning. Agent panels share the global Workflow step numbers. Click a bar to inspect its token breakdown."),
            ("Step Duration", "Bars show time spent in each step after subtracting subagent wait. Dashed lines show each agent's mean. Amber marks system errors and red marks tool errors. Click a bar to open its Workflow step."),
        ),
        example="Open the tallest duration bar in Workflow and compare its token usage. A large prompt may explain slow inference. Inspect tool timings if token usage looks typical.",
    ),
    "tools_section": SectionGuide(
        summary="See which tools the agents called, how long they took, and where they failed.",
        charts=(
            ("Behavioral Diagnostics", "Summary measures include cache hit rate, share of time waiting for tools, and 95th-percentile tool duration. Red highlights a measure outside its suggested range."),
            ("Tool Call Frequency", "Counts calls per tool, grouped by agent when there are several."),
            ("Tool Call Duration", "Each segment is one invocation. Hover for the step number and click to jump to it."),
            ("Tool Outcome Timeline", "Each marker is a tool call. Dots indicate success and crosses indicate failure."),
            ("Skill Calls by Agent", "Thicker connections show more calls from an agent to a skill."),
        ),
        example="Look for clustered failures on the outcome timeline. If searches and reads greatly outnumber edits, check whether the instructions gave the agent enough context to find the right files.",
    ),
    "agents_section": SectionGuide(
        summary="See when the main agent and subagents worked and how much each used.",
        charts=(
            ("Agent Cards", "Summarizes steps, tokens, duration, tool calls, failures, and cache hits. A subagent card can link to its spawn step."),
            ("Agent Swimlane", "Each lane tracks one agent's activity across steps, with user messages in the top lane."),
            ("Token Breakdown by Agent", "Compares each agent's fresh input, cache reads, output, and reasoning."),
        ),
        example="Find the highest token user and inspect its activity in the swimlane. A subagent repeating its parent's searches may have received too little context at delegation.",
    ),
    "context_section": SectionGuide(
        summary="See what fills the context window and how its usage changes over time.",
        charts=(
            ("Context Composition", "Breaks down estimated context use across prompts, tools, rules, skills, conversation, and tool results."),
            ("Context Window Pressure", "Tracks occupancy by step. The dashed line marks the configured limit, and diamonds mark compaction. The single-agent view highlights occupancy above 70% and 90%."),
            ("Controls", "Choose an agent, inspect a pre-compaction snapshot, or adjust the assumed window limit."),
        ),
        example="If the agent starts repeating work, look for high occupancy followed by compaction. When tool results dominate, inspect whether the agent read more of the files or logs than it needed.",
    ),
    "diagnostics_section": SectionGuide(
        summary="Inspect file interactions, likely error causes, and plan progress.",
        charts=(
            ("File Interaction Timeline", "The horizontal axis is step number and each row is a file. Circles are reads, squares are writes, triangles are searches, and stars are skill calls."),
            ("Root Cause Candidates", "Groups failures by type. A step link opens the first matching event."),
            ("Plan Progress Timeline", "Each row tracks a plan item from its start through completion. Red indicates a stalled item."),
        ),
        example="When an edit lands in the wrong place, look for repeated reads and searches around that write. Compare those steps with the plan to see where the agent lost direction.",
    ),
    "deep_dive_section": SectionGuide(
        summary="Inspect run totals, unusually slow or expensive steps, and per-message measurements.",
        charts=(
            ("Timing, Tokens, and Efficiency", "Summarizes duration, token distribution, and generation speed."),
            ("Message Hotspots", "Ranks the largest durations, token loads, and context growth, along with the lowest cache hits. Click a step number to inspect it in Workflow."),
            ("Per-Message Diagnostics", "Lists duration, tokens, generation speed, cache hits, and tool calls for up to 80 steps."),
        ),
        example="When investigating cost, check the lowest cache hits. See whether they follow a context compaction or a switch between agents.",
    ),
    "labels_section": SectionGuide(
        summary="Upload a *_labeled.json sidecar to group steps by phase and action.",
        charts=(
            ("Step Count by Phase / Action", "Shows the number of labeled steps by phase and action."),
            ("Duration by Phase / Action", "Shows the cumulative time spent in each phase and action."),
            ("Step Timeline", "Each bar is a step. Length shows duration, color shows phase, and the label names the action."),
        ),
        example="Compare time spent understanding the task with time spent implementing it. The timeline shows whether implementation repeatedly returned to debugging.",
    ),
}
