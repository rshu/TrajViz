"""Owned comparison outcome prose; report values and evidence stay literal."""

ZH_CN = {
    "supported_finding": "有依据的发现",
    "hypothesis": "待验证假设",
    "improved": "改善",
    "regressed": "退化",
    "unchanged": "无变化",
    "No comparable metric observations are available.": "暂无可比较的指标观测数据。",
    "No significant changes detected.": "未检测到显著变化。",
    "Mixed results — review per-metric deltas.": "结果有好有坏，请查看各项指标的变化。",
    "Intervention effects are unavailable: no successful task pairs are available.":
        "无法评估干预效果：没有成功配对的任务。",
    "Intervention improved {improved} with minor regressions in {regressed} (below guardrail thresholds).":
        "干预改善了 {improved}，但 {regressed} 有轻微退化（未达到保护阈值）。",
    "Intervention improved {improved} without regressions.":
        "干预改善了 {improved}，且未出现退化。",
    "Intervention improved {improved} but regressed {regressed}. Review tradeoffs.":
        "干预改善了 {improved}，但使 {regressed} 退化，请评估这些权衡。",
    "Intervention regressed {regressed} without improvements.":
        "干预使 {regressed} 退化，且未带来改善。",
    "{metric} regressed from {before} to {after} ({relative}% relative)":
        "{metric} 从 {before} 退化至 {after}（相对退化 {relative}%）",
    "{metric} regressed from {before} to {after} ({relative}% relative, p={p_value})":
        "{metric} 从 {before} 退化至 {after}（相对退化 {relative}%，p={p_value}）",
}
