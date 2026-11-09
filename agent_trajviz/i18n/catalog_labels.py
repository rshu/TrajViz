"""Display names for the bundled step taxonomy; canonical IDs remain unchanged."""

PHASE_LABELS = {
    "understand": "理解",
    "plan": "规划",
    "implement": "实施",
    "debug": "调试",
    "validate": "验证",
    "report": "汇报",
}

ACTION_LABELS = {
    "spec_intake": "需求理解",
    "scope_clarification": "范围澄清",
    "constraint_extraction": "约束梳理",
    "tool_discovery": "工具探索",
    "delegated_research": "委派调研",
    "external_research": "外部调研",
    "file_discovery": "文件定位",
    "code_reading": "代码阅读",
    "context_synthesis": "上下文梳理",
    "plan_management": "计划管理",
    "task_breakdown": "任务拆解",
    "sequencing": "执行排序",
    "dependency_mapping": "依赖梳理",
    "risk_planning": "风险规划",
    "implement_api_schema": "接口与结构实现",
    "implement_runtime_logic": "运行逻辑实现",
    "implement_tests": "测试编写",
    "implement_generated_artifacts": "生成产物更新",
    "implement_config": "配置修改",
    "implement_docs": "文档编写",
    "implement_refactor": "代码重构",
    "implement_migration": "迁移实现",
    "debug_reproduction": "问题复现",
    "debug_root_cause": "根因定位",
    "debug_hypothesis_test": "调试假设验证",
    "debug_fix_selection": "修复方案选择",
    "validation_run": "执行验证",
    "validation_review": "验证结果审查",
    "validation_lint": "静态检查",
    "validation_build": "构建验证",
    "validation_unit_tests": "单元测试验证",
    "validation_integration_tests": "集成测试验证",
    "validation_e2e_tests": "端到端测试验证",
    "validation_regression": "回归验证",
    "validation_performance": "性能验证",
    "progress_update": "进度更新",
    "final_reporting": "最终汇报",
    "change_summary": "变更总结",
    "blockers_risks": "阻碍与风险",
    "next_steps_handoff": "后续步骤交接",
}

RESERVED_PHASE_LABELS = {"user": "用户", "unknown": "未知"}
RESERVED_ACTION_LABELS = {"user_prompt": "用户提示", "unknown": "未知"}

ZH_CN = {
    **PHASE_LABELS, **ACTION_LABELS, **RESERVED_PHASE_LABELS, **RESERVED_ACTION_LABELS,
    "user prompt": "用户提示",
}
ZH_CN.update({key + ": {count}": value + "：{count}" for key, value in tuple(ZH_CN.items())})
