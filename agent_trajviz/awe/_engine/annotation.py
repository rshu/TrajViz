"""Annotation task records + taxonomy options for human calibration.

A task record gives the annotator everything needed to diagnose INDEPENDENTLY of
the detector (the detector's own prediction is stored separately, for scoring, so
annotators are not anchored). Annotators fill the blank `annotation` block using
the closed taxonomy options."""

from __future__ import annotations

from .gold import build_gold_reference
from .outcomes import resolved
from .adapters import load_normalized
from .detect import detect
from .rootcause import build_blame
from .arbiter import auto_adjudicate

CAPABILITIES = [
    "requirement_understanding",
    "task_planning",
    "code_localization",
    "code_editing",
    "code_verification",
    "self_repair_loop",
    "tool_use",
    "other_capability",
]
CONFIDENCE = ["low", "medium", "high"]

# Mirror of configs/taxonomy.slice.yaml (the human-facing doc). Keep in sync.
# The two judgment-heavy capabilities mirror agent_trajviz.awe._engine.judge.{REQUIREMENT,PLANNING}_ERRORS.
ERRORS_BY_CAPABILITY = {
    "requirement_understanding": [
        "misread_requirement",
        "missed_constraint",
        "incomplete_requirement_coverage",
        "scope_misjudged",
        "solved_unrelated_problem",
    ],
    "task_planning": ["no_decomposition", "flawed_plan", "premature_implementation", "abandoned_plan"],
    "code_localization": ["missed_relevant_file", "wrong_file_localization", "cross_module_context_missing"],
    "code_editing": [
        "incorrect_patch",
        "relevant_change_omitted",
        "no_patch_produced",
        "gave_up_early",
        "excessive_extra_edits",
        "partial_fix_only",
        "regression_introduced",
    ],
    # v0.39: partial_fix_only / regression_introduced (code_editing) and
    # regression_test_missing (code_verification) are LIVE on the eval+ tier —
    # emitted only when a per-test harness report exists for the case
    # (data/pertest/<agent>/<instance_id>.json; agent_trajviz.awe._engine.outcomes.load_pertest).
    # gating_test_modified was demoted to a signal (gold_tests_edited — its fact
    # calibrated flat, RR 1.01) and replaced by test_expectation_rewritten.
    "code_verification": [
        "relevant_test_not_run",
        "unsupported_success_claim",
        "no_test_after_final_edit",
        "test_expectation_rewritten",
        "authored_test_never_run",
        "regression_test_missing",
        "test_result_misinterpreted",
    ],
    # v0.42: retry_without_change demoted to a signal (protective, RR 0.40).
    "self_repair_loop": ["repair_not_reverified", "repeated_ineffective_attempt", "reverted_own_change"],
    "tool_use": ["tool_failure_not_retried", "repeated_patch_apply_failure", "hallucinated_path_unrecovered"],
}


def taxonomy_options() -> dict:
    return ERRORS_BY_CAPABILITY


def trajectory_summary(traj, max_text: int = 90) -> list[dict]:
    rows = []
    for s in traj.steps:
        rows.append(
            {
                "step_id": s.step_id,
                "actor": s.actor,
                "action": s.canonical_action if s.type == "tool_call" else s.type,
                "tool": s.native_tool,
                "files": s.files,
                "text": (s.text or s.output_excerpt or "")[:max_text],
            }
        )
    return rows


def build_task_record(agent: str, inst: str) -> dict:
    gold = build_gold_reference(inst)
    traj = load_normalized(agent, inst)
    return {
        "case_id": f"{agent}__{inst}",
        "instance_id": inst,
        "agent": agent,
        "repo": gold.repo,
        "resolved": resolved(agent, inst),
        "problem_statement": gold.problem_statement[:700],
        "gold_files": gold.gold_files,
        "n_steps": len(traj.steps) if traj else 0,
        "trajectory_summary": trajectory_summary(traj) if traj else [],
        "label_options": {
            "capabilities": CAPABILITIES,
            "errors_by_capability": taxonomy_options(),
            "confidence": CONFIDENCE,
        },
        "annotation": {  # BLANK — the annotator fills this
            "primary_capability": None,
            "primary_error_type": None,
            "secondary_errors": [],
            "onset_step": None,
            "evidence_steps": [],
            "confidence": None,
            "rationale": "",
        },
    }


def primary_from_blame(blame: dict) -> dict:
    """Interpret a blame dict as the detector's calibration prediction. Reports the
    CONFIDENT primary (None when conjunctive — i.e. the detector abstains rather than
    name an uncertain cause, which is also how an arbiter REFUTE surfaces), plus the
    dominant-mass capability and whether the arbiter corroborated the primary."""
    p = blame.get("primary")
    cb = blame.get("capability_blame", {})
    arb = (p or {}).get("arbiter") or {}
    return {
        "primary_capability": p["capability"] if p else None,
        "primary_error_type": p["error_type"] if p else None,
        "status": blame.get("status"),
        "dominant_capability": (max(cb, key=cb.get) if cb else None),
        "arbiter_corroborated": bool(arb.get("corroborated")),
    }


def detector_label(agent: str, inst: str) -> dict:
    """The full-pipeline prediction for construct validity: grounded rules + (cached)
    judge + (cached) arbiter. So detector-vs-human kappa measures the system as
    shipped, including the LLM complement."""
    d = detect(inst, agent, use_trajectory=True)  # judge auto-included if cached
    b = build_blame(d, adjudicate=auto_adjudicate(agent))  # arbiter applied if cached
    return primary_from_blame(b)
