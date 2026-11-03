"""Validate frozen, scoped remedy experiments; never infer efficacy from diagnosis.

This checks evidence integrity and re-derives paired outcomes. It cannot establish
that supplied logs are authentic or that the declared experiment was conducted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from . import config

ARMS = ("baseline", "active_control", "treatment")
METRICS = ("resolved", "failure_surfaced", "target_fault", "p2p_pass_rate", "cost")
ENDPOINTS = {"fix_rate": "resolved", "failure_surfacing": "failure_surfaced", "budget_efficiency": "cost"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _strings(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and bool(v.strip()) for v in value)
        and len(value) == len(set(value))
    )


def recipe_digest(entry):
    """Bind efficacy evidence to diagnosis and deployment semantics, not just text."""
    fields = (
        "id",
        "capability",
        "error_type",
        "evidence_signal",
        "selection_rules",
        "artifact_type",
        "artifact_content",
        "artifact_plug_in",
        "runtime_trigger",
        "scaffold_layer",
        "hook_event",
        "expected_effect",
        "contract",
        "applicability",
        "runtime_contract",
    )
    payload = {name: getattr(entry, name) for name in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _hashed_file(ref, root):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("invalid_evidence_reference")
    name = config.safe_relative_path(ref["path"])
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence_path_outside_project")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError("evidence_hash_mismatch")
    return path


def _interval(values, tail):
    """Deterministic paired bootstrap over tasks, with seeds averaged within task."""
    rng = random.Random(0)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(4000))
    return [means[int(tail * len(means))], means[int((1 - tail) * len(means)) - 1]]


def validate_verification(entry, root=None):
    """Fail closed for empty/stale records, unpaired runs or unsupported benefits.

    Each arm must have a hashed run log. Treatment must improve against both
    controls on declared endpoints within harm margins. Version 1 also requires
    target-fault reduction; version 2 makes that mechanism check explicit and
    adds resolution preservation and separate false-alarm controls for surfacing.
    Intervals are task-clustered and Bonferroni-adjusted across these comparisons.
    Passing supports only the recorded held-out tasks and declared context.
    """
    root = Path(root or config.PROJECT_ROOT)
    try:
        verification = entry.verification
        if (
            not isinstance(verification, dict)
            or type(verification.get("schema_version")) is not int
            or verification["schema_version"] not in (1, 2)
        ):
            raise ValueError("missing_experiment_record")
        goal_aligned = verification["schema_version"] == 2
        protocol = json.loads(_hashed_file(verification.get("protocol"), root).read_text(encoding="utf-8"))
        record = json.loads(_hashed_file(verification.get("results"), root).read_text(encoding="utf-8"))
        if not isinstance(protocol, dict) or not isinstance(record, dict):
            raise ValueError("invalid_experiment_record")
        if (
            not isinstance(protocol.get("experiment_id"), str)
            or not protocol["experiment_id"]
            or record.get("experiment_id") != protocol["experiment_id"]
            or record.get("protocol_sha256") != verification["protocol"]["sha256"]
            or protocol.get("remedy_id") != entry.id
            or protocol.get("recipe_sha256") != recipe_digest(entry)
            or protocol.get("artifact_sha256") != hashlib.sha256(entry.artifact_content.encode()).hexdigest()
        ):
            raise ValueError("experiment_or_artifact_mismatch")
        # Bind the same diagnosis taxonomy used to measure target-fault recurrence.
        _hashed_file(protocol.get("taxonomy"), root)
        if (
            protocol["taxonomy"]["path"] != "configs/taxonomy.slice.yaml"
            or not isinstance(protocol.get("analysis_commit"), str)
            or len(protocol["analysis_commit"]) != 40
            or any(c not in "0123456789abcdef" for c in protocol["analysis_commit"])
            or not isinstance(protocol.get("judge_model"), str)
            or not protocol["judge_model"]
        ):
            raise ValueError("missing_analysis_provenance")
        scope = protocol.get("scope")
        if (
            not isinstance(scope, dict)
            or set(scope) != {"agents", "benchmarks", "languages"}
            or not all(_strings(v) and "all" not in v for v in scope.values())
            or not _strings(protocol.get("held_out_tasks"))
            or len(protocol["held_out_tasks"]) < 2
            or protocol.get("allocation") != "paired_randomized"
            or type(protocol.get("randomization_seed")) is not int
            or protocol.get("oracle_free_runtime_reviewed") is not True
            or not isinstance(protocol.get("cost_unit"), str)
            or not protocol["cost_unit"].strip()
            or protocol.get("endpoints") != entry.expected_effect
            or not entry.expected_effect
            or not set(entry.expected_effect) <= ENDPOINTS.keys()
        ):
            raise ValueError("invalid_scope_or_protocol")
        cost_margin = protocol.get("max_cost_increase_fraction")
        p2p_margin = protocol.get("max_p2p_drop")
        if not _number(cost_margin) or cost_margin < 0 or not _number(p2p_margin) or not 0 <= p2p_margin <= 1:
            raise ValueError("missing_harm_margins")
        resolution_margin, false_alarm_margin, negatives = 0, 0, []
        require_fault_reduction = True
        if goal_aligned:
            resolution_margin = protocol.get("max_resolution_drop")
            require_fault_reduction = protocol.get("require_target_fault_reduction")
            if (
                protocol.get("verification_policy") != "goal_aligned_v2"
                or not _number(resolution_margin)
                or not 0 <= resolution_margin <= 1
                or type(require_fault_reduction) is not bool
            ):
                raise ValueError("missing_goal_aligned_policy")
            if "failure_surfacing" in entry.expected_effect:
                false_alarm_margin = protocol.get("max_false_alarm_increase")
                negatives = protocol.get("negative_control_tasks")
                if (
                    not _number(false_alarm_margin)
                    or not 0 <= false_alarm_margin <= 1
                    or not _strings(negatives)
                    or not set(negatives) < set(protocol["held_out_tasks"])
                ):
                    raise ValueError("missing_false_alarm_controls")
        rows = record.get("runs")
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing_paired_runs")
        groups, run_ids = defaultdict(dict), set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid_run")
            if (
                not isinstance(row.get("run_id"), str)
                or not row["run_id"]
                or row["run_id"] in run_ids
                or row.get("arm") not in ARMS
                or type(row.get("seed")) is not int
                or row.get("task_id") not in protocol["held_out_tasks"]
                or any(
                    row.get(k) not in scope[s]
                    for k, s in (("agent", "agents"), ("benchmark", "benchmarks"), ("language", "languages"))
                )
            ):
                raise ValueError("invalid_run_identity_or_scope")
            _hashed_file(row.get("log"), root)
            run_ids.add(row["run_id"])
            for metric in METRICS:
                value = row.get(metric)
                if (
                    not _number(value)
                    or value < 0
                    or (metric != "cost" and value > 1)
                    or (metric in METRICS[:3] and value not in (0, 1))
                ):
                    raise ValueError("invalid_outcome")
            if goal_aligned and "failure_surfacing" in entry.expected_effect:
                false_alarm = row.get("false_alarm_rate")
                if not _number(false_alarm) or not 0 <= false_alarm <= 1:
                    raise ValueError("missing_false_alarm_measurement")
            key = tuple(row[k] for k in ("task_id", "agent", "benchmark", "language", "seed"))
            if row["arm"] in groups[key]:
                raise ValueError("duplicate_paired_arm")
            groups[key][row["arm"]] = row
        if (
            any(set(g) != set(ARMS) for g in groups.values())
            or {k[0] for k in groups} != set(protocol["held_out_tasks"])
            or any(
                {r[k] for r in rows} != set(scope[s])
                for k, s in (("agent", "agents"), ("benchmark", "benchmarks"), ("language", "languages"))
            )
        ):
            raise ValueError("incomplete_pairing_or_scope")
        # Per-context checks prevent pooling an improvement from one agent or
        # language over harm to another. A task is the independent sampling unit.
        contexts = sorted({key[1:4] for key in groups})
        comparisons = []
        required_checks = (
            len(entry.expected_effect) + 2 + int(require_fault_reduction) + int(goal_aligned) + int(bool(negatives))
        )
        tail = 0.05 / (2 * len(contexts) * 2 * required_checks)
        for context in contexts:
            for control in ARMS[:2]:
                values = defaultdict(lambda: defaultdict(list))
                for key, arms in groups.items():
                    if key[1:4] != context:
                        continue
                    treatment, baseline = arms["treatment"], arms[control]
                    for endpoint in entry.expected_effect:
                        # Sensitivity is measured on failure cases; negative controls
                        # measure false alarms separately rather than diluting them.
                        if endpoint == "failure_surfacing" and key[0] in negatives:
                            continue
                        metric = ENDPOINTS[endpoint]
                        delta = treatment[metric] - baseline[metric]
                        values[endpoint][key[0]].append(-delta if metric == "cost" else delta)
                    if goal_aligned:
                        values["resolution_preservation"][key[0]].append(
                            treatment["resolved"] - baseline["resolved"] + resolution_margin
                        )
                        if key[0] in negatives:
                            values["false_alarm_limit"][key[0]].append(
                                baseline["false_alarm_rate"] - treatment["false_alarm_rate"] + false_alarm_margin
                            )
                    values["target_fault_reduction"][key[0]].append(
                        baseline["target_fault"] - treatment["target_fault"]
                    )
                    values["p2p_preservation"][key[0]].append(
                        treatment["p2p_pass_rate"] - baseline["p2p_pass_rate"] + p2p_margin
                    )
                    values["cost_limit"][key[0]].append(baseline["cost"] * (1 + cost_margin) - treatment["cost"])
                if len(values["target_fault_reduction"]) < 2:
                    raise ValueError("insufficient_tasks_in_context")
                if negatives and len(values["false_alarm_limit"]) < 2:
                    raise ValueError("insufficient_negative_controls_in_context")
                for endpoint, task_values in values.items():
                    if len(task_values) < 2:
                        raise ValueError("insufficient_tasks_for_endpoint")
                    if not all(_number(v) for task in task_values.values() for v in task):
                        raise ValueError("nonfinite_comparison")
                    ci = _interval([sum(v) / len(v) for v in task_values.values()], tail)
                    if not all(_number(v) for v in ci):
                        raise ValueError("nonfinite_interval")
                    required = endpoint != "target_fault_reduction" or require_fault_reduction
                    strict = endpoint in ENDPOINTS or endpoint == "target_fault_reduction"
                    if required and (ci[0] < 0 or (strict and ci[0] == 0)):
                        raise ValueError("benefit_or_harm_check_failed")
                    comparisons.append(
                        {
                            "context": list(context),
                            "control": control,
                            "endpoint": endpoint,
                            "interval": ci,
                            "n_tasks": len(task_values),
                            **({"required": required} if goal_aligned else {}),
                        }
                    )
        return {
            "valid": True,
            "reason": "scoped_paired_evidence",
            "scope": scope,
            "experiment_id": protocol["experiment_id"],
            "comparisons": comparisons,
            **({"verification_policy": "goal_aligned_v2"} if goal_aligned else {}),
        }
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        return {"valid": False, "reason": str(error)}


def verification_scope(entry, case):
    result = validate_verification(entry)
    if not result["valid"]:
        return "uncertain", "invalid_verification_record"
    for field, key in (("agent", "agents"), ("benchmark", "benchmarks"), ("language", "languages")):
        if not case.get(field):
            return "uncertain", "verification_scope_unavailable"
        if case[field] not in result["scope"][key]:
            return "rejected", "outside_verification_scope"
    return None
