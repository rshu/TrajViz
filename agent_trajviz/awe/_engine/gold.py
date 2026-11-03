"""Build a GoldReference from a local ARGUS requirements file.

This is the objective backbone: gold source files (from the gold patch), the
tests that gate success, and gold-derived task/dimension OPPORTUNITIES (a task
property, fixed across agents — so capability denominators are comparable)."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field

from . import config
from .benchmark import resolve_profile
from .diffutil import parse_unified_diff, is_test_file
from .evidence import changed_lines as _changed_lines


def _as_list(v) -> list[str]:
    """Strict test-ID list; retain legacy JSON/Python-encoded list support."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                raise ValueError("test IDs must be a list or an encoded list") from None
    if not isinstance(v, list) or any(not isinstance(x, str) or not x.strip() for x in v) or len(v) != len(set(v)):
        raise ValueError("test IDs must be unique nonempty strings in a list")
    return list(v)


@dataclass
class GoldReference:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_files: list[str]  # source files changed by the gold patch (tests excluded)
    gold_symbols: list[str]  # best-effort def/class names touched by gold
    gold_patch_loc: int
    test_patch_files: list[str]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    is_multi_file: bool
    is_cross_module: bool
    modules: list[str]
    # False when the benchmark's gating-test identifiers are private verifier
    # data. In that setting, not naming a FAIL_TO_PASS id is not an agent fault.
    gating_tests_visible: bool = True
    opportunities: dict[str, bool | None] = field(default_factory=dict)
    dimension_opportunities: dict[str, bool] = field(default_factory=dict)
    # {file: {"added": set, "removed": set}} of the reference change's line content.
    gold_changed_lines: dict = field(default_factory=dict)
    # {file: [(old_start, old_count), ...]} hunk ranges in old-file coordinates
    # (comparable with the agent patch's ranges: both diff against base_commit).
    gold_hunks: dict = field(default_factory=dict)
    benchmark_profile: dict = field(default_factory=dict)


def build_gold_reference(instance_id: str) -> GoldReference:
    path = config.requirements_path(instance_id)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    return gold_from_requirement(instance_id, d)


def gold_from_requirement(instance_id: str, d: dict) -> GoldReference:
    """Build a reference from a canonical task record without global path changes."""
    if type(d.get("gating_tests_visible", True)) is not bool:
        raise ValueError("gating_tests_visible must be a boolean")

    profile = resolve_profile(d)
    gold_patch_text = d.get("gold_patch", "") or ""
    gold_diffs = parse_unified_diff(gold_patch_text)
    gold_files = [fd.path for fd in gold_diffs if not is_test_file(fd.path, profile)]
    gold_symbols = sorted({s for fd in gold_diffs for s in fd.symbols})
    gold_patch_loc = sum(fd.loc for fd in gold_diffs)

    test_files = sorted(changed for changed in {fd.path for fd in parse_unified_diff(d.get("test_patch", "") or "")})

    modules = sorted({os.path.dirname(f) for f in gold_files})
    is_multi_file = len(set(gold_files)) > 1
    is_cross_module = len(modules) > 1

    fail_to_pass = _as_list(d.get("FAIL_TO_PASS", []))
    pass_to_pass = _as_list(d.get("PASS_TO_PASS", []))

    # Task-level capability opportunities (gold-derived, fixed across agents).
    # self_repair_loop is encountered-only -> None here (set later from trajectory).
    opportunities = {
        "requirement_understanding": True,
        "code_localization": True,
        "task_planning": bool(is_multi_file or gold_patch_loc >= 30),
        "tool_use": True,
        "code_editing": True,
        "code_verification": bool(fail_to_pass or pass_to_pass),
        "self_repair_loop": None,
    }
    # Dimension-level opportunities are where gold grounding adds the most:
    # only score cross-module localization when the gold actually spans modules.
    dimension_opportunities = {
        "multi_file_localization": is_multi_file,
        "cross_module_localization": is_cross_module,
        "regression_risk": bool(pass_to_pass),
    }

    return GoldReference(
        instance_id=instance_id,
        repo=d.get("repo", ""),
        base_commit=d.get("base_commit", ""),
        problem_statement=d.get("problem_statement", ""),
        gold_files=gold_files,
        gold_symbols=gold_symbols,
        gold_patch_loc=gold_patch_loc,
        test_patch_files=test_files,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        is_multi_file=is_multi_file,
        is_cross_module=is_cross_module,
        modules=modules,
        gating_tests_visible=profile["gating_tests_visible"],
        benchmark_profile=profile,
        opportunities=opportunities,
        dimension_opportunities=dimension_opportunities,
        gold_changed_lines=_changed_lines(gold_patch_text),
        gold_hunks={fd.path: list(fd.hunks) for fd in gold_diffs if fd.hunks},
    )
