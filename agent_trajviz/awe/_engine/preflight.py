"""Read-only validation of canonical benchmark inputs before scored runs.

This validates artifact structure and joins, not patch applicability or whether
the benchmark's evaluator is correct. Those require executing its harness.
"""

from __future__ import annotations

import json

from . import config
from .benchmark import resolve_profile
from .diffutil import is_test_file, parse_unified_diff
from .gold import _as_list
from .outcomes import load_outcomes, load_overrides, parse_pertest, pertest_path


def _diff(text, reference=False, profile=None):
    if not isinstance(text, str):
        raise ValueError("patch must be a string")
    files = parse_unified_diff(text)
    if text.strip() and not files:
        raise ValueError("nonempty patch has no parseable file sections")
    for f in files:
        config.safe_relative_path(f.path)
        if f.unbounded_lines or f.invalid_hunks:
            raise ValueError("changed payload outside valid hunks: " + f.path)
        if f.loc and not f.hunks:
            raise ValueError("changed lines have no valid declared hunk: " + f.path)
        if any(not h["complete"] for h in f.hunk_contents):
            raise ValueError("incomplete or invalid hunk: " + f.path)
    if reference and not any(f.loc and not is_test_file(f.path, profile) for f in files):
        raise ValueError("full scoring requires a changed reference source file")
    return files


def validate_inputs(agents, instances):
    """Return errors/warnings without modifying inputs, caches or outputs.

    Missing optional trajectories/per-test records are coverage facts. Missing
    final patches for known outcomes are errors: absence cannot distinguish
    export loss from an intentionally empty patch (which should be a real file).
    Unknown/unattempted/excluded cases remain outside scoring.
    """
    agents, instances = list(agents), list(instances)
    report = {
        "schema_version": 1,
        "agents": agents,
        "instance_ids": instances,
        "errors": [],
        "warnings": [],
        "counts": {
            "tasks": len(instances),
            "known_outcomes": 0,
            "unknown_or_unattempted": 0,
            "excluded": 0,
            "pertest_records": 0,
            "missing_pertest": 0,
            "withheld_pertest": 0,
        },
    }

    def issue(level, code, where, detail):
        report[level].append({"code": code, "where": where, "detail": str(detail)})

    for kind, values in (("agent", agents), ("instance", instances)):
        if not values or len(values) != len(set(values)):
            issue("errors", "invalid_selection", kind, "must be nonempty and unique")
        for value in values:
            try:
                config.safe_id(value)
            except ValueError as exc:
                issue("errors", "unsafe_id", kind, exc)
    if report["errors"]:
        return report
    # Check joins against the whole data root, even when the scored scope is a
    # subset. Legitimate outcomes outside --limit are not orphan artifacts.
    all_ids = set(config.list_instances())
    tasks = {}
    for iid in instances:
        try:
            d = json.loads(config.requirements_path(iid).read_text(encoding="utf-8"))
            if not isinstance(d, dict) or d.get("instance_id") != iid:
                raise ValueError("instance_id must match requirements filename")
            for key in ("repo", "base_commit", "problem_statement"):
                if not isinstance(d.get(key), str) or not d[key].strip():
                    raise ValueError(key + " must be a nonempty string")
            if type(d.get("gating_tests_visible", True)) is not bool:
                raise ValueError("gating_tests_visible must be a boolean")
            profile = resolve_profile(d)
            _diff(d.get("gold_patch"), reference=True, profile=profile)
            if "test_patch" in d:
                _diff(d["test_patch"])
            else:
                issue("warnings", "missing_test_patch", iid, "evaluator test-file evidence unavailable")
            f2p, p2p = (_as_list(d[key]) for key in ("FAIL_TO_PASS", "PASS_TO_PASS"))
            if set(f2p) & set(p2p):
                raise ValueError("F2P and P2P test sets overlap")
            if not f2p:
                issue("warnings", "empty_f2p", iid, "No F2P selectors; P2P can still open verification opportunities")
            tasks[iid] = (f2p, p2p, profile)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue("errors", "invalid_requirement", iid, exc)
    try:
        overrides = load_overrides()
        for (agent, iid), entry in overrides.items():
            config.safe_id(agent)
            config.safe_id(iid)
            if agent in agents and iid not in all_ids:
                raise ValueError("override has no requirement: " + agent + "/" + iid)
    except (OSError, ValueError, TypeError) as exc:
        issue("errors", "invalid_overrides", "overrides", exc)
        overrides = {}
    for agent in agents:
        try:
            outcomes = load_outcomes(agent)
            for iid in outcomes:
                config.safe_id(iid)
                if iid not in all_ids:
                    raise ValueError("outcome has no requirement: " + iid)
        except (OSError, ValueError, TypeError) as exc:
            issue("errors", "invalid_outcomes", agent, exc)
            continue
        for iid in instances:
            where = agent + "/" + iid
            override = overrides.get((agent, iid))
            outcome = override["resolved"] if override else outcomes.get(iid)
            if outcome is None:
                key = "excluded" if override else "unknown_or_unattempted"
                report["counts"][key] += 1
                continue
            report["counts"]["known_outcomes"] += 1
            try:
                _diff(config.patch_path(agent, iid).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                issue("errors", "invalid_final_patch", where, exc)
            if not any((config.TRAJECTORY_DIR / agent / (iid + ext)).exists() for ext in (".json", ".jsonl")):
                issue("warnings", "missing_trajectory", where, "trajectory-dependent evidence may be unavailable")
            path = pertest_path(agent, iid)
            if not path.exists():
                report["counts"]["missing_pertest"] += 1
                continue
            report["counts"]["pertest_records"] += 1
            try:
                expected = tasks.get(iid)
                pt = parse_pertest(
                    json.loads(path.read_text(encoding="utf-8")),
                    iid,
                    expected_f2p=expected[0] if expected else None,
                    expected_p2p=expected[1] if expected else None,
                )
                if pt is None:
                    raise ValueError("per-test record belongs to a different task or has no record")
                if pt.get("outcome_conflict"):
                    policy = expected[2]["resolution_policy"] if expected else "additional_conditions"
                    issue(
                        "errors" if policy == "all_tests_pass" else "warnings",
                        "unexplained_failure",
                        where,
                        "all recorded tests passed but overall verdict failed; inspect extra evaluator conditions",
                    )
                if pt["resolved"] != outcome and override:
                    report["counts"]["withheld_pertest"] += 1
                    issue(
                        "warnings",
                        "superseded_pertest",
                        where,
                        "documented outcome override supersedes this per-test verdict; "
                        "loader withholds this stale record",
                    )
                elif pt["resolved"] != outcome:
                    raise ValueError(
                        "per-test verdict disagrees with effective outcome; "
                        "reconcile provenance or withhold the stale per-test record"
                    )
            except (OSError, ValueError, TypeError) as exc:
                issue("errors", "invalid_pertest", where, exc)
    return report


def require_valid_inputs(agents, instances):
    report = validate_inputs(agents, instances)
    if report["errors"]:
        details = "\n".join(e["where"] + ": " + e["detail"] for e in report["errors"])
        raise ValueError("Input preflight failed before scoring:\n" + details)
    return report
