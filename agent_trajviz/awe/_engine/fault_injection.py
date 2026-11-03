"""Executable controlled-fixture pilot; no paid judges or human labels.

Programs and fixed evaluators run in disposable directories. F2P/P2P outcomes
come exclusively from completed evaluator records. Crashes, timeouts and bad
records are ungraded, never substituted with a failed task verdict.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import config
from .detect import detect
from .gold import _as_list, gold_from_requirement
from .outcomes import parse_pertest
from .patches import patch_from_text
from .trajectory import SCHEMA_VERSION

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fault_injection"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _diff(baseline, current):
    return "".join(
        "".join(
            difflib.unified_diff(
                baseline.get(name, "").splitlines(True),
                current.get(name, "").splitlines(True),
                fromfile="a/" + name if name in baseline else "/dev/null",
                tofile="b/" + name if name in current else "/dev/null",
            )
        )
        for name in sorted(set(baseline) | set(current))
    )


def _grade(doc, f2p, p2p):
    if not isinstance(doc, dict) or type(doc.get("resolved")) is not bool:
        raise ValueError("evaluator must emit an object with a boolean resolved verdict")
    tests = doc.get("tests")
    if not isinstance(tests, list):
        raise ValueError("evaluator must emit a tests list")
    actual = {}
    for test in tests:
        if (
            not isinstance(test, dict)
            or set(test) != {"id", "passed"}
            or not isinstance(test.get("id"), str)
            or type(test.get("passed")) is not bool
            or test["id"] in actual
        ):
            raise ValueError("test records require unique IDs and boolean grades")
        actual[test["id"]] = test["passed"]
    if set(actual) != set(f2p + p2p):
        raise ValueError("evaluator test coverage differs from the fixture contract")
    if doc["resolved"] != all(actual.values()):
        raise ValueError("evaluator verdict contradicts the complete test grades")
    record = {"patch_successfully_applied": True, "resolved": doc["resolved"], "tests_status": {}}
    for key, ids in (("FAIL_TO_PASS", f2p), ("PASS_TO_PASS", p2p)):
        record["tests_status"][key] = {
            "success": [i for i in ids if actual[i]],
            "failure": [i for i in ids if not actual[i]],
        }
    return parse_pertest(record, expected_f2p=f2p, expected_p2p=p2p)


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def _execute(runtime, evaluator, work, f2p, p2p, timeout):
    result = {
        "argv": [runtime, evaluator],
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "status": "ungraded",
        "pertest": None,
    }
    try:
        proc = subprocess.run(
            [runtime, evaluator],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        )
        result.update(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        if proc.returncode != 0:
            result["error"] = "evaluator process failed; no task verdict assigned"
            return result
        result["pertest"] = _grade(json.loads(proc.stdout), f2p, p2p)
        result["status"] = "graded"
    except subprocess.TimeoutExpired as exc:
        result.update(stdout=_text(exc.stdout), stderr=_text(exc.stderr), error="evaluator timeout")
    except (OSError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def run_fixture(fixture_dir, runtime, timeout=10):
    """Run a trusted local fixture, paired restorations and a harmless control.

    This API executes fixture code. It is intended for bundled/reviewed fixtures,
    not arbitrary downloaded benchmark manifests. Expected labels describe
    detector behavior for planted changes, not independently adjudicated causes.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    runtime = os.fspath(runtime)
    folder = Path(fixture_dir)
    spec = json.loads((folder / "fixture.json").read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported executable fixture schema")
    source, evaluator = (config.safe_relative_path(spec[k]) for k in ("source", "evaluator"))
    if source == evaluator:
        raise ValueError("evaluator must be separate from the mutable source")
    f2p, p2p = (_as_list(spec[k]) for k in ("FAIL_TO_PASS", "PASS_TO_PASS"))
    if not f2p or set(f2p) & set(p2p):
        raise ValueError("fixture requires nonempty F2P and disjoint F2P/P2P sets")
    baseline = {source: (folder / "baseline" / source).read_text(encoding="utf-8")}
    reference = {source: (folder / "reference" / source).read_text(encoding="utf-8")}
    evaluator_text = (folder / evaluator).read_text(encoding="utf-8")
    gold_patch = _diff(baseline, reference)
    iid = "pilot-" + config.safe_id(spec["language"])
    gold = gold_from_requirement(
        iid,
        {
            "instance_id": iid,
            "repo": "awe-executable-fixtures",
            "base_commit": _sha(baseline[source]),
            "problem_statement": "Clamp both bounds while preserving addition and in-range values.",
            "gold_patch": gold_patch,
            "test_patch": "",
            "FAIL_TO_PASS": f2p,
            "PASS_TO_PASS": p2p,
        },
    )
    mutations = spec["mutations"]
    names = [config.safe_id(m["name"]) for m in mutations]
    if len(names) != len(set(names)):
        raise ValueError("duplicate mutation names")
    result = {
        "language": spec["language"],
        "status": "blocked",
        "passed": False,
        "fixture_sha256": _sha(
            json.dumps(
                {"spec": spec, "baseline": baseline, "reference": reference, "evaluator": evaluator_text},
                sort_keys=True,
            )
        ),
        "evaluator_sha256": _sha(evaluator_text),
        "gold_patch": gold_patch,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
        "cases": [],
    }
    try:
        version = subprocess.run([runtime, "--version"], capture_output=True, text=True, timeout=5, check=True)
        result["runtime"] = {"executable": runtime, "version": (version.stdout + version.stderr).strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = "runtime unavailable: " + str(exc)
        return result
    with tempfile.TemporaryDirectory(prefix="awe-fault-pilot-") as td:
        work = Path(td)
        (work / evaluator).parent.mkdir(parents=True, exist_ok=True)
        (work / evaluator).write_text(evaluator_text, encoding="utf-8")
        previous = set()

        def case(name, state, expected_shape, expected_types, kind, pair=None):
            nonlocal previous
            for path in previous - set(state):
                (work / path).unlink()
            for path, text in state.items():
                config.safe_relative_path(path)
                if path == evaluator:
                    raise ValueError("mutation cannot alter the evaluator")
                (work / path).parent.mkdir(parents=True, exist_ok=True)
                (work / path).write_text(text, encoding="utf-8")
            previous = set(state)
            # Derive patch evidence from the actual materialized program.
            actual = {p: (work / p).read_text(encoding="utf-8") for p in state}
            patch_text = _diff(baseline, actual)
            row = {
                "name": name,
                "kind": kind,
                "pair": pair,
                "patch": patch_text,
                "source_sha256": {p: _sha(t) for p, t in actual.items()},
                "expected_shape": expected_shape,
                "expected_candidate_types": sorted(expected_types),
                "detection": None,
                "passed": False,
                **_execute(runtime, evaluator, work, f2p, p2p, timeout),
            }
            try:
                stable = (
                    all((work / p).read_text(encoding="utf-8") == text for p, text in actual.items())
                    and (work / evaluator).read_text(encoding="utf-8") == evaluator_text
                )
            except OSError:
                stable = False
            if not stable:
                row.update(
                    status="ungraded",
                    pertest=None,
                    error="source or evaluator changed during evaluation; grade withheld",
                )
            if row["status"] == "graded":
                pt = row["pertest"]
                detection = detect(
                    iid,
                    "pilot",
                    use_trajectory=False,
                    use_judge=False,
                    gold_override=gold,
                    patch_override=patch_from_text("pilot", iid, patch_text),
                    resolved_override=pt["resolved"],
                    pertest_override=pt,
                    pertest_log_override=None,
                )
                row["detection"] = detection
                observed = sorted(c["error_type"] for c in detection["candidates"])
                row["observed_candidate_types"] = observed
                row["passed"] = pt["shape"] == expected_shape and observed == sorted(expected_types)
            result["cases"].append(row)
            return row

        def withhold(row, reason):
            row.update(status="ungraded", pertest=None, detection=None, passed=False, control_error=reason)
            row.pop("observed_candidate_types", None)

        base = case("baseline", baseline, "unfixed", ["no_patch_produced"], "baseline")
        ref = case("reference", reference, "resolved", [], "reference")
        baseline_valid = (
            base["status"] == "graded"
            and set(base["pertest"]["f2p_failed"]) == set(f2p)
            and not base["pertest"]["p2p_broken"]
        )
        reference_valid = ref["status"] == "graded" and ref["pertest"]["resolved"]
        result["baseline_contract_passed"] = baseline_valid
        result["reference_contract_passed"] = reference_valid
        if not baseline_valid or not reference_valid:
            result["error"] = "baseline/reference evaluator contract failed; mutants not executed"
            for row in result["cases"]:
                withhold(row, result["error"])
            return result
        harmless = case(
            "harmless_comment",
            {source: reference[source] + spec["harmless_comment"]},
            "resolved",
            [],
            "harmless_control",
        )
        if harmless["status"] != "graded" or not harmless["pertest"]["resolved"]:
            result["error"] = "harmless control failed; mutants not executed"
            withhold(harmless, result["error"])
            return result
        for mutant in mutations:
            if "redirect_to" in mutant:
                target = config.safe_relative_path(mutant["redirect_to"])
                if target in (source, evaluator):
                    raise ValueError("redirect mutation requires an unused source path")
                state = {**baseline, target: reference[source]}
            else:
                old, new = mutant["old"], mutant["new"]
                if not old or reference[source].count(old) != 1:
                    raise ValueError("mutation must replace exactly one source occurrence")
                state = {source: reference[source].replace(old, new, 1)}
            injected = case(
                mutant["name"],
                state,
                mutant["expected_shape"],
                mutant["expected_candidate_types"],
                "mutant",
                pair=mutant["name"],
            )
            restored = case(
                "restore_after_" + mutant["name"], reference, "resolved", [], "restoration", pair=mutant["name"]
            )
            if restored["status"] != "graded" or not restored["pertest"]["resolved"]:
                result["error"] = "restoration control failed; pair withheld and remaining mutants not executed"
                withhold(injected, result["error"])
                withhold(restored, result["error"])
                return result
    result["passed"] = all(row["passed"] for row in result["cases"])
    result["status"] = "passed" if result["passed"] else "failed"
    return result


def run_pilot(languages=("python", "javascript"), timeout=10):
    languages = list(languages)
    if not languages or len(languages) != len(set(languages)):
        raise ValueError("select nonempty unique fixture languages")
    fixtures = []
    for language in languages:
        if language not in ("python", "javascript"):
            raise ValueError("unsupported bundled fixture language: " + language)
        runtime = sys.executable if language == "python" else shutil.which("node")
        if runtime is None:
            fixtures.append(
                {
                    "language": language,
                    "status": "skipped",
                    "passed": False,
                    "reason": "Node.js runtime unavailable",
                    "cases": [],
                }
            )
        else:
            fixtures.append(run_fixture(FIXTURES / language, runtime, timeout))
    executed = [f for f in fixtures if f["status"] != "skipped"]
    cases = [c for f in fixtures for c in f["cases"]]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, text=True, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=config.PROJECT_ROOT, text=True, timeout=5)
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    code = {p.name: _sha(p.read_text(encoding="utf-8")) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "schema_version": 1,
        "scope": "controlled executable toy fixtures; no human ground truth",
        "provenance": {
            "git_commit": commit,
            "dirty_checkout": dirty,
            "awe_code_sha256": _sha(json.dumps(code, sort_keys=True)),
            "trajectory_schema": SCHEMA_VERSION,
            "requested_languages": languages,
            "timeout_seconds": timeout,
            "forced_environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        },
        "summary": {
            "fixtures_attempted": len(executed),
            "fixtures_skipped": len(fixtures) - len(executed),
            "cases_attempted": len(cases),
            "cases_graded": sum(c["status"] == "graded" for c in cases),
            "cases_passed": sum(c["passed"] for c in cases),
            "ungraded_cases": sum(c["status"] != "graded" for c in cases),
            "passed": bool(executed) and all(f["passed"] for f in executed),
        },
        "fixtures": fixtures,
    }
