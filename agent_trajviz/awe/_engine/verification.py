"""Observed verification metadata, independent of fault detection and scoring.

Counts require a bounded runner summary from a single test command. Missing,
hidden, duplicate-ID, compound-command and conflicting summaries stay unknown.
Recorded state fields are preserved only when explicitly present in a record;
command text never establishes a revision, interpreter or imported artifact.
"""

from __future__ import annotations

import re
from collections import defaultdict
from .trajectory import OBSERVED, is_test_command, _segments, _tokens, _segment_head

SCHEMA_VERSION = 3
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PYTEST = re.compile(r"^\s*=*\s*(.*?)\s+in\s+\d+(?:\.\d+)?(?:s| seconds?)\s*=*\s*$")
_STATUS = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?)")
_UNITTEST = re.compile(r"^Ran (\d+) tests? in \d+(?:\.\d+)?s$")
_UNIT_VERDICT = re.compile(r"^(OK|FAILED)(?: \(([^()]*)\))?$")


def ref(position, step):
    return {"position": position, "step_id": step.step_id, "thread": step.subagent_id, "call_id": step.call_id}


def tool_observations(traj):
    """(call position, call, result position, result, linkage status).

    One pending call per ID is unambiguous. Parallel reuse of an ID poisons
    that ID for the remainder of its thread; guessing which result belongs
    to which call could attach a success to the wrong action. ID-less calls
    require immediately adjacent tool results within their thread.
    """
    if traj is None:
        return []
    threads = defaultdict(list)
    for pos, step in enumerate(traj.steps):
        threads[step.subagent_id].append((pos, step))
    rows = []
    for steps in threads.values():
        pending, poisoned, anonymous = {}, set(), []
        for i, (pos, step) in enumerate(steps):
            if step.type == "tool_call":
                row = [pos, step, None, None, "missing_result"]
                rows.append(row)
                if step.call_id:
                    if step.call_id in poisoned:
                        row[4] = "ambiguous_call_id"
                    elif step.call_id in pending:
                        pending.pop(step.call_id)[4] = "ambiguous_call_id"
                        row[4] = "ambiguous_call_id"
                        poisoned.add(step.call_id)
                    else:
                        pending[step.call_id] = row
                else:
                    anonymous.append(row)
            elif step.type == "tool_result" and not step.call_id:
                if len(anonymous) == 1 and i and steps[i - 1][1] is anonymous[0][1]:
                    anonymous[0][2:5] = [pos, step, "adjacent_result"]
                elif len(anonymous) > 1:
                    for row in anonymous:
                        row[4] = "ambiguous_anonymous_calls"
                anonymous = []
            elif step.type == "tool_result" and step.call_id:
                row = pending.pop(step.call_id, None)
                if row is not None:
                    row[2:5] = [pos, step, "call_id"]
    for row in rows:
        aggregation = (row[3].args or {}).get("output_aggregation") if row[3] else None
        if aggregation and aggregation.get("completion_step") is None:
            row[2:5] = [None, None, "running_session"]
        if row[3] is not None and (row[1].observability != OBSERVED or row[3].observability != OBSERVED):
            row[4] = "unobserved_result"
            row[2:4] = [None, None]
    return sorted(rows, key=lambda row: row[0])


def _single_test_command(command):
    if any(token in command for token in ("$(", "`", "<", ">")):
        return False  # shell substitution/redirection can hide or forge output
    segments = _segments(command)
    if not segments or not is_test_command(segments[-1]):
        return False
    head, args = _segment_head(_tokens(segments[-1]))
    supported = head in ("pytest", "py.test", "runtests.py", "manage.py")
    if re.fullmatch(r"python[\d.]*", head):
        supported = (len(args) >= 2 and args[0] == "-m" and args[1] in ("pytest", "unittest")) or (
            bool(args) and args[0].rsplit("/", 1)[-1] in ("runtests.py", "manage.py")
        )
    if not supported:
        return False  # tox/make/hatch can aggregate several environments/suites
    # Preparation may change directory, but cannot print a fictitious summary.
    for segment in segments[:-1]:
        head, _ = _segment_head(_tokens(segment))
        if head != "cd":
            return False
    return True


def test_summary(output):
    """Parse exact pytest/unittest summaries; absence is not zero execution.

    Multiple or contradictory summaries are unknown, as are incomplete runs.
    unittest errors can be failed imports or executed test bodies, so execution
    counts abstain when the summary cannot distinguish them.
    """
    lines = [_ANSI.sub("", line).strip() for line in (output or "").splitlines()]
    pytest, unit_runs, verdicts = [], [], []
    malformed = False
    for line in lines:
        m = _PYTEST.fullmatch(line)
        if m:
            body = m.group(1).strip("= ")
            if body == "no tests ran":
                pytest.append(
                    {
                        "executed": 0,
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "deselected": 0,
                        "xfailed": 0,
                        "xpassed": 0,
                    }
                )
            else:
                tokens = body.split(", ")
                matches = [_STATUS.fullmatch(token) for token in tokens]
                if all(matches) and any(not m[2].startswith("warning") for m in matches):
                    c = {
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "deselected": 0,
                        "xfailed": 0,
                        "xpassed": 0,
                    }
                    seen = set()
                    for match in matches:
                        name = match[2]
                        name = {"error": "errors", "warning": "warnings"}.get(name, name)
                        if name in seen:
                            malformed = True
                            break
                        seen.add(name)
                        c[name] = int(match[1])
                    else:
                        c["executed"] = (
                            None if c["errors"] else sum(c[k] for k in ("passed", "failed", "xfailed", "xpassed"))
                        )
                        pytest.append(c)
                elif any(matches):
                    malformed = True
        m = _UNITTEST.fullmatch(line)
        if m:
            unit_runs.append(int(m[1]))
        m = _UNIT_VERDICT.fullmatch(line)
        if m:
            verdicts.append((m[1], m[2] or ""))
    if malformed:
        return {"status": "unknown", "reason": "ambiguous_summary"}
    if len(pytest) == 1 and not unit_runs and not verdicts:
        c = pytest[0]
        return {"status": "reported", "runner": "pytest", "counts": c}
    if not pytest and len(unit_runs) == len(verdicts) == 1:
        total, (outcome, detail) = unit_runs[0], verdicts[0]
        fields = {}
        for item in detail.split(", ") if detail else []:
            m = re.fullmatch(r"(skipped|failures|errors|expected failures|unexpected successes)=(\d+)", item)
            if not m or m[1] in fields:
                return {"status": "unknown", "reason": "ambiguous_summary"}
            fields[m[1]] = int(m[2])
        skipped, errors = fields.get("skipped", 0), fields.get("errors", 0)
        if outcome == "FAILED" and not any(fields.get(k, 0) for k in ("failures", "errors", "unexpected successes")):
            return {"status": "unknown", "reason": "conflicting_summary"}
        if sum(fields.values()) > total or (
            outcome == "OK" and any(fields.get(k, 0) for k in ("failures", "errors", "unexpected successes"))
        ):
            return {"status": "unknown", "reason": "conflicting_summary"}
        return {
            "status": "reported",
            "runner": "unittest",
            "counts": {
                "total": total,
                "executed": None if errors else total - skipped,
                "passed": None,
                "failed": fields.get("failures", 0),
                "skipped": skipped,
                "errors": errors,
                "expected_failures": fields.get("expected failures", 0),
                "unexpected_successes": fields.get("unexpected successes", 0),
            },
        }
    return {"status": "unknown", "reason": "absent_or_multiple_summaries"}


def _recorded_state(call, result):
    records = []
    conflicts = []
    for name, step in (("call", call), ("result", result)):
        if step and step.observability == OBSERVED and (step.args or {}).get("verification_state_conflicts"):
            conflicts.extend(step.args["verification_state_conflicts"])
        state = (step.args or {}).get("verification_state") if step else None
        if step is None or step.observability != OBSERVED or not isinstance(state, dict):
            continue
        fields = {
            k: v for k, v in state.items() if k in ("cwd", "revision", "interpreter", "import_paths", "artifact_hashes")
        }
        if fields:
            records.append({"source": name, "fields": fields})
    # Two records retain conflicts rather than merging them into a fictitious state.
    return {
        "status": "conflicting" if conflicts else ("recorded" if records else "not_recorded"),
        "records": records,
        "conflicts": conflicts,
    }


def _recorded_tests(call, result):
    records = []
    for step in (call, result):
        if step is None or step.observability != OBSERVED:
            continue
        args = step.args or {}
        if args.get("verification_tests_conflicts"):
            return {"status": "unknown", "reason": "conflicting_recorded_tests"}
        record = args.get("verification_tests")
        if record is not None:
            records.append(record)
    if not records:
        return {"status": "unknown", "reason": "tests_not_recorded"}
    if any(r != records[0] for r in records[1:]):
        return {"status": "unknown", "reason": "conflicting_recorded_tests"}
    record = records[0]
    if not isinstance(record, dict) or type(record.get("complete")) is not bool:
        return {"status": "unknown", "reason": "invalid_recorded_tests"}
    tests = record.get("results")
    if not isinstance(tests, list):
        return {"status": "unknown", "reason": "invalid_recorded_tests"}
    seen, normalized = set(), []
    for item in tests:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("test_id"), str)
            or not item["test_id"].strip()
            or item["test_id"] in seen
            or item.get("status") not in ("passed", "failed", "skipped", "error", "not_run")
        ):
            return {"status": "unknown", "reason": "invalid_recorded_tests"}
        seen.add(item["test_id"])
        normalized.append({"test_id": item["test_id"], "status": item["status"]})
    return {"status": "recorded", "complete": record["complete"], "results": normalized}


def _typed_summary(tests):
    if tests.get("status") != "recorded" or not tests.get("complete"):
        return None
    counts = {key: 0 for key in ("passed", "failed", "skipped", "errors", "not_run")}
    for test in tests["results"]:
        counts["errors" if test["status"] == "error" else test["status"]] += 1
    counts["executed"] = None if counts["errors"] else counts["passed"] + counts["failed"]
    return {"status": "reported", "runner": "explicit_test_records", "counts": counts}


def verification_observations(traj):
    runs = []
    for cp, call, rp, result, linkage in tool_observations(traj):
        command = str((call.args or {}).get("command") or "")
        if not call.test_run or not is_test_command(command):
            continue
        summary = {"status": "unknown", "reason": linkage}
        if result is not None:
            summary = (
                test_summary(result.output_excerpt)
                if _single_test_command(command)
                else {"status": "unknown", "reason": "compound_command"}
            )
        result_ref = ref(rp, result) if result else None
        aggregation = (result.args or {}).get("output_aggregation") if result else None
        state_result = result
        if aggregation:
            completion = next(
                (
                    (p, s)
                    for p, s in enumerate(traj.steps)
                    if s.step_id == aggregation["completion_step"] and s.subagent_id == result.subagent_id
                ),
                None,
            )
            if completion:
                result_ref = ref(*completion)
                state_result = completion[1]
        tests = _recorded_tests(call, state_result) if result else {"status": "unknown", "reason": linkage}
        if tests.get("reason") in ("invalid_recorded_tests", "conflicting_recorded_tests"):
            summary = {"status": "unknown", "reason": tests["reason"]}
        typed = _typed_summary(tests)
        if typed is not None:
            if summary.get("status") == "reported" and any(
                summary.get("counts", {}).get(k) != typed["counts"].get(k)
                for k in ("passed", "failed", "skipped", "errors")
                if summary.get("counts", {}).get(k) is not None
            ):
                summary = {"status": "unknown", "reason": "conflicting_summary_and_tests"}
            else:
                summary = typed
        code = (result.args or {}).get("exit_code") if result else None
        runs.append(
            {
                "call": ref(cp, call),
                "result": result_ref,
                "output_aggregation": aggregation,
                "command": command,
                "linkage": "session_chunks" if aggregation else linkage,
                "exit_code": code if type(code) is int else None,
                "summary": summary,
                "tests": tests,
                "state": _recorded_state(call, state_result),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "available": traj is not None,
        "runs": runs,
        "scope": "observed runner summaries; absent counts/state remain unknown",
    }


def known_zero_test_calls(traj):
    """Object identities of calls with unambiguous reported zero execution.

    Unknown/missing outputs retain the existing intent-based credit policy.
    This helper never treats a test attempt as absent from the trajectory.
    """
    if traj is None:
        return set()
    return {
        id(traj.steps[r["call"]["position"]])
        for r in verification_observations(traj)["runs"]
        if r["summary"].get("status") == "reported"
        and type((r["summary"].get("counts") or {}).get("executed")) is int
        and r["summary"]["counts"]["executed"] == 0
    }
