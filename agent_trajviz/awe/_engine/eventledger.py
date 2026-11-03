"""Offline ledger of observed failures and later successes on the same action.

This records protocol links and recovery evidence, not semantic dependencies,
terminal impact, or causal attribution. Repeated failures on the same target
form one episode until an observed later success closes it. Different threads
have independent timelines even when their native call IDs collide.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from .trajectory import OBSERVED, EDIT, WRITE, RUN_COMMAND, RUN_TEST, classify_tool_error, looks_like_test_failure


def _ref(position, step):
    return {"position": position, "step_id": step.step_id, "thread": step.subagent_id, "call_id": step.call_id}


def _identity(call):
    command = str((call.args or {}).get("command") or "").strip()
    files = tuple(sorted(set(call.files or [])))
    if not command and not files:
        return None  # no identifiable target: never infer recovery
    return call.canonical_action, call.native_tool, command, files


def _failure(call, result):
    code = (result.args or {}).get("exit_code")
    if call.test_run or call.canonical_action == RUN_TEST:
        failed = (
            code != 0 if type(code) is int else (result.tool_error or looks_like_test_failure(result.output_excerpt))
        )
        return bool(failed), "test_failure"
    if result.tool_error:
        return True, "tool_error"
    if type(code) is int:
        return code != 0, "nonzero_exit"
    subtype = (
        classify_tool_error(result.output_excerpt) if call.canonical_action in (RUN_COMMAND, EDIT, WRITE) else None
    )
    return bool(subtype), "tool_error"


def event_ledger(traj):
    if traj is None:
        return {"schema_version": 1, "available": False, "events": [], "counts": {}, "missing_result_calls": 0}
    threads = defaultdict(list)
    for position, step in enumerate(traj.steps):
        threads[step.subagent_id].append((position, step))
    events, missing = [], 0
    for thread in threads.values():
        results = defaultdict(deque)
        for position, step in thread:
            if step.type == "tool_result" and step.call_id:
                results[step.call_id].append((position, step))
        pending = {}
        for index, (position, call) in enumerate(thread):
            if call.type != "tool_call":
                continue
            identity = _identity(call)
            result_pair = None
            if call.call_id:
                queue = results[call.call_id]
                while queue and queue[0][0] <= position:
                    queue.popleft()
                if queue:
                    result_pair = queue.popleft()
            else:
                result_pair = next(
                    ((p, s) for p, s in thread[index + 1 : index + 3] if s.type == "tool_result" and not s.call_id),
                    None,
                )
            if result_pair is None or result_pair[1].observability != OBSERVED:
                missing += 1
                if identity in pending:
                    pending[identity]["unobserved_followups"] += 1
                continue
            result_pos, result = result_pair
            failed, kind = _failure(call, result)
            old = pending.get(identity) if identity is not None else None
            # A parallel call started before the failure became visible cannot
            # demonstrate a response/recovery from that failure.
            ordered = old is not None and position > old["latest_failure"]["result"]["position"]
            call_ref, result_ref = _ref(position, call), _ref(result_pos, result)
            edge = {"from": call_ref, "to": result_ref, "relation": "tool_result", "basis": "observed_protocol"}
            if failed:
                occurrence = {
                    "call": call_ref,
                    "result": result_ref,
                    "exit_code": (result.args or {}).get("exit_code"),
                    "tool_error": result.tool_error,
                    "excerpt": result.output_excerpt[:180],
                }
                if ordered:
                    old["failures"].append(occurrence)
                    old["latest_failure"] = occurrence
                    old["links"].append(edge)
                    continue
                event = {
                    "event_id": f"event:{position}",
                    "kind": kind,
                    "action": call.canonical_action,
                    "command": (call.args or {}).get("command", ""),
                    "files": sorted(set(call.files or [])),
                    "origin": call_ref,
                    "failures": [occurrence],
                    "latest_failure": occurrence,
                    "lifecycle": "no_recovery_observed",
                    "recovery": None,
                    "unobserved_followups": 0,
                    "links": [edge],
                    "terminal_relevance": "unknown",
                    "causal_validation": "not_tested",
                }
                events.append(event)
                if identity is not None:
                    pending[identity] = event
            elif ordered:
                old["lifecycle"] = "later_success_observed"
                old["recovery"] = {"call": call_ref, "result": result_ref, "excerpt": result.output_excerpt[:180]}
                old["links"].extend(
                    [
                        edge,
                        {
                            "from": old["latest_failure"]["result"],
                            "to": call_ref,
                            "relation": "same_action_later_success",
                            "basis": "matching_action_target_and_thread",
                        },
                    ]
                )
                del pending[identity]
    return {
        "schema_version": 1,
        "available": True,
        "events": events,
        "counts": dict(Counter(e["lifecycle"] for e in events)),
        "missing_result_calls": missing,
        "scope": "all_threads",
        "causal_validation": "not_tested",
    }
