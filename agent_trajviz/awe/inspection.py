"""Reference-free inspection of an already-loaded trajectory using AWE evidence.

This module reads the supplied snapshot only. It does not reload sources, consult
benchmark data, run commands, contact a model, or assign task-correctness blame.
"""
from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any

from .native import _execution_tool, project_native

_SCHEMA_VERSION = 1
_LIMITATIONS = [
    "These are observations from the loaded recording, not correctness attribution.",
    "A later successful action does not establish that the task was solved or that a remedy caused improvement.",
    "Unrecorded commands, hidden output, and external test runs remain unknown.",
    "File interactions come from recognized recorded arguments; shell scripts can touch additional files.",
    "Stored source fingerprints identify the loaded snapshot and are not freshly verified against disk.",
]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _records(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _parts(step: dict) -> list[dict]:
    """Use canonical calls once, preserving original part locations when known."""
    parts = _records(step.get("parts"))
    calls = _records(step.get("tool_calls"))
    if not calls:
        return [{**part, "_inspection_original_part": index} for index, part in enumerate(parts)]
    remaining = list(enumerate(calls))
    merged = []
    for part_index, part in enumerate(parts):
        if part.get("type") not in {"tool_call", "tool"} and not part.get("tool_name"):
            merged.append({**part, "_inspection_original_part": part_index})
            continue
        identity = str(part.get("tool_id") or part.get("id") or "")
        match = next(((index, call) for index, call in remaining
                      if identity and str(call.get("tool_id") or call.get("id") or "") == identity), None)
        if match is None and not identity:
            match = next(((index, call) for index, call in remaining
                          if not (call.get("tool_id") or call.get("id"))
                          and call.get("tool_name") == part.get("tool_name")
                          and call.get("input") == part.get("input")), None)
        if match is not None:
            remaining.remove(match)
            index, call = match
            merged.append({**part, **call, "_inspection_original_part": part_index,
                           "_inspection_tool_index": index})
    merged.extend({**call, "_inspection_original_part": None, "_inspection_tool_index": index}
                  for index, call in remaining)
    return merged


def _projection(run, steps: list[dict], allowed: set[int]):
    """Keep result pairing local to each canonical tool part, across all threads."""
    from agent_trajviz.insight.telemetry_quality import tool_field_observed
    snapshot = []
    originals, tools = {}, {}
    session_ids = {s.get("session_id") for s in steps if isinstance(s.get("session_id"), str) and s["session_id"]}
    thread_keys = {}
    for position, source in enumerate(steps):
        sid = source.get("session_id") if isinstance(source.get("session_id"), str) else ""
        agent = source.get("agent") if isinstance(source.get("agent"), str) else ""
        if getattr(run, "format", "") == "claude_code" and source.get("is_sub_agent"):
            thread_keys[position] = f"{sid or 'session'}::child:{agent}" if agent else f"unknown-child:{position}"
        elif sid:
            thread_keys[position] = sid if len(session_ids) > 1 else None
        elif source.get("is_sub_agent"):
            thread_keys[position] = agent or f"unknown-child:{position}"
        else:
            thread_keys[position] = None
    for position, source in enumerate(steps):
        step = copy.deepcopy(source)
        # Projection positions are internal; only an unambiguous source index
        # earns a link back to Workflow. Never manufacture one for custom data.
        index = source.get("index")
        originals[position] = index if type(index) is int and index in allowed else None
        step["index"] = position
        step.setdefault("role", "system")
        parts = _parts(step)
        for part_index, part in enumerate(parts):
            if part.get("type") in {"tool", "tool_call"} or part.get("tool_name"):
                tools[position, part_index] = copy.deepcopy(part)
                part["_inspection_native_call_id"] = part.get("tool_id") or part.get("id") or None
                part["tool_id"] = f"inspection:{position}:{part_index}"
                if not tool_field_observed(part, "tool_inputs"):
                    part["input"] = {}
                if not tool_field_observed(part, "tool_outputs"):
                    part["output"] = None
        step["parts"], step["tool_calls"] = parts, []
        snapshot.append(step)
    proxy = SimpleNamespace(steps=snapshot, source=getattr(run, "source", None),
                            sha256=getattr(run, "sha256", None), format=getattr(run, "format", "unknown"))
    trajectory = project_native(proxy, str(proxy.format), "loaded-session", verify_source=False)
    for event in trajectory.steps:
        source = event.source_ref
        position, part = source.get("workflow_step"), source.get("part")
        source["projection_row"] = position
        source["workflow_step"] = originals.get(position)
        source["workflow_index"] = originals.get(position)
        tool = tools.get((position, part), {})
        source["part_index"] = tool.get("_inspection_original_part", part)
        source["tool_call_index"] = tool.get("_inspection_tool_index")
        event.subagent_id = thread_keys.get(position)
        source["thread_id"] = event.subagent_id or "main"
        source["session_id"] = steps[position].get("session_id") if type(position) is int else None
        source["native_call_id"] = tool.get("tool_id") or tool.get("id") or None
        source["outcome_observed"] = tool_field_observed(tool, "tool_outcomes")
        if event.type in {"tool_call", "tool_result"}:
            # A canonical part already contains its own native result. A pending
            # call reusing an ID must not consume another part's later result.
            event.call_id = f"inspection:{position}:{part}"
    return trajectory, tools


def _source_ref(trajectory, ref: dict | None) -> dict:
    if not isinstance(ref, dict):
        return {}
    position = ref.get("position")
    if type(position) is not int or not 0 <= position < len(trajectory.steps):
        return {}
    event = trajectory.steps[position]
    source = event.source_ref
    return {"workflow_index": source.get("workflow_index"), "part_index": source.get("part_index"),
            "tool_call_index": source.get("tool_call_index"),
            "thread_id": source.get("thread_id", "main"), "call_id": source.get("native_call_id"),
            "awe_step_id": event.step_id}


def _enrich(trajectory, value):
    """Add explicit Workflow references without relabeling AWE's own step IDs."""
    if isinstance(value, list):
        return [_enrich(trajectory, item) for item in value]
    if isinstance(value, dict):
        result = {key: _enrich(trajectory, item) for key, item in value.items()}
        if "position" in value and "step_id" in value:
            result.update(_source_ref(trajectory, value))
        return result
    return value


def _indices(*values) -> list[int]:
    return list(dict.fromkeys(index for value in values for index in
                             (value if isinstance(value, list) else [value]) if type(index) is int))


def inspect_trajectory(run) -> dict[str, Any]:
    """Inspect the loaded session without reference files, execution, or a model.

    ``run`` is a trajectory returned by :func:`agent_trajviz.load`. The result is
    a JSON-serializable observation report. Workflow references are supplied only
    for unique canonical step indices; local AWE event IDs are kept separate.
    No step or source artifact is modified.
    """
    from ._engine.eventledger import event_ledger
    from ._engine.symptoms import analyze
    from ._engine.trajectory import EDIT, HIDDEN, READ, RUN_COMMAND, SEARCH, WRITE
    from ._engine.verification import test_summary, tool_observations
    from agent_trajviz.insight.telemetry_quality import field_observed, tool_field_observed
    from agent_trajviz.insight.tool_failure import tool_call_failed

    steps = _records(getattr(run, "steps", None))
    frequencies = Counter(step.get("index") for step in steps if type(step.get("index")) is int)
    allowed = {index for index, count in frequencies.items() if count == 1}
    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION, "mode": "trajectory_only", "available": bool(steps),
        "source": {"format": str(getattr(run, "format", "unknown")),
                   "name": str(getattr(run, "source", "") or "Loaded session").replace("\\", "/").rsplit("/", 1)[-1],
                   "sha256": getattr(run, "sha256", None), "identity_basis": "loaded_snapshot"},
        "workflow_indices": [step["index"] for step in steps if type(step.get("index")) is int and step["index"] in allowed],
        "summary": {}, "counts": {}, "findings": [], "files": [], "tests": [], "failures": [],
        "recovery": [], "timing": [], "ledger": {}, "symptoms": {}, "coverage": {},
        "outcome": {"status": "unknown", "basis": "No independent task/evaluator evidence was supplied."},
        "limitations": list(_LIMITATIONS),
    }
    if not steps:
        report["reason"] = "Load a trajectory to inspect its recorded workflow."
        return report

    trajectory, tools = _projection(run, steps, allowed)
    # Outcome coverage is authoritative. Missing outcomes cannot be promoted to
    # a failure or a successful recovery merely because a numeric placeholder
    # or quoted output happens to resemble a result.
    observations = tool_observations(trajectory)
    known_failure_keys = set()
    for _cp, call, _rp, _result, _linkage in observations:
        source = call.source_ref
        key = source.get("projection_row"), source.get("part")
        tool = tools.get(key, {})
        if source.get("outcome_observed") and tool_call_failed(tool):
            known_failure_keys.add(key)
    ledger_trajectory = copy.deepcopy(trajectory)
    for event in ledger_trajectory.steps:
        if event.type != "tool_result":
            continue
        key = event.source_ref.get("projection_row"), event.source_ref.get("part")
        if not event.source_ref.get("outcome_observed"):
            event.observability = HIDDEN
        elif key not in known_failure_keys:
            code = event.args.get("exit_code")
            if (_execution_tool(event.native_tool) and code != 0) or (type(code) is int and code != 0):
                # A completed API response alone does not prove that its shell
                # process exited successfully (it may still be running).
                event.observability = HIDDEN
            # AWE's reference-backed ledger may flag error-looking prose. This
            # inspection counts explicit recorded outcomes only: quoted traces
            # in successful output cannot become new failed calls.
            event.output_excerpt = ""
    ledger = _enrich(trajectory, event_ledger(ledger_trajectory))
    for episode in ledger.get("events", []):
        recovery = episode.get("recovery")
        if recovery:
            position = recovery.get("result", {}).get("position")
            if type(position) is int and 0 <= position < len(trajectory.steps):
                recovery["excerpt"] = trajectory.steps[position].output_excerpt[:180]
    ledger["failure_basis"] = "recorded_status_error_or_exit"
    symptoms = _enrich(trajectory, analyze(trajectory))
    report["ledger"], report["symptoms"] = ledger, symptoms

    failures, by_position = {}, {}
    for cp, call, rp, result, _linkage in observations:
        source = _source_ref(trajectory, {"position": cp})
        tool = tools.get((call.source_ref.get("projection_row"), call.source_ref.get("part")), {})
        by_position[cp] = (call, result, tool)
        if (call.source_ref.get("projection_row"), call.source_ref.get("part")) in known_failure_keys:
            failures[cp] = {
                **source, "tool_name": call.native_tool, "command": str(call.args.get("command") or ""),
                "exit_code": (result.args.get("exit_code") if result else None),
                "status": str(tool.get("status") or "unknown"),
                "output_excerpt": str(result.output_excerpt if result else tool.get("error") or "")[:700],
                "result_workflow_index": _source_ref(trajectory, {"position": rp}).get("workflow_index"),
            }
    for episode in ledger.get("events", []):
        origin = episode.get("origin") or {}
        call, result, _tool = by_position.get(origin.get("position"), (None, None, {}))
        for occurrence in episode.get("failures", []):
            ref = occurrence.get("call") or {}
            cp = ref.get("position")
            if cp not in failures and cp in by_position:
                c, r, t = by_position[cp]
                failures[cp] = {
                    **_source_ref(trajectory, ref), "tool_name": c.native_tool,
                    "command": str(c.args.get("command") or ""), "exit_code": occurrence.get("exit_code"),
                    "status": str(t.get("status") or "unknown"), "output_excerpt": occurrence.get("excerpt") or "",
                    "result_workflow_index": _source_ref(trajectory, occurrence.get("result")).get("workflow_index"),
                }
        recovery = episode.get("recovery") or {}
        recovery_ref = recovery.get("call") or {}
        failure_steps = [o.get("call", {}).get("workflow_index") for o in episode.get("failures", [])]
        report["recovery"].append({
            **_source_ref(trajectory, origin), "command": episode.get("command") or "",
            "tool_name": call.native_tool if call else "", "files": list(episode.get("files") or []),
            "failures": len(episode.get("failures", [])), "lifecycle": episode.get("lifecycle"),
            "recovery_workflow_index": recovery_ref.get("workflow_index"),
            "step_indices": _indices(failure_steps, recovery_ref.get("workflow_index")),
            "output_excerpt": recovery.get("excerpt") or episode.get("latest_failure", {}).get("excerpt") or "",
        })
    report["failures"] = [failures[key] for key in sorted(failures)]

    for row in symptoms.get("verification", {}).get("runs", []):
        summary = row.get("summary") or {}
        counts = summary.get("counts") if summary.get("status") == "reported" else None
        failed = any(type((counts or {}).get(k)) is int and counts[k] > 0 for k in ("failed", "errors"))
        result = row.get("result")
        cp = (row.get("call") or {}).get("position")
        _call, event, _tool = by_position.get(cp, (None, None, {}))
        output_summary = None
        if summary.get("status") != "reported" and event is not None:
            candidate = test_summary(event.output_excerpt)
            if candidate.get("status") == "reported":
                output_summary = {**candidate, "basis": "unattributed_output_text"}
        process_code = row.get("exit_code")
        process_status = ("nonzero_exit" if type(process_code) is int and process_code != 0
                          else "zero_exit" if type(process_code) is int else
                          str((_tool or {}).get("status") or "unknown"))
        report["tests"].append({
            **_source_ref(trajectory, row.get("call")), "command": row.get("command") or "",
            "exit_code": row.get("exit_code"), "runner": summary.get("runner"),
            "summary_status": summary.get("status", "unknown"), "counts": counts,
            "status": "reported_failures" if failed else "reported_summary" if counts is not None else
                      "output_summary_only" if output_summary else "unknown",
            "process_status": process_status, "output_summary": output_summary,
            "output_excerpt": str(event.output_excerpt if event else "")[-700:],
            "result_workflow_index": (result or {}).get("workflow_index"),
            "reason": summary.get("reason"),
        })

    files = {}
    for cp, call, _rp, _result, _linkage in observations:
        action = call.canonical_action
        if action == RUN_COMMAND and call.files:
            # The native projection's command files are recognized read targets.
            action = READ
        if action not in {READ, EDIT, WRITE, SEARCH}:
            continue
        source = _source_ref(trajectory, {"position": cp})
        for path in call.files:
            if not isinstance(path, str) or not path:
                continue
            row = files.setdefault(path, {"path": path, "actions": [], "step_indices": [], "threads": []})
            if action not in row["actions"]:
                row["actions"].append(action)
            row["step_indices"] = _indices(row["step_indices"], source.get("workflow_index"))
            if source["thread_id"] not in row["threads"]:
                row["threads"].append(source["thread_id"])
    for row in files.values():
        row["first_step"] = row["step_indices"][0] if row["step_indices"] else None
        row["last_step"] = row["step_indices"][-1] if row["step_indices"] else None
    report["files"] = list(files.values())

    timed = 0
    for step in steps:
        duration = _number(step.get("duration"))
        if duration is None or not field_observed(step, "duration"):
            continue
        timed += 1
        if duration >= 15:
            report["timing"].append({"workflow_index": step.get("index") if type(step.get("index")) is int and step["index"] in allowed else None,
                                     "duration_seconds": duration, "timing_source": str(step.get("timing_source") or "unknown"),
                                     "incomplete": not bool(step.get("time_completed_ms"))})
    report["timing"].sort(key=lambda row: row["duration_seconds"], reverse=True)

    observed_outputs = sum(result is not None for _, _, _, result, _ in observations)
    observed_outcomes = 0
    for _cp, call, _rp, result, _linkage in observations:
        tool = tools.get((call.source_ref.get("projection_row"), call.source_ref.get("part")), {})
        code = result.args.get("exit_code") if result else None
        status = str(tool.get("status") or "").lower()
        if call.source_ref.get("outcome_observed") and (
            type(code) is int or tool_call_failed(tool) or status not in {"", "pending", "running", "unknown"}
        ):
            observed_outcomes += 1
    input_calls = sum(tool_field_observed(tool, "tool_inputs") and isinstance(tool.get("input"), dict)
                      for tool in tools.values())
    summary = {"steps": len(steps), "tool_calls": len(observations), "tool_results": observed_outputs,
               "failed_calls": len(failures), "missing_results": len(observations) - observed_outputs,
               "failure_episodes": len(report["recovery"]),
               "later_successes": sum(row["lifecycle"] == "later_success_observed" for row in report["recovery"]),
               "files": len(files), "edited_files": sum(bool(set(row["actions"]) & {EDIT, WRITE}) for row in files.values()),
               "test_runs": len(report["tests"]), "test_summaries_reported": sum(row["summary_status"] == "reported" for row in report["tests"]),
               "test_summary_texts": sum(bool(row["output_summary"]) for row in report["tests"]),
               "threads": len({event.subagent_id or "main" for event in trajectory.steps}), "slow_steps": len(report["timing"])}
    report["summary"] = report["counts"] = summary
    report["coverage"] = {"tool_inputs": {"observed": input_calls, "total": len(observations)},
                          "tool_outputs": {"observed": observed_outputs, "total": len(observations)},
                          "tool_outcomes": {"observed": observed_outcomes, "total": len(observations)},
                          "step_timing": {"observed": timed, "total": len(steps)},
                          "test_summaries": {"observed": summary["test_summaries_reported"], "total": summary["test_runs"]},
                          "capture_complete": None}

    def add(kind, title, observation, next_check, rows):
        evidence = [{key: row.get(key) for key in ("workflow_index", "part_index", "thread_id", "call_id", "output_excerpt")}
                    for row in rows]
        report["findings"].append({"id": f"inspection:{kind}", "kind": kind, "severity": "warning" if kind in
                                   {"tool_failures", "reported_test_failures", "zero_tests_executed"} else "info",
                                   "title": title, "observation": observation, "next_check": next_check,
                                   "step_indices": _indices([row.get("workflow_index") for row in rows]),
                                   "evidence": evidence, "status": "observed"})
    if failures:
        add("tool_failures", f"{len(failures)} tool calls recorded failures",
            "The recording contains failed statuses, nonzero exits, or recognizable tool-error evidence.",
            "Inspect the recorded error and surrounding actions before deciding whether it needs a change.", report["failures"])
    recovered = [row for row in report["recovery"] if row["lifecycle"] == "later_success_observed"]
    if recovered:
        add("later_success", f"{len(recovered)} failed actions have a later successful call",
            "The same command or file action later completed successfully in the same thread; task recovery remains unknown.",
            "Compare the failure and later result to check which observed changes were relevant.", recovered)
        report["findings"][-1]["step_indices"] = _indices(*(r["step_indices"] for r in recovered))
    if summary["missing_results"]:
        missing = [_source_ref(trajectory, {"position": cp}) for cp, _, _, result, _ in observations if result is None]
        add("missing_results", f"{summary['missing_results']} tool calls have no linked observed result",
            "The loaded snapshot does not contain an unambiguous observed result for these calls.",
            "Check whether the session was interrupted, output was hidden, or additional logs are available.", missing)
    failing_tests = [row for row in report["tests"] if row["status"] == "reported_failures"]
    if failing_tests:
        add("reported_test_failures", f"{len(failing_tests)} test commands reported failures or errors",
            "The captured runner summaries explicitly report failing tests or errors; this is not an independent task evaluation.",
            "Open the test output and compare any later test commands and changed files.", failing_tests)
    elif not report["tests"]:
        add("no_visible_tests", "No recognized test command is visible",
            "No supported test invocation is identifiable in the captured tool calls; external or omitted tests remain unknown.",
            "Check whether relevant validation was performed outside the recording or by a custom runner.", [])
    output_summaries = [row for row in report["tests"] if row["output_summary"]]
    if output_summaries:
        add("output_summary_text", f"Output contains test-summary text for {len(output_summaries)} commands",
            "These summaries occur in recorded output, but cannot be attributed to one supported test invocation. They do not establish test execution or task correctness.",
            "Inspect the command and full output; a direct test invocation can provide attributable runner evidence.", output_summaries)
    for kind, rows in _symptom_groups(symptoms).items():
        evidence = []
        for row in rows:
            source = row.get("call") or {}
            evidence.append({**source, "output_excerpt": str(row.get("evidence") or "")[:500]})
        label = {"verification_evidence_unknown": "test commands need output review",
                 "unanswered_tool_call": "tool calls need result review",
                 "ambiguous_tool_linkage": "tool calls have ambiguous result links",
                 "zero_tests_executed": "test commands reported zero execution",
                 "all_tests_skipped": "test commands reported only skipped tests"}.get(kind)
        title = f"{len(rows)} {label}" if label else f"{len(rows)} recorded {kind.replace('_', ' ')} observations"
        add(kind, title,
            "AWE found a recorded verification or evidence pattern for review; it is unscored and has unknown task impact.",
            "Review the cited command, result and evidence fields before changing the workflow.", evidence)
    if report["timing"]:
        add("slow_steps", f"{len(report['timing'])} recorded steps lasted at least 15 seconds",
            "Elapsed durations are recorded; they do not isolate model, network, queue, or tool causes.",
            "Inspect the timing source and recorded work in these steps before attributing the delay.", report["timing"][:5])
    if not report["findings"]:
        add("recording_summary", "Recorded workflow is ready to inspect",
            "No supported failure or verification symptom was identified in the available evidence; task correctness remains unknown.",
            "Review the file actions, commands and recording coverage for this session.", [])
    return report


def _symptom_groups(symptoms: dict) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in _records(symptoms.get("symptoms")):
        groups[str(row.get("kind") or "evidence_gap")].append(row)
    return dict(groups)
