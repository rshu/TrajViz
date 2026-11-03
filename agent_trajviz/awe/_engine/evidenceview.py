"""Bounded, source-linked process views; never consume reference patches/grades."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .trajectory import HIDDEN, OBSERVED
from .verification import tool_observations

VIEW_VERSION = 2
FOCUS_ACTIONS = {
    "code_localization": {"search", "read"},
    "code_editing": {"edit", "write"},
    "code_verification": {"run_test", "run_command"},
    "self_repair_loop": {"edit", "write", "run_test"},
    "tool_use": {"other", "run_command"},
}


def excerpt(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    if limit < 5:
        return text[:limit]
    head = (limit - 3) * 2 // 3
    return text[:head] + " … " + text[-(limit - 3 - head) :]


def _row(position, step, limit):
    if step.observability != OBSERVED:
        return None
    if step.type == "tool_result":
        kind, text = "feedback", step.output_excerpt
    elif step.type == "tool_call":
        kind = "plans" if step.canonical_action == "plan" else step.canonical_action
        if kind == "plans":
            text = "; ".join(
                "[%s] %s" % (it.get("status"), it.get("content"))
                for it in (step.args or {}).get("items", [])
                if isinstance(it, dict)
            )
        else:
            text = json.dumps({"args": step.args, "files": step.files}, ensure_ascii=False, sort_keys=True)
    elif step.type == "reasoning":
        kind, text = "thinks", step.text
    elif step.type in ("message", "final_answer"):
        kind, text = ("instruction" if step.actor in ("user", "system") else "says"), step.text
    else:
        return None
    body = excerpt(text, limit)
    return {
        "step": step.step_id,
        "position": position,
        "kind": kind,
        "thread": step.subagent_id,
        "call_id": step.call_id,
        "tool": step.native_tool,
        "status": step.status,
        "source_ref": step.source_ref,
        "text": body,
        "output_aggregation": (step.args or {}).get("output_aggregation"),
        "text_chars": len(str(text or "")),
        "text_truncated": body != str(text or ""),
    }


def build_view(problem_statement, traj, max_rows=28, max_text=220, focus_steps=None, focus_cap=None):
    """Select atomic call/result units, instructions/plans and temporal coverage.

    Selection uses only observed process, never gold file names or final grader
    results. Arbitrary middle evidence remains retrievable by stable step IDs.
    Full task constraints are retained separately from the process row budget.
    """
    if type(max_rows) is not int or max_rows < 1 or type(max_text) is not int or max_text < 8:
        raise ValueError("positive row budget and text budget >=8 required")
    rows = {p: r for p, s in enumerate(traj.steps) if (r := _row(p, s, max_text)) is not None}
    linked, owned = [], set()
    for cp, call, rp, result, linkage in tool_observations(traj):
        group = tuple(p for p in (cp, rp) if p in rows)
        if group:
            linked.append(group)
            owned.update(group)
    linked.extend((p,) for p in rows if p not in owned)
    focus = set(focus_steps or [])
    actions = FOCUS_ACTIONS.get(focus_cap, set())
    priority = {
        "instruction": 9,
        "plans": 9,
        "thinks": 6,
        "says": 5,
        "run_test": 7,
        "edit": 4,
        "write": 4,
        "search": 2,
        "read": 2,
    }
    n = len(traj.steps)

    def rank(group):
        score = max(priority.get(rows[p]["kind"], 1) for p in group)
        if any(traj.steps[p].tool_error for p in group):
            score += 7
        if any(traj.steps[p].canonical_action in actions for p in group):
            score += 10
        if any(rows[p]["step"] in focus for p in group):
            score += 100
        if min(group) < 3 or max(group) >= n - 3:
            score += 4
        return (-score, min(group))

    selected = set()
    for group in sorted(linked, key=rank):
        if len(selected) + len(group) <= max_rows:
            selected.update(group)
    kept = [rows[p] for p in sorted(selected)]
    omitted = [rows[p] for p in sorted(set(rows) - selected)]
    manifest = {
        "view_version": VIEW_VERSION,
        "row_budget": max_rows,
        "text_budget": max_text,
        "total_steps": n,
        "eligible_rows": len(rows),
        "retained_rows": len(kept),
        "omitted_rows": len(omitted),
        "omitted_by_kind": dict(Counter(r["kind"] for r in omitted)),
        "omitted_step_ids": [r["step"] for r in omitted],
        "truncated_step_ids": [r["step"] for r in kept if r["text_truncated"]],
        "hidden_or_unrenderable_steps": n - len(rows),
        "focus_steps": sorted(focus),
        "focus_capability": focus_cap,
        "source": dict(traj.source),
        "task_truncated": False,
    }
    return {
        "problem_statement": problem_statement or "",
        "rows": kept,
        "hidden_reasoning": any(s.observability == HIDDEN for s in traj.steps),
        "omissions": manifest,
    }


def render_rows(evidence):
    lines = []
    for r in evidence["rows"]:
        # Older evidence fixtures omit metadata, so safe defaults remain valid.
        thread = r.get("thread") or "main"
        call = (" call=" + str(r["call_id"])) if r.get("call_id") else ""
        tool = (" " + r["tool"]) if r.get("tool") else ""
        aggregation = r.get("output_aggregation")
        timing = ""
        if aggregation:
            completion = aggregation.get("completion_step")
            timing = (
                f" [session aggregate; available at step {completion}; sources={aggregation['source_steps']}]"
                if completion is not None
                else " [session still running; final outcome unknown]"
            )
        lines.append(f"[{r['step']}] thread={thread}{call} {r['kind']}{tool}{timing}: {r.get('text', '')}")
    if evidence.get("omissions"):
        m = evidence["omissions"]
        lines.append(
            "VIEW COVERAGE: "
            + json.dumps(
                {
                    k: m[k]
                    for k in (
                        "eligible_rows",
                        "retained_rows",
                        "omitted_rows",
                        "omitted_by_kind",
                        "truncated_step_ids",
                        "hidden_or_unrenderable_steps",
                        "task_truncated",
                    )
                },
                sort_keys=True,
            )
        )
        lines.append("Omitted or truncated evidence is unknown. Do not infer absence from this view.")
    return lines


def retrieve_steps(traj, step_ids, max_chars=8000):
    """Offline bounded canonical evidence retrieval; reports every truncation."""
    if type(max_chars) is not int or max_chars < 8:
        raise ValueError("retrieval budget must be >=8")
    wanted, remaining, rows = set(step_ids), max_chars, []
    for position, step in enumerate(traj.steps):
        if step.step_id not in wanted or remaining < 8:
            continue
        row = _row(position, step, remaining)
        if row:
            rows.append(row)
            remaining -= len(row["text"])
    return {
        "rows": rows,
        "budget": max_chars,
        "used_chars": max_chars - remaining,
        "unreturned_step_ids": sorted(wanted - {r["step"] for r in rows}),
    }


def retrieve_source(traj, step_id, max_chars=8000, max_source_bytes=64 * 1024 * 1024):
    """Inspect one original native record with an integrity check and output cap.

    Explicit offline retrieval only; it does not automatically enter a judge
    prompt or supply official test outcomes/reference patches.
    """
    if type(max_chars) is not int or max_chars < 8:
        raise ValueError("retrieval budget must be >=8")
    step = next((s for s in traj.steps if s.step_id == step_id), None)
    if step is None or not traj.source.get("path") or not traj.source.get("sha256"):
        raise ValueError("missing step or source provenance")
    path = Path(traj.source["path"])
    with path.open("rb") as stream:
        body = stream.read(max_source_bytes + 1)
    if len(body) > max_source_bytes:
        raise ValueError("source exceeds read budget")
    if hashlib.sha256(body).hexdigest() != traj.source["sha256"]:
        raise ValueError("source changed since normalization")
    ref = step.source_ref
    if "record" in ref:
        record = [line for line in body.decode().splitlines() if line.strip()][ref["record"]]
        value = json.loads(record)
    elif "pointer" in ref:
        value = json.loads(body)
        for part in ref["pointer"].split("/")[1:]:
            key = part.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
    else:
        raise ValueError("source reference is unavailable")
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {
        "step": step_id,
        "source_ref": ref,
        "text": excerpt(text, max_chars),
        "truncated": len(text) > max_chars,
        "source_verified": True,
    }
