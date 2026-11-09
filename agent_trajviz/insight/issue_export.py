"""Agent-ready JSON export of Overview Issues with step evidence.

The dashboard Issues cards are triage UI; this module builds a file an agent
can consume to address each issue — ranked cards plus clipped step/tool
context so the agent can locate failures without re-reading the full trace.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .assistant import _first_user_task
from .issues import OverviewIssue, prepare_overview_issues
from .presenters.issues import issue_suggested_action as issue_suggested_action
from .issue_judge import _step_window_indices
from .session import LoadedSession
from .tool_failure import tool_call_failed

SCHEMA_ID = "trajviz.issues_agent_export.v1"

_STEP_RADIUS = 1
_EVIDENCE_STEP_CAP = 20
_TOOLS_PER_STEP = 8
_TEXT_CLIP = 2000
_INPUT_CLIP = 1500
_OUTPUT_CLIP = 4000
_ERROR_CLIP = 2000

_PURPOSE = (
    "Feed this file to a coding agent that should address detected trajectory "
    "issues. Work issues in order. Prefer judgment.fix when present; otherwise "
    "follow suggested_action. Use evidence.steps as grounding — do not invent "
    "step numbers. Prefer durable fixes (agent instructions, skills, environment) "
    "over one-shot user-message edits unless judgment.where says otherwise."
)

_INSTRUCTIONS = (
    "Address each issue in ranked order (severity first).",
    "When judgment is present, apply judgment.fix at judgment.where.",
    "Otherwise apply suggested_action using evidence.steps for context.",
    "Cite only step indices that appear in related_steps or evidence.steps.",
    "After changes, re-run the failing commands or checks shown in the evidence when possible.",
)

def _clip(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _stringify_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tool_evidence(tool_call: dict) -> dict[str, Any]:
    name = str(tool_call.get("tool_name") or "")
    entry: dict[str, Any] = {"tool": name}
    if tool_call.get("status"):
        entry["status"] = str(tool_call.get("status"))
    if tool_call_failed(tool_call):
        entry["failed"] = True

    err = tool_call.get("error") or tool_call.get("error_type")
    if err:
        entry["error"] = _clip(str(err), _ERROR_CLIP)

    inp = tool_call.get("input")
    if isinstance(inp, dict):
        for key in (
            "command",
            "file_path",
            "filePath",
            "path",
            "pattern",
            "query",
            "glob_pattern",
            "glob",
            "content",
            "old_string",
            "new_string",
        ):
            if inp.get(key) is not None and str(inp.get(key)).strip():
                entry[key] = _clip(str(inp[key]), _INPUT_CLIP)
        # If nothing named matched, keep a clipped JSON dump.
        if not any(k in entry for k in (
            "command", "file_path", "filePath", "path", "pattern", "query",
            "glob_pattern", "glob", "content", "old_string", "new_string",
        )) and inp:
            entry["input"] = _clip(json.dumps(inp, ensure_ascii=False), _INPUT_CLIP)
    elif isinstance(inp, str) and inp.strip():
        entry["input"] = _clip(inp, _INPUT_CLIP)

    out = _stringify_output(tool_call.get("output"))
    if out.strip():
        entry["output"] = _clip(out, _OUTPUT_CLIP)
    return entry


def _ordered_evidence_indices(issue_steps: tuple[int, ...], step_map: dict[int, dict]) -> list[int]:
    window = _step_window_indices(issue_steps, radius=_STEP_RADIUS)
    ordered: list[int] = []
    for idx in list(issue_steps) + window:
        if idx not in ordered and idx in step_map:
            ordered.append(idx)
        if len(ordered) >= _EVIDENCE_STEP_CAP:
            break
    return ordered


def _step_evidence(step: dict, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index,
        "role": step.get("role"),
    }
    agent = step.get("agent") or step.get("agent_id")
    if agent:
        row["agent"] = str(agent)
    preview = step.get("text_preview")
    if isinstance(preview, str) and preview.strip():
        row["text_preview"] = _clip(preview, _TEXT_CLIP)
    tools = []
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        tools.append(_tool_evidence(tc))
        if len(tools) >= _TOOLS_PER_STEP:
            break
    if tools:
        row["tools"] = tools
    duration = step.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        row["duration_s"] = round(float(duration), 3)
    return row


def _pack_issue(issue: OverviewIssue, number: int, step_map: dict[int, dict]) -> dict[str, Any]:
    evidence_idxs = _ordered_evidence_indices(issue.steps, step_map)
    judgment = None
    if issue.judgment is not None:
        j = issue.judgment
        judgment = {
            "where": j.where,
            "fix": j.fix,
            "also": j.also or "",
            "confidence": j.confidence,
        }
    return {
        "id": number,
        "kind": issue.kind,
        "title": issue.title,
        "detail": issue.detail,
        "why": issue.why,
        "source_id": issue.source_id,
        "related_steps": list(issue.steps),
        "suggested_action": issue_suggested_action(issue),
        "judgment": judgment,
        "evidence": {
            "steps": [_step_evidence(step_map[i], i) for i in evidence_idxs],
        },
    }


def _session_token(session: LoadedSession) -> str:
    """Stable-ish id fragment for ``issues_<id>.json`` download names."""
    raw = getattr(session, "raw", None)
    raw = raw if isinstance(raw, dict) else {}

    candidates: list[Any] = []
    md = raw.get("metadata")
    if isinstance(md, dict):
        candidates.append(md.get("session_id"))
    info = raw.get("info")
    if isinstance(info, dict):
        candidates.append(info.get("id"))
    for step in getattr(session, "steps", None) or []:
        if isinstance(step, dict) and step.get("session_id"):
            candidates.append(step.get("session_id"))
            break
    path = getattr(session, "path", "") or ""
    if path:
        candidates.append(os.path.splitext(os.path.basename(path))[0])
    sha = raw.get("_source_sha256")
    run = getattr(session, "run", None)
    if not sha and run is not None:
        sha = getattr(run, "sha256", None)
    if sha:
        candidates.append(str(sha)[:12])

    for value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        safe = re.sub(r"[^\w.-]+", "-", text).strip("-.")
        if safe:
            return safe[:80]
    return "unknown"


def issues_export_filename(session: LoadedSession) -> str:
    """Download basename: ``issues_<id>.json``."""
    return f"issues_{_session_token(session)}.json"


def build_issues_agent_export(
    session: LoadedSession,
    issues: list[OverviewIssue] | None = None,
) -> dict[str, Any]:
    """Build the agent-ready issues JSON object (not written to disk)."""
    ranked = list(issues) if issues is not None else prepare_overview_issues(session)
    step_map = {
        int(s.get("index", i)): s
        for i, s in enumerate(session.steps or [])
    }
    user_task = _first_user_task(session.steps or [])
    session_meta: dict[str, Any] = {
        "path": getattr(session, "path", "") or "",
        "format": getattr(session, "format", "") or "",
        "steps_total": int(getattr(session, "steps_total", None) or len(session.steps or [])),
    }
    token = _session_token(session)
    if token != "unknown":
        session_meta["session_id"] = token
    sha = None
    raw = getattr(session, "raw", None)
    if isinstance(raw, dict):
        sha = raw.get("_source_sha256")
    run = getattr(session, "run", None)
    if sha is None and run is not None:
        sha = getattr(run, "sha256", None)
    if sha:
        session_meta["sha256"] = str(sha)
    if user_task:
        session_meta["user_task"] = user_task
    wall = getattr(session, "wall_clock", None)
    if wall:
        session_meta["wall_clock"] = str(wall)

    return {
        "schema": SCHEMA_ID,
        "purpose": _PURPOSE,
        "instructions": list(_INSTRUCTIONS),
        "session": session_meta,
        "issue_count": len(ranked),
        "issues": [_pack_issue(issue, i + 1, step_map) for i, issue in enumerate(ranked)],
    }


def write_issues_agent_export(
    session: LoadedSession,
    dest: str,
    issues: list[OverviewIssue] | None = None,
    *,
    filename: str | None = None,
) -> str:
    """Write the agent export JSON under *dest* (file or directory). Return the file path."""
    payload = build_issues_agent_export(session, issues=issues)
    name = filename or issues_export_filename(session)
    if os.path.isdir(dest):
        path = os.path.join(dest, name)
    else:
        path = dest
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path
