"""Prompt evidence for one recorded request, without treating a transcript as a request archive.

Native request/state readers materialize only the selected request. Other
formats expose explicitly estimated transcript candidates: no file contents,
tool definitions, pruning policy or hidden instructions are inferred.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from .telemetry_quality import field_observed, tool_field_observed
from .token_explain import _est_tokens

_ESTIMATE_LIMITATIONS = [
    "These are earlier transcript candidates, not a captured model request or proof of context membership.",
    "System instructions, tool definitions, pruning, provider conversion and hidden context may be missing.",
    "An assistant step can merge several model calls; this boundary is an estimate.",
]
_TOKEN_BASIS = (
    "Character estimate of recorded text/JSON: basic CJK characters / 1.4 plus other characters / 2.5. "
    "This is not billed usage; provider framing and opaque or non-text inputs are not measured."
)
_OMITTED_RESULT = "Tool results without a proven completion before this boundary are omitted."
_TRANSCRIPT_EVENT_LABELS = {
    "transcript_user_message": "User message recorded in transcript",
    "transcript_system_message": "System or developer message recorded in transcript",
    "transcript_tool_result": "Completed tool result recorded in transcript",
    "transcript_compaction": "Compaction checkpoint recorded in transcript",
}
_EVENT_SOURCE_LIMIT = 8


def _number(value: Any) -> int | float | None:
    try:
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    except OverflowError:
        return None


def _identity(step: dict, position: int) -> tuple[str, str, str, str, str, str]:
    """Keep parallel children separate even when their native session ids match."""
    child = bool(step.get("is_sub_agent") or step.get("parent_agent_id"))
    agent = str(step.get("agent") or "") if child else ""
    if agent == "compaction":
        agent = ""
    sid, parent_sid = str(step.get("session_id") or ""), str(step.get("parent_session_id") or "")
    owner = "child" if child else "main"
    if child and not agent and not (sid and parent_sid and sid != parent_sid):
        owner = f"unidentified-child-step-{position}"
    return (sid, str(step.get("source_path") or ""), str(step.get("parent_agent_id") or ""), agent, owner,
            str(step.get("branch_id") or ""))


def _is_compaction(step: dict) -> bool:
    return bool(step.get("is_compaction_checkpoint") or step.get("role") == "compaction"
                or step.get("message_type") == "compaction")


def _matching_child_checkpoint(step: dict, target: dict) -> bool:
    """A distinct child session can identify a checkpoint whose agent label is a mode."""
    sid, parent = step.get("session_id"), step.get("parent_session_id")
    return bool(_is_compaction(step) and isinstance(sid, str) and sid and isinstance(parent, str) and parent
                and sid != parent and sid == target.get("session_id") and parent == target.get("parent_session_id")
                and (step.get("source_path") or "") == (target.get("source_path") or "")
                and (step.get("branch_id") or "") == (target.get("branch_id") or ""))


def _source(step: dict, position: int) -> dict:
    return {"step_index": step.get("index", position), "raw_index": step.get("raw_index"),
            "source_path": str(step.get("source_path") or ""),
            "message_id": str(step.get("message_id") or step.get("id") or "")}


def _step_text(step: dict) -> list[dict]:
    if not field_observed(step, "text"):
        return []
    parts = [{"type": "text", "text": part["text"]} for part in step.get("parts", [])
             if isinstance(part, dict) and part.get("type") == "text"
             and isinstance(part.get("text"), str) and part["text"]]
    if not parts and isinstance(step.get("text_preview"), str) and step["text_preview"]:
        parts = [{"type": "text", "text": step["text_preview"]}]
    return parts


def _tool_observed(call: dict, field: str) -> bool:
    metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
    telemetry = metadata.get("_otel")
    if isinstance(telemetry, dict) and not isinstance(telemetry.get("fields"), dict):
        return False
    return tool_field_observed({"metadata": metadata}, field)


def _step_messages(step: dict, before_ms: int | float | None = None, omissions: list[str] | None = None) -> list[dict]:
    role = str(step.get("role") or "")
    if role not in {"user", "assistant", "system", "developer", "tool"}:
        return []
    completed = _number(step.get("time_completed_ms"))
    content = ([] if role == "assistant" and completed is not None and before_ms is not None and completed > before_ms
               else _step_text(step))
    results = []
    for call in step.get("tool_calls", []):
        if not isinstance(call, dict):
            continue
        started = _number(call.get("time_start"))
        if started is not None and before_ms is not None and started > before_ms:
            continue
        if _tool_observed(call, "tool_inputs"):
            content.append({"type": "tool_use", "id": str(call.get("tool_id") or ""),
                            "name": str(call.get("tool_name") or ""), "input": call.get("input", {})})
        end = _number(call.get("time_end"))
        if _tool_observed(call, "tool_outputs") and call.get("output") not in (None, ""):
            if end is not None and before_ms is not None and end <= before_ms:
                results.append({"type": "tool_result", "tool_use_id": str(call.get("tool_id") or ""),
                                "content": call["output"]})
            elif omissions is not None and _OMITTED_RESULT not in omissions:
                omissions.append(_OMITTED_RESULT)
    messages = [{"role": role, "content": content}] if content else []
    if results:
        messages.append({"role": "user", "content": results})
    return messages


def _summary(step: dict) -> str:
    for part in step.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "compaction" and isinstance(part.get("summary"), str):
            return part["summary"]
    return "\n".join(part["text"] for part in _step_text(step))


def _child_checkpoint_key(step: dict) -> tuple | None:
    sid, parent = step.get("session_id"), step.get("parent_session_id")
    if not isinstance(sid, str) or not sid or not isinstance(parent, str) or not parent or sid == parent:
        return None
    return ("child_checkpoint", sid, parent, str(step.get("source_path") or ""), str(step.get("branch_id") or ""))


def _queue_event(pending: dict, key: tuple, event: dict, ready_at: int | float | None,
                 serial: int, *, requires_clock: bool = False) -> None:
    queues = pending.setdefault(key, {"untimed": [], "timed": [], "results": []})
    if ready_at is None:
        queues["untimed"].append((serial, event))
    else:
        heapq.heappush(queues["results" if requires_clock else "timed"], (ready_at, serial, event))


def _take_events(pending: dict, key: tuple | None, boundary_ms: int | float | None) -> list[tuple]:
    queues = pending.get(key)
    if queues is None:
        return []
    ready = queues["untimed"]
    queues["untimed"] = []
    for field in ("timed", "results"):
        if field == "results" and boundary_ms is None:
            continue
        heap = queues[field]
        while heap and (boundary_ms is None or heap[0][0] <= boundary_ms):
            _when, serial, event = heapq.heappop(heap)
            ready.append((serial, event))
    if not any(queues.values()):
        pending.pop(key, None)
    return ready


def _event_source(step: dict, position: int, field: str, **extra: Any) -> dict:
    return {"record": "normalized_step", **_source(step, position), "field": field, **extra}


def _enqueue_transcript_events(step: dict, position: int, pending: dict, serial: int) -> int:
    key = ("context", *_identity(step, position))
    role = step.get("role")
    kind = ("transcript_compaction" if _is_compaction(step) else
            "transcript_user_message" if role == "user" and _step_text(step) else
            "transcript_system_message" if role in {"system", "developer"} and _step_text(step) else None)
    if kind:
        event = {"kind": kind, "label": _TRANSCRIPT_EVENT_LABELS[kind], "evidence": "observed_transcript", "count": 1,
                 "source_refs": [_event_source(step, position, "parts")], "source_refs_truncated": False}
        # A distinct native child session identifies its mode-labelled checkpoint;
        # shared-session or unidentified children never receive that association.
        event_key = (_child_checkpoint_key(step) or key) if _is_compaction(step) else key
        _queue_event(pending, event_key, event, _number(step.get("time_created_ms")), serial)
        serial += 1
    for index, call in enumerate(step.get("tool_calls", [])):
        if not isinstance(call, dict) or not _tool_observed(call, "tool_outputs"):
            continue
        end, start = _number(call.get("time_end")), _number(call.get("time_start"))
        if end is None or end < 0 or start is not None and start > end:
            continue
        metadata = call.get("metadata") if isinstance(call.get("metadata"), dict) else {}
        telemetry = metadata.get("_otel") if isinstance(metadata.get("_otel"), dict) else {}
        fields = telemetry.get("fields") if isinstance(telemetry.get("fields"), dict) else {}
        explicit_empty = (metadata.get("_output_observed") is True
                          or fields.get("tool_outputs") == "observed")
        if call.get("output") in (None, "") and not explicit_empty:
            continue
        name = str(call.get("tool_name") or "")
        event = {"kind": "transcript_tool_result", "label": _TRANSCRIPT_EVENT_LABELS["transcript_tool_result"],
                 "evidence": "observed_transcript", "count": 1, "tool_name": name,
                 "source_refs": [_event_source(step, position, "tool_calls", index=index, tool_name=name,
                                               tool_id=str(call.get("tool_id") or ""),
                                               part_id=str(call.get("part_id") or ""))],
                 "source_refs_truncated": False}
        source_created = _number(step.get("time_created_ms"))
        ready_at = max(end, source_created) if source_created is not None else end
        _queue_event(pending, key, event, ready_at, serial, requires_clock=True)
        serial += 1
    return serial


def _compact_events(events: list[tuple]) -> list[dict]:
    """At most four event groups per boundary; counts retain omitted references."""
    grouped: dict[str, dict] = {}
    names: dict[str, set[str]] = {}
    for _serial, event in sorted(events, key=lambda item: item[0]):
        kind = event["kind"]
        group = grouped.setdefault(kind, {"kind": kind, "label": event["label"], "evidence": "observed_transcript",
                                          "count": 0, "source_refs": [], "source_refs_truncated": False})
        group["count"] += event["count"]
        if event.get("tool_name"):
            names.setdefault(kind, set()).add(event["tool_name"])
        for source in event["source_refs"]:
            if len(group["source_refs"]) < _EVENT_SOURCE_LIMIT:
                group["source_refs"].append(source)
            else:
                group["source_refs_truncated"] = True
    for kind, values in names.items():
        if len(values) == 1:
            grouped[kind]["tool_name"] = next(iter(values))
    return list(grouped.values())


def _transcript_catalog(steps: list[dict]) -> list[dict]:
    rows = []
    previous: dict[tuple[str, str, str, str, str, str], str] = {}
    pending: dict[tuple, dict] = {}
    serial = 0
    for position, step in enumerate(steps):
        if step.get("role") != "assistant" or _is_compaction(step):
            serial = _enqueue_transcript_events(step, position, pending, serial)
            continue
        identity_key = _identity(step, position)
        sid, path, parent, agent, owner, branch = identity_key
        identity = ":".join(quote(value, safe="") for value in (sid, path, parent, agent, owner, branch))
        index = step.get("index", position)
        identifier = f"transcript:{identity}:{position}"
        actor_kind = "unknown_child" if owner.startswith("unidentified-child") else "subagent" if owner == "child" else "main"
        boundary_ms = _number(step.get("time_created_ms"))
        observed = _take_events(pending, ("context", *identity_key), boundary_ms)
        observed += _take_events(pending, _child_checkpoint_key(step), boundary_ms)
        rows.append({"id": identifier, "label": f"Step #{index} · transcript candidates",
                     "evidence": "estimated", "scope": "trajectory_estimate", "model": str(step.get("model_id") or ""),
                     "session_id": sid, "agent_id": agent, "step_index": index,
                     "timestamp_ms": boundary_ms, "missing_references": [],
                     "context_id": f"candidate:{identity}", "actor_kind": actor_kind,
                     "actor_name": (agent or sid) if actor_kind == "subagent" else "",
                     "branch_id": branch or None,
                     "prompt_events": _compact_events(observed),
                     "previous_candidate_id": previous.get(identity_key),
                     "_position": position})
        previous[identity_key] = identifier
        # This response's attached results can only belong to a later boundary.
        serial = _enqueue_transcript_events(step, position, pending, serial)
    return rows


def _transcript_detail(steps: list[dict], row: dict) -> dict:
    position = row["_position"]
    target = steps[position]
    identity = _identity(target, position)
    system, messages, system_sources, message_sources = [], [], [], []
    compacted = False
    omissions: list[str] = []
    before_ms = _number(target.get("time_created_ms"))
    for prior_position, step in enumerate(steps[:position]):
        if _identity(step, prior_position) != identity and not _matching_child_checkpoint(step, target):
            continue
        created = _number(step.get("time_created_ms"))
        if created is not None and before_ms is not None and created > before_ms:
            continue
        if _is_compaction(step):
            compacted = True
            messages = []
            message_sources = []
            summary = _summary(step)
            if summary:
                messages.append({"role": "user", "content": [{"type": "text", "text": summary}],
                                 "_prompt_source": "compaction_summary"})
                message_sources.append(_source(step, prior_position))
            continue
        additions = _step_messages(step, before_ms, omissions)
        if additions:
            instructions = step.get("role") in {"system", "developer"}
            (system if instructions else messages).extend(additions)
            (system_sources if instructions else message_sources).append(_source(step, prior_position))
    limitations = [*_ESTIMATE_LIMITATIONS, *omissions]
    if identity[4].startswith("unidentified-child"):
        limitations.append("This child has no distinct recorded identity; earlier child context cannot be safely associated.")
    if compacted:
        limitations.append("An observed compaction bounds the transcript candidates; retained context is not fully known.")
    if not system and not messages:
        limitations.append("No earlier readable input was recorded for this boundary; an empty list does not mean an empty prompt.")
    return {**{key: value for key, value in row.items() if key != "_position"}, "system": system, "tools": [],
            "messages": messages, "parameters": {}, "sources": system_sources + message_sources,
            "limitations": limitations}


def _inspect_transcript(raw: dict, steps: list[dict], request_id: str | None = None) -> dict:
    del raw
    rows = _transcript_catalog(steps)
    wanted = request_id if request_id is not None else rows[-1]["id"] if rows else None
    row = next((row for row in rows if row["id"] == wanted), None)
    if request_id is not None and row is None:
        raise ValueError(f"Unknown prompt request id: {request_id}")
    detail = _transcript_detail(steps, row) if row is not None else None
    catalog = [{key: value for key, value in row.items() if key != "_position"} for row in rows]
    return {"available": bool(rows), "requests": catalog, "selected_request": detail,
            "limitations": list(_ESTIMATE_LIMITATIONS) if rows else ["No model-request or assistant-step boundary is recorded."]}


def _json_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _measurable(value: Any) -> tuple[Any, bool]:
    """Keep non-text content intact in the view, but do not count base64 as tokens."""
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url", "input_image", "audio", "input_audio", "video", "document"}:
            return None, True
        result, partial = {}, False
        for key, item in value.items():
            if key in {"encrypted_content", "signature"}:
                partial = True
                continue
            measured, omitted = _measurable(item)
            result[key] = measured
            partial |= omitted
        return result, partial
    if isinstance(value, list):
        result, partial = [], False
        for item in value:
            measured, omitted = _measurable(item)
            if measured is not None:
                result.append(measured)
            partial |= omitted
        return result, partial
    return value, False


def _components(detail: dict) -> tuple[list[dict], dict]:
    components, seen, totals = [], Counter(), {"system": 0, "tools": 0, "messages": 0}
    partial = bool(detail.get("missing_references") or detail.get("unsupported_fields")
                   or detail.get("scope") != "recorded_request")
    for field, kind in (("system", "system"), ("tools", "tool"), ("messages", "message")):
        values = detail.get(field, [])
        values = values if isinstance(values, list) else [values] if values else []
        for position, value in enumerate(values):
            fingerprint = hashlib.sha256((kind + ":" + _json_text(value)).encode()).hexdigest()[:20]
            seen[fingerprint] += 1
            component_id = f"{kind}:{fingerprint}:{seen[fingerprint]}"
            measurable, omitted = _measurable(value)
            text = _json_text(measurable) if measurable not in (None, [], "") else ""
            estimate = _est_tokens(text)
            partial |= omitted
            role = str(value.get("role") or "") if isinstance(value, dict) else ""
            role = role or ("system" if kind == "system" else "")
            label = (str(value.get("name") or "") if kind == "tool" and isinstance(value, dict) else "")
            if kind == "system" and isinstance(value, dict) and value.get("section"):
                label = str(value["section"])
            label = label or f"{role or kind} {position + 1}"
            identity = None
            if detail.get("scope") == "native_prompt_state" and str(detail.get("id", "")).startswith("pi:"):
                if kind == "system" and isinstance(value, dict) and isinstance(value.get("section"), str):
                    identity = f"pi:system:{value['section']}"
                elif (kind == "tool" and isinstance(value, dict) and isinstance(value.get("name"), str)
                      and value["name"]):
                    identity = f"pi:tool:{value['name']}"
                elif kind == "message":
                    refs = detail.get("message_refs", [])
                    ref = refs[position] if isinstance(refs, list) and position < len(refs) else None
                    if isinstance(ref, dict) and isinstance(ref.get("entry_id"), str) and ref["entry_id"]:
                        identity = f"pi:message:{ref.get('origin', '')}:{ref['entry_id']}"
            components.append({"id": component_id, "identity": identity, "kind": kind, "label": label, "role": role,
                               "content": value, "estimated_tokens": estimate,
                               "source_refs": [{"request_id": detail["id"], "field": field, "index": position}]})
            totals[field] += estimate
    return components, {"evidence": "estimated", "total": sum(totals.values()), **totals,
                        "partial": partial, "basis": _TOKEN_BASIS}


def _previous_row(rows: list[dict], selected: dict) -> dict | None:
    selected_id = selected["id"]
    position = next((index for index, row in enumerate(rows) if row.get("id") == selected_id), None)
    if position is None:
        return None
    identity = (selected.get("session_id"), selected.get("agent_id"))
    previous_id = selected.get("previous_request_id")
    branch = selected.get("branch_id")
    for row in reversed(rows[:position]):
        if (row.get("session_id"), row.get("agent_id")) != identity:
            continue
        if previous_id and row.get("id") == previous_id or not previous_id and branch and row.get("branch_id") == branch:
            return row
    return None


def _brief_component(part: dict) -> dict:
    preview = json.dumps(part["content"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**{key: part[key] for key in ("id", "kind", "label", "role", "estimated_tokens")},
            "preview": preview[:500], "preview_truncated": len(preview) > 500, "preview_format": "json"}


def _changes(selected: dict, previous: dict | None, *, estimated_candidates: bool = False) -> dict:
    candidate_basis = ({"evidence": "estimated", "lineage_verified": False,
                        "basis": "adjacent_transcript_candidates",
                        "previous_candidate_id": previous["id"] if previous else None}
                       if estimated_candidates else {})
    if previous is None:
        return {"available": False, "previous_request_id": None, **candidate_basis,
                "reason": ("No earlier candidate boundary with the same recorded agent identity is available."
                           if estimated_candidates else "No previous request on a recorded matching lineage is available.")}
    if selected.get("missing_references") or previous.get("missing_references"):
        return {"available": False, "previous_request_id": previous["id"], **candidate_basis,
                "reason": "Unresolved references prevent a reliable prompt delta."}
    before, before_estimate = _components(previous)
    after, after_estimate = _components(selected)
    before_ids, after_ids = [part["id"] for part in before], [part["id"] for part in after]
    before_set, after_set = set(before_ids), set(after_ids)
    retained = before_set & after_set
    before_counts = Counter(part.get("identity") for part in before if part.get("identity"))
    after_counts = Counter(part.get("identity") for part in after if part.get("identity"))
    before_named = {part["identity"]: part for part in before if part.get("identity") and before_counts[part["identity"]] == 1}
    rewritten = []
    for part in ([] if estimated_candidates else after):
        identity = part.get("identity")
        old = before_named.get(identity) if identity and after_counts[identity] == 1 else None
        if old is not None and old["id"] != part["id"]:
            rewritten.append({"identity": identity, "kind": part["kind"], "label": part["label"],
                              "before_id": old["id"], "after_id": part["id"],
                              "before": old["content"], "after": part["content"],
                              "estimated_token_delta": part["estimated_tokens"] - old["estimated_tokens"]})
    return {"available": True, "previous_request_id": previous["id"], **candidate_basis,
            "added": [value for value in after_ids if value not in before_set],
            "removed": [value for value in before_ids if value not in after_set],
            "added_components": [_brief_component(part) for part in after if part["id"] not in before_set],
            "removed_components": [_brief_component(part) for part in before if part["id"] not in after_set],
            "retained": len(retained),
            "rewritten_components": rewritten,
            "groups_changed": {field: _json_text(previous.get(field, [])) != _json_text(selected.get(field, []))
                               for field in ("system", "tools", "messages")},
            "order_changed": [value for value in before_ids if value in retained]
            != [value for value in after_ids if value in retained],
            "estimated_token_delta": after_estimate["total"] - before_estimate["total"],
            "reason": (("Differences between our estimated transcript-candidate arrays only. Adjacency does not verify "
                        "request lineage or actual context membership; no prompt rewrite is inferred.")
                       if estimated_candidates else
                       "Changes in recorded request/state components, not evidence of model attention or causal influence. "
                       "Added/removed IDs identify changed content; proven rewrites also appear in both lists.")}


def _token_count(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number == int(number) else None


def _observed_prompt(step: dict | None) -> tuple[int | None, str | None]:
    """Use declared occupancy or an unambiguous, fully reported usage equation."""
    if step is None:
        return None, None
    tokens = step.get("tokens")
    if not isinstance(tokens, dict):
        return None, None
    context = _token_count(tokens.get("context_window"))
    if context is not None and field_observed(step, "context"):
        return context, "recorded_context_occupancy"
    peak = _token_count(tokens.get("prompt_peak"))
    if peak is not None and field_observed(step, "context"):
        return peak, "recorded_largest_request_in_step"
    required = ("input", "output", "cache_read", "cache_write", "total")
    missing_labels = set(step.get("_metrics_unavailable_fields") or [])
    if missing_labels & {"Input Tokens", "Output Tokens", "Cache Read", "Cache Write", "Total Tokens"}:
        return None, None
    if not all(field_observed(step, f"tokens.{key}") for key in required):
        return None, None
    counts = {key: _token_count(tokens.get(key)) for key in required}
    if any(value is None for value in counts.values()):
        return None, None
    inp, out, cached, written, total = (counts[key] for key in required)
    reasoning = _token_count(tokens.get("reasoning", 0))
    if reasoning is None:
        return None, None
    readings = set()
    for prompt in (inp, inp + cached + written):
        if total in {prompt + out, prompt + out + reasoning}:
            readings.add(prompt)
    if len(readings) != 1:
        return None, None
    return readings.pop(), "validated_recorded_input_and_cache_counts"


def _evolution(rows: list[dict], steps: list[dict]) -> list[dict]:
    mapped: dict[int, dict | None] = {}
    for position, step in enumerate(steps):
        index = step.get("index", position)
        if isinstance(index, int) and not isinstance(index, bool):
            mapped[index] = None if index in mapped else step
    evolution = []
    for order, row in enumerate(rows, 1):
        index = row.get("step_index")
        step = mapped.get(index) if isinstance(index, int) and not isinstance(index, bool) else None
        if step is not None and row.get("session_id") and step.get("session_id") != row["session_id"]:
            step = None
        if step is not None and row.get("agent_id") and step.get("agent") != row["agent_id"]:
            step = None
        occupancy, basis = _observed_prompt(step)
        context_id = row.get("context_id")
        if not isinstance(context_id, str) or not context_id:
            context_id = "native:" + ":".join(quote(str(value or ""), safe="") for value in (
                str(row["id"]).split(":", 1)[0], row.get("session_id"), row.get("agent_id"),
                row.get("branch_id"), row.get("source_path")))
        actor_kind = row.get("actor_kind") or ("subagent" if row.get("agent_id") else "main")
        evolution.append({"id": row["id"], "label": row["label"], "order": order,
                          **{key: row.get(key) for key in ("step_index", "agent_id", "session_id", "model", "timestamp_ms",
                                                         "evidence", "scope", "branch_id", "previous_request_id",
                                                         "previous_candidate_id")},
                          "context_id": context_id, "actor_kind": actor_kind,
                          "actor_name": str(row.get("actor_name") or row.get("agent_id") or ""),
                          "observed_prompt_tokens": occupancy, "observed_prompt_basis": basis,
                          "prompt_events": row.get("prompt_events", []),
                          **({"transition": row["transition"]} if "transition" in row else {})})
    return evolution


def inspect_prompt_construction(raw: dict, steps: list[dict], request_id: str | None = None,
                                *, format_name: str = "") -> dict:
    """Inspect native request evidence or explicit transcript candidates.

    A missing id selects the last recorded boundary; an unknown id raises
    ``ValueError``. Native evidence is selected from preserved raw fields, so
    normalized JSON exports retain the same behavior after reloading.
    """
    if request_id is not None and not isinstance(request_id, str):
        raise TypeError("request_id must be a string or None")
    inspect: Callable[..., dict] = _inspect_transcript
    if isinstance(raw.get("_claude_prompt_records"), list) and raw["_claude_prompt_records"]:
        from .prompt_claude import inspect_claude

        inspect = inspect_claude
    elif isinstance(raw.get("_pi_entries"), list) and raw["_pi_entries"]:
        from .prompt_pi import inspect_pi

        inspect = inspect_pi
    data = inspect(raw, steps, request_id)
    data = dict(data)
    selected = data.get("selected_request")
    if request_id is not None and (not isinstance(selected, dict) or selected.get("id") != request_id):
        raise ValueError(f"Unknown prompt request id: {request_id}")
    metadata = raw.get("metadata")
    data["format"] = format_name or (str(metadata.get("source") or "") if isinstance(metadata, dict) else "")
    data["evolution"] = _evolution(data.get("requests", []), steps)
    if isinstance(selected, dict):
        selected = dict(selected)
        selected["components"], selected["token_estimate"] = _components(selected)
        candidates = selected.get("scope") == "trajectory_estimate"
        if candidates:
            previous_row = next((row for row in data.get("requests", [])
                                 if row.get("id") == selected.get("previous_candidate_id")
                                 and row.get("scope") == "trajectory_estimate"), None)
        else:
            previous_row = _previous_row(data.get("requests", []), selected)
        previous = inspect(raw, steps, previous_row["id"]).get("selected_request") if previous_row else None
        selected["changes"] = _changes(selected, previous, estimated_candidates=candidates)
        data["selected_request"] = selected
    return data
