"""Claude Code's saved OTLP spans and correlated logs → trajectory operations.

Profile version 1 follows Claude Code's documented beta span names and event
attributes. It never infers transcript messages, agent identities or model/tool
associations from timestamp proximity. The complete selected graph is retained
so unmapped operations and evidence survive a normalized export.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from ...errors import LoadError
from .claude_code import _cc_build_step
from .otlp import OtlpCapture

PROFILE_VERSION = 1
_PREFIX = "claude_code."
_OPERATIONS = {"interaction", "llm_request", "tool", "compaction"}
_TOKEN_KEYS = {
    "input": ("input_tokens",),
    "output": ("output_tokens",),
    "cache_read": ("cache_read_tokens", "cache_read_input_tokens"),
    "cache_write": ("cache_creation_tokens", "cache_creation_input_tokens"),
    "reasoning": ("reasoning_tokens",),
}
_FIELDS = tuple(f"tokens.{key}" for key in (*_TOKEN_KEYS, "total")) + (
    "duration", "text", "tool_inputs", "tool_outputs", "tool_outcomes", "context",
)


def _text(value):
    return value if isinstance(value, str) else ""


def _attrs(record):
    attrs = dict(record.get("resource", {}))
    attrs.update(record.get("attributes", {}))
    return attrs


def _id(record):
    return record.get("trace_id", ""), record.get("span_id", "")


def _name(record):
    attrs = record.get("attributes", {})
    return _text(record.get("name")) or _text(attrs.get("span.type"))


def _event_name(record):
    attrs = record.get("attributes", {})
    name = _text(attrs.get("event.name")) or _text(record.get("event_name"))
    if not name and isinstance(record.get("body"), str):
        body = record["body"]
        if body.startswith(_PREFIX):
            name = body
    return name.removeprefix(_PREFIX)


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value) if "." in value else int(value)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, (int, float)):
        try:
            if math.isfinite(value) and value >= 0:
                return value
        except OverflowError:
            pass
    return None


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.lower())
    return None


def _token_count(value):
    """Counts are nonnegative integers, unlike fractional millisecond timing."""
    number = _number(value)
    if number is None or number > 2**63 - 1:
        return None
    if isinstance(number, float) and not number.is_integer():
        return None
    return int(number)


def _milliseconds(nano):
    return nano // 1_000_000 if isinstance(nano, int) and nano >= 0 else None


def _interval(span):
    start = span.get("start_time_unix_nano")
    end = span.get("end_time_unix_nano")
    if isinstance(start, int) and isinstance(end, int) and start > 0 and end >= start:
        return start, end, (end - start) / 1_000_000
    return None, None, None


def _duration_ms(record):
    duration = _interval(record)[2]
    return duration if duration is not None else _number(_attrs(record).get("duration_ms"))


def _iso(nano):
    if not isinstance(nano, int) or nano <= 0:
        return ""
    try:
        return datetime.fromtimestamp(nano // 1_000_000_000, UTC).strftime("%Y-%m-%dT%H:%M:%S") + (
            f".{nano % 1_000_000_000:09d}Z")
    except (ValueError, OverflowError, OSError):
        return ""


def _metadata(span, kind):
    return {
        "kind": kind, "profile": "claude_code", "profile_version": PROFILE_VERSION,
        "trace_id": span.get("trace_id", ""), "span_id": span.get("span_id", ""),
        "parent_span_id": span.get("parent_span_id", ""),
        "start_time_unix_nano": span.get("start_time_unix_nano"),
        "end_time_unix_nano": span.get("end_time_unix_nano"),
        "fields": dict.fromkeys(_FIELDS, "unavailable"),
        "reasons": dict.fromkeys(_FIELDS, "not_recorded"),
        "source": deepcopy(span.get("source", {})),
    }


def _observed(meta, field):
    meta["fields"][field] = "observed"
    meta["reasons"].pop(field, None)


def _content(attrs, keys, meta, field):
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() in ("<REDACTED>", "[REDACTED]"):
            meta["reasons"][field] = "redacted"
            continue
        if not isinstance(value, (str, dict, list)):
            meta["reasons"][field] = "invalid"
            continue
        truncated = _boolean(attrs.get(f"{key}_truncated")) is True
        truncated = truncated or (isinstance(value, str) and "[TRUNCATED" in value)
        if truncated:
            meta["fields"][field] = "partial"
            meta["reasons"][field] = "truncated"
        else:
            _observed(meta, field)
        return value
    return None


def _json_object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _as_content(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _span_kind(span):
    return _name(span).removeprefix(_PREFIX) if _name(span).startswith(_PREFIX) else ""


def _session_index(capture):
    """Resolve session identity from explicit attributes and graph evidence."""
    spans = capture.spans
    by_id = {_id(span): span for span in spans}
    assigned = {}
    trace_sessions = defaultdict(set)
    for span in spans:
        sid = _text(_attrs(span).get("session.id"))
        if sid:
            assigned[_id(span)] = sid
            trace_sessions[span.get("trace_id", "")].add(sid)
    inherited = _inherit_graph_values(spans, by_id, assigned)
    for span in spans:
        key = _id(span)
        if inherited.get(key):
            assigned[key] = inherited[key]
        else:
            sids = trace_sessions[span.get("trace_id", "")]
            if len(sids) == 1:
                assigned[key] = next(iter(sids))
            elif not sids:
                assigned[key] = f"trace:{span.get('trace_id', '')}"
    return assigned, by_id, trace_sessions


def _inherit_graph_values(spans, by_id, explicit):
    """Nearest explicit ancestor values, memoized even for reversed batches."""
    values = dict(explicit)
    for span in spans:
        key, trail, seen = _id(span), [], set()
        while key in by_id and key not in values and key not in seen:
            seen.add(key)
            trail.append(key)
            current = by_id[key]
            key = current.get("trace_id", ""), current.get("parent_span_id", "")
        value = values.get(key, "")
        for key in reversed(trail):
            values[key] = value
    return values


def list_otlp_sessions(capture: OtlpCapture) -> list[dict[str, Any]]:
    """List Claude Code sessions without merging distinct sessions by timing."""
    assigned, _, _ = _session_index(capture)
    sessions = {}
    for span in capture.spans:
        if _span_kind(span) not in _OPERATIONS:
            continue
        sid = assigned.get(_id(span))
        if not sid:
            continue
        item = sessions.setdefault(sid, {
            "id": sid, "label": sid, "profile": "claude_code", "trace_ids": [],
            "span_count": 0, "interaction_count": 0, "incomplete": sid.startswith("trace:"),
            "started_at": "", "finished_at": "",
        })
        trace = span.get("trace_id", "")
        if trace not in item["trace_ids"]:
            item["trace_ids"].append(trace)
        item["span_count"] += 1
        item["interaction_count"] += _span_kind(span) == "interaction"
        start, end, _ = _interval(span)
        start_text, end_text = _iso(start), _iso(end)
        if start_text and (not item["started_at"] or start_text < item["started_at"]):
            item["started_at"] = start_text
        if end_text and (not item["finished_at"] or end_text > item["finished_at"]):
            item["finished_at"] = end_text
    for item in sessions.values():
        item["trace_ids"].sort()
    return sorted(sessions.values(), key=lambda item: (item["started_at"], item["id"]))


def _request_id(record):
    attrs = _attrs(record)
    return _text(attrs.get("request_id")) or _text(attrs.get("gen_ai.response.id"))


def _tool_id(record):
    attrs = _attrs(record)
    return _text(attrs.get("tool_use_id")) or _text(attrs.get("gen_ai.tool.call.id"))


def _correlation_index(spans, agents, kind, identify):
    counts = Counter()
    for span in spans:
        if _span_kind(span) != kind or not (identity := identify(span)):
            continue
        trace, agent = span.get("trace_id", ""), agents.get(_id(span), "")
        counts[("trace", trace, identity)] += 1
        counts[("lane", trace, agent, identity)] += 1
        counts[("all", identity)] += 1
        counts[("agent", agent, identity)] += 1
    return counts


def _correlated_logs(span, agent, identity, lookup, counts):
    """Attach evidence only when its explicit IDs select one operation.

    Producer request/call IDs may repeat across traces and agent lanes. An
    unqualified log must not fill every operation that happens to share an ID.
    A log's span ID is stronger evidence when it identifies this operation
    directly (normally Claude events carry the enclosing interaction's ID).
    """
    if not identity:
        return []
    trace = span.get("trace_id", "")
    candidates = list(lookup[(trace, identity)])
    if trace:
        candidates += lookup[("", identity)]
    related = []
    for log in candidates:
        log_agent = _text(_attrs(log).get("agent_id"))
        if log_agent and log_agent != agent:
            continue
        log_trace = log.get("trace_id", "")
        key = (("lane", log_trace, log_agent, identity) if log_agent else ("trace", log_trace, identity)) if log_trace else (
            ("agent", log_agent, identity) if log_agent else ("all", identity))
        if _id(log) == _id(span) or counts[key] == 1:
            related.append(log)
    return related


def _merge_model_attributes(span, related, meta):
    """Prefer valid span counts across semantic aliases, then fill from logs."""
    sources = [_attrs(span)] + [_attrs(log) for log in related
                               if _event_name(log) in ("api_request", "assistant_response", "api_error")]
    merged = dict(sources[0])
    for source in sources[1:]:
        for key, value in source.items():
            merged.setdefault(key, value)
    for keys in _TOKEN_KEYS.values():
        present = [(key, source[key]) for source in sources for key in keys if key in source]
        valid = [_token_count(value) for _, value in present]
        valid = [value for value in valid if value is not None]
        if not valid:
            continue
        for key in keys:
            merged.pop(key, None)
        merged[keys[0]] = valid[0]
        if any(value != valid[0] for value in valid[1:]):
            meta.setdefault("conflicts", []).append(keys[0])
    return merged


def _fill_usage(step, attrs):
    tokens, meta = step["tokens"], step["_otel"]
    for field, keys in _TOKEN_KEYS.items():
        for key in keys:
            if key not in attrs:
                continue
            value = _token_count(attrs[key])
            if value is not None:
                tokens[field] = value
                _observed(meta, f"tokens.{field}")
                break
            meta["reasons"][f"tokens.{field}"] = "invalid"
    components = ("input", "output", "cache_read", "cache_write")
    if all(meta["fields"][f"tokens.{key}"] == "observed" for key in components):
        tokens["total"] = sum(tokens[key] for key in components)
        _observed(meta, "tokens.total")


def _base_step(span, sid, kind, *, role, agent, parents):
    start, end, duration = _interval(span)
    attrs = _attrs(span)
    if duration is None:
        duration = _number(attrs.get("duration_ms"))
    step = _cc_build_step([], role=role, agent=agent, session_id=sid,
                          timestamp_ms=_milliseconds(start),
                          step_id=f"otel:{span.get('trace_id', '')}:{span.get('span_id', '')}",
                          parent_id=span.get("parent_span_id", ""),
                          source_path=_text(span.get("source", {}).get("path")))
    step["time_completed_ms"] = _milliseconds(end)
    step["duration"] = duration / 1000 if duration is not None else None
    step["timing_source"] = "otlp_span" if start is not None else "otlp_reported"
    step["_otel"] = _metadata(span, kind)
    if duration is not None:
        _observed(step["_otel"], "duration")
    step["parent_agent_id"] = parents.get(agent, "")
    step["parent_session_id"] = sid if agent else ""
    step["message_id"] = _text(attrs.get("message.uuid"))
    step["model_id"] = _text(attrs.get("model")) or _text(attrs.get("gen_ai.request.model"))
    step["cwd"] = _text(attrs.get("cwd")) or _text(attrs.get("process.working_directory"))
    step["tokens"] = {"total": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    return step


def _agent_graph(spans, by_id):
    agents, parents, info = {}, {}, {}
    for span in spans:
        attrs = _attrs(span)
        own = _text(attrs.get("agent_id"))
        if own:
            agents[_id(span)] = own
            if "parent_agent_id" in attrs:
                parents[own] = _text(attrs.get("parent_agent_id"))
    # No prefix stripping: two producers can legitimately use 'a' and 'agent-a'.
    inherited = _inherit_graph_values(spans, by_id, agents)
    nearest_tools = _inherit_graph_values(spans, by_id, {
        _id(span): span for span in spans if _span_kind(span) == "tool"
    })
    for span in spans:
        key = _id(span)
        parent_key = span.get("trace_id", ""), span.get("parent_span_id", "")
        ancestor_agent = inherited.get(parent_key, "")
        if key not in agents:
            agents[key] = inherited.get(key, "")
        elif ancestor_agent and agents[key] != ancestor_agent:
            parents.setdefault(agents[key], ancestor_agent)
        own = agents.get(key, "")
        if own:
            attrs = _attrs(span)
            agent_info = info.setdefault(own, {"agent_id": own, "parent_agent_id": parents.get(own, "")})
            description = _text(attrs.get("query_source")) or _text(attrs.get("query_source_safe"))
            if description:
                agent_info.setdefault("description", description)
            ancestor = nearest_tools.get(parent_key)
            if ancestor and inherited.get(_id(ancestor), "") != own:
                agent_info.setdefault("spawned_by", _text(_attrs(ancestor).get("tool_use_id")))
    return agents, parents, info


def _outcome(record, meta):
    attrs = _attrs(record)
    success = _boolean(attrs.get("success"))
    code = record.get("status", {}).get("code", 0)
    if success is not None:
        _observed(meta, "tool_outcomes")
        return "success" if success else "error"
    if code in (1, 2):
        _observed(meta, "tool_outcomes")
        return "success" if code == 1 else "error"
    return "unknown"


def _tool(span, children, related):
    attrs = _attrs(span)
    meta = _metadata(span, "tool")
    tool_id = _text(attrs.get("tool_use_id")) or _text(attrs.get("gen_ai.tool.call.id"))
    call = {"type": "tool_call", "tool_name": _text(attrs.get("tool_name")) or "?",
            "tool_id": tool_id, "title": "", "input": {}, "output": "", "error": None,
            "status": "unknown", "time_start": None, "time_end": None,
            "duration_ms": None, "metadata": {"_otel": meta}}
    wall_ms = _duration_ms(span)
    meta["wall_duration_ms"] = wall_ms
    meta["wall_start_time_unix_nano"] = span.get("start_time_unix_nano")
    meta["wall_end_time_unix_nano"] = span.get("end_time_unix_nano")
    executions = [s for s in children if _span_kind(s) == "tool.execution"]
    waits = [s for s in children if _span_kind(s) == "tool.blocked_on_user"]
    meta["permission_spans"] = [_id(s)[1] for s in waits]
    wait_durations = [_duration_ms(s) for s in waits]
    meta["permission_wait_ms"] = (sum(wait_durations)
                                  if waits and all(value is not None for value in wait_durations) else None)
    meta["execution_spans"] = [_id(s)[1] for s in executions]
    if len(executions) == 1:
        execution = executions[0]
        start, end, duration = _interval(execution)
        if duration is None:
            duration = _duration_ms(execution)
        call.update(time_start=_milliseconds(start), time_end=_milliseconds(end), duration_ms=duration)
        # Keep exact execution bounds separate from the parent tool's wall
        # interval and its millisecond display timestamps.
        meta["execution_start_time_unix_nano"] = start
        meta["execution_end_time_unix_nano"] = end
        if duration is not None:
            call["metadata"]["timing_source"] = "otlp_execution"
            _observed(meta, "duration")
        call["status"] = _outcome(execution, meta)
        error = _text(_attrs(execution).get("error")) or _text(execution.get("status", {}).get("message"))
        call["error"] = error or None
    elif len(executions) > 1:
        meta["reasons"]["duration"] = "multiple_execution_intervals"
    # Parent UNSET is expected even for failure; permission rejection is a
    # decision, not a measured execution failure.
    attrs_with_logs = dict(attrs)
    # A permission callback can rewrite inputs. The result describes the
    # executed call; a preceding decision describes what was proposed.
    related_by_precedence = sorted(related, key=lambda log: _event_name(log) != "tool_result")
    for log in related_by_precedence:
        values = _attrs(log)
        for key, value in values.items():
            attrs_with_logs.setdefault(key, value)
        if _event_name(log) == "tool_result":
            log_success = _boolean(values.get("success"))
            if (log_success is not None and call["status"] != "unknown"
                    and (call["status"] == "success") != log_success):
                meta.setdefault("conflicts", []).append("tool_outcomes")
            if call["status"] == "unknown":
                call["status"] = _outcome(log, meta)
                call["error"] = _text(values.get("error")) or _text(values.get("error_type")) or None
            if call["duration_ms"] is None and (duration := _number(values.get("duration_ms"))) is not None:
                call["duration_ms"] = duration
                call["metadata"]["timing_source"] = "otlp_tool_result"
                _observed(meta, "duration")
    raw_input = _content(attrs_with_logs, ("tool_input",), meta, "tool_inputs")
    parameters_only = raw_input is None
    if parameters_only:
        raw_input = _content(attrs_with_logs, ("tool_parameters",), meta, "tool_inputs")
    if raw_input is not None:
        parsed = _json_object(raw_input)
        if parsed is not None:
            call["input"] = parsed
            if parameters_only:
                # tool_parameters is a documented subset, unlike tool_input.
                meta["fields"]["tool_inputs"] = "partial"
                meta["reasons"]["tool_inputs"] = "selected_attributes_only"
            if "command" not in call["input"]:
                command = call["input"].get("full_command", call["input"].get("bash_command"))
                if isinstance(command, str):
                    call["input"]["command"] = command
        else:
            meta["fields"]["tool_inputs"] = "unavailable"
            meta["reasons"]["tool_inputs"] = "invalid_json"
            meta["raw_input"] = raw_input
    else:
        subset = {key: attrs_with_logs[key] for key in ("file_path", "full_command", "skill_name", "subagent_type")
                  if key in attrs_with_logs and attrs_with_logs[key] != "<REDACTED>"}
        if subset:
            if "full_command" in subset:
                subset["command"] = subset.pop("full_command")
            call["input"] = subset
            meta["fields"]["tool_inputs"] = "partial"
            meta["reasons"]["tool_inputs"] = "selected_attributes_only"
    output = _content(attrs_with_logs, ("new_context", "tool_output"), meta, "tool_outputs")
    for event in span.get("events", []):
        if event.get("name") == "tool.output":
            candidate_meta = deepcopy(meta)
            candidate = _content(event.get("attributes", {}), ("output", "content", "diff"),
                                 candidate_meta, "tool_outputs")
            if candidate is not None:
                output = candidate
                meta["fields"]["tool_outputs"] = candidate_meta["fields"]["tool_outputs"]
                if "tool_outputs" in candidate_meta["reasons"]:
                    meta["reasons"]["tool_outputs"] = candidate_meta["reasons"]["tool_outputs"]
                else:
                    meta["reasons"].pop("tool_outputs", None)
                break
            if output is None:
                meta["fields"]["tool_outputs"] = candidate_meta["fields"]["tool_outputs"]
                meta["reasons"]["tool_outputs"] = candidate_meta["reasons"]["tool_outputs"]
    if output is not None:
        call["output"] = _as_content(output)
    meta["correlated_logs"] = [deepcopy(log.get("source", {})) for log in related]
    request_ids = {_text(_attrs(log).get("request_id")) for log in related}
    request_ids.discard("")
    if _text(attrs.get("request_id")):
        request_ids.add(attrs["request_id"])
    meta["request_ids"] = sorted(request_ids)
    return call


def convert_otlp_claude(capture: OtlpCapture, session_id: str | None = None) -> dict[str, Any]:
    """Project one saved Claude session into steps, retaining its source graph."""
    if not capture.spans:
        raise LoadError("OTLP import requires traces; logs or metrics alone cannot reconstruct a trajectory.",
                        code="missing_traces")
    if not any(_span_kind(span) for span in capture.spans):
        raise LoadError("No supported Claude Code span names were found in the OTLP capture.",
                        code="unsupported_producer")
    if not any(_span_kind(span) in _OPERATIONS for span in capture.spans):
        raise LoadError("The capture has no supported Claude Code interaction, model or tool operations.",
                        code="no_steps")
    sessions = list_otlp_sessions(capture)
    if session_id is None:
        if len(sessions) != 1:
            choices = ", ".join(item["id"] for item in sessions)
            exc = LoadError(f"OTLP capture contains multiple sessions; select session_id from: {choices}",
                            code="ambiguous_session")
            exc.sessions = sessions
            raise exc
        session_id = sessions[0]["id"]
    selected = next((item for item in sessions if item["id"] == session_id), None)
    if selected is None:
        raise LoadError(f"Session {session_id!r} is not present in the OTLP capture.", code="session_not_found")
    assigned, all_by_id, trace_sessions = _session_index(capture)
    spans = [span for span in capture.spans if assigned.get(_id(span)) == session_id]
    selected_ids = {_id(span) for span in spans}
    logs = []
    for log in capture.logs:
        explicit = _text(_attrs(log).get("session.id"))
        trace = log.get("trace_id", "")
        if explicit == session_id or (not explicit and (
                _id(log) in selected_ids or trace_sessions.get(trace) == {session_id}
                or (session_id == f"trace:{trace}" and not trace_sessions.get(trace)))):
            logs.append(log)
    warnings = list(capture.warnings)
    if selected["incomplete"]:
        warnings.append("No session.id was recorded; this trajectory contains one trace only and may be incomplete.")
    missing_parents = [span for span in spans if span.get("parent_span_id")
                       and (span.get("trace_id", ""), span["parent_span_id"]) not in all_by_id]
    if missing_parents:
        warnings.append(f"{len(missing_parents)} spans reference parents absent from the capture.")
    agents, parents, agent_info = _agent_graph(spans, all_by_id)
    children = defaultdict(list)
    for span in spans:
        children[(span.get("trace_id", ""), span.get("parent_span_id", ""))].append(span)
    logs_by_request, logs_by_tool, logs_by_span = defaultdict(list), defaultdict(list), defaultdict(list)
    logs_by_client = defaultdict(list)
    request_counts = _correlation_index(spans, agents, "llm_request", _request_id)
    client_counts = _correlation_index(spans, agents, "llm_request",
                                       lambda span: _text(_attrs(span).get("client_request_id")))
    tool_counts = _correlation_index(spans, agents, "tool", _tool_id)
    for log in logs:
        request_id = _request_id(log)
        if request_id:
            logs_by_request[(log.get("trace_id", ""), request_id)].append(log)
        client_id = _text(_attrs(log).get("client_request_id"))
        if client_id:
            logs_by_client[(log.get("trace_id", ""), client_id)].append(log)
        tool_id = _tool_id(log)
        if tool_id:
            logs_by_tool[(log.get("trace_id", ""), tool_id)].append(log)
        logs_by_span[_id(log)].append(log)
    steps, model_steps, tool_steps = [], defaultdict(list), []
    for raw_index, span in enumerate(spans):
        kind, attrs = _span_kind(span), _attrs(span)
        agent = agents.get(_id(span), "")
        if kind not in _OPERATIONS:
            continue
        role, mapped_kind = {
            "interaction": ("user", "prompt"), "llm_request": ("assistant", "model_request"),
            "tool": ("tool", "tool"), "compaction": ("compaction", "compaction"),
        }[kind]
        step = _base_step(span, session_id, mapped_kind, role=role, agent=agent, parents=parents)
        step["raw_index"] = raw_index
        meta = step["_otel"]
        request_id = _request_id(span)
        correlated = _correlated_logs(span, agent, request_id, logs_by_request, request_counts)
        if kind == "llm_request":
            client_id = _text(attrs.get("client_request_id"))
            seen_logs = {id(log) for log in correlated}
            for log in _correlated_logs(span, agent, client_id, logs_by_client, client_counts):
                # A final-attempt client ID cannot join a different server request.
                if (id(log) not in seen_logs and (not request_id or not _request_id(log)
                                                  or request_id == _request_id(log))):
                    correlated.append(log)
                    seen_logs.add(id(log))
            merged = _merge_model_attributes(span, correlated, meta)
            if meta.get("conflicts"):
                meta["conflicts"] = sorted(set(meta["conflicts"]))
                warnings.append(f"Conflicting span/log usage for request {request_id!r}; retained the first valid counts, preferring spans.")
            _fill_usage(step, merged)
            if step["duration"] is None and (duration := _number(merged.get("duration_ms"))) is not None:
                step["duration"] = duration / 1000
                step["timing_source"] = "otlp_api_request"
                _observed(meta, "duration")
            content = _content(merged, ("response.model_output", "response"), meta, "text")
            step["finish"] = _text(merged.get("stop_reason"))
            if not step["finish"] and isinstance(merged.get("gen_ai.response.finish_reasons"), list):
                step["finish"] = ", ".join(str(value) for value in merged["gen_ai.response.finish_reasons"])
            success = _boolean(merged.get("success"))
            status = span.get("status", {}).get("code", 0)
            if status == 2:
                success = False
            elif success is None and status == 1:
                success = True
            elif success is None and any(_event_name(log) == "api_error" for log in correlated):
                success = False
            if success is False:
                # Model failures are scaffold failures, not fabricated tool calls.
                step["finish"] = "error"
            step["model_id"] = _text(merged.get("model")) or _text(merged.get("gen_ai.request.model"))
            step["message_id"] = _text(merged.get("message.uuid"))
            if not request_id:
                server_ids = {_request_id(log) for log in correlated if _request_id(log)}
                if len(server_ids) == 1:
                    request_id = server_ids.pop()
            meta.update(request_id=request_id, client_request_id=client_id, attempts=merged.get("attempt"),
                        ttft_ms=_number(merged.get("ttft_ms")),
                        first_content_ms=_number(merged.get("first_content_ms")),
                        reported_duration_ms=_number(merged.get("duration_ms")),
                        success=success,
                        error=_text(merged.get("error")) or _text(span.get("status", {}).get("message")),
                        error_class=_text(merged.get("error_class")),
                        status_code=merged.get("status_code"),
                        retry_events=deepcopy([event for event in span.get("events", [])
                                              if event.get("name") == "gen_ai.request.attempt"]),
                        correlated_logs=[deepcopy(log.get("source", {})) for log in correlated])
            if request_id:
                model_steps[(span.get("trace_id", ""), agent, request_id)].append(step)
        elif kind == "interaction":
            content = _content(attrs, ("user_prompt", "new_context"), meta, "text")
            if content is None:
                # A user_prompt event can be joined by the interaction span ID,
                # never by whichever prompt happens to be closest in time.
                for log in logs_by_span[_id(span)]:
                    if _event_name(log) == "user_prompt":
                        content = _content(_attrs(log), ("prompt", "prompt_text", "user_prompt"), meta, "text")
                        if content is not None:
                            break
        elif kind == "tool":
            tool_id = _tool_id(span)
            related_logs = _correlated_logs(span, agent, tool_id, logs_by_tool, tool_counts)
            call = _tool(span, children[_id(span)], related_logs)
            if call["metadata"]["_otel"].get("conflicts"):
                warnings.append(f"Conflicting span/log outcome for tool {tool_id!r}; retained execution evidence.")
            step["parts"] = [call]
            step["tool_calls"] = [call]
            step["tool_call_count"] = 1
            step["error_count"] = int(call["status"] == "error")
            step["text_preview"] = f"[Tool: {call['tool_name']}]"
            for field in ("tool_inputs", "tool_outputs", "tool_outcomes"):
                meta["fields"][field] = call["metadata"]["_otel"]["fields"][field]
                if field in call["metadata"]["_otel"]["reasons"]:
                    meta["reasons"][field] = call["metadata"]["_otel"]["reasons"][field]
                else:
                    meta["reasons"].pop(field, None)
            tool_steps.append(step)
            content = None
        else:
            content = _content(attrs, ("summary", "new_context"), meta, "text")
            step.update(message_type="compaction", is_compaction_checkpoint=True,
                        compaction_reason=_text(attrs.get("reason")))
        if kind != "tool" and content is not None:
            text = _as_content(content)
            step["parts"] = [{"type": "text", "text": text}]
            step["text_preview"] = text
        steps.append(step)
    # Join a logical tool only when an explicit request ID identifies exactly
    # one model operation in the same agent lane. Ambiguous evidence stays detached.
    attached = set()
    for step in tool_steps:
        call = step["tool_calls"][0]
        request_ids = call["metadata"]["_otel"]["request_ids"]
        targets = [target for request_id in request_ids
                   for target in model_steps.get((step["_otel"]["trace_id"], step["agent"], request_id), [])]
        if len(targets) == 1:
            target = targets[0]
            target["tool_calls"].append(call)
            target["parts"].append(call)
            target["tool_call_count"] += 1
            target["error_count"] += int(call["status"] == "error")
            attached.add(step["id"])
    steps = [step for step in steps if step["id"] not in attached]
    for step in steps:
        if not step["tool_calls"]:
            continue
        for field in ("tool_inputs", "tool_outputs", "tool_outcomes"):
            states = [call["metadata"]["_otel"]["fields"][field] for call in step["tool_calls"]]
            if all(state == "observed" for state in states):
                _observed(step["_otel"], field)
            elif any(state in ("observed", "partial") for state in states):
                step["_otel"]["fields"][field] = "partial"
                step["_otel"]["reasons"][field] = "incomplete_tool_evidence"
            else:
                step["_otel"]["fields"][field] = "unavailable"

    agent_depths = {"": 0}

    def depth(agent):
        trail, seen = [], set()
        while agent and agent not in seen and agent not in agent_depths:
            seen.add(agent)
            trail.append(agent)
            agent = parents.get(agent, "")
        level = agent_depths.get(agent, 0)
        for item in reversed(trail):
            level += 1
            agent_depths[item] = level
        return level

    steps.sort(key=lambda step: (step["time_created_ms"] is None,
                                step["_otel"].get("start_time_unix_nano") or 0, step["id"]))
    for index, step in enumerate(steps):
        step["index"] = index
        step["session_depth"] = depth(step["agent"])
    starts = [s["start_time_unix_nano"] for s in spans if _interval(s)[0] is not None]
    ends = [s["end_time_unix_nano"] for s in spans if _interval(s)[1] is not None]
    assistant = [step for step in steps if step["role"] == "assistant"]
    calls = [call for step in steps for call in step["tool_calls"]]
    tool_counts = Counter(call["tool_name"] for call in calls)
    prompt = next((s["text_preview"] for s in steps if s["role"] == "user" and not s["agent"]), "")
    model = next((s["model_id"] for s in assistant if s["model_id"]), "")
    sub_agents = []
    for agent in sorted({s["agent"] for s in steps if s["agent"]}):
        own = [s for s in steps if s["agent"] == agent]
        sub_agents.append({**agent_info.get(agent, {}), "agent_id": agent,
                           "parent_agent_id": parents.get(agent, ""), "session_id": session_id,
                           "parent_session_id": session_id, "session_depth": depth(agent),
                           "turns": len(own), "tool_calls": sum(s["tool_call_count"] for s in own),
                           "source": "otlp"})
    source_info = {key: value for key, value in capture.raw.items() if key not in {"requests", "_input_files"}}
    return {
        "metadata": {"session_id": session_id, "agent": "claude-code", "model": model,
                     "directory": "", "directory_name": "", "timestamp_utc": _iso(min(starts)) if starts else "",
                     "sub_agent_count": len(sub_agents), "event_count": len(spans) + len(logs)},
        "input": {"prompt": prompt, "prompt_length": len(prompt)},
        "output": {"patch": "", "patch_length": 0, "patch_lines": 0, "has_patch": False, "error": None},
        "timing": {"total_duration": (max(ends) - min(starts)) / 1e9 if starts and ends else 0,
                   "started_at": _iso(min(starts)) if starts else "",
                   "finished_at": _iso(max(ends)) if ends else ""},
        "token_usage": {"total_tokens": sum(s["tokens"]["total"] for s in assistant),
                        "prompt_tokens": sum(s["tokens"]["input"] for s in assistant),
                        "completion_tokens": sum(s["tokens"]["output"] for s in assistant)},
        "stats": {"total_messages": len(steps), "user_messages": sum(s["role"] == "user" for s in steps),
                  "assistant_messages": len(assistant), "total_tool_calls": len(calls),
                  "tool_call_breakdown": dict(tool_counts),
                  "failed_tool_calls": sum(c["status"] == "error" for c in calls), "reasoning_steps": 0},
        "trajectory": [], "_source_format": "otlp", "_otlp_format": True,
        "_otlp_parsed_steps": steps, "_otlp_sub_agents": sub_agents,
        "_source_sha256": capture.sha256, "_load_warnings": list(dict.fromkeys(warnings)),
        "_input_files": list(capture.raw.get("_input_files", [])),
        "_otel": {"profile": "claude_code", "profile_version": PROFILE_VERSION,
                  "session_id": session_id, "trace_ids": selected["trace_ids"],
                  "mapping": "operation", "coverage": "captured_operations",
                  "incomplete": selected["incomplete"] or bool(missing_parents),
                  "limitations": (["session_identity_missing"] if selected["incomplete"] else [])
                                 + (["parents_missing"] if missing_parents else []),
                  "source": deepcopy(source_info), "spans": deepcopy(spans), "logs": deepcopy(logs),
                  "agents": sub_agents},
    }
