"""Conservative, request-scoped reading of Claude native prompt records.

Request objects retain a prefix reference and a tail rather than duplicating
the growing transcript. Only the selected request is materialized. References
are resolved in original recording order, within the same session and agent;
later definitions never repair an earlier request silently.
"""

from __future__ import annotations

import copy
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .formats.common import _iso_to_epoch_ms

_TRANSPORT_KEYS = frozenset({
    "headers", "http_headers", "request_headers", "authorization", "api_key", "apikey",
    "x-api-key", "access_token", "auth_token", "client_secret", "cookie", "cookies",
    "authentication", "credentials", "auth", "password",
})
_PARAMETERS = frozenset({
    "model", "max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "thinking",
    "tool_choice", "output_config", "stream", "service_tier", "inference_geo",
})
_RECORD_FIELDS = frozenset({
    "type", "id", "requestId", "sessionId", "agentId", "isSidechain", "version", "timestamp",
    "params", "shapeHash", "base", "keep", "tail", "querySource", "parentUuid", "uuid",
})
_NATIVE_LIMIT = "Native request recording is version-dependent; missing input cannot be inferred from transcript messages."
_PROVIDER_LIMIT = "This reconstructs recorded client input, not provider-side prompt processing."
_MISSING_SHAPE = "System instructions and tool definitions are not fully recorded for this request."
_MISSING_MESSAGES = "Some request messages are unavailable because recorded references are missing or invalid."
_DELTA_LIMIT = "Explicit base/prefix replay has not been independently validated against an outgoing request."
_NO_BASE = "A request delta has no resolvable explicit base; its prefix is unavailable."
_UNKNOWN_FIELDS = "Unsupported request or shape fields may affect the reconstructed input."
_TRANSPORT_LIMIT = "Transport credentials and headers are omitted from prompt evidence."
_PREFIX_LIMIT = "Request prefix inspection reached its safety limit; some retained references are unavailable."
_BASELINE_LIMIT = "The previous prompt is unavailable; changes are not inferred from neighboring requests."
_MAX_MESSAGES = 100_000
_MAX_MISSING = 256
_MAX_PREFIX_WALK = 1024
_MAX_PREFIX_CACHE = 4


def _boundary_copy(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Copy a metadata boundary without changing system/tool/message bodies."""
    result = {key: copy.deepcopy(item) for key, item in value.items()
              if str(key).lower() not in _TRANSPORT_KEYS}
    return result, len(result) != len(value)


def preserve_prompt_record(record: dict[str, Any], *, source_path: str, raw_index: int,
                           session_id: str, agent_id: str) -> dict[str, Any]:
    """Preserve native evidence and provenance, omitting transport secrets."""
    native, omitted = _boundary_copy(record)
    if isinstance(native.get("params"), dict):
        native["params"], params_omitted = _boundary_copy(native["params"])
        omitted |= params_omitted
    if record.get("type") == "api-request-shape" and isinstance(native.get("shape"), dict):
        native["shape"], shape_omitted = _boundary_copy(native["shape"])
        omitted |= shape_omitted
    return {"record": native, "source_path": source_path, "raw_index": raw_index,
            "session_id": session_id, "agent_id": agent_id, "transport_omitted": omitted}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _source(row: dict[str, Any], kind: str, reference: str = "") -> dict[str, Any]:
    index = row.get("raw_index")
    return {"source_path": _text(row.get("source_path")),
            "raw_index": index if isinstance(index, int) and not isinstance(index, bool) else None,
            "session_id": _text(row.get("session_id")), "agent_id": _text(row.get("agent_id")),
            "kind": kind, "reference": reference,
            **({"transport_omitted": True} if row.get("transport_omitted") else {})}


def _same_json(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False


@dataclass
class _Blob:
    message: dict[str, Any] | None
    source: dict[str, Any]
    missing: str = ""


@dataclass
class _Request:
    catalog: dict[str, Any]
    row: dict[str, Any]
    native: dict[str, Any]
    shape: dict[str, Any] | None
    shape_source: dict[str, Any] | None
    base: _Request | None
    keep: int
    tail: tuple[_Blob, ...]
    length: int
    limitations: list[str]
    unsupported_fields: list[str]
    message_missing: tuple[str, ...]
    message_issues: tuple[str, ...]
    prefix_issue: str
    prefix_cache: dict[int, tuple[str, ...]] = field(default_factory=dict)


def _prefix_missing(state: _Request, count: int) -> list[str]:
    """Report only unavailable messages actually retained by a prefix."""
    if not state.message_missing:
        return []
    if count >= state.length:
        return list(state.message_missing)
    pending: list[tuple[_Request, int, list[str]]] = []
    missing: list[str] = []
    current: _Request | None = state
    while current is not None and count:
        if not current.message_missing:
            break
        if count >= current.length:
            missing = list(current.message_missing)
            break
        if count in current.prefix_cache:
            missing = list(current.prefix_cache[count])
            break
        if len(pending) >= _MAX_PREFIX_WALK:
            missing = ["prefix:inspection-limit"]
            break
        local = list(current.message_issues)
        if count > current.keep:
            local.extend(blob.missing for blob in current.tail[:count - current.keep] if blob.missing)
        pending.append((current, count, local))
        count = min(count, current.keep)
        if count and current.base is None and current.prefix_issue:
            local.append(current.prefix_issue)
        current = current.base
    for current, size, local in reversed(pending):
        missing = list(dict.fromkeys([*local, *missing]))
        if len(missing) > _MAX_MISSING:
            missing = missing[:_MAX_MISSING] + ["references:truncated"]
        if len(current.prefix_cache) >= _MAX_PREFIX_CACHE:
            current.prefix_cache.pop(next(iter(current.prefix_cache)))
        current.prefix_cache[size] = tuple(missing)
    return missing


def _valid_system(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, dict) for item in value))


def _valid_tools(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _materialize(state: _Request) -> list[tuple[int, _Blob]]:
    """Replay a selected explicit lineage without recursion or dense gaps."""
    operations = []
    current: _Request | None = state
    while current is not None:
        operations.append(current)
        current = current.base
    refs: list[tuple[int, _Blob]] = []
    for operation in reversed(operations):
        del refs[bisect_left(refs, operation.keep, key=lambda item: item[0]):]
        refs.extend((operation.keep + index, blob) for index, blob in enumerate(operation.tail))
    return refs


def _parameters(record: dict[str, Any]) -> dict[str, Any]:
    params = record.get("params")
    if not isinstance(params, dict):
        return {}
    return {key: copy.deepcopy(value) for key, value in params.items() if key in _PARAMETERS
            and not (isinstance(value, float) and not math.isfinite(value))}


def _tool_changes(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """Name-level tool changes only when recorded names are unambiguous."""
    old_names = [_text(tool.get("name")) for tool in before]
    new_names = [_text(tool.get("name")) for tool in after]
    if (not all(old_names) or not all(new_names) or len(set(old_names)) != len(old_names)
            or len(set(new_names)) != len(new_names)):
        return {}
    old, new = dict(zip(old_names, before, strict=True)), dict(zip(new_names, after, strict=True))
    return {"added_tools": [name for name in new if name not in old],
            "removed_tools": [name for name in old if name not in new],
            "updated_tools": [name for name in new if name in old and not _same_json(old[name], new[name])],
            "order_changed": set(old) == set(new) and old_names != new_names}


def _prompt_events(state: _Request, baseline: _Request | None, *, keep_valid: bool,
                   tail_valid: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Describe prompt evolution without guessing a neighboring baseline."""
    own_source = _source(state.row, "request", _text(state.native.get("id")))
    refs = [own_source]
    if baseline is not None:
        refs.insert(0, _source(baseline.row, "request", _text(baseline.native.get("id"))))
    events: list[dict[str, Any]] = []

    def event(kind: str, label: str, **extra: Any) -> None:
        events.append({"kind": kind, "label": label, "source_refs": copy.deepcopy(refs), **extra})

    prefix_missing = _prefix_missing(baseline, state.keep) if baseline is not None and state.keep else []
    retained = state.keep if keep_valid and (state.keep == 0 or (baseline is not None and not prefix_missing)) else None
    known_tail = sum(blob.message is not None for blob in state.tail)
    removed = baseline.length - state.keep if baseline is not None and not baseline.message_missing else None
    reset = state.keep == 0 if baseline is not None and keep_valid else None
    transition = {
        "baseline_request_id": baseline.catalog["id"] if baseline is not None else None,
        "baseline_evidence": "explicit_base" if baseline is not None else "unavailable",
        "recorded_keep": state.keep if keep_valid else None,
        "recorded_tail_count": len(state.tail) if tail_valid else None,
        "retained_message_count": retained,
        "known_tail_message_count": known_tail if tail_valid else None,
        "added_message_count": known_tail if baseline is not None and tail_valid else None,
        "removed_message_count": removed,
        "reset": reset,
    }
    if baseline is None:
        event("request_snapshot", "Recorded request snapshot")
        event("baseline_unavailable", "Previous prompt unavailable")
        if state.shape_source is not None:
            events.append({"kind": "prompt_shape_recorded", "label": "Recorded instruction and tool snapshot",
                           "source_refs": [copy.deepcopy(state.shape_source)]})
        if state.keep:
            event("context_prefix_unavailable", "Retained prefix unavailable", recorded_count=state.keep)
        if tail_valid:
            event("message_tail_recorded", "Recorded message tail", recorded_count=len(state.tail),
                  known_count=known_tail)
        return events, transition

    shape_sources = [source for source in (baseline.shape_source, state.shape_source) if source is not None]
    before, after = baseline.shape or {}, state.shape or {}
    if (_valid_system(before.get("system")) and _valid_system(after.get("system"))
            and not _same_json(before["system"], after["system"])):
        events.append({"kind": "system_changed", "label": "System instructions changed",
                       "source_refs": copy.deepcopy(shape_sources)})
    if (_valid_tools(before.get("tools")) and _valid_tools(after.get("tools"))
            and not _same_json(before["tools"], after["tools"])):
        changes = _tool_changes(before["tools"], after["tools"])
        events.append({"kind": "tools_changed", "label": "Tool definitions changed",
                       "source_refs": copy.deepcopy(shape_sources),
                       **changes})
        for field_name, kind, label in (("added_tools", "tool_added", "Tool added"),
                                        ("removed_tools", "tool_removed", "Tool removed"),
                                        ("updated_tools", "tool_updated", "Tool updated")):
            for name in changes.get(field_name, []):
                events.append({"kind": kind, "label": label, "tool_name": name, "target_id": name,
                               "count": 1, "source_refs": copy.deepcopy(shape_sources)})
    if reset:
        event("context_reset", "Recorded context reset", removed_count=removed)
    elif state.keep:
        event("context_prefix_retained" if retained is not None else "context_prefix_unavailable",
              "Retained recorded message prefix" if retained is not None else "Retained prefix unavailable",
              recorded_count=state.keep, known_count=retained)
    if state.keep < baseline.length and not reset:
        event("message_tail_replaced", "Replaced prior message tail", removed_count=removed)
    if tail_valid:
        event("message_tail_recorded", "Recorded message tail", recorded_count=len(state.tail), known_count=known_tail)
    return events, transition


def inspect_claude(raw: dict[str, Any], steps: list[dict[str, Any]],
                   request_id: str | None = None) -> dict[str, Any]:
    """List native requests and reconstruct the selected recorded input.

    A positive ``keep`` is replayed only with an explicit reference to an
    earlier request owned by the same session/agent. Missing or unsupported
    lineage stays partial; the preceding transcript is never substituted.
    Unknown selection identifiers raise :class:`ValueError`.
    """
    del steps  # Native request IDs do not prove a transcript response mapping.
    rows = raw.get("_claude_prompt_records")
    if not isinstance(rows, list):
        rows = []
    stores: dict[tuple[str, str], dict[str, Any]] = {}
    requests: list[_Request] = []
    id_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("record"), dict):
            continue
        record = row["record"]
        owner = (_text(row.get("session_id")), _text(row.get("agent_id")))
        store = stores.setdefault(owner, {"shapes": {}, "blobs": {}, "requests": {}})
        kind = record.get("type")
        if kind in ("api-request-shape", "api-request-blob"):
            shape_record = kind == "api-request-shape"
            name = _text(record.get("shapeHash" if shape_record else "hash"))
            value = record.get("shape" if shape_record else "message")
            table = store["shapes" if shape_record else "blobs"]
            if not name:
                continue
            existing = table.get(name)
            if name in table and (existing is None or not _same_json(existing[0], value)):
                table[name] = None  # Conflicting identities remain ambiguous.
            elif isinstance(value, dict):
                table[name] = (value, _source(row, "shape" if shape_record else "blob", name))
            else:
                table[name] = None
            continue
        if kind != "api-request":
            continue

        native_id = _text(record.get("id")) or _text(record.get("requestId"))
        key = "claude:" + ":".join(quote(item, safe="") for item in (*owner, native_id or f"line-{row.get('raw_index')}"))
        id_counts[key] = id_counts.get(key, 0) + 1
        identifier = key if id_counts[key] == 1 else f"{key}:duplicate-{id_counts[key]}"
        limitations = [_NATIVE_LIMIT, _PROVIDER_LIMIT]
        missing: list[str] = []
        message_issues: list[str] = []
        prefix_issue = ""
        unsupported = sorted(str(field) for field in record if field not in _RECORD_FIELDS
                             and str(field).lower() not in _TRANSPORT_KEYS)
        if isinstance(record.get("params"), dict):
            unsupported.extend(f"params.{key}" for key in record["params"] if key not in _PARAMETERS)
        if row.get("transport_omitted"):
            limitations.append(_TRANSPORT_LIMIT)
        shape_hash = _text(record.get("shapeHash"))
        shape_entry = store["shapes"].get(shape_hash) if shape_hash else None
        shape, shape_source = shape_entry if shape_entry is not None else (None, None)
        if shape_source is not None and shape_source.get("transport_omitted"):
            limitations.append(_TRANSPORT_LIMIT)
        if shape is None:
            missing.append(f"shape:{shape_hash or 'unrecorded'}")
            limitations.append(_MISSING_SHAPE)
        else:
            unsupported.extend(f"shape.{key}" for key in shape if key not in {"system", "tools"})
            if not _valid_system(shape.get("system")) or not _valid_tools(shape.get("tools")):
                missing.append(f"shape:{shape_hash}")
                limitations.append(_MISSING_SHAPE)
        keep = record.get("keep", 0)
        keep_valid = isinstance(keep, int) and not isinstance(keep, bool) and 0 <= keep <= _MAX_MESSAGES
        if not keep_valid:
            missing.append("prefix:invalid")
            message_issues.append("prefix:invalid")
            limitations.append(_MISSING_MESSAGES)
            keep = 0
        base = None
        native_base = _text(record.get("base"))
        reference_base = store["requests"].get(native_base) if native_base and keep_valid else None
        if reference_base is not None and keep > reference_base.length:
            reference_base = None
        if keep:
            candidate = reference_base
            if candidate is None or keep > candidate.length:
                prefix_issue = f"base:{native_base or 'unrecorded'}"
                missing.append(prefix_issue)
                limitations.append(_NO_BASE)
            else:
                base = candidate
                limitations.append(_DELTA_LIMIT)
                missing.extend(_prefix_missing(base, keep))
        tail = record.get("tail")
        tail_valid = isinstance(tail, list) and len(tail) <= _MAX_MESSAGES
        if not tail_valid:
            tail = []
            missing.append("messages:unrecorded")
            message_issues.append("messages:unrecorded")
        refs = []
        for position, name in enumerate(tail):
            identity = name if isinstance(name, str) else ""
            entry = store["blobs"].get(identity) if identity else None
            message, source = entry if entry is not None else (None, _source(row, "blob", identity))
            if (message is None or _text(message.get("role")) not in {"user", "assistant", "system", "developer", "tool"}
                    or not isinstance(message.get("content"), (str, list))):
                missing_ref = f"blob:{identity or f'invalid-{position}'}"
                missing.append(missing_ref)
                refs.append(_Blob(None, source, missing_ref))
            else:
                refs.append(_Blob(message, source))
        if any(item.startswith(("blob:", "messages:", "prefix:")) for item in missing):
            limitations.append(_MISSING_MESSAGES)
        if "prefix:inspection-limit" in missing:
            limitations.append(_PREFIX_LIMIT)
        if unsupported:
            limitations.append(_UNKNOWN_FIELDS)
        if native_id and native_id in store["requests"]:
            limitations.append("Duplicate native request identifiers make request lineage ambiguous.")
        if reference_base is None:
            limitations.append(_BASELINE_LIMIT)
        missing = list(dict.fromkeys(missing))
        if len(missing) > _MAX_MISSING:
            missing = missing[:_MAX_MISSING] + ["references:truncated"]
            limitations.append("Missing reference details exceed the display limit.")
        catalog = {"id": identifier, "label": native_id or f"#{row.get('raw_index')}",
                   "evidence": "reconstructed", "scope": "recorded_request",
                   "model": _text(_parameters(record).get("model")),
                   "session_id": owner[0], "agent_id": owner[1], "step_index": None,
                   "timestamp_ms": _iso_to_epoch_ms(record.get("timestamp")),
                   "missing_references": missing,
                   "source_path": _text(row.get("source_path")), "raw_index": row.get("raw_index"),
                   "message_count": keep + len(refs)}
        if reference_base is not None:
            catalog["previous_request_id"] = reference_base.catalog["id"]
        state = _Request(catalog, row, record, shape, shape_source, base, keep, tuple(refs),
                         keep + len(refs), list(dict.fromkeys(limitations)), unsupported,
                         tuple(item for item in missing if not item.startswith("shape:")),
                         tuple(message_issues), prefix_issue)
        catalog["prompt_events"], catalog["transition"] = _prompt_events(
            state, reference_base, keep_valid=keep_valid, tail_valid=tail_valid)
        requests.append(state)
        if native_id:
            store["requests"][native_id] = state if native_id not in store["requests"] else None

    selected = next((item for item in requests if item.catalog["id"] == request_id), None) if request_id is not None else (
        requests[-1] if requests else None)
    if request_id is not None and selected is None:
        raise ValueError("Unknown prompt construction request_id")
    detail = None
    if selected is not None:
        refs = _materialize(selected)
        sources = [_source(selected.row, "request", _text(selected.native.get("id")))]
        if selected.shape_source is not None:
            sources.append(selected.shape_source)
        sources.extend(blob.source for _, blob in refs if blob.message is not None)
        system = selected.shape.get("system") if selected.shape else None
        tools = selected.shape.get("tools") if selected.shape else None
        detail = {**selected.catalog,
                  "system": copy.deepcopy(system) if _valid_system(system) else [],
                  "tools": copy.deepcopy(tools) if _valid_tools(tools) else [],
                  "messages": [copy.deepcopy(blob.message) for _, blob in refs if blob.message is not None],
                  "message_indices": [index for index, blob in refs if blob.message is not None],
                  "parameters": _parameters(selected.native),
                  "sources": list({(source["source_path"], source["raw_index"], source["kind"], source["reference"]): source
                                   for source in sources}.values()),
                  "limitations": selected.limitations,
                  "unsupported_fields": selected.unsupported_fields}
    return {"available": bool(requests), "requests": [item.catalog for item in requests],
            "selected_request": detail,
            "limitations": [_NATIVE_LIMIT, _PROVIDER_LIMIT] if requests else [
                "This trajectory does not contain native Claude request records."]}
