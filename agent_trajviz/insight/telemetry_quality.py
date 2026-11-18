"""Evidence coverage for imported telemetry, independent of placeholder values.

The normalized step model uses numeric zero internally. OTel evidence flags
are authoritative: an absent observation must never become a measured zero.
Native formats retain their established analysis behavior.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

TOKEN_FIELDS = ("total", "input", "output", "reasoning", "cache_read", "cache_write")
QUALITY_FIELDS = tuple(f"tokens.{key}" for key in TOKEN_FIELDS) + (
    "duration", "text", "tool_inputs", "tool_outputs", "tool_outcomes", "context",
)


def is_otel_step(step: dict) -> bool:
    return isinstance(step.get("_otel"), dict)


def field_observed(step: dict, field: str) -> bool:
    """Whether a step supplies evidence; native steps keep existing semantics."""
    metadata = step.get("_otel")
    return not isinstance(metadata, dict) or metadata.get("fields", {}).get(field) == "observed"


def tool_field_observed(tool: dict, field: str) -> bool:
    native_metadata = tool.get("metadata") or {}
    if field == "tool_outcomes" and native_metadata.get("_outcome_observed") is False:
        return False
    if field == "tool_inputs" and native_metadata.get("_input_observed") is False:
        return False
    if field == "tool_outputs" and native_metadata.get("_output_observed") is False:
        return False
    metadata = native_metadata.get("_otel")
    return not isinstance(metadata, dict) or metadata.get("fields", {}).get(field) == "observed"


def _items(steps: list[dict], field: str) -> list[dict]:
    if field.startswith("tool_"):
        return [
            (tool.get("metadata") or {}).get("_otel", {})
            for step in steps for tool in step.get("tool_calls", [])
        ]
    if field.startswith("tokens.") or field in {"duration", "context"}:
        return [step["_otel"] for step in steps if (step.get("_otel") or {}).get("kind") == "model_request"]
    return [step["_otel"] for step in steps if (step.get("_otel") or {}).get("kind") in {"prompt", "model_request"}]


def data_quality(steps: list[dict], raw: dict | None = None) -> dict[str, Any]:
    """Describe coverage over captured operations, without claiming capture completeness."""
    metadata = (raw or {}).get("_otel")
    otel = isinstance(metadata, dict) or any(is_otel_step(step) for step in steps)
    if not otel:
        return {"source": "native", "basis": "source_records", "capture_complete": None, "fields": {}}
    metadata = metadata if isinstance(metadata, dict) else {}
    fields: dict[str, dict] = {}
    for field in QUALITY_FIELDS:
        items = _items(steps, field)
        states = Counter(item.get("fields", {}).get(field, "unknown") for item in items)
        observed = states["observed"]
        if not items:
            status = "unknown"
        elif observed == len(items):
            status = "observed"
        elif observed or states["partial"]:
            status = "partial"
        elif states["unknown"]:
            status = "unknown"
        else:
            status = "unavailable"
        reasons = Counter(
            str(item.get("reasons", {}).get(field) or "not_recorded")
            for item in items if item.get("fields", {}).get(field) != "observed"
        )
        fields[field] = {
            "status": status, "observed": observed, "partial": states["partial"], "total": len(items),
            "reasons": dict(sorted(reasons.items())),
        }
    return {
        "source": "otlp", "profile": metadata.get("profile", "claude_code"),
        "profile_version": metadata.get("profile_version", 1),
        "basis": "captured_operations", "capture_complete": None,
        "fields": fields,
        "limitations": list(metadata.get("limitations", [])),
    }


def supports(steps: list[dict], field: str, *, require_complete: bool = True) -> bool:
    """Whether evidence is sufficient for an analysis depending on *field*."""
    if not any(is_otel_step(step) for step in steps):
        return True
    items = _items(steps, field)
    observed = sum(item.get("fields", {}).get(field) == "observed" for item in items)
    return bool(items) and observed == len(items) if require_complete else observed > 0


def public_token_totals(totals: dict, quality: dict) -> dict:
    if quality["source"] != "otlp":
        return totals
    return {
        key: value if quality["fields"][f"tokens.{key}"]["observed"] else None
        for key, value in totals.items()
    }


def view_availability(view_id: str, quality: dict, steps: list[dict] | None = None) -> dict:
    """Typed view-envelope availability, including dependencies for empty results."""
    dependencies = {
        "context_pressure": ("context",), "context_usage": ("context", "text", "tool_inputs", "tool_outputs"),
        "premature_compactions": ("context",),
        "file_interactions": ("tool_inputs",), "tool_selection": ("tool_inputs",),
        "skill_calls": ("tool_inputs",),
        "edit_thrash": ("tool_inputs",), "repeated_searches": ("tool_inputs",),
        "fruitless_streaks": ("tool_inputs", "tool_outputs", "tool_outcomes"),
        "error_clusters": ("tool_outputs", "tool_outcomes"),
        "failure_patterns": ("tool_outcomes",), "failure_chains": ("tool_outcomes",),
        "plan": ("tool_inputs", "tool_outputs"), "phases": ("tool_inputs",),
        "phase_regressions": ("tool_inputs",), "tool_outcomes": ("tool_outcomes",),
        "token_usage": ("tokens.total",), "agent_tokens": ("tokens.total",),
        "step_durations": ("duration",),
        "bottlenecks": ("duration",),
    }.get(view_id, ())
    if quality["source"] != "otlp":
        return {"status": "unknown", "basis": quality["basis"], "fields": {}}
    if view_id in {"token_usage", "agent_tokens"}:
        fields = {f"tokens.{key}": quality["fields"][f"tokens.{key}"] for key in TOKEN_FIELDS}
        complete = all(field["status"] == "observed" for field in fields.values())
        available = any(field["observed"] or field["partial"] for field in fields.values())
        status = "observed" if complete else "partial" if available else "unavailable"
        return {"status": status, "basis": quality["basis"], "fields": fields}
    if view_id == "tool_durations":
        items = [(call.get("metadata") or {}).get("_otel", {})
                 for step in (steps or []) for call in step.get("tool_calls", [])]
        observed = sum(item.get("fields", {}).get("duration") == "observed" for item in items)
        partial = sum(item.get("fields", {}).get("duration") == "partial" for item in items)
        status = ("observed" if items and observed == len(items) else "partial" if observed or partial
                  else "unavailable")
        reasons = Counter(str(item.get("reasons", {}).get("duration") or "not_recorded")
                          for item in items if item.get("fields", {}).get("duration") != "observed")
        return {"status": status, "basis": quality["basis"], "fields": {"tool_duration": {
            "status": status, "observed": observed, "partial": partial, "total": len(items),
            "reasons": dict(sorted(reasons.items())),
        }}}
    fields = {field: quality["fields"][field] for field in dependencies}
    statuses = {field["status"] for field in fields.values()}
    status = ("unavailable" if statuses & {"unavailable", "unknown"}
              else "partial" if "partial" in statuses else "observed")
    if view_id == "bottlenecks" or (
        status == "partial" and view_id not in {"tool_outcomes", "token_usage", "agent_tokens", "step_durations"}
    ):
        status = "unavailable"
    return {"status": status, "basis": quality["basis"], "fields": fields}
