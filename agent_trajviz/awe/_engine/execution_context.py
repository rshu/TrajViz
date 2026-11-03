"""Declared execution metadata; never infer hook availability from a trajectory."""

from __future__ import annotations
import json
import os
from pathlib import Path

from . import config

FIELDS = {"model", "provider", "scaffold_revision", "languages", "capabilities"}


def _context(value, allowed=FIELDS):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown or invalid execution-context fields")
    result = dict(value)
    for key, item in result.items():
        if key in ("languages", "capabilities"):
            if (
                not isinstance(item, list)
                or any(not isinstance(s, str) or not s.strip() for s in item)
                or len(item) != len(set(item))
            ):
                raise ValueError("Execution-context lists require distinct nonempty strings")
            result[key] = sorted(item)
        elif not isinstance(item, str) or not item.strip():
            raise ValueError("Execution-context identities require nonempty strings")
    return result


def load_manifest():
    path = os.environ.get("AWE_EXECUTION_CONTEXT")
    if not path:
        return {"schema_version": 1, "agents": {}, "instances": {}}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(doc, dict)
        or type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
        or set(doc) - {"schema_version", "agents", "instances"}
    ):
        raise ValueError("Invalid execution-context manifest")
    result = {"schema_version": 1}
    for section in ("agents", "instances"):
        mapping = doc.get(section, {})
        if not isinstance(mapping, dict):
            raise ValueError("Execution-context sections must be objects")
        result[section] = {}
        for identity, context in mapping.items():
            config.safe_id(identity)
            result[section][identity] = _context(context, FIELDS if section == "agents" else {"languages"})
    return result


def for_case(agent, instance_id, profile=None, manifest=None):
    manifest = load_manifest() if manifest is None else manifest
    declared = {**manifest.get("agents", {}).get(agent, {}), **manifest.get("instances", {}).get(instance_id, {})}
    profile = profile or {}
    name = profile.get("name")
    benchmark = name if name and name != "unspecified" else config.BENCHMARK_NAME
    context = {
        "benchmark": benchmark,
        "languages": [],
        "model": None,
        "provider": None,
        "scaffold_revision": None,
        "capabilities": None,
        **declared,
        "provenance": "declared; runtime capabilities and identities require adapter checks",
    }
    return {
        "benchmark": benchmark,
        "language": context["languages"][0] if len(context["languages"]) == 1 else None,
        "execution_context": context,
    }
