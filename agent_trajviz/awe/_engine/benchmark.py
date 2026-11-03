"""Explicit benchmark semantics shared by ingestion, diagnosis and reports."""

from __future__ import annotations
import json
from contextvars import ContextVar
from contextlib import contextmanager
import os
from pathlib import Path

_ACTIVE = ContextVar("awe_benchmark_profile", default=None)


@contextmanager
def using_profile(profile):
    token = _ACTIVE.set(profile)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


DEFAULTS = {
    "name": "unspecified",
    "gating_tests_visible": True,
    "test_file_include": [],
    "test_file_exclude": [],
    "evaluator_test_reset": "unknown",
    "resolution_policy": "additional_conditions",
}


def resolve_profile(requirement=None):
    if requirement is None and _ACTIVE.get() is not None:
        return dict(_ACTIVE.get())
    profile = dict(DEFAULTS)
    path = os.environ.get("AWE_BENCHMARK_PROFILE")
    if path:
        configured = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(configured, dict):
            raise ValueError("benchmark profile must be an object")
        profile.update(configured)
    requirement = requirement or {}
    embedded = requirement.get("benchmark_profile", {})
    if not isinstance(embedded, dict):
        raise ValueError("benchmark_profile must be an object")
    profile.update(embedded)
    if "gating_tests_visible" in requirement:
        profile["gating_tests_visible"] = requirement["gating_tests_visible"]
    unknown = set(profile) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown benchmark profile keys: " + ", ".join(sorted(unknown)))
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        raise ValueError("profile name must be nonempty")
    if type(profile["gating_tests_visible"]) is not bool:
        raise ValueError("gating_tests_visible must be a boolean")
    if profile["evaluator_test_reset"] not in ("yes", "no", "unknown"):
        raise ValueError("evaluator_test_reset must be yes/no/unknown")
    if profile["resolution_policy"] not in ("all_tests_pass", "additional_conditions"):
        raise ValueError("unsupported resolution_policy")
    for key in ("test_file_include", "test_file_exclude"):
        values = profile[key]
        if not isinstance(values, list) or any(
            not isinstance(v, str) or not v.strip() or v.startswith("/") or "\\" in v for v in values
        ):
            raise ValueError(key + " must contain relative path globs")
        profile[key] = list(values)
    return profile
