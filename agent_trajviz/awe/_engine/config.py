"""Locations of the ARGUS data inputs we read (no ARGUS code dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from .benchmark import resolve_profile

# Override with AWE_ARGUS_ROOT if the ARGUS checkout moves.
ARGUS_ROOT = Path(os.environ.get("AWE_ARGUS_ROOT", os.environ.get("TRAJVIZ_REFERENCE_ROOT", ".")))
DATA = ARGUS_ROOT / "data"

REQUIREMENTS_DIR = DATA / "requirements"  # <instance_id>.json  (gold patch + tests)
PATCH_DIR = DATA / "patch"  # <agent>/<instance_id>.diff (agent edit-set)
TRAJECTORY_DIR = DATA / "trajectory"  # <agent>/<instance_id>.{json,jsonl}
LABELS_DIR = DATA / "labels"  # <agent>/<instance_id>_labeled.json (LLM phase labels)

_DEFAULT_AGENTS = ["claude_code", "codex", "opencode_glm", "opencode_gpt", "opencode_opus"]
# A comma-separated override keeps benchmark ingestion out of the core.  This is
# especially useful for Harbor exports, where the run-group name is the agent
# identity and is not known when this package is installed.
AGENTS = (
    [a.strip() for a in os.environ["AWE_AGENTS"].split(",") if a.strip()]
    if os.environ.get("AWE_AGENTS")
    else _DEFAULT_AGENTS
)

# This project's own data tree (outputs + caches), distinct from the ARGUS inputs.
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_DATA = Path(
    os.environ.get(
        "AWE_PROJECT_DATA", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "trajviz" / "evaluation"
    )
)
# Outputs may be isolated without losing input caches, overrides or per-test data.
OUTPUT_DATA = Path(os.environ.get("AWE_OUTPUT_DATA", PROJECT_DATA))
# An external dataset must not silently inherit the bundled corpus's label.
# An explicit display label wins; otherwise use the configured profile name.
BENCHMARK_NAME = os.environ.get("AWE_BENCHMARK_NAME") or (
    resolve_profile()["name"]
    if os.environ.get("AWE_BENCHMARK_PROFILE")
    else "unspecified"
    if os.environ.get("AWE_ARGUS_ROOT")
    else "SWE-bench Verified"
)
JUDGE_CACHE_DIR = PROJECT_DATA / "cache" / "judge"  # <model>/<agent>/<instance>.json

# --- LLM judge (compatible API; legacy OpenRouter configuration names) ---------------------------------------------------
# Powers the two primarily judgment-based capabilities (Requirement Understanding,
# Task Planning); requirement understanding also has a limited lexical anchor. Entirely OPTIONAL: with no API key
# the objective 5-capability slice runs unchanged, and cached verdicts are read
# back WITHOUT a key (so reports stay offline/reproducible after one judge run).
# The configured endpoint speaks a compatible chat API; AWE_JUDGE_MODEL is its model ID,
# e.g. openai/gpt-5, google/gemini-2.5-pro,
# anthropic/claude-sonnet-4.5. Pick a NON-Claude judge when scoring claude_code
# to avoid self-preference bias — see LLM_JUDGE.md.
OPENROUTER_BASE_URL = os.environ.get("AWE_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = os.environ.get("AWE_JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
JUDGE_TEMPERATURE = float(os.environ.get("AWE_JUDGE_TEMPERATURE", "0"))
# Headroom for REASONING models (deepseek-v4/r1, gpt-5, etc.): the reasoning trace
# is generated before the JSON and counts against this cap, so a low value
# truncates the verdict. The verdict itself is ~200 tokens; this is just a ceiling,
# so non-reasoning models pay nothing extra. judge_one() also retries on truncation.
JUDGE_MAX_TOKENS = int(os.environ.get("AWE_JUDGE_MAX_TOKENS", "2400"))
# Panel abstention bar (Raj et al. arXiv:2607.28802 §6: selective voting trades
# coverage for precision — 0.96 category precision at 68% coverage under
# unanimity). A winning error label whose panel agreement is BELOW this fraction
# is recorded but demoted to low confidence (blame-ineligible) and marked
# `abstained`. 0 (default) preserves the v0.32 strict-majority behavior; 1.0
# demands unanimity.
JUDGE_MIN_AGREE = float(os.environ.get("AWE_JUDGE_MIN_AGREE", "0"))


def model_slug(model: str | None = None) -> str:
    """Filesystem-safe form of a model id, for cache partitioning.
    COLLISION-RESISTANT for realistic OpenRouter ids, NOT fully injective:
    '/'->'__' and ':'->'___' separate `org/model:free` from `org/model_free`,
    but ids with literal underscores ADJACENT to a separator can still collide
    (e.g. `a/_b` vs `a:b`). Full escaping would fix that at the cost of
    invalidating every existing cache directory; revisit only if such ids
    appear. Slugs without ':' are unchanged from the previous scheme."""
    return (model or JUDGE_MODEL).replace("/", "__").replace(":", "___")


def safe_id(value: str) -> str:
    """A single unambiguous filesystem component used for artifact joins."""
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or value in (".", "..")
        or any(c in value for c in "/\\:")
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"Unsafe artifact ID: {value!r}")
    return value


def safe_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or any(c in value for c in "\\:")
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"Unsafe diff path: {value!r}")
    return value


def eval_path(agent: str) -> Path:
    return DATA / f"eval_{safe_id(agent)}.json"


def requirements_path(instance_id: str) -> Path:
    return REQUIREMENTS_DIR / f"{safe_id(instance_id)}.json"


def patch_path(agent: str, instance_id: str) -> Path:
    return PATCH_DIR / safe_id(agent) / f"{safe_id(instance_id)}.diff"


def list_instances() -> list[str]:
    return sorted(p.stem for p in REQUIREMENTS_DIR.glob("*.json"))


def require_instances(instances):
    """Fail loudly on an empty task list — a misconfigured AWE_ARGUS_ROOT must not
    silently produce a perfect-looking scorecard over 0 tasks."""
    items = list(instances or [])
    if not items:
        raise ValueError(
            f"No instances under {DATA}/requirements/ — is AWE_ARGUS_ROOT set "
            f"correctly? (currently {ARGUS_ROOT}). Found 0 requirement files."
        )
    return items
