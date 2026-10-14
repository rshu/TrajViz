"""Label every step of a trajectory with a phase and an action (``agent-trajviz-label``).

Assistant steps are classified by the model against the taxonomy; user steps
get the deterministic ``user/user_prompt`` label, and any other role gets
``unknown/unknown`` rather than being dropped. The output is a sidecar JSON with
one record per parsed step, keyed by ``index`` (the parser's index) and
``raw_index`` (the position in the source file), so every label maps back to
the step it describes. ``--assistant-only`` keeps just the assistant records.

Usage:
    agent-trajviz-label trajectory.json
    agent-trajviz-label trajectory.json -o labels.json --assistant-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from .. import _fileio
from . import llm
from .taxonomy import (
    DEFAULT_TAXONOMY,
    UNKNOWN_LABEL,
    build_step_message,
    build_system_prompt,
    build_valid_sets,
    load_taxonomy,
    parse_label_response,
)

logger = logging.getLogger(__name__)

USER_DEFAULT_LABEL = {"phase": "user", "action": "user_prompt"}

#: Identifier of the sidecar layout. Existing sidecars carry it and the
#: dashboard reads it, so it is part of the data format, not a code version.
SIDECAR_SCHEMA = "trajectory_labels.v2"


class OutputSafetyError(ValueError):
    """Raised when the label output would replace the source trajectory."""


def _ensure_output_is_not_input(trajectory_path: str | Path, output_path: str | Path) -> None:
    _fileio.ensure_output_does_not_overwrite(
        output_path,
        [trajectory_path],
        exc=OutputSafetyError,
        message="Output path would overwrite input trajectory: {source}",
    )


def load_all_steps(trajectory_path: str) -> list[dict[str, Any]]:
    """Load and normalize every step of a trajectory, preserving parser indices."""
    from ..insight.loaders import load_trajectory
    from ..insight.parser import parse_steps

    raw = load_trajectory(trajectory_path)
    if "_error" in raw:
        raise ValueError(f"Failed to load trajectory: {raw['_error']}")
    return parse_steps(raw)


def _is_empty_assistant_step(step: dict[str, Any]) -> bool:
    has_text = bool((step.get("text_preview", "") or "").strip())
    has_tools = bool(step.get("tool_calls"))
    has_reasoning = any(
        isinstance(part, dict) and part.get("type") == "reasoning" and part.get("text")
        for part in step.get("parts", [])
    )
    return not (has_text or has_tools or has_reasoning)


def _serialize_step(step: dict[str, Any], *, fallback_index: int, label: dict[str, str],
                    label_source: str) -> dict[str, Any]:
    """One sidecar record; the source index is carried over unchanged."""
    step_index = step.get("index", fallback_index)
    tokens = step.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}
    return {
        "index": step_index,
        "raw_index": step.get("raw_index", step_index),
        "role": str(step.get("role", "unknown") or "unknown").lower(),
        "phase": label["phase"],
        "action": label["action"],
        "label_source": label_source,
        "time_created_ms": step.get("time_created_ms"),
        "time_completed_ms": step.get("time_completed_ms"),
        "duration_s": step.get("duration"),
        "tokens_total": tokens.get("total", 0),
        "tool_calls": [tc.get("tool_name", "?") for tc in step.get("tool_calls", []) if isinstance(tc, dict)],
        "finish": step.get("finish", ""),
        "agent": step.get("agent", "") or step.get("agent_id", ""),
        "model_id": step.get("model_id", ""),
        "text_preview": (step.get("text_preview", "") or "")[:200],
        "round": step.get("round"),
        "is_sub_agent": step.get("is_sub_agent", False),
        "session_id": step.get("session_id", ""),
        "executor_id": step.get("agent", "") or step.get("agent_id", ""),
    }


def _classify(step: dict[str, Any], *, system_prompt: str, valid: tuple[set[str], set[str], dict[str, str]],
              base_url: str, api_key: str, model: str, provider: str, temperature: float | None,
              max_tokens: int | None, max_content_chars: int) -> dict[str, str] | None:
    """Ask the model for one assistant step's label; retry once, then give up."""
    user_message = build_step_message(step, max_chars=max_content_chars)
    for attempt in range(2):
        try:
            response = llm.call_llm(
                base_url=base_url, api_key=api_key, model=model, system_prompt=system_prompt,
                user_message=user_message, provider=provider, temperature=temperature, max_tokens=max_tokens,
            )
            return parse_label_response(response, *valid)
        except Exception as exc:  # network, HTTP or response-shape errors from the provider
            # HTTP errors can echo endpoint credentials; malformed provider
            # responses can echo prompt text. Log the error class, not its body.
            error_type = type(exc).__name__
            if attempt == 0:
                logger.warning("labeling index %s failed (%s); retrying", step.get("index"), error_type)
                time.sleep(1)
            else:
                logger.error("labeling index %s failed again (%s); labeled unknown", step.get("index"), error_type)
    return None


def label_trajectory(
    trajectory_path: str,
    output_path: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_content_chars: int = 8000,
    delay: float = 0.0,
    taxonomy_path: str | None = None,
    user_phase: str = USER_DEFAULT_LABEL["phase"],
    user_action: str = USER_DEFAULT_LABEL["action"],
    assistant_only: bool = False,
) -> dict[str, Any]:
    """Label a trajectory and write the sidecar to *output_path*; return its content.

    Refuses (``OutputSafetyError``) to write over the trajectory itself, and
    replaces an existing sidecar only once the new one is completely written.
    """
    _ensure_output_is_not_input(trajectory_path, output_path)
    taxonomy_file = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY
    taxonomy_mapping, taxonomy_version = load_taxonomy(taxonomy_file)
    valid = build_valid_sets(taxonomy_mapping)
    system_prompt = build_system_prompt(taxonomy_file.read_text(encoding="utf-8"))

    logger.info("Loading trajectory: %s", trajectory_path)
    all_steps = load_all_steps(trajectory_path)
    roles = [str(step.get("role", "")).lower() for step in all_steps]
    assistant_count, user_count = roles.count("assistant"), roles.count("user")
    logger.info("Found %d steps (%d assistant, %d user)", len(all_steps), assistant_count, user_count)

    labeled_steps: list[dict[str, Any]] = []
    assistant_seen = assistant_unknown = default_count = 0
    for position, step in enumerate(all_steps):
        role = str(step.get("role", "unknown") or "unknown").lower()
        if role == "user":
            label, label_source = {"phase": user_phase, "action": user_action}, "default"
            default_count += 1
        elif role != "assistant":
            label, label_source = dict(UNKNOWN_LABEL), "fallback"
            default_count += 1
        else:
            assistant_seen += 1
            classified = None if _is_empty_assistant_step(step) else _classify(
                step, system_prompt=system_prompt, valid=valid, base_url=base_url, api_key=api_key,
                model=model, provider=provider, temperature=temperature, max_tokens=max_tokens,
                max_content_chars=max_content_chars,
            )
            label = classified or dict(UNKNOWN_LABEL)
            label_source = "llm" if "unknown" not in (label["phase"], label["action"]) else "fallback"
            if label_source == "fallback":
                assistant_unknown += 1
            logger.info("assistant %d/%d (index %s): %s/%s", assistant_seen, assistant_count,
                        step.get("index", position), label["phase"], label["action"])
        labeled_steps.append(_serialize_step(step, fallback_index=position, label=label, label_source=label_source))
        if delay > 0 and role == "assistant" and assistant_seen < assistant_count:
            time.sleep(delay)

    if len(labeled_steps) != len(all_steps):
        raise RuntimeError("Internal error: not every parsed index received a label")
    if assistant_only:
        labeled_steps = [s for s in labeled_steps if s.get("role") == "assistant"]
        if len(labeled_steps) != assistant_count:
            raise RuntimeError("Internal error: assistant-only output lost assistant records")

    output = {
        "schema_version": SIDECAR_SCHEMA,
        "trajectory_file": os.path.abspath(trajectory_path),
        "taxonomy_version": taxonomy_version,
        "model": model,
        "labeled_at": datetime.now(UTC).isoformat(),
        "defaults": {"user": {"phase": user_phase, "action": user_action}, "other": dict(UNKNOWN_LABEL)},
        "counts": {
            "total": len(labeled_steps),
            "assistant": assistant_count,
            "user": user_count,
            "default_or_fallback": default_count + assistant_unknown,
        },
        "steps": labeled_steps,
    }
    _fileio.write_json_atomic(output_path, output)
    logger.info("Done: %d records; %d/%d assistant steps classified. Output: %s",
                len(labeled_steps), assistant_count - assistant_unknown, assistant_count, output_path)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-trajviz-label",
        description="Label trajectory steps with phase/action tags using an LLM; one record per step.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="Path to a trajectory file of any supported format")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: <input>_labeled.json)")
    parser.add_argument("--assistant-only", action="store_true", help="Write only the assistant steps' records")
    parser.add_argument("--model", default=None, help="Model name (default: LABEL_MODEL)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: LABEL_BASE_URL)")
    parser.add_argument("--api-key", default=None, help="API key (default: LABEL_API_KEY)")
    parser.add_argument("--provider", default=None, choices=("openai", "anthropic"),
                        help="API flavour (default: LABEL_PROVIDER, else openai)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max response tokens (default: 1024)")
    parser.add_argument("--max-content-chars", type=int, default=8000, help="Max characters per step sent")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between model calls")
    parser.add_argument("--taxonomy", default=None, help="Taxonomy file (default: the bundled taxonomy.md)")
    parser.add_argument("--user-phase", default=USER_DEFAULT_LABEL["phase"], help="Phase recorded for user steps")
    parser.add_argument("--user-action", default=USER_DEFAULT_LABEL["action"], help="Action recorded for user steps")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from .._extras import require_or_exit

    require_or_exit("llm", "The step labeler", "agent-trajviz-label")
    # Configuration is read here, at the command's entry, never at import.
    llm.load_env_files()
    base_url = args.base_url or os.getenv("LABEL_BASE_URL", "")
    api_key = args.api_key or os.getenv("LABEL_API_KEY", "")
    model = args.model or os.getenv("LABEL_MODEL", "")
    provider = args.provider or os.getenv("LABEL_PROVIDER", "openai")
    temperature = args.temperature if args.temperature is not None else llm.env_float("LABEL_TEMPERATURE", 0.3)
    max_tokens = args.max_tokens if args.max_tokens is not None else llm.env_int("LABEL_MAX_TOKENS", 1024)
    for value, name, flag in ((base_url, "LABEL_BASE_URL", "--base-url"), (api_key, "LABEL_API_KEY", "--api-key"),
                              (model, "LABEL_MODEL", "--model")):
        if not value:
            parser.error(f"{name} not set (use {flag} or .env)")

    output_path = args.output or str(Path(args.input).with_name(f"{Path(args.input).stem}_labeled.json"))

    from .._cli import configure_logging

    configure_logging(logging.INFO, "%(message)s")
    try:
        label_trajectory(
            trajectory_path=args.input, output_path=output_path, base_url=base_url, api_key=api_key,
            model=model, provider=provider, temperature=temperature, max_tokens=max_tokens,
            max_content_chars=args.max_content_chars, delay=args.delay, taxonomy_path=args.taxonomy,
            user_phase=args.user_phase, user_action=args.user_action, assistant_only=args.assistant_only,
        )
    except OutputSafetyError as exc:
        parser.error(str(exc))
    except (ValueError, OSError) as exc:
        print(f"agent-trajviz-label: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
