"""Milestone extraction, segmentation, and per-segment comparison."""

from __future__ import annotations

import re

from agent_trajviz.insight.diagnostics import _without_heredoc_bodies
from agent_trajviz.insight.patterns import _VALIDATION_COMMAND_PATTERNS
from agent_trajviz.insight.shell_cmd import (
    _cmd_script_tokens,
    _is_shell_assignment,
    _is_windows_timeout_invocation,
    _powershell_payload,
    _PS_INTERPRETERS,
    _PYTHON_INTERPRETER_RE,
    _PYTHON_VALUE_OPTIONS,
    _shell_name,
    _shell_segments,
    _skip_wrapper_options,
)
from agent_trajviz.insight.telemetry_quality import tool_field_observed
from agent_trajviz.insight.tool_failure import tool_call_failed
from agent_trajviz.tool_vocab import BASH_TOOL_NAMES

from .canonical import CanonicalAction, compute_action_cost, DEFAULT_TOKEN_RATE


# ---------------------------------------------------------------------------
# Default validation patterns
# ---------------------------------------------------------------------------

DEFAULT_VALIDATION_PATTERNS = {"test", "build", "lint", "check", "verify"}

_PROCESS_SESSION = re.compile(r"(?m)^Process running with session ID ([^\s]+)\s*$")
_PROCESS_EXIT = re.compile(r"(?m)^Process exited with code (-?\d+)\s*$")
_TAIL_OUTPUT_FILTER = re.compile(r"\s*tail(?:\s+(?:-[1-9][0-9]*|-n\s*-?[1-9][0-9]*))?\s*")
_STDERR_TO_STDOUT = re.compile(r"\s+2>&1\s*$")
_PYTEST_RUNNERS = {"pytest", "py.test"}
_RUNNER_VALUE_OPTIONS = {
    "--with", "--with-editable", "--project", "--directory", "--python",
    "--group", "--extra", "--package", "--env-file", "--index", "--index-url",
}


def _runs_validation(
    command: str, patterns: set[str], nesting: int = 0, *, pytest_tail: bool = False,
) -> bool:
    """Recognize runner argv, never words in script bodies or file arguments."""
    if nesting > 3:
        return False
    command = _without_heredoc_bodies(command).strip()
    # A shell's successful exit only grounds its final sequential command or
    # an && chain. Pipes, background jobs, and || branches can hide a failed
    # validation. A final pytest-to-tail pipeline is eligible only when the
    # caller separately checks the captured pytest summary.
    quote = ""
    tail = 0
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in {"'", '"'}:
            quote = char
        elif char in {"|", "&"}:
            if char == "&" and ((index and command[index - 1] in "<>") or command[index + 1:index + 2] == ">"):
                index += 1  # A descriptor redirect (2>&1 / &>file), not a background job.
                continue
            if command.startswith("&&", index):
                index += 2
                continue
            if char == "|" and pytest_tail and _TAIL_OUTPUT_FILTER.fullmatch(command[index + 1:]):
                producer = _STDERR_TO_STDOUT.sub("", command[tail:index])
                segments = _shell_segments(producer)
                return (bool(segments) and _validation_segment(segments[-1], _PYTEST_RUNNERS, nesting)
                        and _validation_segment(segments[-1], patterns, nesting))
            return False
        elif char in {";", "\n"}:
            tail = index + 1
        index += 1
    command = command[tail:]
    return not pytest_tail and any(_validation_segment(tokens, patterns, nesting)
                                   for tokens in _shell_segments(command))


def _validation_segment(tokens: list[str], patterns: set[str], nesting: int) -> bool:
    index = 0
    for _ in range(12):
        while index < len(tokens) and _is_shell_assignment(tokens[index]):
            index += 1
        if index >= len(tokens):
            return False
        name = _shell_name(tokens[index])
        if name == "command":
            index += 1
            if index < len(tokens) and tokens[index] in {"-v", "-V"}:
                return False
            index = _skip_wrapper_options(tokens, index, name)
            continue
        if name in {"env", "sudo", "time", "nice", "nohup"}:
            index = _skip_wrapper_options(tokens, index + 1, name)
            continue
        if name == "timeout":
            if _is_windows_timeout_invocation(tokens, index):
                return False
            index = _skip_wrapper_options(tokens, index + 1, name) + 1
            continue
        if name == "!":
            return False  # Shell success here means the validation failed.
        if name in {"if", "then", "elif", "while", "until", "do"}:
            index += 1
            continue
        if name in {"bash", "dash", "ksh", "sh", "zsh"}:
            for i in range(index + 1, len(tokens) - 1):
                if not tokens[i].startswith("-"):
                    break
                if "c" in tokens[i].lstrip("-"):
                    return _runs_validation(tokens[i + 1], patterns, nesting + 1)
            return False
        if name in {"cmd", "cmd.exe"}:
            rest = _cmd_script_tokens(tokens, index)
            return (_runs_validation(rest[0], patterns, nesting + 1) if len(rest) == 1
                    else bool(rest) and nesting < 3 and _validation_segment(rest, patterns, nesting + 1))
        if name in _PS_INTERPRETERS:
            payload = _powershell_payload(tokens, index, nesting)
            if payload is None or payload[0] != "command":
                return False
            body = payload[1]
            return (_runs_validation(body, patterns, nesting + 1) if isinstance(body, str)
                    else nesting < 3 and _validation_segment(body, patterns, nesting + 1))
        if name in {"uv", "poetry", "pipenv", "pdm", "hatch"}:
            if index + 1 >= len(tokens) or tokens[index + 1] != "run":
                return False
            index += 2
            while index < len(tokens) and tokens[index].startswith("-"):
                option = tokens[index]
                index += 1
                if option == "--":
                    break
                if option in _RUNNER_VALUE_OPTIONS:
                    index += 1
            continue
        if _PYTHON_INTERPRETER_RE.fullmatch(name):
            index += 1
            while index < len(tokens):
                option = tokens[index]
                if option == "-m" and index + 1 < len(tokens):
                    index += 1
                    break
                if option in {"-c", "-", "--"} or not option.startswith("-"):
                    return False
                index += 2 if option in _PYTHON_VALUE_OPTIONS else 1
            else:
                return False
            name = tokens[index].lower()
        args = tokens[index + 1:]
        if name == "git":
            args = tokens[_skip_wrapper_options(tokens, index + 1, name):]
        # pytest's -v means verbose, unlike its uppercase -V version flag.
        if any(token in {"--version", "-V", "--help", "-h", "--collect-only", "--co", "--fixtures"}
               for token in args):
            return False
        argv = [name, *[token.lower() for token in args]]
        options = {arg.split("=", 1)[0] for arg in argv[1:]}
        no_run_options = {
            "pytest": {"--markers", "--setup-plan", "--setup-only", "--fixtures-per-test", "--cache-show"},
            "make": {"-n", "--just-print", "--dry-run", "--recon", "-q", "--question"},
            "ruff": {"--show-settings", "--show-files"},
            "tox": {"-l", "--listenvs", "--listenvs-all", "--showconfig"},
            "nox": {"-l", "--list", "--list-sessions"},
            "ctest": {"-n", "--show-only"},
            "jest": {"--listtests", "--showconfig"},
            "eslint": {"--print-config"},
        }
        if options & no_run_options.get(name, set()):
            return False
        if name == "make" and any(re.fullmatch(r"-[^-]*[nq][^-]*", arg) for arg in args):
            return False
        if name == "vitest" and len(argv) > 1 and argv[1] == "list":
            return False
        if name == "ruff":
            subcommand = argv[1] if len(argv) > 1 else ""
            if subcommand != "check" and not (subcommand == "format" and "--check" in options):
                return False
        candidates = _VALIDATION_COMMAND_PATTERNS if patterns == DEFAULT_VALIDATION_PATTERNS else patterns
        return any(argv[:len(parts)] == parts for pattern in candidates
                   if (parts := pattern.lower().split()))
    return False


def _successful_completion(tool: dict, *, process_poll: bool = False) -> bool | None:
    """None means the process is still running or its outcome was not observed."""
    if not tool_field_observed(tool, "tool_outcomes"):
        return None
    if tool_call_failed(tool):
        return False
    metadata = tool.get("metadata") or {}
    for key in ("exit", "exit_code"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value == 0
    output = tool.get("output")
    if isinstance(output, str):
        exits = _PROCESS_EXIT.findall(output)
        if exits:
            return int(exits[-1]) == 0
        if _PROCESS_SESSION.search(output):
            return None
    if process_poll:
        # A successful poll request says nothing about process completion.
        return None
    return True if str(tool.get("status") or "").lower() in {"success", "completed", "complete", "succeeded", "ok"} else None


def _has_passing_pytest_summary(tool: dict) -> bool:
    """Require one complete terminal summary with an executed, passing test."""
    from agent_trajviz.awe._engine.verification import test_summary

    output = tool.get("output")
    if (not isinstance(output, str) or not output.strip()
            or (tool.get("metadata") or {}).get("truncated") is True
            or not tool_field_observed(tool, "tool_outputs")):
        return False
    summary = test_summary(output)
    if summary.get("runner") != "pytest":
        return False
    if test_summary(output.strip().rsplit("\n", 1)[-1]) != summary:
        return False
    counts = summary["counts"]
    return counts["passed"] > 0 and counts["failed"] == counts["errors"] == 0


def _first_passing_validation(steps: list[dict], patterns: set[str]) -> int | None:
    pending: set[tuple[str, str]] = set()
    for step in steps:
        owner = str(step.get("session_id") or "")
        for tool in step.get("tool_calls", []):
            args = tool.get("input")
            if not isinstance(args, dict):
                continue
            command = args.get("command")
            if tool.get("tool_name") in BASH_TOOL_NAMES and isinstance(command, str):
                if not _runs_validation(command, patterns):
                    if ("|" in command and _runs_validation(command, patterns, pytest_tail=True)
                            and _successful_completion(tool) is True and _has_passing_pytest_summary(tool)):
                        return step["index"]
                    continue
                completed = _successful_completion(tool)
                if completed is True:
                    return step["index"]
                output = tool.get("output")
                session = _PROCESS_SESSION.search(output) if isinstance(output, str) else None
                if completed is None and session:
                    pending.add((owner, session[1]))
            elif tool.get("tool_name") == "write_stdin":
                key = (owner, str(args.get("session_id") or ""))
                if key not in pending:
                    continue
                completed = _successful_completion(tool, process_poll=True)
                if completed is True:
                    return step["index"]
                if completed is False:
                    pending.discard(key)
    return None


# ---------------------------------------------------------------------------
# Milestone identity
# ---------------------------------------------------------------------------

# Defined here, next to extract_milestones, as the single source of truth:
# charts.py and rendering.py each used to carry their own copy. The list also
# fixes the display ORDER of the milestone timeline, which is why it is a list
# and not just the label map's keys.
MILESTONE_NAMES = [
    "first_relevant_file",
    "first_edit",
    "first_surviving_edit",
    "first_passing_validation",
    "final_patch",
]

MILESTONE_LABELS = {
    "first_relevant_file": "First Relevant File",
    "first_edit": "First Edit",
    "first_surviving_edit": "First Surviving Edit",
    "first_passing_validation": "First Passing Validation",
    "final_patch": "Last Recognized Edit",
}


# ---------------------------------------------------------------------------
# Milestone extraction
# ---------------------------------------------------------------------------

def extract_milestones(
    actions: list[CanonicalAction],
    validation_patterns: set[str] | None = None,
    target_files: set[str] | None = None,
    *,
    steps: list[dict] | None = None,
) -> dict[str, int | None]:
    """Extract first-occurrence milestones from a canonical action sequence.

    Args:
        target_files: Set of normalized file paths from the patch footprint or anchor.
            When provided, first_relevant_file and first_edit are grounded to these files.
        steps: Original parsed steps, when available. Grounds passing validation
            to an executed runner and observed successful completion, including
            asynchronous process polling. Actions-only callers retain legacy behavior.

    Returns dict with milestone name → step_index or None.
    """
    if validation_patterns is None:
        validation_patterns = DEFAULT_VALIDATION_PATTERNS

    import os
    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    norm_targets = {_norm(f) for f in target_files} if target_files else None

    def _is_target(target: str) -> bool:
        """Normalize-and-suffix-match against the target set (hoisted
        from three verbatim copies below; behavior unchanged)."""
        nt = _norm(target)
        return any(
            nt == t or nt.endswith("/" + t) or t.endswith("/" + nt)
            for t in norm_targets
        )

    milestones: dict[str, int | None] = {
        "first_relevant_file": None,
        "first_edit": None,
        "first_surviving_edit": None,
        "first_passing_validation": _first_passing_validation(steps, validation_patterns) if steps is not None else None,
        "final_patch": None,
    }

    last_successful_write_step = None

    for a in actions:
        if a.action_type == "REASON":
            continue

        # first_relevant_file: earliest FILE_READ or SEARCH referencing a target file
        if milestones["first_relevant_file"] is None:
            if a.action_type in ("FILE_READ", "SEARCH"):
                if norm_targets is not None:
                    if _is_target(a.target):
                        milestones["first_relevant_file"] = a.step_index
                elif a.effect_label in ("justified", "survived"):
                    # Fallback: use effect label when no target_files provided
                    milestones["first_relevant_file"] = a.step_index

        # first_edit: earliest FILE_WRITE to a target file (or any file if no targets)
        if milestones["first_edit"] is None and a.action_type == "FILE_WRITE":
            if norm_targets is None or _is_target(a.target):
                milestones["first_edit"] = a.step_index

        # first_surviving_edit: earliest FILE_WRITE with effect_label=survived
        # Grounded to target files when available
        if milestones["first_surviving_edit"] is None:
            if a.action_type == "FILE_WRITE" and a.effect_label == "survived":
                if norm_targets is None or _is_target(a.target):
                    milestones["first_surviving_edit"] = a.step_index

        # first_passing_validation: earliest recognized validation COMMAND that
        # completed. assign_effect_labels marks validation commands as
        # "justified" (reason=validation_command), NOT "survived", so match on
        # that; fall back to a pattern match for commands labeled "survived".
        if steps is None and milestones["first_passing_validation"] is None and a.action_type == "COMMAND":
            reason = a.effect_detail.get("reason") if isinstance(a.effect_detail, dict) else None
            base_cmd = a.target.split("/")[-1] if "/" in a.target else a.target
            if reason == "validation_command" or (
                a.effect_label in ("justified", "survived")
                and any(vp in base_cmd for vp in validation_patterns)
            ):
                milestones["first_passing_validation"] = a.step_index

        # The public final_patch key represents the last recognized successful write.
        if a.action_type == "FILE_WRITE" and a.effect_label == "survived":
            last_successful_write_step = a.step_index

    milestones["final_patch"] = last_successful_write_step
    return milestones


# ---------------------------------------------------------------------------
# Milestone deltas
# ---------------------------------------------------------------------------

def compute_milestone_deltas(
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
) -> dict[str, int | None]:
    """Compute step-index deltas for milestones present in both trajectories."""
    deltas: dict[str, int | None] = {}
    for name in ref_milestones:
        ref_val = ref_milestones.get(name)
        cmp_val = cmp_milestones.get(name)
        if ref_val is not None and cmp_val is not None:
            deltas[f"{name}_delta"] = cmp_val - ref_val
        else:
            deltas[f"{name}_delta"] = None
    return deltas


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _milestone_order(milestones: dict[str, int | None]) -> list[str]:
    """Return milestone names in order of occurrence, excluding None."""
    present = [(name, step) for name, step in milestones.items() if step is not None]
    present.sort(key=lambda x: x[1])
    return [name for name, _ in present]


def segment_by_milestones(
    actions: list[CanonicalAction],
    milestones: dict[str, int | None],
) -> list[dict]:
    """Partition actions into segments between consecutive milestones.

    Returns list of {label, actions, start_step, end_step}.
    """
    order = _milestone_order(milestones)
    if not order:
        non_reason = [a for a in actions if a.action_type != "REASON"]
        return [{
            "label": "start → end",
            "actions": non_reason,
            "start_step": non_reason[0].step_index if non_reason else 0,
            "end_step": non_reason[-1].step_index if non_reason else 0,
        }]

    non_reason = [a for a in actions if a.action_type != "REASON"]
    boundaries = [("start", 0)]
    for name in order:
        boundaries.append((name, milestones[name]))
    boundaries.append(("end", non_reason[-1].step_index + 1 if non_reason else 0))

    segments = []
    for i in range(len(boundaries) - 1):
        label = f"{boundaries[i][0]} → {boundaries[i+1][0]}"
        start = boundaries[i][1]
        end = boundaries[i + 1][1]
        seg_actions = [a for a in non_reason if start <= a.step_index < end]
        segments.append({
            "label": label,
            "actions": seg_actions,
            "start_step": start,
            "end_step": end,
        })

    return segments


# ---------------------------------------------------------------------------
# Segment comparison
# ---------------------------------------------------------------------------

def _unique_step_count(actions: list[CanonicalAction]) -> int:
    """Count unique step indices (original trajectory steps, not action count)."""
    return len({a.step_index for a in actions})


def compare_segments(
    ref_segments: list[dict],
    cmp_segments: list[dict],
    ref_milestones: dict[str, int | None],
    cmp_milestones: dict[str, int | None],
    ref_actions: list[CanonicalAction],
    cmp_actions: list[CanonicalAction],
    token_rate: float = DEFAULT_TOKEN_RATE,
) -> dict:
    """Compare segments. Paired when milestone order matches, per-trajectory when not.

    Paired segments use real within-segment alignment to compute recall and precision.
    """
    from .alignment import align_trajectories
    from .alignment import compute_alignment_metrics

    ref_order = _milestone_order(ref_milestones)
    cmp_order = _milestone_order(cmp_milestones)

    if ref_order == cmp_order:
        # Paired comparison with real within-segment alignment
        paired = []
        for ref_seg, cmp_seg in zip(ref_segments, cmp_segments, strict=False):
            ref_seg_actions = ref_seg["actions"]
            cmp_seg_actions = cmp_seg["actions"]

            if ref_seg_actions and cmp_seg_actions:
                seg_alignment = align_trajectories(ref_seg_actions, cmp_seg_actions)
                seg_metrics = compute_alignment_metrics(
                    seg_alignment, ref_seg_actions, cmp_seg_actions, token_rate)
                recall = seg_metrics["reference_recall"]
                precision = seg_metrics["behavioral_precision"]
                overhead = seg_metrics["overhead_ratio"]
            else:
                ref_cost = sum(compute_action_cost(a, token_rate) for a in ref_seg_actions)
                cmp_cost = sum(compute_action_cost(a, token_rate) for a in cmp_seg_actions)
                recall = 1.0 if not ref_seg_actions else 0.0
                precision = 1.0 if not cmp_seg_actions else 0.0
                overhead = cmp_cost / ref_cost if ref_cost > 0 else 0.0

            paired.append({
                "segment": ref_seg["label"],
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "overhead": round(overhead, 2),
            })
        return {
            "milestone_order_matches": True,
            "segment_comparison": paired,
        }
    else:
        # Per-trajectory segments — report step counts and tokens
        ref_segs = []
        for seg in ref_segments:
            ref_segs.append({
                "segment": seg["label"],
                "steps": _unique_step_count(seg["actions"]),
                "tokens": sum(a.cost.token_share for a in seg["actions"]),
            })
        cmp_segs = []
        for seg in cmp_segments:
            cmp_segs.append({
                "segment": seg["label"],
                "steps": _unique_step_count(seg["actions"]),
                "tokens": sum(a.cost.token_share for a in seg["actions"]),
            })
        return {
            "milestone_order_matches": False,
            "reference_segments": ref_segs,
            "compared_segments": cmp_segs,
        }
