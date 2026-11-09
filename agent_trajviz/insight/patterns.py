"""Pattern detection for recurring tool sequences, workflow phases, and anti-patterns."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter

from agent_trajviz.insight.diagnostics import _extract_bash_paths
from agent_trajviz.insight.formats.common import as_text
from agent_trajviz.insight.parser import spawned_child_session_id
from agent_trajviz.insight.shell_cmd import shell_runs_search, tool_call_hint
from agent_trajviz.insight.token_explain import (
    CJK_CHARS_PER_TOKEN,
    _est_tokens,
)
from agent_trajviz.insight.tool_failure import step_has_success, tool_call_failed
from agent_trajviz.tool_vocab import (
    BASH_TOOL_NAMES,
    WRITE_PATH_KEYS,
    WRITE_TOOL_NAMES as _WRITE_TOOL_NAMES,
    write_target_path,
)


# ---------------------------------------------------------------------------
# Expected phase ordering for anomaly detection
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["understand", "plan", "implement", "debug", "validate", "report"]
_PHASE_RANK: dict[str, int] = {name: i for i, name in enumerate(_PHASE_ORDER)}

# Cap to avoid pathological inputs
_MAX_STEPS = 2000

# Lowercased, and compared only through _is_plan_tool. This vocabulary used to
# be written twice — an exact-spelling set here for the phase classifier and a
# separate lowercase tuple inside extract_plan_history — so the two disagreed in
# both directions: "TodoUpdate" set the plan phase but produced no plan history,
# "todo_write" produced plan history but never the plan phase. One set, one
# comparison, so a new spelling only has to be added here.
_PLAN_TOOL_NAMES = frozenset({
    "todowrite", "todo_write", "todoupdate", "taskcreate", "taskupdate",
    "tasklist", "enterplanmode",
})
_READ_TOOL_NAMES = {"Read", "read", "WebFetch"}
_SEARCH_TOOL_NAMES = {
    *BASH_TOOL_NAMES,
    "Grep", "Glob", "grep", "glob", "find", "ToolSearch", "WebSearch",
}
# Every entry must be COMMAND-SHAPED. These are matched as substrings of the
# whole whitespace-normalised command, so a bare English word matches prose:
# "lint", "check" and "verify" used to live here and fired on `git checkout`,
# `cat checklist.md`, `subprocess.check_call`, `astropy.io.fits.verify`, and —
# dominantly — on standalone words inside quoted script bodies such as
# `python -c "... # first verify the bug exists ..."`. A word-boundary match
# would not have helped: the false positive IS a standalone word. Keep the
# verb attached to its tool.
_VALIDATION_COMMAND_PATTERNS = (
    "pytest", "python -m pytest", "unittest", "tox", "nox", "go test",
    "cargo test", "npm test", "pnpm test", "yarn test", "jest", "vitest",
    "mvn test", "gradle test", "bazel test", "make test", "ctest", "ruff",
    "flake8", "pylint", "mypy", "eslint",
    "ruff check", "cargo check", "npm run lint", "pnpm lint", "yarn lint",
    "make lint", "make check", "golangci-lint", "pre-commit run",
    "tsc --noemit", "git diff --check",
    # Builds compile the code, so they validate it too. Only unambiguous
    # spellings are listed: a bare "make" is a substring of "cat Makefile".
    "npm run build", "pnpm build", "yarn build", "cargo build", "go build",
    "tsc", "cmake --build", "make -j", "ninja",
)


# ``python -m`` modules that install/bootstrap tooling — not a smoke run of
# the agent's own code.
_PYTHON_TOOLING_MODULES = frozenset({
    "pip", "pip3", "venv", "ensurepip", "virtualenv", "build", "installer",
    "pip._internal", "pip._vendor",
})
_PYTHON_VALUE_FLAGS = frozenset({
    "-W", "-X", "-Q", "--check-hash-based-pycs",
})


def _smoke_runs_command(command: str) -> bool:
    """True when a shell command executes a project entrypoint.

    Counts ``python3 app.py``, ``python -m pkg.main``, and ``./build/run``.
    Version probes, bare interpreters, ``-c`` one-liners, and tooling modules
    (``pip`` / ``venv`` / …) do not count — they never exercise the written code.
    """
    lowered = " ".join(str(command).lower().split())
    if not lowered:
        return False
    tokens = lowered.split()
    first = tokens[0]
    if first.startswith("./") or first.startswith(".\\"):
        return True
    if not first.startswith("python"):
        return False
    if "--version" in tokens or "-V" in tokens:
        return False
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-c":
            return False
        if tok == "-m":
            if i + 1 >= len(tokens):
                return False
            return tokens[i + 1] not in _PYTHON_TOOLING_MODULES
        if tok in _PYTHON_VALUE_FLAGS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return True  # positional script path
    return False  # bare ``python3``


def _is_plan_tool(tool_name: object) -> bool:
    """Return True when a tool name is a plan/TODO tool, in any export's casing."""
    return str(tool_name or "").lower() in _PLAN_TOOL_NAMES


def _has_plan_snapshot(step: dict) -> bool:
    """Return True when a step carries an explicit TODO/task snapshot."""
    for part in step.get("parts", []):
        if part.get("type") != "snapshot":
            continue
        data = part.get("data", {})
        if isinstance(data, dict) and ("todos" in data or "items" in data):
            return True
    return False


def _is_validation_command(command: str) -> bool:
    """Return True when a shell command appears to run validation."""
    lowered = " ".join(str(command).lower().split())
    if not lowered:
        return False
    return any(pattern in lowered for pattern in _VALIDATION_COMMAND_PATTERNS)


def classify_structural_phase(step: dict) -> str:
    """Classify one assistant step into a deterministic workflow phase.

    The classifier is intentionally conservative. It maps directly observable
    tool behavior to one of the six workflow phases used by the semantic
    labeler: understand, plan, implement, debug, validate, report.
    """
    if step.get("role") != "assistant":
        return "unknown"

    tool_calls = step.get("tool_calls", [])
    if _has_plan_snapshot(step):
        return "plan"

    has_plan_tool = False
    has_write = False
    has_read_or_search = False
    has_validation = False
    has_failure = False

    for tc in tool_calls:
        tool_name = tc.get("tool_name", "")
        if _is_plan_tool(tool_name):
            has_plan_tool = True
        if tool_name in _WRITE_TOOL_NAMES:
            has_write = True
        if tool_name in _READ_TOOL_NAMES or tool_name in _SEARCH_TOOL_NAMES:
            has_read_or_search = True
        if tool_call_failed(tc):
            has_failure = True

        inp = tc.get("input", {})
        if isinstance(inp, dict):
            command = inp.get("command", "")
            if command and _is_validation_command(command):
                has_validation = True

    if has_plan_tool:
        return "plan"
    if has_write:
        return "implement"
    if has_validation:
        return "validate"
    if has_failure:
        return "debug"
    if has_read_or_search:
        return "understand"
    if step.get("finish") in ("stop", "end_turn") or not tool_calls:
        return "report"
    return "unknown"


def build_structural_phase_segments(steps: list[dict]) -> list[dict]:
    """Group assistant steps into contiguous structural workflow phases."""
    segments: list[dict] = []
    current: dict | None = None

    for step in steps[:_MAX_STEPS]:
        phase = classify_structural_phase(step)
        if phase == "unknown":
            continue

        step_idx = step.get("index", 0)
        if current is None or current["name"] != phase:
            if current is not None:
                segments.append(current)
            current = {
                "name": phase,
                "start_idx": step_idx,
                "end_idx": step_idx,
            }
        else:
            current["end_idx"] = step_idx

    if current is not None:
        segments.append(current)

    return segments


# ---------------------------------------------------------------------------
# 1. Tool Sequence Detection
# ---------------------------------------------------------------------------

def _extract_tool_names(step: dict) -> list[str]:
    """Extract tool-call names from a single step."""
    names: list[str] = []
    for tc in step.get("tool_calls", []):
        name = tc.get("tool_name") or tc.get("name", "")
        if name:
            names.append(name)
    return names


def detect_tool_sequences(
    steps: list[dict],
    min_freq: int = 3,
    max_n: int = 5,
) -> list[dict]:
    """Find recurring tool-call n-grams across the trajectory.

    Parameters
    ----------
    steps : list[dict]
        Parsed trajectory steps, each with a ``tool_calls`` list.
    min_freq : int
        Minimum frequency for an n-gram to be reported.
    max_n : int
        Maximum n-gram length to consider (inclusive).

    Returns
    -------
    list[dict]
        Each entry: ``{"sequence": [str], "frequency": int, "step_indices": [int]}``,
        sorted by frequency descending.
    """
    if not steps:
        return []

    # Build flat list of (tool_name, originating_step_index) pairs
    tool_stream: list[tuple[str, int]] = []
    for step in steps[:_MAX_STEPS]:
        idx = step.get("index", 0)
        for name in _extract_tool_names(step):
            tool_stream.append((name, idx))

    if len(tool_stream) < 2:
        return []

    results: list[dict] = []

    for n in range(2, max_n + 1):
        if n > len(tool_stream):
            break

        # Count n-grams and track their start indices in the stream
        ngram_indices: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(tool_stream) - n + 1):
            gram = tuple(tool_stream[i + k][0] for k in range(n))
            ngram_indices.setdefault(gram, []).append(i)

        for gram, stream_positions in ngram_indices.items():
            # Count NON-OVERLAPPING occurrences so a single contiguous burst of
            # one tool (Read×4 -> three overlapping (Read,Read)) is not inflated
            # into a frequent sequence.
            occ_positions: list[int] = []
            last_end = -1
            for pos in sorted(stream_positions):
                if pos >= last_end:
                    occ_positions.append(pos)
                    last_end = pos + n
            if len(occ_positions) < min_freq:
                continue
            step_indices = sorted({tool_stream[pos][1] for pos in occ_positions})
            results.append({
                "sequence": list(gram),
                "frequency": len(occ_positions),
                "step_indices": step_indices,
            })

    # Sort by frequency descending, then by shorter sequences first for ties
    results.sort(key=lambda r: (-r["frequency"], len(r["sequence"])))
    return results


# ---------------------------------------------------------------------------
# 2. Failure Pattern Detection
# ---------------------------------------------------------------------------

def _step_has_success(step: dict) -> bool:
    """Return True if the step has at least one successful tool call.

    Delegates to ``tool_failure.step_has_success`` so the recovery walk shares
    one definition of "failed" with every other surface. The local copy this
    replaces was case-sensitive and consulted the exit code only when the
    status was blank, which made OpenCode's ``completed`` + non-zero
    ``metadata.exit`` read as a success here and a failure everywhere else.
    """
    return step_has_success(step)


def detect_failure_patterns(steps: list[dict]) -> list[dict]:
    """Detect recurring failure patterns and their recovery paths.

    Reuses :func:`.diagnostics.cluster_errors` to group errors, then for
    each cluster finds the most common sequence of tool calls between the
    error and the next successful step.

    Returns
    -------
    list[dict]
        Each entry: ``{"cluster_label": str, "count": int,
        "example_error": str, "recovery_path": [str] | None}``.
    """
    if not steps:
        return []

    from .diagnostics import cluster_errors

    clusters = cluster_errors(steps)
    if not clusters:
        return []

    # Build step-index lookup
    step_map: dict[int, dict] = {s.get("index", i): s for i, s in enumerate(steps)}
    all_indices = sorted(step_map.keys())

    results: list[dict] = []

    for cluster in clusters:
        # Collect recovery paths for every error occurrence in this cluster
        recovery_paths: list[tuple[str, ...]] = []

        for err_step_idx in cluster["steps"]:
            # Find steps between the error and the next success
            path_names: list[str] = []
            found_recovery = False

            # Walk forward from the step *after* the error
            pos = _bisect_right(all_indices, err_step_idx)
            for j in range(pos, min(pos + 20, len(all_indices))):
                next_idx = all_indices[j]
                next_step = step_map[next_idx]
                names = _extract_tool_names(next_step)
                path_names.extend(names)
                if _step_has_success(next_step):
                    found_recovery = True
                    break

            if found_recovery and path_names:
                recovery_paths.append(tuple(path_names))

        # Most common recovery path
        recovery_path: list[str] | None = None
        if recovery_paths:
            counter = Counter(recovery_paths)
            most_common = counter.most_common(1)[0][0]
            recovery_path = list(most_common)

        results.append({
            "cluster_label": f"{cluster['tool']}: {cluster['pattern']}",
            "count": cluster["count"],
            "example_error": cluster["pattern"],
            "recovery_path": recovery_path,
            "steps": list(cluster.get("steps", [])),
            "error_class": cluster.get("error_class", "tool"),
        })

    return results


_bisect_right = bisect_right


# ---------------------------------------------------------------------------
# 3. Phase Anomaly Detection
# ---------------------------------------------------------------------------

def detect_phase_anomalies(
    steps: list[dict],
    phases: list[dict],
) -> list[dict]:
    """Detect backward phase transitions that may indicate rework or regression.

    Parameters
    ----------
    steps : list[dict]
        Full parsed trajectory steps (used to compute total step count).
    phases : list[dict]
        Phase list with keys ``name``, ``start_idx``, ``end_idx`` (e.g.
        from manual annotation or an external classifier).

    Returns
    -------
    list[dict]
        Each entry: ``{"from_phase": str, "to_phase": str, "step_idx": int,
        "span_fraction": float, "category": str, "explanation": str}``.
        ``category`` is ``"intentional_iteration"`` or ``"unintentional_drift"``.
        ``span_fraction`` is the share of the trajectory the regressed phase
        covers — a size, deliberately NOT named ``confidence``: it shrinks as
        the regression gets shorter, which is the opposite of certainty.
    """
    if not phases or not steps:
        return []

    total_steps = len(steps[:_MAX_STEPS])
    if total_steps == 0:
        return []

    anomalies: list[dict] = []

    for i in range(1, len(phases)):
        prev_phase = phases[i - 1]
        curr_phase = phases[i]

        prev_name = prev_phase.get("name", "").lower().strip()
        curr_name = curr_phase.get("name", "").lower().strip()

        prev_rank = _PHASE_RANK.get(prev_name)
        curr_rank = _PHASE_RANK.get(curr_name)

        # Skip phases not in the expected ordering
        if prev_rank is None or curr_rank is None:
            continue

        # Backward transition
        if curr_rank < prev_rank:
            transition_step = curr_phase.get("start_idx", 0)
            regressed_steps = curr_phase.get("end_idx", transition_step) - transition_step + 1
            span_fraction = round(regressed_steps / total_steps, 4) if total_steps > 0 else 0.0

            # Categorize: check if a planning step precedes the regression
            category = "unintentional_drift"
            window_start = max(0, transition_step - 3)
            for s in steps[window_start:transition_step]:
                for tc in s.get("tool_calls", []):
                    if _is_plan_tool(tc.get("tool_name")):
                        category = "intentional_iteration"
                        break
                if category == "intentional_iteration":
                    break

            if category == "intentional_iteration":
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"A planning step precedes the transition, suggesting "
                    f"intentional iteration (spans {regressed_steps} step(s), "
                    f"{span_fraction * 100:.1f}% of trajectory)."
                )
            else:
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"The regressed phase spans {regressed_steps} step(s) "
                    f"({span_fraction * 100:.1f}% of trajectory), "
                    f"suggesting rework or unexpected context switch."
                )

            anomalies.append({
                "from_phase": prev_name,
                "to_phase": curr_name,
                "step_idx": transition_step,
                "span_fraction": span_fraction,
                "category": category,
                "explanation": explanation,
            })

    return anomalies


# ---------------------------------------------------------------------------
# 4. Plan Progress Analysis (TodoWrite tracking)
# ---------------------------------------------------------------------------

def extract_plan_history(steps: list[dict]) -> list[dict]:
    """Extract plan snapshots from TodoWrite tool calls.

    Returns a list of plan snapshots:
    ``[{"step": int, "items": [{"content": str, "status": str}]}]``
    """
    history: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name") or tc.get("name", "")
            # Same vocabulary as the phase classifier: TodoWrite (Claude Code),
            # todowrite (OpenCode), TaskCreate, EnterPlanMode, ... Plan-mode
            # tools that carry no todo list are dropped by the guard below, so a
            # wider vocabulary cannot invent snapshots.
            if not _is_plan_tool(name):
                continue
            inp = tc.get("input", tc.get("arguments", {}))
            if not isinstance(inp, dict):
                continue
            todos = inp.get("todos", [])
            if not isinstance(todos, list) or not todos:
                continue
            items = [
                {"content": as_text(t.get("content")), "status": as_text(t.get("status")) or "?"}
                for t in todos if isinstance(t, dict)
            ]
            history.append({"step": s.get("index", 0), "items": items})
    return history


def detect_plan_phases(plan_history: list[dict]) -> list[dict]:
    """Group plan snapshots into phases.

    A phase boundary is detected when the item set changes completely
    (no content overlap with the previous snapshot).
    """
    if not plan_history:
        return []

    phases: list[dict] = []
    current_contents: set[str] = set()
    phase_start = 0

    for i, snapshot in enumerate(plan_history):
        new_contents = {item["content"] for item in snapshot["items"]}
        if i > 0 and current_contents and not current_contents & new_contents:
            # Complete content change — new phase
            phases.append({
                "phase_index": len(phases),
                "start_snapshot": phase_start,
                "end_snapshot": i - 1,
                "start_step": plan_history[phase_start]["step"],
                "end_step": plan_history[i - 1]["step"],
                "item_count": len(current_contents),
            })
            phase_start = i
        current_contents = new_contents

    # Final phase
    if plan_history:
        phases.append({
            "phase_index": len(phases),
            "start_snapshot": phase_start,
            "end_snapshot": len(plan_history) - 1,
            "start_step": plan_history[phase_start]["step"],
            "end_step": plan_history[-1]["step"],
            "item_count": len(current_contents),
        })

    return phases


def compute_plan_metrics(plan_history: list[dict]) -> dict:
    """Compute plan-level metrics from plan history.

    Returns dict with: item durations, stalled items, plan reset count.
    """
    if not plan_history:
        return {"items": [], "stalled": [], "plan_resets": 0, "total_items": 0}

    # Track per-item first in_progress and completed steps
    item_first_progress: dict[str, int] = {}
    item_completed: dict[str, int] = {}

    for snapshot in plan_history:
        step = snapshot["step"]
        for item in snapshot["items"]:
            content = item["content"]
            status = item["status"]
            if status == "in_progress" and content not in item_first_progress:
                item_first_progress[content] = step
            if status == "completed" and content not in item_completed:
                item_completed[content] = step

    items = []
    stalled = []
    # Insertion-ordered, NOT a set: iterating a set follows per-process string
    # hash randomisation, which made the Plan Progress Timeline's row order and
    # the "stalled items" the Overview names differ between runs on the same
    # file. First-appearance order is also the order the agent wrote the plan,
    # which is how a reader expects to read it.
    all_contents: dict[str, None] = {}
    for snapshot in plan_history:
        for item in snapshot["items"]:
            all_contents.setdefault(item["content"], None)

    for content in all_contents:
        start = item_first_progress.get(content)
        end = item_completed.get(content)
        duration = (end - start) if start is not None and end is not None else None
        entry = {"content": content, "start_step": start, "end_step": end, "duration_steps": duration}
        items.append(entry)
        if start is not None and end is None or duration is not None and duration > 20:
            stalled.append(entry)

    phases = detect_plan_phases(plan_history)
    return {
        "items": items,
        "stalled": stalled,
        "plan_resets": max(0, len(phases) - 1),
        "total_items": len(all_contents),
    }


# ---------------------------------------------------------------------------
# 5. Sub-Agent Delegation Analysis
# ---------------------------------------------------------------------------

def _get_step_info(step: dict, trajectory: list[dict]) -> dict:
    """Retrieve the trajectory 'info' dict corresponding to a parsed step.

    Matches by the step's 'raw_index' (original position in trajectory);
    returns {} if raw_index is missing or out of range.
    """
    raw_idx = step.get("raw_index")
    if raw_idx is not None and 0 <= raw_idx < len(trajectory):
        entry = trajectory[raw_idx]
        info = entry.get("info", {}) if isinstance(entry, dict) else {}
        return info if isinstance(info, dict) else {}
    return {}


def extract_subagent_sessions(
    steps: list[dict], trajectory: list[dict],
) -> list[dict]:
    """Identify child sessions, keeping native Claude agents across interleaving.

    Returns list of sessions:
    ``[{"session_id": str, "start_step": int, "end_step": int,
        "step_count": int, "spawn_step": int | None}]``
    """
    sessions: list[dict] = []
    current_session: dict | None = None
    native_sessions: dict[str, dict] = {}

    for s in steps[:_MAX_STEPS]:
        info = _get_step_info(s, trajectory)
        # Prefer step fields (may be inferred from Task spawn metadata).
        is_sub = bool(s.get("is_sub_agent")) or bool(info.get("isSubAgent", False))
        session_id = s.get("session_id") or info.get("sessionID", "")
        if is_sub and "parent_agent_id" in s:
            session_id = s.get("agent") or session_id
        step_idx = s.get("index", 0)

        if "parent_agent_id" in s:
            if current_session is not None:
                sessions.append(current_session)
                current_session = None
            if is_sub and session_id:
                if session_id not in native_sessions:
                    native_sessions[session_id] = {
                        "session_id": session_id, "start_step": step_idx,
                        "end_step": step_idx, "step_count": 0, "spawn_step": None,
                    }
                native_sessions[session_id]["end_step"] = step_idx
                native_sessions[session_id]["step_count"] += 1
            continue

        if is_sub and session_id:
            if current_session is None or current_session["session_id"] != session_id:
                if current_session is not None:
                    sessions.append(current_session)
                current_session = {
                    "session_id": session_id,
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "step_count": 1,
                    "spawn_step": None,
                }
            else:
                current_session["end_step"] = step_idx
                current_session["step_count"] += 1
        else:
            if current_session is not None:
                sessions.append(current_session)
                current_session = None

    if current_session is not None:
        sessions.append(current_session)
    sessions.extend(native_sessions.values())
    sessions.sort(key=lambda session: session["start_step"])

    # OpenCode-style exports record the spawned child session on the parent
    # task tool part's metadata rather than on the child messages.  Build the
    # child-session -> delegation-step map in one pass so consolidated
    # CodeArts/OpenCode exports can identify the actual delegation step.
    spawn_by_child: dict[str, int] = {}
    for step in steps[:_MAX_STEPS]:
        if "parent_agent_id" in step:
            for call in step.get("tool_calls", []):
                metadata = call.get("metadata")
                child = metadata.get("agentId") if isinstance(metadata, dict) else None
                if isinstance(child, str) and child:
                    spawn_by_child.setdefault(child, step.get("index", 0))
        raw_idx = step.get("raw_index")
        if not isinstance(raw_idx, int) or not (0 <= raw_idx < len(trajectory)):
            continue
        entry = trajectory[raw_idx]
        parts = entry.get("parts", []) if isinstance(entry, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            state = part.get("state", {})
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            child_id = spawned_child_session_id(
                metadata,
                caller_session_id=str(step.get("session_id") or ""),
            )
            if child_id:
                spawn_by_child.setdefault(child_id, step.get("index", 0))

    for session in sessions:
        if session["spawn_step"] is None:
            session["spawn_step"] = spawn_by_child.get(session["session_id"])

    return sessions


def compute_subagent_metrics(
    sessions: list[dict], steps: list[dict],
) -> list[dict]:
    """Compute per-session metrics: tokens, tools, duration."""
    results = []
    for session in sessions:
        start = session["start_step"]
        end = session["end_step"]
        session_steps = [s for s in steps if start <= s.get("index", 0) <= end
                         and ("parent_agent_id" not in s or s.get("agent") == session["session_id"])]

        total_tokens = sum(s["tokens"]["total"] for s in session_steps)
        total_tools = sum(s["tool_call_count"] for s in session_steps)
        durations = [s["duration"] for s in session_steps if s.get("duration") is not None]
        total_duration = sum(durations)

        results.append({
            **session,
            "total_tokens": total_tokens,
            "total_tools": total_tools,
            "total_duration": round(total_duration, 2),
        })
    return results


# ---------------------------------------------------------------------------
# 6. Fruitless Search & Anti-Pattern Detection
# ---------------------------------------------------------------------------


def _is_search_call(tc: dict) -> bool:
    """A tool call counts as a search only if it actually searches.

    Bash is in ``_SEARCH_TOOL_NAMES`` because agents run grep/rg/find through
    it, but ordinary shell commands (mkdir, cp, chmod, git add) that legitimately
    emit no output must not be mistaken for empty searches.
    """
    name = tc.get("tool_name")
    if name not in _SEARCH_TOOL_NAMES:
        return False
    if name in BASH_TOOL_NAMES:
        inp = tc.get("input", {})
        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
        return shell_runs_search(cmd) if isinstance(cmd, str) else False
    return True


def _is_fruitless_step(step: dict) -> bool:
    """Whether every search call demonstrably succeeded with an empty result.

    Missing, failed or unresolved results cannot establish that a search found
    nothing. They break a streak rather than being counted as wasted work.
    """
    from .telemetry_quality import tool_field_observed

    tool_calls = step.get("tool_calls", [])
    if not tool_calls:
        return False

    search_calls = [tc for tc in tool_calls if _is_search_call(tc)]
    if not search_calls:
        return False

    # Method 1: Check individual tool call outputs/results
    for tc in search_calls:
        status = str(tc.get("status") or "").lower()
        if (status not in {"success", "completed", "complete", "done", "ok"}
                or tool_call_failed(tc)
                or not tool_field_observed(tc, "tool_outcomes")
                or not tool_field_observed(tc, "tool_outputs")):
            return False
        if "output" not in tc and "result" not in tc:
            return False
        output = tc.get("output", tc.get("result"))
        if output is None:
            return False
        if isinstance(output, str):
            stripped = output.strip()
            if stripped and stripped not in ("", "No matches found", "No results", "No files found"):
                return False
        elif isinstance(output, dict):
            content = output.get("content", output.get("text", ""))
            if isinstance(content, str) and content.strip():
                return False
            # Content-block lists (e.g. [{"type": "text", "text": ...}]) are
            # non-empty results too — mirror the bare-list handling below.
            if isinstance(content, list) and content:
                return False
        elif isinstance(output, list):
            if output:
                return False
        else:
            return False

    return True


def detect_fruitless_streaks(steps: list[dict]) -> list[dict]:
    """Detect consecutive steps with empty search/grep tool output.

    Returns list of streaks:
    ``[{"start_step": int, "end_step": int, "length": int, "tools": [str]}]``
    """
    streaks: list[dict] = []
    current_streak: dict | None = None

    for s in steps[:_MAX_STEPS]:
        if not s.get("tool_calls"):
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None
            continue

        if _is_fruitless_step(s):
            tool_names = [tc.get("tool_name", "?") for tc in s["tool_calls"]]
            step_idx = s.get("index", 0)
            if current_streak is None:
                current_streak = {
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "length": 1,
                    "tools": tool_names,
                }
            else:
                current_streak["end_step"] = step_idx
                current_streak["length"] += 1
                current_streak["tools"].extend(tool_names)
        else:
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None

    if current_streak is not None and current_streak["length"] >= 3:
        streaks.append(current_streak)

    return streaks


def detect_tool_selection_antipatterns(steps: list[dict]) -> list[dict]:
    """Detect Bash commands used for file reading when Read tool exists.

    Returns list of flagged steps.
    """
    # Only flag commands where sed/cat/head IS the primary command reading a file,
    # not when used in pipes (e.g., "grep ... | head -20" is legitimate).
    _BASH_READ_PATTERNS = [
        re.compile(r"^sed\s+-n\s+'"),          # sed as primary command
        re.compile(r"^cat\s+[\"']?[/\\a-zA-Z]"),  # cat as primary command
        re.compile(r"^head\s+-[n0-9]"),        # head as primary command
        re.compile(r"^tail\s+-[n0-9]"),        # tail as primary command
    ]

    flagged: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            # OpenCode emits lowercase tool names — match both spellings,
            # like _is_search_call above.
            if tc.get("tool_name") not in BASH_TOOL_NAMES:
                continue
            inp = tc.get("input", {})
            cmd = inp.get("command", "") if isinstance(inp, dict) else ""
            for pattern in _BASH_READ_PATTERNS:
                if pattern.search(cmd):
                    flagged.append({
                        "step": s.get("index", 0),
                        "command": cmd[:80],
                        "pattern": pattern.pattern,
                    })
                    break
    return flagged


# ---------------------------------------------------------------------------
# 6b. Failed same-path edit retries + repeated identical search
# ---------------------------------------------------------------------------

_EDIT_THRASH_WINDOW = 15
_EDIT_THRASH_MIN = 3
_REPEATED_SEARCH_MIN = 3


def detect_edit_thrash(steps: list[dict]) -> list[dict]:
    """Detect repeated Write/Edit on one path that includes failed attempts.

    Successful iterate/fix loops on the same file are normal and are NOT
    flagged. A cluster qualifies only when ≥3 write attempts hit the same path
    within a short window **and** at least one of those attempts failed.

    Returns
    -------
    list[dict]
        ``{"path", "count", "fail_count", "steps", "start_step", "end_step"}``
    """
    # path -> list of (step_idx, failed)
    by_path: dict[str, list[tuple[int, bool]]] = {}
    for s in steps[:_MAX_STEPS]:
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            name = tc.get("tool_name") or ""
            if name not in _WRITE_TOOL_NAMES:
                continue
            inp = tc.get("input", {})
            path = write_target_path(inp) if isinstance(inp, dict) else ""
            if not path:
                continue
            path = path.replace("\\", "/")
            by_path.setdefault(path, []).append((idx, tool_call_failed(tc)))

    thrash: list[dict] = []
    for path, events in by_path.items():
        if len(events) < _EDIT_THRASH_MIN:
            continue
        best: list[tuple[int, bool]] | None = None
        best_fails = 0
        for i in range(len(events)):
            window = [events[i]]
            for j in range(i + 1, len(events)):
                if events[j][0] - events[i][0] > _EDIT_THRASH_WINDOW:
                    break
                window.append(events[j])
            if len(window) < _EDIT_THRASH_MIN:
                continue
            fail_count = sum(1 for _, failed in window if failed)
            if fail_count < 1:
                continue
            if best is None or len(window) > len(best):
                best = window
                best_fails = fail_count
        if best is None:
            continue
        step_list = [idx for idx, _ in best]
        thrash.append({
            "path": path,
            "count": len(best),
            "fail_count": best_fails,
            "steps": step_list,
            "start_step": step_list[0],
            "end_step": step_list[-1],
        })

    thrash.sort(key=lambda t: (-t["fail_count"], -t["count"], t["start_step"], t["path"]))
    return thrash


def _search_signature(tc: dict) -> str | None:
    """Normalize a search tool call into a comparable signature, or None."""
    if not _is_search_call(tc):
        return None
    name = tc.get("tool_name") or ""
    inp = tc.get("input", {}) if isinstance(tc.get("input"), dict) else {}
    if name in ("Grep", "grep"):
        pattern = str(inp.get("pattern") or "").strip()
        path = str(inp.get("path") or "").strip()
        if not pattern:
            return None
        return f"grep:{pattern}|{path}"
    if name in ("Glob", "glob"):
        pattern = str(inp.get("pattern") or "").strip()
        path = str(inp.get("path") or "").strip()
        if not pattern:
            return None
        return f"glob:{pattern}|{path}"
    if name in BASH_TOOL_NAMES:
        cmd = str(inp.get("command") or "")
        normalized = " ".join(cmd.split())
        if not normalized:
            return None
        return f"bash:{normalized}"
    # Other search tools — use name + primary arg
    for key in ("pattern", "query", "path"):
        if inp.get(key):
            return f"{name}:{inp[key]}"
    return f"{name}:"


def detect_repeated_searches(steps: list[dict]) -> list[dict]:
    """Detect the same search signature used ≥3 times with empty results.

    Complements :func:`detect_fruitless_streaks` (consecutive-only) by catching
    non-consecutive repeats of an identical query/pattern.

    Returns
    -------
    list[dict]
        ``{"signature": str, "count": int, "steps": [int], "display": str}``
    """
    # signature -> step indices where that search was fruitless on the step
    by_sig: dict[str, list[int]] = {}
    for s in steps[:_MAX_STEPS]:
        if not _is_fruitless_step(s):
            continue
        idx = int(s.get("index", 0))
        seen_this_step: set[str] = set()
        for tc in s.get("tool_calls") or []:
            sig = _search_signature(tc)
            if not sig or sig in seen_this_step:
                continue
            seen_this_step.add(sig)
            by_sig.setdefault(sig, []).append(idx)

    out: list[dict] = []
    for sig, step_list in by_sig.items():
        unique_steps = sorted(set(step_list))
        if len(unique_steps) < _REPEATED_SEARCH_MIN:
            continue
        # Prefer non-consecutive repeats (fruitless streaks already cover runs).
        consecutive = all(
            unique_steps[i] + 1 == unique_steps[i + 1]
            for i in range(len(unique_steps) - 1)
        )
        if consecutive:
            continue
        display = sig
        if sig.startswith("grep:"):
            display = sig[len("grep:"):]
        elif sig.startswith("glob:"):
            display = sig[len("glob:"):]
        elif sig.startswith("bash:"):
            display = sig[len("bash:"):]
        out.append({
            "signature": sig,
            "count": len(unique_steps),
            "steps": unique_steps,
            "display": display[:80],
        })

    out.sort(key=lambda r: (-r["count"], r["steps"][0], r["signature"]))
    return out


# ---------------------------------------------------------------------------
# Oversized single tool outputs
# ---------------------------------------------------------------------------

# A tool result stays in the conversation history, so one oversized output is
# paid for again on every later step until a compaction drops it. Absolute
# floor: any single output this big is noise whatever the window size.
# Relative rule: smaller outputs still hurt when they fill a chunk of the
# window the agent actually has.
_OVERSIZED_ABS_TOKENS = 5000
_OVERSIZED_REL_TOKENS = 2000
_OVERSIZED_REL_SHARE = 0.15
_OVERSIZED_MAX_REPORTED = 5
# All-CJK text yields the most tokens per character; below this length the
# relative floor (_OVERSIZED_REL_TOKENS) is unreachable, so skip _est_tokens.
_OVERSIZED_MIN_CHARS = int(_OVERSIZED_REL_TOKENS * CJK_CHARS_PER_TOKEN)


def detect_oversized_outputs(steps: list[dict]) -> list[dict]:
    """Find single tool outputs big enough to tax every later step.

    Flags one output when it estimates to ≥5k tokens, or ≥2k tokens while
    covering ≥15% of the next step's context occupancy (input + cache read —
    the prompt the output lands in).
    Typical sources: whole-file ``cat``, unbounded ``find``, verbose test runs.

    Returns
    -------
    list[dict]
        ``{"step", "tool", "tokens", "share" (int % or None), "title"}``,
        largest first, capped at ``_OVERSIZED_MAX_REPORTED`` entries.
    """
    # Occupancy per step index: the context a step's prompt occupied. A tool
    # output lands in the *next* prompt, so shares are computed against the
    # next step that reports tokens, not the step that produced the output.
    occupancy_by_index: dict[int, int] = {}
    for s in steps[:_MAX_STEPS]:
        tokens = s.get("tokens") or {}
        occupancy = int(tokens.get("input") or 0) + int(tokens.get("cache_read") or 0)
        if occupancy > 0:
            occupancy_by_index[int(s.get("index", 0))] = occupancy
    ordered_indices = sorted(occupancy_by_index)

    def _next_occupancy(index: int) -> int | None:
        pos = bisect_right(ordered_indices, index)
        if pos >= len(ordered_indices):
            return None
        return occupancy_by_index[ordered_indices[pos]]

    out: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            output = as_text(tc.get("output"))
            if not output or output == "(no output)":
                continue
            if len(output) < _OVERSIZED_MIN_CHARS:
                continue
            est = _est_tokens(output)
            nxt = _next_occupancy(idx)
            share = round(est / nxt * 100) if nxt else None
            big_absolute = est >= _OVERSIZED_ABS_TOKENS
            big_relative = (
                est >= _OVERSIZED_REL_TOKENS
                and share is not None
                and share >= _OVERSIZED_REL_SHARE * 100
            )
            if not (big_absolute or big_relative):
                continue
            out.append({
                "step": idx,
                "tool": str(tc.get("tool_name") or "tool"),
                "tokens": est,
                "share": share,
                "title": str(tc.get("title") or tc.get("tool_name") or "")[:80],
            })

    out.sort(key=lambda r: (-r["tokens"], r["step"]))
    return out[:_OVERSIZED_MAX_REPORTED]


# ---------------------------------------------------------------------------
# Blind retries (identical call repeated after a failure)
# ---------------------------------------------------------------------------


def _retry_input_key(tc: dict) -> str:
    """Comparison key for "the same call": tool name plus normalized input."""
    inp = tc.get("input")
    if isinstance(inp, (dict, list)):
        body = json.dumps(inp, sort_keys=True, default=str, ensure_ascii=False)
    else:
        body = as_text(inp)
    return f"{as_text(tc.get('tool_name'))}:{body}"


def detect_blind_retries(steps: list[dict]) -> list[dict]:
    """Find a failed call retried verbatim — and failing again.

    Only consecutive identical calls count (same tool, same input, nothing in
    between): that is the agent resubmitting the exact call. A retry that
    *succeeds* is exempt — transient errors (timeouts, races) deserve one
    verbatim shot. What is flagged is the wasted pattern: unchanged input in
    an unchanged environment fails the same way twice.

    Returns
    -------
    list[dict]
        ``{"tool", "attempts", "failed", "steps": [int], "display"}``,
        one entry per qualifying run.
    """
    # Flatten calls in execution order: steps in order, calls within a step.
    flat: list[tuple[int, dict]] = []
    for s in steps[:_MAX_STEPS]:
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            flat.append((idx, tc))
    keys = [_retry_input_key(tc) for _idx, tc in flat]

    out: list[dict] = []
    i = 0
    while i < len(flat):
        key = keys[i]
        j = i + 1
        while j < len(flat) and keys[j] == key:
            j += 1
        run = flat[i:j]
        if len(run) >= 2:
            failures = [k for k, (_idx, tc) in enumerate(run) if tool_call_failed(tc)]
            # First attempt failed and at least one verbatim retry failed too.
            if failures and failures[0] == 0 and len(failures) >= 2:
                tc0 = run[0][1]
                display = (
                    as_text(tc0.get("title"))
                    or tool_call_hint(tc0)
                    or key
                )[:80]
                out.append({
                    "tool": as_text(tc0.get("tool_name")) or "tool",
                    "attempts": len(run),
                    "failed": len(failures),
                    "steps": sorted({idx for idx, _tc in run}),
                    "display": display,
                })
        i = j

    out.sort(key=lambda r: (-r["attempts"], r["steps"][0]))
    return out


# ---------------------------------------------------------------------------
# User repeated messages
# ---------------------------------------------------------------------------


def _user_typed_text(step: dict) -> str:
    """The text a human actually typed.

    Frameworks inject synthetic user messages (e.g. OpenCode's
    "Continue if you have next steps..." after compaction). Those parts are
    marked ``synthetic`` and must not count as the user speaking.
    """
    saw_text_part = False
    for p in step.get("parts") or []:
        if not isinstance(p, dict) or p.get("type") != "text":
            continue
        saw_text_part = True
        if p.get("synthetic"):
            continue
        text = as_text(p.get("text"))
        if text:
            return text
    if saw_text_part:
        return ""  # synthetic-/empty-only: the framework, not the user
    return as_text(step.get("text_preview"))


def detect_user_repeats(steps: list[dict]) -> list[dict]:
    """Find user messages sent verbatim two or more times.

    The exact same text re-sent is the most direct evidence available that the
    previous attempt did not satisfy the user — no inference required. Only
    whitespace is normalized (line wraps and indentation should not hide a
    repeat); anything else — "继续" vs "请继续" — is a different message and
    stays out, keeping the rule explainable in one sentence. Synthetic
    (framework-injected) messages never count as the user speaking.

    Returns
    -------
    list[dict]
        ``{"message", "count", "steps": [int]}``, most repeated first.
    """
    by_message: dict[str, list[int]] = {}
    for s in steps[:_MAX_STEPS]:
        if s.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", _user_typed_text(s)).strip()
        if not text:
            continue
        by_message.setdefault(text, []).append(int(s.get("index", 0)))

    out = [
        {"message": text, "count": len(idxs), "steps": sorted(idxs)}
        for text, idxs in by_message.items()
        if len(idxs) >= 2
    ]
    out.sort(key=lambda r: (-r["count"], r["steps"][0]))
    return out


# ---------------------------------------------------------------------------
# Code changes without any validation
# ---------------------------------------------------------------------------

# Only writes to code files count: a session that edits docs or config has
# nothing a test run would catch, so flagging it would be noise.
_CODE_EXTENSIONS = (
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".java", ".go", ".rs", ".rb", ".php", ".cs", ".swift",
    ".kt", ".kts", ".scala", ".sh", ".bash", ".vue", ".svelte",
    ".m", ".mm", ".sql",
)


def detect_unvalidated_writes(steps: list[dict]) -> list[dict]:
    """Code files written by write tools while nothing checked the result.

    Flags a session whose write tools successfully changed code files but in
    which no shell command ran a test/build/lint (the
    ``_VALIDATION_COMMAND_PATTERNS`` whitelist) or executed the program
    itself (``python3 app.py`` — a smoke run checks the result too). Writes
    through shell redirection (``sed -i``, ``echo >``) are not counted — the
    rule stays "write tools touched code, nothing checked the result".

    Returns
    -------
    list[dict]
        ``{"files", "steps": [int]}`` — one entry, or empty when the run is
        clean (no code writes, or validation ran at least once).
    """
    write_steps: list[int] = []
    paths: set[str] = set()
    for s in steps[:_MAX_STEPS]:
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            name = as_text(tc.get("tool_name"))
            inp = tc.get("input")
            if name in _WRITE_TOOL_NAMES:
                if tool_call_failed(tc):
                    continue
                path = write_target_path(inp if isinstance(inp, dict) else {})
                if path and path.lower().endswith(_CODE_EXTENSIONS):
                    write_steps.append(idx)
                    paths.add(path)
            elif isinstance(inp, dict):
                command = as_text(inp.get("command"))
                if _is_validation_command(command) or _smoke_runs_command(command):
                    return []

    if not write_steps:
        return []
    return [{"files": len(paths), "steps": sorted(set(write_steps))}]


# ---------------------------------------------------------------------------
# Redundant re-reads (same file, same range, nothing changed in between)
# ---------------------------------------------------------------------------

# WebFetch is excluded: a URL's content can change behind the agent's back,
# so re-fetching is never provably redundant.
_FILE_READ_TOOLS = _READ_TOOL_NAMES - {"WebFetch"}

# Shell rewrites that change file content without going through write tools.
# Mirrors the Codex / converge in-place edit recognition (sed -i, redirects).
_SED_IN_PLACE_RE = re.compile(r"(?:^|\s)(?:-i\S*|--in-place\b)")
_AWK_IN_PLACE_RE = re.compile(r"(?:^|\s)(?:-i\s+inplace\b|--in-place\b)")
_SHELL_REDIRECT_WRITE_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|echo|printf)\b[^;&|]*?(?:>>|>)",
    re.IGNORECASE,
)


def _is_compaction_step(step: dict) -> bool:
    """True when this step is (or contains) a compaction checkpoint."""
    if step.get("is_compaction_checkpoint") or step.get("role") == "compaction":
        return True
    return any(
        isinstance(p, dict) and p.get("type") == "compaction"
        for p in (step.get("parts") or [])
    )


def _shell_rewrite_paths(command: str) -> list[str] | None:
    """Paths a bash command rewrote, or ``None`` if it is not a rewrite.

    Returns an empty list when the command looks like a rewrite but no path
    tokens could be extracted — callers should invalidate every live read in
    that conversation rather than keep false-positive re-reads.
    """
    text = str(command or "")
    if not text.strip():
        return None
    lowered = text.lower()
    is_rewrite = False
    if "sed" in lowered and _SED_IN_PLACE_RE.search(text) or ("awk" in lowered or "gawk" in lowered) and _AWK_IN_PLACE_RE.search(text) or "tee " in lowered or "git apply" in lowered or "patch " in lowered or _SHELL_REDIRECT_WRITE_RE.search(text):
        is_rewrite = True
    if not is_rewrite:
        return None
    return _extract_bash_paths(text)


def _paths_match_rewrite(tracked: str, rewritten: str) -> bool:
    """True when a tracked read path is the same file a shell rewrite touched."""
    a = tracked.replace("\\", "/").rstrip("/")
    b = rewritten.replace("\\", "/").rstrip("/")
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _invalidate_reads(
    last_valid: dict[str, dict[str, dict[str, int]]],
    conv: str,
    path: str | None,
) -> None:
    """Drop live read signatures for *path*, or the whole conversation if None."""
    if path is None:
        last_valid.pop(conv, None)
        return
    by_path = last_valid.get(conv)
    if by_path is None:
        return
    if path in by_path:
        by_path.pop(path, None)
        return
    for tracked in [p for p in by_path if _paths_match_rewrite(p, path)]:
        by_path.pop(tracked, None)


def detect_redundant_re_reads(steps: list[dict]) -> list[dict]:
    """The same file read again while nothing could have changed it.

    A re-read is redundant only when, since the previous read of the *same
    path with the same parameters* (offset/limit — a different range is
    different content) *in the same conversation*, no successful write
    touched the file (write tool *or* shell rewrite like ``sed -i`` /
    redirects) and no compaction dropped it from context. Sub-agents hold
    separate contexts, so their reads are judged independently. Those events
    are exactly the cases where re-reading is justified, which keeps this
    card the honest opposite of the post-compaction thin-read signal.

    Returns
    -------
    list[dict]
        ``{"path", "count", "steps": [int]}`` per path, most redundant first.
    """
    # Nested so invalidation is O(signatures for that path), not O(all).
    # conversation -> path -> params-json -> last still-valid read step.
    last_valid: dict[str, dict[str, dict[str, int]]] = {}
    rows: dict[str, dict] = {}
    for s in steps[:_MAX_STEPS]:
        conv = as_text(s.get("session_id"))
        if _is_compaction_step(s):
            last_valid.pop(conv, None)
            continue
        idx = int(s.get("index", 0))
        for tc in s.get("tool_calls") or []:
            name = as_text(tc.get("tool_name"))
            inp = tc.get("input")
            inp = inp if isinstance(inp, dict) else {}
            if name in _WRITE_TOOL_NAMES:
                if tool_call_failed(tc):
                    continue
                path = write_target_path(inp)
                if path:
                    _invalidate_reads(last_valid, conv, path)
            elif name in BASH_TOOL_NAMES:
                if tool_call_failed(tc):
                    continue
                rewritten = _shell_rewrite_paths(as_text(inp.get("command")))
                if rewritten is None:
                    continue
                if not rewritten:
                    # Rewrite shape, unknown target — drop all live reads.
                    _invalidate_reads(last_valid, conv, None)
                else:
                    for path in rewritten:
                        _invalidate_reads(last_valid, conv, path)
            elif name in _FILE_READ_TOOLS:
                path = write_target_path(inp)
                if not path:
                    continue
                params = json.dumps(
                    {k: v for k, v in inp.items() if k not in WRITE_PATH_KEYS},
                    sort_keys=True,
                    default=str,
                )
                by_params = last_valid.setdefault(conv, {}).setdefault(path, {})
                if params in by_params:
                    row = rows.setdefault(
                        path, {"path": path, "count": 0, "steps": []}
                    )
                    row["count"] += 1
                    row["steps"].append(idx)
                by_params[params] = idx

    out = list(rows.values())
    out.sort(key=lambda r: (-r["count"], r["steps"][0]))
    return out
