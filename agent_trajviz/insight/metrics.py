"""Metric computation, health verdicts, and agent summaries."""

import math
import statistics
from collections.abc import Callable

from agent_trajviz.tool_vocab import SPAWN_TOOL_NAMES

from .tool_failure import tool_call_failed


def effective_agent(s: dict) -> str:
    """Return the best agent identifier for a step.

    User turns always map to ``""`` (main).  For assistant turns:
    - Native Claude Code: uses ``agent`` (children share the root sessionId).
    - CodeArts: uses ``session_id`` as the grouping key since each
      sub-agent gets a distinct session, and ``agent`` may be the same
      generic name (e.g. "Agent") for both main and sub-agent messages.
    Falls back to empty string for the main agent.
    """
    if s.get("role") == "user":
        return ""
    agent = s.get("agent", "") or ""
    suffix_sub = isinstance(agent, str) and agent.endswith("(subagent)")
    # Sub-agent: isSubAgent set, or the OpenCode "(subagent)" name suffix.
    # CodeArts groups by session_id (agent name is often generic).
    if s.get("is_sub_agent") or suffix_sub:
        if "parent_agent_id" in s:
            return agent or s.get("session_id") or ""
        return s.get("session_id") or agent or ""
    # Custom steps may name children without the explicit is_sub_agent flag.
    # A named main agent with is_sub_agent=False maps to "".
    if "is_sub_agent" not in s:
        return agent
    return ""


def _clip_agent_label(name: str, max_len: int) -> str:
    if max_len <= 1 or len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _preset_used_by_tagged_subagents(mode: str, steps: list[dict]) -> bool:
    if not mode:
        return False
    for s in steps:
        if not isinstance(s, dict) or not s.get("is_sub_agent"):
            continue
        if str(s.get("agent") or "").strip() == mode:
            return True
    return False


def _name_used_by_parent_agent(name: str, steps: list[dict]) -> bool:
    if not name:
        return False
    for s in steps:
        if not isinstance(s, dict) or s.get("is_sub_agent"):
            continue
        if str(s.get("agent") or "").strip() == name:
            return True
    return False


def tagged_subagent_display_label(
    agent_id: str,
    steps: list[dict],
    *,
    max_len: int = 20,
) -> str | None:
    """Label a lane as ``main`` / ``sub {id}`` when the run tags sub-agents.

    DSH (and similar) share a generic preset name across parent and children.
    Returns ``None`` when the run has no ``is_sub_agent`` steps, or when this
    lane is mixed, so callers can keep format-specific names (OpenCode modes).
    """
    if not agent_id:
        return None
    tagged = any(isinstance(s, dict) and s.get("is_sub_agent") for s in steps)
    if not tagged:
        return None
    sid = agent_id
    mode = ""
    if "::" in agent_id:
        sid, mode = agent_id.split("::", 1)
        mode = mode.strip()

    lane_sub = False
    lane_parent = False
    title = ""
    child_names: set[str] = set()
    for s in steps:
        if not isinstance(s, dict):
            continue
        step_sid = str(s.get("session_id") or "")
        if step_sid != sid and effective_agent(s) not in {agent_id, sid}:
            continue
        step_mode = str(s.get("agent") or "").strip()
        if mode and step_mode and step_mode != mode:
            if not (
                step_mode == "compaction"
                or s.get("role") == "compaction"
                or s.get("is_compaction_checkpoint")
            ):
                continue
        if s.get("is_sub_agent"):
            lane_sub = True
            if step_mode and step_mode != "compaction":
                child_names.add(step_mode)
        elif s.get("role") in ("assistant", "user", "compaction"):
            lane_parent = True
        if not title:
            candidate = str(s.get("session_title") or "").strip()
            if candidate:
                title = candidate

    if lane_sub and not lane_parent:
        if title:
            return _clip_agent_label(title, max_len)
        distinctive = [
            name for name in child_names
            if not _name_used_by_parent_agent(name, steps)
        ]
        if len(distinctive) == 1:
            return _clip_agent_label(distinctive[0], max_len)
        short = sid[:12] if len(sid) > 12 else sid
        if short:
            return f"sub {short}"
        return _clip_agent_label(mode or agent_id, max_len)
    if lane_parent and not lane_sub:
        if mode and not _preset_used_by_tagged_subagents(mode, steps):
            return _clip_agent_label(mode, max_len)
        return "main"
    return None


def agent_display_label(agent_id: str, steps: list[dict], *, max_len: int = 20) -> str:
    """Card / token-chart / Overview label for an ``effective_agent`` id.

    Empty id is ``main``. Tagged DSH-style runs reuse
    ``tagged_subagent_display_label`` so cards match the swimlane. Other
    formats keep the truncated id.
    """
    if not agent_id:
        return "main"
    tagged = tagged_subagent_display_label(agent_id, steps, max_len=max_len)
    if tagged:
        return tagged
    return agent_id[:8] + "\u2026" if len(agent_id) > 8 else agent_id


def disambiguate_agent_labels(
    agent_ids: list[str],
    steps: list[dict],
    *,
    max_len: int = 20,
) -> dict[str, str]:
    """Unique display labels; suffix a short session id when names collide."""
    from collections import Counter

    raw = {aid: agent_display_label(aid, steps, max_len=max_len) for aid in agent_ids}
    counts = Counter(raw.values())
    out: dict[str, str] = {}
    for aid, label in raw.items():
        if counts[label] > 1 and aid:
            sid = aid.split("::", 1)[0] if "::" in aid else aid
            suffix = sid[-6:] if len(sid) > 6 else sid
            out[aid] = f"{label} ({suffix})"
        else:
            out[aid] = label
    return out


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile using nearest-rank (q in [0, 1])."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    # Nearest-rank: ceil(q * n) - 1 (0-based). Truncating int((n-1)*q) would
    # pull every percentile one rank low (p95 of 10 values -> the 9th value).
    idx = max(0, min(len(vals) - 1, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def tool_call_duration_ms(tc: dict) -> float | None:
    """Best-effort duration of one tool call in milliseconds.

    OTLP calls prefer their observed ``duration_ms``: span durations retain
    nanosecond precision while display endpoints are rounded to milliseconds.
    An observed zero is a measurement, rather than absent timing.

    Native fallback chain:

    1. ``time_created``/``time_updated`` — OpenCode part wall-clock from when
       the tool call was issued until the result was written (includes queue).
    2. ``time_start``/``time_end`` — execution window only (OpenCode
       ``state.time``, or formats that stamp run bounds).
    3. ``duration_ms``, then ``metadata.totalDurationMs`` (Claude Code).

    Returns ``None`` when no usable timing exists. Shared by analytics,
    message metrics, tool stats, and hotspot decomposition so they can never
    disagree about the same tool call.
    """
    from .telemetry_quality import tool_field_observed

    if not tool_field_observed(tc, "duration"):
        return None
    metadata = tc.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("_otel"), dict):
        reported = tc.get("duration_ms")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            try:
                precise = float(reported)
            except OverflowError:
                precise = None
            if precise is not None and math.isfinite(precise) and precise >= 0:
                return precise
    created, updated = tc.get("time_created"), tc.get("time_updated")
    if (
        isinstance(created, (int, float))
        and isinstance(updated, (int, float))
        and updated >= created
    ):
        return float(updated - created)
    ts, te = tc.get("time_start"), tc.get("time_end")
    if isinstance(ts, (int, float)) and isinstance(te, (int, float)) and te >= ts:
        return float(te - ts)
    dm = tc.get("duration_ms")
    if not isinstance(dm, (int, float)):
        meta = tc.get("metadata")
        dm = meta.get("totalDurationMs") if isinstance(meta, dict) else None
    return float(dm) if isinstance(dm, (int, float)) and dm > 0 else None


def is_spawn_tool_call(tc: dict) -> bool:
    return (tc.get("tool_name") or "") in SPAWN_TOOL_NAMES


def tool_call_stats_duration_ms(tc: dict) -> float | None:
    """Duration for aggregate tool stats (excludes spawn/delegation tools).

    Spawn wall-clock is the child's run and is already reflected in child
    steps' tools/durations; counting it again inflates Tool time / wait %.
    """
    if is_spawn_tool_call(tc):
        return None
    return tool_call_duration_ms(tc)


def spawn_wait_seconds(step: dict) -> float:
    """Wall-clock spent blocked on spawn/delegation tools for *step*.

    Parallel ``task``/``Agent`` calls overlap, so this is the **max** timed
    spawn duration on the step (not the sum). Returns ``0`` when none.
    """
    wait = 0.0
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict) or not is_spawn_tool_call(tc):
            continue
        ms = tool_call_duration_ms(tc)
        if ms is not None:
            wait = max(wait, ms / 1000.0)
    return wait


def step_duration_excluding_spawn(step: dict) -> float | None:
    """Step duration with spawn/delegation wait removed.

    Used by the Step Duration chart and by :func:`generation_seconds` for the
    output tok/s denominator. Parent messages blocked on ``task``/``Agent``
    report wall-clock that is really the child's run; subtracting that wait
    avoids double-counting.
    """
    raw = step.get("duration")
    if (
        not isinstance(raw, (int, float))
        or isinstance(raw, bool)
        or not math.isfinite(raw)
    ):
        return None
    from .telemetry_quality import field_observed, is_otel_step

    if step.get("timing_source") == "model_request":
        return float(raw)
    if is_otel_step(step):
        return float(raw) if step["_otel"].get("kind") == "model_request" and field_observed(step, "duration") else None
    return max(0.0, float(raw) - spawn_wait_seconds(step))


def tool_call_window_ms(tc: dict) -> tuple[float, float] | None:
    """Absolute ``(start, end)`` ms window of one tool call, or ``None``.

    Native calls use the same fallback order as :func:`tool_call_duration_ms`:
    part wall-clock (``time_created``/``time_updated``) before execution bounds
    (``time_start``/``time_end``). OTLP unions use exact nanosecond execution
    bounds in :func:`non_spawn_tool_seconds` instead of display timestamps.
    """
    from .telemetry_quality import tool_field_observed

    if not tool_field_observed(tc, "duration"):
        return None
    for lo_key, hi_key in (("time_created", "time_updated"), ("time_start", "time_end")):
        lo, hi = tc.get(lo_key), tc.get(hi_key)
        if (
            isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and not isinstance(lo, bool)
            and not isinstance(hi, bool)
            and hi >= lo
        ):
            return float(lo), float(hi)
    return None


def non_spawn_tool_seconds(step: dict) -> float:
    """Wall-clock consumed by timed non-spawn tool calls on *step* (seconds).

    Agents issue tool calls in parallel within one step, so summing per-call
    durations double-counts the overlap and can exceed the step's own wall
    clock. Where the format stamps a window — part-level
    ``time_created``/``time_updated`` or state-level ``time_start``/``time_end``,
    in :func:`tool_call_window_ms` precedence — the union of those windows is
    exact, so that is used. Calls carrying only a duration (Claude Code's
    ``metadata.totalDurationMs``) have no window to union and still contribute
    their full length, which leaves the result an upper bound for those
    exports; that residual is a data limitation, not something measurable here.
    OTLP execution windows are unioned as integer nanoseconds before conversion
    to seconds, retaining intervals shorter than one display millisecond.
    """
    windows: list[tuple[float, float]] = []
    precise_windows: list[tuple[int, int]] = []
    unstamped = 0.0
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        ms = tool_call_stats_duration_ms(tc)
        if ms is None:
            continue
        metadata = tc.get("metadata")
        otel = metadata.get("_otel") if isinstance(metadata, dict) else None
        if isinstance(otel, dict):
            lo, hi = otel.get("execution_start_time_unix_nano"), otel.get("execution_end_time_unix_nano")
            if (isinstance(lo, int) and not isinstance(lo, bool) and lo > 0
                    and isinstance(hi, int) and not isinstance(hi, bool) and hi >= lo):
                precise_windows.append((lo, hi))
            else:
                # Log-only timings and older normalized captures have no
                # precise bounds. Their observed durations remain an upper
                # bound rather than a fabricated rounded execution interval.
                unstamped += ms / 1000.0
            continue
        window = tool_call_window_ms(tc)
        if window is not None:
            windows.append(window)
        else:
            unstamped += ms / 1000.0

    total = unstamped
    if precise_windows:
        precise_windows.sort()
        cur_start_ns, cur_end_ns = precise_windows[0]
        duration_ns = 0
        for start_ns, end_ns in precise_windows[1:]:
            if start_ns > cur_end_ns:
                duration_ns += cur_end_ns - cur_start_ns
                cur_start_ns, cur_end_ns = start_ns, end_ns
            else:
                cur_end_ns = max(cur_end_ns, end_ns)
        duration_ns += cur_end_ns - cur_start_ns
        total += duration_ns / 1_000_000_000
    if windows:
        windows.sort()
        cur_start, cur_end = windows[0]
        for start, end in windows[1:]:
            if start > cur_end:  # disjoint: close the run and start a new one
                total += (cur_end - cur_start) / 1000.0
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        total += (cur_end - cur_start) / 1000.0
    return total


def wall_clock_fmt(metrics: dict) -> tuple[float, str]:
    """Return (wall_seconds, formatted_string) for wall-clock time."""
    wall = metrics["wall_clock"] if isinstance(metrics["wall_clock"], (int, float)) else metrics["total_duration"]
    if wall is None:
        return 0.0, "n/a"
    fmt = f"{wall:.0f}s" if wall < 3600 else f"{wall / 60:.1f}m"
    return wall, fmt


def generation_seconds(step: dict) -> tuple[float, float] | None:
    """Return ``(generation_s, capped_tool_wait_s)`` excluding spawn wait.

    Output tok/s uses generation time as the denominator so a long Bash call
    does not look like near-zero throughput. Tool wait is capped so overlapping
    stamps cannot drive generation negative. Returns ``None`` without duration.
    """
    from .telemetry_quality import field_observed, is_otel_step

    if step.get("timing_source") == "model_request":
        duration = step.get("duration")
        return ((float(duration), 0.0) if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and math.isfinite(duration) and duration >= 0 else None)
    if is_otel_step(step):
        if step["_otel"].get("kind") != "model_request" or not field_observed(step, "duration"):
            return None
        # A request span already measures the model operation. Tools are
        # separate operations, not waits to subtract from this duration.
        duration = step.get("duration")
        return (float(duration), 0.0) if isinstance(duration, (int, float)) and duration >= 0 else None
    remaining = step_duration_excluding_spawn(step)
    if remaining is None:
        return None
    tool_s = min(non_spawn_tool_seconds(step), remaining)
    return remaining - tool_s, tool_s


def _raw_summary(raw: dict) -> tuple[dict, dict | None]:
    """Extract the ``output`` dict and ``session_raw.summary`` dict from raw."""
    output = raw.get("output", {}) if isinstance(raw.get("output"), dict) else {}
    session_raw = raw.get("session_raw", {}) if isinstance(raw.get("session_raw"), dict) else {}
    summary = session_raw.get("summary") if isinstance(session_raw.get("summary"), dict) else None
    return output, summary


def _churn(summary: dict | None) -> int | None:
    """Total line churn (additions + deletions), or None when unavailable."""
    if summary and "additions" in summary and "deletions" in summary:
        return summary["additions"] + summary["deletions"]
    return None


def build_message_metrics(steps: list[dict]) -> list[dict]:
    """Build per-message metrics used for diagnostics tables and charts."""
    from .parser import cache_read_share, infer_non_cache_input

    rows: list[dict] = []
    for s in steps:
        tokens = s.get("tokens", {})
        tok_total = tokens.get("total", 0) or 0
        tok_input = tokens.get("input", 0) or 0
        tok_output = tokens.get("output", 0) or 0
        tok_reasoning = tokens.get("reasoning", 0) or 0
        cache_read = tokens.get("cache_read", 0) or 0
        non_cache = infer_non_cache_input(
            total_tokens=tok_total,
            input_tokens=tok_input,
            output_tokens=tok_output,
            reasoning_tokens=tok_reasoning,
            cache_read_tokens=cache_read,
            cache_write_tokens=tokens.get("cache_write", 0) or 0,
        )
        duration = s.get("duration")

        tool_time_sum = non_spawn_tool_seconds(s)

        part_counts: dict[str, int] = {}
        for p in s.get("parts", []):
            pt = p.get("type", "unknown")
            part_counts[pt] = part_counts.get(pt, 0) + 1

        rows.append({
            "index": s.get("index", 0),
            "role": s.get("role", "?"),
            "agent": effective_agent(s),
            "model_id": s.get("model_id", ""),
            "finish": s.get("finish", ""),
            "duration": duration,
            "tokens_total": tok_total,
            "tokens_input": tok_input,
            "tokens_output": tok_output,
            "cache_read": cache_read,
            "non_cache_tokens": non_cache,
            "cache_ratio": cache_read_share(cache_read, tok_total),
            "tokens_per_sec": (tok_total / duration) if duration and duration > 0 else None,
            "non_cache_per_sec": (non_cache / duration) if duration and duration > 0 else None,
            # Same domain rule as the session total and analytics.py: without a
            # positive input there is no ratio. The `max(1, ...)` floor this
            # replaces published the output count itself as a ratio.
            "output_input_ratio": (tok_output / tok_input) if tok_input > 0 else None,
            "tool_calls": s.get("tool_call_count", 0),
            "errors": s.get("error_count", 0),
            "tool_time_sum": tool_time_sum,
            "tool_time_share": (tool_time_sum / duration) if duration and duration > 0 else 0.0,
            "reasoning_parts": part_counts.get("reasoning", 0),
            "text_parts": part_counts.get("text", 0),
            "patch_parts": part_counts.get("patch", 0),
        })
    from .telemetry_quality import field_observed, is_otel_step

    for step, row in zip(steps, rows, strict=True):
        if step.get("timing_source") == "model_request":
            row["tool_time_share"] = None
        if not is_otel_step(step):
            continue
        for key, field in {
            "tokens_total": "tokens.total", "tokens_input": "tokens.input",
            "tokens_output": "tokens.output", "cache_read": "tokens.cache_read",
            "duration": "duration", "errors": "tool_outcomes",
        }.items():
            if not field_observed(step, field):
                row[key] = None
        if not all(field_observed(step, f"tokens.{key}") for key in ("total", "cache_read")):
            row["cache_ratio"] = None
        if field_observed(step, "tokens.input"):
            # Claude telemetry reports fresh input separately from cache
            # reads/writes, so no total-minus-cache inference is necessary.
            row["non_cache_tokens"] = tokens_input = (step.get("tokens") or {}).get("input", 0)
            row["non_cache_per_sec"] = tokens_input / row["duration"] if row["duration"] else None
        else:
            row["non_cache_tokens"] = row["non_cache_per_sec"] = None
        if not field_observed(step, "tokens.total") or not field_observed(step, "duration"):
            row["tokens_per_sec"] = None
        if not all(field_observed(step, f"tokens.{key}") for key in ("input", "output")):
            row["output_input_ratio"] = None
        if not field_observed(step, "text"):
            row["reasoning_parts"] = row["text_parts"] = row["patch_parts"] = None
        # Claude response events deliberately omit thinking/tool blocks.
        # Ordinary captured text cannot establish that those blocks were absent.
        row["reasoning_parts"] = row["patch_parts"] = None
        calls = step.get("tool_calls") or []
        if calls and not any(tool_call_stats_duration_ms(call) is not None for call in calls):
            row["tool_time_sum"] = None
        # An explicit request/tool join establishes provenance, not nested
        # timing. A tool's execution may start after this model span ends.
        row["tool_time_share"] = None
        row["availability"] = dict(step["_otel"].get("fields", {}))
    return rows


def _compute_command_metrics(steps: list[dict]) -> dict:
    """Compute command execution success rate from tool calls with exit codes."""
    cmd_total = 0
    cmd_failures = 0
    for s in steps:
        for tc in s.get("tool_calls", []):
            meta = tc.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            if "exit" in meta:
                cmd_total += 1
                # exit=None (cancelled/unfinished, no exit code recorded) is not
                # a failure — matches tool_call_failed / cluster_errors.
                if meta["exit"] not in (None, 0):
                    cmd_failures += 1
    if cmd_total == 0:
        return {"command_success_rate": None, "command_call_count": None, "command_failures": None}
    return {
        "command_success_rate": round((cmd_total - cmd_failures) / cmd_total, 4),
        "command_call_count": cmd_total,
        "command_failures": cmd_failures,
    }


def output_throughput(steps: list[dict]) -> dict:
    """Output tokens over generation time, measured on one set of assistant steps.

    Tokens and seconds come from exactly the same steps, so untimed output
    (common for some sub-agent final messages) cannot inflate the rate. The
    denominator is :func:`generation_seconds` -- step duration minus spawn wait
    and timed non-spawn tool time -- so blocked Bash/script waits are not
    counted as generation. When tool stamps are missing those waits stay in the
    denominator, which under-states the rate rather than inflating it.

    Every surface that reports a generation rate -- the session metrics, each
    agent's row, the health verdict -- goes through this function, so a
    single-agent run shows one number everywhere.

    Returns ``rate`` (``None`` when nothing is measurable), the ``tokens`` and
    ``seconds`` it divides, ``tool_wait_seconds`` taken out of the window,
    ``timed_steps`` it covered and ``total_steps`` (all assistant steps).
    """
    total_steps = 0
    timed_steps = 0
    tokens = 0.0
    seconds = 0.0
    tool_wait = 0.0
    for s in steps:
        if s.get("role") != "assistant":
            continue
        from .telemetry_quality import field_observed, is_otel_step
        if is_otel_step(s) and s["_otel"].get("kind") != "model_request":
            continue
        total_steps += 1
        if not field_observed(s, "tokens.output") or not field_observed(s, "duration"):
            continue
        duration = s.get("duration")
        output_tokens = (s.get("tokens") or {}).get("output")
        has_duration = (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and duration > 0
        )
        has_output_tokens = (
            isinstance(output_tokens, (int, float))
            and not isinstance(output_tokens, bool)
            and math.isfinite(output_tokens)
            and output_tokens >= 0
        )
        if not (has_duration and has_output_tokens):
            continue
        gen_pair = generation_seconds(s)
        if gen_pair is None:
            continue
        gen_s, tool_wait_s = gen_pair
        if gen_s <= 0:
            # Tool stamps consumed the whole step, so this step has no
            # measurable generation window. Counting its output tokens against
            # zero seconds would inflate the rate without bound; leave it out
            # and let the coverage figure say so.
            continue
        timed_steps += 1
        seconds += gen_s
        tool_wait += tool_wait_s
        tokens += output_tokens
    return {
        "rate": (round(tokens / seconds, 1)
                 if seconds > 0 and (tokens > 0 or any("_otel" in step for step in steps)) else None),
        "tokens": tokens,
        "seconds": seconds,
        "tool_wait_seconds": tool_wait,
        "timed_steps": timed_steps,
        "total_steps": total_steps,
    }


def _compute_timing_metrics(steps: list[dict]) -> dict:
    """Compute response latency, output throughput, and timing coverage.

    ``time_to_first_token`` / ``time_to_last_token`` are **response latencies**,
    not token-level measurements despite their names: each is the first user
    message's creation stamp to the first (resp. last) assistant message's
    *completion* stamp, so the whole turn is inside the figure. The first
    response is the first assistant message that began and completed at or
    after that user message. One that began earlier answered something else,
    and counting it gave a negative or a zero latency: a Codex sub-agent's
    first task arrives as a non-user message, and a turn cut off before
    ``task_complete`` is taken to end where the next message begins, which
    can be that user message. "At" keeps a reply that whole-second stamps put
    in its question's second. A message without a creation stamp is judged by
    its completion alone, since nothing shows it began earlier. With no such
    message ``time_to_first_token`` is ``None``; it is never negative.
    No supported export records streaming or first-delta timing — the step
    model carries only ``time_created_ms``/``time_completed_ms`` — so a true
    time-to-first-token is not computable here. The names are kept because
    callers already read them.

    Output throughput is :func:`output_throughput` over every assistant step.
    """
    first_user_created = None
    # (created or None, completed) of each assistant step with a completion, in step order.
    asst_spans: list[tuple[int | float | None, int | float]] = []

    for s in steps:
        role = s.get("role", "")
        t_created = s.get("time_created_ms")
        t_completed = s.get("time_completed_ms")
        if role == "user" and first_user_created is None and isinstance(t_created, (int, float)):
            first_user_created = t_created
        if role == "assistant" and isinstance(t_completed, (int, float)):
            began = t_created if isinstance(t_created, (int, float)) else None
            asst_spans.append((began, t_completed))

    first_asst_completed = None
    if first_user_created is not None:
        first_asst_completed = next((
            ended for began, ended in asst_spans
            if ended >= first_user_created and (began is None or began >= first_user_created)
        ), None)
    last_asst_completed = asst_spans[-1][1] if asst_spans else None

    result: dict = {}
    if first_user_created is not None and first_asst_completed is not None:
        # First-response latency: the first assistant turn is COMPLETE at this
        # point, so this is not a time-to-first-token (see the docstring).
        result["time_to_first_token"] = round((first_asst_completed - first_user_created) / 1000, 3)
    else:
        result["time_to_first_token"] = None

    throughput = output_throughput(steps)
    timed, total = throughput["timed_steps"], throughput["total_steps"]
    result["output_tokens_per_sec"] = throughput["rate"]
    result["output_throughput_timed_steps"] = timed
    result["output_throughput_total_steps"] = total
    result["output_throughput_coverage_pct"] = round(timed / total * 100, 1) if total else None
    result["output_throughput_incomplete"] = timed < total
    result["output_throughput_timed_tokens"] = throughput["tokens"]
    result["output_throughput_timed_seconds"] = round(throughput["seconds"], 3)
    result["output_throughput_tool_wait_seconds"] = round(throughput["tool_wait_seconds"], 3)

    if first_user_created is not None and last_asst_completed is not None:
        # Full-response latency, same caveat: last assistant message completed.
        result["time_to_last_token"] = round((last_asst_completed - first_user_created) / 1000, 3)
    else:
        result["time_to_last_token"] = None

    return result


def _compute_plan_metrics(steps: list[dict]) -> dict:
    """Compute plan tracking from todo snapshot parts."""
    snapshots = []
    for s in steps:
        for p in s.get("parts", []):
            if p.get("type") == "snapshot":
                data = p.get("data", {})
                if isinstance(data, dict) and ("todos" in data or "items" in data):
                    snapshots.append(data)
    if not snapshots:
        return {"plan_items": None, "plan_completion_ratio": None, "plan_update_count": None}

    # First snapshot for plan_items count
    first = snapshots[0]
    items = first.get("todos", first.get("items", []))
    plan_items = len(items) if isinstance(items, list) else 0

    # Last snapshot for completion ratio
    last = snapshots[-1]
    last_items = last.get("todos", last.get("items", []))
    if isinstance(last_items, list) and last_items:
        completed = sum(
            1 for item in last_items
            if (isinstance(item, dict) and item.get("completed", False))
            or (isinstance(item, dict) and item.get("status") in ("completed", "done"))
        )
        plan_completion_ratio = round(completed / len(last_items), 4)
    else:
        plan_completion_ratio = None

    return {
        "plan_items": plan_items,
        "plan_completion_ratio": plan_completion_ratio,
        "plan_update_count": len(snapshots),
    }


def _compute_token_stats(total_tokens, total_duration, steps, message_rows, raw):
    """Token breakdown, throughput, and cache metrics."""
    output, summary = _raw_summary(raw)

    assistant_rows = [r for r in message_rows if r.get("role") == "assistant"]
    assistant_tokens = [r["tokens_total"] for r in assistant_rows if r["tokens_total"] is not None]
    token_rates = [r["tokens_per_sec"] for r in assistant_rows if r.get("tokens_per_sec") is not None]
    cache_rows = [r for r in assistant_rows if r["tokens_total"] is not None and r["tokens_total"] > 0]
    cache_ratios = [r["cache_ratio"] for r in cache_rows if r["cache_ratio"] is not None]
    # Steps whose record is self-inconsistent (cache_read > total). Averaging
    # only the rest would bias the result downward precisely because the
    # excluded steps are the cache-heavy ones, so the count is published and
    # the mean is withheld when any step is unusable.
    cache_unusable = len(cache_rows) - len(cache_ratios)
    non_cache_total = sum(r["non_cache_tokens"] or 0 for r in message_rows)
    cache_dominant = sum(1 for r in cache_rows if (r["cache_ratio"] or 0) >= 0.90)
    total_io = total_tokens["input"] + total_tokens["output"]
    churn = _churn(summary) or 0
    return {
        "tokens": total_tokens,
        "non_cache_tokens": non_cache_total,
        "non_cache_ratio": round(non_cache_total / total_tokens["total"] * 100, 1) if total_tokens["total"] else 0,
        # Two "per step" averages on purpose. `avg_tokens_per_step` divides by
        # ALL steps (user turns included, which report no tokens) and is kept
        # emitted for continuity with figures already reported. Only the assistant
        # variant shares a denominator with median_step_tokens / p95_step_tokens
        # below, so that is the one to compare against them.
        "avg_tokens_per_step": round(total_tokens["total"] / len(steps)) if steps else 0,
        "avg_tokens_per_assistant_step": (
            round(total_tokens["total"] / len(assistant_rows)) if assistant_rows else 0
        ),
        "tokens_per_second": round(total_tokens["total"] / total_duration, 1) if total_duration else 0,
        # Meaningless without a real input total: `max(1, 0)` turned a session
        # whose every input was rejected into "Out/In 9184.0".
        "output_input_ratio": (
            round(total_tokens["output"] / total_tokens["input"], 3)
            if total_tokens["input"] > 0 else None
        ),
        "median_step_tokens": round(statistics.median(assistant_tokens)) if assistant_tokens else 0,
        "p95_step_tokens": round(_percentile(assistant_tokens, 0.95)) if assistant_tokens else 0,
        "median_tokens_per_second": round(statistics.median(token_rates), 1) if token_rates else 0,
        "avg_cache_ratio": (
            round(statistics.mean(cache_ratios) * 100, 1)
            if cache_ratios and not cache_unusable else None
        ),
        "cache_ratio_unusable_steps": cache_unusable,
        "cache_dominant_steps": cache_dominant,
        "assistant_steps": len(assistant_rows),
        "user_steps": sum(1 for r in message_rows if r.get("role") == "user"),
        "input_tokens": total_tokens["input"],
        "output_tokens": total_tokens["output"],
        "cache_read_tokens": total_tokens["cache_read"],
        "tokens_per_patch_line": round(total_io / output.get("patch_lines", 0), 1) if output.get("patch_lines", 0) > 0 else None,
        "tokens_per_churn_line": round(total_io / churn, 1) if churn > 0 else None,
    }


def _compute_tool_stats(steps, total_tokens_total, message_rows, wait_denom: float):
    """Tool frequency, success rate, duration, and load metrics.

    Timed aggregates omit spawn/delegation tools. ``wait_denom`` should be
    session wall-clock (not summed step durations) when subagents overlap.
    """
    tool_count = 0
    tool_breakdown: dict[str, int] = {}
    tool_status_breakdown: dict[str, int] = {}
    tool_success = 0
    tool_fail = 0
    tool_durations: list[float] = []
    delegated_total = 0.0
    delegated_calls = 0
    unknown_outcomes = 0
    from .telemetry_quality import tool_field_observed

    for s in steps:
        tool_count += s["tool_call_count"]
        # Delegation wall-clock stays out of tool-execution time on purpose,
        # but it must not vanish: Claude Code stamps timing ONLY on the spawn
        # call, so dropping it leaves those exports with no timing at all.
        # Use the shared per-step helper rather than summing calls — parallel
        # task/Agent calls overlap, and step_duration_excluding_spawn already
        # subtracts exactly this quantity.
        delegated_total += spawn_wait_seconds(s)
        delegated_calls += sum(
            1 for tc in (s.get("tool_calls") or [])
            if isinstance(tc, dict) and is_spawn_tool_call(tc)
            and tool_call_duration_ms(tc) is not None
        )
        for tc in s["tool_calls"]:
            name = tc["tool_name"]
            tool_breakdown[name] = tool_breakdown.get(name, 0) + 1
            status = tc.get("status", "unknown")
            tool_status_breakdown[status] = tool_status_breakdown.get(status, 0) + 1
            if not tool_field_observed(tc, "tool_outcomes"):
                unknown_outcomes += 1
            elif tool_call_failed(tc):
                tool_fail += 1
            else:
                # A call carrying no failure signal — no failure status, no
                # error/error_type, no non-zero metadata.exit — scores as a
                # success, so an in-flight or never-stamped call is optimistic
                # by design. `tool_failure.tool_call_failed` is the single
                # source of truth for that judgement; do not reintroduce a
                # status allow-list here, it can only drift away from it.
                tool_success += 1
            v = tool_call_stats_duration_ms(tc)
            if v is not None:
                tool_durations.append(v / 1000.0)

    assistant_rows = [r for r in message_rows if r.get("role") == "assistant"]
    tool_time_total = sum(r["tool_time_sum"] or 0 for r in message_rows)
    avg_td = statistics.mean(tool_durations) if tool_durations else 0
    denom = float(wait_denom) if wait_denom > 0 else 0.0
    # No timed non-spawn call means "this export records no per-tool timing",
    # which is not the same claim as "the tools took zero seconds". Report
    # None so every surface renders n/a instead of a fabricated zero.
    timed = bool(tool_durations) or tool_time_total > 0
    # Native mirrors can omit outcomes entirely. OTel rates explicitly use
    # the observed population; native session-wide rates require coverage.
    complete_outcomes = not unknown_outcomes or any("_otel" in s for s in steps)
    return {
        "tool_call_count": tool_count,
        "tool_breakdown": tool_breakdown,
        "tool_status_breakdown": tool_status_breakdown,
        "tool_success": tool_success,
        "tool_fail": tool_fail,
        # No tool calls means no rate: None ("n/a"), not a 0% success.
        "tool_success_rate": (
            round(tool_success / (tool_success + tool_fail) * 100, 1)
            if tool_success + tool_fail and complete_outcomes else None
        ),
        **({"tool_unknown_outcomes": unknown_outcomes, "tool_observed_outcomes": tool_success + tool_fail}
           if unknown_outcomes or any("_otel" in s for s in steps) else {}),
        "tokens_per_tool": round(total_tokens_total / tool_count) if tool_count else 0,
        "tool_time_total": round(tool_time_total, 2) if timed else None,
        "tool_wait_share": round(tool_time_total / denom * 100, 1) if timed and denom else None,
        "avg_tool_duration": round(avg_td, 3) if timed else None,
        "p95_tool_duration": round(_percentile(tool_durations, 0.95), 3) if tool_durations else None,
        "max_tool_duration": round(max(tool_durations), 3) if tool_durations else None,
        # Wall-clock the parent spent blocked on sub-agents, kept separate from
        # tool-execution time so delegation is visible rather than discarded.
        "delegation_time_total": round(delegated_total, 2) if delegated_calls else None,
        "delegated_call_count": delegated_calls,
        "multi_tool_steps": sum(1 for r in assistant_rows if r["tool_calls"] >= 2),
        "no_tool_assistant_steps": sum(1 for r in assistant_rows if r["tool_calls"] == 0),
        "patch_steps": sum(1 for r in assistant_rows if (r["patch_parts"] or 0) > 0),
        "tool_calls_per_min": round(tool_count / (denom / 60), 2) if denom > 0 else None,
        "tool_time_fraction": round(tool_time_total / denom, 4) if timed and denom > 0 else None,
        "tool_system_failure_rate": (
            round(tool_fail / (tool_success + tool_fail), 4)
            if tool_success + tool_fail and complete_outcomes else None
        ),
    }


def _compute_efficiency_stats(steps, message_rows, raw):
    """Behavioral, structural, and change-scope metrics."""
    roles: dict[str, int] = {}
    agent_breakdown: dict[str, int] = {}
    model_breakdown: dict[str, int] = {}
    finish_breakdown: dict[str, int] = {}
    reasoning_parts = text_parts = snapshot_parts = 0
    for s in steps:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
        agent = effective_agent(s)
        if agent:
            agent_breakdown[agent] = agent_breakdown.get(agent, 0) + 1
        model = s.get("model_id", "")
        if model:
            model_breakdown[model] = model_breakdown.get(model, 0) + 1
        finish = s.get("finish", "")
        if finish:
            finish_breakdown[finish] = finish_breakdown.get(finish, 0) + 1
        for p in s.get("parts", []):
            pt = p.get("type", "")
            if pt == "reasoning":
                reasoning_parts += 1
            elif pt == "text":
                text_parts += 1
            elif pt == "snapshot":
                snapshot_parts += 1

    output, summary = _raw_summary(raw)
    file_status_raw = raw.get("file_status")
    asst_durs = [s["duration"] for s in steps if s.get("role") == "assistant" and s.get("duration") is not None]
    user_n, asst_n = roles.get("user", 0), roles.get("assistant", 0)
    agent_labels = disambiguate_agent_labels(list(agent_breakdown), steps)
    return {
        "messages_breakdown": roles,
        "agent_breakdown": agent_breakdown,
        "agent_labels": agent_labels,
        "model_breakdown": model_breakdown,
        "finish_breakdown": finish_breakdown,
        "reasoning_parts": reasoning_parts,
        "text_parts": text_parts,
        "snapshot_parts": snapshot_parts,
        "patch_lines": output.get("patch_lines", 0),
        "has_patch": output.get("has_patch", False),
        "patch_error": output.get("error"),
        "files_changed": (
            summary.get("files") if summary and "files" in summary
            else len(file_status_raw) if isinstance(file_status_raw, list) else None
        ),
        "additions": summary.get("additions") if summary else None,
        "deletions": summary.get("deletions") if summary else None,
        "churn": _churn(summary),
        "net_change": (summary["additions"] - summary["deletions"]) if summary and "additions" in summary and "deletions" in summary else None,
        "user_turns": user_n,
        "assistant_turns": asst_n,
        "autonomy_ratio": round(asst_n / (user_n + asst_n), 4) if (user_n + asst_n) > 0 else None,
        "p50_duration": round(_percentile(asst_durs, 0.50), 2) if asst_durs else None,
        "p90_duration": round(_percentile(asst_durs, 0.90), 2) if asst_durs else None,
        "p99_duration": round(_percentile(asst_durs, 0.99), 2) if asst_durs else None,
    }


def session_wall_clock_seconds(
    steps: list[dict],
    timing: dict | None = None,
) -> float | None:
    """Calendar elapsed time for the session in seconds.

    Prefers ``timing.total_duration`` from the loader when present. Otherwise
    derives wall-clock from the span of step ``time_created_ms`` /
    ``time_completed_ms``. That matters for multi-agent traces where summed
    step durations double-count overlapping parent wait and child work.
    """
    if isinstance(timing, dict):
        td = timing.get("total_duration")
        if isinstance(td, (int, float)) and not isinstance(td, bool) and td > 0:
            return float(td)
    stamps: list[float] = []
    for s in steps:
        for key in ("time_created_ms", "time_completed_ms"):
            v = s.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stamps.append(float(v))
    if len(stamps) < 2:
        return None
    span = (max(stamps) - min(stamps)) / 1000.0
    return span if span > 0 else None


TOKEN_FIELDS = ("total", "input", "output", "reasoning", "cache_read", "cache_write")


def sum_usable_tokens(
    steps: list[dict],
    *,
    predicate: Callable[[dict], bool] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Sum per-step token fields, returning ``(totals, unusable_step_counts)``.

    Every token total in TrajViz goes through here, and here goes through
    ``parser.usable_token_count``: a negative input (OpenCode double-subtracts
    the cache read against providers whose ``input_tokens`` is already
    cache-exclusive) is not a token count and must not be summed — it dragged a
    reported session Input below zero by millions. Rejected steps are counted per field
    so the surfaces can say "partial" instead of printing a confident wrong
    total, or clamping it to a false zero.

    *predicate* restricts the steps counted (per-agent rows pass the agent
    test). Written as one helper rather than two accumulators because the
    per-agent sum previously bypassed the domain check and published the
    negative that the session total had already rejected.
    """
    from .parser import usable_token_count

    totals: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
    unusable: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
    for s in steps:
        if predicate is not None and not predicate(s):
            continue
        from .telemetry_quality import field_observed, is_otel_step
        if is_otel_step(s) and s["_otel"].get("kind") != "model_request":
            continue
        tokens = s.get("tokens") or {}
        for k in TOKEN_FIELDS:
            if not field_observed(s, f"tokens.{k}"):
                continue
            reported = tokens.get(k)
            if reported in (None, 0):
                continue
            usable = usable_token_count(reported)
            if usable is None:
                unusable[k] += 1
                continue
            totals[k] += usable
    return totals, unusable


def compute_metrics(steps: list[dict], raw: dict, message_rows: list[dict] | None = None) -> dict:
    """Aggregate metrics from parsed steps and raw trajectory.

    **Rate metrics come in two denominator families, and the dict does not
    otherwise say which one a value used.**

    - *Per model time*, divided by ``total_duration`` = the sum of per-step
      durations: ``tokens_per_second``, ``median_tokens_per_second``.
    - *Per session wall clock*, divided by ``wall_clock``: ``tool_wait_share``,
      ``tool_calls_per_min``, ``tool_time_fraction``.

    The two bases are not interchangeable: summed step durations double-count a
    parent blocked on an overlapping sub-agent, so they differ from wall-clock
    whenever work overlaps, at times by more than 10x. Wall-clock is the only
    base on which a share cannot exceed 1, which is why the tool shares use it
    and must not be moved. Conversely ``tokens_per_second`` divides the
    cumulative cache-read context (re-counted every turn) by elapsed time and
    is inflated ~(#turns)x, so health verdicts use ``output_tokens_per_sec``
    (output tokens per second of model time) instead.
    """
    if message_rows is None:
        message_rows = build_message_metrics(steps)

    # Duration stats
    from .telemetry_quality import data_quality, field_observed, is_otel_step

    durations = [
        s["duration"] for s in steps if s.get("duration") is not None
        and (not is_otel_step(s) or (s["_otel"].get("kind") == "model_request" and field_observed(s, "duration")))
    ]
    total_duration = sum(durations)
    # token_unusable is the per-field count of steps whose value was not a
    # usable token count; see sum_usable_tokens for why they are dropped.
    total_tokens, token_unusable = sum_usable_tokens(steps)
    from .parser import _optional_token_count

    reasoning_tokens_reported = any(
        _optional_token_count(s["tokens"], "reasoning") is not None for s in steps
    )

    timing = raw.get("timing", {}) if isinstance(raw.get("timing"), dict) else {}
    wall_clock = session_wall_clock_seconds(steps, timing)
    if wall_clock is None:
        wall_clock = total_duration

    result = {
        "total_steps": len(steps),
        "total_duration": round(total_duration, 2),
        "avg_duration": round(total_duration / len(durations), 2) if durations else 0,
        "median_duration": round(statistics.median(durations), 2) if durations else 0,
        "p95_duration": round(_percentile(durations, 0.95), 2) if durations else 0,
        "max_duration": round(max(durations), 2) if durations else 0,
        "wall_clock": wall_clock,
        "reasoning_tokens_reported": reasoning_tokens_reported,
        "token_unusable_steps": token_unusable,
        "input_tokens_unusable_steps": token_unusable.get("input", 0),
        **_compute_token_stats(total_tokens, total_duration, steps, message_rows, raw),
        **_compute_tool_stats(steps, total_tokens["total"], message_rows, float(wall_clock)),
        **_compute_efficiency_stats(steps, message_rows, raw),
        **_compute_command_metrics(steps),
        **_compute_timing_metrics(steps),
        **_compute_plan_metrics(steps),
    }
    # Native mirrors explicitly identify absent clocks. Preserve observed
    # zero-duration records, and keep legacy/custom formats' fallback behavior.
    capabilities = raw.get("_capabilities")
    missing_native_clocks = isinstance(capabilities, dict) and capabilities.get("has_timing") is False
    native_cursor = raw.get("_cursor_format") is True
    if not durations and (missing_native_clocks or native_cursor):
        for key in ("total_duration", "avg_duration", "median_duration", "p95_duration", "max_duration",
                    "tokens_per_second", "median_tokens_per_second"):
            result[key] = None
        if session_wall_clock_seconds(steps, timing) is None:
            result["wall_clock"] = None
    quality = data_quality(steps, raw)
    if quality["source"] == "otlp":
        from .telemetry_quality import public_token_totals

        result["data_quality"] = quality
        result["metric_coverage"] = {key: value["status"] for key, value in quality["fields"].items()}
        result["measurement_basis"] = "captured_operations"
        if session_wall_clock_seconds(steps, timing) is None:
            result["wall_clock"] = None
        result["tokens"] = public_token_totals(total_tokens, quality)
        for key, field in {
            "input_tokens": "tokens.input", "output_tokens": "tokens.output", "cache_read_tokens": "tokens.cache_read",
            "non_cache_tokens": "tokens.input", "total_duration": "duration", "avg_duration": "duration",
            "median_duration": "duration", "p95_duration": "duration", "max_duration": "duration",
            "median_step_tokens": "tokens.total", "p95_step_tokens": "tokens.total",
            "avg_tokens_per_step": "tokens.total", "avg_tokens_per_assistant_step": "tokens.total",
            "tokens_per_second": "tokens.total", "median_tokens_per_second": "tokens.total",
            "tokens_per_tool": "tokens.total", "tool_success": "tool_outcomes", "tool_fail": "tool_outcomes",
            "reasoning_parts": "text", "text_parts": "text", "patch_steps": "text", "has_patch": "text",
        }.items():
            if not quality["fields"][field]["observed"]:
                result[key] = None
        # Ratios need complete compatible populations, rather than dividing
        # subtotals that happened to be observed on different operations.
        for key, fields in {
            "avg_cache_ratio": ("tokens.total", "tokens.cache_read"),
            "non_cache_ratio": ("tokens.total", "tokens.input", "tokens.cache_write"),
            "output_input_ratio": ("tokens.output", "tokens.input"),
        }.items():
            if not all(quality["fields"][field]["status"] == "observed" for field in fields):
                result[key] = None
        if not durations:
            result["tokens_per_second"] = result["median_tokens_per_second"] = None
        if any(quality["fields"][field]["status"] != "observed" for field in ("tokens.total", "duration")):
            result["tokens_per_second"] = None
        observed_totals = [row["tokens_total"] for row in message_rows
                           if row["role"] == "assistant" and row["tokens_total"] is not None]
        if observed_totals:
            result["avg_tokens_per_assistant_step"] = round(sum(observed_totals) / len(observed_totals))
        if not quality["fields"]["tool_inputs"]["observed"]:
            for key in ("command_call_count", "command_failures", "command_success_rate"):
                result[key] = None
        if quality["fields"]["tool_outcomes"]["status"] != "observed":
            result["command_failures"] = result["command_success_rate"] = None
        for key in ("plan_items", "plan_update_count", "plan_completion_ratio"):
            result[key] = None
        for key in ("reasoning_parts", "patch_steps", "has_patch", "patch_lines"):
            result[key] = None
        result["reasoning_tokens_reported"] = quality["fields"]["tokens.reasoning"]["observed"] > 0
    return result


def compute_diagnostic_metrics(
    steps: list[dict],
    trajectory: list[dict],
    *,
    tool_fail: int | None = None,
) -> dict:
    """Sub-agent counts, tool-error chip, and edit precision.

    Pass *tool_fail* from ``compute_metrics`` to avoid a second tool-call scan.
    """
    from .patterns import (
        extract_subagent_sessions,
        compute_subagent_metrics,
    )

    sessions = extract_subagent_sessions(steps, trajectory)
    sa_metrics = compute_subagent_metrics(sessions, steps)
    error_count = (
        int(tool_fail)
        if tool_fail is not None
        else sum(1 for s in steps for tc in s.get("tool_calls", []) if tool_call_failed(tc))
    )

    from agent_trajviz.tool_vocab import WRITE_TOOL_NAMES as edit_tools
    edit_total = 0
    edit_success = 0
    edit_unknown = 0
    from .telemetry_quality import tool_field_observed

    for s in steps:
        for tc in s.get("tool_calls", []):
            if tc.get("tool_name") in edit_tools:
                edit_total += 1
                if not tool_field_observed(tc, "tool_outcomes"):
                    edit_unknown += 1
                elif not tool_call_failed(tc):
                    edit_success += 1

    return {
        "subagent_session_count": len(sessions),
        "subagent_total_steps": sum(s.get("step_count", 0) for s in sa_metrics),
        "subagent_total_tokens": sum(s.get("total_tokens", 0) for s in sa_metrics),
        "classified_error_count": error_count,
        "edit_total": edit_total,
        "edit_success": edit_success,
        "edit_precision": round(edit_success / edit_total * 100, 1) if edit_total and not edit_unknown else None,
    }


def compute_health_verdict(metrics: dict, step_analytics: list[dict]) -> list[dict]:
    """Compute a health verdict with color-coded status for key metrics."""
    quality = metrics.get("data_quality", {})
    if quality.get("source") == "otlp":
        # Reuse the established thresholds only where the required evidence
        # is complete. The temporary numeric defaults never escape here.
        safe = dict(metrics)
        safe.pop("data_quality", None)
        for key in ("tool_success_rate", "tool_fail"):
            if safe.get(key) is None:
                safe[key] = 0
        verdicts = compute_health_verdict(safe, step_analytics)
        dependencies = {
            "Cache Efficiency": ("tokens.total", "tokens.cache_read"),
            "Tool Success": ("tool_outcomes",), "Errors": ("tool_outcomes",),
            "Throughput": ("tokens.output", "duration"),
        }
        for verdict in verdicts:
            fields = dependencies[verdict["metric"]]
            missing = [field for field in fields if quality["fields"][field]["status"] != "observed"]
            if missing or (verdict["metric"] == "Throughput" and metrics.get("output_tokens_per_sec") is None):
                verdict.update(status="unavailable", label="N/A", detail=(
                    "Insufficient captured evidence: " + ", ".join(missing or ["measurable model duration"])
                ))
            else:
                verdict["detail"] += "; captured operations only"
        return verdicts
    verdicts = []

    # Cache efficiency
    avg_cache = metrics.get("avg_cache_ratio")
    cache_unusable = metrics.get("cache_ratio_unusable_steps") or 0
    if not metrics.get("tokens", {}).get("total", 0):
        # No per-step token data (e.g. a format without token usage): a 0% cache
        # ratio here means "unknown", not "poor".
        verdicts.append({"metric": "Cache Efficiency", "status": "good", "label": "N/A", "detail": "No token data"})
    elif not isinstance(avg_cache, (int, float)) or isinstance(avg_cache, bool):
        # Self-inconsistent token records: refuse to rate rather than rate an
        # impossible value "strong", which is what a share above 25,000% used to get.
        detail = (
            f"{cache_unusable} step(s) report more cache-read tokens than the step total"
            if cache_unusable else "Cache share unavailable"
        )
        verdicts.append({"metric": "Cache Efficiency", "status": "warn", "label": "N/A", "detail": detail})
    elif avg_cache >= 60:
        verdicts.append({"metric": "Cache Efficiency", "status": "good", "label": f"{avg_cache}%",
                         "detail": f"Avg cache read {avg_cache}% — strong cache reuse"})
    elif avg_cache >= 30:
        verdicts.append({"metric": "Cache Efficiency", "status": "warn", "label": f"{avg_cache}%",
                         "detail": f"Avg cache read {avg_cache}% — moderate cache reuse"})
    else:
        verdicts.append({"metric": "Cache Efficiency", "status": "bad", "label": f"{avg_cache}%",
                         "detail": f"Avg cache read {avg_cache}% — most input tokens are fresh"})

    # Tool success rate
    tool_rate = metrics.get("tool_success_rate", 0)
    tool_count = metrics.get("tool_call_count", 0)
    if tool_count == 0:
        verdicts.append({"metric": "Tool Success", "status": "good", "label": "N/A", "detail": "No tool calls"})
    elif tool_rate is None:
        verdicts.append({"metric": "Tool Success", "status": "unavailable", "label": "N/A",
                         "detail": "Tool outcomes were not recorded for every call"})
    elif tool_rate >= 95:
        verdicts.append({"metric": "Tool Success", "status": "good", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success across {tool_count} calls"})
    elif tool_rate >= 80:
        verdicts.append({"metric": "Tool Success", "status": "warn", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success — {metrics.get('tool_fail', 0)} failures out of {tool_count} calls"})
    else:
        verdicts.append({"metric": "Tool Success", "status": "bad", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success — high failure rate across {tool_count} calls"})

    # Generation throughput — output tokens per second of model time
    # (assistant step duration minus spawn wait and timed tool waits).
    # NOTE: use output_tokens_per_sec, not tokens_per_second, which divides the
    # cumulative cache-read context (re-counted every turn) by wall time and is
    # inflated ~(#turns)x, making the verdict structurally "good".
    gen_rate = metrics.get("output_tokens_per_sec")
    timed_steps = metrics.get("output_throughput_timed_steps")
    throughput_steps = metrics.get("output_throughput_total_steps")
    incomplete_timing = metrics.get("output_throughput_incomplete", False)
    excl_tools = (metrics.get("output_throughput_tool_wait_seconds") or 0) > 0
    coverage_note = ""
    if (
        incomplete_timing
        and isinstance(timed_steps, int)
        and isinstance(throughput_steps, int)
        and throughput_steps > 0
    ):
        coverage_note = f"; based on {timed_steps}/{throughput_steps} assistant steps with timing"
    if excl_tools:
        coverage_note += "; tool wait excluded from denominator"
    if gen_rate is None:
        detail = "No timing/output-token data"
        if coverage_note:
            detail += coverage_note
        verdicts.append({"metric": "Throughput", "status": "good", "label": "N/A", "detail": detail})
    elif gen_rate >= 50:
        verdicts.append({"metric": "Throughput", "status": "good", "label": f"{gen_rate} gen tok/s",
                         "detail": f"{gen_rate} gen tok/s — strong throughput{coverage_note}"})
    elif gen_rate >= 20:
        verdicts.append({"metric": "Throughput", "status": "warn", "label": f"{gen_rate} gen tok/s",
                         "detail": f"{gen_rate} gen tok/s — moderate throughput{coverage_note}"})
    else:
        verdicts.append({"metric": "Throughput", "status": "bad", "label": f"{gen_rate} gen tok/s",
                         "detail": f"{gen_rate} gen tok/s — low throughput{coverage_note}"})

    # Failed tool calls — tool_fail counts failing tool CALLS (already reflected
    # in Tool Success); label accordingly rather than as "error steps".
    failed_calls = metrics.get("tool_fail", 0)
    if metrics.get("tool_unknown_outcomes", 0):
        status, detail = "unavailable", (
            f"{failed_calls} observed failed call(s); "
            f"{metrics['tool_unknown_outcomes']} tool outcomes were not recorded"
        )
    elif failed_calls == 0:
        status, detail = "good", "No failed tool calls"
    elif failed_calls <= 2:
        status, detail = "warn", f"{failed_calls} failed tool call(s)"
    else:
        status, detail = "bad", f"{failed_calls} failed tool calls — agent may be struggling"
    verdicts.append({"metric": "Errors", "status": status,
                     "label": "N/A" if status == "unavailable" else str(failed_calls), "detail": detail})

    return verdicts


def extract_agent_info(steps: list[dict]) -> tuple[str, str, str]:
    """Return (model_id, provider_id, agent_id) for the session header.

    Model and provider come from the last assistant step that recorded them,
    so a mid-session switch (Pi provider retries, etc.) is what the header
    shows. Agent id is the first non-empty agent field.
    """
    model_id = provider_id = agent_id = ""
    for s in steps:
        if s.get("role") == "assistant" and s.get("model_id"):
            model_id = s["model_id"]
            provider_id = s.get("provider_id", "") or provider_id
            if not agent_id and s.get("agent"):
                agent_id = s["agent"]
    if not agent_id:
        for s in steps:
            if s.get("agent"):
                agent_id = s["agent"]
                break
    return model_id, provider_id, agent_id


def compute_agent_summary(steps: list[dict], raw: dict) -> list[dict]:
    """Compute per-agent summary statistics from parsed steps.

    Returns a list of agent summary dicts sorted by first appearance,
    containing tokens, duration, tool calls, errors, cache efficiency, and the
    agent's generation rate (``output_tokens_per_sec``, from
    :func:`output_throughput` over that agent's assistant steps).
    """
    from collections import defaultdict

    from .parser import cache_read_share
    from .telemetry_quality import data_quality, is_otel_step, public_token_totals

    def included(step: dict) -> bool:
        return step.get("role") == "assistant" or (is_otel_step(step) and step.get("role") == "tool")

    if not steps:
        return []

    # Determine agent ordering by first appearance (assistant steps only)
    agent_order: list[str] = []
    seen: set[str] = set()
    for s in steps:
        if not included(s):
            continue
        agent = effective_agent(s)
        if agent not in seen:
            agent_order.append(agent)
            seen.add(agent)

    # Accumulate per-agent stats
    stats: dict[str, dict] = defaultdict(lambda: {
        "step_count": 0,
        "total_duration_s": 0.0, "tool_call_count": 0, "error_count": 0,
    })
    agent_steps: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        if not included(s):
            continue
        agent = effective_agent(s)
        agent_steps[agent].append(s)
        d = stats[agent]
        d["step_count"] += 1
        dur = s.get("duration")
        if isinstance(dur, (int, float)) and (not is_otel_step(s) or s["_otel"].get("kind") == "model_request"):
            d["total_duration_s"] += dur
        d["tool_call_count"] += s.get("tool_call_count", 0)
        d["error_count"] += s.get("error_count", 0)

    # Build spawning map from native Claude Code child transcripts.
    spawned_by_map: dict[str, str] = {}  # agent_id -> spawned_by_tool_call_id
    cc_sub_agents = raw.get("_claude_code_sub_agents") or raw.get("_codex_sub_agents") or raw.get("_otlp_sub_agents", [])
    if isinstance(cc_sub_agents, list):
        for sa in cc_sub_agents:
            if isinstance(sa, dict) and sa.get("agent_id"):
                spawned_by_map[sa["agent_id"]] = sa.get("spawned_by", "")

    # Build tool_call_id -> step_index map for spawning correlation
    tool_call_step_map: dict[str, int] = {}
    for s in steps:
        for tc in s.get("tool_calls", []):
            tid = tc.get("tool_id", "")
            if tid:
                tool_call_step_map[tid] = s.get("index", 0)

    # Tokens go through the shared helper rather than a local `+=`: the Agents
    # tab reported a negative Input for sessions whose Overview had already
    # switched to "partial", because only the session total applied the domain
    # check. Dropped steps are counted per agent for the same reason they are
    # counted per session — a surface must be able to say the row is partial.
    agent_tokens = {
        agent_id: sum_usable_tokens(
            steps,
            predicate=lambda s, a=agent_id: (
                s.get("role") == "assistant" and effective_agent(s) == a
            ),
        )
        for agent_id in agent_order
    }

    labels = disambiguate_agent_labels(agent_order, steps)
    result = []
    for agent_id in agent_order:
        d = stats[agent_id]
        tok_totals, tok_unusable = agent_tokens[agent_id]
        label = labels.get(agent_id, agent_id if agent_id else "main")
        total_tok = tok_totals["total"]
        cache_read = tok_totals["cache_read"]
        # Same domain rule as the session-level ratio: a share that exceeds 1
        # is not a share. Without this the Agents tab still rendered shares near
        # 10,000% on the very sessions where the Overview had switched to n/a.
        _cache_share = cache_read_share(cache_read, total_tok)
        cache_pct = None if _cache_share is None else round(_cache_share * 100, 1)
        dur = d["total_duration_s"]
        # The session's generation-rate code over this agent's steps, so a
        # single-agent card and the Overview cannot disagree. Dividing every
        # token (cache reads included) by wall-clock instead read 70-120x the
        # generation rate on cache-heavy runs.
        throughput = output_throughput(agent_steps[agent_id])

        # Spawning correlation
        spawned_by_tool = spawned_by_map.get(agent_id, "")
        spawned_by_step = tool_call_step_map.get(spawned_by_tool) if spawned_by_tool else None

        row = {
            "agent_id": agent_id,
            "label": label,
            "step_count": d["step_count"],
            "total_tokens": total_tok,
            "input_tokens": tok_totals["input"],
            "output_tokens": tok_totals["output"],
            "reasoning_tokens": tok_totals["reasoning"],
            "cache_read_tokens": cache_read,
            "cache_write_tokens": tok_totals["cache_write"],
            "token_unusable_steps": tok_unusable,
            "input_tokens_unusable_steps": tok_unusable["input"],
            "total_duration_s": round(dur, 2),
            "tool_call_count": d["tool_call_count"],
            "error_count": d["error_count"],
            "cache_efficiency_pct": cache_pct,
            "output_tokens_per_sec": throughput["rate"],
            "output_throughput_timed_steps": throughput["timed_steps"],
            "spawned_by_step": spawned_by_step,
        }
        capabilities = raw.get("_capabilities")
        missing_native_clocks = isinstance(capabilities, dict) and capabilities.get("has_timing") is False
        if (missing_native_clocks or raw.get("_cursor_format") is True) and not any(
            isinstance(step.get("duration"), (int, float)) for step in agent_steps[agent_id]
        ):
            row["total_duration_s"] = None
        quality = data_quality(agent_steps[agent_id], raw)
        if quality["source"] == "otlp":
            visible = public_token_totals(tok_totals, quality)
            for field, key in {
                "total": "total_tokens", "input": "input_tokens", "output": "output_tokens",
                "reasoning": "reasoning_tokens", "cache_read": "cache_read_tokens", "cache_write": "cache_write_tokens",
            }.items():
                row[key] = visible[field]
            row["data_quality"] = quality
            if not all(quality["fields"][f"tokens.{key}"]["status"] == "observed" for key in ("total", "cache_read")):
                row["cache_efficiency_pct"] = None
            if not quality["fields"]["duration"]["observed"]:
                row["total_duration_s"] = None
            if not quality["fields"]["tool_outcomes"]["observed"]:
                row["error_count"] = None
        result.append(row)
    return result


def generate_agent_insights(agent_summaries: list[dict]) -> list[str]:
    """Generate insight strings for multi-agent patterns."""
    if len(agent_summaries) <= 1:
        return []
    if any(row.get("data_quality", {}).get("source") == "otlp" for row in agent_summaries):
        # These heuristics assume complete transcript token totals and use
        # message count as the error-rate denominator. Imported operations
        # have a different basis; their measured rows remain available.
        return []

    insights: list[str] = []
    total_tokens = sum(a["total_tokens"] for a in agent_summaries)
    if total_tokens <= 0:
        return insights

    for a in agent_summaries:
        share = a["total_tokens"] / total_tokens * 100
        if share > 60:
            insights.append(
                f"Agent {a['label']} consumed {share:.0f}% of total tokens"
            )
        if a["cache_efficiency_pct"] == 0 and a["total_tokens"] > 0 and a["cache_read_tokens"] == 0:
            insights.append(
                f"Agent {a['label']} has 0% cache efficiency — no prompt caching"
            )
        if a["step_count"] > 0 and a["error_count"] / a["step_count"] > 0.10:
            rate = a["error_count"] / a["step_count"] * 100
            insights.append(
                f"Agent {a['label']} has {rate:.0f}% error rate ({a['error_count']} errors in {a['step_count']} steps)"
            )
        tool_share = a["tool_call_count"] / max(1, sum(x["tool_call_count"] for x in agent_summaries)) * 100
        if tool_share > 70 and len(agent_summaries) > 1:
            insights.append(
                f"Agent {a['label']} made {tool_share:.0f}% of all tool calls"
            )

    return insights
