"""Codex CLI JSONL → internal trajectory."""

import ast
import json
import re
import shlex

from ..shell_cmd import _shell_segments
from .common import _iso_to_epoch_ms
from .codex_files import agent_path as _codex_agent_path
from .codex_files import own_codex_events, parent_thread_id, session_meta


def _codex_tool_status(output: object) -> str:
    """Read classic results and code-mode text blocks without inspecting stdout.

    Recent rollouts wrap tool results in ``input_text`` blocks. A code-mode
    invocation may report several shell results; any nonzero exit fails that
    invocation. Structured exits take precedence over incidental error wording.
    """
    pending = [(output, 0)]
    texts: list[str] = []
    exits: list[int] = []
    explicit_error = False
    while pending:
        value, depth = pending.pop()
        if depth > 8:
            continue
        if isinstance(value, str):
            texts.append(value)
            explicit_error |= bool(re.search(
                r"(?im)^(?:script failed\b|error executing code:|failed to spawn code-mode host\b)",
                value[:200],
            ))
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                continue
            if isinstance(decoded, (dict, list)):
                pending.append((decoded, depth + 1))
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, dict):
            metadata = value.get("metadata")
            for candidate in (value.get("exit_code"),
                              metadata.get("exit_code") if isinstance(metadata, dict) else None):
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    exits.append(candidate)
            explicit_error |= value.get("isError") is True
            if value.get("type") in ("input_text", "output_text", "text"):
                pending.append((value.get("text"), depth + 1))
            if isinstance(value.get("content"), list):
                pending.append((value["content"], depth + 1))
    if explicit_error or any(exits):
        return "error"
    if exits:
        return "success"
    for text in texts:
        if re.search(r"(?i)\b(?:error:|traceback \(most recent call last\))", text[:200]):
            return "error"
        if any(int(code) != 0 for code in re.findall(r"(?i)exited with code\s+(-?\d+)", text)):
            return "error"
    return "success"


def _codex_spawn_info(meta: dict) -> dict:
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawned = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    return spawned if isinstance(spawned, dict) else {}


def _codex_declared_context_window(events: list[dict]) -> int | None:
    """First positive ``model_context_window`` the export declares, if any.

    Codex states its window on ``task_started`` (``payload.model_context_window``)
    and repeats it on every ``token_count`` (``payload.info.model_context_window``).
    No Codex message carries a model id, so this declaration is the only
    evidence of the real window size — without it every Codex run fell back to
    the 128k default while actually running in 258,400.
    """
    for event in events:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        info = payload.get("info")
        for value in (
            payload.get("model_context_window"),
            info.get("model_context_window") if isinstance(info, dict) else None,
        ):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
    return None


def _convert_codex_to_internal(events: list[dict], *, source_path: str = "",
                               child_event_lists: dict[str, dict] | None = None,
                               child_warnings: list[str] | None = None) -> dict:
    """Convert Codex CLI JSONL events into the TrajViz internal format.

    Codex emits newline-delimited JSON with event types:
    - session_meta: session ID, cwd, model, version
    - turn_context: turn metadata
    - response_item: messages (user/assistant/developer), function_call, function_call_output, reasoning
    - event_msg: task_started, task_complete, token_count, agent_message

    We group these into "messages" matching the OpenCode internal format:
    each assistant turn = one message with parts (text, reasoning, tool calls).
    """
    events, history_warnings = own_codex_events(events)
    # The first owning header is authoritative; inherited histories can contain
    # another thread's header and must never replace the child's identity.
    meta = session_meta(events)
    model = (meta.get("model") if isinstance(meta.get("model"), str) else "") or next((
        event["payload"]["model"] for event in events
        if isinstance(event, dict) and event.get("type") == "turn_context"
        and isinstance(event.get("payload"), dict) and isinstance(event["payload"].get("model"), str)
    ), "")
    owner_id = meta.get("id", "")
    owner_parent = parent_thread_id(meta)
    agent_path = _codex_agent_path(meta)
    spawned = _codex_spawn_info(meta)
    depth = spawned.get("depth", 1 if owner_parent else 0) if isinstance(spawned, dict) else 0
    declared_window = _codex_declared_context_window(events)

    # Group events into assistant turns.
    # Pattern: user message → (reasoning → assistant text → function_calls → function_call_outputs)* → task_complete
    messages: list[dict] = []
    pending_tool_calls: dict[str, dict] = {}  # call_id -> function_call payload
    current_parts: list[dict] = []
    current_role = None
    current_timestamp = None
    current_tokens: dict | None = None
    current_raw_index: int | None = None
    current_model = model
    previous_cumulative_usage: dict | None = None
    previous_usage_snapshot: tuple | None = None

    usage_fields = (
        "total_tokens", "input_tokens", "output_tokens",
        "reasoning_output_tokens", "cached_input_tokens",
    )

    def _usage_value(usage: dict, field: str) -> int | float:
        value = usage.get(field, 0)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    def _flush_message():
        nonlocal current_parts, current_role, current_timestamp, current_tokens, current_raw_index
        if current_parts and current_role:
            info_block = {
                "role": current_role,
                "time": {"created": current_timestamp or 0},
                "modelID": current_model,
                "providerID": meta.get("model_provider", "openai"),
                "sessionID": owner_id, "parentSessionID": owner_parent,
                "sessionDepth": depth, "isSubAgent": bool(owner_parent),
                "agent": agent_path if owner_parent else "", "sourcePath": source_path,
                "rawIndex": current_raw_index if current_raw_index is not None else 0,
            }
            if current_tokens:
                info_block["tokens"] = current_tokens
            messages.append({"info": info_block, "parts": current_parts})
        current_parts = []
        current_role = None
        current_timestamp = None
        current_tokens = None
        current_raw_index = None

    for raw_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "turn_context" and isinstance(event.get("payload"), dict):
            candidate_model = event["payload"].get("model")
            if isinstance(candidate_model, str) and candidate_model:
                current_model = candidate_model
        ordinal = event.get("ordinal")
        event_index = ordinal if isinstance(ordinal, int) and not isinstance(ordinal, bool) else raw_index
        # Codex records timestamps as ISO-8601 strings; the internal contract is
        # epoch milliseconds, so convert here (parse_steps and every timing
        # consumer discard non-numeric timestamps).
        ts = _iso_to_epoch_ms(event.get("timestamp"))

        if etype == "response_item":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            item_type = payload.get("type", "")
            role = payload.get("role", "")

            if item_type == "message":
                # Role change → flush previous message
                if role != current_role and current_parts:
                    _flush_message()

                current_role = role
                if current_raw_index is None:
                    current_raw_index = event_index
                if not current_timestamp:
                    current_timestamp = ts

                for content in (payload.get("content") or []):
                    if not isinstance(content, dict):
                        continue
                    ctype = content.get("type", "")
                    text = content.get("text", "")
                    if ctype == "output_text" or ctype == "input_text":
                        current_parts.append({"type": "text", "text": text})

            elif item_type == "reasoning":
                # If we already have tool call parts, this reasoning starts a new turn
                has_tool_parts = any(p.get("type") in ("tool_call", "tool") for p in current_parts)
                if has_tool_parts and current_role == "assistant":
                    _flush_message()
                current_role = current_role or "assistant"
                if current_raw_index is None:
                    current_raw_index = event_index
                if not current_timestamp:
                    current_timestamp = ts
                summary = payload.get("summary", [])
                summary_text = " ".join(
                    s.get("text", "") for s in summary if isinstance(s, dict)
                ) if isinstance(summary, list) else ""
                current_parts.append({"type": "reasoning", "text": summary_text})

            elif item_type in ("function_call", "custom_tool_call"):
                if current_role != "assistant" and current_parts:
                    _flush_message()
                current_role = "assistant"
                if current_raw_index is None:
                    current_raw_index = event_index
                if not current_timestamp:
                    current_timestamp = ts

                call_id = payload.get("call_id", "")
                name = payload.get("name", "exec_command")
                args_str = payload.get("arguments") or payload.get("input") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                if not isinstance(args, dict):
                    args = {"raw": args_str}

                # Determine tool name and build normalized input
                cmd = args.get("cmd") or args.get("command") or ""
                # The classic `shell` tool records argv (["bash", "-lc", "ls"]);
                # every downstream consumer treats a command as a shell string.
                if isinstance(cmd, list) and all(isinstance(x, str) for x in cmd):
                    cmd = shlex.join(cmd)
                elif not isinstance(cmd, str):
                    cmd = ""
                if not cmd and isinstance(args_str, str):
                    if name == "exec":
                        commands = _extract_codex_exec_commands(args_str)
                        cmd = " ; ".join(commands) if commands else args_str
                    elif name == "apply_patch":
                        cmd = args_str
                tool_name = _classify_codex_command(name, cmd)
                normalized_input = _build_codex_tool_input(tool_name, cmd, args, name)

                pending_tool_calls[call_id] = {
                    "name": name,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input": normalized_input,
                    "cmd": cmd,
                }

            elif item_type in ("function_call_output", "custom_tool_call_output"):
                # An output can arrive after task_complete flushed the turn
                # (role/timestamp reset to None); restore them so the final
                # flush's role guard does not silently drop this part.
                current_role = current_role or "assistant"
                if current_raw_index is None:
                    current_raw_index = event_index
                current_timestamp = current_timestamp or ts
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                tc = pending_tool_calls.pop(call_id, {})

                status = _codex_tool_status(output)

                current_parts.append({
                    "type": "tool_call" if tc else "tool",
                    "tool_name": tc.get("tool_name", "Bash"),
                    "tool_id": call_id,
                    "status": status,
                    "input": tc.get("input", {}),
                    "output": output,
                })

        elif etype == "event_msg":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            msg_type = payload.get("type", "")
            if msg_type == "token_count":
                # last_token_usage is the per-response delta. Codex can repeat an
                # identical notification, so use the cumulative + last snapshots
                # together to de-duplicate it. Cumulative counters alone are not a
                # safe key: parallel agents can report the same cumulative value
                # with different, legitimate last-usage deltas.
                usage = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
                last_usage = usage.get("last_token_usage")
                cumulative = usage.get("total_token_usage")
                tu = None
                cumulative_snapshot = (
                    tuple(_usage_value(cumulative, field) for field in usage_fields)
                    if isinstance(cumulative, dict) else None
                )
                last_snapshot = (
                    tuple(_usage_value(last_usage, field) for field in usage_fields)
                    if isinstance(last_usage, dict) else None
                )
                snapshot = (cumulative_snapshot, last_snapshot)
                if snapshot != (None, None):
                    if snapshot == previous_usage_snapshot:
                        continue
                    previous_usage_snapshot = snapshot

                if isinstance(last_usage, dict):
                    tu = last_usage
                elif isinstance(cumulative, dict):
                    if previous_cumulative_usage is None or any(
                        _usage_value(cumulative, field)
                        < _usage_value(previous_cumulative_usage, field)
                        for field in usage_fields
                    ):
                        tu = cumulative
                    else:
                        tu = {
                            field: _usage_value(cumulative, field)
                            - _usage_value(previous_cumulative_usage, field)
                            for field in usage_fields
                        }

                if isinstance(cumulative, dict):
                    previous_cumulative_usage = {
                        field: _usage_value(cumulative, field) for field in usage_fields
                    }

                if isinstance(tu, dict):
                    token_delta = {
                        "total": _usage_value(tu, "total_tokens"),
                        "input": _usage_value(tu, "input_tokens"),
                        "output": _usage_value(tu, "output_tokens"),
                        "reasoning": _usage_value(tu, "reasoning_output_tokens"),
                        "cache": {"read": _usage_value(tu, "cached_input_tokens"), "write": 0},
                    }
                    if not any(token_delta[field] for field in ("total", "input", "output", "reasoning")) \
                            and not token_delta["cache"]["read"]:
                        continue
                    # The prompt of THIS request (Codex input includes cached
                    # tokens). Occupancy needs the largest single request in the
                    # step, not the sum the usage totals rightly keep.
                    request_prompt = token_delta["input"]
                    request_cached = token_delta["cache"]["read"]
                    if current_tokens is None:
                        current_tokens = token_delta
                        current_tokens["prompt_peak"] = request_prompt
                        current_tokens["prompt_peak_cache_read"] = request_cached
                    else:
                        if request_prompt > current_tokens.get("prompt_peak", 0):
                            current_tokens["prompt_peak"] = request_prompt
                            current_tokens["prompt_peak_cache_read"] = request_cached
                        # A displayed step may contain several model responses
                        # (for example commentary followed by a tool call). Each
                        # response has its own last_token_usage delta, so retain
                        # all of them instead of replacing the earlier usage.
                        for field in ("total", "input", "output", "reasoning"):
                            current_tokens[field] += token_delta[field]
                        current_tokens["cache"]["read"] += token_delta["cache"]["read"]
            elif msg_type == "task_complete":
                _flush_message()
                # Codex's own statement that the turn ended normally. Every
                # "did this run stop normally?" reader keys off the last assistant
                # step's finish, so record it as `stop`, the value the other
                # formats use. Only the turn's final assistant message is marked;
                # a rollout cut off before task_complete keeps an empty finish.
                # The event's timestamp is also the turn's only recorded end:
                # without it that message would run on to the next turn's start
                # and book the idle time between turns as its inference. A stamp
                # earlier than the message itself is no end at all.
                for prior in reversed(messages):
                    prior_info = prior.get("info") or {}
                    if prior_info.get("role") == "user":
                        break
                    if prior_info.get("role") == "assistant":
                        prior_info.setdefault("finish", "stop")
                        turn_time = prior_info["time"]
                        created = turn_time.get("created")
                        if (isinstance(ts, int) and isinstance(created, (int, float))
                                and ts >= created):
                            turn_time.setdefault("completed", ts)
                        break

    # Drain tool calls that never received a function_call_output (session
    # interrupted/truncated mid-command) so the final — often most diagnostic —
    # invocation is not silently dropped from the timeline.
    if pending_tool_calls:
        current_role = current_role or "assistant"
        for call_id, tc in pending_tool_calls.items():
            current_parts.append({
                "type": "tool_call",
                "tool_name": tc.get("tool_name", "Bash"),
                "tool_id": call_id,
                "status": "error",  # interrupted: call never produced an output
                "input": tc.get("input", {}),
                "output": "",
            })
        pending_tool_calls.clear()

    # Flush any remaining parts
    _flush_message()

    # Approximate each remaining completion as the next message's start so
    # per-step durations exist (Codex is single-session; parse_steps backfills
    # a final message left open from the session end timestamp). A completion
    # task_complete already recorded is never replaced: the next turn's start
    # would put the idle time between turns inside the earlier turn's step.
    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"].setdefault("completed", nxt_t)

    first_ts_iso = events[0].get("timestamp") if events and isinstance(events[0], dict) else None
    last_ts_iso = events[-1].get("timestamp") if events and isinstance(events[-1], dict) else None
    directory = meta.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3)
        if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages
    )

    # Build metadata
    info = {
        "id": owner_id,
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": "",
        "version": meta.get("cli_version", ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    result = {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": owner_id,
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1],
            "agent": "codex",
            "model": model,
            "context_window_limit": declared_window,
            "source": "codex",
            "model_provider": meta.get("model_provider", "openai"),
            "originator": meta.get("originator", "Codex CLI"),
            "server_version": meta.get("cli_version", ""),
            "timestamp_utc": first_ts_iso or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": first_ts_iso or "",
            "finished_at": last_ts_iso or "",
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_codex_format": True,
        "_load_warnings": list(child_warnings or []) + history_warnings,
    }
    if child_event_lists:
        _merge_codex_children(result, child_event_lists, owner_id)
    return result


def _merge_codex_children(result: dict, children: dict[str, dict], root_id: str) -> None:
    """Merge verified child operations once; keep each owning session distinct."""
    messages = result["messages"]
    agents = []
    sessions = [{"id": root_id, "source_path": next((
        message["info"].get("sourcePath", "") for message in messages
    ), "")}]
    for identity, info in sorted(children.items()):
        child = _convert_codex_to_internal(info["events"], source_path=info.get("source_path", ""))
        result["_load_warnings"].extend(child.get("_load_warnings", []))
        meta = session_meta(info["events"])
        parent = info.get("parent_thread_id") or parent_thread_id(meta)
        if not child["messages"]:
            result["_load_warnings"].append(f"Codex child {identity} contained no own operations.")
            continue
        label = _codex_agent_path(meta) or identity
        spawned = _codex_spawn_info(meta)
        depth = spawned.get("depth") if isinstance(spawned, dict) else None
        if not isinstance(depth, int) or isinstance(depth, bool):
            depth = 1
        for message in child["messages"]:
            message["info"].update(sessionID=identity, parentSessionID=parent,
                                   sessionDepth=depth, agent=label, isSubAgent=True)
        # Retain this thread's own final bound before global interleaving;
        # the parent's later completion is not the child's duration.
        final_time = child["messages"][-1]["info"].get("time", {})
        child_end = _iso_to_epoch_ms(child.get("timing", {}).get("finished_at"))
        created = final_time.get("created")
        if isinstance(created, (int, float)) and isinstance(child_end, int) and child_end >= created:
            final_time.setdefault("completed", child_end)
        messages.extend(child["messages"])
        agents.append({"agent_id": identity, "agent_path": label,
                       "parent_agent_id": "" if parent == root_id else parent,
                       "session_id": identity, "parent_session_id": parent, "session_depth": depth,
                       "source_path": info.get("source_path", ""), "model": child["metadata"].get("model", ""),
                       "spawned_by": ""})
        sessions.append({"id": identity, "parent_thread_id": parent,
                         "source_path": info.get("source_path", ""), "sha256": info.get("sha256", ""),
                         "history_start_ordinal": meta.get("subagent_history_start_ordinal")})
    # Explicit tool-result identifiers correlate a logical spawn with its child.
    # Canonical task names are unique only within their verified parent thread.
    for message in messages:
        caller = message["info"].get("sessionID")
        for part in message.get("parts", []):
            if str(part.get("tool_name", "")).split(".")[-1] != "spawn_agent":
                continue
            output = part.get("output")
            try:
                output = json.loads(output) if isinstance(output, str) else output
            except ValueError:
                continue
            if not isinstance(output, dict):
                continue
            target = output.get("agent_id") or output.get("thread_id") or output.get("task_name")
            matches = [agent for agent in agents if agent["parent_session_id"] == caller
                       and target in (agent["agent_id"], agent["agent_path"])]
            if len(matches) == 1:
                agent = matches[0]
                agent["spawned_by"] = part.get("tool_id", "")
                part.setdefault("state", {}).setdefault("metadata", {}).update(
                    sessionId=agent["agent_id"], parentSessionId=caller,
                )
    messages.sort(key=lambda message: (
        message["info"].get("time", {}).get("created", 0),
        message["info"].get("sessionID", ""), message["info"].get("rawIndex", 0),
    ))
    result["_codex_sub_agents"] = agents
    result["_codex_sessions"] = sessions
    result["metadata"]["sub_agent_count"] = len(agents)
    result["token_usage"]["total_tokens"] = sum((message["info"].get("tokens") or {}).get("total", 0)
                                               for message in messages)
    result["_load_warnings"] = list(dict.fromkeys(result["_load_warnings"]))

def _extract_codex_exec_commands(source: str) -> list[str]:
    """Extract literal quoted ``cmd`` values without evaluating code-mode input."""
    if not isinstance(source, str):
        return []
    pattern = re.compile(r"""(?:["']?cmd["']?)\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")
    commands = []
    for match in pattern.finditer(source):
        literal = match.group(1)
        try:
            command = json.loads(literal) if literal.startswith('"') else ast.literal_eval(literal)
        except (ValueError, SyntaxError, RecursionError):
            continue
        if isinstance(command, str):
            commands.append(command)
    return commands


def _classify_codex_command(func_name: str, cmd: str) -> str:
    """Map a Codex exec_command to a TrajViz tool name.

    Codex uses exec_command for everything; we infer the intent from the command.
    """
    func_lower = str(func_name).lower().strip()
    cmd_lower = str(cmd).lower().strip()

    if func_lower == "apply_patch" or "tools.apply_patch" in cmd_lower:
        return "Write"
    if not cmd_lower and func_lower not in ("exec", "exec_command"):
        # Preserve non-shell Codex tools (for example spawn_agent or wait)
        # instead of flattening every call into an empty Bash command.
        return str(func_name) or "Bash"

    # Direct cat redirection writes stdin; it must not enter the Read branch.
    if cmd_lower.startswith(("cat >", "cat >>")):
        return "Write"

    # File reading
    if any(cmd_lower.startswith(p) for p in ["cat ", "head ", "tail ", "sed -n", "less "]):
        return "Read"
    if cmd_lower.startswith("rg ") or cmd_lower.startswith("grep "):
        return "Grep"
    if cmd_lower.startswith("find ") or cmd_lower.startswith("ls "):
        return "Glob"

    # File writing
    for pattern in ["cat >", "cat >>", "tee ", "echo >", "echo >>",
                     "sed -i", "patch ", "git apply"]:
        if pattern in cmd_lower:
            return "Write"

    # Test, git, python, and all other shell commands are Bash
    return "Bash"


_RG_SEARCH_VALUE_OPTIONS = frozenset({
    "-A", "-B", "-C", "-m", "-g", "-t", "-T", "-j", "--after-context", "--before-context", "--context",
    "--max-count", "--glob", "--iglob", "--type", "--type-not", "--type-add", "--type-clear", "--threads",
    "--encoding", "--max-columns", "--max-depth", "--max-filesize", "--sort", "--sortr", "--ignore-file",
    "--include", "--exclude", "--exclude-dir", "--label", "--color", "--colors", "--pre", "--pre-glob",
})
_GREP_SEARCH_VALUE_OPTIONS = frozenset({
    "-A", "-B", "-C", "-m", "-D", "-d", "--after-context", "--before-context", "--context", "--max-count",
    "--include", "--exclude", "--exclude-dir", "--label", "--devices", "--directories",
})


def _command_operands(parts: list[str], value_options: frozenset[str]) -> list[str]:
    """Positional operands of one shell stage, excluding flags and redirects."""
    operands = []
    index, options = 1, True
    while index < len(parts):
        token = parts[index]
        if re.fullmatch(r"[0-9]*[<>]+", token):
            index += 2
        elif re.match(r"^[0-9]*[<>]", token):
            index += 1
        elif options and token == "--":
            options = False
            index += 1
        elif options and token.startswith("-") and token != "-":
            option, separator, _ = token.partition("=")
            index += 2 if option in value_options and not separator else 1
        else:
            operands.append(token)
            index += 1
    return operands


def _search_input(parts: list[str]) -> tuple[str, str]:
    """Keep search patterns/scopes separate from option values and later pipes."""
    operands = []
    name = parts[0].rsplit("/", 1)[-1] if parts else ""
    value_options = _RG_SEARCH_VALUE_OPTIONS if name in {"rg", "ripgrep"} else _GREP_SEARCH_VALUE_OPTIONS
    pattern = ""
    has_pattern = False
    index, options = 1, True
    while index < len(parts):
        token = parts[index]
        option, separator, value = token.partition("=")
        if options and option in {"-e", "--regexp", "-f", "--file"}:
            if not separator and index + 1 < len(parts):
                value = parts[index + 1]
            if not has_pattern:
                pattern = value
            has_pattern = True
            index += 1 if separator else 2
        elif options and token.startswith(("-e", "-f")) and len(token) > 2:
            if not has_pattern:
                pattern = token[2:]
            has_pattern = True
            index += 1
        elif re.fullmatch(r"[0-9]*[<>]+", token):
            index += 2
        elif re.match(r"^[0-9]*[<>]", token):
            index += 1
        elif options and token == "--":
            options = False
            index += 1
        elif options and token.startswith("-") and token != "-":
            index += 2 if option in value_options and not separator else 1
        else:
            operands.append(token)
            index += 1
    if not has_pattern and operands:
        pattern = operands.pop(0)
    return pattern, operands[-1] if operands else ""


def _build_codex_tool_input(
    tool_name: str,
    cmd: str,
    raw_args: dict,
    func_name: str = "",
) -> dict:
    """Build a normalized input dict for Codex commands.

    Maps the raw Codex exec_command args into the format that
    canonical.py expects for each tool type:
    - Read: {"file_path": "..."}
    - Write: {"file_path": "..."}
    - Grep/Glob: {"pattern": "...", "path": "..."}
    - Bash: {"command": "..."}
    """
    raw_text = raw_args.get("raw", "") if isinstance(raw_args, dict) else ""

    if tool_name in ("Read",):
        segments = _shell_segments(cmd)
        parts = segments[0] if segments else []
        reader = parts[0].rsplit("/", 1)[-1] if parts else ""
        values = frozenset({"-n", "--lines", "-c", "--bytes"}) if reader in {"head", "tail"} else frozenset()
        if reader == "sed":
            values = frozenset({"-e", "--expression", "-f", "--file"})
        operands = _command_operands(parts, values)
        if reader == "sed" and not any(
            option.partition("=")[0] in values or option.startswith(("-e", "-f"))
            for option in parts[1:] if option.startswith("-")
        ):
            operands = operands[1:]  # the first positional operand is sed's program
        return {"file_path": operands[-1] if operands else "", "command": cmd}

    elif tool_name in ("Write",):
        # Extract file path from: sed -i, cat >, patch, git apply
        patch_match = re.search(
            r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
            cmd,
            flags=re.MULTILINE,
        )
        if patch_match:
            file_path = patch_match.group(1).strip()
        else:
            segments = _shell_segments(cmd)
            parts = segments[0] if segments else []
            file_path = ""
            for index, token in enumerate(parts):
                if token in {">", ">>", "1>", "1>>"} and index + 1 < len(parts):
                    file_path = parts[index + 1]
                    break
                match = re.match(r"^(?:1)?>>?(.+)$", token)
                if match:
                    file_path = match[1]
                    break
            if not file_path:
                for p in reversed(parts):
                    if not p.startswith("-") and ("." in p or "/" in p):
                        file_path = p
                        break
        result = {"file_path": file_path, "command": cmd}
        if str(func_name).lower() == "apply_patch" or cmd.startswith("*** Begin Patch"):
            result["patch"] = raw_text or cmd
        return result

    elif tool_name in ("Grep",):
        segments = _shell_segments(cmd)
        pattern, path = _search_input(segments[0] if segments else [])
        return {"pattern": pattern, "path": path, "command": cmd}

    elif tool_name in ("Glob",):
        segments = _shell_segments(cmd)
        parts = segments[0] if segments else []
        if parts and parts[0].rsplit("/", 1)[-1] == "find":
            rest = parts[1:]
            while rest and rest[0] in {"-H", "-L", "-P"}:
                rest = rest[1:]
            path = rest[0] if rest and not rest[0].startswith("-") else ""
        else:
            operands = _command_operands(parts, frozenset({"-I", "--ignore", "--hide", "--sort", "--format", "--time-style", "--block-size", "-T", "--tabsize", "-w", "--width"}))
            path = operands[-1] if operands else ""
        return {"path": path, "command": cmd}

    else:
        # Bash — pass cmd as "command" so canonical.py can parse it
        if not cmd and isinstance(raw_args, dict) and raw_args:
            return dict(raw_args)
        result = {"command": cmd}
        if raw_text and raw_text != cmd:
            result["raw_input"] = raw_text
        return result
