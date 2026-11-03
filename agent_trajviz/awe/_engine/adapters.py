"""Per-scaffold adapters: raw trajectory -> NormalizedTrajectory.

Edits are NOT relied upon from trajectories (the authoritative edit-set is the
final patch in agent_trajviz.awe._engine.patches); adapters focus on read-set, search, test execution,
reasoning visibility, and message text for claim detection."""

from __future__ import annotations

import json
import hashlib
import posixpath
import re
import shlex

from . import config
from .evidence import norm_line
from .trajectory import (
    codex_result_header,
    claude_code_exit_code,
    PLAN,
    plan_items,
    NormalizedStep,
    NormalizedTrajectory,
    repo_relative,
    classify_command,
    command_read_targets,
    is_test_command,
    SCHEMA_VERSION,
    SEARCH,
    READ,
    EDIT,
    WRITE,
    RUN_COMMAND,
    DELEGATE,
    OTHER,
    OBSERVED,
    HIDDEN,
)


_APPLY_PATCH_RE = re.compile(r"\*\*\* (?:Update|Add|Delete) File:\s*(.+)")

# Keep complete ordered edit text; normalized line keys are overlap features only.


def _content_lines(text) -> list:
    """Normalized, de-duplicated lines of an edit's old/new content —
    the SAME normalization as the gold/patch changed-line maps (evidence
    norm_line), so trajectory-edit vs gold comparisons share a key space."""
    if not isinstance(text, str) or not text:
        return []
    out, seen = [], set()
    for ln in text.splitlines():
        n = norm_line(ln)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _edit_content_args(old, new) -> dict:
    args = {}
    if isinstance(old, str):
        args["old_content"] = old
    if isinstance(new, str):
        args["new_content"] = new
    ol, nl = _content_lines(old), _content_lines(new)
    if ol:
        args["old_lines"] = ol
    if nl:
        args["new_lines"] = nl
    return args


def _first_text(*values):
    return next((v for v in values if isinstance(v, str)), None)


def _recorded_verification_state(*records):
    typed = [
        record.get("verification_tests")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("verification_tests"), dict)
    ]
    extra = (
        {"verification_tests": typed[0]}
        if typed and all(t == typed[0] for t in typed)
        else ({"verification_tests_conflicts": typed} if typed else {})
    )
    states = [
        record.get("verification_state")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("verification_state"), dict)
    ]
    if not states:
        return extra
    # Do not merge contradictory records into a fabricated single state.
    if any(state != states[0] for state in states[1:]):
        return {**extra, "verification_state_conflicts": states}
    return {**extra, "verification_state": dict(states[0])}


class _Builder:
    def __init__(self, agent: str, instance_id: str):
        self.agent = agent
        self.instance_id = instance_id
        self.steps: list[NormalizedStep] = []
        self._sid = 0
        self._source_ref = {}
        self._timestamp = None

    def at(self, reference, payload):
        self._source_ref = reference
        self._timestamp = payload.get("timestamp", payload.get("created_at"))

    def add(self, **kw) -> None:
        kw.setdefault("source_ref", dict(self._source_ref))
        kw.setdefault("timestamp", self._timestamp)
        self.steps.append(NormalizedStep(step_id=self._sid, **kw))
        self._sid += 1

    def rel(self, path):
        return repo_relative(path, self.instance_id, self.agent)

    def done(self) -> NormalizedTrajectory:
        return NormalizedTrajectory(self.agent, self.instance_id, self.steps)


# ----------------------------------------------------------------- Claude Code

_CC_TOOL = {
    "Read": READ,
    "Edit": EDIT,
    "Write": WRITE,
    "MultiEdit": EDIT,
    "NotebookEdit": EDIT,
    "Grep": SEARCH,
    "Glob": SEARCH,
    "LS": SEARCH,
    "Agent": DELEGATE,
    "Task": DELEGATE,
    "TodoWrite": PLAN,
}


def adapt_claude_code(raw: dict, agent: str, instance_id: str) -> NormalizedTrajectory:
    b = _Builder(agent, instance_id)

    def tool_step(name, inp, call_id, turn, sub):
        inp = inp or {}
        files, args, test = [], {}, False
        ca = _CC_TOOL.get(name, OTHER)
        if ca in (READ, EDIT, WRITE):
            r = b.rel(inp.get("file_path"))
            if r:
                files = [r]
            args = {"path": inp.get("file_path")}
            if ca == READ:
                for k in ("offset", "limit"):
                    if inp.get(k) is not None:
                        args[k] = inp[k]
            if ca in (EDIT, WRITE):
                old, new = inp.get("old_string"), _first_text(inp.get("new_string"), inp.get("content"))
                if name == "MultiEdit" and isinstance(inp.get("edits"), list):
                    old = "\n".join(e.get("old_string", "") for e in inp["edits"] if isinstance(e, dict))
                    new = "\n".join(e.get("new_string", "") for e in inp["edits"] if isinstance(e, dict))
                args.update(_edit_content_args(old, new))
        elif ca == SEARCH:
            args = {"pattern": inp.get("pattern"), "path": inp.get("path")}
        elif ca == DELEGATE:
            args = {"subagent_type": inp.get("subagent_type"), "description": inp.get("description")}
        elif ca == PLAN:
            args = {"items": plan_items(inp.get("todos"))}
        elif name == "Bash":
            cmd = inp.get("command", "") or ""
            ca = classify_command(cmd)
            test = is_test_command(cmd)
            files = [b.rel(p) or p for p in command_read_targets(cmd)]
            args = {"command": cmd}
        b.add(
            turn_id=turn,
            actor="agent",
            type="tool_call",
            canonical_action=ca,
            native_tool=name,
            files=files,
            args={**args, **_recorded_verification_state(inp)},
            test_run=test,
            call_id=call_id,
            subagent_id=sub,
            native_input=inp,
        )

    def walk(traj, sub=None, pointer="/trajectory"):
        for position, m in enumerate(traj or []):
            b.at({"pointer": f"{pointer}/{position}"}, m)
            role = m.get("role")
            turn = m.get("index", 0)
            content = m.get("content")
            if role == "event":
                continue
            if isinstance(content, str):
                b.add(
                    turn_id=turn,
                    actor=("agent" if role == "assistant" else "user"),
                    type="message",
                    text=content,
                    subagent_id=sub,
                )
                continue
            for block_position, blk in enumerate(content or []):
                if not isinstance(blk, dict):
                    continue
                b.at({"pointer": f"{pointer}/{position}/content/{block_position}"}, m)
                t = blk.get("type")
                if t == "text":
                    b.add(
                        turn_id=turn,
                        actor=("agent" if role == "assistant" else "user"),
                        type="message",
                        text=blk.get("text", ""),
                        subagent_id=sub,
                    )
                elif t == "thinking":
                    b.add(
                        turn_id=turn,
                        actor="agent",
                        type="reasoning",
                        text=blk.get("thinking") or blk.get("text", ""),
                        observability=OBSERVED,
                        subagent_id=sub,
                    )
                elif t == "tool_call":
                    tool_step(blk.get("name", ""), blk.get("input"), blk.get("id"), turn, sub)
                elif t == "tool_result":
                    out = str(blk.get("output"))
                    ec = claude_code_exit_code(out)
                    b.add(
                        turn_id=turn,
                        actor="tool",
                        type="tool_result",
                        tool_error=bool(blk.get("is_error")),
                        args={**({"exit_code": ec} if ec is not None else {}), **_recorded_verification_state(blk)},
                        output_excerpt=out,
                        call_id=blk.get("tool_call_id"),
                        subagent_id=sub,
                    )

    walk(raw.get("trajectory", []))
    for index, sa in enumerate(raw.get("sub_agents", [])):
        walk(sa.get("trajectory", []), sub=sa.get("agent_id"), pointer=f"/sub_agents/{index}/trajectory")
    return b.done()


# -------------------------------------------------------------------- OpenCode

_OC_TOOL = {
    "read": READ,
    "edit": EDIT,
    "write": WRITE,
    "grep": SEARCH,
    "glob": SEARCH,
    "list": SEARCH,
    "webfetch": OTHER,
    "task": DELEGATE,
    "todowrite": PLAN,
}


def adapt_opencode(raw: dict, agent: str, instance_id: str) -> NormalizedTrajectory:
    b = _Builder(agent, instance_id)
    for turn, m in enumerate(raw.get("messages", [])):
        role = (m.get("info") or {}).get("role", "")
        for part_position, p in enumerate(m.get("parts", []) or []):
            b.at({"pointer": f"/messages/{turn}/parts/{part_position}"}, m)
            t = p.get("type")
            if t == "text":
                b.add(
                    turn_id=turn,
                    actor=("agent" if role == "assistant" else "user"),
                    type="message",
                    text=p.get("text", ""),
                )
            elif t == "tool":
                tool = p.get("tool", "")
                st = p.get("state") or {}
                inp = st.get("input") or {}
                files, args, test = [], {}, False
                ca = _OC_TOOL.get(tool, OTHER)
                if tool in ("apply_patch", "patch"):
                    # opencode_gpt edits via an apply_patch tool (input.patchText,
                    # '*** Update File:' format) — unmapped, these steps were
                    # canonical OTHER: the long-standing "gpt edits invisible in
                    # the trajectory" capture gap.
                    ca = EDIT
                    body = inp.get("patchText") or inp.get("patch") or ""
                    files = [b.rel(f.strip()) or f.strip() for f in _APPLY_PATCH_RE.findall(body)]
                    args = _apply_patch_args(body, files)
                elif ca in (READ, EDIT, WRITE):
                    r = b.rel(inp.get("filePath") or inp.get("path"))
                    if r:
                        files = [r]
                    args = {"path": inp.get("filePath") or inp.get("path")}
                    if ca == READ:
                        for k in ("offset", "limit"):
                            if inp.get(k) is not None:
                                args[k] = inp[k]
                    if ca in (EDIT, WRITE):
                        args.update(
                            _edit_content_args(
                                _first_text(inp.get("oldString"), inp.get("old_string")),
                                _first_text(inp.get("newString"), inp.get("new_string"), inp.get("content")),
                            )
                        )
                elif ca == SEARCH:
                    args = {"pattern": inp.get("pattern"), "include": inp.get("include")}
                elif ca == PLAN:
                    args = {"items": plan_items(inp.get("todos"))}
                elif tool == "bash":
                    cmd = inp.get("command", "") or ""
                    ca = classify_command(cmd)
                    test = is_test_command(cmd)
                    files = [b.rel(x) or x for x in command_read_targets(cmd)]
                    args = {"command": cmd}
                # Positive error statuses only: an interrupted/'running'/'pending'
                # last call is not a tool ERROR (the old negative list flagged them).
                err = st.get("status") in ("error", "failed")
                b.add(
                    turn_id=turn,
                    actor="agent",
                    type="tool_call",
                    canonical_action=ca,
                    native_tool=tool,
                    files=files,
                    args={**args, **_recorded_verification_state(inp, st)},
                    test_run=test,
                    call_id=p.get("callID"),
                    native_input=inp,
                    status=st.get("status", ""),
                )
                if st.get("status") in ("pending", "running"):
                    continue
                if "output" not in st and not st.get("error") and st.get("status") not in ("error", "failed"):
                    continue
                # The bash tool records the process exit status in
                # state.metadata.exit (v19) — ground truth for run outcomes,
                # previously dropped in favour of text heuristics.
                md = st.get("metadata") or {}
                res_args = {"exit_code": md["exit"]} if isinstance(md.get("exit"), int) else {}
                b.add(
                    turn_id=turn,
                    actor="tool",
                    type="tool_result",
                    tool_error=bool(err),
                    args={**res_args, **_recorded_verification_state(st, md)},
                    status=st.get("status", ""),
                    output_excerpt=str(st.get("output", st.get("error", ""))),
                    call_id=p.get("callID"),
                )
    return b.done()


# ----------------------------------------------------------------------- Codex


def _codex_text(content) -> str:
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


def _apply_patch_content(text: str) -> tuple:
    """(old_text, new_text) from an apply_patch body: '-' lines are the replaced
    content, '+' lines the replacement (headers/markers excluded)."""
    old, new = [], []
    for ln in (text or "").splitlines():
        if ln.startswith("***") or ln.startswith("@@"):
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            new.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"):
            old.append(ln[1:])
    return "\n".join(old), "\n".join(new)


def _apply_patch_sections(text: str) -> list:
    """[(raw_file, old_text, new_text)] per '*** Update/Add File:' section, in
    order of appearance (same order _APPLY_PATCH_RE.findall yields)."""
    sections: list = []
    cur, old, new = None, [], []
    for ln in (text or "").splitlines():
        m = _APPLY_PATCH_RE.search(ln)
        if m:
            if cur is not None:
                sections.append((cur, "\n".join(old), "\n".join(new)))
            cur, old, new = m.group(1).strip(), [], []
            continue
        if ln.startswith("***") or ln.startswith("@@"):
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            new.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"):
            old.append(ln[1:])
    if cur is not None:
        sections.append((cur, "\n".join(old), "\n".join(new)))
    return sections


def _apply_patch_args(text: str, resolved_files: list) -> dict:
    """Edit-content args for an apply_patch body. Single-file patches use the
    flat old_lines/new_lines form; multi-file patches store a PER-FILE map
    (keyed by the resolved path, positional with the section order) — a flat
    union would attribute every file's lines to every file, breaking the
    revert/gold-overlap comparisons."""
    sections = _apply_patch_sections(text)
    if not sections:
        old, new = _apply_patch_content(text)
        return _edit_content_args(old, new)
    if len(sections) == 1:
        _, old, new = sections[0]
        return _edit_content_args(old, new)
    per = {}
    for (_, old, new), rf in zip(sections, resolved_files, strict=False):
        c = _edit_content_args(old, new)
        if c and rf:
            per[rf] = c
    return {"per_file_lines": per} if per else {}


_CWD_PROBE = "__cwd_probe__"


def _command_cwd(cmd: str, b: _Builder, initial: str = "") -> str:
    """The working directory a `cd` inside THIS command establishes ('' = repo
    root). PER-COMMAND only: each Codex exec runs in a fresh shell at the repo
    root unless workdir is explicitly supplied, so cwd must never persist across commands — the target pattern is
    `cd pkg && cat sub/a.py` within one command. Approximations: the LAST cd
    wins and applies to the whole command (the dominant real form is a leading
    cd); absolute targets outside the known repository remain absolute rather
    than being falsely resolved against the repository root."""
    cwd = initial
    for seg in re.split(r"(?:&&|\|\||;)", cmd or ""):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        if not toks or toks[0] != "cd":
            continue
        args = [t for t in toks[1:] if not t.startswith("-")]
        if not args:  # `cd`, `cd -`, `cd -P`: unknown → root
            cwd = ""
            continue
        target = args[0]
        if target.startswith("/"):
            # Distinctive probe suffix survives repo_relative intact, so the
            # sandbox-root case strips cleanly to '' (a bare '/x' suffix did
            # not: repo_relative reduced root+'/x' to 'x', which then leaked
            # into every subsequent path as a phantom directory).
            r = repo_relative(posixpath.normpath(f"{target}/{_CWD_PROBE}"), b.instance_id, b.agent) or ""
            if r.endswith(_CWD_PROBE):
                r = r[: -len(_CWD_PROBE)].rstrip("/")
            cwd = r if r else ""
        elif target.startswith("~"):
            cwd = ""
        else:
            joined = posixpath.normpath(f"{cwd}/{target}" if cwd else target)
            cwd = "" if joined in (".", "..") or joined.startswith("..") else joined
    return cwd


def _resolve_read(b: _Builder, cwd: str, path: str) -> str:
    # An ABSOLUTE raw path never takes the cwd prefix: b.rel() strips the
    # sandbox prefix to a repo-relative string, which would otherwise be
    # indistinguishable from a genuinely relative target and get double-prefixed
    # ('pkg/pkg/a.py').
    if path.startswith("/"):
        return b.rel(path) or path
    r = b.rel(path) or path
    if cwd and r and not r.startswith("/"):
        return posixpath.normpath(f"{cwd}/{r}")
    return r


def adapt_codex(raw_lines: list[dict], agent: str, instance_id: str) -> NormalizedTrajectory:
    b = _Builder(agent, instance_id)
    # Codex yields a shell command after ~1s ("Process running with session ID
    # N") and streams the rest through write_stdin(session_id=N) calls. The
    # verdict of a long test run therefore lands in a LATER chunk. v19: each
    # chunk is merged back into the originating exec_command's result (full
    # text re-excerpted, exit code set when the process ends), so the test
    # call's own result carries its outcome; the write_stdin steps stay as
    # their own call/result pairs for fidelity.
    sessions: dict = {}  # session_id -> (result step index, accumulated body)
    call_name: dict = {}  # call_id -> (tool name, session_id)
    for turn, o in enumerate(raw_lines):
        b.at({"record": turn}, o)
        if o.get("type") != "response_item":
            continue
        pl = o.get("payload") or {}
        pt = pl.get("type")
        if pt == "message":
            role = pl.get("role", "")
            actor = "agent" if role == "assistant" else "user"
            b.add(turn_id=turn, actor=actor, type="message", text=_codex_text(pl.get("content")))
        elif pt == "reasoning":
            summaries = pl.get("summary") or []
            if isinstance(summaries, list):
                for summary in summaries:
                    if isinstance(summary, dict) and isinstance(summary.get("text"), str):
                        b.add(
                            turn_id=turn, actor="agent", type="reasoning", text=summary["text"], observability=OBSERVED
                        )
            b.add(turn_id=turn, actor="agent", type="reasoning", text="[encrypted reasoning]", observability=HIDDEN)
        elif pt in ("function_call", "custom_tool_call"):
            name = pl.get("name", "")
            # custom_tool_call (e.g. apply_patch) carries its body in `input`,
            # function_call in `arguments` — reading only `arguments` left every
            # custom apply_patch with no files and no content (the codex
            # edit-capture gap). Coerce non-string payloads (some emit dicts).
            raw_args = pl.get("arguments") or pl.get("input") or ""
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            try:
                a = json.loads(raw_args or "{}")
            except Exception:
                a = {}
            if not isinstance(a, dict):
                a = {"input": a}
            cmd = a.get("cmd", "")
            patch_files = _APPLY_PATCH_RE.findall(cmd or raw_args)
            if patch_files or "apply_patch" in (name or ""):
                # Codex edits go through apply_patch (shell heredoc or custom
                # tool). A shell-form patch honors a cd in ITS OWN command; the
                # custom-tool form has no shell, so its paths are root-relative.
                initial = (
                    _command_cwd("cd " + shlex.quote(a["workdir"]), b)
                    if isinstance(a, dict) and a.get("workdir")
                    else ""
                )
                cwd = _command_cwd(cmd, b, initial) if cmd else ""
                files = [_resolve_read(b, cwd, f.strip()) for f in patch_files]
                b.add(
                    turn_id=turn,
                    actor="agent",
                    type="tool_call",
                    canonical_action=EDIT,
                    native_tool="apply_patch",
                    files=files,
                    args=_apply_patch_args(cmd or raw_args, files),
                    native_input=a if isinstance(a, dict) and a else {"input": raw_args},
                    call_id=pl.get("call_id"),
                )
            elif name == "exec_command" or "cmd" in a:
                initial = _command_cwd("cd " + shlex.quote(a["workdir"]), b) if a.get("workdir") else ""
                cwd = _command_cwd(cmd, b, initial)
                ca = classify_command(cmd)
                files = [_resolve_read(b, cwd, x) for x in command_read_targets(cmd)]
                b.add(
                    turn_id=turn,
                    actor="agent",
                    type="tool_call",
                    canonical_action=ca,
                    native_tool=name or "exec_command",
                    files=files,
                    args={
                        "command": cmd,
                        **_recorded_verification_state(a),
                        **({"workdir": a["workdir"]} if a.get("workdir") else {}),
                    },
                    native_input=a,
                    test_run=is_test_command(cmd),
                    call_id=pl.get("call_id"),
                )
                call_name[pl.get("call_id")] = ("exec_command", None)
            elif name == "update_plan":
                b.add(
                    turn_id=turn,
                    actor="agent",
                    type="tool_call",
                    canonical_action=PLAN,
                    native_tool=name,
                    native_input=a,
                    args={
                        "items": plan_items(a.get("plan") if isinstance(a, dict) else None),
                        "explanation": (a.get("explanation") if isinstance(a, dict) else None),
                    },
                    call_id=pl.get("call_id"),
                )
            else:
                args = a if isinstance(a, dict) else {}
                b.add(
                    turn_id=turn,
                    actor="agent",
                    type="tool_call",
                    canonical_action=OTHER,
                    native_tool=name,
                    args=args,
                    native_input=args,
                    call_id=pl.get("call_id"),
                )
                if name == "write_stdin":
                    call_name[pl.get("call_id")] = ("write_stdin", args.get("session_id"))
        elif pt in ("function_call_output", "custom_tool_call_output"):
            out = pl.get("output", "")
            exit_code = None
            recorded = _recorded_verification_state(pl)
            if isinstance(out, str):
                # Codex wraps shell results as {"output": ..., "metadata":
                # {"exit_code": N}} — unwrap so the excerpt is the real output and
                # the exit code (the ground-truth success signal) is preserved.
                try:
                    j = json.loads(out)
                    if isinstance(j, dict) and ("output" in j or "metadata" in j):
                        recorded.update(_recorded_verification_state(j, j.get("metadata")))
                        exit_code = (j.get("metadata") or {}).get("exit_code")
                        out = j.get("output", out)
                except Exception:
                    pass
            out = str(out)
            hdr_exit, session, body = codex_result_header(out)
            if exit_code is None and hdr_exit is not None:
                exit_code = hdr_exit
            args = {**({"exit_code": exit_code} if type(exit_code) is int else {}), **recorded}
            kind, sid = call_name.get(pl.get("call_id"), (None, None))
            if kind == "write_stdin" and sid in sessions:
                # continuation chunk: fold into the originating result
                idx, acc = sessions[sid]
                acc = acc + body
                orig = b.steps[idx]
                orig.output_excerpt = acc
                aggregation = orig.args["output_aggregation"]
                aggregation["source_steps"].append(b._sid)
                if type(exit_code) is int:
                    aggregation["completion_step"] = b._sid
                    orig.args = {**(orig.args or {}), "exit_code": exit_code}
                    sessions.pop(sid, None)
                else:
                    sessions[sid] = (idx, acc)
                args["continuation_of"] = orig.call_id
            b.add(
                turn_id=turn, actor="tool", type="tool_result", args=args, output_excerpt=out, call_id=pl.get("call_id")
            )
            if kind == "exec_command" and session is not None and exit_code is None:
                origin = b.steps[-1]
                origin.args["output_aggregation"] = {
                    "kind": "session_chunks",
                    "source_steps": [origin.step_id],
                    "completion_step": None,
                    "initial_exit_code": None,
                }
                sessions[session] = (len(b.steps) - 1, body)
    return b.done()


# ---------------------------------------------------------------- Harbor ATIF

_ATIF_TOOL = {
    "read": READ,
    "read_file": READ,
    "edit": EDIT,
    "multiedit": EDIT,
    "notebookedit": EDIT,
    "write": WRITE,
    "write_file": WRITE,
    "grep": SEARCH,
    "glob": SEARCH,
    "ls": SEARCH,
    "list": SEARCH,
    "search": SEARCH,
    "search_files": SEARCH,
    "agent": DELEGATE,
    "task": DELEGATE,
    "delegate": DELEGATE,
}
_WORKSPACE_ROOT_RE = re.compile(r"`?(/workspace/[^/`\s,]+)")


def _atif_args(value) -> dict:
    """ATIF permits either an object or a JSON string for tool arguments."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def adapt_atif(raw: dict, agent: str, instance_id: str) -> NormalizedTrajectory:
    """Normalize Harbor's Agent Trajectory Interchange Format (ATIF).

    Harbor already combines an assistant turn, its tool calls, and their
    observations in one step.  We split those into the same atomic message /
    reasoning / tool_call / tool_result records used by the native adapters.
    The authoritative edit set still comes from the submitted final patch.
    """
    b = _Builder(agent, instance_id)

    # Harbor tasks normally run under /workspace/<repo>.  Prefer the explicit
    # scaffold cwd, then the task prompt, then the first per-step cwd.  Keeping a
    # single repository root avoids stripping a legitimate package subdirectory.
    roots = []
    extra = (raw.get("agent") or {}).get("extra") or {}
    for cwd in extra.get("cwds", []) or []:
        if isinstance(cwd, str) and cwd.startswith("/"):
            roots.append(cwd.rstrip("/"))
    for step in raw.get("steps", []) or []:
        msg = step.get("message")
        if isinstance(msg, str):
            roots.extend(m.group(1).rstrip("/") for m in _WORKSPACE_ROOT_RE.finditer(msg))
    if not roots:
        for step in raw.get("steps", []) or []:
            cwd = (step.get("extra") or {}).get("cwd")
            if isinstance(cwd, str) and cwd.startswith("/"):
                roots.append(cwd.rstrip("/"))
                break
    repo_root = roots[0] if roots else ""

    def rel(path):
        if not isinstance(path, str) or not path:
            return None
        p = b.rel(path) or path
        if repo_root and (p == repo_root or p.startswith(repo_root + "/")):
            return p[len(repo_root) :].lstrip("/")
        # Bounded fallback for standard Harbor workspaces when a scaffold omits
        # cwd metadata.  Only the first /workspace/<repo>/ pair is stripped.
        if p.startswith("/workspace/"):
            parts = p.split("/", 3)
            return parts[3] if len(parts) == 4 else ""
        return p

    def command_cwd(cmd: str) -> str:
        cwd = ""
        for seg in re.split(r"(?:&&|\|\||;)", cmd or ""):
            try:
                toks = shlex.split(seg)
            except ValueError:
                toks = seg.split()
            if not toks or toks[0] != "cd":
                continue
            targets = [t for t in toks[1:] if not t.startswith("-")]
            if not targets or targets[0].startswith("~"):
                cwd = ""
            elif targets[0].startswith("/"):
                r = rel(targets[0]) or ""
                cwd = "" if r.startswith("/") else r
            else:
                joined = posixpath.normpath(f"{cwd}/{targets[0]}" if cwd else targets[0])
                cwd = "" if joined in (".", "..") or joined.startswith("..") else joined
        return cwd

    def resolve(cwd: str, path: str) -> str:
        if path.startswith("/"):
            return rel(path) or path
        p = rel(path) or path
        return posixpath.normpath(f"{cwd}/{p}") if cwd else p

    for turn, step in enumerate(raw.get("steps", []) or []):
        b.at({"pointer": f"/steps/{turn}"}, step)
        source = step.get("source", "")
        actor = "agent" if source in ("agent", "assistant") else source or "system"
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            b.add(turn_id=turn, actor="agent", type="reasoning", text=reasoning, observability=OBSERVED)
        message = step.get("message")
        if isinstance(message, str) and message:
            b.add(turn_id=turn, actor=actor, type="message", text=message)

        calls = step.get("tool_calls") or []
        call_ids = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("function_name") or call.get("name") or "")
            lname = name.lower()
            inp = _atif_args(call.get("arguments") or call.get("input"))
            call_id = call.get("tool_call_id") or call.get("id")
            call_ids.append(call_id)
            files, args, test = [], {}, False
            ca = _ATIF_TOOL.get(lname, OTHER)

            if lname in ("apply_patch", "patch"):
                body = inp.get("patchText") or inp.get("patch") or inp.get("input") or ""
                ca = EDIT
                files = [resolve("", f.strip()) for f in _APPLY_PATCH_RE.findall(body)]
                args = _apply_patch_args(body, files)
            elif ca in (READ, EDIT, WRITE):
                path = inp.get("file_path") or inp.get("filePath") or inp.get("path") or inp.get("filename")
                r = rel(path)
                if r:
                    files = [r]
                args = {"path": path}
                if ca == READ:
                    for key in ("offset", "limit"):
                        if inp.get(key) is not None:
                            args[key] = inp[key]
                else:
                    old = _first_text(inp.get("old_string"), inp.get("oldString"))
                    new = _first_text(inp.get("new_string"), inp.get("newString"), inp.get("content"))
                    args.update(_edit_content_args(old, new))
            elif ca == SEARCH:
                args = {
                    "pattern": inp.get("pattern") or inp.get("query"),
                    "path": inp.get("path"),
                    "include": inp.get("include"),
                }
            elif ca == DELEGATE:
                args = {
                    "subagent_type": inp.get("subagent_type"),
                    "description": inp.get("description") or inp.get("prompt"),
                }
            elif lname in ("bash", "exec_command", "shell"):
                cmd = inp.get("command") or inp.get("cmd") or ""
                cwd = command_cwd(cmd)
                patch_files = _APPLY_PATCH_RE.findall(cmd)
                if patch_files:
                    ca = EDIT
                    files = [resolve(cwd, f.strip()) for f in patch_files]
                    args = {"command": cmd, **_apply_patch_args(cmd, files)}
                else:
                    ca = classify_command(cmd)
                    test = is_test_command(cmd)
                    files = [resolve(cwd, p) for p in command_read_targets(cmd)]
                    args = {"command": cmd}

            b.add(
                turn_id=turn,
                actor="agent",
                type="tool_call",
                canonical_action=ca,
                native_tool=name,
                files=files,
                args={**args, **_recorded_verification_state(inp, call)},
                native_input=inp,
                status=call.get("status", ""),
                test_run=test,
                call_id=call_id,
            )

        observation = step.get("observation")
        if isinstance(observation, dict) and isinstance(observation.get("results"), list):
            results = observation["results"]
        elif observation is not None:
            results = [{"source_call_id": call_ids[0] if len(call_ids) == 1 else None, "content": observation}]
        else:
            results = []
        for result in results:
            if not isinstance(result, dict):
                result = {"content": result}
            rextra = result.get("extra") or {}
            err = bool(
                result.get("is_error")
                or result.get("error")
                or rextra.get("tool_result_is_error")
                or result.get("status") in ("error", "failed")
            )
            exit_code = result.get("exit_code")
            if not isinstance(exit_code, int):
                meta = (rextra.get("tool_result_metadata") or {}).get("tool_use_result") or {}
                exit_code = meta.get("exit_code")
            out = result.get("content", result.get("output", ""))
            b.add(
                turn_id=turn,
                actor="tool",
                type="tool_result",
                tool_error=err,
                args={
                    **({"exit_code": exit_code} if type(exit_code) is int else {}),
                    **_recorded_verification_state(result, rextra),
                },
                output_excerpt=str(out),
                call_id=result.get("source_call_id") or result.get("tool_call_id"),
            )
    return b.done()


# --------------------------------------------------------------------- Pi v3

_PI_TOOL = {
    "read": READ,
    "read_file": READ,
    "edit": EDIT,
    "write": WRITE,
    "grep": SEARCH,
    "glob": SEARCH,
    "search": SEARCH,
    "bash": RUN_COMMAND,
    "shell": RUN_COMMAND,
}


def adapt_pi(lines: list[dict], agent: str, instance_id: str) -> NormalizedTrajectory:
    """Normalize Pi's v3 event-stream transcript written by Harbor as pi.txt."""
    b = _Builder(agent, instance_id)
    session = next((event for event in lines if event.get("type") == "session"), {})
    repo_root = str(session.get("cwd") or "").rstrip("/")
    turn = 0

    def rel(path):
        if not isinstance(path, str) or not path:
            return None
        if repo_root and (path == repo_root or path.startswith(repo_root + "/")):
            return path[len(repo_root) :].lstrip("/")
        if path.startswith("/workspace/"):
            parts = path.split("/", 3)
            return parts[3] if len(parts) == 4 else ""
        return b.rel(path) or path

    def command_cwd(cmd: str) -> str:
        cwd = ""
        for seg in re.split(r"(?:&&|\|\||;)", cmd or ""):
            try:
                toks = shlex.split(seg)
            except ValueError:
                toks = seg.split()
            if not toks or toks[0] != "cd":
                continue
            targets = [tok for tok in toks[1:] if not tok.startswith("-")]
            if not targets or targets[0].startswith("~"):
                cwd = ""
            elif targets[0].startswith("/"):
                value = rel(targets[0]) or ""
                cwd = "" if value.startswith("/") else value
            else:
                joined = posixpath.normpath(f"{cwd}/{targets[0]}" if cwd else targets[0])
                cwd = "" if joined in (".", "..") or joined.startswith("..") else joined
        return cwd

    def resolve(cwd: str, path: str) -> str:
        value = rel(path) or path
        if path.startswith("/"):
            return value
        return posixpath.normpath(f"{cwd}/{value}") if cwd else value

    def add_tool(block: dict) -> None:
        name = str(block.get("name") or "")
        lname = name.lower()
        inp = block.get("arguments") or block.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        call_id = block.get("id") or block.get("toolCallId")
        ca = _PI_TOOL.get(lname, OTHER)
        files, args, test = [], {}, False
        if lname in ("bash", "shell"):
            cmd = inp.get("command") or inp.get("cmd") or ""
            cwd = command_cwd(cmd)
            patch_files = _APPLY_PATCH_RE.findall(cmd)
            if patch_files:
                ca = EDIT
                files = [resolve(cwd, path.strip()) for path in patch_files]
                args = {"command": cmd, **_apply_patch_args(cmd, files)}
            else:
                ca = classify_command(cmd)
                test = is_test_command(cmd)
                files = [resolve(cwd, path) for path in command_read_targets(cmd)]
                args = {"command": cmd}
        elif ca in (READ, EDIT, WRITE):
            path = inp.get("file_path") or inp.get("path") or inp.get("filename")
            value = rel(path)
            files = [value] if value else []
            args = {"path": path}
            if ca in (EDIT, WRITE):
                args.update(
                    _edit_content_args(inp.get("old_string"), _first_text(inp.get("new_string"), inp.get("content")))
                )
        elif ca == SEARCH:
            args = {"pattern": inp.get("pattern") or inp.get("query"), "path": inp.get("path")}
        b.add(
            turn_id=turn,
            actor="agent",
            type="tool_call",
            canonical_action=ca,
            native_tool=name,
            files=files,
            args={**args, **_recorded_verification_state(inp, block)},
            native_input=inp,
            test_run=test,
            call_id=call_id,
        )

    for position, event in enumerate(lines):
        b.at({"record": position}, event)
        etype = event.get("type")
        if etype == "turn_start":
            turn += 1
            continue
        if etype == "message_end":
            message = event.get("message") or {}
            role = message.get("role", "")
            if role not in ("assistant", "user"):
                continue
            actor = "agent" if role == "assistant" else "user"
            content = message.get("content") or []
            if isinstance(content, str):
                b.add(turn_id=turn, actor=actor, type="message", text=content)
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking" and actor == "agent":
                    b.add(
                        turn_id=turn,
                        actor="agent",
                        type="reasoning",
                        text=block.get("thinking") or block.get("text") or "",
                        observability=OBSERVED,
                    )
                elif btype == "text":
                    b.add(turn_id=turn, actor=actor, type="message", text=block.get("text") or "")
                elif btype in ("toolCall", "tool_call") and actor == "agent":
                    add_tool(block)
        elif etype == "tool_execution_end":
            result = event.get("result") or {}
            content = result.get("content", "") if isinstance(result, dict) else result
            if isinstance(content, list):
                output = "\n".join(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content
                )
            else:
                output = str(content)
            b.add(
                turn_id=turn,
                actor="tool",
                type="tool_result",
                tool_error=bool(event.get("isError") or (result.get("isError") if isinstance(result, dict) else False)),
                args=_recorded_verification_state(event, result),
                output_excerpt=output,
                call_id=event.get("toolCallId"),
            )
    return b.done()


# ----------------------------------------------------------------- dispatch

_NORMTRAJ_CACHE = config.PROJECT_DATA / "cache" / "normtraj"


def load_normalized(agent: str, instance_id: str) -> NormalizedTrajectory | None:
    """Parse + DISK-CACHE the normalized trajectory. The parse is the slow step in a
    full report run (judge/arbiter/dossier all call this), so cache it like the
    trajectory signals. Entries embed trajectory.SCHEMA_VERSION: a cache written by
    an older adapter/heuristic is recomputed, not silently served (dataclass
    defaults would otherwise mask missing fields forever)."""
    cache = _NORMTRAJ_CACHE / agent / f"{instance_id}.json"
    from .provenance import trajectory_fingerprint

    source = trajectory_fingerprint(agent, instance_id)
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            valid_schema = d.pop("schema_version", None) == SCHEMA_VERSION
            valid_source = d.pop("source_fingerprint", None) == source
            if valid_schema and valid_source:
                return NormalizedTrajectory.from_dict(d)
        except Exception:
            pass
    traj = _parse_normalized(agent, instance_id)
    if traj is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, "source_fingerprint": source, **traj.to_dict()}), encoding="utf-8"
            )
        except Exception:
            pass
    return traj


def _parse_normalized(agent: str, instance_id: str) -> NormalizedTrajectory | None:
    path = config.TRAJECTORY_DIR / agent / f"{instance_id}.json"
    jsonl = config.TRAJECTORY_DIR / agent / f"{instance_id}.jsonl"

    def done(traj, source_path, body):
        traj.source = {
            "path": str(source_path),
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "record_index": "nonempty JSONL records or JSON pointer",
        }
        return traj

    if agent == "codex" or (jsonl.exists() and not path.exists()):
        if not jsonl.exists():
            return None
        body = jsonl.read_bytes()
        lines = [json.loads(l) for l in body.decode(errors="replace").splitlines() if l.strip()]
        if lines and lines[0].get("type") == "session":
            return done(adapt_pi(lines, agent, instance_id), jsonl, body)
        return done(adapt_codex(lines, agent, instance_id), jsonl, body)
    if not path.exists():
        return None
    body = path.read_bytes()
    raw = json.loads(body.decode(errors="replace"))
    if str(raw.get("schema_version", "")).startswith("ATIF-"):
        return done(adapt_atif(raw, agent, instance_id), path, body)
    if agent.startswith("opencode"):
        return done(adapt_opencode(raw, agent, instance_id), path, body)
    if agent == "claude_code" or raw.get("format") == "ccsession-trajectory":
        return done(adapt_claude_code(raw, agent, instance_id), path, body)
    # fallback by shape
    if "messages" in raw:
        return done(adapt_opencode(raw, agent, instance_id), path, body)
    return done(adapt_claude_code(raw, agent, instance_id), path, body)
