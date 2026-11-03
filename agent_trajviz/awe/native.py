"""Native session projection for the bundled evaluation engine.

Source references identify TrajViz workflow rows; they are not invented native
record offsets. Observed results, threads and source fingerprints are retained.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import shlex
from pathlib import Path
from typing import Any

ADAPTER_VERSION = "trajviz-native-v1"
_INSTALLED_FOR = None


@lru_cache(maxsize=1)
def adapter_code_hash() -> str:
    """Bind projected evidence to the installed parser implementation."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _canonical(agent: str, instance_id: str) -> Path | None:
    from agent_trajviz.awe._engine import config
    config.safe_id(agent)
    config.safe_id(instance_id)
    base = config.TRAJECTORY_DIR / agent
    candidates = [base / (instance_id + suffix) for suffix in (".json", ".jsonl")]
    if agent == "codex":
        candidates.reverse()
    return next((path for path in candidates if path.is_file()), None)


def _files(path: Path) -> list[Path]:
    """Cover sibling child logs as well as the canonical parent file."""
    files = [path]
    # Claude/Codex export trees use a session-stem directory; harness exports
    # use siblings. Restrict discovery to this session tree, never all runs.
    children = path.with_suffix("")
    if children.is_dir():
        files += sorted(p for p in children.rglob("*") if p.is_file() and p.suffix in {".jsonl", ".json"})
    subagents = path.parent / "subagents"
    if subagents.is_dir():
        files += sorted(p for p in subagents.rglob("session*.jsonl") if p.is_file())
    if path.name.startswith("session.v"):
        files += sorted(p for p in path.parent.parent.rglob(path.name) if p != path and p.is_file())
    return list(dict.fromkeys(files))


def _fingerprints(path: Path) -> list[dict[str, Any]]:
    return [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in _files(path)]


def _native_candidate(path: Path):
    """Retain upstream's established archive adapters, project native sessions."""
    from agent_trajviz import load
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict) and (isinstance(raw.get("trajectory"), list)
                                      or str(raw.get("schema_version", "")).startswith("ATIF-")):
            return None
    run = load(path)
    if run.format == "otlp":
        raise ValueError("AWE requires native transcript and benchmark evidence")
    if run.format == "claude_code" and path.suffix != ".jsonl":
        return None
    if run.format not in {"claude_code", "codex", "cursor", "opencode", "pi", "dsh", "codearts", "icode"}:
        return None
    return run


def project_native(run, agent: str, instance_id: str):
    """Project recorded TrajViz rows into AWE's canonical evidence ontology."""
    from agent_trajviz.awe._engine import adapters as ad
    from agent_trajviz.awe._engine import trajectory as t

    result = t.NormalizedTrajectory(agent, instance_id)
    source_path = Path(run.source) if run.source else None
    source_files = _fingerprints(source_path) if source_path and source_path.is_file() else []
    result.source = {"path": str(source_path.resolve()) if source_path else None,
                     "sha256": run.sha256, "adapter": ADAPTER_VERSION,
                     "native_format": run.format, "files": source_files, "adapter_sha256": adapter_code_hash(),
                     "record_index": "TrajViz workflow row and part; native result offsets may be unknown"}
    names = {"read": t.READ, "read_file": t.READ, "readfile": t.READ,
             "edit": t.EDIT, "edit_file": t.EDIT, "replace_string": t.EDIT,
             "multiedit": t.EDIT, "notebookedit": t.EDIT,
             "write": t.WRITE, "write_file": t.WRITE, "writefile": t.WRITE,
             "grep": t.SEARCH, "glob": t.SEARCH, "ls": t.SEARCH, "search": t.SEARCH,
             "agent": t.DELEGATE, "task": t.DELEGATE, "spawn_agent": t.DELEGATE,
             "todowrite": t.PLAN, "update_plan": t.PLAN}

    def add(step: dict, part_index: int, **kw):
        source = {"workflow_step": step["index"], "part": part_index,
                  "source": step.get("source_path") or run.source,
                  "native_record": step.get("raw_index"), "kind": "trajviz_workflow_projection"}
        sub = step.get("agent") or step.get("session_id") if step.get("is_sub_agent") else None
        result.steps.append(t.NormalizedStep(step_id=len(result.steps), turn_id=step["index"],
                           subagent_id=sub, source_ref=source,
                           timestamp=step.get("time_created_ms"), **kw))

    for step in run.steps:
        actor = {"assistant": "agent", "user": "user", "system": "system", "tool": "tool"}.get(step["role"], "system")
        parts = step.get("parts") or []
        # Tool calls are already represented in parts for some formats. Other
        # formats retain them separately; add only calls not represented there.
        calls = [p for p in parts if p.get("type") in {"tool_call", "tool"}]
        represented = {str(c.get("tool_id") or c.get("id") or "") for c in calls}
        parts = [*parts, *(c for c in step.get("tool_calls", [])
                          if str(c.get("tool_id") or c.get("id") or "") not in represented)]
        for part_index, part in enumerate(parts):
            kind = part.get("type")
            text = part.get("text") or part.get("thinking") or ""
            hidden = bool(part.get("redacted") or part.get("encrypted") or part.get("observability") == "hidden")
            if kind in {"text", "reasoning", "thinking"} and isinstance(text, str) and (text or hidden):
                add(step, part_index, actor=actor, type="reasoning" if kind != "text" else "message", text=text,
                    observability="hidden" if hidden else part.get("observability", "observed"))
                continue
            if kind not in {"tool", "tool_call"} and not part.get("tool_name"):
                continue
            native = str(part.get("tool_name") or part.get("name") or "")
            name = native.rsplit(".", 1)[-1].lower()
            inp = part.get("input") or {}
            if not isinstance(inp, dict):
                inp = {"input": inp}
            args, files = {}, []
            command = inp.get("command") or inp.get("cmd")
            action = names.get(name, t.OTHER)
            if isinstance(command, str) and command:
                action = t.classify_command(command)
                args["command"] = command
                builder = ad._Builder(agent, instance_id)
                workdir = inp.get("workdir") or inp.get("cwd") or ""
                root = str(step.get("cwd") or "").rstrip("/")
                if root and isinstance(workdir, str) and workdir.startswith(root + "/"):
                    workdir = workdir[len(root) + 1:]
                initial = ad._command_cwd("cd " + shlex.quote(str(workdir)), builder) if workdir and hasattr(ad, "_command_cwd") else ""
                cwd = ad._command_cwd(command, builder, initial) if hasattr(ad, "_command_cwd") else initial
                files = [ad._resolve_read(builder, cwd, p) if hasattr(ad, "_resolve_read")
                         else t.repo_relative(p, instance_id, agent) or p for p in t.command_read_targets(command)]
            path = inp.get("file_path") or inp.get("filePath") or inp.get("path") or inp.get("filename")
            if isinstance(path, str) and path and action in {t.READ, t.EDIT, t.WRITE}:
                relative = t.repo_relative(path, instance_id, agent) or path
                cwd = str(step.get("cwd") or "").rstrip("/")
                if cwd and relative.startswith(cwd + "/"):
                    relative = relative[len(cwd) + 1:]
                files = [relative]
                args["path"] = path
                for key in ("offset", "limit"):
                    if key in inp:
                        args[key] = inp[key]
            if action in {t.EDIT, t.WRITE}:
                args.update(ad._edit_content_args(inp.get("old_string", inp.get("oldText")),
                            inp.get("new_string", inp.get("newText", inp.get("content")))))
            if "apply_patch" in name:
                body = command or inp.get("patch") or inp.get("input") or ""
                if isinstance(body, str):
                    patch_files = ad._APPLY_PATCH_RE.findall(body)
                    if patch_files:
                        action, files = t.EDIT, patch_files
                        args.update(ad._apply_patch_args(body, files))
            if action == t.SEARCH:
                args.update(pattern=inp.get("pattern", inp.get("query")), path=path)
            if action == t.PLAN:
                args["items"] = t.plan_items(inp.get("todos", inp.get("plan", inp.get("items"))))
            if action == t.DELEGATE:
                args.update(subagent_type=inp.get("subagent_type"), description=inp.get("description"))
            metadata = part.get("metadata") if isinstance(part.get("metadata"), dict) else {}
            args.update(ad._recorded_verification_state(inp, metadata, part))
            call_id = part.get("tool_id") or part.get("id") or None
            add(step, part_index, actor="agent", type="tool_call", canonical_action=action,
                native_tool=native, args=args, files=files, test_run=bool(isinstance(command, str) and t.is_test_command(command)),
                call_id=call_id, native_input=inp, status=str(part.get("status") or ""))
            # Empty output is still an observed result only when the transcript
            # records completion. Pending calls never gain an invented result.
            status = str(part.get("status") or "").lower()
            output = part.get("output")
            observed = output is not None and (bool(output) or status in {"success", "completed", "error", "failed"})
            if not observed:
                continue
            output = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            result_args = ad._recorded_verification_state(metadata, part)
            exit_code = metadata.get("exit_code", metadata.get("exitCode"))
            if run.format == "codex":
                try:
                    wrapper = json.loads(output)
                except (ValueError, TypeError):
                    wrapper = None
                if isinstance(wrapper, dict) and isinstance(wrapper.get("metadata"), dict):
                    recorded = wrapper["metadata"].get("exit_code")
                    if type(recorded) is int:
                        exit_code = recorded
                    if isinstance(wrapper.get("output"), str):
                        output = wrapper["output"]
                if type(exit_code) is not int:
                    exit_code, _session_id, output = t.codex_result_header(output)
            if type(exit_code) is not int:
                exit_code = t.claude_code_exit_code(output)
            if type(exit_code) is int:
                result_args["exit_code"] = exit_code
            add(step, part_index, actor="tool", type="tool_result", native_tool=native,
                call_id=call_id, output_excerpt=output, args=result_args,
                tool_error=bool(part.get("error") or status in {"error", "failed"}), status=status)
    return result


def install_native_adapters(reference_root=None) -> None:
    """Extend only the bundled engine; safe across isolated worker module copies."""
    global _INSTALLED_FOR
    from agent_trajviz.awe._engine import adapters

    if _INSTALLED_FOR is adapters:
        return
    original_parse = adapters._parse_normalized

    def parse(agent, instance_id):
        path = _canonical(agent, instance_id)
        if path is not None:
            run = _native_candidate(path)
            if run is not None:
                return project_native(run, agent, instance_id)
        return original_parse(agent, instance_id)

    adapters._parse_normalized = parse
    _INSTALLED_FOR = adapters
