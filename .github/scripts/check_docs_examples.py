"""Run explicitly marked documentation examples against synthetic saved sessions.

A ``<!-- trajviz:run -->`` comment immediately before a Python fence opts it
in. ``<!-- trajviz:run extras=charts -->`` opts in only with ``--extras charts``.
Each page gets its own process, temporary files and Python namespace. Network
access and child processes are disabled while examples run.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit

_MARKER = re.compile(r"^\s*<!--\s*trajviz:run(?:\s+extras=([a-z, ]+))?\s*-->\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_EXTRAS = frozenset({"charts", "llm", "app", "zstd", "all"})


class ExampleError(RuntimeError):
    """A documentation example or its explicit execution marker is invalid."""


@dataclass(frozen=True)
class Example:
    line: int
    code: str
    extras: frozenset[str]


def read_examples(path: Path) -> list[Example]:
    """Read only opted-in Python fences; unmarked fences are never executed."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    examples = []
    pending: tuple[int, frozenset[str]] | None = None
    fence: tuple[str, int, str, int] | None = None
    code: list[str] = []
    for number, line in enumerate(lines, 1):
        if fence is not None:
            char, length, language, start = fence
            if re.fullmatch(rf" {{0,3}}{re.escape(char)}{{{length},}}\s*", line):
                if pending is not None:
                    examples.append(Example(start + 1, "".join(code), pending[1]))
                fence, pending, code = None, None, []
            else:
                code.append(line)
            continue
        marker = _MARKER.fullmatch(line.rstrip("\r\n"))
        if marker:
            if pending is not None:
                raise ExampleError(f"{path}:{number}: execution marker has no Python fence")
            extras = frozenset(item.strip() for item in (marker[1] or "").split(",") if item.strip())
            if extras - _EXTRAS:
                raise ExampleError(f"{path}:{number}: unknown extras: {sorted(extras - _EXTRAS)}")
            pending = number, extras
            continue
        if "<!--" in line and "trajviz:run" in line:
            raise ExampleError(f"{path}:{number}: malformed execution marker")
        match = _FENCE.fullmatch(line.rstrip("\r\n"))
        if match:
            if pending is not None and match[2].strip() != "python":
                raise ExampleError(f"{path}:{number}: executable fences must use python")
            fence = match[1][0], len(match[1]), match[2], number
            continue
        if pending is not None and line.strip():
            raise ExampleError(f"{path}:{number}: execution marker must precede a Python fence")
    if pending is not None or fence is not None:
        raise ExampleError(f"{path}: unterminated fence or execution marker")
    return examples


def _import_package(package_root: Path | None, expected_version: str | None) -> ModuleType:
    if package_root is not None:
        sys.path.insert(0, str(package_root.resolve()))
    module = importlib.import_module("agent_trajviz")
    if package_root is not None and not Path(module.__file__).resolve().is_relative_to(package_root.resolve()):
        raise ExampleError("agent_trajviz was imported outside --package-root")
    if expected_version is not None and module.__version__ != expected_version:
        raise ExampleError(f"expected version {expected_version}, imported {module.__version__}")
    return module


def seed_inputs(directory: Path, tv: ModuleType) -> None:
    """Generate small native sessions and a selectable capture; no recorded data."""
    for filename, sid, output_tokens in (("capture.jsonl", "docs-first", 5),
                                         ("capture-2.jsonl", "docs-second", 8)):
        records = [
            {"type": "user", "uuid": sid + "-user", "sessionId": sid,
             "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": "Inspect parser.py"}},
            {"type": "assistant", "uuid": sid + "-read", "sessionId": sid,
             "timestamp": "2026-01-01T00:00:01Z", "message": {"id": sid + "-message-1", "role": "assistant",
                 "model": "claude-docs-fixture", "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 3},
                 "content": [{"type": "tool_use", "id": sid + "-tool", "name": "Read", "input": {"file_path": "parser.py"}}]}},
            {"type": "user", "uuid": sid + "-result", "sessionId": sid,
             "timestamp": "2026-01-01T00:00:02Z", "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": sid + "-tool", "content": "def parse(): return []", "is_error": False}]}},
            {"type": "assistant", "uuid": sid + "-done", "sessionId": sid,
             "timestamp": "2026-01-01T00:00:03Z", "message": {"id": sid + "-message-2", "role": "assistant",
                 "model": "claude-docs-fixture", "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": output_tokens},
                 "content": [{"type": "text", "text": "Parser inspected."}]}},
        ]
        (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    spans = []
    for index, sid in enumerate(("docs-first", "docs-second"), 1):
        trace, parent = f"{index:032x}", f"{index * 2:016x}"
        def attrs(**values):
            return [{"key": key, "value": {"intValue" if isinstance(value, int) else "stringValue": str(value)}}
                    for key, value in values.items()]
        spans.extend([
            {"traceId": trace, "spanId": parent, "name": "claude_code.interaction",
             "startTimeUnixNano": str(1767225600000000000 + index * 10_000_000_000),
             "endTimeUnixNano": str(1767225604000000000 + index * 10_000_000_000),
             "attributes": attrs(**{"session.id": sid, "user_prompt": "Inspect parser.py"}), "status": {"code": 0}},
            {"traceId": trace, "spanId": f"{index * 2 + 1:016x}", "parentSpanId": parent,
             "name": "claude_code.llm_request",
             "startTimeUnixNano": str(1767225601000000000 + index * 10_000_000_000),
             "endTimeUnixNano": str(1767225603000000000 + index * 10_000_000_000),
             "attributes": attrs(**{"session.id": sid, "model": "claude-docs-fixture", "input_tokens": 12,
                                    "output_tokens": 4, "response.model_output": "Parser inspected.", "stop_reason": "end_turn"}),
             "status": {"code": 0}},
        ])
    traces = {"resourceSpans": [{"resource": {"attributes": attrs(**{"service.name": "claude-code"})},
                                "scopeSpans": [{"scope": {"name": "claude-code"}, "spans": spans}]}]}
    manifest = {"type": "agent-trajviz-otlp-bundle", "schema_version": 1, "profile": "claude_code", "traces": ["traces.json"]}
    with zipfile.ZipFile(directory / "sessions.zip", "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("traces.json", json.dumps(traces))
    run = tv.load(directory / "capture.jsonl")
    labels = {"schema_version": 2, "taxonomy_version": "docs-v1", "model": "docs-fixture",
              "trajectory_file": "capture.jsonl", "steps": [
                  {"index": step["index"], "raw_index": step.get("raw_index", step["index"]), "role": step["role"],
                   "phase": "user" if step["role"] == "user" else "understand",
                   "action": "user_prompt" if step["role"] == "user" else "code_reading",
                   "label_source": "default", "duration_s": step.get("duration"),
                   "tokens_total": step.get("tokens", {}).get("total", 0),
                   "tool_calls": [tool["tool_name"] for tool in step.get("tool_calls", [])],
                   "agent": step.get("agent", "main"), "text_preview": step.get("text_preview", "")}
                  for step in run.steps]}
    (directory / "labels.json").write_text(json.dumps(labels), encoding="utf-8")


def _offline_audit(event: str, _arguments: tuple) -> None:
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"}:
        raise RuntimeError("documentation examples must not use network access or child processes")


def run_page(path: Path, package_root: Path | None, extras: frozenset[str], expected_version: str | None) -> dict:
    """Run one page in an isolated subprocess, as invoked by the CLI."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    sys.addaudithook(_offline_audit)
    tv = _import_package(package_root, expected_version)
    examples = read_examples(path)
    executed = skipped = 0
    with tempfile.TemporaryDirectory(prefix="trajviz-docs-") as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            seed_inputs(Path(temporary), tv)
            namespace = {"__name__": "__docs_example__"}
            for example in examples:
                if example.extras - extras:
                    skipped += 1
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile("\n" * (example.line - 1) + example.code, str(path), "exec"), namespace)
                except Exception as exc:
                    raise ExampleError(f"{path}:{example.line}: {type(exc).__name__}: {exc}") from exc
                executed += 1
        finally:
            os.chdir(previous)
    return {"page": path.name, "executed": executed, "skipped": skipped,
            "version": tv.__version__, "package": str(Path(tv.__file__).resolve())}


class _Ids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, _tag, attributes):
        self.ids.update(value for key, value in attributes if key == "id" and value is not None)


def check_api_reference(path: Path, docs: list[Path], module: ModuleType) -> int:
    """Ensure the facade reference covers its public names and guide links resolve."""
    parser = _Ids()
    parser.feed(path.read_text(encoding="utf-8"))
    missing = sorted(set(module.__all__) - parser.ids)
    if missing:
        raise ExampleError(f"{path}: missing public API anchors: {', '.join(missing)}")
    for doc in docs:
        for target in re.findall(r"\]\(([^\s)]+)\)", doc.read_text(encoding="utf-8")):
            parts = urlsplit(target)
            if not parts.scheme and parts.path.endswith("api/agent_trajviz.html") and parts.fragment:
                if unquote(parts.fragment) not in parser.ids:
                    raise ExampleError(f"{doc}: unknown public API link: {target}")
    return len(module.__all__)


def _extras(raw: str) -> frozenset[str]:
    chosen = frozenset(value.strip() for value in raw.split(",") if value.strip())
    if chosen - _EXTRAS:
        raise ExampleError(f"unknown extras: {sorted(chosen - _EXTRAS)}")
    if "all" in chosen:
        chosen |= _EXTRAS
    if "app" in chosen:
        chosen |= {"charts", "llm"}
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", "--docs-dir", dest="docs_dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--extras", default="")
    parser.add_argument("--api-html", type=Path)
    parser.add_argument("--run-page", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        extras = _extras(args.extras)
        if args.run_page:
            print(json.dumps(run_page(args.run_page.resolve(), args.package_root, extras, args.expected_version)))
            return 0
        docs = sorted(args.docs_dir.resolve().glob("*.md"))
        if not docs or not any(read_examples(path) for path in docs):
            raise ExampleError("no explicitly marked Python examples found")
        results = []
        for path in docs:
            if path.stem == "awe":
                target = _import_package(args.package_root, args.expected_version)
                release = tuple(int(part) for part in re.match(r"(\d+)\.(\d+)", target.__version__).groups())
                if release < (0, 4):
                    results.append({"page": path.name, "executed": 0, "skipped": len(read_examples(path)),
                                    "version": target.__version__, "package": str(Path(target.__file__).resolve())})
                    continue
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "--run-page", str(path), "--extras", args.extras]
            if args.package_root is not None:
                command += ["--package-root", str(args.package_root.resolve())]
            if args.expected_version:
                command += ["--expected-version", args.expected_version]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise ExampleError(result.stderr.strip() or f"{path}: example process failed")
            results.append(json.loads(result.stdout.strip().splitlines()[-1]))
        module = _import_package(args.package_root, args.expected_version)
        coverage = check_api_reference(args.api_html, docs, module) if args.api_html else None
        print(json.dumps({"version": module.__version__, "package": str(Path(module.__file__).resolve()),
                          "pages": len(results), "executed": sum(row["executed"] for row in results),
                          "skipped": sum(row["skipped"] for row in results), "public_api_anchors": coverage,
                          "results": results}))
        return 0
    except (ExampleError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
