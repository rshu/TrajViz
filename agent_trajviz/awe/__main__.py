"""Command-line import, export and isolated AWE operation runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from collections.abc import Sequence

from .. import __version__
from . import EvaluationSettings, available_operations, load_run, run_operation, write_report


def _json_file(value):
    if value is None:
        return None
    return json.loads(Path(value).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-trajviz-awe", description="Import saved AWE runs or execute the bundled evaluation engine")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect JSON, ZIP or a saved run directory")
    inspect.add_argument("source")
    inspect.add_argument("--output", help="Write the complete versioned JSON envelope")
    export = commands.add_parser("export", help="Export original artifacts, JSON or an offline HTML report")
    export.add_argument("source")
    export.add_argument("--output", required=True)
    commands.add_parser("operations", help="List supported operations and availability")
    run = commands.add_parser("run", help="Execute an allowlisted operation in an isolated job")
    run.add_argument("operation")
    run.add_argument("--reference-root", help="Benchmark inputs; required for dataset analysis operations")
    run.add_argument("--project-data")
    run.add_argument("--output-root")
    run.add_argument("--agents", default="")
    run.add_argument("--agent")
    run.add_argument("--instance-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--benchmark-profile", help="Benchmark profile JSON file")
    run.add_argument("--benchmark-name", default="")
    run.add_argument("--execution-context", help="Execution context JSON file")
    run.add_argument("--judge-model", default="")
    run.add_argument("--base-url", default="")
    run.add_argument("--api-key-env", default="AWE_API_KEY", help="Environment variable holding the session key")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=2400)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--allow-network", action="store_true", help="Explicitly enable configured model requests for this operation")
    run.add_argument("--payload", help="Operation-specific JSON request file")
    args = parser.parse_args(argv)
    try:
        if args.command in {"inspect", "export"}:
            saved = load_run(args.source)
            if args.command == "inspect":
                if args.output:
                    print(saved.write(args.output))
                else:
                    print(json.dumps({"source": saved.source, "cases": len(saved.cases),
                                      "artifacts": sorted(saved.artifacts), "warnings": saved.warnings,
                                      "manifest": saved.manifest}, ensure_ascii=False, indent=2))
            else:
                suffix = Path(args.output).suffix.lower()
                output = (saved.export(args.output) if suffix == ".zip" else
                          write_report(saved, args.output) if suffix in {".html", ".htm"} else saved.write(args.output))
                print(output)
            return 0
        settings = EvaluationSettings(reference_root=getattr(args, "reference_root", ""),
                    **({} if args.command == "operations" else {
                        "project_data": args.project_data, "output_root": args.output_root,
                        "agents": tuple(value.strip() for value in args.agents.split(",") if value.strip()),
                        "benchmark_profile": _json_file(args.benchmark_profile), "benchmark_name": args.benchmark_name,
                        "execution_context": _json_file(args.execution_context), "judge_model": args.judge_model,
                        "base_url": args.base_url, "api_key": os.environ.get(args.api_key_env, ""),
                        "temperature": args.temperature, "max_tokens": args.max_tokens,
                        "timeout": args.timeout, "allow_network": args.allow_network}))
        if args.command == "operations":
            print(json.dumps(available_operations(settings), ensure_ascii=False, indent=2))
            return 0
        result = run_operation(args.operation, settings, agent=args.agent, instance_id=args.instance_id,
                               limit=args.limit, payload=_json_file(args.payload))
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.success else 2
    except (ValueError, OSError, TypeError) as exc:
        # Import errors are controlled validation messages; credentials never
        # enter the CLI argument list or these saved-run validation exceptions.
        parser.exit(2, f"AWE request failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
