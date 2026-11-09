"""``agent-trajviz``: the Insight dashboard, trajectory views as JSON with
``--json``, or a standalone HTML report with ``--report``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__, _serve
from .._cli import configure_logging
from .._extras import require_or_exit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-trajviz", description="TrajViz Insight dashboard, JSON views, and HTML reports",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _serve.add_server_arguments(parser, default_port=7860)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--json",
        metavar="TRAJECTORY",
        help="Write trajectory views as JSON files and print the output directory. "
        "Use -o to set the directory (default: ./<name>.views)",
    )
    mode.add_argument(
        "--report",
        metavar="TRAJECTORY",
        help="Write a standalone HTML report for TRAJECTORY (requires the charts extra)",
    )
    mode.add_argument("--list-sessions", metavar="CAPTURE",
                      help="List the selectable sessions of a saved OTLP capture as JSON and exit")
    parser.add_argument("--format", metavar="FORMAT", help="Require a trajectory format, for example otlp")
    parser.add_argument("--session-id", metavar="ID",
                        help="With --json or --report, select one session from a saved OTLP capture")
    mode.add_argument("--list-views", action="store_true", help="List the view ids --json can write, and exit")
    mode.add_argument(
        "--schemas",
        metavar="DIR",
        help="Write the JSON Schema of every view to DIR as <id>.schema.json, print DIR and exit",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="With --json: the directory to write. With --report: the HTML file or directory "
        "(default: a temp file). The path is printed",
    )
    parser.add_argument(
        "--view",
        action="append",
        metavar="ID",
        help="With --json: write only this view (repeatable; see --list-views)",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Render the HTML report with dark charts and tokens (with --report)",
    )
    parser.add_argument(
        "--inline-plotly",
        action="store_true",
        help="Embed plotly.js in the report so it works offline (with --report; adds several MB)",
    )
    return parser


def _export_json(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ..api import load
    from ..errors import LoadError
    from ..views import VIEW_CATALOG, export_json

    unknown = [vid for vid in args.view or () if vid not in VIEW_CATALOG]
    if unknown:
        parser.error(f"unknown view id(s): {', '.join(unknown)} (see --list-views)")
    out_dir = Path(args.output) if args.output else Path.cwd() / f"{Path(args.json).stem}.views"
    try:
        run = load(args.json, format=args.format, session_id=args.session_id)
        export_json(run, out_dir, only=args.view)
    except (LoadError, OSError, ValueError) as exc:
        print(f"agent-trajviz: {exc}", file=sys.stderr)
        return 1
    print(out_dir)
    return 0


def _write_schemas(directory: Path) -> int:
    import json

    from ..views import view_schemas

    try:
        directory.mkdir(parents=True, exist_ok=True)
        for vid, schema in view_schemas().items():
            (directory / f"{vid}.schema.json").write_text(
                json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"agent-trajviz: {exc}", file=sys.stderr)
        return 1
    print(directory)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    # Installed plugins can add formats and views, so they run before any of them is listed or used.
    from .. import plugins
    from ..errors import PluginError

    try:
        plugins.load_plugins()
    except PluginError as exc:
        print(f"agent-trajviz: {exc}", file=sys.stderr)
        return 1

    from ..api import FORMATS

    if args.format is not None and args.format not in FORMATS:
        parser.error(f"unknown format {args.format!r}; expected one of: {', '.join(FORMATS)}")
    if args.session_id is not None and not args.session_id:
        parser.error("--session-id must be a non-empty string")

    if args.list_views:
        from ..views import VIEW_CATALOG

        for vid, info in VIEW_CATALOG.items():
            print(f"{vid:24} {info['kind']:8} {info['title']}")
        return 0

    if args.schemas:
        return _write_schemas(Path(args.schemas))
    if args.list_sessions:
        import json

        from ..api import list_sessions
        from ..errors import LoadError

        try:
            print(json.dumps(list_sessions(args.list_sessions, format=args.format), indent=2))
        except (LoadError, ValueError, OSError) as exc:
            print(f"agent-trajviz: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.json:
        return _export_json(parser, args)
    if args.view:
        parser.error("--view applies only to --json")

    if args.report:
        require_or_exit("charts", "The HTML report", "agent-trajviz")
        from .report import ReportError, report_from_path

        try:
            report_options = {"dark": args.dark, "inline_plotly": args.inline_plotly}
            if args.format or args.session_id:
                report_options.update(format_hint=args.format or "", session_id=args.session_id)
            path = report_from_path(args.report, args.output, **report_options)
        except (ReportError, OSError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.format or args.session_id:
        parser.error("Use --format with --json, --report or --list-sessions and --session-id with --json or --report")

    require_or_exit("app", "The dashboard", "agent-trajviz")
    try:
        auth = _serve.auth_from_env()
    except ValueError as exc:
        parser.error(str(exc))
    # The public launcher: it drops server-path inputs on an exposed launch
    # and applies the presentation Gradio only accepts at launch().
    from ..dashboard import launch_insight

    launch_insight(host=args.host, port=args.port, share=args.share, auth=auth, max_file_size=args.max_file_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
