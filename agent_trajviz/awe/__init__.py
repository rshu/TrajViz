"""Import, inspect, export and execute Agent Workflow Evaluation (AWE) analyses.

Loaded-session inspection and saved results need only Python's standard library.
Benchmark operations use the bundled engine in an isolated worker process.
Model requests occur only through explicitly enabled network operations.
"""
from __future__ import annotations

import html
from pathlib import Path

from .runtime import EvaluationSettings, OperationResult, available_operations, run_operation
from .inspection import inspect_trajectory
from .saved import SavedRun, load_run

__all__ = ["EvaluationSettings", "OperationResult", "SavedRun", "available_operations",
           "inspect_trajectory", "load_run", "run_operation", "write_report"]


def write_report(run: SavedRun, dest: str | Path) -> Path:
    """Write a self-contained AWE HTML report without executing imported HTML."""
    from . import render
    from .saved import _atomic_output

    run._check_integrity()
    path = run._check_output(dest)
    sections = ["<h1>AWE run</h1>", render.provenance_html(run)]
    for index, case in enumerate(render.all_cases(run)):
        sections.append(f'<section id="case-{index}">{render.case_html(case)}</section>')
        for kind in ("Attribution", "Action/result episodes"):
            graph = render.case_graph(run, case, kind)
            sections.append(render.graph_html(graph, "", run=run, namespace=f"case-{index}:{kind}"))
            for node in (graph or {}).get("nodes", []):
                sections.append(render.node_html(graph, str(node.get("id", "")), run=run))
    sections.extend([render.symptoms_html(run), render.recommendations_html(run), render.rsi_html(run)])
    for name in render.report_options(run):
        sections.append(f"<h2>{html.escape(name)}</h2>{render.report_html(run, name)}")
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>AWE run · TrajViz</title><style>'
                'body{font:15px system-ui,sans-serif;line-height:1.5;color:#172033;background:#fff;'
                'max-width:1200px;margin:auto;padding:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;'
                'background:#f5f7fb;padding:16px;border-radius:8px}table{border-collapse:collapse;display:block;'
                'overflow:auto}td,th{padding:8px;border:1px solid #dce2eb}svg{max-width:100%;height:auto}'
                'section{margin-top:32px}details{margin:10px 0}summary{cursor:pointer}'
                '</style><body>' + "\n".join(sections) + '</body></html>')
    with _atomic_output(path) as stream:
        stream.write(document.encode("utf-8"))
    return path
