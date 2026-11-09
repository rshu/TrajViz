"""Safe, dependency-free views of saved AWE evidence and run artifacts.

These views display recorded evidence. They never execute an imported HTML file
or turn a graph relation into a claim of causality.
"""
from __future__ import annotations

import hashlib
import html
import csv
import io
import json
import math
import re
import textwrap
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ..i18n import attrs, text
from ..i18n.catalog_awe import ZH_CN
from .summary import attribution_summary_html

_STYLE = """<style>
.awe-view{font-size:14px;line-height:1.55;color:var(--body-text-color,#192331);min-width:0}
.awe-view *{box-sizing:border-box}.awe-view h3{font-size:17px;margin:14px 0 8px}
.awe-view h4{font-size:14px;margin:12px 0 6px}.awe-muted{color:var(--body-text-color-subdued,#64748b)}
.awe-panel{border:1px solid var(--border-color-primary,#dbe3ed);border-radius:10px;padding:12px;margin:10px 0;min-width:0}
.awe-view summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}
.awe-view pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-height:550px;overflow:auto;padding:10px;background:var(--background-fill-secondary,#f4f7fb);border-radius:6px;font:12px/1.55 ui-monospace,monospace}
.awe-view table{border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed}
.awe-view table.awe-score-matrix{min-width:760px}
.awe-view td,.awe-view th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid var(--border-color-primary,#e1e7ef);overflow-wrap:anywhere}
.awe-view .awe-report-scroll{width:100%;max-width:100%;overflow-x:auto;overscroll-behavior-x:contain}
.awe-view .awe-report-scroll table{width:max-content;min-width:100%;table-layout:auto}
.awe-view .awe-report-scroll th,.awe-view .awe-report-scroll td{white-space:nowrap;overflow-wrap:normal;word-break:normal}
.awe-view .awe-report-cell-details{width:clamp(220px,28vw,380px);white-space:normal}
.awe-view .awe-report-cell-details summary{white-space:normal;overflow-wrap:anywhere;font-weight:400}
.awe-view .awe-report-cell-details pre{max-height:260px;margin:6px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.awe-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:10px}
.awe-node{display:block;text-decoration:none!important;color:inherit!important;border:1px solid #bfccdc;border-left:4px solid #7c66e8;border-radius:8px;padding:10px;margin:6px 0;overflow-wrap:anywhere}
.awe-node.awe-selected{background:var(--background-fill-secondary,#f3f0ff);border-color:#7956d9}
.awe-tag{display:inline-block;font-size:11px;padding:2px 7px;margin-right:5px;border-radius:4px;background:#eaf0f8;color:#344560}
.awe-edges{font-size:12px;overflow-wrap:anywhere}.awe-warning{border-left:4px solid #b7791f;padding:8px 12px;background:var(--background-fill-secondary,#fff8eb)}
.awe-graph-scroll{max-width:100%;max-height:650px;overflow:auto;overscroll-behavior-x:contain;border:1px solid #dbe3ed;border-radius:10px;margin:10px 0;background:#fbfcff}
.awe-graph-scroll svg{display:block;max-width:none;font-family:system-ui,sans-serif}
@media(max-width:650px){.awe-view td,.awe-view th{padding:6px}.awe-panel{padding:10px}.awe-view table{font-size:11px}}
</style>"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _pre(value: Any) -> str:
    return "<pre>" + _escape(_text(value)) + "</pre>"


def _label(value: str) -> str:
    """Translate an authored title without marking arbitrary artifact titles."""
    return text(value) if value in ZH_CN else _escape(value)


def _details(title: str, value: Any, *, opened: bool = False) -> str:
    if value is None:
        return ""
    return (f"<details class='awe-panel'{' open' if opened else ''}><summary>{_label(title)}</summary>"
            + _pre(value) + "</details>")


def _view(body: str) -> str:
    return _STYLE + "<div class='awe-view'>" + body + "</div>"


def notice(message: str) -> str:
    return _view("<p class='awe-muted'>" + text(message) + "</p>")


def identity(case: Mapping[str, Any], index: int = 0) -> str:
    value = case.get("case_id")
    if isinstance(value, str) and value:
        return value
    if case.get("agent") and case.get("instance_id"):
        return f"{case['agent']}__{case['instance_id']}"
    return f"case-{index + 1}"


def all_cases(run: Any) -> list[dict[str, Any]]:
    """Include symptom-only outcomes without fabricating a fault diagnosis."""
    cases = list(run.cases)
    seen = {identity(case, i) for i, case in enumerate(cases)}
    symptoms = run.symptoms
    if isinstance(symptoms, dict):
        for case in symptoms.get("cases", []):
            if isinstance(case, dict) and identity(case) not in seen:
                seen.add(identity(case))
                cases.append(case)
    # A graph-only export still has case identities and original graph records.
    # Keep them selectable without synthesizing faults, outcomes or a dossier.
    for graph in _graphs(run):
        if identity(graph) not in seen:
            seen.add(identity(graph))
            cases.append(graph)
    return cases


def outcome(case: Mapping[str, Any]) -> str:
    result = case.get("resolved")
    if result is True:
        return "solved"
    if result is False:
        return "failed"
    # A dossier is restricted to failed cases by the AWE data contract. Symptom
    # records carry an explicit outcome; absent general records remain unknown.
    if "blame_status" in case and "resolved" not in case:
        return "failed"
    return "unknown"


def case_options(run: Any, agent: str = "All agents", result: str = "All outcomes", query: str = "") -> list[tuple[str, str]]:
    options = []
    search = query.strip().casefold()
    for i, case in enumerate(all_cases(run)):
        key = identity(case, i)
        status = outcome(case)
        if agent != "All agents" and case.get("agent") != agent:
            continue
        if result != "All outcomes" and status != result:
            continue
        label = f"{case.get('agent') or 'unknown agent'} / {case.get('instance_id') or key} · {status}"
        if not search or search in label.casefold() or search in key.casefold():
            options.append((label, key))
    return options


def selected_case(run: Any, key: str | None) -> dict[str, Any] | None:
    return next((case for i, case in enumerate(all_cases(run)) if identity(case, i) == key), None)


def _task_context_html(task: Mapping[str, Any]) -> str:
    body = ("<details class='awe-panel'><summary>" + text('Task and reference context') + '</summary>')
    problem = task.get("problem_statement")
    if isinstance(problem, str):
        body += "<div class='awe-task-text'>" + _markdown_report(problem) + "</div>"
        remaining = {key: value for key, value in task.items() if key != "problem_statement"}
    else:
        remaining = dict(task)
    if remaining:
        body += ('<h4>' + text('Reference context and additional task fields') + '</h4>') + _pre(remaining)
    body += _details("Raw task artifact", dict(task))
    return body + "</details>"


def _code_evidence_value(value: Any, depth: int = 0) -> str:
    """Render recorded strings as source text, retaining unknown JSON shapes."""
    if isinstance(value, str):
        return _pre(value)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return "".join(_pre(item) for item in value)
    if isinstance(value, Mapping) and value and depth < 3:
        parts = []
        for name, item in value.items():
            # First-level group names are labels; deeper names may be literal
            # file paths and must retain their recorded spelling.
            label = str(name).replace("_", " ") if depth == 0 else str(name)
            parts.append("<div class='awe-code-field'><h4>" + _escape(label) + "</h4>"
                         + _code_evidence_value(item, depth + 1) + "</div>")
        return "".join(parts)
    return _pre(value)


def _code_evidence_html(value: Any) -> str:
    if value is None:
        return ""
    return (("<details class='awe-panel'><summary>" + text('Agent and reference code') + '</summary>')
            + _code_evidence_value(value) + _details("Raw code evidence artifact", value) + "</details>")


def case_html(case: Mapping[str, Any] | None) -> str:
    if case is None:
        return notice("Choose a case to inspect its recorded evidence.")
    title = f"{case.get('agent') or 'Unknown agent'} / {case.get('instance_id') or identity(case)}"
    body = f"<h3>{_escape(title)}</h3><p><span class='awe-tag'>{_escape(outcome(case))}</span>"
    if case.get("blame_status"):
        body += "<span class='awe-tag'>" + _escape(case["blame_status"]) + "</span>"
    body += ("</p><p class='awe-muted'>" + text('Recorded diagnostic attribution; causal effects remain untested unless explicitly documented.') + '</p>')
    body += attribution_summary_html(dict(case))
    task = case.get("task")
    if isinstance(task, dict):
        body += _task_context_html(task)
    faults = case.get("faults")
    if isinstance(faults, list):
        if not faults:
            body += ('<p>' + text('No supported fault is recorded for this case.') + '</p>')
        for index, fault in enumerate(faults):
            if not isinstance(fault, dict):
                continue
            heading = f"{index + 1}. {fault.get('capability', 'Unknown capability')} → {fault.get('error_type', 'Unknown fault')}"
            heading += " · primary" if fault.get("is_primary") else ""
            body += f"<details class='awe-panel'{' open' if fault.get('is_primary') else ''}><summary>{_escape(heading)}</summary>"
            body += "<p>" + text("Recorded blame weight:") + " " + _escape(fault.get("blame_weight", "unknown"))
            body += " · " + text("confidence:") + " " + _escape(fault.get("confidence", "unknown")) + "</p>"
            for key, label in (("detector_evidence", "Detector evidence"), ("code_evidence", "Agent and reference code"),
                               ("evidence_chain", "Evidence chain, qualifications and checks"), ("audit", "Independent audit and mismatches"),
                               ("countermeasures", "Matched remedy specifications"), ("countermeasure_decisions", "Remedy selection decisions"),
                               ("recommendation", "Recommendation"), ("recommendation_support", "Recommendation support")):
                if key in fault:
                    body += (_code_evidence_html(fault[key]) if key == "code_evidence"
                             else _details(label, fault[key], opened=key == "detector_evidence"))
            body += _details("Complete fault record", fault) + "</details>"
    elif "analysis" in case:
        body += ('<p>' + text('Symptom observation only; this record does not assign fault blame.') + '</p>')
    for key, label in (("attribution_trace", "Exact attribution policy trace"), ("arbiter", "Arbiter review"),
                       ("fault_side", "Model, scaffold and grader context"), ("execution_context", "Execution context"),
                       ("event_ledger", "Action/result ledger and recovery episodes"),
                       ("thinking_process", "Recorded reasoning and action excerpts"),
                       ("symptom_analysis", "Unscored symptom observations"), ("analysis", "Unscored symptom analysis")):
        if key in case:
            body += _details(label, case[key])
    body += _details("Complete original case", dict(case))
    return _view(body)


def _graphs(run: Any) -> list[dict[str, Any]]:
    graphs = run.graphs
    if isinstance(graphs, dict):
        for key in ("cases", "graphs"):
            if isinstance(graphs.get(key), list):
                return [g for g in graphs[key] if isinstance(g, dict)]
        if "nodes" in graphs:
            return [graphs]
    return []


def case_graph(run: Any, case: Mapping[str, Any] | None, kind: str = "Attribution") -> dict[str, Any] | None:
    if case is None:
        return None
    for graph in _graphs(run):
        if identity(graph) == identity(case):
            if kind == "Action/result episodes":
                return graph.get("action_graph") if isinstance(graph.get("action_graph"), dict) else None
            return graph
    return None


def node_options(graph: dict[str, Any] | None) -> list[tuple[str, str]]:
    return [(f"{node.get('number') or node.get('position', '')} · {node.get('title') or node.get('id')}", str(node["id"]))
            for node in (graph or {}).get("nodes", []) if isinstance(node, dict) and "id" in node]


def _graph_anchor(node_id: str, namespace: str = "") -> str:
    value = namespace + "\0" + node_id if namespace else node_id
    return "awe-node-" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _graph_svg(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], selected: str | None, *, namespace: str = "") -> str:
    """Draw typed provenance links with inert SVG and fragment-only anchors."""
    if not nodes:
        return ""
    phases: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        phases.setdefault(str(node.get("phase", "Events")), []).append(node)
    def phase_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    ordered = sorted(phases, key=phase_key)
    column_width, row_height, box_width, box_height = 390, 132, 238, 98
    width = max(320, 34 + len(ordered) * column_width)
    height = max(190, 65 + max(len(rows) for rows in phases.values()) * row_height)
    positions = {str(node["id"]): (24 + column * column_width, 48 + row * row_height)
                 for column, phase in enumerate(ordered) for row, node in enumerate(phases[phase])}
    marker_source = "\0".join(positions)
    if namespace:
        marker_source = namespace + "\0" + marker_source
    marker = "awe-arrow-" + hashlib.sha256(marker_source.encode()).hexdigest()[:12]
    svg = (f"<div class='awe-graph-scroll' role='region' {attrs('Scrollable provenance graph', attribute='aria-label')} tabindex='0'><svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
           f"viewBox='0 0 {width} {height}' role='img' {attrs('Numbered AWE provenance graph', attribute='aria-label')}>"
           "<title>Recorded evidence and attribution decisions</title>"
           "<desc>Arrows are typed provenance links, not established causal effects. Node links lead to recorded details below.</desc>"
           f"<defs><marker id='{marker}' markerWidth='7' markerHeight='7' refX='6' refY='3.5' orient='auto'>"
           "<path d='M0,0 L7,3.5 L0,7 Z' fill='#74859b'/></marker></defs>")
    for column, phase in enumerate(ordered):
        label = "Recorded events" if phase == "Events" else "Stage " + phase
        svg += f"<text x='{24 + column * column_width}' y='25' fill='#51627a' font-size='12' font-weight='600'>{_escape(label)}</text>"
    for edge in edges:
        source, target = str(edge.get("from")), str(edge.get("to"))
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        if x1 == x2:
            start_x, start_y, end_x, end_y = x1 + box_width // 2, y1 + box_height, x2 + box_width // 2, y2
            path = f"M{start_x},{start_y} C{start_x + 36},{start_y + 16} {end_x + 36},{end_y - 16} {end_x},{end_y}"
            label_x, label_y = start_x + 40, (start_y + end_y) // 2
        else:
            start_x, start_y, end_x, end_y = x1 + box_width, y1 + box_height // 2, x2, y2 + box_height // 2
            bend = (start_x + end_x) // 2
            path = f"M{start_x},{start_y} C{bend},{start_y} {bend},{end_y} {end_x},{end_y}"
            label_x, label_y = bend, (start_y + end_y) // 2 - 6
        relation = str(edge.get("relation", "recorded relation"))
        shown_relation = relation if len(relation) <= 28 else relation[:27] + "…"
        svg += (f"<path d='{path}' fill='none' stroke='#74859b' stroke-width='1.4' marker-end='url(#{marker})'>"
                f"<title>{_escape(source)} → {_escape(target)}: {_escape(relation)}</title></path>"
                f"<text x='{label_x}' y='{label_y}' fill='#5f6f84' text-anchor='middle' font-size='9' "
                "paint-order='stroke' stroke='#fbfcff' stroke-width='3' stroke-linejoin='round'>"
                f"{_escape(shown_relation)}</text>")
    for node in nodes:
        key = str(node["id"])
        x, y = positions[key]
        anchor = _graph_anchor(key, namespace)
        active = key == selected
        stroke, fill = ("#7956d9", "#f0ebff") if active else ("#bdc9da", "#ffffff")
        title = str(node.get("title") or key)
        number = str(node.get("number") or node.get("position", ""))
        lines = textwrap.wrap(title, width=28, break_long_words=True, break_on_hyphens=False)
        svg += (f"<a href='#{anchor}'><title>{_escape(number + ' ' + title)}</title>"
                f"<rect x='{x}' y='{y}' width='{box_width}' height='{box_height}' rx='9' fill='{fill}' stroke='{stroke}' stroke-width='{'2' if active else '1'}'/>"
                f"<text x='{x + 12}' y='{y + 20}' fill='#7453c5' font-size='11' font-weight='700'>{_escape(number)}</text>")
        for index, line in enumerate(lines[:2]):
            shown = line + ("…" if index == 1 and len(lines) > 2 else "")
            svg += f"<text x='{x + 12}' y='{y + 39 + index * 15}' fill='#1c2d45' font-size='12' font-weight='600'>{_escape(shown)}</text>"
        state = str(node.get("status") or node.get("provenance") or node.get("kind") or "recorded")
        svg += f"<text x='{x + 12}' y='{y + 83}' fill='#64748b' font-size='10'>{_escape(state[:36])}</text></a>"
    return svg + "</svg></div>"


def source_details(run: Any, source: Any) -> dict[str, Any]:
    """Resolve a recorded JSON pointer only inside the imported artifact set."""
    if run is None or not isinstance(source, dict):
        return {"available": False, "reason": "No imported source artifact is available."}
    artifact = source.get("artifact")
    pointer = source.get("json_pointer")
    if not isinstance(artifact, str) or not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        return {"available": False, "reason": "No supported artifact/JSON-pointer reference was recorded."}
    matches = [name for name in run.artifacts if name == artifact or name.rsplit("/", 1)[-1] == artifact]
    if len(matches) != 1:
        return {"available": False, "artifact": artifact, "json_pointer": pointer,
                "reason": "The referenced artifact is absent or ambiguous in this saved run."}
    value = run.artifacts[matches[0]]
    try:
        for segment in pointer.split("/")[1:] if pointer else []:
            key = segment.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not key.isdigit() or (len(key) > 1 and key.startswith("0")):
                    raise KeyError(key)
                value = value[int(key)]
            elif isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(key)
    except (KeyError, IndexError, ValueError):
        return {"available": False, "artifact": matches[0], "json_pointer": pointer,
                "reason": "The recorded pointer does not resolve in the imported artifact."}
    return {"available": True, "artifact": matches[0], "json_pointer": pointer, "value": value}


def graph_html(graph: dict[str, Any] | None, selected: str | None = None, *, run: Any = None, namespace: str = "") -> str:
    if graph is None:
        return notice("No saved graph is available for this case. Import attribution_graphs.json alongside faults.json.")
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
    body = ("<p class='awe-muted'>" + text('Numbered provenance graph. Edges document recorded inputs and decisions; they do not establish causal effects.') + '</p>')
    body += ("<p class='awe-muted'>" + text('Scroll within the graph to explore its stages, or use the graph node selector for focused details.') + '</p>')
    if graph.get("scope"):
        body += "<p>" + _escape(graph["scope"]) + "</p>"
    phases: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        phases.setdefault(str(node.get("phase", "Events")), []).append(node)
    body += _graph_svg(nodes, edges, selected, namespace=namespace)
    body += ("<details class='awe-panel'><summary>" + text('Numbered card overview') + "</summary><div class='awe-grid'>")
    for phase, rows in phases.items():
        body += "<section><h4>" + (text("Stage {phase}", phase=phase) if phase != "Events" else text("Events")) + "</h4>"
        for node in rows:
            anchor = _graph_anchor(str(node.get("id")), namespace)
            selected_class = " awe-selected" if str(node.get("id")) == selected else ""
            body += f"<a class='awe-node{selected_class}' href='#{anchor}'>"
            body += "<b>" + _escape(str(node.get("number", "")) + " " + str(node.get("title", node.get("id", "Node")))) + "</b>"
            if node.get("status"):
                body += "<br><span class='awe-muted'>" + _escape(node["status"]) + "</span>"
            for line in node.get("lines", []) if isinstance(node.get("lines"), list) else []:
                body += "<br>" + _escape(line)
            body += "</a>"
        body += "</section>"
    body += "</div></details>"
    if edges:
        body += ('<h4>' + text('Recorded relations') + "</h4><div class='awe-edges'>")
        titles = {str(n.get("id")): str(n.get("number") or n.get("title") or n.get("id")) for n in nodes}
        for edge in edges:
            body += ("<p>" + _escape(titles.get(str(edge.get("from")), edge.get("from"))) + " → "
                     + _escape(titles.get(str(edge.get("to")), edge.get("to"))) + " · "
                     + _escape(edge.get("relation", "unspecified")) + "</p>")
        body += "</div>"
    # Anchors are derived IDs, never imported HTML identifiers or event handlers.
    for node in nodes:
        anchor = _graph_anchor(str(node.get("id")), namespace)
        body += f"<details class='awe-panel' id='{anchor}'><summary>"
        body += _escape(str(node.get("number", "")) + " " + str(node.get("title", node.get("id", "Node"))))
        body += "</summary>" + _pre(node)
        if run is not None:
            body += _details("Original artifact value", source_details(run, node.get("source")), opened=True)
        body += "</details>"
    return _view(body if nodes else body + ('<p>' + text('No graph nodes were recorded.') + '</p>'))


def node_html(graph: dict[str, Any] | None, selected: str | None, *, run: Any = None) -> str:
    node = next((n for n in (graph or {}).get("nodes", []) if isinstance(n, dict) and str(n.get("id")) == selected), None)
    if node is None:
        return notice("Choose a graph node to inspect its original source pointer, evidence and decision details.")
    related = [edge for edge in (graph or {}).get("edges", []) if isinstance(edge, dict) and selected in (edge.get("from"), edge.get("to"))]
    body = _details("Selected node", node, opened=True)
    if run is not None:
        body += _details("Original artifact value", source_details(run, node.get("source")), opened=True)
    return _view(body + _details("Incoming and outgoing relations", related, opened=True))


def symptoms_html(run: Any, case: Mapping[str, Any] | None = None) -> str:
    doc = run.symptoms
    body = ("<p class='awe-muted'>" + text('Unscored observations across solved, failed and unknown outcomes. Prevalence is not detector precision or causal effect.') + '</p>')
    if isinstance(doc, dict):
        body += _details("Outcome summary", doc.get("summary"), opened=True)
        rows = doc.get("cases", [])
        if case is not None:
            rows = [c for c in rows if isinstance(c, dict) and identity(c) == identity(case)]
        for row in rows:
            body += _details(f"{identity(row)} · {outcome(row)}", row, opened=case is not None)
    if case is not None and case.get("symptom_analysis") is not None:
        body += _details("Selected dossier symptom analysis", case["symptom_analysis"], opened=True)
    if doc is None and not (case and case.get("symptom_analysis")):
        body += ('<p>' + text('No symptom export was included in this run.') + '</p>')
    return _view(body)


def recommendations_html(run: Any, case: Mapping[str, Any] | None = None) -> str:
    doc = run.recommendations
    if not isinstance(doc, dict):
        return notice("No recommendation plan was included. Case-level remedy fields remain available in the evidence panel.")
    body = ("<p class='awe-warning'>" + text('These are proposed experiments. Coverage rank does not estimate benefit, and a recipe is not an installed intervention.') + '</p>')
    body += _details("Experimental shortlist", doc.get("shortlist"), opened=True)
    candidates = doc.get("candidates", [])
    if case is not None:
        candidates = [c for c in candidates if isinstance(c, dict) and identity(case) in c.get("case_ids", [])]
    for candidate in candidates:
        recipe = candidate.get("recipe") or {}
        title = str(candidate.get("id", "Candidate")) + " · " + str(recipe.get("name") or recipe.get("title") or "Experimental remedy")
        body += _details(title, candidate, opened=case is not None)
    if not candidates:
        body += ('<p>' + text('No selected recipe matches this case or run.') + '</p>')
    for key in ("prerequisites", "measurement", "limitations", "selection_decisions", "review_queues", "prioritization_policy"):
        if key in doc:
            body += _details(key.replace("_", " ").capitalize(), doc[key])
    return _view(body)


def report_options(run: Any) -> list[str]:
    return sorted(name for name in run.artifacts if name.lower().endswith((".md", ".txt", ".csv", ".html", ".json")))


def _measurement(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, float):
        return format(value, ".5g") if math.isfinite(value) else "Non-finite recorded value"
    return _text(value)


def _report_cell(value: Any) -> str:
    if value is None:
        return "<td>" + text("Not recorded") + "</td>"
    content = _measurement(value)
    if len(content) <= 120 and "\n" not in content and "\r" not in content:
        return "<td>" + _escape(content) + "</td>"
    preview = " ".join(content[:80].split())
    if len(content) > 80:
        preview += "…"
    return ("<td><details class='awe-report-cell-details'><summary>"
            + text("Preview ({count} characters): {preview}", count=f"{len(content):,}", preview=preview)
            + "</summary>" + _pre(content) + "</details></td>")


def _report_table(headers: list[str], rows: list[list[Any]], *, localize_headers: bool = False) -> str:
    width = max(len(headers), max((len(row) for row in rows), default=0))
    labels = [text(header) if localize_headers else _escape(header) for header in headers]
    labels.extend(text("Additional column {index}", index=index + 1) for index in range(width - len(headers)))
    body = f"<div class='awe-report-scroll' role='region' {attrs('Scrollable report table', attribute='aria-label')} tabindex='0'><table><thead><tr>"
    body += "".join("<th scope='col'>" + label + "</th>" for label in labels)
    body += "</tr></thead><tbody>"
    for row in rows[:2000]:
        values = [*row, *([None] * (width - len(row)))]
        body += "<tr>" + "".join(_report_cell(value) for value in values) + "</tr>"
    body += "</tbody></table></div>"
    if len(rows) > 2000:
        body += ("<p class='awe-muted'>" + text('Showing the first 2,000 rows. The complete original artifact remains available below and in downloads.') + '</p>')
    return body


def _report_inline(value: str) -> str:
    """A small escaped inline subset; no raw HTML, images or executable URLs."""
    pattern = re.compile(r"(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^\)\n]+\))")
    parts, start = [], 0
    for match in pattern.finditer(value):
        parts.append(_escape(value[start:match.start()]))
        token = match.group()
        if token.startswith("`"):
            parts.append("<code>" + _escape(token[1:-1]) + "</code>")
        elif token.startswith("**"):
            parts.append("<strong>" + _escape(token[2:-2]) + "</strong>")
        else:
            label, url = token[1:].split("](", 1)
            url = url[:-1]
            try:
                parsed = urlsplit(url)
                safe = parsed.scheme in {"https", "http"} and bool(parsed.netloc) and not (parsed.username or parsed.password)
            except ValueError:
                safe = False
            if safe:
                parts.append(f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">{_escape(label)}</a>')
            else:
                parts.append(_escape(label + " (" + url + ")"))
        start = match.end()
    return "".join(parts) + _escape(value[start:])


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _markdown_report(text: str) -> str:
    lines, output, index = text.splitlines(), [], 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker, code = fence[1], []
            index += 1
            while index < len(lines) and not re.fullmatch(r"\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", lines[index]):
                code.append(lines[index])
                index += 1
            output.append(_pre("\n".join(code)))
        elif index + 1 < len(lines) and "|" in line and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", lines[index + 1]):
            headers, rows = _markdown_cells(line), []
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_markdown_cells(lines[index]))
                index += 1
            output.append(_report_table(headers, rows))
            continue
        else:
            heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            bullet = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", line)
            if heading:
                level = min(6, len(heading[1]) + 3)
                output.append(f"<h{level}>" + _report_inline(heading[2]) + f"</h{level}>")
            elif bullet:
                items = [bullet[1]]
                while index + 1 < len(lines):
                    following = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", lines[index + 1])
                    if not following:
                        break
                    items.append(following[1])
                    index += 1
                output.append("<ul>" + "".join("<li>" + _report_inline(item) + "</li>" for item in items) + "</ul>")
            elif line.strip():
                output.append("<p>" + _report_inline(line) + "</p>")
        index += 1
    return "".join(output)


_REPORT_CAPABILITIES = (
    ("requirement_understanding", "Requirements"), ("task_planning", "Planning"),
    ("code_localization", "Localization"), ("code_editing", "Editing"),
    ("code_verification", "Verification"), ("self_repair_loop", "Self-repair"), ("tool_use", "Tool use"),
)


def _agent_records(value: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "capabilities" in value or "scores" in value or "associations" in value or isinstance(value.get("agent"), str):
            return [(str(value.get("agent") or "Agent not recorded"), value)]
        records = value.get("agents") if isinstance(value.get("agents"), Mapping) else value
        return [(str(name), row) for name, row in records.items() if isinstance(row, Mapping)]
    if isinstance(value, list):
        return [(str(row.get("agent") or "Agent not recorded"), row) for row in value if isinstance(row, Mapping)]
    return []


def _interval(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return "[" + ", ".join(_measurement(number) for number in value) + "]"
    return "Not recorded"


def _score_report(value: Any) -> str:
    records = [(agent, row) for agent, row in _agent_records(value)
               if isinstance(row.get("capabilities"), Mapping) or isinstance(row.get("scores"), Mapping)]
    if not records:
        return ""
    body = ('<h4>' + text('Recorded capability scores') + "</h4><p class='awe-muted'>" + text('Stored measurements only. Missing scores and intervals remain unavailable. Blame-based scores and sampling intervals do not establish causal effects.') + '</p>')
    body += ("<div style='overflow-x:auto'><table class='awe-score-matrix'><thead><tr><th scope='col'>" + text('Agent') + '</th>')
    body += "".join("<th scope='col'>" + text(label) + "</th>" for _, label in _REPORT_CAPABILITIES) + "</tr></thead><tbody>"
    details, scope = [], []
    for agent, row in records:
        capabilities = row.get("capabilities") if isinstance(row.get("capabilities"), Mapping) else {}
        scores = row.get("scores") if isinstance(row.get("scores"), Mapping) else {}
        body += "<tr><th scope='row'>" + _escape(agent) + "</th>"
        for capability, _capability_label in _REPORT_CAPABILITIES:
            metric = capabilities.get(capability)
            metric = metric if isinstance(metric, Mapping) else {}
            score = metric.get("score", scores.get(capability))
            color = "transparent"
            if type(score) in (int, float) and 0 <= score <= 100 and math.isfinite(score):
                color = ("#fee2e2", "#ffedd5", "#fef3c7", "#dcfce7", "#bbf7d0")[min(4, int(score // 25))]
            body += f"<td style='background:{color}'>" + _label(_measurement(score))
            body += "<br><small>" + text("Recorded CI:") + " " + _label(_interval(metric.get("score_ci"))) + "</small></td>"
            if metric:
                details.append([agent, capability, metric.get("opportunity"), metric.get("fault_incidence"),
                                _interval(metric.get("score_identification_bounds"))])
        body += "</tr>"
        scope.append([agent, row.get("n_tasks", row.get("n")), row.get("n_failed"), row.get("failure_rate"), row.get("n_requested")])
    body += ('</tbody></table></div><h4>' + text('Recorded cohort coverage') + '</h4>')
    body += _report_table(["Agent", "Scored tasks", "Failed tasks", "Failure rate", "Requested tasks"], scope, localize_headers=True)
    if details:
        body += ('<h4>' + text('Assessment coverage and identification bounds') + "</h4><p class='awe-muted'>" + text('Identification bounds describe unresolved attribution mass; they are separate from sampling confidence intervals.') + '</p>')
        body += _report_table(["Agent", "Capability", "Opportunities", "Fault incidence", "Identification bounds"], details, localize_headers=True)
    return body


def _discovery_report(value: Any) -> str:
    records = _agent_records(value)
    associations, coverage, behaviors = [], [], []
    for agent, record in records:
        rows = record.get("associations")
        if isinstance(rows, list):
            coverage.append([agent, record.get("n_solved"), record.get("n_failed"), record.get("n_unknown"), record.get("scope")])
            for row in rows:
                if isinstance(row, Mapping):
                    associations.append([agent, row.get("feature"), row.get("mean_solved"), row.get("mean_failed"),
                                         row.get("diff"), _interval([row.get("lo"), row.get("hi")]) if "lo" in row or "hi" in row else "Not recorded",
                                         row.get("p_holm"), row.get("risk_ratio")])
        for name, metric in record.items():
            if isinstance(metric, Mapping) and "rate" in metric and "fail_given_mode" in metric:
                behaviors.append([agent, name, record.get("n"), metric.get("n"), metric.get("rate"), metric.get("fail_given_mode")])
    if not associations and not coverage and not behaviors:
        return ""
    body = ('<h4>' + text('Recorded discovery comparisons') + "</h4><p class='awe-warning'>" + text('Observational associations, not causal effects. Prevalence is not detector precision; missing cohort scope and measurements remain unknown.') + '</p>')
    if coverage:
        body += _report_table(["Agent", "Solved cohort", "Failed cohort", "Unknown outcomes", "Recorded scope"], coverage, localize_headers=True)
    if associations:
        body += _report_table(["Agent", "Behavior", "Solved mean", "Failed mean", "Failed − solved", "Recorded CI", "Holm p", "Risk ratio"], associations, localize_headers=True)
    if behaviors:
        body += _report_table(["Agent", "Behavior", "Observed runs", "Mode count", "Prevalence", "Failure given mode"], behaviors, localize_headers=True)
    return body


def report_html(run: Any, name: str | None) -> str:
    if not name or name not in run.artifacts:
        return notice("Choose a report, scorecard, discovery result or original artifact.")
    body = "<h3>" + _escape(name) + "</h3>"
    value = run.artifacts[name]
    extension = name.rsplit(".", 1)[-1].lower()
    if extension == "html":
        body += ("<p class='awe-muted'>" + text('Imported HTML is displayed as source text and is never executed.') + '</p>')
        return _view(body + _pre(value))
    if extension == "md" and isinstance(value, str):
        body += _markdown_report(value)
    elif extension == "csv" and isinstance(value, str):
        rows = list(csv.reader(io.StringIO(value)))
        if rows:
            body += _report_table(rows[0], rows[1:])
    elif extension == "json":
        rendered = _score_report(value) + _discovery_report(value)
        if not rendered:
            return _view(body + _pre(value))
        body += rendered
    else:
        return _view(body + _pre(value))
    return _view(body + _details("Complete original artifact", value))


def provenance_html(run: Any) -> str:
    body = ('<h3>' + text('Saved run provenance') + '</h3><p>' + text('Imported artifacts are preserved independently of the currently loaded native trajectory.') + '</p>')
    for message in run.warnings:
        body += "<p class='awe-warning'>" + _escape(message) + "</p>"
    body += _details("Run manifest", run.manifest, opened=True)
    body += _details("Imported artifact inventory", sorted(run.artifacts), opened=True)
    # Portable envelopes contain base64 for every original file. Construct only
    # display metadata here; exporting that envelope just to omit its payload
    # would duplicate large runs and embed binary attachments in the page.
    body += _details("Import metadata and fingerprints", {
        "source": getattr(run, "source", None), "imported": True,
        "fingerprints": getattr(run, "fingerprints", {}),
    })
    return _view(body)


def rsi_html(run: Any) -> str:
    body = ("<p class='awe-warning'>" + text('RSI proposals describe uninstalled experiments. Diagnostic feedback is privileged development data and must not be supplied to a task agent.') + '</p>')
    names = [name for name in run.artifacts if name.rsplit("/", 1)[-1] in ("rsi_proposals.json", "rsi_feedback.json")]
    if not names:
        body += ('<p>' + text('No RSI exchange files were included.') + '</p>')
    for name in names:
        body += _details(name, run.artifacts[name], opened=name.endswith("rsi_proposals.json"))
    return _view(body)
