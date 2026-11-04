"""Safe, dependency-free presentation of loaded-session inspection evidence."""
from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any

from ..i18n import attrs, text
from ..i18n.catalog_awe import ZH_CN

_STYLE = """<style>
.awe-inspection{font:14px/1.55 system-ui,sans-serif;color:var(--ov-text,#192331);min-width:0;max-width:100%}
.awe-inspection *{box-sizing:border-box}.awe-inspection h3{font-size:17px;line-height:1.35;margin:0 0 10px}
.awe-inspection h4{font-size:14px;margin:10px 0 5px}.awe-inspection p{margin:6px 0;overflow-wrap:anywhere}
.awe-inspection .ai-panel{border:1px solid var(--ov-border,#dbe3ed);border-radius:10px;padding:14px;margin:12px 0;min-width:0}
.awe-inspection .ai-muted{color:var(--ov-muted,#5b6473)}.awe-inspection .ai-small{font-size:12px}
.awe-inspection .ai-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,130px),1fr));gap:8px;margin:12px 0}
.awe-inspection .ai-stat{padding:9px 11px;border-radius:7px;background:var(--ov-card,#f7f9fc);border:1px solid var(--ov-border,#dbe3ed);min-width:0}
.awe-inspection .ai-stat strong{display:block;font-size:21px;font-variant-numeric:tabular-nums;line-height:1.3}
.awe-inspection .ai-stat span{font-size:12px;color:var(--ov-muted,#5b6473)}
.awe-inspection .ai-tag{display:inline-block;border-radius:5px;padding:1px 7px;font-size:11px;background:var(--ov-card,#eef2f7);color:var(--ov-muted,#5b6473);vertical-align:middle;margin:0 5px 3px 0}
.awe-inspection .ai-finding{border-left:3px solid #64748b}.awe-inspection .ai-warning{border-left-color:#b45309}
.awe-inspection .ai-finding-title{font-size:15px;font-weight:650;overflow-wrap:anywhere}.awe-inspection .ai-next{padding-top:7px;margin-top:8px;border-top:1px solid var(--ov-border,#dbe3ed)}
.awe-inspection summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}.awe-inspection summary:focus-visible{outline:2px solid #2563eb;outline-offset:3px}
.awe-inspection ul{padding-left:20px;margin:5px 0}.awe-inspection li{margin:4px 0;overflow-wrap:anywhere}
.awe-inspection .ai-step{display:inline-block;font:12px/1.5 system-ui,sans-serif;vertical-align:middle;margin:2px 4px 2px 0;padding:2px 7px;border:1px solid var(--ov-border,#cbd5e1);border-radius:5px;background:var(--ov-card,#f7f9fc);color:var(--ov-text,#192331);white-space:nowrap;text-decoration:none}
.awe-inspection button.ai-step,.trajectory-viz .awe-inspection button.ai-step{font:500 12px/1.5 system-ui,sans-serif;min-height:0;height:auto;padding:2px 7px;border:1px solid var(--ov-border,#cbd5e1);border-radius:5px;background:var(--ov-card,#f7f9fc);color:var(--ov-text,#192331);box-shadow:none;cursor:pointer}
.awe-inspection button.ai-step:hover{border-color:#2563eb;color:#1d4ed8}.awe-inspection button.ai-step:focus-visible{outline:2px solid #2563eb;outline-offset:2px}
.awe-inspection .ai-cited-steps{margin-top:5px}.awe-inspection .ai-cited-steps summary{font-size:12px;font-weight:500}
.awe-inspection pre{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-height:320px;overflow:auto;padding:9px;border-radius:6px;background:var(--ov-card,#f7f9fc);margin:6px 0;color:var(--ov-text,#192331)}
.awe-inspection .ai-table-scroll{width:100%;max-width:100%;overflow-x:auto;overscroll-behavior-x:contain;margin-top:10px}
.awe-inspection table{border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed;min-width:570px}
.awe-inspection th,.awe-inspection td{text-align:left;vertical-align:top;border-bottom:1px solid var(--ov-border,#dbe3ed);padding:8px;overflow-wrap:anywhere}
.awe-inspection th{color:var(--ov-muted,#5b6473);font-weight:600}.awe-inspection td pre{max-height:150px;padding:5px;margin:2px 0}
.awe-inspection .ai-table-empty{margin-top:8px;color:var(--ov-muted,#5b6473)}
@media(max-width:600px){.awe-inspection .ai-panel{padding:11px}.awe-inspection .ai-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.awe-inspection h3{font-size:16px}.awe-inspection th,.awe-inspection td{padding:6px}}
</style>"""


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _text(value: Any, fallback: str = "Not recorded") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _escape(value: Any, fallback: str = "Not recorded") -> str:
    if value is None or value == "":
        return text(fallback)
    return html.escape(_text(value, fallback), quote=True)


def _prose(value: Any, fallback: str = "Not recorded") -> str:
    """Mark known application prose, while preserving unfamiliar recorded text."""
    if value is None or value == "":
        return text(fallback)
    return (text(value) if isinstance(value, str) and value.endswith(".") and value in ZH_CN
            else _escape(value))


_FINDING_TITLES = {
    "tool_failures": "{count} tool calls recorded failures",
    "later_success": "{count} failed actions have a later successful call",
    "missing_results": "{count} tool calls have no linked observed result",
    "reported_test_failures": "{count} test commands reported failures or errors",
    "output_summary_text": "Output contains test-summary text for {count} commands",
    "verification_evidence_unknown": "{count} test commands need output review",
    "unanswered_tool_call": "{count} tool calls need result review",
    "ambiguous_tool_linkage": "{count} tool calls have ambiguous result links",
    "zero_tests_executed": "{count} test commands reported zero execution",
    "all_tests_skipped": "{count} test commands reported only skipped tests",
    "slow_steps": "{count} recorded steps lasted at least 15 seconds",
}


def _finding_title(row: Mapping[str, Any]) -> str:
    title = row.get("title")
    kind = row.get("kind")
    template = _FINDING_TITLES.get(kind) if isinstance(kind, str) else None
    if isinstance(title, str) and template:
        import re

        match = re.search(r"\d+", title)
        if match and template.format(count=match.group()) == title:
            return text(template, count=match.group())
    if (kind, title) in (("no_visible_tests", "No recognized test command is visible"),
                          ("recording_summary", "Recorded workflow is ready to inspect")):
        return text(title)
    return _escape(title, "Recorded observation")


def _number(value: Any) -> str:
    return f"{value:,}" if type(value) is int and value >= 0 else "—"


def _indices(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(index for index in value if type(index) is int and index >= 0))


def _step(index: Any, allowed: set[int], workflow_links: bool) -> str:
    if type(index) is not int or index < 0:
        return ""
    label = text("Step {index}", index=index)
    if workflow_links and index in allowed:
        return (f"<button type='button' class='ai-step' {attrs('Open Workflow step {index}', attribute='aria-label', index=index)} "
                f"onclick=\"if(typeof window.tvGotoWorkflowStep==='function'){{window.tvGotoWorkflowStep({index});}}\">"
                f"{label}</button>")
    return f"<span class='ai-step'>{label}</span>"


def _steps(value: Any, allowed: set[int], workflow_links: bool) -> str:
    return "".join(_step(index, allowed, workflow_links) for index in _indices(value))


def _limited_steps(value: Any, allowed: set[int], workflow_links: bool, limit: int, *, expandable: bool = False) -> str:
    indices = _indices(value)
    body = _steps(indices[:limit], allowed, workflow_links)
    remainder = len(indices) - limit
    if remainder > 0:
        if expandable:
            body += ("<details class='ai-cited-steps'><summary>" + text("All cited steps ({count})", count=len(indices)) + "</summary>"
                     "<div class='ai-muted ai-small'>" + text("Additional cited steps") + "</div>"
                     + _steps(indices[limit:], allowed, workflow_links) + "</details>")
        else:
            body += "<span class='ai-muted ai-small'>" + text("+{count} more", count=remainder) + "</span>"
    return body


def _row_steps(row: Mapping[str, Any], allowed: set[int], workflow_links: bool, *, limit: int | None = None) -> str:
    indices = _indices(row.get("step_indices"))
    if type(row.get("workflow_index")) is int and row["workflow_index"] >= 0:
        indices = list(dict.fromkeys([row["workflow_index"], *indices]))
    body = (_limited_steps(indices, allowed, workflow_links, limit, expandable=True) if limit is not None
            else _steps(indices, allowed, workflow_links)) or ("<span class='ai-muted'>" + text("No mapped step") + "</span>")
    result_index = row.get("result_workflow_index")
    if type(result_index) is int and result_index >= 0 and result_index not in indices:
        body += "<div class='ai-muted ai-small'>" + text("Result:") + " " + _step(result_index, allowed, workflow_links) + "</div>"
    return body


def _list_text(value: Any) -> str:
    return ", ".join(_text(item) for item in value) if isinstance(value, list) else _text(value)


def _test_summary(row: Mapping[str, Any]) -> str:
    status = row.get("summary_status")
    body = _escape(status.replace("_", " ") if isinstance(status, str) else status)
    reason = row.get("reason")
    if isinstance(reason, str) and reason.strip():
        body += "<div class='ai-muted'>" + text("Reason:") + " " + _escape(reason.replace("_", " ")) + "</div>"
    if row.get("counts") is not None:
        body += "<div>" + _escape(row["counts"]) + "</div>"
    candidate = row.get("output_summary")
    if isinstance(candidate, Mapping) and candidate.get("status") == "reported" and isinstance(candidate.get("counts"), Mapping):
        body += ("<div class='ai-muted ai-small' style='margin-top:6px;'>"
                 "<strong>" + text("Summary text in output (command attribution unknown)") + "</strong><br>"
                 + _escape(candidate.get("counts")) + "</div>")
    return body


def _test_result(row: Mapping[str, Any]) -> str:
    labels = {"output_summary_only": "Output text only", "reported_summary": "Runner summary reported",
              "reported_failures": "Runner summary reports failures", "unknown": "Not determined"}
    status = row.get("status")
    if isinstance(status, str) and status in labels:
        return text(labels[status])
    return _escape(status.replace("_", " ") if isinstance(status, str) else status)


def _process_result(row: Mapping[str, Any]) -> str:
    code = row.get("exit_code")
    body = text("Exit code {code}", code=code) if type(code) is int else text("Exit code not recorded")
    status = row.get("process_status")
    if isinstance(status, str) and status.strip() and status not in {"zero_exit", "nonzero_exit", "unknown"}:
        body += "<div class='ai-muted'>" + _escape(status.replace("_", " ")) + "</div>"
    return body


def _pre(value: Any) -> str:
    return "<pre>" + _escape(value) + "</pre>"


def _details(title: str, body: str, *, opened: bool = False, **params: Any) -> str:
    return (f"<details class='ai-panel'{' open' if opened else ''}>"
            f"<summary>{text(title, **params)}</summary>{body}</details>")


def _table(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f"<p class='ai-table-empty'>{text(empty)}</p>"
    return (f"<div class='ai-table-scroll' tabindex='0' {attrs('Scrollable evidence table', attribute='aria-label')}><table><thead><tr>"
            + "".join(f"<th scope='col'>{text(header)}</th>" for header in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _finding(row: Mapping[str, Any], index: int, allowed: set[int], workflow_links: bool) -> str:
    severity = "warning" if row.get("severity") == "warning" else "info"
    body = (f"<article class='ai-panel ai-finding ai-{severity}'>"
            f"<div class='ai-finding-title'>{index}. {_finding_title(row)}</div>"
            + _row_steps(row, allowed, workflow_links, limit=8)
            + f"<p>{_prose(row.get('observation'), 'Review the recorded evidence below.')}</p>")
    check = row.get("next_check")
    if isinstance(check, str) and check.strip():
        body += f"<p class='ai-next'><strong>{text('Next check:')}</strong> {_prose(check)}</p>"
    evidence = []
    for record in _records(row.get("evidence")):
        context = []
        for key, label in (("part_index", "Part"), ("thread_id", "Thread"), ("call_id", "Call")):
            if record.get(key) is not None and record.get(key) != "":
                context.append(f"{text(label)} {_escape(record[key])}")
        excerpt = record.get("output_excerpt")
        evidence.append("<div>" + _row_steps(record, allowed, workflow_links)
                        + ("<span class='ai-muted ai-small'>" + " · ".join(context) + "</span>" if context else "")
                        + (_pre(excerpt) if excerpt is not None and excerpt != "" else "") + "</div>")
    if evidence:
        body += ("<details><summary>" + text("Recorded evidence") + "</summary>" + "".join(evidence) + "</details>")
    return body + "</article>"


def inspection_html(result: dict, *, workflow_links: bool = True) -> str:
    """Render an inspection without deriving new verdicts or trusting raw step ids.

    Clickable references require integer canonical indices in ``workflow_indices``.
    Set ``workflow_links=False`` for an exported report without the live Workflow UI.
    Recorded strings are escaped, including JSON shown in collapsed details.
    """
    data = result if isinstance(result, Mapping) else {}
    if data.get("available") is not True:
        reason = _prose(data.get("reason"), "Load a trajectory above to inspect its recorded workflow.")
        return (_STYLE + f"<section class='awe-inspection' {attrs('Trajectory inspection', attribute='aria-label')}>"
                "<div class='ai-panel'><h3>" + text("Inspect a loaded trajectory") + "</h3>"
                f"<p>{reason}</p></div></section>")

    allowed = set(_indices(data.get("workflow_indices")))
    summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else {}
    source = data.get("source") if isinstance(data.get("source"), Mapping) else {}
    outcome = data.get("outcome") if isinstance(data.get("outcome"), Mapping) else {}
    findings = _records(data.get("findings"))
    body = f"<section class='ai-panel' {attrs('At a glance', attribute='aria-label')}><h3>" + text("At a glance") + "</h3>"
    if source.get("name") or source.get("format"):
        identity = [_escape(source[key]) for key in ("name", "format") if source.get(key)]
        body += "<p class='ai-muted'>" + " · ".join(identity) + "</p>"
    body += "<div class='ai-stats'>"
    for key, label in (("steps", "Recorded steps"), ("tool_calls", "Tool calls"),
                       ("failed_calls", "Recorded failures"), ("files", "File targets"),
                       ("test_runs", "Test-related calls")):
        body += f"<div class='ai-stat'><strong>{_number(summary.get(key))}</strong><span>{text(label)}</span></div>"
    body += "</div>"
    if findings:
        body += "<h4>" + text("What stands out") + "</h4><ul>" + "".join(
            f"<li>{_finding_title(finding)} "
            + _limited_steps(finding.get("step_indices"), allowed, workflow_links, 4) + "</li>" for finding in findings[:3]) + "</ul>"
        next_check = next((finding["next_check"] for finding in findings
                           if isinstance(finding.get("next_check"), str) and finding["next_check"].strip()), None)
        if next_check:
            body += f"<p class='ai-next'><strong>{text('Start here:')}</strong> {_prose(next_check)}</p>"
    else:
        body += "<p>" + text("No inspection findings were produced from the available recording. Review recording coverage below.") + "</p>"
    body += ("<p class='ai-muted ai-small'><strong>" + text("Task outcome:") + "</strong> "
             + _escape(outcome.get("status"), "unknown") + ". "
             + _prose(outcome.get("basis"), "Independent task or evaluator evidence has not been supplied.") + "</p></section>")

    body += f"<section {attrs('Recorded findings', attribute='aria-label')}><h3>" + text("Recorded findings") + "</h3>"
    body += "".join(_finding(finding, i + 1, allowed, workflow_links) for i, finding in enumerate(findings))
    if not findings:
        body += "<p class='ai-muted'>" + text("No ranked findings are available for this recording.") + "</p>"
    body += "</section>"

    files = _records(data.get("files"))
    rows = [[_escape(row.get("path")), _escape(_list_text(row.get("actions"))),
             _row_steps(row, allowed, workflow_links), _escape(_list_text(row.get("threads")))] for row in files]
    body += _details("Recorded file targets ({count})",
        "<p class='ai-muted'>" + text("Recorded arguments identify these targets. An attempted call does not confirm that a file changed.") + "</p>" + _table(
        ["File", "Recorded actions", "Steps", "Threads"], rows, "No file targets were extracted from the recorded actions."), count=len(files))

    tests = _records(data.get("tests"))
    rows = [[_row_steps(row, allowed, workflow_links), _pre(row.get("command")),
             _test_result(row), _process_result(row),
             _test_summary(row),
             _pre(row.get("output_excerpt"))] for row in tests]
    body += _details("Recorded test activity ({count})", "<p class='ai-muted'>" + text("Test-related commands, process outcomes and any runner summaries present in the recording.") + "</p>" + _table(
        ["Step", "Command", "Observed test result", "Recorded process result", "Runner summary", "Recorded output"], rows,
        "No test commands were identified in the recorded tool calls."), count=len(tests))

    failures = _records(data.get("failures"))
    rows = [[_row_steps(row, allowed, workflow_links), _escape(row.get("tool_name")),
             _pre(row.get("command")), _escape(row.get("status")), _escape(row.get("exit_code")),
             _pre(row.get("output_excerpt"))] for row in failures]
    body += _details("Recorded tool failures ({count})", _table(
        ["Step", "Tool", "Command", "Status", "Exit code", "Recorded output"], rows,
        "No failed tool calls were identified in the available recording."), count=len(failures))

    recovery = _records(data.get("recovery"))
    rows = [[_row_steps(row, allowed, workflow_links), _pre(row.get("command")),
             _escape(row.get("lifecycle")), _step(row.get("recovery_workflow_index"), allowed, workflow_links) or text("Not recorded"),
             _pre(row.get("output_excerpt"))] for row in recovery]
    body += _details("Retries and later outcomes ({count})", "<p class='ai-muted'>" + text("A later command outcome describes the recording; it does not establish that the task was solved.") + "</p>" + _table(
        ["Steps", "Command", "Observed sequence", "Later outcome step", "Recorded context"], rows,
        "No retry or failure-recovery episodes were identified in the available recording."), count=len(recovery))

    timing = _records(data.get("timing"))
    rows = [[_row_steps(row, allowed, workflow_links), _escape(row.get("duration_seconds")),
             _escape(row.get("timing_source")), text("Incomplete") if row.get("incomplete") is True else text("As recorded")] for row in timing]
    body += _details("Slow recorded steps ({count})", _table(
        ["Step", "Seconds", "Timing source", "Coverage"], rows,
        "No slow-step timing observations are available."), count=len(timing))

    limitations = data.get("limitations") if isinstance(data.get("limitations"), list) else []
    limitations = [value for value in limitations if isinstance(value, str) and value.strip()]
    coverage = data.get("coverage")
    coverage_body = ("<ul>" + "".join(f"<li>{_prose(value)}</li>" for value in limitations) + "</ul>"
                     if limitations else "<p>" + text("Additional recording limitations were not provided.") + "</p>")
    if isinstance(coverage, Mapping) and coverage:
        coverage_body += "<h4>" + text("Recorded-field coverage") + "</h4>" + _table(["Field", "Coverage"], [
            [_escape(str(key).replace("_", " ")), _escape(value)] for key, value in coverage.items()], "")
    body += _details("Recording coverage and limits", coverage_body)
    if isinstance(data.get("ledger"), Mapping):
        body += _details("Action and result ledger", _pre(data["ledger"]))
    if isinstance(data.get("symptoms"), Mapping):
        body += _details("Recorded symptom observations", _pre(data["symptoms"]))
    body += _details("Full inspection record", _pre(dict(data)))
    return _STYLE + f"<section class='awe-inspection' {attrs('Trajectory inspection', attribute='aria-label')}>" + body + "</section>"
