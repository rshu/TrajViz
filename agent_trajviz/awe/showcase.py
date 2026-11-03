"""Safe offline presentation of explicitly saved evaluation showcase data."""
from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

_CAPABILITIES = {"requirement_understanding": "Requirement understanding", "task_planning": "Task planning",
                 "code_localization": "Code localization", "code_editing": "Code editing",
                 "code_verification": "Code verification", "self_repair_loop": "Self-repair loop", "tool_use": "Tool use"}
_COLORS = ("#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be185d")


def _text(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return html.escape(str(value), quote=True)


def _number(value: Any, field: str, *, maximum: float | None = None) -> None:
    if value is None:
        return
    if (type(value) not in (int, float) or (isinstance(value, float) and not math.isfinite(value))
            or value < 0 or (maximum is not None and value > maximum)):
        raise ValueError(f"{field} must be a finite nonnegative number in its supported range")


def validate_showcase(value: Any) -> dict[str, Any]:
    """Reject malformed numeric evidence; text remains text and is escaped on output."""
    if not isinstance(value, dict):
        raise ValueError("Saved showcase must be a JSON object")
    for key in ("dataset", "insights", "agents_meta"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"Saved showcase {key} must be an object")
    for key in ("cases", "countermeasures"):
        if not isinstance(value.get(key), list) or any(not isinstance(row, dict) for row in value[key]):
            raise ValueError(f"Saved showcase {key} must contain objects")
    json.dumps(value, allow_nan=False)
    dataset = value["dataset"]
    for key in ("n_tasks", "n_repos", "n_runs", "n_scored_runs", "n_solved", "n_failed", "n_excluded"):
        _number(dataset.get(key), "dataset." + key)
    properties = dataset.get("task_props") or {}
    if not isinstance(properties, dict):
        raise ValueError("Saved task properties must be an object")
    for key in ("n", "single_file", "multi_file", "cross_module", "median_gold_files", "median_gold_loc", "max_gold_files"):
        _number(properties.get(key), "task_props." + key)
    for agent, row in value["insights"].items():
        if not isinstance(row, dict):
            raise ValueError("Each agent insight must be an object")
        for key in ("n", "n_failed", "n_solved", "n_excluded"):
            _number(row.get(key), "insights." + key)
        _number(row.get("failure_rate"), "insights.failure_rate", maximum=1)
        if row.get("n") is not None and row.get("n_failed") is not None and row["n_failed"] > row["n"]:
            raise ValueError("Failed cases cannot exceed the recorded agent run count")
        scores = row.get("scores") or {}
        if not isinstance(scores, dict):
            raise ValueError("Recorded scores must be an object")
        for score in scores.values():
            _number(score, "insights.scores", maximum=100)
        for key in ("ci", "ci95", "confidence_intervals"):
            intervals = row.get(key)
            if intervals is not None and not isinstance(intervals, dict):
                raise ValueError("Recorded confidence intervals must be an object")
        mass = row.get("cap_mass") or {}
        if not isinstance(mass, dict):
            raise ValueError("Recorded fault mass must be an object")
        for amount in mass.values():
            _number(amount, "insights.cap_mass")
        if agent in value["agents_meta"] and not isinstance(value["agents_meta"][agent], dict):
            raise ValueError("Recorded agent metadata must be an object")
    for key in ("agg_primary", "primary_capabilities", "attribution_statuses"):
        counts = value.get(key) or {}
        if not isinstance(counts, dict):
            raise ValueError(f"{key} must be an object")
        for count in counts.values():
            _number(count, key)
    severities = value.get("edit_severity") or []
    if not isinstance(severities, list):
        raise ValueError("Recorded edit severity must be a list")
    for count in severities:
        _number(count, "edit_severity")
    validation = value.get("validation") or {}
    if not isinstance(validation, dict):
        raise ValueError("Recorded validation must be an object")
    for key in ("unit_tests", "evidence_mismatches"):
        _number(validation.get(key), "validation." + key)
    for case in value["cases"]:
        faults = case.get("faults") or []
        if not isinstance(faults, list) or any(not isinstance(fault, dict) for fault in faults):
            raise ValueError("Recorded case faults must contain objects")
        for fault in faults:
            _number(fault.get("blame_weight"), "fault.blame_weight", maximum=1)
    for entry in value["countermeasures"]:
        _number(entry.get("empirical_support"), "countermeasure.empirical_support")
        if entry.get("references") is not None and not isinstance(entry["references"], list):
            raise ValueError("Recorded catalog references must be a list")
    return value


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class='muted'>Not recorded.</p>"
    return ("<div class='table-wrap'><table><thead><tr>" + "".join("<th>" + _text(h) + "</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + _text(v) + "</td>" for v in row)
                                                + "</tr>" for row in rows) + "</tbody></table></div>")


def _details(title: str, value: Any, *, opened: bool = False) -> str:
    if value is None or value == {} or value == [] or value == "":
        return ""
    body = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else str(value)
    return ("<details" + (" open" if opened else "") + "><summary>" + _text(title)
            + "</summary><pre>" + _text(body) + "</pre></details>")


def _score_chart(rows: list[tuple[str, float | int | None]]) -> str:
    if not rows:
        return "<p class='muted'>Scores were not recorded.</p>"
    height = 36 * len(rows) + 30
    out = [f"<svg viewBox='0 0 740 {height}' role='img' aria-label='Recorded capability scores on a zero to one hundred scale'>"]
    for index, (label, score) in enumerate(rows):
        y = index * 36 + 10
        out.append(f"<text x='0' y='{y + 17}'>{_text(label)}</text>")
        if score is not None:
            width = float(score) * 3.8
            out.append(f"<rect x='280' y='{y}' width='{width:.3f}' height='24' rx='4' fill='{_COLORS[index % len(_COLORS)]}'/>")
            out.append(f"<text x='{290 + width:.3f}' y='{y + 17}'>{float(score):.1f}</text>")
        else:
            out.append(f"<text x='280' y='{y + 17}'>Not recorded</text>")
    out.append("</svg>")
    return "".join(out)


def _safe_url(value: Any) -> str | None:
    if not isinstance(value, str) or any(ord(character) < 32 for character in value):
        return None
    if value.startswith("arXiv:"):
        value = "https://arxiv.org/abs/" + quote(value[6:], safe="/.")
    elif value.startswith("doi:"):
        value = "https://doi.org/" + quote(value[4:], safe="/.")
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    return value if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username and not parsed.password else None


def render_showcase(value: Any) -> str:
    """Render recorded values as escaped HTML/SVG; no executable data or invented coverage."""
    data = validate_showcase(value)
    dataset = data["dataset"]
    out = ["<header><p class='eyebrow'>TrajViz · saved evaluation</p><h1>" + _text(dataset.get("benchmark")) + "</h1>",
           "<p>Recorded results and their evidence. Missing measurements remain unrecorded; proposed experiments retain their recorded status.</p></header>"]
    out.append("<section><h2>Run scope</h2>" + _table(["Recorded measure", "Value"],
        [[key.replace("_", " "), dataset.get(key)] for key in ("n_tasks", "n_repos", "n_runs", "n_scored_runs", "n_solved", "n_failed", "n_excluded")]) + "</section>")
    comparisons, outcomes = [], []
    agent_sections = []
    for agent, insight in data["insights"].items():
        metadata = data["agents_meta"].get(agent) or {}
        name = metadata.get("name") or agent
        for capability, score in (insight.get("scores") or {}).items():
            comparisons.append((str(name) + " · " + _CAPABILITIES.get(capability, capability), score))
        total, failed = insight.get("n"), insight.get("n_failed")
        outcomes.append([name, total, failed, insight.get("n_solved"), insight.get("failure_rate")])
        agent_sections.append("<details><summary>" + _text(name) + " · recorded scorecard</summary>"
                              + _table(["Capability", "Score", "Recorded confidence interval"],
                                [[_CAPABILITIES.get(capability, capability), score,
                                  (insight.get("ci") or insight.get("ci95") or insight.get("confidence_intervals") or {}).get(capability)]
                                 for capability, score in (insight.get("scores") or {}).items()])
                              + _details("Agent metadata", metadata) + _details("Additional recorded insight", insight) + "</details>")
    out.append("<section><h2>Aggregate comparisons</h2>" + _score_chart(comparisons)
               + _table(["Agent", "Scored runs", "Failed", "Solved (recorded)", "Failure rate (recorded)"], outcomes)
               + "".join(agent_sections) + "</section>")
    out.append("<section><h2>Attribution summaries</h2>" + _table(["Primary capability", "Recorded count"],
               [[_CAPABILITIES.get(key, key), count] for key, count in (data.get("primary_capabilities") or {}).items()])
               + _table(["Attribution status", "Recorded count"], [[key, count] for key, count in (data.get("attribution_statuses") or {}).items()])
               + _table(["Primary fault", "Recorded count"], [[key, count] for key, count in (data.get("agg_primary") or {}).items()]) + "</section>")
    out.append("<section><h2>Case evidence</h2>")
    if not data["cases"]:
        out.append("<p class='muted'>No case gallery was supplied.</p>")
    for index, case in enumerate(data["cases"]):
        out.append(f"<details id='case-{index}'><summary>" + _text(case.get("instance_id") or case.get("case_id"))
                   + " · " + _text(case.get("blame_status")) + "</summary>")
        task = case.get("task") if isinstance(case.get("task"), dict) else {}
        out.append("<p>" + _text(case.get("agent")) + " · " + _text(case.get("repo")) + "</p><p>"
                   + _text(case.get("problem") or task.get("problem_statement")) + "</p>")
        out.append(_details("Task and reference files", case.get("task") or case.get("gold_files")))
        for fault in case.get("faults") or []:
            out.append("<article><h3>" + _text(fault.get("label") or fault.get("error_type")) + "</h3>"
                       + _table(["Recorded field", "Value"], [["Primary", fault.get("is_primary")], ["Weight", fault.get("blame_weight")],
                         ["Confidence", fault.get("confidence")], ["Evidence strength", fault.get("strength")], ["Audit", fault.get("audit")]])
                       + _details("Evidence chain", fault.get("evidence_chain") or fault.get("links"), opened=True)
                       + _details("Evidence qualifications", fault.get("strength_basis"))
                       + _details("Code evidence", fault.get("code_evidence")) + _details("Counterfactual check", fault.get("counterfactual"))
                       + _details("Proposed experiments", fault.get("recommendation")) + "</article>")
        out.append(_details("Recorded reasoning and actions", case.get("thinking_process")) + "</details>")
    out.append("</section><section><h2>Recommendations and catalog</h2>"
               + _details("Recorded recommendation plan", data.get("recommendation_plan"), opened=True))
    for entry in data["countermeasures"]:
        out.append("<details><summary>" + _text(entry.get("summary") or entry.get("id")) + " · " + _text(entry.get("status")) + "</summary>"
                   + _table(["Recorded field", "Value"], [[key.replace("_", " "), entry.get(key)]
                             for key in ("type", "capability", "error_type", "artifact_maturity", "empirical_support")])
                   + _details("Proposed implementation", entry.get("artifact")) + _details("Applicability", entry.get("applies_when")))
        for reference in entry.get("references") or []:
            if not isinstance(reference, Mapping):
                out.append("<p>" + _text(reference) + "</p>")
                continue
            title = _text(reference.get("title") or reference.get("id"))
            url = _safe_url(reference.get("id"))
            out.append("<p>" + ("<a rel='noopener noreferrer' href='" + _text(url) + "'>" + title + "</a>" if url else title) + "</p>")
        out.append("</details>")
    out.append("</section><section><h2>Validation and provenance</h2>"
               + _table(["Recorded validation", "Value"], [[key.replace("_", " "), value] for key, value in (data.get("validation") or {}).items()])
               + _details("Run provenance", data.get("run_provenance"), opened=True)
               + _details("Discovery observations", data.get("discovery")) + _details("Task properties", dataset.get("task_props"))
               + _details("Judge example", data.get("judge_example")) + _details("Arbiter example", data.get("arbiter_example"))
               + "<p class='muted'>These records describe supplied evidence. They do not establish causal effects or measured intervention benefit.</p></section>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>TrajViz · saved evaluation</title><style>"
            "body{font:15px system-ui,sans-serif;line-height:1.55;max-width:1120px;margin:auto;padding:24px;color:#172033;background:#fff}"
            "header,section{margin-bottom:28px}h1{font-size:30px}h2{font-size:22px}.eyebrow,.muted{color:#64748b}"
            "details,article{border:1px solid #dbe3ef;border-radius:8px;padding:12px;margin:10px 0}summary{cursor:pointer;font-weight:600}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:12px;border-radius:6px}.table-wrap{overflow:auto}"
            "table{border-collapse:collapse;width:100%;margin:10px 0}td,th{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top;overflow-wrap:anywhere}"
            "svg{width:100%;height:auto}svg text{font:12px system-ui,sans-serif;fill:#172033}@media(max-width:600px){body{padding:12px}h1{font-size:24px}}"
            "</style><body>" + "".join(out) + "<footer><a href='https://github.com/rshu/TrajViz'>TrajViz</a></footer></body></html>")
