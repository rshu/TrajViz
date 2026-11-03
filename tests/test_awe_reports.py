"""Readable reports preserve stored statistics and keep imported markup inert."""
from __future__ import annotations

from html.parser import HTMLParser
import html
import json
import re
from types import SimpleNamespace

from agent_trajviz.awe import render


def _report(name, value):
    return render.report_html(SimpleNamespace(artifacts={name: value}), name)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.names.append(tag)
        if tag == "a":
            self.links.append(dict(attrs).get("href"))


def test_markdown_scorecard_is_readable_with_recorded_intervals_and_safe_markup():
    text = """# Capability Diagnosis

## Capability scores
| Capability | Score (95% CI) | Opportunities |
|---|---|---|
| Code Editing | 75 [60, 90] | 4 |

- **Observed** score, `recorded` evidence.
- <img src=x onerror=alert(1)>

```python
<script>source()</script>
```
[unsafe](javascript:alert(1))
[reference](https://example.com/report)
"""
    output = _report("reports/agent.md", text)
    tags = Tags()
    tags.feed(output)
    assert {"h3", "h4", "table", "ul", "li", "strong", "code", "pre"} <= set(tags.names)
    assert "75 [60, 90]" in output and "95% CI" in output
    assert "img" not in tags.names and "script" not in tags.names
    assert tags.links == ["https://example.com/report"]
    assert "&lt;script&gt;source()&lt;/script&gt;" in output
    assert "Complete original artifact" in output


def test_csv_table_keeps_quoted_cells_zeroes_and_missing_fields_distinct():
    output = _report("failure_composition.csv", 'agent,mode,mass\nalpha,"mode, with comma",0\nbeta,unknown\n')
    assert "<table>" in output and "mode, with comma" in output
    assert "<td>0</td>" in output and "<td>Not recorded</td>" in output
    assert "Complete original artifact" in output


def test_structured_scores_render_heatmap_and_separate_sampling_ci_from_bounds():
    value = {"agent": "alpha", "n_tasks": 5, "n_failed": 1, "n_requested": 7,
             "capabilities": {"code_editing": {"score": 75, "score_ci": [60, 90], "opportunity": 5,
                                                "fault_incidence": 0.2, "score_identification_bounds": [50, 75]},
                              "tool_use": {"score": 0, "score_ci": None}}}
    output = _report("scorecard.json", value)
    assert "awe-score-matrix" in output and "Recorded capability scores" in output
    # Eight score columns retain readable width inside the existing scroll region.
    assert ".awe-view table.awe-score-matrix{min-width:760px}" in output
    assert "<div style='overflow-x:auto'><table class='awe-score-matrix'>" in output
    assert "Recorded CI: [60, 90]" in output
    assert "Recorded CI: Not recorded" in output
    assert "Identification bounds" in output and "[50, 75]" in output
    assert "separate from sampling confidence intervals" in output
    assert "Recorded cohort coverage" in output
    assert "<td>5</td>" in output and "<td>7</td>" in output


def test_cross_agent_digest_never_invents_intervals_or_unmeasured_scores():
    value = {"alpha": {"n": 3, "n_failed": 1, "scores": {"code_editing": 70}},
             "beta": {"n": 4, "scores": {"tool_use": 0}}}
    output = _report("insights.json", value)
    assert "alpha" in output and "beta" in output and "awe-score-matrix" in output
    assert "Recorded CI: Not recorded" in output
    assert "[0, 0]" not in output and "[70, 70]" not in output
    assert "Not recorded" in output and "#fee2e2" in output


def test_structured_cohort_discovery_shows_recorded_effects_and_unknown_scope():
    value = [{"agent": "alpha", "n_failed": 7, "n_solved": 8,
              "associations": [{"feature": "ran_no_test", "mean_solved": 0, "mean_failed": 0.4,
                                "diff": 0.4, "lo": 0.1, "hi": 0.7, "p_holm": 0.03, "risk_ratio": None}]}]
    output = _report("discovery/cohort.json", value)
    assert "Recorded discovery comparisons" in output
    assert "Observational associations, not causal effects" in output
    assert "Solved cohort" in output and "Unknown outcomes" in output
    assert "ran_no_test" in output and "[0.1, 0.7]" in output and "0.03" in output
    assert "Not recorded" in output


def test_recorded_behavior_prevalence_is_a_comparison_table():
    value = [{"agent": "alpha", "n": 8,
              "analysis_paralysis": {"n": 0, "rate": 0, "fail_given_mode": None}}]
    output = _report("discovery/discovered_behaviors.json", value)
    assert "Mode count" in output and "Prevalence" in output and "Failure given mode" in output
    assert "analysis_paralysis" in output and "<td>0</td>" in output and "Not recorded" in output
    assert "Prevalence is not detector precision" in output


def test_unknown_json_and_imported_html_remain_escaped_raw_source():
    output = _report("custom.json", {"unknown": "<svg onload=alert(1)>"})
    assert "&lt;svg onload=alert(1)&gt;" in output and "<table class='awe-score-matrix'>" not in output
    html_output = _report("ATTRIBUTION.html", "<script>danger()</script>")
    assert "never executed" in html_output and "<script>" not in html_output


def test_report_rendering_does_not_mutate_stored_data():
    value = {"alpha": {"scores": {"code_editing": None}, "future": {"keep": [1]}}}
    _report("insights.json", value)
    assert value == {"alpha": {"scores": {"code_editing": None}, "future": {"keep": [1]}}}


def test_offline_graph_sources_resolve_and_reused_node_ids_have_unique_anchors(tmp_path):
    from agent_trajviz import awe

    cases = [{"case_id": f"agent__task-{index}", "agent": "agent", "instance_id": f"task-{index}",
              "blame_status": "unattributed", "faults": [],
              "task": {"problem_statement": f"ORIGINAL-SOURCE-{index}"}} for index in range(2)]
    graphs = []
    for index, case in enumerate(cases):
        nodes = [{"id": node_id, "title": node_id, "phase": phase,
                  "source": {"artifact": "faults.json", "json_pointer": f"/{index}/task/problem_statement"}}
                 for node_id, phase in (("task", 1), ("conclusion", 2))]
        edges = [{"from": "task", "to": "conclusion", "relation": "documents"}]
        graphs.append({"case_id": case["case_id"], "agent": "agent", "instance_id": case["instance_id"],
                       "nodes": nodes, "edges": edges, "action_graph": {"nodes": nodes, "edges": edges}})
    directory = tmp_path / "saved"
    directory.mkdir()
    (directory / "faults.json").write_text(json.dumps(cases), encoding="utf-8")
    (directory / "attribution_graphs.json").write_text(json.dumps({"graphs": graphs}), encoding="utf-8")
    run = awe.load_run(directory)
    output = awe.write_report(run, tmp_path / "report.html").read_text(encoding="utf-8")

    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids, self.targets = [], []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if "id" in values:
                self.ids.append(values["id"])
            if tag == "a" and values.get("href", "").startswith("#"):
                self.targets.append(values["href"][1:])

    parsed = Anchors()
    parsed.feed(output)
    assert len(parsed.ids) == len(set(parsed.ids))
    assert len({target for target in parsed.targets if target.startswith("awe-node-")}) == 8
    assert set(parsed.targets) <= set(parsed.ids)
    values = re.findall(r"<summary>Original artifact value</summary><pre>(.*?)</pre>", output, re.DOTALL)
    resolved = [json.loads(html.unescape(value)) for value in values]
    assert len(resolved) == 16 and all(value["available"] is True for value in resolved)
    assert {value["value"] for value in resolved} == {"ORIGINAL-SOURCE-0", "ORIGINAL-SOURCE-1"}
