"""Saved case summaries prioritize recorded insight and preserve original evidence."""
from __future__ import annotations

import copy
import html
import json
from html.parser import HTMLParser

from agent_trajviz.awe import render


class _SummaryText(HTMLParser):
    """Read user-visible summary text without its original-record disclosures."""
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.root = None
        self.parts = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            return
        self.depth += 1
        if "awe-attribution-summary" in dict(attrs).get("class", "").split():
            self.root = self.depth

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if self.root == self.depth:
            self.root = None
        self.depth -= 1

    def handle_data(self, data):
        if self.root is not None:
            self.parts.append(data)

    @property
    def text(self):
        return " ".join(self.parts)


def _saved_case():
    # Saved dossiers identify their primary in faults, rather than a top-level
    # primary object. Keep this representative shape distinct from native UI results.
    return {
        "agent": "example_agent",
        "instance_id": "example_task",
        "resolved": False,
        "blame_status": "primary",
        "task": {"problem_statement": "Repair the sample behavior.", "gold_files": ["sample.py"]},
        "faults": [{
            "capability": "code_editing", "error_type": "incorrect_patch",
            "is_primary": True, "blame_weight": 1.0,
            "evidence_chain": {
                "strength": "deductive",
                "causal_support": {"status": "not_tested"},
                "links": [{"kind": "observation", "source": "eval_outcome", "statement": "A gating test failed."}],
                "counterfactual": {"status": "not_tested", "what_would_confirm": "Rerun the gating test after the proposed change."},
            },
        }],
    }


def test_saved_case_leads_with_recorded_diagnosis_evidence_and_check():
    case = _saved_case()
    original = copy.deepcopy(case)
    output = render.case_html(case)
    summary = _SummaryText(output).text
    assert "Code Editing" in summary
    assert "incorrect patch" in summary.lower()
    assert "A gating test failed." in summary
    assert "Rerun the gating test after the proposed change." in summary
    assert "not tested" in summary.lower()
    assert output.index("awe-attribution-summary") < output.index("Task and reference context</summary>")
    assert "<details class='awe-panel'><summary>Task and reference context</summary>" in output
    assert "Raw task artifact" in output and "Complete original case" in output
    assert json.dumps(case, ensure_ascii=False, indent=2) in html.unescape(output)
    assert case == original


def test_saved_summary_does_not_treat_absent_evidence_as_a_tested_remedy():
    case = _saved_case()
    del case["faults"][0]["evidence_chain"]
    summary = _SummaryText(render.case_html(case)).text
    assert "A gating test failed." not in summary
    assert "Rerun the gating test after the proposed change." not in summary
    assert "expect pass" not in summary.lower()
    assert "100%" not in summary


def test_saved_summary_keeps_recorded_html_inert():
    case = _saved_case()
    unsafe = "<script>fixture()</script><img src=x onerror=fixture()>"
    chain = case["faults"][0]["evidence_chain"]
    chain["links"][0]["statement"] = unsafe
    chain["counterfactual"]["what_would_confirm"] = unsafe
    output = render.case_html(case)
    summary = _SummaryText(output).text
    assert unsafe in summary
    assert "<script>" not in output and "<img" not in output
