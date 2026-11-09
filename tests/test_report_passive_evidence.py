"""Standalone reports retain evidence without live-dashboard navigation targets."""

from __future__ import annotations

import copy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from agent_trajviz.insight.presenters.issues import (
    IssueJudgment,
    OverviewIssue,
    render_overview_issues_html,
)
from agent_trajviz.insight.presenters.overview import build_overview_outputs
from agent_trajviz.insight.presenters.patterns import (
    build_antipattern_html,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from agent_trajviz.insight.rendering import build_root_cause_html, render_agent_summary_cards
from agent_trajviz.insight.report import build_report_html
from agent_trajviz.insight.session import LoadError, load_session


class _Elements(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.handlers, self.classes, self.text = [], set(), []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key.startswith("on"):
                self.handlers.append(value or "")
            if key == "class":
                self.classes.update(value.split())

    def handle_data(self, value):
        self.text.append(value)


@pytest.fixture
def evidence_session():
    path = Path(__file__).parent / "fixtures/native_claude_code/astropy__astropy-13033.jsonl"
    session = load_session(str(path))
    assert not isinstance(session, LoadError)
    # Controlled positive presentation cases supplement the public native run;
    # this tests every report presenter regardless of that run's detected cases.
    session.tool_sequences = [{"sequence": ["Read", "Edit"], "frequency": 3, "step_indices": [0, 7, 12]}]
    session.failure_patterns = [
        {
            "cluster_label": "Fixture tool failure",
            "count": 2,
            "example_error": "<script>fixture()</script>",
            "steps": [7, 12],
            "recovery_path": ["Read", "Edit"],
        }
    ]
    session.fruitless_streaks = [{"start_step": 7, "end_step": 12, "length": 6}]
    session.tool_selection = [{"step": 7}]
    session.plan_metrics = {"stalled": [{"content": "Fixture plan", "start_step": 7, "end_step": 12}]}
    session.clusters = [{"tool": "Bash", "count": 2, "pattern": "fixture failure", "first_step": 7, "last_step": 12}]
    session.show_root_cause = True
    session.agent_summaries[0]["spawned_by_step"] = 7
    return session


def test_actual_export_uses_passive_indices_in_all_reused_evidence_panels(evidence_session):
    before = copy.deepcopy(
        (
            evidence_session.tool_sequences,
            evidence_session.failure_patterns,
            evidence_session.clusters,
            evidence_session.agent_summaries,
        )
    )
    output = build_report_html(evidence_session)
    elements = _Elements(output)
    assert "insight-step-index" in elements.classes
    assert "insight-step-link" not in elements.classes
    assert not any(
        "tvGotoWorkflowStep" in handler
        or "tvFocusWorkflowCard" in handler
        or "'Workflow'" in handler
        or "wf-card-" in handler
        for handler in elements.handlers
    )
    assert "click a step to open Workflow" not in output
    text = "".join(elements.text)
    for index in [0, 7, 12]:
        assert f"#{index}" in text
    assert "Spawned at step #7" in text
    assert "steps 7..12" in text
    assert "step numbers refer to the recorded trajectory" in text
    assert "&lt;script&gt;fixture()&lt;/script&gt;" in output
    assert (
        evidence_session.tool_sequences,
        evidence_session.failure_patterns,
        evidence_session.clusters,
        evidence_session.agent_summaries,
    ) == before


def test_live_presenters_keep_workflow_navigation_by_default(evidence_session):
    panels = [
        render_tool_sequences_html(evidence_session.tool_sequences),
        render_failure_patterns_html(evidence_session.failure_patterns),
        build_antipattern_html(evidence_session),
        build_root_cause_html(evidence_session.clusters),
        render_agent_summary_cards(evidence_session.agent_summaries),
        build_overview_outputs(evidence_session)["issues_html"],
    ]
    assert all("tvGotoWorkflowStep(7)" in output for output in panels)
    assert "click a step to open Workflow" in panels[-1]


@pytest.mark.parametrize("judged", [False, True])
def test_passive_issues_hint_and_fix_details_preserve_evidence(judged):
    judgment = (
        IssueJudgment(where="fixture.py:7", fix="Check the recorded failure", confidence="high") if judged else None
    )
    issue = OverviewIssue(
        kind="error",
        title="Fixture failure",
        detail="Recorded error",
        why="Recorded recovery",
        steps=(7, 12),
        judgment=judgment,
    )
    output = render_overview_issues_html([issue], workflow_links=False)
    elements = _Elements(output)
    assert not elements.handlers
    assert "Workflow" not in output
    assert "#7" in output and "#12" in output
    assert "step numbers refer to the recorded trajectory" in output
    if judged:
        assert "Check the recorded failure" in output and "fixture.py:7" in output
