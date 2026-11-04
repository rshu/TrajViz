"""Optional saved fields degrade honestly without hiding the original record."""

from __future__ import annotations

import copy

import pytest

from i18n_test_helpers import unmark_html

from agent_trajviz.awe.render import case_html
from agent_trajviz.awe.saved import load_run
from agent_trajviz.awe.summary import attribution_summary_html, primary_fault


def _case():
    return {"case_id": "agent__task", "agent": "agent", "instance_id": "task", "resolved": False,
            "faults": [{"capability": "code_editing", "error_type": "incorrect_patch", "blame_weight": 1,
                        "is_primary": True, "evidence_chain": {
                            "links": [{"kind": "observation", "statement": "Recorded gating test failed.",
                                       "source": "eval_outcome"}],
                            "counterfactual": {"status": "not_tested", "what_would_confirm": "Rerun the gating test."},
                        }}]}


def _render_without_mutation(case):
    before = copy.deepcopy(case)
    run = load_run(case)
    imported = copy.deepcopy(run.cases)
    out = unmark_html(case_html(run.cases[0]))
    assert case == before and run.cases == imported
    assert "Complete original case" in out
    return out


@pytest.mark.parametrize("value", [1, True, "future format", {"future": "shape"}, None])
def test_imported_optional_scorecard_shape_does_not_break_case_view(value):
    case = _case()
    case["scorecard"] = value
    out = _render_without_mutation(case)
    assert "Code Editing: incorrect patch." in out
    assert "Recorded gating test failed." in out
    assert "Capability assessment coverage was not recorded." in out
    assert "0 of 0 capabilities" not in out


@pytest.mark.parametrize("value", [1, True, "future format", {"future": "shape"}, None])
def test_imported_optional_links_shape_does_not_break_case_view(value):
    case = _case()
    case["faults"][0]["evidence_chain"]["links"] = value
    out = _render_without_mutation(case)
    summary = unmark_html(attribution_summary_html(case))
    assert "Code Editing: incorrect patch." in summary
    assert "Expand the recorded evidence below" in summary
    assert "Recorded gating test failed." not in summary
    assert "Evidence chain, qualifications and checks" in out


@pytest.mark.parametrize("field", ["capability", "error_type"])
@pytest.mark.parametrize("value", [[], {}, 1, None, ""])
def test_malformed_fault_identity_does_not_become_a_primary_finding(field, value):
    case = _case()
    case["faults"][0][field] = value
    out = _render_without_mutation(case)
    assert "At a glance" in out
    assert primary_fault(case) is None
    assert "No single dominant fault" in unmark_html(attribution_summary_html(case))
    case["primary"] = {"capability": case["faults"][0]["capability"], "error_type": case["faults"][0]["error_type"]}
    assert primary_fault(case) is None


@pytest.mark.parametrize("value", [1, True, "future format", {"future": "shape"}, None])
def test_summary_helpers_accept_absent_or_unrecognized_fault_container(value):
    case = {"resolved": False, "faults": value}
    assert primary_fault(case) is None
    assert "No single dominant fault" in unmark_html(attribution_summary_html(case))


def test_valid_arrays_keep_recorded_evidence_and_coverage():
    case = _case()
    case["scorecard"] = [{"capability": "code_editing", "assessed": True},
                         {"capability": "task_planning", "assessed": False}]
    out = _render_without_mutation(case)
    assert "Code Editing: incorrect patch." in out
    assert "Recorded gating test failed." in out
    assert "2 capabilities" in out
    assert "1 of 2 capabilities assessed; 1 unassessed." in out
    assert "Recorded check (not tested): Rerun the gating test." in out


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, {}, []])
def test_unrecognized_assessment_values_do_not_invent_coverage(value):
    case = _case()
    case["scorecard"] = [{"capability": "code_editing", "assessed": value}]
    out = _render_without_mutation(case)
    assert "Capability assessment coverage was not recorded." in out
    assert "capabilities assessed;" not in unmark_html(attribution_summary_html(case))


def test_partial_valid_scorecard_counts_only_boolean_assessments():
    case = _case()
    case["scorecard"] = [{"capability": "code_editing", "assessed": True},
                         {"capability": "task_planning", "assessed": False},
                         {"capability": "tool_use", "assessed": "false"},
                         {"assessed": True}]
    out = _render_without_mutation(case)
    assert "1 of 2 capabilities assessed; 1 unassessed." in out


@pytest.mark.parametrize("value", [True, False])
def test_boolean_weight_never_promotes_a_fault(value):
    case = _case()
    case["faults"][0]["blame_weight"] = value
    out = _render_without_mutation(case)
    assert primary_fault(case) is None
    assert "No single dominant fault" in unmark_html(attribution_summary_html(case))
    assert "Code Editing: incorrect patch." not in out
