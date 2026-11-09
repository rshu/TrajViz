"""Readable attribution summaries stay grounded in recorded evidence."""
import copy

import pytest

from i18n_test_helpers import unmark_html

from agent_trajviz.awe.summary import attribution_summary_html, primary_fault
from agent_trajviz.insight.rendering import build_attribution_html


def record():
    return {
        "available": True,
        "primary": {"capability": "code_editing", "error_type": "incorrect_patch"},
        "blame_status": "primary",
        "scorecard": [
            {"capability": "code_editing", "assessed": True, "blamed": True, "weight": 1.0,
             "tier": "deductive", "top_error": "incorrect_patch"},
            {"capability": "code_localization", "assessed": True, "blamed": False},
            {"capability": "task_planning", "assessed": False, "blamed": False},
        ],
        "faults": [{"capability": "code_editing", "error_type": "incorrect_patch", "is_primary": True,
                    "blame_weight": 1.0, "evidence_chain": {
                        "links": [{"kind": "observation", "source": "test_report", "statement": "The gating test failed."},
                                  {"kind": "inference", "statement": "The change caused a regression."}],
                        "counterfactual": {"status": "not_tested", "what_would_confirm": "Rerun the gating test after the change."},
                    }}],
        "dossier": {"task": {"problem_statement": "Repair the reported regression."}},
    }


def test_summary_is_first_reading_guide_with_recorded_support_and_check():
    data = record()
    before = copy.deepcopy(data)
    out = unmark_html(build_attribution_html(data))
    summary = unmark_html(attribution_summary_html(data))
    assert "At a glance" in out
    assert "Code Editing: incorrect patch." in summary
    assert "The gating test failed." in summary and "test_report" in summary
    assert "The change caused a regression." not in summary
    assert "Recorded check (not tested): Rerun the gating test after the change." in summary
    assert "2 of 3 capabilities assessed; 1 unassessed." in summary
    assert "Not assessed does not mean passed." in summary
    assert "Attribution weights are not probabilities." in summary
    assert out.index("At a glance") < out.index("score-dim-grid") < out.index("Task and reference context")
    assert out.index("At a glance") < out.index("Primary cause:")
    assert data == before


def test_summary_retains_every_recorded_observation_including_late_test_failure():
    data = record()
    statements = ["The reference changes one file.", "The agent edited the correct file.",
                  "Content overlap is 0%.", "Changed-region overlap is 60%.", "The gating tests failed."]
    data["faults"][0]["evidence_chain"]["links"] = [
        {"kind": "observation", "statement": statement} for statement in statements
    ] + [{"kind": "inference", "statement": "Unproven causal claim."}]
    summary = unmark_html(attribution_summary_html(data))
    assert "<ul " in summary and summary.count("<li ") == len(statements)
    for statement in statements:
        assert statement in summary
    assert "Unproven causal claim." not in summary


def test_raw_context_is_preserved_and_collapsed():
    out = unmark_html(build_attribution_html(record()))
    assert "Repair the reported regression." in out
    assert "<details class='attr-evidence-detail' open>" not in out
    assert "Task and reference context" in out
    assert "no fault found" in out and "insufficient evidence or no opportunity" in out
    assert ">clean<" not in out and ">n/a<" not in out


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), "invalid"])
def test_zero_or_invalid_weights_never_promote_a_primary_summary(weight):
    data = record()
    data["faults"][0]["blame_weight"] = weight
    summary = unmark_html(attribution_summary_html(data))
    assert primary_fault(data) is None
    assert "No single dominant fault" in summary
    assert "Code Editing: incorrect patch." not in summary
    assert "The gating test failed." not in summary


def test_zero_weight_primary_marker_does_not_show_as_primary_in_fault_panel():
    data = record()
    data["faults"][0]["blame_weight"] = 0
    out = unmark_html(build_attribution_html(data))
    assert "Primary cause:" not in out
    assert "★ primary" not in out
    assert "No single dominant cause" in out


def test_saved_primary_marker_is_accepted_but_explicit_abstention_is_retained():
    data = record()
    del data["primary"]
    assert primary_fault(data) == data["faults"][0]
    data["primary"] = None
    assert primary_fault(data) is None
    assert "No single dominant fault" in unmark_html(attribution_summary_html(data))


def test_solved_outcome_does_not_promote_a_stale_fault():
    data = record()
    data["resolved"] = True
    summary = unmark_html(attribution_summary_html(data))
    assert primary_fault(data) is None
    assert "The recorded outcome is solved." in summary
    assert "Code Editing: incorrect patch." not in summary
    assert "The gating test failed." not in summary


def test_missing_primary_check_and_coverage_are_explicit():
    out = unmark_html(attribution_summary_html({}))
    assert "outcome is unknown" in out
    assert "no confirming check was recorded" in out
    assert "coverage was not recorded" in out
    assert "0 of 0" not in out
    assert "fault attribution" in unmark_html(attribution_summary_html({"analysis": {"symptom": "retries"}}))


def test_summary_escapes_all_recorded_prose_and_identifiers():
    data = record()
    data["faults"][0]["evidence_chain"] = {
        "links": [{"kind": "observation", "statement": "<script>bad</script>", "source": "<img src=x>"}],
        "counterfactual": {"status": "<img src=y>", "what_would_confirm": "<script>check</script>"},
    }
    out = unmark_html(attribution_summary_html(data))
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;bad&lt;/script&gt;" in out
    assert "&lt;img src=x&gt;" in out and "&lt;img src=y&gt;" in out


def test_unavailable_diagnosis_explains_next_action_without_scores():
    out = unmark_html(build_attribution_html({"available": False, "reason": "Missing <reference>"}))
    assert "Diagnosis unavailable" in out and "Missing &lt;reference&gt;" in out
    assert "Task and reference data above" in out and "Diagnose failure" in out
    assert "Trajectory-only diagnostics remain available in Overview" in out
    assert "score-dim-grid" not in out and "At a glance" not in out
