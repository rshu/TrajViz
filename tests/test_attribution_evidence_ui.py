"""Evidence rendering works independently of model configuration."""

import pytest

from agent_trajviz.insight.rendering import build_attribution_html


def _rendered_evidence(**changes):
    """Small public-result shapes; no corpus records or external backend required."""
    chain = {
        "strength": "deductive",
        "strength_basis": "Per-test outcomes joined to the final patch.",
        "links": [{"kind": "observation", "source": "test_report",
                   "statement": "A gating test failed.",
                   "quotes": [{"step": 60, "text": "The reported failure."}]}],
        "counterfactual": {"status": "not_tested", "what_would_confirm": "Apply the proposed change and rerun the gating test."},
    }
    chain.update(changes)
    return build_attribution_html({
        "available": True,
        "blame_status": "primary",
        "primary": {"capability": "code_editing", "error_type": "partial_fix_only"},
        "scorecard": [],
        "faults": [{"capability": "code_editing", "error_type": "partial_fix_only",
                    "blame_weight": 1.0, "is_primary": True, "evidence_chain": chain}],
    })


def test_evidence_step_numbers_have_a_distinct_namespace_without_workflow_links():
    out = _rendered_evidence()
    assert "AWE step 60" in out
    assert "Evidence step references use AWE numbering" in out
    assert "may differ from Workflow step numbers" in out
    assert "href=" not in out


def test_evidence_basis_and_untested_check_are_accessible_details():
    out = _rendered_evidence()
    assert "<details class='attr-evidence-detail'>" in out
    assert "Evidence basis</summary>" in out
    assert "Per-test outcomes joined to the final patch." in out
    assert "Proposed counterfactual check &mdash; Not tested" in out
    assert "Apply the proposed change and rerun the gating test." in out
    assert "A gating test failed." in out


def test_evidence_context_escapes_backend_prose_and_status():
    out = _rendered_evidence(strength_basis="<script>basis</script>",
                             counterfactual={"status": "<img src=x>", "what_would_confirm": "<script>check</script>"})
    assert "<script>" not in out and "<img src=x>" not in out
    assert "&lt;script&gt;basis&lt;/script&gt;" in out
    assert "&lt;script&gt;check&lt;/script&gt;" in out
    assert "&lt;img src=x&gt;" in out


@pytest.mark.parametrize("basis,counterfactual", [(None, None), ("", {}), ([], []), (123, {"status": "not_tested", "what_would_confirm": None})])
def test_missing_evidence_context_does_not_create_empty_details(basis, counterfactual):
    out = _rendered_evidence(strength_basis=basis, counterfactual=counterfactual)
    assert "attr-evidence-detail" not in out
    assert "A gating test failed." in out


def test_counterfactual_without_status_does_not_claim_it_was_tested():
    out = _rendered_evidence(counterfactual={"what_would_confirm": "Rerun the test after the proposed change."})
    assert "Proposed counterfactual check &mdash; Status not provided" in out
    assert "Rerun the test after the proposed change." in out
