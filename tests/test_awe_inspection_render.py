"""Inspection views preserve observations and link only mapped canonical rows."""
from __future__ import annotations

import copy
from html.parser import HTMLParser

import pytest

from agent_trajviz.awe.inspection_render import inspection_html


class _Buttons(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []

    def handle_starttag(self, tag, attrs):
        if tag == "button":
            self.buttons.append(dict(attrs))


def _result():
    return {
        "schema_version": 1, "mode": "trajectory_only", "available": True,
        "source": {"name": "session.jsonl", "format": "codex"},
        "workflow_indices": [0, 4, 9],
        "summary": {"steps": 10, "tool_calls": 6, "failed_calls": 1, "files": 2, "test_runs": 1},
        "outcome": {"status": "unknown", "basis": "No independent evaluator outcome was supplied."},
        "findings": [{
            "id": "failure-1", "kind": "tool_failure", "severity": "warning", "title": "A command failed",
            "observation": "The recorded test process exited with code 1.", "next_check": "Review the assertion failure.",
            "step_indices": [4],
            "evidence": [{"workflow_index": 4, "part_index": 2, "thread_id": "main", "call_id": "call-1",
                          "output_excerpt": "1 failed, 12 passed"}],
        }],
        "files": [{"path": "src/module.py", "actions": ["read", "edit"], "step_indices": [0, 4], "threads": ["main"]}],
        "tests": [{"workflow_index": 4, "command": "pytest tests/test_module.py", "status": "failed", "exit_code": 1,
                   "counts": {"failed": 1, "passed": 12}, "summary_status": "reported failures", "output_excerpt": "1 failed, 12 passed"}],
        "failures": [{"workflow_index": 4, "tool_name": "shell", "command": "pytest tests/test_module.py", "status": "failed",
                      "exit_code": 1, "output_excerpt": "1 failed, 12 passed"}],
        "recovery": [{"workflow_index": 4, "step_indices": [4, 9], "command": "pytest tests/test_module.py",
                      "lifecycle": "failed_then_later_success", "recovery_workflow_index": 9, "output_excerpt": "Later process exited 0."}],
        "timing": [{"workflow_index": 9, "duration_seconds": 21.5, "timing_source": "model_request", "incomplete": True}],
        "coverage": {"tool_results": {"observed": 6, "total": 6}},
        "limitations": ["A command exit status does not establish task correctness."],
        "ledger": {"events": [{"step_id": 800, "workflow_index": 4, "kind": "tool_call"}]},
        "symptoms": {"observations": [{"step_id": 900, "kind": "recorded retry"}]},
    }


def test_reading_guide_preserves_evidence_then_collapses_full_records():
    data = _result()
    before = copy.deepcopy(data)
    out = inspection_html(data)
    assert out.index("At a glance") < out.index("Recorded findings") < out.index("Recorded file targets")
    assert "The recorded test process exited with code 1." in out
    assert "Review the assertion failure." in out and "1 failed, 12 passed" in out
    assert "No independent evaluator outcome was supplied." in out
    assert "A command exit status does not establish task correctness." in out
    assert "src/module.py" in out and "model_request" in out and "Incomplete" in out
    assert "Action and result ledger" in out and "Recorded symptom observations" in out
    assert "File targets" in out and "Test-related calls" in out
    assert "Recorded test activity" in out
    assert "An attempted call does not confirm that a file changed." in out
    assert "<details class='ai-panel' open>" not in out
    assert data == before


def test_only_allowlisted_canonical_indices_produce_workflow_buttons():
    data = _result()
    data["findings"][0]["step_indices"] = [0, 4, 800, -1, True, "9", "4);alert(1)"]
    data["workflow_indices"] = [0, 4, -1, True, "9"]
    out = inspection_html(data)
    parser = _Buttons()
    parser.feed(out)
    assert parser.buttons
    assert {button["aria-label"] for button in parser.buttons} == {"Open Workflow step 0", "Open Workflow step 4"}
    assert all(button["onclick"] in {
        "if(typeof window.tvGotoWorkflowStep==='function'){window.tvGotoWorkflowStep(0);}",
        "if(typeof window.tvGotoWorkflowStep==='function'){window.tvGotoWorkflowStep(4);}",
    } for button in parser.buttons)
    assert "Open Workflow step 800" not in out and "Open Workflow step 900" not in out
    assert "window.tvGotoWorkflowStep(9)" not in out
    assert "window.tvGotoWorkflowStep(800)" not in out
    assert "window.tvGotoWorkflowStep(900)" not in out


def test_large_citation_sets_are_compact_without_losing_any_mapped_steps():
    data = _result()
    data["workflow_indices"] = list(range(12))
    data["findings"][0]["step_indices"] = list(range(12))
    out = inspection_html(data)
    glance = out.split("aria-label='At a glance'>", 1)[1].split("</section>", 1)[0]
    assert glance.count("<button") == 4
    assert "+8 more" in glance
    article = out.split("<article", 1)[1].split("</article>", 1)[0]
    default_steps, expanded = article.split("<details class='ai-cited-steps'>", 1)
    assert default_steps.count("<button") == 8
    more_steps = expanded.split("</details>", 1)[0]
    assert "All cited steps (12)" in more_steps
    assert more_steps.count("<button") == 4
    for index in range(8, 12):
        assert f"Open Workflow step {index}" in more_steps
    assert "<details class='ai-cited-steps' open" not in out
    assert ".trajectory-viz .awe-inspection button.ai-step" in out


def test_exported_inspection_retains_step_labels_without_live_navigation():
    out = inspection_html(_result(), workflow_links=False)
    assert "Step 4" in out and "Step 9" in out
    assert "onclick=" not in out and "<button" not in out


def test_test_result_links_preserve_the_separate_observed_result_step():
    data = _result()
    data["tests"][0]["result_workflow_index"] = 9
    out = inspection_html(data)
    assert "Result: <button type='button' class='ai-step' aria-label='Open Workflow step 9'" in out
    data["workflow_indices"] = [4]
    out = inspection_html(data)
    assert "Result: <span class='ai-step'>Step 9</span>" in out
    assert "window.tvGotoWorkflowStep(9)" not in out


def test_missing_link_allowlist_never_guesses_workflow_targets():
    data = _result()
    del data["workflow_indices"]
    out = inspection_html(data)
    assert "Step 4" in out
    assert "onclick=" not in out


def test_unknown_test_counts_explain_why_the_summary_is_not_assessed():
    data = _result()
    data["tests"][0].update(status="unknown", exit_code=0, summary_status="unknown", counts=None,
                             reason="compound_command")
    out = inspection_html(data)
    assert "Reason: compound command" in out
    assert "Recorded process result" in out and "Observed test result" in out
    assert "<td>Exit code 0</td>" in out
    assert "<td>Not determined</td>" in out
    assert "Process status" not in out


def test_output_summary_text_is_separate_from_verified_command_counts():
    data = _result()
    data["tests"][0].update(status="output_summary_only", exit_code=0, process_status="zero_exit",
                             summary_status="unknown", counts=None, reason="compound_command",
                             output_summary={"status": "reported", "runner": "pytest", "counts": {"passed": 24},
                                             "basis": "unattributed_output_text"})
    out = inspection_html(data)
    assert "<td>Output text only</td>" in out
    assert "<td>Exit code 0</td>" in out
    assert "Reason: compound command" in out
    assert "Summary text in output (command attribution unknown)" in out
    assert "&quot;passed&quot;: 24" in out
    assert "<td>Passed</td>" not in out


def test_all_recorded_strings_are_escaped_in_prose_tables_details_and_sources():
    data = _result()
    hostile = "<img src=x onerror=alert(1)> ' \" </script>"
    data["source"] = {"name": hostile, "format": hostile}
    finding = data["findings"][0]
    for key in ("title", "observation", "next_check"):
        finding[key] = hostile
    finding["evidence"][0].update(thread_id=hostile, call_id=hostile, output_excerpt=hostile)
    data["files"][0]["path"] = hostile
    data["tests"][0].update(command=hostile, output_excerpt=hostile, reason=hostile)
    data["ledger"] = {hostile: hostile, "step_id": hostile}
    data["limitations"] = [hostile]
    out = inspection_html(data)
    assert "<img" not in out and "</script>" not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "&#x27;" in out and "&quot;" in out
    assert "window.tvGotoWorkflowStep(4)" in out
    parser = _Buttons()
    parser.feed(out)
    assert all("alert" not in button["onclick"] for button in parser.buttons)


@pytest.mark.parametrize("value", [None, 1, True, "future shape", {}, [1, "bad", None]])
def test_optional_container_shapes_do_not_crash_or_invent_findings(value):
    data = {"available": True, "summary": value, "source": value, "outcome": value,
            "findings": value, "files": value, "tests": value, "failures": value,
            "recovery": value, "timing": value, "coverage": value, "limitations": value,
            "workflow_indices": value}
    out = inspection_html(data)
    assert "At a glance" in out
    assert "No ranked findings are available" in out
    assert "Independent task or evaluator evidence has not been supplied." in out
    assert "onclick=" not in out


@pytest.mark.parametrize("count", [None, -1, True, "0", {}, []])
def test_unknown_counts_are_not_displayed_as_zero_successes(count):
    out = inspection_html({"available": True, "summary": {"failed_calls": count}})
    assert "<strong>—</strong><span>Recorded failures</span>" in out
    body = out.split("</style>", 1)[1]
    assert "error-free" not in body and "100%" not in body


def test_no_loaded_session_is_a_compact_next_action():
    out = inspection_html({"available": False})
    assert "Load a trajectory above" in out
    assert "Recorded findings" not in out and "<table>" not in out
    assert "reference" not in out.lower()


def test_unavailable_reason_is_recorded_as_text():
    out = inspection_html({"available": False, "reason": "<script>unavailable</script>"})
    assert "<script>" not in out and "&lt;script&gt;unavailable&lt;/script&gt;" in out
