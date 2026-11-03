"""Saved showcase text never becomes executable markup or fabricated evidence."""
from __future__ import annotations

import copy
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agent_trajviz.awe.runtime import EvaluationSettings, run_operation
from agent_trajviz.awe.showcase import render_showcase


def saved_showcase():
    return {
        "dataset": {"benchmark": "Local regression tasks", "n_tasks": 2, "n_runs": 2},
        "insights": {"custom_pi": {"n": 2, "n_failed": 1, "failure_rate": 0.5,
                                   "scores": {"code_editing": 78.5, "tool_use": None},
                                   "ci95": {"code_editing": [65.0, 91.0]}}},
        "agents_meta": {"custom_pi": {"name": "Custom Pi"}},
        "cases": [{"agent": "custom_pi", "instance_id": "regression-case", "repo": "scratch",
                   "blame_status": "attributed", "task": {"problem_statement": "Repair a real regression"},
                   "faults": [{"error_type": "verification_gap", "is_primary": True, "blame_weight": 1,
                               "strength": "associational", "links": [{"text": "No focused test was run"}],
                               "code_evidence": [{"code": "assert repaired()"}],
                               "counterfactual": {"status": "proposed"}}]}],
        "countermeasures": [{"id": "focused-test", "summary": "Add focused regression test",
                             "status": "proposed", "empirical_support": 0,
                             "references": [{"id": "https://example.org/study", "title": "Recorded source"}]}],
        "validation": {"evidence_mismatches": 1, "notes": "Only supplied evidence was checked"},
    }


class MarkupAudit(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.tags = []
        self.text = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        assert tag not in {"script", "iframe", "object", "embed", "img"}
        for key, value in attrs:
            assert not key.lower().startswith("on")
            if key in {"href", "src"}:
                assert urlsplit(value).scheme in {"https", "http"}

    def handle_data(self, value):
        self.text.append(value)


def test_saved_text_and_references_are_escaped_in_every_output_context():
    data = saved_showcase()
    attack = "</script><img src=x onerror='alert(1)'><svg onload=alert(2)> & \""
    data["dataset"]["benchmark"] = attack
    data["agents_meta"]["custom_pi"]["name"] = attack
    data["insights"]["custom_pi"]["ci95"]["code_editing"] = attack
    data["cases"][0].update(instance_id=attack, agent=attack, repo=attack, blame_status=attack)
    fault = data["cases"][0]["faults"][0]
    fault.update(error_type=attack, strength=attack, audit=attack, links=[{"text": attack}],
                 code_evidence=attack, counterfactual=attack, recommendation=attack)
    entry = data["countermeasures"][0]
    entry.update(summary=attack, status=attack, artifact=attack,
                 references=[{"id": "javascript:alert(1)", "title": attack},
                             {"id": "https://example.org/' onmouseover='alert(1)", "title": attack},
                             {"id": "https://user:password@example.org/", "title": attack},
                             {"id": "https://[malformed", "title": attack}])
    data["validation"]["fp"] = attack
    original = copy.deepcopy(data)
    output = render_showcase(data)
    audit = MarkupAudit(output)
    assert "svg" in audit.tags and "details" in audit.tags
    assert "&lt;img" in output and attack in "".join(audit.text)
    assert "href='javascript:" not in output and "user:password@" not in output
    assert data == original


@pytest.mark.parametrize("location", ["runs", "properties", "failed", "mismatches", "support", "score", "weight"])
def test_untrusted_numeric_metadata_is_rejected(location):
    data = saved_showcase()
    invalid = "<img src=x onerror=alert(1)>"
    if location == "runs":
        data["dataset"]["n_runs"] = invalid
    elif location == "properties":
        data["dataset"]["task_props"] = {"multi_file": invalid}
    elif location == "failed":
        data["insights"]["custom_pi"]["n_failed"] = invalid
    elif location == "mismatches":
        data["validation"]["evidence_mismatches"] = invalid
    elif location == "support":
        data["countermeasures"][0]["empirical_support"] = invalid
    elif location == "score":
        data["insights"]["custom_pi"]["scores"]["code_editing"] = invalid
    else:
        data["cases"][0]["faults"][0]["blame_weight"] = invalid
    with pytest.raises(ValueError, match="finite nonnegative"):
        render_showcase(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, 101])
def test_score_range_and_finite_number_validation(value):
    data = saved_showcase()
    data["insights"]["custom_pi"]["scores"]["code_editing"] = value
    with pytest.raises(ValueError):
        render_showcase(data)


def test_dynamic_scorecards_preserve_recorded_evidence_without_invented_findings():
    output = render_showcase(saved_showcase())
    text = "".join(MarkupAudit(output).text)
    assert "Custom Pi" in text and "78.5" in text and "[65.0, 91.0]" in text
    assert "Repair a real regression" in text and "No focused test was run" in text
    assert "assert repaired()" in text and "associational" in text
    assert "Not recorded" in text and "evidence mismatches1" in text
    assert "7/7" not in text and "0 mismatches" not in text
    assert "Claude Code" not in text and "<script" not in output


def test_offline_showcase_operation_preserves_original_input_bytes(tmp_path):
    source = tmp_path / "supplied-showcase.json"
    data = saved_showcase()
    data["dataset"]["benchmark"] = "<img src=x onerror=alert(1)>"
    source.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    original = source.read_bytes()
    result = run_operation("showcase", EvaluationSettings(output_root=tmp_path / "jobs"),
                           payload={"showcase_data": str(source)})
    assert result.success, (result.reason, result.logs)
    output = Path(result.output_dir) / "showcase_results.html"
    MarkupAudit(output.read_text(encoding="utf-8"))
    assert "&lt;img" in output.read_text(encoding="utf-8")
    assert source.read_bytes() == original
    manifest = json.loads((Path(result.output_dir) / "operation-manifest.json").read_text())
    assert manifest["network_enabled"] is False
    assert "showcase_results.html" in manifest["artifact_sha256"]


def test_invalid_counts_fail_the_actual_operation_before_writing_html(tmp_path):
    data = saved_showcase()
    data["dataset"]["n_runs"] = "<img src=x onerror=alert(1)>"
    result = run_operation("showcase", EvaluationSettings(output_root=tmp_path / "jobs"),
                           payload={"showcase_data": data})
    assert not result.success
    assert not list((tmp_path / "jobs").rglob("showcase_results.html"))
