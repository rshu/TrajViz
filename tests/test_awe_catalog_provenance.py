"""Historical catalog observations must not masquerade as current-run evidence."""
import copy

from agent_trajviz.awe._engine import config, digest, remedy_plan, report
from agent_trajviz.awe._engine.catalog_notes import historical_calibration_text
from agent_trajviz.awe._engine.countermeasures import countermeasures_for


def test_historical_note_preserves_numbers_and_studies_without_active_internal_links():
    raw = "RR 0.83 / RR 0.95, 2,500 runs; TDFlow, not our cohort data (DESIGN.md §0y). Do not cite our calibration."
    rendered = historical_calibration_text(raw)
    assert "not recomputed for this run" in rendered
    assert "not evidence of current-run remedy efficacy" in rendered
    for retained in ("0.83", "0.95", "2,500", "TDFlow"):
        assert retained in rendered
    assert "our cohort" not in rendered and "our calibration" not in rendered
    assert "DESIGN.md" not in rendered
    assert "DESIGN.md" in raw


def test_plan_and_agent_report_keep_raw_catalog_metadata_but_qualify_display():
    entry = next(e for e in countermeasures_for(error_type="incorrect_patch") if e.id == "CM-EDIT-INCORRECT-1")
    raw = copy.deepcopy(entry.calibration)
    case = {"case_id": "a__i", "agent": "a", "instance_id": "i", "blame_status": "primary",
            "faults": [{"capability": "code_editing", "error_type": "incorrect_patch", "blame_weight": 1,
                        "is_primary": True, "detector_evidence": {"region_overlap": 0.8}}]}
    plan = remedy_plan.build_plan([case])
    candidate = next(c for c in plan["candidates"] if c["id"] == entry.id)
    assert candidate["recipe"]["calibration"] == raw
    assert candidate["recipe"]["calibration_provenance"]["computed_for_current_run"] is False
    assert candidate["recipe"]["calibration_provenance"]["raw_metadata_field"] == "calibration"
    for markdown in (remedy_plan.to_markdown(plan), "\n".join(remedy_plan.insights_section(plan)),
                     "\n".join(report.remedy_lines(entry, 1))):
        assert "Historical" in markdown or "historical" in markdown
        assert "not recomputed for this run" in markdown
        assert "0.83" in markdown
        assert "our cohort" not in markdown
        assert "DESIGN.md" not in markdown
    summary = remedy_plan.showcase_summary(plan)
    assert summary["cautions"][0]["raw_catalog_metadata"]["calibration"] == raw
    assert summary["cautions"][0]["computed_for_current_run"] is False
    assert entry.calibration == raw


def test_digest_current_fault_side_section_has_no_unavailable_internal_reference(monkeypatch):
    monkeypatch.setattr(config, "AGENTS", ["a"])
    markdown = "\n".join(digest._fault_side_section({"a": {"fault_side": {"model": 2}}}))
    assert "bundled fault-side assessment" in markdown
    assert "DESIGN.md" not in markdown
    assert "awe/faultside.py" not in markdown
