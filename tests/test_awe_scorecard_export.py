"""Scorecard JSON must be the exact computation rendered in Markdown."""
import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

from agent_trajviz.awe._engine import report, runs


def test_generator_exports_the_single_computed_scorecard_and_coverage(tmp_path, monkeypatch):
    scores = {
        "agent": "agent", "n_requested": 3, "n_tasks": 2, "n_failed": 1, "failure_rate": 0.5,
        "unattributed": 0, "unscored_instances": [{"instance_id": "unattempted", "reason": "unknown outcome"}],
        "capabilities": {
            "code_editing": {"score": 62.5, "score_ci": (25.0, 88.0), "rate": 0.375,
                             "opportunity": 2, "fault_incidence": 0.5,
                             "unattributed_opportunity_mass": 0.0,
                             "score_identification_bounds": (62.5, 62.5), "top_errors": []},
        },
    }
    before = copy.deepcopy(scores)
    scorer = Mock(return_value=scores)
    monkeypatch.setattr(report, "agent_capability_scores", scorer)
    monkeypatch.setattr(report, "_examples", lambda *args: ({}, Counter()))
    monkeypatch.setattr(report, "strength_composition", lambda *args: {"by_mass": {"deductive": 1, "associational": 0, "model_inferred": 0}})
    monkeypatch.setattr(report, "tool_behavior", lambda *args: {"n_tool_calls": 0})
    script = Path(report.__file__).parent / "scripts/gen_report.py"
    spec = importlib.util.spec_from_file_location("scorecard_generator_test", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    monkeypatch.setattr(generator, "OUT", tmp_path)
    monkeypatch.setattr(generator, "pipeline_args", lambda: object())
    monkeypatch.setattr(generator, "selected_inputs", lambda args: (["agent"], ["one", "two", "unattempted"]))
    generator.main()
    scorer.assert_called_once_with("agent", ["one", "two", "unattempted"], use_trajectory=True)
    assert scores == before
    stored = json.loads((tmp_path / "agent.json").read_text())
    assert stored["capabilities"] == json.loads(json.dumps(before["capabilities"]))
    assert stored["n_requested"] == 3 and stored["n_tasks"] == 2
    assert stored["unscored_instances"] == before["unscored_instances"]
    assert stored["instance_ids"] == ["one", "two", "unattempted"]
    assert "task_planning" not in stored["capabilities"]
    markdown = (tmp_path / "agent.md").read_text()
    assert "62.5 [25, 88]" in markdown
    assert "2 tasks" in markdown


def test_snapshot_keeps_scorecard_json_and_markdown(tmp_path, monkeypatch):
    dossier, reports = tmp_path / "dossier", tmp_path / "reports"
    dossier.mkdir()
    reports.mkdir()
    (dossier / "INSIGHTS.md").write_text("Recorded cohort")
    payload = {"agent": "agent", "capabilities": {"code_editing": {"score": 62.5, "score_ci": [25, 88]}}}
    (reports / "agent.json").write_text(json.dumps(payload))
    (reports / "agent.md").write_text("62.5 [25, 88]")
    (reports / "unrelated.tmp").write_text("not a report")
    monkeypatch.setattr(runs, "_manifest", lambda path: {"agents": ["agent"], "n_instances": 2})
    dest = runs.snapshot("run", dossier_dir=dossier, reports_dir=reports, runs_dir=tmp_path / "runs")
    assert json.loads((dest / "reports/agent.json").read_text()) == payload
    assert (dest / "reports/agent.md").read_text() == "62.5 [25, 88]"
    manifest = json.loads((dest / "manifest.json").read_text())
    assert "reports/agent.json" in manifest["artifacts"]
    assert "reports/agent.md" in manifest["artifacts"]
    assert not (dest / "reports/unrelated.tmp").exists()
