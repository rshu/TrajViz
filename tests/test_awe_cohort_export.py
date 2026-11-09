"""Strict-JSON exports must preserve genuine unbounded cohort effects."""
import json
import math

import pytest

from agent_trajviz.awe import _worker
from agent_trajviz.awe._engine import cohort, config


def test_export_preserves_internal_infinity_and_distinguishes_zero_and_unknown():
    source = {
        "agent": "recorded-agent", "n_failed": 2, "n_solved": 2,
        "associations": [
            {"feature": "perfect_separation", "binary": True, **cohort.binary_association([1, 1], [0, 0])},
            {"feature": "protective", "binary": True, **cohort.binary_association([0, 0], [1, 1])},
            {"feature": "unobserved_stratum", "binary": True, **cohort.binary_association([1, 1], [1, 1])},
            {"feature": "finite", "binary": True, **cohort.binary_association([1, 1, 1, 0], [1, 0, 0, 0])},
        ],
    }
    exported = cohort.export_discovery(source)
    restored = json.loads(json.dumps(exported, allow_nan=False))
    rows = restored["associations"]
    assert rows[0]["risk_ratio"] == "Infinity"
    assert rows[0]["risk_ratio_status"] == "positive_infinity"
    assert "recorded sample" in rows[0]["risk_ratio_interpretation"]
    assert rows[1]["risk_ratio"] == 0.0
    assert rows[1]["risk_ratio_status"] == "finite"
    assert rows[2]["risk_ratio"] is None
    assert rows[2]["risk_ratio_status"] == "undefined"
    assert rows[3]["risk_ratio"] == 3.0
    assert math.isinf(source["associations"][0]["risk_ratio"])
    assert "risk_ratio_status" not in source["associations"][0]


@pytest.mark.parametrize("field,value", [("diff", float("nan")), ("lo", float("-inf")), ("risk_ratio", float("nan"))])
def test_invalid_nonfinite_measurements_are_rejected(field, value):
    with pytest.raises(ValueError):
        cohort.export_discovery({"associations": [{"binary": True, "risk_ratio": 1.0, field: value}]})


def test_discovery_worker_exports_unbounded_effect_and_readable_report(tmp_path, monkeypatch):
    rows = [(False, {"hallucinated_path": 1.0})] * 6 + [(True, {"hallucinated_path": 0.0})] * 6
    discovery = cohort.discover(rows, B=100)
    discovery["agent"] = "recorded-agent"
    monkeypatch.setattr(config, "AGENTS", ["recorded-agent"])
    monkeypatch.setattr(config, "list_instances", lambda: ["task"])
    monkeypatch.setattr(config, "require_instances", lambda ids: ids)
    monkeypatch.setattr(cohort, "discover_agent", lambda agent, ids: discovery)
    result = _worker._discovery({"operation": "cohort", "agent": None, "limit": None, "output": str(tmp_path)})
    exported = json.loads((tmp_path / "discovery/cohort_contrast.json").read_text())
    assert exported[0]["associations"][0]["risk_ratio"] == "Infinity"
    assert exported[0]["associations"][0]["risk_ratio_status"] == "positive_infinity"
    markdown = (tmp_path / "discovery/cohort_contrast.md").read_text()
    assert "RR=∞" in markdown
    assert "Associational, not causal" in markdown
    assert result["report"] == "cohort_contrast"
