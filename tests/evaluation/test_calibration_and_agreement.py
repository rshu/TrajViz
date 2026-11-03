"""Independent-rater reliability and calibration controls on synthetic cohorts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_trajviz.awe._engine import agreement, calibrate, sampling


def test_cohen_kappa_missing_labels_and_chance_adjustment():
    assert agreement.cohen_kappa([], []) is None
    assert agreement.cohen_kappa([None], ["editing"]) is None
    assert agreement.cohen_kappa(["editing"], ["editing"]) == 1
    assert agreement.cohen_kappa(["editing", "verification"], ["verification", "editing"]) == -1
    # Three of four agree, while chance agreement from the marginals is 1/2.
    assert agreement.cohen_kappa(
        ["edit", "edit", "verify", "verify"], ["edit", "verify", "verify", "verify"]
    ) == pytest.approx(0.5)
    assert agreement.cohen_kappa([None, "edit", "verify"], ["edit", None, "verify"]) == 1


def test_fleiss_kappa_checks_equal_rater_counts_and_disagreement():
    assert agreement.fleiss_kappa([]) is None
    assert agreement.fleiss_kappa([{"edit": 1}]) is None
    assert agreement.fleiss_kappa([{"edit": 2}, {"verify": 3}]) is None
    assert agreement.fleiss_kappa([{"edit": 3}]) == 1
    assert agreement.fleiss_kappa([{"edit": 3}, {"verify": 3}]) == 1
    # Each item has a 2:1 split; expected agreement is 1/2, observed is 1/3.
    assert agreement.fleiss_kappa([{"edit": 2, "verify": 1}, {"edit": 1, "verify": 2}]) == pytest.approx(-1 / 3)


def test_independent_human_consensus_and_detector_abstentions():
    ids = ["one", "two", "three", "four", "unannotated"]
    human = {
        "alice": {cid: {"cause": label} for cid, label in zip(ids, ["edit", "edit", "verify", "verify"], strict=False)},
        "bob": {cid: {"cause": label} for cid, label in zip(ids, ["edit", "verify", "verify", "verify"], strict=False)},
    }
    ceiling = agreement.human_human(human, ids, "cause")
    assert ceiling == {"pairwise_mean_kappa": 0.5, "fleiss_kappa": pytest.approx(7 / 15), "n_pairs": 1}
    assert agreement.consensus(human, ids, "cause") == {
        "one": "edit",
        "two": "edit",
        "three": "verify",
        "four": "verify",
        "unannotated": None,
    }
    detector = {"one": {"cause": "edit"}, "two": {"cause": "verify"}, "four": {"cause": "verify"}}
    assert agreement.detector_vs_consensus(detector, human, ids, "cause") == {
        "n": 3,
        "accuracy": pytest.approx(2 / 3),
        "kappa": pytest.approx(0.4),
    }
    assert agreement.detector_vs_consensus({}, {}, ids, "cause") == {"n": 0, "accuracy": None, "kappa": None}
    assert agreement.human_human({}, ids, "cause") == {
        "pairwise_mean_kappa": None,
        "fleiss_kappa": None,
        "n_pairs": 0,
    }


def test_onset_alignment_uses_paired_evidence_only():
    assert agreement.onset_metrics([1, None, 5, 9], [2, 4, 5, None]) == {
        "n": 2,
        "exact_match": 0.5,
        "mean_abs_dist": 0.5,
    }
    assert agreement.onset_metrics([None], [2]) == {"n": 0, "exact_match": None, "mean_abs_dist": None}


@pytest.mark.parametrize("solved,failed", [([], []), ([1.0, 2.0], []), ([], [4.0, 8.0])])
def test_calibration_preserves_missing_cohort_measurements(monkeypatch, solved, failed):
    monkeypatch.setattr(calibrate, "edit_ratio_distribution", lambda: (solved, failed))
    result = calibrate.suggest_over_edit_threshold()
    assert result["n_solved"] == len(solved) and result["n_failed"] == len(failed)
    assert set(result["fire_rate_solved_failed"]) >= {3.0, 5.0, 8.0}
    assert 0 not in result["fire_rate_solved_failed"]
    for cohort, data, index in [("solved", solved, 0), ("failed", failed, 1)]:
        if not data:
            assert set(result[f"{cohort}_pctiles"].values()) == {None}
            assert all(rates[index] is None for rates in result["fire_rate_solved_failed"].values())


def test_calibration_quantiles_and_threshold_fire_rates(monkeypatch):
    monkeypatch.setattr(calibrate, "edit_ratio_distribution", lambda: ([1.0, 2.0, 3.0, 4.0], [2.0, 6.0, 10.0, 14.0]))
    result = calibrate.suggest_over_edit_threshold()
    assert result["solved_pctiles"] == {50: 3.0, 75: 4.0, 90: 4.0, 95: 4.0, 99: 4.0}
    assert result["failed_pctiles"] == {50: 10.0, 75: 14.0, 90: 14.0, 95: 14.0, 99: 14.0}
    assert result["fire_rate_solved_failed"][3.0] == (0.25, 0.75)
    assert result["fire_rate_solved_failed"][5.0] == (0.0, 0.75)
    assert result["fire_rate_solved_failed"][14.0] == (0.0, 0.0)


def test_edit_distribution_excludes_unknown_outcomes_missing_reference_and_missing_ratio(monkeypatch):
    monkeypatch.setattr(calibrate.config, "AGENTS", ["agent"])
    monkeypatch.setattr(
        calibrate.config, "list_instances", lambda: ["solved", "failed", "unknown", "broken", "no_ratio"]
    )
    outcomes = {"solved": True, "failed": False, "unknown": None, "broken": False, "no_ratio": True}
    monkeypatch.setattr(calibrate, "resolved", lambda agent, iid: outcomes[iid])

    def gold(iid):
        if iid == "broken":
            raise ValueError("reference unavailable")
        return SimpleNamespace(instance_id=iid)

    monkeypatch.setattr(calibrate, "build_gold_reference", gold)
    monkeypatch.setattr(calibrate, "load_agent_patch", lambda agent, iid: None)
    monkeypatch.setattr(
        calibrate,
        "localization_signals",
        lambda gold, patch, resolved: {
            "edit_size_ratio": {"solved": 1.5, "failed": 9.0, "no_ratio": None}[gold.instance_id],
        },
    )
    assert calibrate.edit_ratio_distribution() == ([1.5], [9.0])


def test_stratified_sample_balances_agents_and_causes_reproducibly(monkeypatch):
    agents = ["agent_a", "agent_b"]
    instances = [
        *(f"edit-{i}" for i in range(6)),
        *(f"verify-{i}" for i in range(6)),
        "solved-1",
        "solved-2",
        "unknown",
    ]
    monkeypatch.setattr(sampling.config, "AGENTS", agents)
    monkeypatch.setattr(sampling.config, "list_instances", lambda: instances)
    monkeypatch.setattr(sampling, "resolved", lambda agent, iid: None if iid == "unknown" else iid.startswith("solved"))
    monkeypatch.setattr(sampling, "_primary_capability", lambda agent, iid: iid.split("-")[0])
    failed, solved = sampling.stratified_sample(n=10, seed=42, solved_fraction=0.2)
    assert (failed, solved) == sampling.stratified_sample(n=10, seed=42, solved_fraction=0.2)
    assert len(failed) == 8 and len(set(failed)) == 8
    assert len(solved) == 2 and all(iid.startswith("solved") for _, iid in solved)
    for agent in agents:
        for cause in ["edit", "verify"]:
            assert sum(a == agent and iid.startswith(cause) for a, iid in failed) == 2
    assert all(iid != "unknown" for _, iid in failed + solved)


def test_calibration_cli_reports_absent_cohort_as_unknown(monkeypatch, capsys):
    source = Path(calibrate.__file__).parent / "scripts" / "calibrate.py"
    spec = importlib.util.spec_from_file_location("calibration_cli_control", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(calibrate, "edit_ratio_distribution", lambda: ([], []))
    monkeypatch.setattr(module, "blame_sensitivity", lambda: {})
    monkeypatch.setattr(module, "attribution_sensitivity", lambda: {})
    module.main()
    output = capsys.readouterr().out
    assert "n_solved=0  n_failed=0" in output
    assert "None" in output and "NA" in output
    assert "0%" not in output
