"""Comparison outcomes translate presentation without changing report data."""

from __future__ import annotations

from copy import deepcopy
import html
from html.parser import HTMLParser
import json

import pytest

from agent_trajviz.converge import app
from agent_trajviz.converge.batch import BatchResult
from agent_trajviz.converge.intervention import detect_guardrail_regressions, generate_recommendation
from agent_trajviz.converge.rendering import build_comparison_report_html
from agent_trajviz.i18n import catalog, translate, validate_catalog


class _Surface(HTMLParser):
    def __init__(self):
        super().__init__()
        self.markers = []
        self.visible = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-tv-i18n" in attrs:
            self.markers.append(attrs)

    def handle_data(self, data):
        self.visible.append(data)


def _surface(rendered):
    parsed = _Surface()
    parsed.feed(rendered)
    return parsed


def _delta(direction, before=1.0, after=1.2, *, p_value=None):
    return {
        "direction": direction, "before_mean": before, "after_mean": after, "delta": after - before,
        "significant": p_value is not None, "p_value": p_value,
    }


@pytest.mark.parametrize("deltas,guardrails", [
    ({}, []),
    ({"alignment_f1": _delta("unchanged")}, []),
    ({"alignment_f1": _delta("improved")}, []),
    ({"alignment_f1": _delta("improved"), "overhead_ratio": _delta("regressed")}, []),
    ({"alignment_f1": _delta("improved"), "overhead_ratio": _delta("regressed")},
     [{"metric": "overhead_ratio"}]),
    ({"alignment_f1": _delta("regressed")}, []),
])
def test_engine_recommendation_branches_are_localized_without_changing_values(deltas, guardrails):
    report = {
        "intervention_effect": deltas, "guardrail_warnings": guardrails,
        "recommendation": generate_recommendation(deltas, guardrails),
    }
    original = deepcopy(report)
    surface = _surface(app._recommendation_html(report))
    assert report == original
    assert "".join(surface.visible) == original["recommendation"]
    assert len(surface.markers) == 1
    marker = surface.markers[0]
    params = json.loads(marker["data-tv-i18n-args"])
    key = marker["data-tv-i18n"]
    assert key in catalog()
    assert translate(key, "zh-CN", **params) != original["recommendation"]
    for value in params.values():
        assert value in translate(key, "zh-CN", **params)


@pytest.mark.parametrize("message", [
    "Mixed results — review per-metric deltas.",
    "Intervention effects are unavailable: no successful task pairs are available.",
])
def test_other_known_recommendations_are_localized(message):
    surface = _surface(app._recommendation_html({"recommendation": message}))
    assert [marker["data-tv-i18n"] for marker in surface.markers] == [message]
    assert translate(message, "zh-CN") != message


def test_recommendation_metric_identifiers_are_escaped_parameters():
    metric = '<img src=x> "improved" & Reference'
    report = {
        "intervention_effect": {metric: _delta("improved")},
        "recommendation": f"Intervention improved {metric} without regressions.",
    }
    rendered = app._recommendation_html(report)
    marker = _surface(rendered).markers[0]
    assert json.loads(marker["data-tv-i18n-args"]) == {"improved": metric}
    assert marker["data-tv-i18n"] == "Intervention improved {improved} without regressions."
    assert "<img" not in rendered
    assert html.escape(metric) in rendered


@pytest.mark.parametrize("before,after,p_value", [
    (1.0, 1.4, None),
    (1.0, 1.12345, 0.001),
    (0.0, 0.4, None),
])
def test_generated_guardrails_translate_and_preserve_exact_numbers(before, after, p_value):
    deltas = {
        "alignment_f1": _delta("improved"),
        "overhead_ratio": _delta("regressed", before, after, p_value=p_value),
    }
    warnings = detect_guardrail_regressions(deltas)
    assert len(warnings) == 1
    original = deepcopy(warnings[0])
    rendered = app._guardrail_warning_html(warnings[0])
    surface = _surface(rendered)
    assert warnings[0] == original
    assert "".join(surface.visible) == original["warning"]
    marker = surface.markers[0]
    params = json.loads(marker["data-tv-i18n-args"])
    assert params["metric"] == "overhead_ratio"
    assert params["before"] == f"{before:.4f}"
    assert params["after"] == f"{after:.4f}"
    assert translate(marker["data-tv-i18n"], "zh-CN", **params) != original["warning"]


def test_unknown_report_prose_and_directions_remain_literal():
    value = 'Intervention improved <img src=x> /private/path with custom evidence'
    assert app._recommendation_html({"recommendation": value}) == html.escape(value)
    assert app._guardrail_warning_html({"warning": value}) == html.escape(value)
    assert app._direction_html(value) == html.escape(value)
    # Even template-shaped text must agree with the report's metric fields.
    mismatch = {"recommendation": "Intervention improved Reference without regressions.",
                "intervention_effect": {"alignment_f1": _delta("improved")}}
    assert "data-tv-i18n" not in app._recommendation_html(mismatch)


@pytest.mark.parametrize("direction", ["improved", "regressed", "unchanged"])
def test_known_directions_translate(direction):
    marker = _surface(app._direction_html(direction)).markers[0]
    assert marker["data-tv-i18n"] == direction
    assert translate(direction, "zh-CN") != direction


def test_batch_callback_translates_evidence_levels_but_preserves_pattern_ids(monkeypatch, tmp_path):
    results = [
        BatchResult(str(i), {"alignment": {}, "patterns": [{"type": "hypothesis"}] +
                             ([{"type": "supported_finding"}] if i == 0 else [])})
        for i in range(3)
    ]
    original = deepcopy(results)
    monkeypatch.setattr(app, "_run_manifest", lambda *args: results)
    rendered = app.batch_report_html("manifest.json", 50, False, data_root=tmp_path)
    assert results == original
    surface = _surface(rendered)
    levels = [marker["data-tv-i18n"] for marker in surface.markers
              if marker["data-tv-i18n"] in {"supported_finding", "hypothesis"}]
    assert sorted(levels) == ["hypothesis", "supported_finding"]
    assert "<td>hypothesis</td>" in rendered
    assert "<td>supported_finding</td>" in rendered


def test_before_after_callback_marks_own_outcomes_and_keeps_report_unchanged(monkeypatch, tmp_path):
    from agent_trajviz.converge import intervention

    deltas = {"Reference": _delta("improved"), "overhead_ratio": _delta("regressed")}
    warnings = detect_guardrail_regressions(deltas)
    report = {
        "paired_tasks": 3, "unpaired_before": 0, "unpaired_after": 0,
        "intervention_effect": deltas, "guardrail_warnings": warnings,
        "recommendation": generate_recommendation(deltas, warnings),
    }
    original = deepcopy(report)
    monkeypatch.setattr(app, "_run_manifest", lambda *args: [])
    monkeypatch.setattr(intervention, "build_intervention_report", lambda *args: report)
    rendered = app.before_after_report_html("before.json", "after.json", 50, False, data_root=tmp_path)
    keys = [marker["data-tv-i18n"] for marker in _surface(rendered).markers]
    assert report == original
    assert {"improved", "regressed"} <= set(keys)
    assert "Intervention improved {improved} but regressed {regressed}. Review tradeoffs." in keys
    assert "<td>Reference</td>" in rendered
    assert "Reference" not in keys
    assert any(key.startswith("{metric} regressed from") for key in keys)


def test_missing_milestone_is_a_localized_empty_value_and_steps_stay_numeric():
    rendered = build_comparison_report_html({"ref_milestones": {"first_edit": 2}, "cmp_milestones": {}})
    assert "N/A" in [marker["data-tv-i18n"] for marker in _surface(rendered).markers]
    assert "<td>#2</td>" in rendered
    validate_catalog()
