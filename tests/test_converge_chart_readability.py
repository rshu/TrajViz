"""Milestone collisions and waterfall clipping must not hide recorded values."""

from __future__ import annotations

import copy

import pytest

from agent_trajviz.converge.charts import (
    build_anchor_class_chart,
    build_divergence_waterfall_chart,
    build_milestone_timeline_chart,
    build_segment_cost_chart,
)
from agent_trajviz.converge.milestones import MILESTONE_LABELS, MILESTONE_NAMES


def test_coincident_and_nearby_milestones_have_named_separate_lanes():
    recorded = {
        "first_relevant_file": 3,
        "first_edit": 10,
        "first_surviving_edit": 10,
        "final_patch": 66,
        "first_passing_validation": 68,
    }
    reference = {name: recorded[name] for name in MILESTONE_NAMES}
    compared = {name: step + 2 for name, step in reference.items()}
    original = copy.deepcopy((reference, compared))
    figure = build_milestone_timeline_chart(reference, compared)
    ref = next(trace for trace in figure.data if trace.name == "Reference")
    cmp = next(trace for trace in figure.data if trace.name == "Compared")
    assert list(ref.x) == list(reference.values())
    assert list(cmp.x) == list(compared.values())
    assert len(set(ref.y)) == len(reference) and len(set(cmp.y)) == len(compared)
    assert ref.mode == "markers" and cmp.mode == "markers"
    assert {text.replace("<br>", " ") for text in figure.layout.yaxis.ticktext} == set(MILESTONE_LABELS.values())
    assert figure.layout.xaxis.title.text == "Step Index"
    assert figure.layout.yaxis.visible and figure.layout.legend.orientation == "h"
    assert len({annotation.y for annotation in figure.layout.annotations}) == len(reference)
    assert all(annotation.text == "+2" for annotation in figure.layout.annotations)
    assert (reference, compared) == original


def test_all_milestones_can_share_one_step_without_losing_names_or_delta_hover():
    milestones = dict.fromkeys(MILESTONE_NAMES, 10)
    figure = build_milestone_timeline_chart(milestones, milestones)
    tracks = [trace for trace in figure.data if trace.mode == "markers"]
    assert {trace.name for trace in tracks} == {"Reference", "Compared"}
    assert all(list(trace.x) == [10] * len(MILESTONE_NAMES) for trace in tracks)
    assert set(tracks[0].y).isdisjoint(tracks[1].y)
    assert all(set(trace.text) == set(MILESTONE_LABELS.values()) for trace in tracks)
    assert all("+0 steps" in value for trace in tracks for value in trace.customdata)
    assert all(
        figure.layout.yaxis.range[0] < annotation.y < figure.layout.yaxis.range[1]
        for annotation in figure.layout.annotations
    )


def test_narrow_milestone_chart_wraps_axis_labels_without_truncating_hover():
    milestones = dict.fromkeys(MILESTONE_NAMES, 10)
    figure = build_milestone_timeline_chart(milestones, milestones)
    figure.update_layout(width=330)
    assert figure.layout.width - figure.layout.margin.l - figure.layout.margin.r >= 180
    assert all(len(line) <= 18 for label in figure.layout.yaxis.ticktext for line in label.split("<br>"))
    assert {label.replace("<br>", " ") for label in figure.layout.yaxis.ticktext} == set(MILESTONE_LABELS.values())
    markers = [trace for trace in figure.data if trace.mode == "markers"]
    assert all(set(trace.text) == set(MILESTONE_LABELS.values()) for trace in markers)
    assert all("%{text}" in trace.hovertemplate and "Step: %{x}" in trace.hovertemplate for trace in markers)
    assert len({annotation.y for annotation in figure.layout.annotations}) == len(MILESTONE_NAMES)


def test_milestone_delta_text_keeps_large_integer_values_exact():
    figure = build_milestone_timeline_chart({"first_edit": 1}, {"first_edit": 1234568})
    assert figure.layout.annotations[0].text == "+1234567"
    markers = [trace for trace in figure.data if trace.mode == "markers"]
    assert all("+1234567 steps" in trace.customdata[0] for trace in markers)


@pytest.mark.parametrize("costs", [[16840, 9970], [10, -20], [1]])
def test_waterfall_preserves_values_and_leaves_room_for_outside_labels(costs):
    patterns = [
        {"type": f"pattern-{index}", "estimated_extra_cost": {"tokens": cost}} for index, cost in enumerate(costs)
    ]
    original = copy.deepcopy(patterns)
    figure = build_divergence_waterfall_chart(patterns)
    trace = figure.data[0]
    ordered = sorted(costs, reverse=True)
    assert list(trace.y) == ordered + [sum(costs)]
    assert list(trace.text) == [f"{value:,}" for value in ordered + [sum(costs)]]
    assert trace.cliponaxis is False and trace.textposition == "outside"
    levels = [0]
    for cost in ordered:
        levels.append(levels[-1] + cost)
    assert figure.layout.yaxis.range[1] > max(levels)
    if min(levels) < 0:
        assert figure.layout.yaxis.range[0] < min(levels)
    assert patterns == original


def test_narrow_converge_titles_and_legends_have_separate_bands():
    milestones = dict.fromkeys(MILESTONE_NAMES, 10)
    figure = build_milestone_timeline_chart(milestones, milestones)
    figure.update_layout(width=330)
    assert figure.layout.title.yref == "container" and figure.layout.title.yanchor == "top"
    assert figure.layout.title.y == 0.94
    assert figure.layout.title.pad.t >= 2
    assert figure.layout.margin.t >= 110
    assert figure.layout.legend.orientation == "h" and figure.layout.legend.y <= 1.03
    waterfall = build_divergence_waterfall_chart([{"type": "recorded", "estimated_extra_cost": {"tokens": 9970}}])
    assert "<br>" in waterfall.layout.title.text
    assert all(len(line) <= 26 for line in waterfall.layout.title.text.split("<br>"))
    assert waterfall.layout.title.text.replace("<br>", " ") == "Divergence Cost Breakdown by Pattern"


def test_narrow_anchor_percentages_do_not_shrink_into_unreadable_text():
    source = {
        "reference": {"write_recall_by_class": {"source": 1, "test": 0}},
        "compared": {"write_recall_by_class": {"source": 0.5, "test": 0}},
        "file_classes": {"source": [], "test": []},
    }
    figure = build_anchor_class_chart(source)
    figure.update_layout(width=330)
    assert list(figure.data[0].y) == [100, 0]
    assert list(figure.data[0].text) == ["100%", ""]
    assert list(figure.data[1].y) == [50, 0]
    assert list(figure.data[1].text) == ["50%", ""]
    assert all(
        trace.textfont.size >= 11 and trace.constraintext == "none" and trace.cliponaxis is False
        for trace in figure.data
    )
    assert figure.layout.uniformtext.minsize >= 11 and figure.layout.uniformtext.mode == "show"
    assert figure.layout.yaxis.range[1] > 100


def test_segment_ticks_keep_full_names_without_colliding_with_an_axis_title():
    label = "first_surviving_edit → first_passing_validation"
    figure = build_segment_cost_chart(
        {
            "reference_segments": [{"segment": label, "tokens": 1234}],
            "compared_segments": [{"segment": label, "tokens": 5678}],
        },
        False,
    )
    assert all(list(trace.x) == [label] for trace in figure.data)
    assert figure.layout.xaxis.title.text is None
    assert figure.layout.xaxis.automargin and figure.layout.xaxis.tickangle == -45
    assert figure.layout.margin.b >= 180 and figure.layout.height >= 500
    assert list(figure.data[0].y) == [1234] and list(figure.data[1].y) == [5678]


def test_anchor_zero_labels_are_omitted_without_removing_values_or_hover():
    source = {
        "reference": {"write_recall_by_class": {"a": 1, "b": 0}},
        "compared": {"write_recall_by_class": {"a": 1, "b": 0}},
        "file_classes": {"a": [], "b": []},
    }
    figure = build_anchor_class_chart(source)
    for trace in figure.data:
        assert list(trace.y) == [100, 0] and list(trace.text) == ["100%", ""]
        assert list(trace.textposition) == ["inside", "outside"]
        assert trace.textangle == -90 and "%{y}%" in trace.hovertemplate
