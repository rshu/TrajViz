"""Builtin taxonomy names are presentation labels, never rewritten data IDs."""

import copy

import plotly.graph_objects as go
import pytest

from agent_trajviz.i18n import catalog, localize_figure, validate_catalog
from agent_trajviz.insight.charts.label_charts import (
    build_label_action_count_chart,
    build_label_action_duration_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_label_timeline_chart,
    build_phase_count_comparison_chart,
    build_phase_duration_comparison_chart,
)


@pytest.mark.parametrize("builder,values,expected", [
    (build_label_phase_count_chart, {"understand": 4, "implement": 2, "Reference": 3}, [4, 2, 3]),
    (build_label_phase_duration_chart, {"understand": 4.23, "implement": 2.47, "Reference": 3.61}, [4.2, 2.5, 3.6]),
])
def test_phase_bars_own_builtin_tick_and_hover_labels_without_rewriting_categories_or_values(builder, values, expected):
    before = copy.deepcopy(values)
    figure = builder(values)
    trace = figure.data[0]
    assert list(trace.x) == ["understand", "implement", "Reference"]
    assert list(trace.y) == expected
    assert list(figure.layout.xaxis.tickvals) == list(trace.x)
    assert list(figure.layout.xaxis.ticktext) == list(trace.x)
    labels = ["understand", "implement", {"literal": "Reference"}]
    metadata = figure.layout.meta["tv_i18n"]
    assert metadata["layout_labels"]["xaxis.ticktext"] == labels
    assert metadata["trace_labels"]["0"]["hovertext"] == labels
    assert "%{hovertext}" in trace.hovertemplate
    assert values == before


@pytest.mark.parametrize("builder,values,expected", [
    (build_label_action_count_chart, {"file_discovery": 4, "understand": 3, "Reference": 2}, [4, 3, 2]),
    (build_label_action_duration_chart, {"file_discovery": 4.23, "understand": 3.47, "Reference": 2.61}, [4.2, 3.5, 2.6]),
])
def test_action_bars_preserve_phase_customdata_and_limit_translation_to_the_correct_taxonomy(builder, values, expected):
    mapping = {"file_discovery": "understand", "understand": "file_discovery", "Reference": "Reference"}
    before = copy.deepcopy((values, mapping))
    figure = builder(values, mapping)
    trace = figure.data[0]
    assert list(trace.y) == ["file_discovery", "understand", "Reference"]
    assert list(trace.x) == expected
    assert list(trace.customdata) == ["understand", "file_discovery", "Reference"]
    metadata = figure.layout.meta["tv_i18n"]
    assert metadata["layout_labels"]["yaxis.ticktext"] == [
        "file_discovery", {"literal": "understand"}, {"literal": "Reference"},
    ]
    hover = metadata["trace_labels"]["0"]["hovertext"]
    assert hover[0]["labels"] == {"action": "file_discovery", "phase": "understand"}
    assert hover[1]["labels"] == {"action": {"literal": "understand"}, "phase": {"literal": "file_discovery"}}
    assert hover[2]["labels"] == {"action": {"literal": "Reference"}, "phase": {"literal": "Reference"}}
    assert metadata["trace_labels"]["1"]["name"] == "understand"
    assert (values, mapping) == before


@pytest.mark.parametrize("builder,reference,compared,expected", [
    (build_phase_count_comparison_chart, {"plan": 3}, {"plan": 5, "Reference": 2}, [[3, 0], [5, 2]]),
    (build_phase_duration_comparison_chart, {"plan": 3.25}, {"plan": 5.75, "Reference": 2.5}, [[3.25, 0], [5.75, 2.5]]),
])
def test_phase_comparisons_translate_phase_display_but_preserve_run_names_and_paired_data(builder, reference,
                                                                                       compared, expected):
    before = copy.deepcopy((reference, compared))
    figure = builder(reference, compared, ref_label="understand", cmp_label="Reference")
    metadata = figure.layout.meta["tv_i18n"]
    assert [trace.name for trace in figure.data] == ["understand", "Reference"]
    assert "traces" not in metadata
    assert all("name" not in fields for fields in metadata["trace_labels"].values())
    assert all(list(trace.x) == ["plan", "Reference"] for trace in figure.data)
    assert [list(trace.y) for trace in figure.data] == expected
    assert metadata["layout_labels"]["xaxis.ticktext"] == ["plan", {"literal": "Reference"}]
    assert all(fields["hovertext"] == ["plan", {"literal": "Reference"}]
               for fields in metadata["trace_labels"].values())
    assert (reference, compared) == before


def test_step_timeline_owns_action_phase_prose_and_user_annotation_but_retains_indices_and_prompt_content():
    steps = [
        {"index": "understand", "role": "user", "phase": "user", "action": "user_prompt", "duration_s": 0,
         "tokens_total": 0, "text_preview": "file_discovery understand <img src=x>"},
        {"index": 7, "role": "assistant", "phase": "understand", "action": "file_discovery", "duration_s": 2.8,
         "tokens_total": 12345},
        {"index": 99, "role": "assistant", "phase": "file_discovery", "action": "Reference", "duration_s": 3.4,
         "tokens_total": 50},
    ]
    before = copy.deepcopy(steps)
    figure = build_label_timeline_chart(steps)
    metadata = figure.layout.meta["tv_i18n"]
    assert list(figure.data[0].x) == [0, 2.8, 3.4]
    assert list(figure.data[0].y) == [0, 1, 2]
    assert list(figure.layout.yaxis.ticktext) == ["understand", "7", "99"]
    assert list(figure.layout.yaxis.range) == [2.5, -0.5]
    assert list(figure.data[0].text) == ["", "file_discovery  (2.8s)", "Reference  (3.4s)"]
    assistant = metadata["trace_labels"]["0"]
    assert assistant["text"][1]["labels"]["action"] == "file_discovery"
    assert assistant["text"][2]["labels"]["action"] == {"literal": "Reference"}
    assert assistant["hovertext"][1]["labels"] == {"phase": "understand", "action": "file_discovery"}
    assert assistant["hovertext"][2]["labels"] == {
        "phase": {"literal": "file_discovery"}, "action": {"literal": "Reference"},
    }
    user_hover = metadata["trace_labels"]["1"]["hovertext"][0]
    assert user_hover["params"] == {"index": "understand", "preview": "file_discovery understand &lt;img src=x&gt;"}
    assert figure.data[1].hovertext[0].endswith("file_discovery understand &lt;img src=x&gt;")
    assert metadata["layout_labels"]["annotations[0].text"] == "<i>user prompt</i>"
    assert steps == before
    original_data = copy.deepcopy(figure.to_plotly_json()["data"])
    original_metadata = copy.deepcopy(metadata)
    localize_figure(figure)
    assert figure.to_plotly_json()["data"] == original_data
    assert figure.layout.meta["tv_i18n"] == original_metadata
    validate_catalog()


def test_nested_owned_label_parameters_escape_literal_arguments_once_and_keep_sources_idempotent():
    figure = go.Figure(go.Bar(x=["file_discovery"], y=[12]))
    localize_figure(figure, trace_labels={0: {"text": {
        "template": "{action}: {description}", "params": {"description": "<raw> understand"},
        "labels": {"action": "file_discovery"},
    }}})
    assert figure.data[0].text == "file_discovery: &lt;raw&gt; understand"
    expression = figure.layout.meta["tv_i18n"]["trace_labels"]["0"]["text"]
    assert expression["params"] == {"description": "&lt;raw&gt; understand"}
    assert expression["labels"] == {"action": "file_discovery"}
    localize_figure(figure)
    assert figure.data[0].text == "file_discovery: &lt;raw&gt; understand"
    assert list(figure.data[0].x) == ["file_discovery"] and list(figure.data[0].y) == [12]
    assert "file_discovery" in catalog()


@pytest.mark.parametrize("labels,params", [
    ({"missing": "understand"}, {}),
    ({"action": "file_discovery"}, {"action": "recorded"}),
    ({"action.name": "file_discovery"}, {}),
    ({"action": {"x": "understand"}}, {}),
])
def test_nested_labels_reject_ambiguous_fields_and_invalid_expressions(labels, params):
    figure = go.Figure(go.Bar(x=["file_discovery"], y=[12]))
    with pytest.raises(ValueError):
        localize_figure(figure, trace_labels={0: {"text": {
            "template": "{action}", "params": params, "labels": labels,
        }}})
    assert list(figure.data[0].x) == ["file_discovery"] and list(figure.data[0].y) == [12]
