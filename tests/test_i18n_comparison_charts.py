"""Comparison charts translate prose without rewriting recorded identities."""

import copy

import plotly.graph_objects as go
import pytest

from agent_trajviz.converge.charts import (
    build_anchor_class_chart,
    build_divergence_waterfall_chart,
    build_milestone_timeline_chart,
)
from agent_trajviz.converge.milestones import MILESTONE_NAMES
from agent_trajviz.i18n import catalog, localize_figure, translate, validate_catalog
from agent_trajviz.insight import chart_data
from agent_trajviz.insight.charts.swimlanes import (
    build_run_group_agent_timeline,
    draw_agent_swimlane,
    draw_run_timelines,
)


def _label_keys(expression):
    if isinstance(expression, str):
        yield expression
    elif isinstance(expression, list):
        for item in expression:
            yield from _label_keys(item)
    elif "template" in expression:
        yield expression["template"]


def test_milestone_names_tick_labels_and_hover_prose_are_owned_but_values_are_not():
    reference = {name: index + 3 for index, name in enumerate(MILESTONE_NAMES)}
    compared = {name: index + 8 for index, name in enumerate(MILESTONE_NAMES)}
    compared["final_patch"] = None
    before = copy.deepcopy((reference, compared))
    figure = build_milestone_timeline_chart(reference, compared)
    metadata = figure.layout.meta["tv_i18n"]
    assert metadata["layout_labels"]["yaxis.ticktext"] == list(figure.layout.yaxis.ticktext)
    markers = [(index, trace) for index, trace in enumerate(figure.data) if trace.mode == "markers"]
    assert len(markers) == 2
    for index, trace in markers:
        labels = metadata["trace_labels"][str(index)]
        assert set(labels) == {"text", "hovertext", "hovertemplate"}
        assert labels["text"] == list(trace.text)
        assert "%{hovertext}" in labels["hovertemplate"] and "%{x}" in labels["hovertemplate"]
        assert "步骤：" in translate(labels["hovertemplate"], "zh-CN")
        assert all("x" not in fields and "y" not in fields and "customdata" not in fields
                   for fields in metadata["trace_labels"].values())
        for expression in (labels["text"], labels["hovertext"], labels["hovertemplate"]):
            assert set(_label_keys(expression)) <= catalog().keys()
    ref = next(trace for _, trace in markers if trace.name == "Reference")
    cmp = next(trace for _, trace in markers if trace.name == "Compared")
    assert "Last Recognized Edit" in ref.text
    assert "Last Recognized<br>Edit" in figure.layout.yaxis.ticktext
    assert translate("Last Recognized<br>Edit", "zh-CN") == "最后识别到的<br>编辑"
    assert list(ref.x) == list(reference.values())
    assert list(cmp.x) == [step for step in compared.values() if step is not None]
    assert "Compared − reference: +5 steps" in ref.customdata
    assert "Matching milestone not recorded in the other run" in ref.customdata
    assert (reference, compared) == before
    validate_catalog()


def test_waterfall_total_display_is_owned_while_pattern_ids_and_values_stay_literal():
    patterns = [
        {"type": "Reference", "estimated_extra_cost": {"tokens": 20}},
        {"type": "Recall (%) <raw>", "estimated_extra_cost": {"tokens": 5}},
    ]
    before = copy.deepcopy(patterns)
    figure = build_divergence_waterfall_chart(patterns)
    assert list(figure.data[0].x) == ["Reference", "Recall (%) <raw>", "Total"]
    assert list(figure.layout.xaxis.tickvals) == list(figure.data[0].x)
    assert list(figure.layout.xaxis.ticktext) == ["Reference", "Recall (%) &lt;raw&gt;", "Total"]
    assert figure.layout.meta["tv_i18n"]["layout_labels"] == {"xaxis.ticktext": [
        {"literal": "Reference"}, {"literal": "Recall (%) &lt;raw&gt;"}, "Total",
    ]}
    assert list(figure.data[0].y) == [20, 5, 25]
    assert list(figure.data[0].text) == ["20", "5", "25"]
    assert patterns == before


def test_anchor_hover_owns_recall_prefix_but_not_recorded_class_names_or_percentages():
    source = {"reference": {"write_recall_by_class": {"Reference": 0.25, "Total": 1}},
              "compared": {"write_recall_by_class": {"Reference": 0.5, "Total": 0}}}
    before = copy.deepcopy(source)
    figure = build_anchor_class_chart(source)
    for index, trace in enumerate(figure.data):
        labels = figure.layout.meta["tv_i18n"]["trace_labels"][str(index)]
        assert set(labels) == {"hovertemplate"}
        assert "召回率：%{y}%" in translate(labels["hovertemplate"], "zh-CN")
        assert "%{x}" in translate(labels["hovertemplate"], "zh-CN")
        assert list(trace.x) == ["Reference", "Total"]
    assert list(figure.data[0].y) == [25, 100]
    assert list(figure.data[1].y) == [50, 0]
    assert source == before


def test_explicit_display_metadata_is_idempotent_and_escapes_only_data_arguments():
    figure = go.Figure(go.Scatter(x=["Total"], y=[42], customdata=["Reference"], name="recorded"))
    data = copy.deepcopy(figure.to_plotly_json()["data"])
    localize_figure(figure, layout_labels={"xaxis.ticktext": [{"literal": "Total"}, "Reference"]},
                    trace_labels={0: {"hovertext": [{"template": "Step {index}",
                                                     "params": {"index": "<raw>"}}]}})
    captured = copy.deepcopy(figure.layout.meta)
    assert figure.data[0].hovertext == ("Step &lt;raw&gt;",)
    assert captured["tv_i18n"]["trace_labels"]["0"]["hovertext"][0]["params"] == {"index": "&lt;raw&gt;"}
    localize_figure(figure)
    assert figure.layout.meta == captured
    assert figure.data[0].hovertext == ("Step &lt;raw&gt;",)
    for key in ("x", "y", "customdata", "name"):
        assert figure.to_plotly_json()["data"][0][key] == data[0][key]


@pytest.mark.parametrize("path", ["xaxis.tickvals", "yaxis.range", "meta.raw", "shapes[0].x0"])
def test_layout_label_api_rejects_plotted_data_fields(path):
    figure = go.Figure(go.Scatter(x=["Total"], y=[42]))
    with pytest.raises(ValueError, match="Only chart titles"):
        localize_figure(figure, layout_labels={path: ["Reference"]})
    assert list(figure.data[0].x) == ["Total"] and list(figure.data[0].y) == [42]


@pytest.mark.parametrize("field", ["x", "y", "customdata", "ids", "marker"])
def test_trace_label_api_rejects_plotted_data_fields(field):
    figure = go.Figure(go.Scatter(x=["Total"], y=[42]))
    with pytest.raises(ValueError, match="Only chart trace names"):
        localize_figure(figure, trace_labels={0: {field: ["Reference"]}})
    assert list(figure.data[0].x) == ["Total"] and list(figure.data[0].y) == [42]


@pytest.mark.parametrize("tokens", [12345, None])
def test_run_timeline_hover_translates_only_authored_prose(tokens):
    source = {"buckets": ["main"], "runs": [{
        "run_id": "raw-id", "label": "Reference <run> (baseline)", "segments": [{
            "agent": "raw-agent", "agent_label": "Recall (%) <agent>", "bucket": "main",
            "start": 5, "end": 7, "tokens": tokens, "tool_calls": 2,
        }],
    }]}
    before = copy.deepcopy(source)
    figure = draw_run_timelines(source)
    trace = figure.data[0]
    assert list(trace.x) == [3] and list(trace.y) == ["Reference <run> (baseline)"]
    assert trace.base == 5 and trace.customdata is None
    assert trace.name == "Recall (%) <agent>" and trace.legendgroup == trace.name
    assert figure.layout.yaxis.categoryarray == ("Reference <run> (baseline)",)
    assert figure.layout.yaxis.tickvals == ("Reference <run> (baseline)",)
    metadata = figure.layout.meta["tv_i18n"]
    assert "traces" not in metadata
    assert metadata["layout_labels"]["yaxis.ticktext"] == [{"literal": "Reference &lt;run&gt; (baseline)"}]
    expression = metadata["trace_labels"]["0"]["hovertext"]
    translated = translate(expression["template"], "zh-CN", **expression["params"])
    assert translated.startswith("<b>Reference &lt;run&gt; (baseline)</b><br>Recall (%) &lt;agent&gt;")
    assert "步骤 5–7（3 步）" in translated and "2 次工具调用" in translated
    assert ("12,345 Token" if tokens is not None else "无 Token 数据") in translated
    assert source == before


@pytest.mark.parametrize("label,spec,translated", [
    ("Reference (baseline)", {"template": "{run} (baseline)", "params": {"run": "Reference"}},
     "Reference （基线）"),
    ("Overview (baseline)", {"template": "Overview (baseline)", "params": {}}, "概览 （基线）"),
])
def test_generated_baseline_display_metadata_reaches_ticks_and_hover_without_changing_run_ids(label, spec, translated):
    runs = [{"run_id": "Reference", "label": label, "label_i18n": spec, "steps": [
        {"index": 3, "role": "assistant", "agent": "main", "tokens": {"total": 100}, "tool_call_count": 2},
    ]}]
    before = copy.deepcopy(runs)
    assert "label_i18n" not in chart_data.run_timelines(runs)["runs"][0]
    figure = build_run_group_agent_timeline(runs)
    assert list(figure.data[0].y) == [label] and figure.data[0].base == 3
    assert list(figure.data[0].x) == [1]
    assert list(figure.layout.yaxis.categoryarray) == [label]
    metadata = figure.layout.meta["tv_i18n"]
    tick = metadata["layout_labels"]["yaxis.ticktext"][0]
    assert translate(tick["template"], "zh-CN", **tick["params"]) == translated
    hover = metadata["trace_labels"]["0"]["hovertext"]
    assert translate(hover["template"], "zh-CN", **hover["params"]).startswith(f"<b>{translated}</b>")
    assert runs == before


@pytest.mark.parametrize("spec", [
    {"template": "Reference", "params": {}},
    {"template": "{run} (baseline)", "params": {"run": "Different"}},
    {"template": "Overview (baseline)", "params": {}},
])
def test_run_display_metadata_cannot_claim_arbitrary_or_mismatched_recorded_names(spec):
    runs = [{"run_id": "raw-id", "label": "Reference (baseline)", "label_i18n": spec, "steps": [
        {"index": 0, "role": "assistant", "agent": "main", "tokens": {"total": 100}},
    ]}]
    figure = build_run_group_agent_timeline(runs)
    assert figure.layout.meta["tv_i18n"]["layout_labels"]["yaxis.ticktext"] == [
        {"literal": "Reference (baseline)"},
    ]
    assert list(figure.data[0].y) == ["Reference (baseline)"]


@pytest.mark.parametrize("tokens", [12345, None])
def test_agent_swimlane_owns_segment_prose_but_preserves_agent_and_workflow_jump(tokens):
    source = {"lanes": [{"label": "Reference <agent>", "color_index": 0, "segments": [
        {"start": 8, "end": 9, "tokens": tokens, "tool_calls": 3},
    ]}]}
    before = copy.deepcopy(source)
    figure = draw_agent_swimlane(source)
    trace = figure.data[0]
    assert list(trace.x) == [2] and list(trace.y) == ["Reference <agent>"]
    assert trace.base == 8 and list(trace.customdata) == [8]
    assert trace.name == "Reference <agent>"
    metadata = figure.layout.meta["tv_i18n"]
    assert "traces" not in metadata
    hover = metadata["trace_labels"]["0"]["hovertext"]
    translated = translate(hover["template"], "zh-CN", **hover["params"])
    assert translated.startswith("Reference &lt;agent&gt;：步骤 8–9")
    assert "3 次工具调用" in translated
    assert ("12,345 Token" if tokens is not None else "无 Token 数据") in translated
    assert source == before
