"""Saved telemetry figures preserve unavailable evidence and measured gaps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.figures import FIGURE_VIEWS, figure
from agent_trajviz.insight.charts import swimlanes, usage

FIXTURE = Path(__file__).parent / "fixtures" / "otlp"
RUN_CHARTS = sorted(vid for vid, spec in tv.VIEW_CATALOG.items() if spec["kind"] == "chart")


def _minimal_file(tmp_path):
    requests = [json.loads(line) for line in (FIXTURE / "claude_code_traces.jsonl").read_text().splitlines()]
    for request in requests:
        for resource in request["resourceSpans"]:
            for scope in resource["scopeSpans"]:
                for span in scope["spans"]:
                    span["attributes"] = [a for a in span["attributes"]
                                          if a["key"] in {"session.id", "tool_name", "tool_use_id", "agent_id"}]
                    span["startTimeUnixNano"] = span["endTimeUnixNano"] = "0"
                    span.pop("events", None)
    path = tmp_path / "minimal.jsonl"
    path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    return path


@pytest.fixture(params=["full", "minimal"])
def run(request, tmp_path):
    return tv.load(FIXTURE if request.param == "full" else _minimal_file(tmp_path))


@pytest.mark.parametrize("view_id", RUN_CHARTS)
@pytest.mark.parametrize("dark", [False, True])
def test_every_trajectory_chart_draws_saved_telemetry(run, view_id, dark):
    assert view_id in FIGURE_VIEWS
    view = run.view(view_id)
    chart = figure(view, dark=dark)
    # Serialization exercises Plotly validation as well as the drawer; unknown
    # observations stay null, never NaN or an invented measured zero.
    json.loads(chart.to_json())
    if view.availability["status"] == "unavailable":
        assert not chart.data
        assert any("Unavailable" in annotation.text for annotation in chart.layout.annotations)
    if view.availability["status"] == "partial":
        assert any("Partial data" in annotation.text for annotation in chart.layout.annotations)


def test_unknown_token_fields_are_gaps_and_absent_categories():
    run = tv.load(FIXTURE)
    view = run.view("token_usage")
    chart = figure(view)
    assert "Reasoning" not in {trace.name for trace in chart.data}
    # Chart data keeps the full step series; multi-agent drawers may split it
    # across panels, so assert gaps on the view payload rather than one trace.
    assert [row["fresh_input"] for row in view.data["steps"]] == [None, 10, 2, 1, None, 2]
    assert sum(value for trace in chart.data for value in trace.y if value is not None) == 57


def test_missing_durations_draw_gaps_without_average_zero():
    data = {"average_s": None, "steps": [
        {"step": 1, "duration_s": None, "spawn_adjusted": False, "error_kind": None, "outlier": False},
    ]}
    chart = usage.draw_step_durations(data)
    assert list(chart.data[0].y) == [None]
    assert not any("Avg:" in annotation.text for annotation in chart.layout.annotations)


def test_unknown_outcomes_are_not_drawn_as_failure_or_success():
    data = {"agents": [{"id": "", "label": "main"}], "calls": [
        {"step": 1, "tool": "Read", "ok": None, "hint": "", "agent": ""},
    ]}
    chart = swimlanes.draw_tool_outcomes(data)
    assert len(chart.data) == 1
    assert chart.data[0].name == "Outcome unavailable"
    assert chart.data[0].marker.symbol == "diamond-open"
    assert list(chart.data[0].hovertext) == ["Outcome unavailable"]


def test_mixed_multi_agent_outcomes_have_explicit_unknown_shape():
    data = {"agents": [{"id": "", "label": "main"}, {"id": "child", "label": "child"}], "calls": [
        {"step": 1, "tool": "Read", "ok": True, "hint": "", "agent": ""},
        {"step": 2, "tool": "Read", "ok": None, "hint": "", "agent": "child"},
    ]}
    chart = swimlanes.draw_tool_outcomes(data)
    child = next(trace for trace in chart.data if trace.name == "child")
    assert list(child.marker.symbol) == ["diamond-open"]
    assert list(child.marker.line.width) == [0]
    assert list(child.hovertext) == ["Outcome unavailable"]
    assert not any(trace.name == "Failure (x)" for trace in chart.data)


def test_swimlane_unknown_tokens_are_labeled_unavailable():
    chart = swimlanes.draw_agent_swimlane({"lanes": [{
        "id": "", "label": "main", "color_index": 0,
        "segments": [{"start": 0, "end": 2, "tokens": None, "tool_calls": 1}],
    }]})
    assert chart.data[0].text == "3 steps"
    assert "Tokens unavailable" in chart.data[0].hovertext
    assert "0 tokens" not in chart.data[0].hovertext


def test_run_timeline_unknown_tokens_are_labeled_unavailable():
    chart = swimlanes.draw_run_timelines({"buckets": ["main"], "runs": [{
        "label": "run", "segments": [{"agent_label": "main", "start": 0, "end": 1,
                                       "bucket": "main", "tokens": None, "tool_calls": 1}],
    }]})
    assert "Tokens unavailable" in chart.data[0].hovertext


def test_native_figures_keep_existing_values_without_telemetry_notice():
    run = tv.load(FIXTURE / "native" / "demo-session.jsonl")
    for view_id in RUN_CHARTS:
        chart = figure(run.view(view_id))
        assert not any("required evidence" in annotation.text or "Partial data" in annotation.text
                       for annotation in chart.layout.annotations)
