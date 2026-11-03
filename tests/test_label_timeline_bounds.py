"""Label timelines occupy their ordinal rows without session-sized edge gaps."""

from __future__ import annotations

from copy import deepcopy

import pytest

import agent_trajviz as tv
from agent_trajviz.figures import figure
from agent_trajviz.insight.charts.label_charts import build_label_timeline_chart


def _step(index, *, role="assistant", phase="understand", action="code_reading", duration=2.0):
    return {"index": index, "role": role, "phase": phase, "action": action,
            "duration_s": duration, "tokens_total": 100, "text_preview": "prompt" if role == "user" else ""}


@pytest.mark.parametrize("size", [1, 2, 104, 203])
@pytest.mark.parametrize("dark", [False, True])
def test_first_and_last_displayed_rows_have_only_half_row_clearance(size, dark):
    steps = [_step(i) for i in range(size)]
    before = deepcopy(steps)
    fig = build_label_timeline_chart(steps, dark=dark)

    # The row coordinate, not trace count or token/duration magnitude, defines
    # the occupied vertical extent. Reversing bounds keeps earliest at top.
    axis = fig.layout.yaxis
    assert tuple(axis.range) == (size - 0.5, -0.5)
    assert axis.autorange is False
    assert tuple(axis.tickvals) == tuple(range(size))
    assert tuple(axis.ticktext) == tuple(str(i) for i in range(size))
    assert len(fig.data[0].y) == size
    assert max(fig.data[0].y) + 0.5 == axis.range[0]
    assert min(fig.data[0].y) - 0.5 == axis.range[1]
    assert list(fig.data[0].x) == [2.0] * size
    assert steps == before


def test_sparse_and_string_indices_remain_labels_without_expanding_the_axis():
    indices = [0, 7, "turn-9", 1_000_000]
    fig = build_label_timeline_chart([_step(i) for i in indices])

    assert tuple(fig.layout.yaxis.range) == (3.5, -0.5)
    assert list(fig.data[0].y) == [0, 1, 2, 3]
    assert list(fig.layout.yaxis.ticktext) == [str(i) for i in indices]
    assert "Step turn-9" in fig.data[0].hovertext[2]


def test_public_label_view_keeps_user_prompts_and_unknown_exclusion_compact():
    source = {"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1", "steps": [
        _step(0, role="user", phase="user", action="user_prompt", duration=0),
        _step(1),
        _step(2, phase="unknown", action="unknown"),
        _step(10, role="user", phase="user", action="user_prompt", duration=0),
        _step(1_000_000, phase="report", action="final_reporting", duration=5),
    ]}
    before = deepcopy(source)
    timeline = tv.label_views(source)["label_timeline"]
    fig = figure(timeline)

    assert [step["index"] for step in timeline.data["steps"]] == [0, 1, 10, 1_000_000]
    assert tuple(fig.layout.yaxis.range) == (3.5, -0.5)
    assert tuple(fig.layout.yaxis.ticktext) == ("0", "1", "10", "1000000")
    assert list(fig.data[0].y) == [0, 1, 2, 3]
    assert list(fig.data[1].y) == [0, 2]
    assert [annotation.y for annotation in fig.layout.annotations] == [0, 2]
    assert source == before


def test_empty_timeline_keeps_the_existing_no_data_state():
    fig = build_label_timeline_chart([])
    assert not fig.data
    assert fig.layout.height == 380
    assert fig.layout.annotations[0].text == "No step data"
