"""Only authored chart chrome can change language; plotted identities cannot."""

import copy
import html

import plotly.graph_objects as go

from agent_trajviz.i18n import localize_figure, translate
from agent_trajviz.insight.charts.activity import build_context_pressure_chart
from agent_trajviz.insight.charts.usage import build_agent_token_chart, draw_tool_usage


_AGENT = 'Input/<img src=x>{agent} & "quoted"'
_TITLE = "Context Window Pressure — {agent}"


def test_dynamic_title_parameters_are_escaped_without_modifying_data():
    series = {
        "window_limit": 200_000,
        "events": [],
        "agents": [{"agent_id": "literal-agent", "label": _AGENT, "points": [{
            "step": 7, "local_turn": 1, "fresh": 600, "cache_read": 400, "occupancy": 1000,
        }]}],
    }
    before = copy.deepcopy(series)
    figure = build_context_pressure_chart([], series=series)
    assert series == before
    occupancy = next(trace for trace in figure.data if trace.name == "Occupancy")
    assert list(occupancy.x) == [7]
    assert list(occupancy.y) == [1000]
    assert occupancy.hovertext[0].startswith(_AGENT + "<br>")
    chrome = figure.layout.meta["tv_i18n"]["layout"]["title.text"]
    assert chrome == {"template": _TITLE, "params": {"agent": html.escape(_AGENT)}}
    assert figure.layout.title.text == "Context Window Pressure — " + html.escape(_AGENT)
    assert translate(chrome["template"], "zh-CN", **chrome["params"]) == (
        "上下文窗口压力 — " + html.escape(_AGENT)
    )
    assert translate(chrome["template"], "en", **chrome["params"]) == (
        "Context Window Pressure — " + html.escape(_AGENT)
    )


def test_repeated_chrome_capture_keeps_explicit_templates_and_existing_metadata():
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3.5, 4.5], name="Input", customdata=[_AGENT, _AGENT]))
    figure.update_layout(title=_TITLE.format(agent=_AGENT), meta={"tv_chart_height": 400})
    original = copy.deepcopy(figure.to_plotly_json()["data"])
    localize_figure(figure, layout_text={"title.text": (_TITLE, {"agent": _AGENT})})
    captured = copy.deepcopy(figure.layout.meta["tv_i18n"])
    localize_figure(figure)
    assert figure.layout.meta["tv_i18n"] == captured
    assert figure.layout.meta["tv_chart_height"] == 400
    assert figure.to_plotly_json()["data"] == original
    assert "traces" not in figure.layout.meta["tv_i18n"]
    figure.update_layout(title="Changed authored title")
    localize_figure(figure)
    assert figure.layout.meta["tv_i18n"]["layout"]["title.text"] == "Changed authored title"
    assert figure.to_plotly_json()["data"] == original


def test_tool_and_agent_names_matching_ui_words_are_never_legend_translations():
    data = {
        "agents": [{"id": "one", "label": "Input"}, {"id": "two", "label": _AGENT}],
        "tools": [
            {"tool": "Output", "calls": 3, "by_agent": {"one": 2, "two": 1}},
            {"tool": "Input", "calls": 4, "by_agent": {"one": 1, "two": 3}},
        ],
    }
    before = copy.deepcopy(data)
    figure = draw_tool_usage(data)
    assert data == before
    assert [trace.name for trace in figure.data] == ["Input", _AGENT]
    assert [list(trace.x) for trace in figure.data] == [[2, 1], [1, 3]]
    assert all(list(trace.y) == ["Output", "Input"] for trace in figure.data)
    assert "traces" not in figure.layout.meta["tv_i18n"]


def test_agent_token_categories_localize_but_agent_labels_remain_categories():
    summaries = [{"label": label, "total_tokens": 100, "input_tokens": 80,
                  "output_tokens": 20, "reasoning_tokens": 5, "cache_read_tokens": 10}
                 for label in ("Input", _AGENT)]
    before = copy.deepcopy(summaries)
    figure = build_agent_token_chart(summaries)
    assert summaries == before
    assert all(list(trace.x) == ["Input", _AGENT] for trace in figure.data)
    owned = figure.layout.meta["tv_i18n"]["traces"]
    assert set(owned.values()) == {"Fresh Input", "Cache Read", "Output", "Reasoning"}
    assert _AGENT not in owned.values() and "Input" not in owned.values()
    assert [list(trace.y) for trace in figure.data] == [[70, 70], [10, 10], [15, 15], [5, 5]]
