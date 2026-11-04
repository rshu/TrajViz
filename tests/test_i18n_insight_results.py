"""Localization marks authored presentation while preserving trajectory data."""

from __future__ import annotations

import html
import json
from html.parser import HTMLParser

from agent_trajviz.i18n import translate
from agent_trajviz.insight.formatting import _metric_chip
from agent_trajviz.insight.presenters.issues import _issue_title_html
from agent_trajviz.insight.presenters.overview import _build_session_detail_html
from agent_trajviz.insight.rendering import _format_tool_call_detail, render_filter_chips


class _Surface(HTMLParser):
    def __init__(self):
        super().__init__()
        self.markers = []
        self.protected_markers = []
        self.protected = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("pre", "code"):
            self.protected += 1
        attributes = dict(attrs)
        if "data-tv-i18n" in attributes:
            self.markers.append(attributes)
            if self.protected:
                self.protected_markers.append(attributes)

    def handle_endtag(self, tag):
        if tag in ("pre", "code"):
            self.protected -= 1


def test_tool_labels_translate_without_marking_name_input_or_output():
    raw = 'Input Workflow Metrics <span data-tv-i18n="Steps">keep this evidence</span>'
    rendered = _format_tool_call_detail({
        "tool_name": "Input", "status": "completed", "input": {"command": raw}, "output": raw,
    })
    surface = _Surface()
    surface.feed(rendered)
    assert {marker["data-tv-i18n"] for marker in surface.markers} >= {"Tool", "Input", "Output"}
    assert not surface.protected_markers
    assert "<code>Input</code>" in rendered
    assert html.escape(raw) in rendered
    assert raw not in rendered


def test_identical_tool_and_metric_names_have_separate_localization_behavior():
    owned = _metric_chip("Steps", "7")
    recorded = _metric_chip("Steps", "7", localize_label=False)
    assert 'data-tv-i18n="Steps"' in owned
    assert "data-tv-i18n" not in recorded
    assert "Steps" in recorded


def test_session_model_and_branch_values_remain_unmarked():
    rendered = _build_session_detail_html({}, {"branch": "Workflow"}, model_id="Steps")
    surface = _Surface()
    surface.feed(rendered)
    keys = {marker["data-tv-i18n"] for marker in surface.markers}
    assert {"Model", "Branch", "Session details"} <= keys
    assert "Workflow" not in keys and "Steps" not in keys
    assert ">Steps</span>" in rendered and ">Workflow</span>" in rendered


def test_agent_filter_keeps_recorded_name_and_canonical_token():
    rendered = render_filter_chips(agent_options=[("agent:recorded-id", "Assistant", 0)])
    button = rendered[rendered.index("data-filter='agent:recorded-id'"):].split("</button>", 1)[0]
    assert ">Assistant" in button
    assert "data-tv-i18n" not in button
    assert "data-filter='Assistant'" in rendered


def test_bottleneck_title_preserves_untrusted_tool_target_as_an_argument():
    tool = 'Input: <img src=x onerror="alert(1)">'
    rendered = _issue_title_html(f"Tool bottleneck: {tool} (12.5s)")
    surface = _Surface()
    surface.feed(rendered)
    assert len(surface.markers) == 1
    marker = surface.markers[0]
    params = json.loads(marker["data-tv-i18n-args"])
    assert params == {"tool": tool, "seconds": "12.5"}
    assert translate(marker["data-tv-i18n"], "zh-CN", **params) == f"工具瓶颈：{tool}（12.5 秒）"
    assert "<img" not in rendered
