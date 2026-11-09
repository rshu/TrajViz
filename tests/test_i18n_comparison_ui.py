"""Comparison guidance and refreshes localize without altering supplied evidence."""

import json
from html.parser import HTMLParser

import gradio as gr

from agent_trajviz.i18n import translate
from agent_trajviz.insight.comparison import run_comparison
from agent_trajviz.insight.ui import awe_tab, comparison_tab


class _Markers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.markers = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-tv-i18n" in attrs:
            self.markers.append((attrs["data-tv-i18n"], json.loads(attrs.get("data-tv-i18n-args", "{}"))))


def _markers(value):
    parser = _Markers()
    parser.feed(value)
    return parser.markers


def test_comparison_placeholders_and_reset_keep_translatable_instructions():
    before = comparison_tab._reset_comparison()
    after = comparison_tab._reset_comparison()
    for first, second in zip(before[:2], after[:2], strict=True):
        assert first == second
        markers = _markers(second)
        assert markers
        for key, args in markers:
            assert translate(key, "zh-CN", **args) != key
    assert "baseline" in before[0]
    assert "compared trajectory" in before[0]


def test_comparison_error_prefix_localizes_but_supplied_error_is_not_a_translation_key():
    evidence = 'Tool coverage <img src=x onerror="run()"> /private/repo/Action.py'
    for source, key in (("ref_raw", "Error loading reference trajectory:"),
                        ("cmp_raw", "Error loading compared trajectory:")):
        args = {"ref_raw": {}, "cmp_raw": {}, source: {"_error": evidence}}
        result = run_comparison(**args)
        assert result["ok"] is False
        assert (key, {}) in _markers(result["report_html"])
        assert all(evidence != marker for marker, _ in _markers(result["report_html"]))
        assert "&lt;img" in result["report_html"] and '<img' not in result["report_html"]
        assert "/private/repo/Action.py" in result["report_html"]
        assert translate(key, "zh-CN") != key


def test_awe_settings_status_preserves_translation_markers_through_html_component():
    with gr.Blocks(analytics_enabled=False):
        refs = awe_tab.layout()
    status = refs.runtime["settings_status"]
    assert isinstance(status, gr.HTML)
    assert _markers(status.value) == [("No evaluation session settings applied.", {})]
    assert translate("No evaluation session settings applied.", "zh-CN") == "尚未应用评估会话设置。"
