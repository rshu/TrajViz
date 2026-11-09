"""The Workflow toolbar must not squeeze filters between desktop-width controls."""

from __future__ import annotations

import re

import gradio as gr

from agent_trajviz.insight.styles import APP_CSS
from agent_trajviz.insight.ui import workflow_tab


def _paths(node, target, ancestors=()):
    current = (*ancestors, node.get("id"))
    if node.get("id") == target:
        yield current
    for child in node.get("children", []):
        yield from _paths(child, target, current)


def test_toolbar_has_scoped_layout_hooks_and_no_fixed_search_minimum():
    with gr.Blocks() as app, gr.Tabs():
        refs = workflow_tab.layout()
    try:
        rows = [
            block
            for block in app.blocks.values()
            if isinstance(block, gr.Row) and "wf-toolbar" in (block.elem_classes or [])
        ]
        assert len(rows) == 1
        layout = app.get_config_file()["layout"]
        for component in [refs.wf_toc_toggle, refs.wf_filter_chips_html, refs.wf_search]:
            path = next(_paths(layout, component._id))
            assert rows[0]._id in path
        assert refs.wf_filter_chips_html.elem_id == "wf-filter-chips"
        assert "wf-toolbar-search" in refs.wf_search.elem_classes
        assert refs.wf_search.min_width == 0
    finally:
        app.close()


def test_small_screen_css_stacks_toolbar_and_gives_chips_and_search_full_width():
    responsive = APP_CSS.split("/* Keep Workflow chips usable", 1)[1]
    assert "@media (max-width: 768px)" in responsive
    toolbar = re.search(r"\.wf-toolbar\s*\{([^}]+)\}", responsive)[1]
    assert "flex-direction: column !important" in toolbar
    assert "align-items: stretch !important" in toolbar
    controls = re.search(
        r"\.wf-toolbar #wf-filter-chips,\s*\.wf-toolbar \.wf-toolbar-search\s*\{([^}]+)\}", responsive
    )[1]
    for declaration in [
        "width: 100% !important",
        "min-width: 0 !important",
        "max-width: 100% !important",
        "flex: 0 0 auto !important",
    ]:
        assert declaration in controls
    bar = re.search(r"\.wf-toolbar #wf-filter-bar\s*\{([^}]+)\}", responsive)[1]
    assert "width: 100%" in bar and "min-width: 0" in bar
    assert "box-sizing: border-box" in bar
