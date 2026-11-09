"""Comparison cards honor the explicit host theme rather than the OS alone."""

from __future__ import annotations

import copy
import re

import pytest

from agent_trajviz.converge.rendering import build_comparison_report_html
from agent_trajviz.converge.styles import CONVERGE_CSS


def _property(selector, name):
    block = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", CONVERGE_CSS)[1]
    return re.search(r"(?:^|;)\s*" + name + r":\s*([^;]+)", block)[1]


def _resolve(value, variables):
    match = re.fullmatch(r"var\((--[\w-]+)(?:,\s*(.*))?\)", value)
    if not match:
        return value
    return _resolve(variables.get(match[1], match[2]), variables)


def _contrast(first, second):
    def luminance(color):
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in rgb]
        return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _palette(dark):
    blocks = re.findall(r":root\s*\{([^}]+)\}", CONVERGE_CSS)
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", blocks[int(dark)]))


@pytest.mark.parametrize("dark", [False, True])
def test_host_theme_overrides_opposite_os_card_heading_and_milestone_cells(dark):
    variables = _palette(not dark)
    variables.update(
        {
            "--block-background-fill": "#1f2937" if dark else "#ffffff",
            "--body-text-color": "#e5e7eb" if dark else "#1f2937",
            "--background-fill-secondary": "#111827" if dark else "#f3f4f6",
        }
    )
    card = _resolve(_property(".cvg-report", "background"), variables)
    text = _resolve(_property(".cvg-report", "color"), variables)
    heading = _resolve(_property(".cvg-report h2", "color"), variables)
    cells = _resolve(_property(".gradio-container .cvg-report table td", "color"), variables)
    header = _resolve(_property(".gradio-container .cvg-report table th", "background"), variables)
    alternate = _resolve(_property(".cvg-milestone-table tr:nth-child(even) td", "background"), variables)
    assert card == variables["--block-background-fill"]
    assert text == heading == cells == variables["--body-text-color"]
    assert min(_contrast(text, card), _contrast(cells, header), _contrast(cells, alternate)) >= 4.5


@pytest.mark.parametrize("dark", [False, True])
def test_standalone_light_and_dark_reports_keep_readable_existing_palette(dark):
    variables = _palette(dark)
    card = _resolve(_property(".cvg-report", "background"), variables)
    text = _resolve(_property(".cvg-report", "color"), variables)
    cells = _resolve(_property(".gradio-container .cvg-report table td", "color"), variables)
    alternate = _resolve(_property(".cvg-milestone-table tr:nth-child(even) td", "background"), variables)
    assert card == variables["--cvg-card"] and text == cells == variables["--cvg-text"]
    assert min(_contrast(text, card), _contrast(cells, alternate)) >= 4.5


def test_rendered_milestone_card_embeds_host_theme_rules_without_changing_evidence():
    report = {
        "ref_milestones": {"first_edit": 13, "first_passing_validation": None},
        "cmp_milestones": {"first_edit": 22, "first_passing_validation": None},
    }
    before = copy.deepcopy(report)
    output = build_comparison_report_html(report)
    assert 'class="cvg-milestone-table"' in output
    assert "#13" in output and "#22" in output and "N/A" in output
    assert ".gradio-container .cvg-report table td" in output
    assert "var(--body-text-color, var(--cvg-text))" in output
    assert report == before
