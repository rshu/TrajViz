"""Compact file-coverage headers keep the full recorded run identity accessible."""

import re
from html.parser import HTMLParser

from agent_trajviz.insight.styles import APP_CSS

from agent_trajviz.insight.run_group import _render_file_matrix_html


class _Headers(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.headers, self.compact = [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "rg-tree-runhead" in values.get("class", "").split():
            self.headers.append(values)
        if "-webkit-line-clamp:2" in values.get("style", ""):
            self.compact.append(values)


def test_file_headers_wrap_in_two_lines_and_retain_escaped_full_hover_identity():
    labels = {
        "a": "Long recorded run identity including model and harness A <script>bad()</script>",
        "b": "Long recorded run identity including model and harness B",
    }
    value = {
        "run_ids": ["a", "b"],
        "labels": labels,
        "file_matrix": [
            {
                "path": "src/a.py",
                "kind": "consensus",
                "n_runs": 2,
                "cells": {"a": {"read": 1, "write": 0}, "b": {"read": 1, "write": 1}},
            }
        ],
    }
    output = _render_file_matrix_html(value)
    parsed = _Headers(output)
    assert [header["title"] for header in parsed.headers] == list(labels.values())
    assert all(header["aria-label"] == header["title"] for header in parsed.headers)
    assert all(
        "max-width:100%" in header["style"] and "overflow:hidden" in header["style"] for header in parsed.headers
    )
    assert len(parsed.compact) == 2
    assert all("white-space:normal" in value["style"] for value in parsed.compact)
    assert "<script>bad()" not in output and "&lt;script&gt;bad()&lt;/script&gt;" in output


def test_header_directory_and_file_rows_share_font_independent_end_columns():
    # Header text is 11px, row text 14px. em tracks would misalign the actual
    # run outcomes even though every row uses the same CSS declaration.
    shared = re.search(
        r"\.rg-tree-head,\s*\.rg-tree-summary,\s*\.rg-tree-file\s*\{([^}]+)\}",
        APP_CSS,
    )[1]
    tracks = re.search(r"grid-template-columns:\s*([^;]+)", shared)[1]
    assert tracks == "minmax(12rem, 1fr) repeat(var(--rg-run-cols, 2), 5.6rem) 14rem"
    # Fixed trailing columns keep all data aligned from the right edge even
    # when directory indentation reduces only the flexible filename column.
    assert "minmax(7em, auto)" not in tracks
    mix = re.search(r"\.rg-tree-mix\s*\{([^}]+)\}", APP_CSS)[1]
    assert "min-width: 0" in mix
    assert "white-space: normal" in mix and "overflow-wrap: anywhere" in mix
