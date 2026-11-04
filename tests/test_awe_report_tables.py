"""Wide report tables keep measurements intact and full evidence accessible."""

from __future__ import annotations

import csv
import io
from html.parser import HTMLParser
from types import SimpleNamespace

from i18n_test_helpers import unmark_html

from agent_trajviz.awe import render


class _TableContent(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags, self.blocks, self.current = [], [], None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "pre":
            self.current = []

    def handle_data(self, text):
        if self.current is not None:
            self.current.append(text)

    def handle_endtag(self, tag):
        if tag == "pre" and self.current is not None:
            self.blocks.append("".join(self.current))
            self.current = None


def _csv_report(headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    original = stream.getvalue()
    run = SimpleNamespace(artifacts={"evidence.csv": original})
    return unmark_html(render.report_html(run, "evidence.csv")), original


def test_compact_identifiers_and_full_csv_precision_have_natural_nowrap_columns():
    identifier = "agent__astropy__astropy-13033"
    number = "2.6760168302945306"
    output, original = _csv_report(["case_id", "risk_ratio", "n"], [[identifier, number, 0]])
    assert "<table>" in output
    assert f"<td>{identifier}</td>" in output
    assert f"<td>{number}</td>" in output
    assert "<td>0</td>" in output
    assert "awe-report-scroll" in output and "Scrollable report table" in output
    assert "table{width:max-content;min-width:100%;table-layout:auto}" in output
    assert "td{white-space:nowrap;overflow-wrap:normal;word-break:normal}" in output
    assert original in _TableContent(output).blocks


def test_long_untrusted_evidence_has_bounded_disclosure_and_exact_full_value():
    evidence = '{"snippet": "<script>fixture()</script>", "recorded": "' + ("evidence " * 40) + '"}'
    output, original = _csv_report(["case_id", "evidence_json"], [["case-1", evidence]])
    parsed = _TableContent(output)
    details = [
        attrs for tag, attrs in parsed.tags if tag == "details" and attrs.get("class") == "awe-report-cell-details"
    ]
    assert len(details) == 1 and "open" not in details[0]
    assert "Preview (" in output and "characters):" in output
    assert evidence in parsed.blocks and original in parsed.blocks
    assert "Complete original artifact" in output
    assert "max-height:260px" in output and "width:clamp(220px,28vw,380px)" in output
    assert not any(tag in {"script", "img", "iframe"} for tag, _ in parsed.tags)
    assert "&lt;script&gt;fixture()&lt;/script&gt;" in output


def test_multiline_cells_keep_exact_lines_while_missing_and_empty_stay_distinct():
    evidence = "First recorded line\r\nSecond recorded line, with comma"
    output, original = _csv_report(["case", "evidence", "blank"], [["case-1", evidence, ""], ["case-2"]])
    assert "awe-report-cell-details" in output
    assert evidence in _TableContent(output).blocks
    assert original in _TableContent(output).blocks
    assert "<td></td>" in output and "<td>Not recorded</td>" in output


def test_row_limit_is_explicit_and_complete_original_csv_remains_available():
    rows = [[f"case-{number}"] for number in range(2001)]
    output, original = _csv_report(["case"], rows)
    parsed = _TableContent(output)
    assert sum(tag == "td" for tag, _ in parsed.tags) == 2000
    assert "Showing the first 2,000 rows" in output
    assert "<td>case-1999</td>" in output and "<td>case-2000</td>" not in output
    assert original in parsed.blocks and "case-2000" in original
