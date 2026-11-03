"""Run-group rankings must not reward or penalize unrecorded observations."""

from __future__ import annotations

import copy
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.run_group import (
    _best_worst_flags,
    build_run_group_scorecard_html,
    build_run_scorecard_row,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "cursor_minimal.json"


class _Rows(HTMLParser):
    def __init__(self, output):
        super().__init__(convert_charrefs=True)
        self.rows, self.current, self.cell = [], None, None
        self.feed(output)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current = []
        elif tag == "td":
            self.cell = {"attrs": dict(attrs), "text": ""}

    def handle_data(self, text):
        if self.cell is not None:
            self.cell["text"] += text

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.current.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.current:
            self.rows.append(self.current)
            self.current = None


def _cursor(tmp_path, *, unknown_only=False, all_failed=False):
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    if unknown_only:
        for key in ("time", "promptTokenBreakdown", "contextUsagePercent"):
            document["info"].pop(key, None)
    for message in document["messages"]:
        if unknown_only:
            message["info"].pop("time", None)
            message["info"].pop("tokens", None)
        for part in message.get("parts", []):
            if part.get("type") == "tool" and (unknown_only or all_failed):
                part["state"] = {
                    "status": "unknown" if unknown_only else "error",
                    "input": part["state"].get("input", {}),
                    "output": "" if unknown_only else "Error: fixture failure",
                }
    source = tmp_path / "cursor.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    return tv.load(source)


def test_partial_cursor_capture_keeps_missing_tool_context_and_clock_metrics_nullable(tmp_path):
    run = _cursor(tmp_path, unknown_only=True)
    original = copy.deepcopy(run.to_dict(include_raw=True))
    assert run.metrics["tool_call_count"] > 0
    assert run.metrics["tool_observed_outcomes"] == 0
    assert run.metrics["tool_success_rate"] is None
    assert not run.context_pressure()["series"]["agents"]
    row = build_run_scorecard_row(run.raw, path=str(run.source), steps=run.steps, fmt=run.format)
    for key in ("tool_success_pct", "peak_occupancy", "peak_pct", "compactions", "wall_clock_s"):
        assert row[key] is None, key
    assert run.to_dict(include_raw=True) == original


def test_partially_observed_outcomes_do_not_turn_an_abstention_into_zero(tmp_path):
    run = _cursor(tmp_path)
    assert run.metrics["tool_observed_outcomes"] > 0 and run.metrics["tool_unknown_outcomes"] > 0
    assert run.metrics["tool_success_rate"] is None
    row = build_run_scorecard_row(run.raw, steps=run.steps, fmt=run.format)
    assert row["tool_success_pct"] is None
    assert row["peak_occupancy"] > 0 and row["compactions"] == 0


def test_observed_failures_remain_a_real_zero_percent_success_rate(tmp_path):
    run = _cursor(tmp_path, all_failed=True)
    assert run.metrics["tool_success_rate"] == 0
    row = build_run_scorecard_row(run.raw, steps=run.steps, fmt=run.format)
    assert row["tool_success_pct"] == 0
    rendered = build_run_group_scorecard_html({"rows": [row]}, include_behavior=False)
    cells = _Rows(rendered).rows[0]
    assert cells[7]["text"] == "0%"
    assert cells[8]["text"] != "N/A" and cells[9]["text"] == "0"


def _row(run_id, **values):
    return {
        "run_id": run_id,
        "label": run_id,
        "error": None,
        "format": "fixture",
        "finished": True,
        "steps": 1,
        "tokens": 100,
        "tool_calls": 1,
        "wall_clock_s": None,
        "wall_clock_fmt": "n/a",
        "tool_success_pct": None,
        "peak_occupancy": None,
        "peak_pct": None,
        "compactions": None,
        **values,
    }


def test_unavailable_metrics_are_na_without_best_or_worst_styles():
    missing = _row("unobserved")
    low = _row(
        "observed-low", wall_clock_s=10, wall_clock_fmt="10s", tool_success_pct=0, peak_occupancy=100, compactions=0
    )
    high = _row(
        "observed-high", wall_clock_s=20, wall_clock_fmt="20s", tool_success_pct=100, peak_occupancy=200, compactions=1
    )
    rows = [missing, low, high]
    flags = _best_worst_flags(rows)
    for key in ("wall_clock_s", "tool_success_pct", "peak_occupancy", "compactions"):
        assert "unobserved" not in flags[key]["best"] | flags[key]["worst"]
    output = build_run_group_scorecard_html({"rows": rows}, include_behavior=False)
    rendered = _Rows(output).rows
    assert [rendered[0][index]["text"] for index in (7, 8, 9)] == ["N/A", "N/A", "N/A"]
    assert all("style" not in rendered[0][index]["attrs"] for index in (4, 7, 8, 9))
    assert rendered[1][7]["text"] == "0%" and rendered[1][9]["text"] == "0"
    assert "N/A metrics lack evidence and are excluded from ranking" in output


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), False])
def test_non_measurements_do_not_enter_best_worst_rankings(invalid):
    flags = _best_worst_flags(
        [
            _row("missing", peak_occupancy=invalid),
            _row("recorded-zero", peak_occupancy=0),
            _row("recorded-ten", peak_occupancy=10),
        ]
    )
    assert flags["peak_occupancy"] == {"best": {"recorded-zero"}, "worst": {"recorded-ten"}}
