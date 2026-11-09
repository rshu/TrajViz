"""Context redraws disclose inferred capacities and preserve explicit overrides."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.context_usage import context_pressure_series
from agent_trajviz.insight.formatting import format_context_pressure_html
from agent_trajviz.insight.session import session_from_run
from agent_trajviz.insight.ui import overview_tab


def _step(model="deepseek-flash"):
    return {
        "index": 0,
        "role": "assistant",
        "model_id": model,
        "agent": "",
        "tokens": {"input": 93700, "output": 100, "total": 93800, "cache_read": 0, "cache_write": 0},
        "parts": [],
        "tool_calls": [],
    }


def test_loaded_model_capacity_is_not_written_into_manual_override_input():
    run = tv.Trajectory.from_steps([_step()])
    session = session_from_run(run)
    before = copy.deepcopy(run.raw)
    values = overview_tab.pack_load(session)
    assert values["diag_window_limit"] is None
    assert "estimated from the model name" in values["diag_pressure_html"]
    assert "does not report" in values["diag_pressure_html"]
    assert run.raw == before


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("override", "Capacity manually overridden to 1,048,576 tokens (user supplied)."),
        ("declared", "Capacity reported by the trajectory: 1,048,576 tokens."),
        ("model", "Capacity estimated from the model name"),
        ("default", "Capacity assumed: 1,048,576 tokens"),
    ],
)
def test_every_numeric_capacity_basis_is_visible_without_mutating_series(source, expected):
    series = context_pressure_series([_step()])
    series["window_source"], series["window_note"] = source, ""
    before = copy.deepcopy(series)
    output = format_context_pressure_html(series)
    assert "class='context-window-basis'" in output
    assert expected in output
    assert series == before


def test_context_callback_roundtrip_keeps_inference_until_user_types_override():
    from agent_trajviz.insight import insight

    with patch.object(insight, "load_env_files"):
        app = insight.build_ui(allow_server_paths=False)
    try:
        callback = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "_rebuild_utilization")
        steps = [_step()]
        raw = {"metadata": {"model": "deepseek-flash"}}
        _fig, detected = callback("__all__", None, "current", steps, raw, False)
        assert "estimated from the model name" in detected and "does not report" in detected
        _fig, manual = callback("__all__", 524288, "current", steps, raw, False)
        assert "manually overridden to 524,288 tokens" in manual
        assert "estimated from the model name" not in manual
        _fig, cleared = callback("__all__", None, "current", steps, raw, False)
        assert "estimated from the model name" in cleared
    finally:
        app.close()


def test_basis_note_is_escaped_and_unknown_capacity_retains_its_reason():
    series = context_pressure_series([_step()])
    series["window_note"] = "<img src=x onerror=alert(1)>"
    output = format_context_pressure_html(series)
    assert "&lt;img" in output and "<img" not in output
    step = _step("unknown-model")
    step["tokens"]["input"] = 175000
    series = context_pressure_series([step])
    output = format_context_pressure_html(series)
    assert "real window is unknown" in output
    assert "window unknown" in output
