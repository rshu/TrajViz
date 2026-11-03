"""Model catalog capacity is an estimate, not a native session declaration."""
from __future__ import annotations

import pytest

from agent_trajviz.insight.context_usage import (
    DEFAULT_CONTEXT_WINDOW_LIMIT,
    context_pressure_series,
    context_pressure_stats,
    resolve_context_window,
)
from agent_trajviz.insight.formatting import format_context_pressure_html


def step(model, occupancy=93_700):
    return {"index": 0, "role": "assistant", "agent": "", "model_id": model,
            "tokens": {"input": occupancy, "output": 100, "total": occupancy + 100,
                       "cache_read": 0, "cache_write": 0}, "tool_calls": [], "parts": []}


@pytest.mark.parametrize("model", [
    "deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
    "deepseek/deepseek-flash", "openrouter/deepseek/deepseek-v4-pro", "DeepSeek/DeepSeek-Flash",
])
def test_current_deepseek_aliases_infer_capacity_without_native_declaration(model):
    raw = {"metadata": {"model": model}}
    limit, source, note = resolve_context_window([step(model)], raw)
    assert limit == 1_048_576
    assert source == "model"
    assert "estimated" in note.lower()
    assert "does not report" in note.lower()
    assert "context_window_limit" not in raw["metadata"]


@pytest.mark.parametrize(("occupancy", "percentage"), [(93_700, 8.9), (108_540, 10.4)])
def test_real_capture_peak_percentages_use_catalog_capacity_and_disclose_inference(occupancy, percentage):
    series = context_pressure_series([step("deepseek-flash", occupancy)])
    stats = context_pressure_stats(series)
    assert stats["window_limit"] == 1_048_576
    assert stats["window_source"] == "model"
    assert stats["peak_occupancy"] == occupancy
    assert stats["peak_pct"] == percentage
    assert "estimated" in format_context_pressure_html(series).lower()


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner", "deepseek-flash-unknown", "deepseek-v4-pro-custom"])
def test_other_deepseek_models_do_not_inherit_capacity_from_a_family_prefix(model):
    limit, source, note = resolve_context_window([step(model)])
    assert (limit, source, note) == (DEFAULT_CONTEXT_WINDOW_LIMIT, "default", "")


def test_declared_capacity_and_explicit_override_take_precedence_over_model_estimate():
    steps = [step("deepseek-flash")]
    raw = {"metadata": {"context_window_limit": 262_144}}
    assert resolve_context_window(steps, raw) == (262_144, "declared", "")
    assert resolve_context_window(steps, raw, override=524_288) == (524_288, "override", "")


def test_occupancy_exceeding_model_catalog_capacity_stays_unknown():
    limit, source, note = resolve_context_window([step("deepseek-flash", 1_100_000)])
    assert limit is None
    assert source == "unknown"
    assert "model's listed" in note
