"""Compaction request billing is distinct from its resulting context size."""
from copy import deepcopy
import json

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.figures import figures
from agent_trajviz.insight.context_usage import detect_compaction_events
from agent_trajviz.insight.formatting import format_context_pressure_html


def _native(tmp_path, before, after, request=173142):
    def tokens(n, out=10):
        return {"input": n, "output": out, "reasoning": 0, "cache": {"read": 0, "write": 0}}
    def assistant(identity, created, occupancy):
        return {"id": identity, "type": "assistant", "sessionID": "root", "agent": "build",
                "model": {"id": "unlisted-context-model", "providerID": "example"}, "finish": "stop",
                "time": {"created": created, "streamed": created + 50, "completed": created + 50},
                "tokens": tokens(occupancy), "content": [{"type": "text", "text": "Continue."}]}
    records = [assistant("before", 1000, before),
               {"id": "compact", "type": "compaction", "sessionID": "root", "status": "completed",
                "reason": "auto", "time": {"created": 2000}, "summary": "Keep relevant context.",
                "tokens": tokens(request, 1229)}]
    if after is not None:
        records.append(assistant("after", 4000, after))
    raw = {"info": {"id": "root", "location": {"directory": "/example"},
                    "time": {"created": 1000, "updated": 5000}}, "messages": records}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(raw))
    run = tv.load(path)
    expected = before + 10 + request + 1229 + (after + 10 if after is not None else 0)
    assert run.metrics["tokens"]["total"] == expected
    return run


def _validate(run):
    for name, view in run.views().items():
        jsonschema.Draft202012Validator(tv.view_schema(name)).validate(json.loads(view.to_json()))
    assert len(run.views()) == len(tv.VIEW_CATALOG)
    rendered = figures(run.views())
    assert len(rendered) == 12
    for chart in rendered.values():
        json.loads(chart.to_json())
    for raw in [False, True]:
        restored = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict(include_raw=raw))))
        assert restored.steps == run.steps
        assert restored.view("premature_compactions").data == run.view("premature_compactions").data
        if raw:
            assert restored.metrics == run.metrics


def test_billed_compaction_request_does_not_claim_context_growth(tmp_path):
    run = _native(tmp_path, 169786, 42308)
    event = detect_compaction_events(run.steps)[0]
    assert (event["occupancy_before"], event["occupancy_after"]) == (169786, 42308)
    assert event["dropped"] == 127478
    assert run.view("premature_compactions").data == []
    assert run.steps[1]["tokens"]["input"] == 173142
    _validate(run)


def test_growing_compaction_with_unknown_window_has_nullable_percentage(tmp_path):
    run = _native(tmp_path, 169786, 180000)
    flag = run.view("premature_compactions").data[0]
    assert flag["grew"] is True
    assert flag["before_pct"] is None
    assert flag["window_limit"] is None
    assert flag["occupancy_after"] == 180000
    series = run.view("context_pressure").data["series"]
    assert "window unknown" in format_context_pressure_html(series).lower()
    _validate(run)


def test_missing_post_compaction_measurement_stays_unknown(tmp_path):
    run = _native(tmp_path, 10000, None, request=9000)
    event = detect_compaction_events(run.steps)[0]
    assert event["occupancy_after"] is None
    assert event["dropped"] is None
    flag = run.view("premature_compactions").data[0]
    assert flag["occupancy_after"] is None
    assert flag["before_pct"] == 7.8
    _validate(run)


@pytest.mark.parametrize("summary_fields", [{"summary": True}, {"mode": "compaction"}, {"agent": "compaction"}])
def test_post_measurement_skips_other_windows_earlier_requests_and_summary_requests(tmp_path, summary_fields):
    run = _native(tmp_path, 169786, 42308)
    before, compact, after = deepcopy(run.steps)
    compact["time_completed_ms"] = 3000
    other = deepcopy(after)
    other.update(session_id="other", time_created_ms=3500)
    other["tokens"].update(input=300000, total=300010)
    earlier = deepcopy(after)
    earlier["time_created_ms"] = 2500
    earlier["tokens"].update(input=190000, total=190010)
    summary = deepcopy(after)
    summary.update(time_created_ms=3500, **summary_fields)
    summary["tokens"].update(input=200000, total=200010)
    steps = [before, compact, other, earlier, summary, after]
    for i, step in enumerate(steps):
        step["index"] = i
    event = detect_compaction_events(steps)[0]
    assert event["occupancy_after"] == 42308


def test_explicit_zero_post_context_is_preserved(tmp_path):
    run = _native(tmp_path, 169786, 42308)
    steps = deepcopy(run.steps)
    steps[1]["tokens"]["context_window"] = 0
    assert detect_compaction_events(steps)[0]["occupancy_after"] == 0
