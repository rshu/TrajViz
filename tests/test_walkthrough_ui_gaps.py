"""Regressions found while walking real sessions through the dashboard."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.insight import build_ui
from agent_trajviz.insight.presenters.label_ui import validate_labels_for_run
from agent_trajviz.insight.ui import upload


def _run():
    return tv.Trajectory.from_steps([
        {"index": 0, "raw_index": 4, "role": "user", "session_id": "native-session", "parts": [], "tool_calls": []},
        {"index": 1, "raw_index": 9, "role": "assistant", "session_id": "native-session", "parts": [], "tool_calls": []},
    ])


@pytest.mark.parametrize("record", [
    {"index": 99}, {"index": True}, {"index": 1, "raw_index": 4},
    {"index": 1, "role": "user"}, {"index": 1, "session_id": "other-session"},
])
def test_contradictory_label_identity_is_rejected(record):
    with pytest.raises(ValueError):
        validate_labels_for_run({"steps": [record]}, _run())


def test_sparse_and_legacy_labels_do_not_require_full_coverage_or_new_fields():
    validate_labels_for_run({"steps": [{"index": 1, "phase": "understand", "action": "code_reading"}]}, _run())
    validate_labels_for_run({"steps": [{"index": 1, "role": "assistant", "raw_index": 9,
                                        "session_id": "native-session"}]}, _run())
    with pytest.raises(ValueError, match="duplicate"):
        validate_labels_for_run({"steps": [{"index": 1}, {"index": 1}]}, _run())


@pytest.fixture(scope="module")
def app():
    app = build_ui()
    yield app
    app.close()


def _handler(app, name):
    return next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == name)


def test_labels_callback_validates_the_actual_loaded_run(app, tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"taxonomy_version": "v1", "steps": [{
        "index": 1, "raw_index": 500, "phase": "understand", "action": "code_reading",
    }]}))
    handler = _handler(app, "do_load_labels")
    result = handler.fn(str(labels), False, _run())
    assert "do not match" in result[1]
    assert not result[2]["visible"]
    compared_state = _handler(app, "on_run_comparison").inputs[5]
    assert compared_state in handler.inputs


def test_reference_capture_selection_reaches_the_comparison(app, tmp_path):
    from tests.test_otlp_e2e import _bundle

    _, archive = _bundle(tmp_path, partial=True)
    inspect = _handler(app, "inspect_reference_sessions")
    choices = inspect.fn(str(archive), "")
    assert choices["visible"] and choices["value"] is None
    assert {value for _, value in choices["choices"]} == {"demo-session", "second-session"}
    hidden = inspect.fn(str(archive), "claude_code")
    assert not hidden["visible"] and hidden["value"] is None and hidden["choices"] == []
    compared = _run()
    selected = tv.load(archive, session_id="second-session")
    with patch("agent_trajviz.insight.ui.comparison_tab.run_comparison", return_value={"report_html": "selected", "ok": True}) as compare:
        result = _handler(app, "on_run_comparison").fn(str(archive), None, "", None, None, compared, False, "second-session")
    assert result[0] == "selected"
    assert compare.call_args.kwargs["ref_steps"] == selected.steps
    assert compare.call_args.kwargs["ref_raw"]["_otel"]["session_id"] == "second-session"
    reference = _handler(app, "on_run_comparison").inputs[-1]
    assert reference.label == "Reference captured session"
    targets = {name for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "inspect_reference_sessions"
               for _, name in fn.targets}
    assert targets == {"change"}


def test_empty_warning_component_has_no_decorative_warning_class(app):
    refs = _handler(app, "_do_load").outputs
    banners = [component for component in refs if getattr(component, "get_block_name", lambda: "")() == "html"
               and "summary-banner" in (getattr(component, "elem_classes", []) or [])]
    assert banners == []
    empty = upload.pack_load(None)
    assert empty["summary_banner"]["value"] == ""
    assert not empty["summary_banner"]["visible"]
    warning = upload.pack_load(None, banner="Specific loading warning")
    assert 'class="summary-banner"' in warning["summary_banner"]["value"]
    assert "Specific loading warning" in warning["summary_banner"]["value"]


def test_misaligned_pairwise_labels_do_not_invent_phase_charts(app, tmp_path):
    run = _run()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(run.to_dict()))
    good = tmp_path / "good-labels.json"
    bad = tmp_path / "bad-labels.json"
    data = {"taxonomy_version": "v1", "steps": [{"index": 1, "phase": "understand", "action": "code_reading"}]}
    good.write_text(json.dumps(data))
    data["steps"][0]["raw_index"] = 999
    bad.write_text(json.dumps(data))
    with patch("agent_trajviz.insight.ui.comparison_tab.run_comparison", return_value={"report_html": "core comparison", "ok": True}):
        result = _handler(app, "on_run_comparison").fn(str(reference), None, "", str(bad), str(good), run, False)
    assert result[0] == "core comparison"
    assert "Label comparison unavailable" in result[3]
    assert not result[1].data and not result[2].data


def test_summary_parent_hides_empty_state_and_reveals_valid_labels(app, tmp_path):
    from agent_trajviz.insight.session import session_from_run

    run = _run()
    assert not upload.pack_load(session_from_run(run))["summary_area"]["visible"]
    labels = tmp_path / "sparse-labels.json"
    labels.write_text(json.dumps({"taxonomy_version": "v1", "steps": [{
        "index": 1, "phase": "understand", "action": "code_reading",
    }]}))
    handler = _handler(app, "do_load_labels")
    assert handler.outputs[0].visible is False
    loaded = handler.fn(str(labels), False, run)
    assert "Labels loaded" in loaded[0]["value"]
    assert loaded[0]["visible"] and loaded[-1]["visible"]
    cleared = handler.fn(None, False, run)
    assert cleared[0]["value"] == ""
    assert not cleared[0]["visible"] and not cleared[-1]["visible"]
    labels.write_text(json.dumps({"taxonomy_version": "v1", "steps": [{"index": 999}]}))
    failed = handler.fn(str(labels), False, run)
    assert "Labels could not be loaded" in failed[0]["value"]
    assert failed[0]["visible"] and failed[-1]["visible"]


def test_label_clear_keeps_current_run_warning_parent_visible(app):
    from agent_trajviz.insight.session import session_from_run

    payload = _run().to_dict(include_raw=True)
    payload["raw"]["_load_warnings"] = ["Some records were unavailable"]
    run = tv.Trajectory.from_dict(payload)
    packed = upload.pack_load(session_from_run(run))
    assert packed["summary_area"]["visible"]
    assert "Some records were unavailable" in packed["summary_banner"]["value"]
    cleared = _handler(app, "do_load_labels").fn(None, False, run)
    assert cleared[0]["value"] == ""
    assert not cleared[0]["visible"] and cleared[-1]["visible"]
    reset = _handler(app, "_reset_labels")
    assert _handler(app, "do_load_labels").outputs[-1] not in reset.outputs


def test_label_clear_does_not_hide_a_failed_load_banner(app):
    result = _handler(app, "do_load_labels").fn(None, False, None, "<p>Could not load this trajectory</p>")
    assert result[0]["value"] == ""
    assert not result[0]["visible"] and result[-1]["visible"]
