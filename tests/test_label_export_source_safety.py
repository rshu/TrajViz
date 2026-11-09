"""File-backed label exports preserve consumed evidence, including copied views."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.views import _jsonable


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _labels(*, trajectory: str = "", embedded_user: bool = True) -> dict:
    records = [
        {"index": 1, "role": "assistant", "phase": "implement", "action": "code_editing", "duration_s": 1.25},
        {"index": 2, "role": "assistant", "phase": "unknown", "action": "unknown", "duration_s": 0},
    ]
    if embedded_user:
        records.insert(0, {"index": 0, "role": "user", "phase": "user", "action": "user_prompt", "duration_s": 0})
    return {"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1", "model": "test-model",
            "trajectory_file": trajectory, "steps": records}


def _write(path: Path, data: dict | None = None) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_labels() if data is None else data) + "\n").encode()
    path.write_bytes(raw)
    return raw


@pytest.mark.parametrize("filename", ["labels_summary.json", "label_phase_counts.json", "index.json"])
@pytest.mark.parametrize("mapping_copy", [dict, copy.deepcopy])
def test_file_labels_are_protected_before_any_output_even_after_mapping_copy(tmp_path, filename, mapping_copy):
    source = tmp_path / filename
    original = _write(source)
    views = mapping_copy(tv.label_views(source))
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(views, tmp_path)
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_label_source_aliases_are_rejected_before_any_output(tmp_path, link):
    source = tmp_path / "labels.json"
    original = _write(source)
    output = tmp_path / "out"
    output.mkdir()
    alias = output / "label_timeline.json"
    try:
        alias.symlink_to(source) if link == "symbolic" else os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"File links unavailable: {exc}")
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(dict(tv.label_views(source)), output)
    assert source.read_bytes() == alias.read_bytes() == original
    assert list(output.iterdir()) == [alias]


@pytest.mark.parametrize("collision_side", ["reference", "compared"])
def test_label_comparison_protects_both_consumed_sidecars(tmp_path, collision_side):
    output = tmp_path / "out"
    source = output / "label_phase_count_comparison.json"
    original = _write(source)
    other = tmp_path / "other.json"
    other_original = _write(other)
    reference, compared = (source, other) if collision_side == "reference" else (other, source)
    views = dict(tv.label_comparison_views(reference, compared))
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(views, output, only=["label_phase_count_comparison"])
    assert source.read_bytes() == original and other.read_bytes() == other_original
    assert list(output.iterdir()) == [source]


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_label_comparison_also_protects_compared_file_aliases(tmp_path, link):
    reference, compared = tmp_path / "first.json", tmp_path / "second.json"
    _write(reference)
    original = _write(compared)
    output = tmp_path / "out"
    output.mkdir()
    alias = output / "label_phase_duration_comparison.json"
    try:
        alias.symlink_to(compared) if link == "symbolic" else os.link(compared, alias)
    except OSError as exc:
        pytest.skip(f"File links unavailable: {exc}")
    views = dict(tv.label_comparison_views(reference, compared))
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(views, output)
    assert compared.read_bytes() == alias.read_bytes() == original
    assert list(output.iterdir()) == [alias]


def test_v1_label_exports_protect_the_native_file_actually_read_for_timeline(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    native = output / "label_timeline.json"
    original = (FIXTURES / "codearts_minimal.json").read_bytes()
    native.write_bytes(original)
    labels = tmp_path / "labels.json"
    _write(labels, _labels(trajectory=str(native), embedded_user=False))
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(tv.label_views(labels), output)
    assert native.read_bytes() == original
    assert list(output.iterdir()) == [native]


def test_consumed_label_identity_is_bound_before_working_directory_changes(tmp_path, monkeypatch):
    source = tmp_path / "labels_summary.json"
    original = _write(source)
    monkeypatch.chdir(tmp_path)
    views = tv.label_views(Path(source.name))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(dict(views), tmp_path)
    assert source.read_bytes() == original


def test_private_export_guards_do_not_change_json_views_or_unknown_semantics(tmp_path):
    source = tmp_path / "private-label-source.json"
    payload = _labels()
    original = _write(source, payload)
    file_views, value_views = tv.label_views(source), tv.label_views(payload)
    assert file_views == value_views
    assert file_views["labels_summary"].data["unknown"] == 1
    assert file_views["labels_summary"].data["classified"] == 1
    output = tmp_path / "out"
    tv.export_json(dict(file_views), output)
    for key, view in file_views.items():
        exported = (output / f"{key}.json").read_text()
        assert json.loads(exported) == value_views[key].to_dict() == view.to_dict()
        assert str(source) not in exported and source.name not in exported
        assert _jsonable(view) == _jsonable(value_views[key])
    assert source.read_bytes() == original


def test_exporting_into_source_folder_is_allowed_when_no_filename_collides(tmp_path):
    source = tmp_path / "labels.json"
    original = _write(source)
    tv.export_json(tv.label_views(source), tmp_path)
    assert source.read_bytes() == original
    assert (tmp_path / "labels_summary.json").is_file()
    assert json.loads((tmp_path / "index.json").read_text())["schema_version"] == tv.VIEWS_SCHEMA_VERSION


def test_nested_dataclass_serialization_keeps_file_provenance_private(tmp_path):
    @dataclasses.dataclass
    class ReportData:
        chart: tv.View

    source = tmp_path / "private-evidence.json"
    payload = _labels()
    _write(source, payload)
    file_chart = tv.label_views(source)["label_timeline"]
    value_chart = tv.label_views(payload)["label_timeline"]
    actual = _jsonable(ReportData(file_chart))
    assert actual == _jsonable(ReportData(value_chart))
    assert str(source) not in json.dumps(actual)
    assert source.name not in json.dumps(actual)


def test_empty_view_selection_still_protects_source_from_index_export(tmp_path):
    source = tmp_path / "index.json"
    original = _write(source)
    with pytest.raises(ValueError, match="overwrite input source"):
        tv.export_json(dict(tv.label_views(source)), tmp_path, only=[])
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]
