"""Component provenance and estimated composition remain distinct from occupancy."""

from __future__ import annotations

import copy

from agent_trajviz.insight.prompt_components import CATEGORIES
from agent_trajviz.insight.ui.prompt_composition import render_component_explorer
from agent_trajviz.insight.ui.prompt_evolution import evolution_chart


def _composition(amount=12, *, partial=False):
    return {"evidence": "estimated", "basis": "serialized_visible_content", "total": amount,
            "partial": partial, "component_count": 1,
            "categories": [{"category": "rules", "estimated_tokens": amount, "component_count": 1}]}


def _detail():
    segment = {"category": "rules", "category_basis": "recorded_section", "logical_id": "rules:project",
               "identity_basis": "native_section", "content_hash": "sha256:version-one",
               "occurrence_id": "request-1:rules:0", "json_pointer": "/content/0/text",
               "estimated_tokens": 12, "kind": "content", "provenance": {"source_refs": [{"raw_index": 3}]}}
    return {"evidence": "reconstructed", "composition": _composition(), "components": [{
        "id": "message-1", "kind": "message", "category": "conversation", "role": "user",
        "content": {"content": [{"type": "text", "text": "Project rules <script>literal()</script> Settings"}]},
        "location": {"field": "messages", "index": 2, "json_pointer": "/messages/2"},
        "source_refs": [{"source_path": "session<script>.jsonl", "raw_index": 3}], "segments": [segment],
    }]}


def test_explorer_groups_segments_with_identity_version_location_and_literal_content():
    detail = _detail()
    original = copy.deepcopy(detail)
    rendered = render_component_explorer(detail)
    assert 'data-tv-i18n="Component explorer"' in rendered
    assert 'data-tv-i18n="Rules"' in rendered
    for value in ("rules:project", "sha256:version-one", "request-1:rules:0", "/content/0/text", "/messages/2"):
        assert value in rendered
    assert "Project rules &lt;script&gt;literal()&lt;/script&gt; Settings" in rendered
    assert "session&lt;script&gt;.jsonl" in rendered and "<script>" not in rendered
    assert 'data-tv-i18n="Settings"' not in rendered
    assert "Reconstructed" in rendered and "serialized_visible_content" in rendered
    assert detail == original
    assert rendered.index("Project rules &lt;script&gt;") < rendered.index('data-tv-i18n="Logical identity"')


def test_single_segment_snapshot_keeps_proven_native_rewrite_highlighting():
    detail = _detail()
    detail["scope"] = "native_prompt_state"
    detail["text_evolution"] = {"comparison_available": True, "evidence": "reconstructed", "blocks": [],
        "rewrites": [{"after_id": "message-1", "before_id": "old-message", "before_text": "Old rules",
                      "after_text": "Project rules <script>literal()</script> Settings"}]}
    rendered = render_component_explorer(detail)
    assert 'data-tv-i18n="Recorded rewrite"' in rendered
    assert "pc-text-added" in rendered and "<script>" not in rendered


def test_framing_segments_expose_allocation_without_repeating_source_text():
    detail = _detail()
    detail["components"][0]["segments"] = [dict(detail["components"][0]["segments"][0],
                                                kind="framing", category="unattributed")]
    rendered = render_component_explorer(detail)
    assert "Serialization framing contributes" in rendered
    assert "Project rules" not in rendered


def test_partial_and_bounded_history_are_disclosed_in_explorer():
    detail = _detail()
    detail["composition"]["partial"] = True
    detail["evidence_quality"] = {"content_complete": False, "missing_references": [
        {"kind": "message", "reference": "missing-blob", "reason": "not_recorded"}]}
    rendered = render_component_explorer(detail, {"complete": False, "computed_requests": 80, "total_requests": 110})
    assert "1 prompt reference could not be resolved" in rendered and "Unrecorded references: 1" in rendered
    assert "80 of 110" in rendered


def test_explorer_caps_component_segment_and_text_previews_without_mutation():
    detail = _detail()
    component = detail["components"][0]
    component["content"]["content"][0]["text"] = "<" * 20000
    component["segments"] *= 20
    detail["components"] = [dict(component, id=f"component-{index}") for index in range(100)]
    original = copy.deepcopy(detail)
    rendered = render_component_explorer(detail)
    assert "Showing" in rendered and "Explorer preview shortened" in rendered
    assert len(rendered) < 1_300_000
    assert detail == original


def test_chart_separates_recorded_occupancy_from_clickable_estimated_categories():
    rows = [{"id": "main-1", "order": 1, "agent_id": "main", "branch_id": "a", "context_id": "main-a",
             "observed_prompt_tokens": 900, "composition": _composition(12)},
            {"id": "child-1", "order": 2, "agent_id": "child<script>", "branch_id": "b", "context_id": "child-b",
             "observed_prompt_tokens": 80, "composition": None},
            {"id": "main-2", "order": 3, "agent_id": "main", "branch_id": "a", "context_id": "main-a",
             "observed_prompt_tokens": None, "composition": _composition(15, partial=True)}]
    chart = evolution_chart({"evolution": rows, "composition_history": {
        "complete": False, "computed_requests": 2, "total_requests": 3}}, "main-2")
    measured = [trace for trace in chart.data if trace.type == "scatter" and trace.mode == "lines+markers"]
    bars = [trace for trace in chart.data if trace.type == "bar"]
    assert [list(trace.y) for trace in measured] == [[900, None], [80]]
    assert list(bars[0].y) == [12, None, 15]
    assert bars[0].yaxis == "y2" and measured[0].yaxis == "y"
    assert list(chart.layout.yaxis3.categoryarray)[-1] == "Request sequence"
    assert chart.layout.barmode == "stack"
    assert chart.layout.xaxis.dtick == 1
    assert [row[0] for row in bars[0].customdata] == ["main-1", "child-1", "main-2"]
    assert "child&lt;script&gt; · b" in str(bars[0].customdata)
    assert any("2 of 3" in note.text for note in chart.layout.annotations)
    assert any("Partial composition" in note.text for note in chart.layout.annotations)
    localized = evolution_chart({"evolution": rows}, "main-1", "zh-CN")
    assert any(trace.name == "规则" for trace in localized.data)
    assert localized.layout.yaxis2.title.text == "分类 Token 估算"


def test_no_occupancy_is_created_when_only_estimated_composition_is_available():
    chart = evolution_chart({"evolution": [{"id": "r1", "order": 1, "composition": _composition()}]}, "r1")
    assert not any(trace.type == "scatter" and trace.mode == "lines+markers" for trace in chart.data)
    assert chart.layout.yaxis.title.text == "Estimated category tokens"
    assert len(CATEGORIES) == 10


def test_missing_all_components_is_an_unknown_composition_gap():
    missing = _composition(0, partial=True)
    missing["component_count"] = 0
    chart = evolution_chart({"evolution": [
        {"id": "known", "order": 1, "composition": _composition(12)},
        {"id": "missing", "order": 2, "composition": missing},
    ]}, "missing")
    bars = [trace for trace in chart.data if trace.type == "bar"]
    assert list(bars[0].y) == [12, None]
    rendered = render_component_explorer({"components": [], "composition": missing,
        "missing_references": ["blob:not-recorded"]})
    assert "No prompt components could be recovered" in rendered and "could not be resolved" in rendered


def test_segment_preview_keeps_recorded_change_paths_and_rejects_excessive_pointer_depth():
    detail = _detail()
    detail["text_evolution"] = {"comparison_available": True, "blocks": [{
        "component_id": "message-1", "path": "/content/0/text", "change": "added", "repeated": True,
        "repeat_count": 2, "repeat_source_component_ids": ["previous-component"],
    }]}
    rendered = render_component_explorer(detail)
    assert "pc-text-repeated" in rendered and "previous-component" in rendered
    assert 'data-tv-i18n="Role"' in rendered and 'data-tv-i18n="User"' in rendered
    detail["components"][0]["content"] = {"child": "Visible ancestor must not be returned"}
    detail["components"][0]["segments"][0]["json_pointer"] = "/child" * 65
    rendered = render_component_explorer(detail)
    assert "Visible ancestor" not in rendered


def test_semantic_titles_show_literal_text_and_one_based_source_record_before_raw_metadata():
    detail = _detail()
    detail["components"][0]["logical_id"] = "stable-parent"
    detail["components"][0]["segments"][0].update(logical_id=None, identity_basis="unavailable")
    rendered = render_component_explorer(detail)
    assert 'data-tv-i18n="User message"' in rendered
    assert "Message 1" not in rendered and "Segment <code>" not in rendered
    assert "record 4" in rendered and "session&lt;script&gt;.jsonl" in rendered
    assert rendered.index("record 4") < rendered.index('data-tv-i18n="Identity and provenance"')
    assert 'data-tv-i18n="Parent component identity"' in rendered and "stable-parent" in rendered
    assert "recorded position within the parent component" in rendered
    assert "/content/0/text" in rendered and "/messages/2" in rendered


def test_tool_title_uses_native_name_and_command_instead_of_array_position():
    detail = _detail()
    component = detail["components"][0]
    component["content"] = {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "rg --files"}}]}
    component["segments"][0]["json_pointer"] = "/content/0"
    rendered = render_component_explorer(detail)
    assert 'data-tv-i18n="Tool call"' in rendered and "<code>Bash</code>" in rendered
    assert "rg --files" in rendered and "Segment <code>" not in rendered


def test_recovered_content_with_unsupported_settings_never_claims_missing_messages():
    detail = _detail()
    detail["composition"]["partial"] = True
    detail["evidence_quality"] = {"content_complete": True, "missing_references": [],
                                  "unsupported_settings": ["cacheControl", "requestDigest"]}
    rendered = render_component_explorer(detail)
    assert "Recorded components recovered" in rendered
    assert "Unsupported request settings" in rendered and "cacheControl" in rendered
    assert "missing prompt evidence" not in rendered and "could not be resolved" not in rendered
    detail["composition"]["categories"][0].update(estimated_tokens=251204, component_count=42)
    rendered = render_component_explorer(detail)
    assert "251.2k" in rendered and "42 components" in rendered


def test_edit_call_title_uses_literal_file_path_when_command_is_absent():
    detail = _detail()
    component = detail["components"][0]
    component["content"] = {"content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "src/auth.py"}}]}
    component["segments"][0]["json_pointer"] = "/content/0"
    rendered = render_component_explorer(detail)
    assert "<code>Edit</code>" in rendered and "src/auth.py</span>" in rendered


def test_truncated_reference_catalog_reports_lower_bound_without_counting_marker():
    from agent_trajviz.insight.ui.prompt_composition import quality_note

    detail = {"evidence_quality": {"content_complete": False, "missing_references": [
        *[{"kind": "message", "reference": f"missing-{index}", "reason": "not_recorded"} for index in range(3)],
        {"kind": "catalog", "reference": "truncated", "reason": "inspection_limit"},
    ]}}
    rendered = quality_note(detail)
    assert "At least 3 prompt references could not be resolved" in rendered
    assert "4 prompt references" not in rendered
    assert "Known unrecorded references: 3" in rendered
    assert "Additional references were omitted after the inspection limit" in rendered


def test_prefix_inspection_limit_keeps_unknown_prefix_count_separate_from_known_gaps():
    from agent_trajviz.insight.ui.prompt_composition import quality_note

    issues = [{"kind": "prefix", "reference": "inspection-limit", "reason": "inspection_limit"}]
    rendered = quality_note({"evidence_quality": {"missing_references": issues}})
    assert "unresolved reference count is unknown" in rendered
    assert "1 prompt reference" not in rendered
    issues.insert(0, {"kind": "message", "reference": "absent", "reason": "not_recorded"})
    rendered = quality_note({"evidence_quality": {"missing_references": issues}})
    assert "At least 1 prompt reference could not be resolved" in rendered
    assert "2 prompt references" not in rendered and "unresolved reference count is unknown" in rendered


def test_legacy_reference_limit_markers_are_excluded_from_known_reference_count():
    from agent_trajviz.insight.ui.prompt_composition import quality_note

    rendered = quality_note({"missing_references": ["blob:absent", "references:truncated", "prefix:inspection-limit"]})
    assert "At least 1 prompt reference could not be resolved" in rendered
    assert "3 prompt references" not in rendered
    assert "Additional references were omitted" in rendered and "unresolved reference count is unknown" in rendered


def test_chart_interaction_revision_tracks_catalog_session_and_panel_shape():
    data = {"evolution": [
        {"id": "request-1", "order": 1, "session_id": "session-one", "composition": _composition()},
        {"id": "request-2", "order": 2, "session_id": "session-one", "composition": _composition()},
    ]}
    first = evolution_chart(data, "request-1").layout.uirevision
    assert evolution_chart(data, "request-2").layout.uirevision == first
    # Bounds of the computed history can move with selection; that does not
    # change the catalog or the meaning of its composition subplot.
    sparse = copy.deepcopy(data)
    sparse["evolution"][0]["composition"] = None
    assert evolution_chart(sparse, "request-2").layout.uirevision == first
    other_session = copy.deepcopy(data)
    for row in other_session["evolution"]:
        row["session_id"] = "session-two"
    assert evolution_chart(other_session, "request-1").layout.uirevision != first
    shorter = {"evolution": data["evolution"][:1]}
    assert evolution_chart(shorter, "request-1").layout.uirevision != first
    measured = copy.deepcopy(data)
    measured["evolution"][0]["observed_prompt_tokens"] = 100
    assert evolution_chart(measured, "request-1").layout.uirevision != first
    no_composition = copy.deepcopy(data)
    for row in no_composition["evolution"]:
        row["composition"] = None
    assert evolution_chart(no_composition, "request-1").layout.uirevision != first
