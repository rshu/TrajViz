"""Timeline measurements stay distinct from recorded prompt-change events."""

from agent_trajviz.insight.ui.prompt_evolution import evolution_chart
from agent_trajviz.insight.ui.prompt_tab import _changes


def _data(tokens):
    return {"evolution": [{"id": f"r{i}", "order": i + 1, "session_id": "s", "agent_id": "",
                           "model": "test", "observed_prompt_tokens": value,
                           "prompt_events": ([{"kind": "compaction", "count": 1}] if i == 2
                                             else [{"kind": "system_section_updated", "target_id": "rules"}])}
                          for i, value in enumerate(tokens)]}


def test_prompt_evolution_preserves_measured_growth_and_compaction_events():
    chart = evolution_chart(_data([100, 500, 200]), "r2")
    assert list(chart.data[0].y) == [100, 500, 200]
    assert any("Context compacted" in str(trace.customdata) for trace in chart.data)
    assert all(row[0].startswith("r") for trace in chart.data for row in trace.customdata)


def test_missing_sizes_have_event_markers_without_a_fake_zero_token_series():
    chart = evolution_chart(_data([None, None]), "r0", "zh-CN")
    assert all(all(isinstance(value, str) for value in trace.y) for trace in chart.data)
    assert "工作流响应边界" in chart.layout.xaxis.title.text


def test_transcript_only_chart_does_not_suggest_unrecorded_instruction_events():
    data = _data([100, 200])
    for row in data["evolution"]:
        row["prompt_events"] = []
    chart = evolution_chart(data, "r1")
    assert list(chart.layout.yaxis2.categoryarray) == ["Request sequence"]


def test_prompt_growth_lines_do_not_mix_child_ownership_or_distinct_branches():
    data = _data([100, 900, 150, 300])
    for row, context, branch in zip(data["evolution"], ("main", "unknown", "main", "branch-b"),
                                    ("a", "a", "a", "b"), strict=True):
        row.update(context_id=context, branch_id=branch)
    data["evolution"][1]["actor_kind"] = "unknown_child"
    chart = evolution_chart(data, "r3")
    measured = [trace for trace in chart.data if trace.mode == "lines+markers"]
    assert [list(trace.y) for trace in measured] == [[100, 150], [900], [300]]
    assert "Unidentified subagent" in measured[1].name


def test_proven_rewrites_expose_escaped_before_and_after_but_not_invented_lineage():
    detail = {"prompt_events": [{"kind": "context_edit", "target_id": "earlier"}],
              "changes": {"available": True, "previous_request_id": "before", "added": [], "removed": [],
                          "retained": 0, "rewritten_components": [{"identity": "message:earlier", "label": "earlier",
                            "before": "long <script>output</script>", "after": "short result"}]}}
    rendered = _changes(detail)
    assert "Before" in rendered and "After" in rendered and "Rewritten components" in rendered
    assert "long &lt;script&gt;" in rendered and "long <script>" not in rendered
    assert "short result" in rendered and "Context edited" in rendered


def test_generated_message_role_labels_are_translatable_outside_literal_code():
    from agent_trajviz.i18n import translate
    from agent_trajviz.insight.ui.prompt_tab import _component_label_html

    label = _component_label_html({"kind": "message", "role": "toolResult", "label": "toolResult 3"})
    assert 'data-tv-i18n="Tool result"' in label and "<code>" not in label
    rewritten = _component_label_html({"kind": "message", "label": "toolResult 3",
                                       "after": {"role": "toolResult", "content": "literal User"}})
    assert rewritten == label
    literal = _component_label_html({"kind": "tool", "label": "User <script>"})
    assert literal == "<code>User &lt;script&gt;</code>"
    assert translate("Before rewrite", "zh-CN") == "改写前"
