"""Readable prompt previews are bounded, literal and color-independent."""

from __future__ import annotations

import copy
from html.parser import HTMLParser

import pytest

from agent_trajviz.insight.ui import prompt_text


def block(text, *, change="added", identity="message-1", path="/content/0/text", **extra):
    return {"component_id": identity, "path": path, "kind": "text", "role": "user", "text": text,
            "text_truncated": False, "source_refs": [], "change": change,
            "repeated": False, "repeat_count": 0, "repeat_source_component_ids": [], **extra}


def detail(*blocks, rewrites=(), **extra):
    return {"scope": "native_prompt_state", "text_evolution": {
        "comparison_available": True, "evidence": "reconstructed", "blocks": list(blocks),
        "rewrites": list(rewrites), "blocks_truncated": False, **extra,
    }}


@pytest.fixture(autouse=True)
def readable_blocks(monkeypatch):
    # Test renderer behavior independently of the separately tested extractor.
    def read(component):
        if "_blocks" in component:
            return component["_blocks"]
        content = component.get("content")
        return [block(content, identity=component.get("id", ""))] if isinstance(content, str) else []

    monkeypatch.setattr(prompt_text, "_readable", read)


class HTML(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_legend_has_labeled_symbols_and_distinct_semantic_colors():
    rendered = prompt_text.legend()
    for status, label in (("added", "Added text"), ("removed", "Removed text"),
                          ("retained", "Retained text"), ("uncompared", "Uncompared text"),
                          ("repeated", "Repeated content")):
        assert f"pc-text-{status}" in rendered and f'data-tv-i18n="{label}"' in rendered
    assert "aria-hidden='true'" in rendered
    assert all(color in prompt_text.CSS for color in ("#14532d", "#7f1d1d", "#334155", "#78350f"))


def test_changed_cards_show_actual_commands_and_escape_all_recorded_fields():
    data = detail(block("pytest -q\nprint('<script>bad()</script>')", kind="tool_arguments",
                        tool_name='<img src=x onerror="bad()">',
                        source_refs=[{"source_path": '<script>path</script>', "raw_index": 4}]),
                  block("Failed: <img src=x onerror=bad()>", change="removed", kind="tool_result"))
    rendered = prompt_text.render_text_evolution(data)
    assert "pytest -q" in rendered and "pc-text-added" in rendered and "pc-text-removed" in rendered
    assert "&lt;script&gt;" in rendered and "&lt;img" in rendered
    assert not any(tag in {"script", "img"} for tag, _ in HTML(rendered).tags)
    assert rendered.count(" open>") == 2
    assert 'data-tv-i18n="Tool arguments"' in rendered


def test_repetition_uses_amber_label_count_and_literal_source_ids():
    data = detail(block("same output", repeated=True, repeat_count=3,
                        repeat_source_component_ids=['previous<script>', "component-two"]))
    rendered = prompt_text.render_text_evolution(data)
    assert "pc-text-repeated" in rendered and "pc-text-repeat-badge" in rendered
    assert "Matches 3 earlier recorded blocks." in rendered
    assert "previous&lt;script&gt;" in rendered
    assert "do not establish unnecessary work" in rendered
    assert "in this prompt" not in rendered


def test_proven_rewrite_has_inline_before_after_diff_and_keeps_cjk_literal():
    rewrite = {"before_id": "before", "after_id": "after", "label": "base<script>",
               "before_text": "检查 old_value 并运行 tests", "after_text": "检查 new_value 并运行 tests",
               "before_truncated": False, "after_truncated": False, "source_refs": []}
    rendered = prompt_text.render_text_evolution(detail(rewrites=[rewrite]))
    assert "<del class='pc-text-removed'>old_value</del>" in rendered
    assert "<ins class='pc-text-added'>new_value</ins>" in rendered
    assert "pc-text-retained" in rendered and "检查" in rendered
    assert 'data-tv-i18n="Before rewrite"' in rendered and 'data-tv-i18n="After rewrite"' in rendered
    assert "base&lt;script&gt;" in rendered
    snapshot = prompt_text.render_component_text({"id": "after", "content": rewrite["after_text"]},
                                                 detail(rewrites=[rewrite]))
    assert "<ins class='pc-text-added'>new_value</ins>" in snapshot
    assert "old_value" not in snapshot


def test_unrelated_or_estimated_blocks_never_become_inferred_rewrites():
    changes = detail(block("old command", change="removed"), block("new command"))
    rendered = prompt_text.render_text_evolution(changes)
    assert "Recorded rewrite" not in rendered and "<ins" not in rendered and "<del" not in rendered
    fake = {"before_text": "old", "after_text": "new"}
    assert "Recorded rewrite" not in prompt_text.render_text_evolution(detail(rewrites=[fake], evidence="estimated"))
    unavailable = detail(rewrites=[fake], comparison_available=False)
    assert "Recorded rewrite" not in prompt_text.render_text_evolution(unavailable)


def test_snapshot_uses_added_repeated_retained_and_uncompared_states():
    component = {"id": "message-1", "content": "Settings remains literal"}
    rendered = prompt_text.render_component_text(component, detail(block(component["content"])))
    assert "pc-text-added" in rendered and 'data-tv-i18n="Settings"' not in rendered
    assert "pc-text-repeated" in prompt_text.render_component_text(component, detail(block("x", repeated=True)))
    assert "pc-text-retained" in prompt_text.render_component_text(component, detail())
    assert "pc-text-uncompared" in prompt_text.render_component_text(component, {})
    assert "pc-text-uncompared" in prompt_text.render_component_text(component, detail(blocks_truncated=True))


def test_old_details_have_plain_readable_fallback_without_rewrite_inference():
    data = {"changes": {"available": True, "added_components": [{"id": "a", "content": "read parser.py"}],
                         "removed_components": [{"id": "b", "content": "read old.py"}]}}
    rendered = prompt_text.render_text_evolution(data)
    assert "read parser.py" in rendered and "read old.py" in rendered
    assert "Recorded rewrite" not in rendered
    assert "plain old snapshot" in prompt_text.render_component_text({"id": "old", "content": "plain old snapshot"}, {})


def test_legacy_explicit_rewrite_keeps_escaped_plain_before_and_after():
    data = {"changes": {"available": True, "rewritten_components": [
        {"identity": "message:old", "before": "long <script>output</script>", "after": "short result"},
    ]}}
    rendered = prompt_text.render_text_evolution(data)
    assert "long &lt;script&gt;output&lt;/script&gt;" in rendered
    assert "short result" in rendered and "Before rewrite" in rendered and "After rewrite" in rendered
    assert "<script>" not in rendered


def test_core_field_and_workflow_source_refs_are_readable():
    rendered = prompt_text.render_text_evolution(detail(block("hi", source_refs=[
        {"request_id": "request-a", "field": "messages", "index": 43, "step_index": 9},
        {"request_id": "request-only<script>"},
    ])))
    assert "messages[43] · #9" in rendered and "request-only&lt;script&gt;" in rendered


def test_definition_and_instruction_text_have_specific_titles():
    rendered = prompt_text.render_text_evolution(detail(block("Instructions", role="system"),
                                                        block("Tool schema", tool_name="Read")))
    assert 'data-tv-i18n="System instructions"' in rendered
    assert 'data-tv-i18n="Tool definition"' in rendered


def test_omitted_small_snapshot_blocks_show_shortened_note():
    component = {"id": "many", "_blocks": [block(str(i), identity="many", path=f"/{i}") for i in range(30)]}
    rendered = prompt_text.render_component_text(component, detail())
    assert rendered.count("<pre") == prompt_text._MAX_COMPONENT_BLOCKS
    assert "Preview shortened" in rendered


def test_large_changes_and_snapshots_are_bounded_without_mutating_originals():
    blocks = [block("<" * 20000, identity=f"message-{i}") for i in range(100)]
    data = detail(*blocks)
    original = copy.deepcopy(data)
    rendered = prompt_text.render_text_evolution(data)
    assert len(rendered) < 250000 and "Showing" in rendered and "Preview shortened" in rendered
    assert rendered.count("<details") <= prompt_text._MAX_CHANGE_CARDS
    assert data == original and len(data["text_evolution"]["blocks"][0]["text"]) == 20000
    component = {"id": "message-1", "_blocks": blocks}
    snapshot = prompt_text.render_component_text(component, data)
    assert len(snapshot) < 45000 and "Preview shortened" in snapshot


def test_large_rewrite_skips_quadratic_sequence_matching(monkeypatch):
    class Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Diff matcher called past budget")

    monkeypatch.setattr(prompt_text, "SequenceMatcher", Forbidden)
    rewrite = {"before_id": "before", "after_id": "after", "before_text": "旧 " * 3000, "after_text": "新 " * 3000}
    rendered = prompt_text.render_text_evolution(detail(rewrites=[rewrite]))
    assert "Inline diff exceeds the preview budget" in rendered and "Preview shortened" in rendered
    assert "<ins" not in rendered and "<del" not in rendered and "pc-text-uncompared" in rendered
    assert len(rendered) < 60000


def test_limited_rewrite_does_not_mark_identical_prefix_as_added_or_removed():
    prefix = "same " * 500
    rewrite = {"before_id": "before", "after_id": "after",
               "before_text": prefix + "old", "after_text": prefix + "new"}
    rendered = prompt_text.render_text_evolution(detail(rewrites=[rewrite]))
    assert "Inline diff exceeds the preview budget" in rendered
    assert prefix + "old" in rendered and prefix + "new" in rendered
    assert "<ins" not in rendered and "<del" not in rendered


def test_recorded_contents_never_receive_translation_markers():
    rendered = prompt_text.render_text_evolution(detail(block("Added text\nSettings\n简体中文")))
    before, body = rendered.split("<pre class='pc-text-content'>", 1)
    body = body.split("</pre>", 1)[0]
    assert "data-tv-i18n" in before and "data-tv-i18n" not in body
    assert body == "Added text\nSettings\n简体中文"


def test_empty_and_malformed_preview_inputs_do_not_crash():
    assert prompt_text.render_text_evolution(None) == ""
    assert "No changed prompt text" in prompt_text.render_text_evolution({})
    assert prompt_text.render_component_text(None, {}) == ""
    assert "No readable text" in prompt_text.render_component_text({"id": "empty", "content": {}}, {})
    data = detail(block("hi", kind=[], tool_name={}))
    assert "hi" in prompt_text.render_text_evolution(data)
