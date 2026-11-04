"""Bundled taxonomy coverage and safe fallback for custom label identities."""

import html
from html.parser import HTMLParser

import pytest

from agent_trajviz.i18n import translate, validate_catalog
from agent_trajviz.i18n.catalog_labels import ACTION_LABELS, PHASE_LABELS
from agent_trajviz.i18n.label_names import label_attrs, label_key, label_spec, label_text
from agent_trajviz.labeling.taxonomy import DEFAULT_TAXONOMY, build_valid_sets, load_taxonomy


def test_display_names_cover_entire_bundled_taxonomy_without_changing_english_ids():
    mapping, _ = load_taxonomy(DEFAULT_TAXONOMY)
    phases, actions, _ = build_valid_sets(mapping)
    assert set(PHASE_LABELS) == phases
    assert set(ACTION_LABELS) == actions
    validate_catalog()
    for kind, values in (("phase", phases), ("action", actions)):
        for value in values:
            assert label_key(kind, value) == value
            assert label_spec(kind, value) == value
            assert translate(value, "en") == value
            assert translate(value, "zh-CN") != value
            assert translate(value + ": {count}", "zh-CN", count=7).endswith("：7")


@pytest.mark.parametrize("kind,value", [("phase", "Reference"), ("action", "Tool Calls"),
                                       ("phase", '<img src=x onerror="run()">'),
                                       ("action", "my_custom_action")])
def test_custom_names_remain_literal_and_html_safe(kind, value):
    assert label_key(kind, value) is None
    assert label_spec(kind, value) == {"literal": html.escape(value)}
    assert label_text(kind, value) == html.escape(value)
    tooltip = label_attrs(kind, value, count=3)
    assert "data-tv-i18n" not in tooltip
    assert tooltip == 'title="' + html.escape(value + ": 3", quote=True) + '"'


def test_reserved_labels_and_tooltip_counts_have_explicit_owned_names():
    assert label_key("phase", "user") == "user"
    assert label_key("action", "user_prompt") == "user_prompt"
    assert label_key("action", "unknown") == "unknown"
    parsed = HTMLParser()
    parsed.feed("<span " + label_attrs("phase", "understand", count=0) + "></span>")
    assert 'data-tv-i18n="understand: {count}"' in parsed.get_starttag_text()
    with pytest.raises(ValueError):
        label_attrs("phase", "raw", attribute="onclick")
    with pytest.raises(ValueError):
        label_key("recorded-name", "understand")
