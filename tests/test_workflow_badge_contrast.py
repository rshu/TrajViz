"""Rendered Workflow labels retain readable contrast in both theme palettes."""

from html.parser import HTMLParser
import re

import pytest

from agent_trajviz.insight.rendering import _format_step_header, render_toc_sidebar, render_workflow_html
from agent_trajviz.insight.styles import APP_CSS, _DARK_TOKENS, _LIGHT_TOKENS


def _declarations(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return {key.strip(): value.strip() for key, value in re.findall(r"([^;{}]+):([^;{}]+);?", source)}


def _color(value, tokens, under):
    while value.startswith("var("):
        value = tokens[value[4:-1]]
    if value in {"inherit", "transparent"}:
        return under
    if value.startswith("#"):
        return tuple(int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))
    if value == "white":
        return (1, 1, 1)
    values = [float(part) for part in re.fullmatch(r"rgba\(([^)]+)\)", value)[1].split(",")]
    return tuple(values[index] / 255 * values[3] + under[index] * (1 - values[3]) for index in range(3))


def _luminance(rgb):
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in rgb]
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _contrast(first, second):
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class _Labels(HTMLParser):
    """Resolve the rendered inline colors against real CSS theme tokens.

    Badge text is small, so every sampled foreground/background needs the
    normal-text contrast threshold. Alpha card fills are composited over the
    actual light/dark surface, rather than assumed to be opaque. Dashboard
    mode also models Gradio's prose rule explicitly coloring each descendant.
    """

    def __init__(self, source, tokens, surface, *, prose_reset=False):
        super().__init__()
        self.tokens = tokens
        background = _color(tokens[surface], tokens, (1, 1, 1))
        foreground = _color(tokens["--ov-text"], tokens, background)
        self.prose_color = _color(tokens["--ov-body-text"], tokens, background) if prose_reset else None
        self.stack = [("body", foreground, background, False)]
        self.samples = []
        self.detail_badge = _declarations(re.search(r"\.dp-header \.dp-badge\s*\{([^}]+)}", APP_CSS)[1])
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        _, inherited_foreground, background, tracked = self.stack[-1]
        # The prose reset applies a specified color to this element; CSS
        # `inherit` must still resolve its parent's computed color, before
        # that reset, rather than inheriting the reset value from itself.
        foreground = self.prose_color if self.prose_color is not None else inherited_foreground
        classes = set(attrs.get("class", "").split())
        style = dict(self.detail_badge) if "dp-badge" in classes else {}
        style.update(_declarations(attrs.get("style", "")))
        if "background" in style:
            background = _color(style["background"], self.tokens, background)
        if "color" in style:
            foreground = _color(style["color"], self.tokens, inherited_foreground)
        tracked = tracked or bool(classes & {"wf-badge", "dp-header", "wf-model-error"})
        if tag not in {"br", "hr", "input", "img", "meta", "link"}:
            self.stack.append((tag, foreground, background, tracked))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        tag, foreground, background, tracked = self.stack[-1]
        if data.strip() and tracked and tag != "style":
            self.samples.append((data.strip(), _contrast(foreground, background)))


def _step(**overrides):
    step = {
        "index": 1, "role": "assistant", "parts": [], "tokens": {"total": 100}, "duration": None,
        "text_preview": "recorded evidence", "tool_calls": [], "tool_call_count": 0, "error_count": 0,
        "has_reasoning": False,
    }
    step.update(overrides)
    return step


_CASES = [
    {"role": role} for role in ("user", "assistant", "task", "system", "compaction", "tool", "developer")
] + [
    {"tool_call_count": 1},
    {"has_reasoning": True},
    {"finish": "stop"},
    {"finish": "error", "error_count": 1},
    {"error_count": 1},
    {"is_sub_agent": True},
    {"role": "user", "is_sub_agent": True},
]


@pytest.mark.parametrize("palette", [_LIGHT_TOKENS, _DARK_TOKENS], ids=["light", "dark"])
@pytest.mark.parametrize("surface", ["--ov-bg", "--ov-card"])
@pytest.mark.parametrize("prose_reset", [False, True], ids=["standalone-report", "dashboard-prose-reset"])
@pytest.mark.parametrize("changes", _CASES)
def test_rendered_role_category_step_and_detail_badges_have_normal_text_contrast(palette, surface, prose_reset, changes):
    tokens = _declarations(palette)
    step = _step(**changes)
    for rendered in (render_workflow_html([step]), render_toc_sidebar([step]), _format_step_header(step)):
        labels = _Labels(rendered, tokens, surface, prose_reset=prose_reset)
        assert labels.samples
        for caption, ratio in labels.samples:
            assert ratio >= 4.5, f"{caption!r} has contrast {ratio:.2f}:1 for {changes} on {surface}"


def test_prose_reset_model_detects_original_unstyled_translation_wrapper_regression():
    rendered = render_workflow_html([_step(role="user")])
    # Reproduce the old markup by removing style only from owned text wrappers.
    old_markup = re.sub(r'(<span\b(?=[^>]*\bdata-tv-i18n=)[^>]*?)\sstyle="[^"]*"', r"\1", rendered)
    tokens = _declarations(_LIGHT_TOKENS)
    standalone = _Labels(old_markup, tokens, "--ov-card")
    dashboard = _Labels(old_markup, tokens, "--ov-card", prose_reset=True)
    assert all(ratio >= 4.5 for _, ratio in standalone.samples)
    assert any(caption == "User" and ratio < 4.5 for caption, ratio in dashboard.samples)
