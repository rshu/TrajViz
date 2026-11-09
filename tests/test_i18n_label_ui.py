"""Phase-summary labels localize without rewriting the uploaded sidecar."""

from copy import deepcopy
from html.parser import HTMLParser
import json

from agent_trajviz.i18n import translate
from agent_trajviz.i18n.label_names import label_key
from agent_trajviz.insight.labels import aggregate_labels
from agent_trajviz.insight.palette import LABEL_PHASE_COLORS
from agent_trajviz.insight.presenters.label_ui import build_label_ui_payload


class _Projection(HTMLParser):
    def __init__(self, source, language="zh-CN"):
        super().__init__(convert_charrefs=True)
        self.language = language
        self.stack = []
        self.markers = []
        self.attributes = []
        self.text = []
        self.unmarked = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        owned_text = "data-tv-i18n" in attrs and "data-tv-i18n-attr" not in attrs
        if "data-tv-i18n" in attrs:
            key = attrs["data-tv-i18n"]
            args = json.loads(attrs.get("data-tv-i18n-args", "{}"))
            attribute = attrs.get("data-tv-i18n-attr")
            self.markers.append((key, args, attribute))
            if attribute:
                attrs[attribute] = translate(key, self.language, **args)
            elif not any(owned for _, owned in self.stack):
                self.text.append(translate(key, self.language, **args))
        self.attributes.append(attrs)
        if tag not in {"input", "hr", "br", "img", "meta", "link"}:
            self.stack.append((tag, owned_text))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not any(owned for _, owned in self.stack):
            self.text.append(data)
            self.unmarked.append(data)

    @property
    def visible(self):
        return "".join(self.text)


def _sidecar(tmp_path):
    phases = ["understand", "plan", "implement", "debug", "validate", "report"]
    actions = ["code_reading", "task_breakdown", "implement_runtime_logic", "debug_root_cause",
               "validation_unit_tests", "final_reporting"]
    records = [{"index": index, "role": "assistant", "phase": phase, "action": action, "duration_s": 1.5}
               for index, (phase, action) in enumerate(zip(phases, actions, strict=True))]
    value = {"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1",
             "model": "implement", "source_note": "understand", "steps": records}
    path = tmp_path / "session_labeled.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_phase_chips_and_count_tooltips_translate_in_both_languages_without_data_changes(tmp_path):
    path, data = _sidecar(tmp_path)
    original = deepcopy(data)
    original_bytes = path.read_bytes()
    output = build_label_ui_payload(str(path))
    chinese = _Projection(output["badge_html"])
    english = _Projection(output["badge_html"], language="en")
    titles = [attrs["title"] for attrs in chinese.attributes if "title" in attrs]
    for record in data["steps"]:
        phase = record["phase"]
        key = label_key("phase", phase)
        caption = translate(key, "zh-CN")
        assert caption != phase and caption in chinese.visible
        assert phase not in chinese.visible
        assert f"{caption}：1" in titles
        assert f"{phase}: 1" in english.visible
        assert (key, {}, None) in chinese.markers
    # Counts are formatted arguments; canonical phase IDs are never translated
    # as if they were a free-form data parameter in a surrounding message.
    tooltip_markers = [(key, args) for key, args, attr in chinese.markers if attr == "title"]
    assert len(tooltip_markers) == 6 and all(args == {"count": 1} for _, args in tooltip_markers)
    assert data == original and path.read_bytes() == original_bytes
    aggregated = aggregate_labels(deepcopy(data))
    assert set(aggregated["phase_counts"]) == {record["phase"] for record in data["steps"]}
    assert set(aggregated["action_counts"]) == {record["action"] for record in data["steps"]}


def test_custom_phase_and_recorded_provenance_remain_literal_and_escaped(tmp_path):
    path, data = _sidecar(tmp_path)
    custom = 'Step Count by Phase <img src=x onerror="run()">'
    data["steps"].append({"index": 9, "role": "assistant", "phase": custom,
                          "action": "custom_action", "duration_s": 2})
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    output = build_label_ui_payload(str(path))
    chinese = _Projection(output["badge_html"])
    provenance = _Projection(output["status_html"])
    assert f"{custom}: 1" in chinese.visible
    assert any(attrs.get("title") == f"{custom}: 1" and "data-tv-i18n" not in attrs
               for attrs in chinese.attributes)
    assert all(key != custom for key, _, _ in chinese.markers)
    assert "&lt;img" in output["badge_html"] and '<img' not in output["badge_html"]
    # A model/source note equal to a builtin taxonomy ID is still provenance.
    assert "implement" in provenance.unmarked and "understand" in provenance.unmarked
    assert path.read_bytes() == before


def _contrast(foreground, background):
    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4
                  for channel in channels]
        return sum(channel * weight for channel, weight in zip(linear, (.2126, .7152, .0722), strict=True))

    first, second = sorted((luminance(foreground), luminance(background)))
    return (second + .05) / (first + .05)


def _styles(attributes):
    return [{key.strip(): value.strip() for key, value in
             (item.split(":", 1) for item in attrs.get("style", "").split(";") if ":" in item)}
            for attrs in attributes]


def test_rendered_phase_and_coverage_badges_meet_text_contrast_in_both_languages(tmp_path):
    path, data = _sidecar(tmp_path)
    data["steps"].append({"index": 9, "role": "assistant", "phase": "custom-phase",
                          "action": "custom-action", "duration_s": 1})
    path.write_text(json.dumps(data), encoding="utf-8")
    output = build_label_ui_payload(str(path))
    snapshots = []
    for language in ("en", "zh-CN"):
        parsed = _Projection(output["badge_html"], language=language)
        styles = _styles(parsed.attributes)
        chips = [style for style in styles if style.get("border-radius") == "10px"]
        assert [style["background"] for style in chips] == list(LABEL_PHASE_COLORS.values()) + ["#6b7280"]
        assert all(_contrast(style["color"], style["background"]) >= 4.5 for style in chips)
        coverage = [style for style in styles if style.get("padding") == "4px 12px"]
        assert len(coverage) == 1 and coverage[0]["background"] == "#059669"
        assert _contrast(coverage[0]["color"], coverage[0]["background"]) >= 4.5
        assert any(style.get("text-decoration") == "underline" and style.get("color") == "#047857"
                   for style in styles)
        snapshots.append((chips, coverage))
    assert snapshots[0] == snapshots[1]
