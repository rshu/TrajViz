"""Run-group presentation localizes without translating recorded identities."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path

from agent_trajviz.i18n import translate, validate_catalog
from agent_trajviz.insight.run_group import (
    _ACTION_MATRIX_PREVIEW,
    _PATTERN_LABELS,
    _RUN_GROUP_MESSAGES,
    _SKILL_MATRIX_PREVIEW,
    _TOOL_MATRIX_PREVIEW,
    LoadedRun,
    _run_label_html,
    _run_label_message,
    _run_swatch_html,
    build_run_group_behavior_html,
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    load_run,
)


class _Presentation(HTMLParser):
    """Project only explicit text markers, as the browser translator does."""

    def __init__(self, source, language="zh-CN"):
        super().__init__(convert_charrefs=True)
        self.language = language
        self.stack = []
        self.markers = []
        self.attributes = []
        self.text = []
        self.unmarked = []
        self.code = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        owned = "data-tv-i18n" in attrs and "data-tv-i18n-attr" not in attrs
        if "data-tv-i18n" in attrs:
            key = attrs["data-tv-i18n"]
            args = json.loads(attrs.get("data-tv-i18n-args", "{}"))
            self.markers.append((key, args, attrs.get("data-tv-i18n-attr")))
            if "data-tv-i18n-attr" in attrs:
                attrs[attrs["data-tv-i18n-attr"]] = translate(key, self.language, **args)
            if owned and not any(marked for _, marked in self.stack):
                self.text.append(translate(key, self.language, **args))
        self.attributes.append(attrs)
        if tag not in {"input", "hr", "br", "img", "meta", "link"}:
            self.stack.append((tag, owned))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not any(marked for _, marked in self.stack):
            self.text.append(data)
            self.unmarked.append(data)
            if any(tag == "code" for tag, _ in self.stack):
                self.code.append(data)

    @property
    def visible(self):
        return "".join(self.text)


def _behavior():
    ids = ["run-one", "run-two", "run-three"]

    def count_row(key, kind="consensus", readers=None):
        readers = ids if readers is None else readers
        return {"key": key, "kind": kind, "n_runs": len(readers),
                "cells": {rid: {"present": rid in readers, "count": 2} for rid in ids}}

    def file_row(path, readers):
        return {"path": path, "kind": "consensus" if len(readers) == 3 else "unique" if len(readers) == 1 else "partial",
                "n_runs": len(readers),
                "cells": {rid: {"read": int(rid in readers), "write": int(rid == ids[0])} for rid in ids}}

    return {
        "baseline_run_id": ids[0], "run_ids": ids,
        "labels": {ids[0]: f"{ids[0]} (baseline)", ids[1]: "second", ids[2]: "third"},
        "consensus_threshold": 2,
        "similarity": {rid: {other: 1.0 if rid == other else .22 for other in ids} for rid in ids},
        "tool_matrix": [count_row("read"), count_row("execute", "partial", ids[:2]),
                        count_row("edit", "unique", ids[1:2])],
        "skill_matrix": [count_row("test-skill")],
        "action_matrix": [{"type": "COMMAND", "target": "pytest", "kind": "consensus", "n_runs": 3,
                           "cells": {rid: {"present": True, "count": 2} for rid in ids}}],
        "file_matrix": [file_row("src/common.py", ids), file_row("src/first.py", ids[:1]),
                        file_row("src/other.py", ids[1:]), file_row("README.md", ids[1:2])],
        "patterns_vs_baseline": {ids[1]: [{"type": "redundant_search", "label": "redundant search", "count": 3}]},
    }


def test_screenshot_sections_badges_and_dynamic_guidance_have_chinese_projection():
    validate_catalog()
    behavior = _behavior()
    original = deepcopy(behavior)
    output = build_run_group_behavior_html({"behavior": behavior, "ok": True})
    parsed = _Presentation(output)
    for expected in ("行为对比", "行为相似度（对齐 F1）", "工具覆盖情况", "技能覆盖情况",
                     "操作覆盖情况", "文件覆盖情况", "覆盖情况", "共有", "部分共有", "独有",
                     "（基线）", "重复搜索", "读取分布"):
        assert expected in parsed.visible
    for english in ("Behavioral comparison", "Tool coverage", "Skill coverage", "Action coverage",
                    "File coverage", "shared", "partial", "unique", "(baseline)", " only"):
        assert english not in parsed.visible
    assert "0.22" in parsed.visible and "3/3" in parsed.visible and "pytest" in parsed.visible
    assert behavior == original
    for key, args, _ in parsed.markers:
        assert translate(key, "zh-CN", **args) != key.format(**args)


def test_expansion_counts_and_three_run_file_summaries_translate_without_english_fragments():
    behavior = _behavior()
    for matrix, preview in (("tool_matrix", _TOOL_MATRIX_PREVIEW), ("skill_matrix", _SKILL_MATRIX_PREVIEW),
                            ("action_matrix", _ACTION_MATRIX_PREVIEW)):
        base = deepcopy(behavior[matrix][0])
        behavior[matrix] = [dict(base, key=f"raw-item-{i}", target=f"raw-target-{i}") for i in range(preview + 7)]
    # Exercise the collapsed mutual-read file preview and its remaining count.
    shared = deepcopy(behavior["file_matrix"][0])
    behavior["file_matrix"].extend(dict(shared, path=f"src/common-{i}.py") for i in range(25))
    parsed = _Presentation(build_run_group_behavior_html({"behavior": behavior}))
    for preview, noun in ((_TOOL_MATRIX_PREVIEW, "工具"), (_SKILL_MATRIX_PREVIEW, "技能"),
                          (_ACTION_MATRIX_PREVIEW, "操作")):
        assert f"显示全部 {preview + 7} 项 {noun}" in parsed.visible
        assert f"显示前 {preview} 项 {noun}" in parsed.visible
    assert "所有运行共同读取（26 个文件）" in parsed.visible
    assert "下方目录树中另有 2 个" in parsed.visible
    for english in ("Show all", "Show first", "actions", "tools", "skills", " only", "(baseline)"):
        assert english not in parsed.visible
    args = [args for key, args, attr in parsed.markers if key.startswith("{count} files read only by") and attr == "aria-label"]
    assert args and all(isinstance(value["count"], int) for value in args)


def test_recorded_values_matching_ui_words_and_html_stay_literal_unmarked():
    behavior = _behavior()
    behavior["labels"]["run-two"] = "Tool coverage"
    behavior["labels"]["run-three"] = 'shared <img src=x onerror="run()">'
    behavior["tool_matrix"][0]["key"] = "Skill coverage"
    behavior["skill_matrix"][0]["key"] = "Action coverage"
    behavior["action_matrix"][0]["target"] = 'Behavioral comparison <script>run()</script>'
    behavior["file_matrix"][0]["path"] = "Tool coverage/shared.py"
    behavior["patterns_vs_baseline"]["run-two"].append(
        {"type": "recorded-custom-type", "label": "redundant search", "count": 5}
    )
    original = deepcopy(behavior)
    output = build_run_group_behavior_html({"behavior": behavior})
    parsed = _Presentation(output)
    assert "Tool coverage" in parsed.unmarked
    assert 'shared <img src=x onerror="run()">' in parsed.unmarked
    assert "Skill coverage" in parsed.code and "Action coverage" in parsed.code
    assert 'Behavioral comparison <script>run()</script>' in parsed.code
    assert "redundant search" in "".join(parsed.unmarked)
    assert "Tool coverage/shared.py" in output
    assert '<img' not in output and '<script' not in output
    assert "&lt;img" in output and "&lt;script&gt;" in output
    assert behavior == original
    assert sum(key == "redundant search" for key, _, _ in parsed.markers) == 1
    # Marking known prose must never make a custom run label a translation key.
    assert not _Presentation(_run_label_html("Tool coverage", "run-one", baseline=True)).markers
    assert not _Presentation(_run_label_html("custom (baseline)", "run-one", baseline=True)).markers
    assert _Presentation(_run_label_html("run-one (baseline)", "run-one", baseline=True)).visible == "run-one （基线）"


def test_scorecard_headers_status_and_owned_errors_translate_while_rows_and_evidence_do_not():
    behavior = _behavior()
    row = {"run_id": "run-one", "label": "run-one (baseline)", "format": "Steps", "finished": True,
           "steps": 17, "wall_clock_fmt": "1m 5s", "tokens": 12034, "tool_calls": 8,
           "tool_success_pct": None, "peak_occupancy": 1234, "peak_pct": 12.5, "compactions": None}
    error_text = 'Tool coverage <b>raw evidence</b>'
    result = {"behavior": behavior, "rows": [row, {"run_id": "other", "label": "Condition", "error": error_text}]}
    original = deepcopy(result)
    output = build_run_group_scorecard_html(result, include_behavior=False)
    parsed = _Presentation(output)
    assert "正常结束" in parsed.visible and "上下文峰值" in parsed.visible
    assert "12,034" in parsed.visible and "1,234 (12.5%)" in parsed.visible and "1m 5s" in parsed.visible
    assert "Steps" in parsed.unmarked and "Condition" in parsed.unmarked and error_text in parsed.unmarked
    assert '<b>raw evidence</b>' not in output and '&lt;b&gt;raw evidence&lt;/b&gt;' in output
    assert result == original
    for message in _RUN_GROUP_MESSAGES:
        assert _Presentation(build_run_group_scorecard_html({"error": message})).markers
    assert not _Presentation(build_run_group_scorecard_html({"error": error_text})).markers
    assert _Presentation(build_run_group_scorecard_html({})).visible.startswith("先在概览中加载")


def test_known_generated_pattern_captions_all_have_translations():
    for caption in _PATTERN_LABELS.values():
        assert translate(caption, "zh-CN") != caption


def test_generated_baseline_and_only_prose_translate_in_hover_and_accessible_labels():
    behavior = _behavior()
    custom_name = "Tool coverage (baseline)"
    behavior["labels"]["run-two"] = custom_name
    original = deepcopy(behavior)
    parsed = _Presentation(build_run_group_behavior_html({"behavior": behavior}))
    descriptions = [(key, args, attr) for key, args, attr in parsed.markers if attr]
    for template in ("{run} (baseline)", "{count} only in {run} (baseline)"):
        assert {attr for key, _, attr in descriptions if key == template} == {"title", "aria-label"}
    mix_messages = [(key, args) for key, args, attr in descriptions
                    if key == "{count} files read only by {run} (baseline)" and attr == "aria-label"]
    assert mix_messages and all(args["run"] == "run-one" for _, args in mix_messages)
    for attrs in parsed.attributes:
        for attribute in ("title", "aria-label"):
            value = attrs.get(attribute, "")
            if "run-one" in value:
                assert "（基线）" in value
                assert "baseline" not in value and " only" not in value
    # Literal supplied names remain literal in both raw identities and owned
    # descriptions, even when their characters resemble generated UI prose.
    assert any(attrs.get("title") == custom_name and "data-tv-i18n" not in attrs for attrs in parsed.attributes)
    assert any(key == "{count} only in {run}" and args.get("run") == custom_name
               for key, args, _ in descriptions)
    assert behavior == original


def test_overview_baseline_attribute_templates_own_every_generated_word():
    label = "Overview (baseline)"
    for description in ("name", "only_in", "read_only"):
        message = _run_label_message(label, "overview", baseline=True, description=description, count=4)
        key, args = message
        assert "run" not in args
        expected = translate(key, "zh-CN", **args)
        assert "概览" in expected and "基线" in expected
        assert "Overview" not in expected and "baseline" not in expected and " only" not in expected
        rendered = _run_swatch_html(0, label, caption_html=_run_label_html(label, "overview", baseline=True),
                                    label_message=message)
        parsed = _Presentation(rendered)
        assert any(attrs.get("title") == expected for attrs in parsed.attributes)
        assert any(attrs.get("aria-label") == expected for attrs in parsed.attributes)
    assert _run_label_message(label, "a-custom-id", baseline=True) is None


def test_only_generated_default_baseline_timeline_labels_have_presentation_specs():
    fixture = Path(__file__).parent / "fixtures" / "native_claude_code" / "astropy__astropy-13033.jsonl"
    loaded = load_run(str(fixture))
    compared = LoadedRun("/recorded/run-two.json", loaded.raw, loaded.steps, loaded.format)
    original = deepcopy(loaded.raw)
    for source, expected in (
        ("/recorded/run-one.json", {"template": "{run} (baseline)", "params": {"run": "run-one"}}),
        ("", {"template": "Overview (baseline)", "params": {}}),
    ):
        baseline = LoadedRun(source, loaded.raw, loaded.steps, loaded.format)
        generated = build_run_group_scorecard([compared], baseline_raw=baseline)
        assert generated["ok"]
        first, second = generated["timeline_runs"]
        assert first["label_i18n"] == expected
        assert "label_i18n" not in second
        assert all("label_i18n" not in row for row in generated["rows"])
        assert "label_i18n" not in generated["behavior"]

        # The exact same text supplied as a custom label remains literal.
        literal = build_run_group_scorecard([compared], baseline_raw=baseline, baseline_label=first["label"])
        assert literal["ok"] and "label_i18n" not in literal["timeline_runs"][0]
        assert generated["rows"] == literal["rows"]
        assert generated["behavior"] == literal["behavior"]
        assert {key: value for key, value in first.items() if key != "label_i18n"} == literal["timeline_runs"][0]
    assert loaded.raw == original


def test_uploaded_runs_with_literal_baseline_suffix_receive_no_presentation_spec():
    fixture = Path(__file__).parent / "fixtures" / "native_claude_code" / "astropy__astropy-13033.jsonl"
    loaded = load_run(str(fixture))
    runs = [LoadedRun(f"/recorded/run-{i}.json", loaded.raw, loaded.steps, loaded.format) for i in range(2)]
    result = build_run_group_scorecard(runs, labels=["run-0 (baseline)", "Overview (baseline)"])
    assert result["ok"]
    assert [run["label"] for run in result["timeline_runs"]] == ["run-0 (baseline)", "Overview (baseline)"]
    assert all("label_i18n" not in run for run in result["timeline_runs"])
