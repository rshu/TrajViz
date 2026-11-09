"""Converge localizes its presentation without rewriting report evidence."""

from html.parser import HTMLParser

from agent_trajviz.converge.charts import build_anchor_class_chart, build_segment_cost_chart
from agent_trajviz.converge.rendering import build_anchor_analysis_html, build_comparison_report_html
from agent_trajviz.i18n import catalog, translate


class _Markers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.keys = []

    def handle_starttag(self, tag, attrs):
        key = dict(attrs).get("data-tv-i18n")
        if key is not None:
            self.keys.append(key)


def test_report_markers_are_owned_text_and_evidence_remains_literal():
    rendered = build_comparison_report_html({
        "task_id": "Metric <script>run()</script>",
        "outcome": {"reference_filename": "Steps", "compared_filename": "Reference"},
        "ref_milestones": {"first_edit": 2, "custom_milestone": 3},
        "patterns": [{
            "type": "Anchor Analysis",
            "steps": [1],
            "estimated_extra_cost": {"tokens": 5},
            "evidence": ["Reference <b>do not translate</b>"],
        }],
    })
    parsed = _Markers()
    parsed.feed(rendered)
    assert "Converge Comparison Report" in parsed.keys
    assert "First Edit" in parsed.keys
    assert "Custom Milestone" not in parsed.keys
    assert "Anchor Analysis" not in parsed.keys
    assert "<th>Steps</th>" in rendered
    assert "<th>Reference</th>" in rendered
    assert "<td>Anchor Analysis</td>" in rendered
    assert "Reference &lt;b&gt;do not translate&lt;/b&gt;" in rendered
    assert "Metric &lt;script&gt;run()&lt;/script&gt;" in rendered
    assert set(parsed.keys) <= catalog().keys()


def test_anchor_labels_preserve_class_names_and_parameter_escaping():
    rendered = build_anchor_analysis_html({
        "total_anchor_files": 1,
        "file_classes": {"Metric <img src=x>": 1},
        "reference": {"time_to_first_anchor_read": 4},
    })
    parsed = _Markers()
    parsed.feed(rendered)
    assert "Write Precision" in parsed.keys
    assert "Step {index}" in parsed.keys
    assert "Metric <img src=x>" not in parsed.keys
    assert "<img src=x>" not in rendered
    assert "<td>Metric &lt;img src=x&gt;</td>" in rendered
    assert translate("File classes: {summary} ({total} total)", "zh-CN", summary="1 Metric", total=1) == (
        "文件类别：1 Metric（共 1 个）"
    )


def test_chart_localization_metadata_does_not_rewrite_plotted_categories():
    anchor = build_anchor_class_chart({
        "file_classes": {"Reference": 1, "Metric": 1},
        "reference": {"write_recall_by_class": {"Reference": 0.5}},
    })
    assert list(anchor.data[0].x) == ["Metric", "Reference"]
    assert list(anchor.data[0].y) == [0, 50]
    meta = anchor.layout.meta["tv_i18n"]
    assert meta["traces"] == {"0": "Reference", "1": "Compared"}
    assert meta["layout"]["yaxis.title.text"] == "Recall (%)"
    assert all(key in catalog() for key in meta["layout"].values())

    segments = build_segment_cost_chart({
        "segment_comparison": [{"segment": "Comparison Report", "overhead": 1.75}],
    }, True)
    assert list(segments.data[0].x) == ["Comparison Report"]
    assert list(segments.data[0].y) == [1.75]
    assert segments.layout.meta["tv_i18n"]["traces"] == {"0": "Overhead Ratio"}
