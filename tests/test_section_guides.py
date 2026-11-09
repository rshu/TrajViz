"""Overview Contents section guides — one per section, rendered at its top."""

import unittest

import gradio as gr

from agent_trajviz.insight.help import SECTION_GUIDES, SECTION_GUIDES_EN
from agent_trajviz.insight.presenters.overview import render_section_guide
from agent_trajviz.insight.ui import overview_tab


class SectionGuideTests(unittest.TestCase):
    def test_every_contents_section_has_a_guide(self):
        self.assertEqual(
            {attr for _, attr in overview_tab.OVERVIEW_SECTIONS},
            set(SECTION_GUIDES),
        )
        self.assertEqual(set(SECTION_GUIDES), set(SECTION_GUIDES_EN))

    def test_guide_renders_summary_charts_and_example(self):
        for section, guide in SECTION_GUIDES.items():
            with self.subTest(section=section):
                self.assertTrue(guide.charts)
                html = render_section_guide(section)
                self.assertIn(f'data-tv-i18n-zh="{guide.summary}"', html)
                self.assertIn(SECTION_GUIDES_EN[section].summary, html)
                for name, desc in guide.charts:
                    self.assertIn(f'data-tv-i18n-zh="{name}"', html)
                    self.assertIn(desc, html)
                self.assertIn(guide.example, html)
                self.assertIn("ⓘ 图表说明与调试示例", html)
                self.assertIn(">Charts<", html)
                self.assertIn(">Debugging example<", html)
                self.assertIn("class='section-guide-tip' tabindex='0'", html)
                self.assertIn("role='tooltip'", html)

    def test_performance_guide_mentions_per_agent_panels(self):
        guide = SECTION_GUIDES["performance_section"]
        blurbs = " ".join(desc for _, desc in guide.charts)
        self.assertIn("每个智能体单独一块图", blurbs)
        self.assertIn("按智能体分图", blurbs)

    def test_each_section_column_starts_with_its_guide(self):
        with gr.Blocks():
            kpi_html = gr.HTML("")
            with gr.Tabs():
                refs = overview_tab.layout(kpi_html)
        for _, attr in overview_tab.OVERVIEW_SECTIONS:
            with self.subTest(section=attr):
                first = getattr(refs, attr).children[0]
                self.assertIsInstance(first, gr.HTML)
                self.assertEqual(first.value, render_section_guide(attr))


if __name__ == "__main__":
    unittest.main()
