"""Tests for the scout (open-ended failure-mode discovery).

Pure parse/aggregate tests + a mock-client round-trip — no ARGUS data, no API key.

Run:  python3 -m unittest tests.test_scout -v
"""

from __future__ import annotations

import json
import unittest


from agent_trajviz.awe._engine import scout


class TestParseModes(unittest.TestCase):
    def test_sanitizes_label_desc_caps_and_steps(self):
        content = json.dumps(
            {
                "modes": [
                    {
                        "label": "Premature Implementation!!",
                        "description": "x" * 250,
                        "covered_by": "not_a_capability",
                        "novel": True,
                        "evidence_steps": ["3", "bad", 5],
                    }
                ]
            }
        )
        m = scout.parse_modes(content)[0]
        self.assertEqual(m["label"], "premature-implementation")
        self.assertEqual(len(m["description"]), 160)
        self.assertIsNone(m["covered_by"])  # invalid capability -> None
        self.assertTrue(m["novel"])  # novel + uncovered
        self.assertEqual(m["evidence_steps"], [3, 5])

    def test_novel_is_false_when_mapped_to_a_real_capability(self):
        content = json.dumps(
            {
                "modes": [
                    {
                        "label": "wrong-edit",
                        "description": "d",
                        "covered_by": "code_editing",
                        "novel": True,
                        "evidence_steps": [1],
                    }
                ]
            }
        )
        m = scout.parse_modes(content)[0]
        self.assertEqual(m["covered_by"], "code_editing")
        self.assertFalse(m["novel"])  # mapping overrides a 'novel' claim

    def test_drops_malformed_and_labelless(self):
        content = json.dumps({"modes": [{"description": "no label"}, "notadict", {"label": "ok", "novel": False}]})
        out = scout.parse_modes(content)
        self.assertEqual([m["label"] for m in out], ["ok"])

    def test_garbage_is_empty(self):
        self.assertEqual(scout.parse_modes("not json at all"), [])


class TestAggregateNovel(unittest.TestCase):
    def test_groups_novel_by_label_counts_and_examples(self):
        recs = [
            {
                "agent": "a",
                "instance_id": "i1",
                "modes": [
                    {
                        "label": "ignored-stack-trace",
                        "description": "missed the traceback line",
                        "covered_by": None,
                        "novel": True,
                        "evidence_steps": [3],
                    },
                    {"label": "wrong-edit", "covered_by": "code_editing", "novel": False},
                ],
            },
            {
                "agent": "b",
                "instance_id": "i2",
                "modes": [
                    {
                        "label": "ignored-stack-trace",
                        "description": "skipped error output",
                        "covered_by": None,
                        "novel": True,
                        "evidence_steps": [9],
                    }
                ],
            },
            None,  # a dead/None record must be tolerated
        ]
        agg = scout.aggregate_novel(recs)
        self.assertEqual(len(agg), 1)  # only the novel label
        g = agg[0]
        self.assertEqual(g["label"], "ignored-stack-trace")
        self.assertEqual(g["count"], 2)
        self.assertEqual(set(g["examples"]), {"a/i1", "b/i2"})
        self.assertTrue(g["description"])  # representative desc kept

    def test_no_novel_modes_yields_empty(self):
        recs = [
            {"agent": "a", "instance_id": "i", "modes": [{"label": "x", "covered_by": "code_editing", "novel": False}]}
        ]
        self.assertEqual(scout.aggregate_novel(recs), [])


class TestMockRoundTrip(unittest.TestCase):
    def test_mock_client_output_parses(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "a failed trace with steps [1] [2]"}]
        r = scout.mock_client(msgs)
        modes = scout.parse_modes(r["content"])
        self.assertTrue(modes)
        self.assertTrue(all(m["label"] for m in modes))

    def test_report_md_lists_candidates(self):
        recs = [
            {
                "agent": "a",
                "instance_id": "i",
                "modes": [
                    {
                        "label": "novel-thing",
                        "description": "a mechanism",
                        "covered_by": None,
                        "novel": True,
                        "evidence_steps": [1],
                    }
                ],
            }
        ]
        md = scout.candidate_report_md(scout.aggregate_novel(recs), len(recs))
        self.assertIn("novel-thing", md)
        self.assertIn("human review", md.lower())


if __name__ == "__main__":
    unittest.main()
