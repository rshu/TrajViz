"""Reporting must distinguish attribution gaps and every applied refutation."""

import unittest
from collections import Counter
from unittest.mock import patch

from agent_trajviz.awe._engine import config, digest


class AttributionReportingTests(unittest.TestCase):
    def test_gap_is_not_reported_as_conjunction(self):
        data = {
            "agent": {
                "n_failed": 4,
                "primary": Counter({"incorrect_patch": 1}),
                "attribution_statuses": {"primary": 1, "conjunctive": 2, "unattributed": 1},
            }
        }
        with patch.object(config, "AGENTS", ["agent"]):
            md = "\n".join(digest._attribution_section(data))
        self.assertIn("| agent | 1 | 2 | 1 | 4 |", md)
        self.assertIn("does not prove multiple causal mechanisms", md)

    def test_all_applied_refutations_are_explained(self):
        data = {
            "agent": {
                "adj_total": 10,
                "adj_refute": 8,
                "arbiter_applied": {
                    "demoted_to_conjunctive": 3,
                    "refuted_reassigned": 4,
                    "unattributed_after_refute": 1,
                    "noted": 2,
                },
            }
        }
        with patch.object(config, "AGENTS", ["agent"]):
            md = digest._arbiter_summary_line(data)
        self.assertIn("applied 8 refutes", md)
        self.assertIn("3 no dominant primary, 4 reassigned, 1 unattributed", md)

    def test_partial_judge_coverage_is_not_claimed_complete(self):
        data = {
            "agent": {
                "n": 10,
                "n_failed": 4,
                "failure_rate": 0.4,
                "judge_opportunities": {"requirement_understanding": 6, "task_planning": 2},
            }
        }
        with patch.object(config, "AGENTS", ["agent"]), patch.object(digest, "_judge_present", return_value=True):
            md = "\n".join(digest._coverage_section(data))
        self.assertIn("| agent | 6 | 2 |", md)
        self.assertNotIn("all 4 were also scored", md)
        self.assertIn("does not measure causal precision", md)

    def test_unequal_cohorts_are_not_presented_as_rectangular(self):
        from .test_digest_headline import _D

        agents, data = _D()
        data[agents[1]]["n"] = 499
        with patch.object(config, "AGENTS", agents):
            md = "\n".join(digest._headline_section(data))
        self.assertIn("999 scored runs (499–500 tasks per agent)", md)
        self.assertNotIn("2 agents × 500", md)
