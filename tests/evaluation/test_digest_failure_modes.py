"""Failure-composition mode decomposition and CSV-row data."""

import unittest
from collections import Counter
from unittest import mock

from agent_trajviz.awe._engine import config, digest


def _data():
    agents = ["agent_a", "agent_b"]
    data = {
        "agent_a": {
            "n_failed": 10,
            "scores": {"code_editing": 80.0},
            "cap_mass": Counter({"code_editing": 4.0}),
            "mode_mass": {"code_editing": {"incorrect_patch": 3.0, "no_patch_produced": 1.0}},
            "mode_count": {"code_editing": {"incorrect_patch": 4, "no_patch_produced": 1}},
        },
        "agent_b": {
            "n_failed": 20,
            "scores": {"code_editing": 70.0},
            "cap_mass": Counter({"code_editing": 2.0}),
            "mode_mass": {"code_editing": {"incorrect_patch": 0.5, "no_patch_produced": 1.5}},
            "mode_count": {"code_editing": {"incorrect_patch": 1, "no_patch_produced": 2}},
        },
    }
    return agents, data


class TestFailureModeComposition(unittest.TestCase):
    def test_markdown_uses_within_phase_denominator_and_overall_mass(self):
        agents, data = _data()
        with mock.patch.object(config, "AGENTS", agents):
            text = "\n".join(digest._failure_mode_share_section(data))
        self.assertIn("#### Editing", text)
        self.assertIn(
            "| `incorrect_patch` | **58.333333333333336%** | **75.0%** | 25.0% |",
            text,
        )
        self.assertIn(
            "| `no_patch_produced` | 41.666666666666664% | 25.0% | **75.0%** |",
            text,
        )

    def test_long_rows_expose_both_denominators(self):
        agents, data = _data()
        with mock.patch.object(config, "AGENTS", agents):
            rows = digest.failure_mode_rows(data)
        row = next(r for r in rows if r["root_cause_mode"] == "incorrect_patch" and r["agent"] == "agent_a")
        self.assertEqual(row["fault_entry_count"], 4)
        self.assertEqual(row["attributed_blame_mass"], 3.0)
        self.assertEqual(row["phase_blame_mass"], 4.0)
        self.assertEqual(row["share_within_phase_pct"], 75.0)
        self.assertEqual(row["share_of_agent_failures_pct"], 30.0)

        overall = next(r for r in rows if r["root_cause_mode"] == "incorrect_patch" and r["agent"] == "__all_agents__")
        self.assertEqual(overall["attributed_blame_mass"], 3.5)
        self.assertEqual(overall["phase_blame_mass"], 6.0)
        self.assertEqual(overall["failed_tasks"], 30)

    def test_phase_mass_is_exact_sum_of_mode_masses(self):
        agents, data = _data()
        with mock.patch.object(config, "AGENTS", agents):
            rows = digest.failure_mode_rows(data)
        overall = [r for r in rows if r["agent"] == "__all_agents__"]
        self.assertEqual(
            sum(r["attributed_blame_mass"] for r in overall),
            overall[0]["phase_blame_mass"],
        )

    def test_old_digest_without_mode_mass_omits_additive_section(self):
        agents, data = _data()
        for record in data.values():
            record.pop("mode_mass")
        with mock.patch.object(config, "AGENTS", agents):
            self.assertEqual(digest._failure_mode_share_section(data), [])

    def test_unattributed_uses_explicit_mass_not_float_remainder(self):
        agents, data = _data()
        data["agent_a"]["cap_mass"]["unattributed"] = 1.0
        with mock.patch.object(config, "AGENTS", agents):
            text = "\n".join(digest._failure_share_section(data))
        row = next(line for line in text.splitlines() if line.startswith("| _unattributed_"))
        self.assertEqual(row, "| _unattributed_ | 10.0% | 0.0% |")


if __name__ == "__main__":
    unittest.main()
