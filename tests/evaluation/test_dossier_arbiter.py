"""Arbiter provenance is visible in the dossier surfaces. arbiter_note is pure;
to_markdown / to_csv_rows are pure over case dicts (no ARGUS data).

Run:  python3 -m unittest tests.test_dossier_arbiter -v
"""

import unittest

from agent_trajviz.awe._engine import dossier


def _case(arb=None):
    return {
        "case_id": "a__i",
        "agent": "a",
        "instance_id": "i",
        "repo": "r",
        "blame_status": "conjunctive" if arb else "primary",
        "arbiter": arb,
        "task": {
            "problem_statement": "do the thing",
            "gold_files": ["pkg/a.py"],
            "multi_file": False,
            "cross_module": False,
        },
        "reasoning_observability": "observed",
        "thinking_process": [],
        "faults": [
            {
                "label": "Code Editing -> relevant_change_omitted",
                "capability": "code_editing",
                "error_type": "relevant_change_omitted",
                "confidence": "high",
                "blame_weight": 1.0,
                "is_primary": False,
                "detector_evidence": {},
                "code_evidence": {},
                "evidence_chain": None,
                "audit": None,
                "recommendation": ["x"],
            }
        ],
    }


class TestArbiterNote(unittest.TestCase):
    def test_refute_note(self):
        s = dossier.arbiter_note(
            {
                "applied": "demoted_to_conjunctive",
                "support": "refute",
                "confidence": "high",
                "target": {"error_type": "relevant_change_omitted"},
                "rationale": "handled fine",
            }
        )
        self.assertIn("refuted", s)
        self.assertIn("relevant_change_omitted", s)

    def test_corroborated_note(self):
        s = dossier.arbiter_note(
            {
                "applied": "corroborated",
                "support": "confirm",
                "confidence": "high",
                "target": {"error_type": "flawed_plan"},
            }
        )
        self.assertIn("corroborated", s)

    def test_none_is_empty(self):
        self.assertEqual(dossier.arbiter_note(None), "")


class TestSurfaces(unittest.TestCase):
    def test_markdown_shows_refute_on_diagnosis(self):
        md = dossier.to_markdown(
            [
                _case(
                    {
                        "applied": "demoted_to_conjunctive",
                        "support": "refute",
                        "confidence": "high",
                        "target": {"error_type": "relevant_change_omitted"},
                        "rationale": "ok",
                    }
                )
            ]
        )
        self.assertIn("refuted", md)

    def test_csv_has_arbiter_support_column(self):
        rows = dossier.to_csv_rows(
            [
                _case(
                    {
                        "applied": "demoted_to_conjunctive",
                        "support": "refute",
                        "confidence": "high",
                        "target": {"error_type": "relevant_change_omitted"},
                    }
                )
            ]
        )
        self.assertEqual(rows[0]["arbiter_support"], "refute")

    def test_csv_blank_when_no_arbiter(self):
        rows = dossier.to_csv_rows([_case(None)])
        self.assertEqual(rows[0]["arbiter_support"], "")

    def test_csv_stamps_only_the_arbiter_target_row(self):
        """The case verdict must not broadcast onto non-target faults — a
        deductive row would otherwise read as adjudicated, violating
        deductive_never_adjudicated for CSV consumers (v0.31)."""
        case = _case(
            {
                "applied": "demoted_to_conjunctive",
                "support": "refute",
                "confidence": "high",
                "target": {"capability": "code_editing", "error_type": "relevant_change_omitted"},
            }
        )
        case["faults"].append(
            {
                "label": "Code Editing -> incorrect_patch",
                "capability": "code_editing",
                "error_type": "incorrect_patch",
                "confidence": "high",
                "blame_weight": 0.0,
                "is_primary": False,
                "detector_evidence": {},
                "code_evidence": {},
                "evidence_chain": None,
                "audit": None,
                "recommendation": ["x"],
            }
        )
        rows = dossier.to_csv_rows([case])
        by_et = {r["error_type"]: r["arbiter_support"] for r in rows}
        self.assertEqual(by_et["relevant_change_omitted"], "refute")
        self.assertEqual(by_et["incorrect_patch"], "")


if __name__ == "__main__":
    unittest.main()
