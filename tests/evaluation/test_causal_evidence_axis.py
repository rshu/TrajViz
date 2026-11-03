import unittest
from agent_trajviz.awe._engine.evidence import build_chain
from agent_trajviz.awe._engine.dossier import to_csv_rows
from .test_batch8_signals import _gold


class TestCausalEvidenceAxis(unittest.TestCase):
    def test_deductive_observation_is_not_causal_proof(self):
        c = {
            "capability": "code_editing",
            "error_type": "incorrect_patch",
            "confidence": "high",
            "evidence": {"hit": ["pkg/a.py"]},
        }
        chain = build_chain(c, {"signals": {}}, _gold(), {}, {}, 1.0, True)
        self.assertEqual(chain["strength"], "deductive")
        self.assertEqual(chain["observation_strength"], "deductive")
        self.assertEqual(chain["causal_support"]["status"], "not_tested")
        self.assertEqual(chain["counterfactual"]["status"], "not_tested")
        f = {
            "capability": c["capability"],
            "error_type": c["error_type"],
            "label": "editing",
            "confidence": "high",
            "blame_weight": 1,
            "evidence_chain": chain,
            "is_primary": True,
        }
        rows = to_csv_rows(
            [
                {
                    "case_id": "a__i",
                    "agent": "a",
                    "instance_id": "i",
                    "repo": "r",
                    "faults": [f],
                    "task": {"gold_files": ["pkg/a.py"], "problem_statement": "issue"},
                }
            ]
        )
        self.assertEqual(rows[0]["observation_strength"], "deductive")
        self.assertEqual(rows[0]["causal_validation"], "not_tested")

    def test_judge_corroboration_does_not_become_intervention(self):
        c = {
            "capability": "code_editing",
            "error_type": "relevant_change_omitted",
            "confidence": "high",
            "evidence": {"read_not_edited": ["pkg/a.py"]},
        }
        d = {"signals": {}, "arbiter": {"support": "confirm", "confidence": "high"}}
        self.assertEqual(build_chain(c, d, _gold(), {}, {}, 1, True)["causal_support"]["status"], "not_tested")
