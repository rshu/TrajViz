import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.evidence import build_chain
from agent_trajviz.awe._engine.signals import localization_signals
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory
from .test_batch8_signals import _gold, _edit, _test_call, _result
from .test_review_fixes_data import _patch


class TestAuthoredEvidenceScope(unittest.TestCase):
    def fixture(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                _edit(0, "tests/test_first.py"),
                _edit(1, "tests/test_second.py"),
                _test_call(2, "pytest tests/test_first.py"),
                _result(3, 2, 0),
            ],
        )
        g = _gold()
        sig = compute(t, g)
        c = next(
            c
            for c in objective_candidates(g, localization_signals(g, _patch(), False), sig, False)
            if c.error_type == "authored_test_never_run"
        )
        return t, g, c

    def test_chain_names_only_unverified_files_and_correct_temporal_scope(self):
        t, g, c = self.fixture()
        self.assertEqual(c.evidence["unrun_authored_test_files"], ["tests/test_second.py"])
        chain = build_chain(
            {"capability": c.capability, "error_type": c.error_type, "evidence": c.evidence},
            {"signals": {}},
            g,
            {},
            {},
            0.3,
            False,
        )
        statement = " ".join(evidence_link["statement"] for evidence_link in chain["links"])
        self.assertIn("tests/test_second.py", statement)
        self.assertNotIn("tests/test_first.py", statement)
        self.assertIn("after their latest edit", statement)
        self.assertNotIn("no test command ever", statement)

    def test_audit_detects_tampered_unverified_file_list(self):
        t, g, c = self.fixture()
        fault = {
            "capability": c.capability,
            "error_type": c.error_type,
            "detector_evidence": {**c.evidence, "unrun_authored_test_files": ["tests/test_first.py"]},
        }
        result = audit.audit_fault(fault, agent="a", inst="i", gold_text="", agent_text="", resolved=False, traj=t)
        self.assertEqual(result["verdict"], "mismatch")
        self.assertTrue(any(m["atom"] == "unrun_authored_test_files" for m in result["mismatches"]))

    def test_neighboring_repair_and_tool_chains_retain_specific_evidence(self):
        for cap, et, source in [
            ("self_repair_loop", "repair_not_reverified", "trajectory:edit_after_failure"),
            ("tool_use", "hallucinated_path_unrecovered", "trajectory:repeated_invalid_target"),
        ]:
            chain = build_chain(
                {"capability": cap, "error_type": et, "evidence": {}}, {"signals": {}}, _gold(), {}, {}, 0.3, False
            )
            self.assertIn(source, [link["source"] for link in chain["links"]])
            self.assertTrue(chain["counterfactual"]["what_would_confirm"])
