import unittest
from agent_trajviz.awe._engine.preprocessing_validation import controlled_validation, project
from agent_trajviz.awe._engine.trajectory import NormalizedStep as S, NormalizedTrajectory as T


class PreprocessingValidation(unittest.TestCase):
    def test_all_independent_observational_controls(self):
        report = controlled_validation()
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(len(report["checks"]), 15)
        self.assertEqual(len(report["cases"]), 6)
        for row in report["cases"].values():
            for projection in row.values():
                self.assertEqual(
                    projection["view"]["retained_rows"] + projection["view"]["omitted_rows"],
                    projection["view"]["eligible_rows"],
                )

    def test_projection_never_mutates_original(self):
        traj = T(
            "a",
            "i",
            [S(0, 0, "tool", "tool_result", output_excerpt="evidence", args={"verification_state": {"revision": "a"}})],
        )
        before = traj.to_dict()
        for mode in ("drop_results", "drop_result_text", "drop_state", "legacy_lossy_proxy"):
            project(traj, mode)
            self.assertEqual(traj.to_dict(), before)
        with self.assertRaises(ValueError):
            project(traj, "unsupported")

    def test_wrong_file_audit_requires_unreached_reference(self):
        from agent_trajviz.awe._engine.audit import audit_fault, MISMATCH, UNVERIFIABLE

        fault = {
            "capability": "code_localization",
            "error_type": "wrong_file_localization",
            "detector_evidence": {"extra_files": ["b.py"]},
        }
        gold = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        patch = "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
        kwargs = dict(agent="a", inst="i", gold_text=gold, agent_text=patch, resolved=False)
        self.assertEqual(audit_fault(fault, traj=None, **kwargs)["verdict"], UNVERIFIABLE)
        traj = T(
            "a",
            "i",
            [
                S(0, 0, "agent", "tool_call", canonical_action="read", call_id="r", files=["a.py"]),
                S(1, 0, "tool", "tool_result", call_id="r", output_excerpt="content"),
            ],
        )
        self.assertEqual(audit_fault(fault, traj=traj, **kwargs)["verdict"], MISMATCH)
