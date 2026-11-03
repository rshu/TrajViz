import unittest
from agent_trajviz.awe._engine.verification import test_summary as _test_summary, verification_observations
from agent_trajviz.awe._engine.symptoms import analyze
from .test_verification_observations import call, result, traj


class VerificationReviewEdges(unittest.TestCase):
    def test_invalid_summary_cannot_hide_before_valid_zero_summary(self):
        for text in (
            "1 passed, 2 passed in 0.2s\nno tests ran in 0.2s",
            "1 passed, unsupported in 0.2s\nno tests ran in 0.2s",
            "no tests ran in 0.2s\nFAILED (failures=1)",
        ):
            self.assertEqual(_test_summary(text)["status"], "unknown")
            self.assertEqual(
                [s["kind"] for s in analyze(traj(call(), result(out=text)))["symptoms"]],
                ["verification_evidence_unknown"],
            )

    def test_failed_unittest_verdict_needs_a_failure_detail(self):
        for text in ("Ran 0 tests in 0.1s\nFAILED", "Ran 1 tests in 0.1s\nFAILED (skipped=1)"):
            self.assertEqual(_test_summary(text)["status"], "unknown")

    def test_shell_substitution_and_redirection_abstain(self):
        for command in ("cd $(echo dir); pytest", "cd `echo dir`; pytest", "pytest > log", "pytest < input"):
            r = verification_observations(traj(call(command=command), result(out="no tests ran in 0.2s")))
            self.assertEqual(r["runs"][0]["summary"]["status"], "unknown")
