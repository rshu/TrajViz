import unittest
from agent_trajviz.awe._engine.trajectory import looks_like_success_claim, NormalizedStep, NormalizedTrajectory
from agent_trajviz.awe._engine.trajsignals import compute
from .test_batch8_signals import _gold


class TestSuccessClaimScope(unittest.TestCase):
    def test_negated_conditional_quoted_or_retracted_claims_abstain(self):
        for text in [
            "I cannot confirm that all tests pass; fix incomplete",
            "Do all tests pass?",
            "Not all tests pass",
            "I haven't successfully fixed the bug",
            "If all tests pass, merge it",
            'The phrase "all tests pass" is misleading',
            "Earlier I thought all tests pass",
            "All tests pass is not true",
        ]:
            self.assertFalse(looks_like_success_claim(text), text)

    def test_positive_claims_remain_detectable(self):
        for text in [
            "All tests pass.",
            "All tests passed.",
            "The fix is complete",
            "This resolves the bug",
            "Not done with docs, but all tests pass",
        ]:
            self.assertTrue(looks_like_success_claim(text), text)

    def test_later_message_withdraws_prior_success(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                NormalizedStep(0, 0, "agent", "message", text="All tests pass"),
                NormalizedStep(1, 1, "agent", "final_answer", text="The fix remains incomplete."),
            ],
        )
        sig = compute(t, _gold())
        self.assertTrue(sig.claimed_success_any)
        self.assertFalse(sig.claimed_success_final)
