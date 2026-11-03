import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.trajectory import NormalizedStep as S, EDIT, observed_test_runs
from .test_batch8_signals import _gold
from .test_verification_observations import call, result, traj


def edit(n, file="package/source.py"):
    return S(n, n, "agent", "tool_call", canonical_action=EDIT, files=[file])


class ZeroExecutionCreditTests(unittest.TestCase):
    def test_known_zero_is_attempt_without_verification_credit(self):
        for text in (
            "no tests ran in 0.2s",
            "3 skipped in 0.2s",
            "Ran 0 tests in 0.001s\nOK",
            "Ran 3 tests in 0.001s\nOK (skipped=3)",
        ):
            t = traj(edit(0), call(1), result(2, out=text))
            s = compute(t, _gold())
            self.assertEqual(s.n_test_attempts, 1)
            self.assertEqual(s.n_test_runs, 0)
            self.assertFalse(s.ran_any_test)
            self.assertFalse(s.ran_suite_wide)
            self.assertTrue(s.no_test_after_final_edit)
            self.assertTrue(audit._redrv_no_test_after_final_edit(t))
            self.assertEqual(observed_test_runs(t), [])

    def test_unknown_preserves_intent_policy(self):
        for t in (traj(call()), traj(call(), result(out="output omitted"))):
            s = compute(t, _gold())
            self.assertTrue(s.ran_any_test)
            self.assertEqual(s.n_test_attempts, 1)

    def test_zero_after_edit_does_not_close_repair_window(self):
        t = traj(call(), result(code=1, out="1 failed in 0.2s"), edit(2), call(3), result(4, out="3 skipped in 0.2s"))
        s = compute(t, _gold())
        self.assertTrue(s.repair_not_reverified)
        self.assertTrue(audit._redrv_repair_not_reverified(t))
        self.assertTrue(s.last_test_failed)
        self.assertTrue(audit._redrv_last_test_failed(t))

    def test_zero_authored_run_cannot_be_protective_evidence(self):
        t = traj(
            edit(0, "tests/test_repro.py"),
            call(1, command="pytest tests/test_repro.py"),
            result(2, out="3 skipped in 0.2s"),
        )
        s = compute(t, _gold())
        self.assertTrue(s.authored_test_never_run)
        self.assertFalse(s.authored_test_ran_never_failed)
        self.assertTrue(audit._redrv_authored_test_never_run(t))

    def test_positive_followup_restores_credit(self):
        t = traj(edit(0), call(1), result(2, out="3 skipped in 0.2s"), call(3), result(4))
        s = compute(t, _gold())
        self.assertEqual((s.n_test_attempts, s.n_test_runs), (2, 1))
        self.assertTrue(s.ran_suite_wide)
        self.assertFalse(s.no_test_after_final_edit)

    def test_sequential_reused_ids_do_not_rewrite_earlier_failures(self):
        t = traj(call(), result(code=1, out="1 failed in 0.2s"), edit(2), call(3), result(4))
        s = compute(t, _gold())
        self.assertEqual(s.n_failure_events, 1)
        self.assertTrue(s.encountered_failure)
        self.assertEqual(audit._failure_event_steps(t), [0])
        self.assertFalse(s.last_test_failed)

    def test_ambiguous_results_do_not_prove_authored_execution(self):
        t = traj(
            edit(0, "tests/test_repro.py"),
            call(1, command="pytest tests/test_repro.py"),
            call(2, command="pytest tests/test_repro.py"),
            result(3),
        )
        self.assertEqual(observed_test_runs(t), [])
        self.assertTrue(compute(t, _gold()).authored_test_never_run)

    def test_attempt_is_investigation_even_with_no_source_patch(self):
        t = traj(call(), result(out="3 skipped in 0.2s"))
        s = compute(t, _gold())
        loc = {"flag": "no_source_edit", "patch_state": "empty", "gold_tests_edited": []}
        candidates = objective_candidates(_gold(), loc, s, False)
        types = {c.error_type for c in candidates}
        self.assertIn("no_patch_produced", types)
        self.assertNotIn("gave_up_early", types)
        self.assertIn("relevant_test_not_run", types)
