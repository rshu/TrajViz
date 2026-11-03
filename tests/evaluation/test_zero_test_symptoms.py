import unittest
from agent_trajviz.awe._engine.symptoms import analyze, zero_test_symptoms
from .test_verification_observations import call, result, traj


class ZeroTestSymptomsTests(unittest.TestCase):
    def test_zero_and_all_skipped_are_distinct(self):
        for text, kind in [
            ("no tests ran in 0.2s", "zero_tests_executed"),
            ("Ran 0 tests in 0.001s\nOK", "zero_tests_executed"),
            ("3 skipped in 0.2s", "all_tests_skipped"),
            ("Ran 3 tests in 0.001s\nOK (skipped=3)", "all_tests_skipped"),
        ]:
            rows = analyze(traj(call(), result(out=text)))["symptoms"]
            self.assertEqual([r["kind"] for r in rows], [kind])
            self.assertEqual(rows[0]["result"]["step_id"], 1)
            self.assertEqual(rows[0]["scoring"], "not_scored")

    def test_missing_unknown_and_positive_runs_do_not_fire(self):
        for t in (
            None,
            traj(call()),
            traj(call(), result(out="completed")),
            traj(call(), result(out="1 passed, 2 skipped in 0.2s")),
            traj(call(), result(out="1 error in 0.2s")),
        ):
            self.assertFalse(
                any(s["kind"] in ("zero_tests_executed", "all_tests_skipped") for s in analyze(t)["symptoms"])
            )

    def test_later_execution_is_not_passing_or_terminal_recovery(self):
        t = traj(call(), result(out="no tests ran in 0.2s", code=5), call(2), result(3, out="1 failed in 0.2s", code=1))
        row = analyze(t)["symptoms"][0]
        self.assertEqual(row["lifecycle"], "later_execution_observed")
        self.assertEqual(row["terminal_relevance"], "unknown")
        self.assertEqual(row["causal_validation"], "not_tested")

    def test_other_thread_or_selector_does_not_establish_later_execution(self):
        for later in (call(2, thread="sub"), call(2, command="pytest tests/other.py")):
            t = traj(call(), result(out="no tests ran in 0.2s"), later, result(3, thread=later.subagent_id))
            self.assertEqual(analyze(t)["symptoms"][0]["lifecycle"], "no_later_execution_observed")

    def test_parallel_future_call_is_not_a_retest(self):
        t = traj(call(cid="a"), call(1, cid="b"), result(2, cid="a", out="no tests ran in 0.2s"), result(3, cid="b"))
        self.assertIsNone(analyze(t)["symptoms"][0]["later_execution"])

    def test_boolean_zero_and_missing_result_cannot_be_evidence(self):
        from agent_trajviz.awe._engine.verification import verification_observations

        observations = verification_observations(traj(call(), result()))
        observations["runs"][0]["summary"]["counts"]["executed"] = False
        self.assertEqual(zero_test_symptoms(observations), [])
        observations["runs"][0]["summary"]["counts"]["executed"] = 0
        observations["runs"][0]["result"] = None
        self.assertEqual(zero_test_symptoms(observations), [])

    def test_collection_only_and_compound_fake_summary_do_not_fire(self):
        for command in ("pytest --collect-only", 'pytest; echo "no tests ran in 0.2s"'):
            self.assertFalse(
                any(
                    s["kind"] in ("zero_tests_executed", "all_tests_skipped")
                    for s in analyze(traj(call(command=command), result(out="no tests ran in 0.2s")))["symptoms"]
                )
            )
