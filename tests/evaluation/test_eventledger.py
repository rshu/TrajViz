import unittest
from agent_trajviz.awe._engine.eventledger import event_ledger
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, READ, RUN_COMMAND, RUN_TEST
from .test_review_fixes_trajectory import _call, _result


class TestEventLedger(unittest.TestCase):
    def ledger(self, steps):
        return event_ledger(NormalizedTrajectory("a", "i", steps))

    def test_repeated_failures_group_until_later_success(self):
        steps = [
            _call(0, "a", RUN_TEST, test=True, cmd="pytest tests/test_x.py"),
            _result(1, "a", args={"exit_code": 1}),
            _call(2, "b", RUN_TEST, test=True, cmd="pytest tests/test_x.py"),
            _result(3, "b", args={"exit_code": 1}),
            _call(4, "c", RUN_TEST, test=True, cmd="pytest tests/test_x.py"),
            _result(5, "c", args={"exit_code": 0}),
        ]
        ledger = self.ledger(steps)
        self.assertEqual(len(ledger["events"]), 1)
        event = ledger["events"][0]
        self.assertEqual(len(event["failures"]), 2)
        self.assertEqual(event["lifecycle"], "later_success_observed")
        self.assertEqual(event["recovery"]["call"]["step_id"], 4)
        self.assertEqual(event["terminal_relevance"], "unknown")
        self.assertEqual(event["causal_validation"], "not_tested")

    def test_other_target_or_thread_does_not_resolve_failure(self):
        ledger = self.ledger(
            [
                _call(0, "same", READ, files=["pkg/a.py"]),
                _result(1, "same", error=True),
                _call(2, "other", READ, files=["pkg/b.py"]),
                _result(3, "other"),
                _call(4, "same", READ, files=["pkg/a.py"], sub="child"),
                _result(5, "same", sub="child"),
            ]
        )
        self.assertEqual(ledger["events"][0]["lifecycle"], "no_recovery_observed")
        self.assertIsNone(ledger["events"][0]["recovery"])

    def test_missing_followup_is_unknown(self):
        ledger = self.ledger(
            [
                _call(0, "a", READ, files=["pkg/a.py"]),
                _result(1, "a", error=True),
                _call(2, "b", READ, files=["pkg/a.py"]),
            ]
        )
        self.assertEqual(ledger["missing_result_calls"], 1)
        self.assertEqual(ledger["events"][0]["unobserved_followups"], 1)

    def test_read_content_is_not_a_tool_failure(self):
        ledger = self.ledger([_call(0, "a", READ, files=["pkg/a.py"]), _result(1, "a", "No such file or directory")])
        self.assertEqual(ledger["events"], [])

    def test_parallel_success_cannot_be_response_to_later_failure(self):
        ledger = self.ledger(
            [
                _call(0, "a", RUN_COMMAND, cmd="cmd"),
                _call(1, "b", RUN_COMMAND, cmd="cmd"),
                _result(2, "b", args={"exit_code": 0}),
                _result(3, "a", args={"exit_code": 1}),
            ]
        )
        self.assertEqual(ledger["events"][0]["lifecycle"], "no_recovery_observed")

    def test_missing_trajectory_has_no_observation_claim(self):
        self.assertFalse(event_ledger(None)["available"])
