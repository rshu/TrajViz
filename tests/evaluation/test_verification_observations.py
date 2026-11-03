import unittest
from agent_trajviz.awe._engine.verification import (
    verification_observations,
    tool_observations,
    test_summary as _test_summary,
)
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, NormalizedStep as S, RUN_TEST


def call(n=0, cid="t", command="pytest", thread=None, **kw):
    return S(
        n,
        n,
        "agent",
        "tool_call",
        canonical_action=RUN_TEST,
        test_run=True,
        call_id=cid,
        subagent_id=thread,
        args={"command": command, **kw},
    )


def result(n=1, cid="t", out="1 passed in 0.1s", thread=None, code=0, **kw):
    return S(
        n, n, "tool", "tool_result", call_id=cid, subagent_id=thread, args={"exit_code": code, **kw}, output_excerpt=out
    )


def traj(*steps):
    return NormalizedTrajectory("a", "i", list(steps))


class VerificationTests(unittest.TestCase):
    def test_known_zero_and_unknown_are_distinct(self):
        self.assertEqual(_test_summary("no tests ran in 0.2s")["counts"]["executed"], 0)
        self.assertEqual(_test_summary("Ran 0 tests in 0.001s\n\nOK")["counts"]["executed"], 0)
        self.assertEqual(_test_summary("Process completed")["status"], "unknown")

    def test_skips_xfails_and_failures(self):
        s = _test_summary("=== 2 passed, 1 failed, 3 skipped, 1 xfailed in 0.2s ===")
        self.assertEqual(s["counts"]["executed"], 4)
        self.assertEqual(s["counts"]["skipped"], 3)
        self.assertEqual(_test_summary("Ran 4 tests in 0.001s\nOK (skipped=4)")["counts"]["executed"], 0)

    def test_collection_and_import_errors_abstain(self):
        self.assertEqual(_test_summary("1 error in 0.2s")["counts"]["executed"], None)
        self.assertEqual(_test_summary("Ran 1 test in 0.001s\nFAILED (errors=1)")["counts"]["executed"], None)

    def test_conflicting_and_multiple_summaries(self):
        for text in (
            "1 passed in 0.1s\nno tests ran in 0.2s",
            "Ran 1 test in 0.1s\nOK (skipped=2)",
            "Ran 1 test in 0.1s\nOK (failures=1)",
            "1 passed, 2 passed in 0.1s",
        ):
            self.assertEqual(_test_summary(text)["status"], "unknown", text)

    def test_thread_scoping_and_sequential_id_reuse(self):
        t = traj(
            call(),
            call(1, thread="sub"),
            result(2, thread="sub", out="no tests ran in 0.2s"),
            result(3),
            call(4),
            result(5, out="2 passed in 0.2s"),
        )
        runs = verification_observations(t)["runs"]
        self.assertEqual([r["summary"]["counts"]["executed"] for r in runs], [1, 0, 2])

    def test_parallel_id_reuse_and_future_results_do_not_get_guessed(self):
        rows = tool_observations(traj(call(), call(1), result(2), result(3)))
        self.assertEqual([r[-1] for r in rows], ["ambiguous_call_id"] * 2)
        rows = tool_observations(traj(result(0), call(1)))
        self.assertIsNone(rows[0][3])

    def test_idless_calls_require_adjacency(self):
        self.assertEqual(tool_observations(traj(call(cid=None), result(cid=None)))[0][-1], "adjacent_result")
        mid = S(1, 1, "agent", "message", text="Pending")
        self.assertIsNone(tool_observations(traj(call(cid=None), mid, result(2, cid=None)))[0][3])
        rows = tool_observations(traj(call(cid=None), call(1, cid=None), result(2, cid=None)))
        self.assertIsNone(rows[0][3])
        self.assertIsNone(rows[1][3])
        self.assertEqual(rows[0][-1], "ambiguous_anonymous_calls")

    def test_hidden_results_and_missing_runs_are_unknown(self):
        r = result()
        r.observability = "hidden"
        self.assertEqual(verification_observations(traj(call(), r))["runs"][0]["summary"]["status"], "unknown")
        self.assertFalse(verification_observations(None)["available"])

    def test_compound_outputs_and_no_run_flags(self):
        for cmd in (
            'pytest; echo "no tests ran in 0.1s"',
            "pytest && pytest other",
            "cat log | pytest",
            "tox",
            "make test",
        ):
            r = verification_observations(traj(call(command=cmd), result()))["runs"][0]
            self.assertEqual(r["summary"]["status"], "unknown", cmd)
        self.assertEqual(verification_observations(traj(call(command="pytest --collect-only"), result()))["runs"], [])
        self.assertEqual(
            verification_observations(traj(call(command="cd repo && pytest"), result()))["runs"][0]["summary"][
                "status"
            ],
            "reported",
        )

    def test_recorded_state_keeps_provenance_and_conflicts(self):
        t = traj(
            call(verification_state={"revision": "abc", "secret": "omit"}),
            result(verification_state={"revision": "def"}),
        )
        state = verification_observations(t)["runs"][0]["state"]
        self.assertEqual(
            state["records"],
            [{"source": "call", "fields": {"revision": "abc"}}, {"source": "result", "fields": {"revision": "def"}}],
        )
        self.assertEqual(
            verification_observations(traj(call(), result()))["runs"][0]["state"]["status"], "not_recorded"
        )

    def test_boolean_exit_code_is_unknown(self):
        r = verification_observations(traj(call(), result(code=True)))["runs"][0]
        self.assertIsNone(r["exit_code"])

    def test_warnings_alone_are_not_zero_execution(self):
        self.assertEqual(_test_summary("1 warning in 0.2s")["status"], "unknown")
