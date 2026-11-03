import unittest
from agent_trajviz.awe._engine.symptoms import analyze
from agent_trajviz.awe._engine.trajectory import NormalizedStep as S, SEARCH
from .test_verification_observations import call, result, traj


def message(text, n=2, thread=None, **kw):
    return S(n, n, "agent", "message", text=text, subagent_id=thread, **kw)


def hidden_result():
    r = result(code=1)
    r.observability = "hidden"
    return r


def error_result():
    r = result(code=0)
    r.tool_error = True
    return r


def rows(t):
    return [r for r in analyze(t)["symptoms"] if r["kind"] == "tool_output_contradiction"]


class ToolContradictionTests(unittest.TestCase):
    def test_failed_command_claims_success(self):
        r = rows(traj(call(), result(code=2), message("The last command succeeded.")))
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["claim"]["position"], 2)
        self.assertEqual(r[0]["scoring"], "not_scored")

    def test_failure_and_test_claims(self):
        self.assertEqual(len(rows(traj(call(), result(), message("The last command failed.")))), 1)
        self.assertEqual(
            len(rows(traj(call(), result(code=1, out="2 failed in 0.2s"), message("The last test run passed.")))), 1
        )
        self.assertEqual(rows(traj(call(), result(code=1, out="unknown"), message("The last test run passed."))), [])

    def test_search_is_bounded_and_action_specific(self):
        for output, claim in [
            ("Found 3 matches.", "The last search found no matches."),
            ("No matches found.", "The last search found matches."),
        ]:
            c = call()
            c.canonical_action = SEARCH
            c.test_run = False
            self.assertEqual(len(rows(traj(c, result(out=output), message(claim)))), 1)
            self.assertEqual(rows(traj(call(), result(out=output), message(claim))), [])
        c = call()
        c.canonical_action = SEARCH
        self.assertEqual(
            rows(traj(c, result(out="source: Found 3 matches."), message("The last search found no matches."))), []
        )

    def test_quotations_questions_conditionals_and_hidden_text_abstain(self):
        for text in (
            '"The last command succeeded."',
            "The last command succeeded?",
            "If the last command succeeded.",
            "```\nThe last command succeeded.\n```",
            "> The last command succeeded.",
            "Example:\nThe last command succeeded.",
        ):
            self.assertEqual(rows(traj(call(), result(code=1), message(text))), [])
        self.assertEqual(
            rows(traj(call(), result(code=1), message("The last command succeeded.", observability="hidden"))), []
        )

    def test_missing_hidden_duplicate_and_parallel_results_abstain(self):
        claim = message("The last command succeeded.", n=4)
        for t in (
            traj(call(), claim),
            traj(call(), hidden_result(), claim),
            traj(call(), call(1), result(2, code=1), claim),
            traj(call(cid="a"), call(1, cid="b"), result(2, cid="a", code=1), claim),
        ):
            self.assertEqual(rows(t), [])

    def test_new_action_or_unrelated_message_ends_claim_window(self):
        for step in (call(2, cid="other"), message("I will investigate.", n=2)):
            self.assertEqual(rows(traj(call(), result(code=1), step, message("The last command succeeded.", n=3))), [])

    def test_thread_scope_and_boolean_exit_codes(self):
        self.assertEqual(rows(traj(call(), result(code=1), message("The last command succeeded.", thread="s"))), [])
        self.assertEqual(rows(traj(call(), result(code=True), message("The last command succeeded."))), [])
        self.assertEqual(rows(traj(call(), error_result(), message("The last command failed."))), [])

    def test_consistent_claims_and_failed_tests_with_echo_do_not_fire(self):
        self.assertEqual(rows(traj(call(), result(), message("The last command succeeded."))), [])
        self.assertEqual(
            rows(
                traj(
                    call(command="pytest; echo complete"),
                    result(out="2 failed in 0.2s"),
                    message("The last test run passed."),
                )
            ),
            [],
        )
