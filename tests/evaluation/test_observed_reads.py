import unittest
from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, READ, RUN_COMMAND, read_observations
from agent_trajviz.awe._engine.trajsignals import compute
from agent_trajviz.awe._engine.signals import localization_signals
from .test_review_fixes_trajectory import _call, _result, _gold


class TestObservedReads(unittest.TestCase):
    def test_failed_read_is_not_reached(self):
        t = NormalizedTrajectory(
            "a", "i", [_call(0, "r", READ, files=["pkg/a.py"]), _result(1, "r", "No such file", error=True)]
        )
        sig = compute(t, _gold())
        self.assertEqual(sig.read_set, [])
        self.assertEqual(sig.attempted_read_set, ["pkg/a.py"])
        loc = localization_signals(_gold(), None, False, sig.read_set)
        self.assertEqual(loc["missed_unreached"], ["pkg/a.py"])
        self.assertEqual(audit._read_set(t), set())

    def test_unanswered_read_is_unknown_not_localization_miss(self):
        t = NormalizedTrajectory("a", "i", [_call(0, "r", READ, files=["pkg/a.py"])])
        sig = compute(t, _gold())
        loc = localization_signals(_gold(), None, False, sig.read_set, sig.unobserved_read_set)
        self.assertEqual(loc["missed_read_unknown"], ["pkg/a.py"])
        self.assertEqual(loc["missed_unreached"], [])
        self.assertEqual(loc["missed_reached"], [])

    def test_successful_later_read_and_content_error_words(self):
        t = NormalizedTrajectory(
            "a",
            "i",
            [
                _call(0, "r", READ, files=["pkg/a.py"]),
                _call(1, "s", RUN_COMMAND, files=["pkg/a.py"]),
                _result(2, "s", "No such file in source content", args={"exit_code": 0}),
            ],
        )
        self.assertEqual(read_observations(t), ({"pkg/a.py"}, {"pkg/a.py"}, set()))
        self.assertEqual(audit._read_set(t), {"pkg/a.py"})
