import unittest
from agent_trajviz.awe._engine.preflight import _diff
from agent_trajviz.awe._engine.diffutil import parse_unified_diff
from agent_trajviz.awe._engine.patches import patch_from_text
from agent_trajviz.awe._engine.outcomes import parse_pertest
from agent_trajviz.awe._engine.detect import objective_candidates
from .test_pr56_review_fixes import _gold, _patch, _report, X, Y, KEEP
from agent_trajviz.awe._engine.signals import localization_signals


class DiagnosticConsistency(unittest.TestCase):
    def test_failed_all_pass_record_has_unknown_cause(self):
        raw = _report([X, Y], [])
        raw["i"]["resolved"] = False
        pt = parse_pertest(raw, "i")
        self.assertEqual(pt["shape"], "unexplained_failure")
        self.assertTrue(pt["outcome_conflict"])
        g = _gold(["pkg/a.py"])
        loc = localization_signals(g, _patch(), False, read_set=set())
        self.assertEqual(objective_candidates(g, loc, None, False, pertest=pt), [])

    def test_p2p_only_retains_regression(self):
        pt = parse_pertest(_report([], [], p2p_bad=[KEEP]), "i")
        g = _gold(["pkg/a.py"])
        loc = localization_signals(g, _patch(), False)
        self.assertEqual(
            {c.error_type for c in objective_candidates(g, loc, None, False, pertest=pt)}, {"regression_introduced"}
        )

    def test_trailing_payload_cannot_inflate_overlap(self):
        text = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-return 0\n+return 2\n+return 1\n"
        fd = parse_unified_diff(text)[0]
        self.assertEqual(fd.added_lines, ["return 2"])
        self.assertEqual(fd.unbounded_lines, 1)
        self.assertTrue(patch_from_text("a", "i", text).unparseable)
        with self.assertRaises(ValueError):
            _diff(text)

    def test_legacy_hunkless_fixture_requires_preflight(self):
        with self.assertRaises(ValueError):
            _diff("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n-x\n+y\n")
