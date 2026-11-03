"""Blame-aware prescription surfacing (Finding B) + single-source dedup (Finding D).

These cover the pure presentation helpers; the full case_record/markdown path needs
the ARGUS data tree and is exercised by gen_dossier + the showcase regeneration."""

import unittest

from agent_trajviz.awe._engine import dossier


class TestBlameAwareSurfacing(unittest.TestCase):
    def test_fix_label_gates_emphasis_by_role(self):
        # the primary cause gets the authoritative fix; a contributing fault is flagged
        self.assertEqual(dossier.fix_label(True), "Recommended fix (primary cause)")
        self.assertEqual(dossier.fix_label(False), "Secondary-fault fix")
        self.assertNotEqual(dossier.fix_label(True), dossier.fix_label(False))

    def test_blame_status_note_makes_conjunctive_explicit(self):
        # a refuted/no-dominant case must read as conjunctive, not as a confident primary
        self.assertIn("conjunctive", dossier.blame_status_note("conjunctive"))
        self.assertIn("single dominant", dossier.blame_status_note("primary"))
        self.assertIn("unattributed", dossier.blame_status_note("unattributed"))

    def test_blame_status_note_passthrough_unknown(self):
        self.assertEqual(dossier.blame_status_note("weird"), "weird")


class TestSingleSourceDedup(unittest.TestCase):
    def test_report_and_dossier_drop_deprecated_recommend_shim(self):
        # Finding D: the surfaces no longer import the deprecated recommendations shim;
        # strategies are derived from the same catalog entries they render.
        import agent_trajviz.awe._engine.report as report
        import agent_trajviz.awe._engine.dossier as dsr

        self.assertFalse(hasattr(report, "recommend"), "report.py should no longer import the recommend shim")
        self.assertFalse(hasattr(dsr, "recommend"), "dossier.py should no longer import the recommend shim")


if __name__ == "__main__":
    unittest.main()
