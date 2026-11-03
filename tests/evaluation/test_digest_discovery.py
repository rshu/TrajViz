"""Unit tests for the INSIGHTS discovery section (pure renderer, no ARGUS)."""

import unittest

from agent_trajviz.awe._engine.digest import _discovery_section, _discovery_block


def _disc():
    # SYNTHETIC fixture exercising the renderer paths (incl. the step_repetition
    # callout). Not corpus values: on the real post-detector-fix data
    # step_repetition is ns (pooled RR=1.22, Holm p≈0.81) and would be gated out.
    # Never label per-agent numbers with pooled cohort ns (the 2026-07-12 lesson).
    return {
        "cohort": {
            "n_failed": 685,
            "n_solved": 1815,
            "associations": [
                {
                    "feature": "step_repetition",
                    "binary": True,
                    "diff": 0.15,
                    "mean_failed": 0.68,
                    "mean_solved": 0.53,
                    "risk_ratio": 1.60,
                    "odds_ratio": 1.9,
                    "p_holm": 0.027,
                    "n_failed": 685,
                    "n_solved": 1815,
                },
                {
                    "feature": "n_tool_calls",
                    "binary": False,
                    "diff": 5.85,
                    "mean_failed": 32.84,
                    "mean_solved": 26.99,
                    "p_holm": 0.027,
                    "n_failed": 685,
                    "n_solved": 1815,
                },
                {
                    "feature": "noise",
                    "binary": True,
                    "diff": 0.01,
                    "mean_failed": 0.5,
                    "mean_solved": 0.49,
                    "risk_ratio": 1.02,
                    "odds_ratio": 1.0,
                    "p_holm": 0.90,
                    "n_failed": 685,
                    "n_solved": 1815,
                },
            ],
        },
        "fragile": {"n": 728, "n_resolved": 1815, "n_destructive": 443},
    }


class TestDiscoverySection(unittest.TestCase):
    def test_renders_significant_and_gates_out_nonsignificant(self):
        md = "\n".join(_discovery_section(_disc()))
        self.assertIn("Discovery signals", md)
        self.assertIn("step_repetition", md)
        self.assertIn("RR=1.60", md)  # binary effect
        self.assertIn("Δmean=+5.85", md)  # continuous effect
        self.assertNotIn("noise", md)  # p_holm 0.90 → gated out

    def test_step_repetition_callout(self):
        md = "\n".join(_discovery_section(_disc()))
        self.assertIn("MAST-3", md)
        self.assertIn("1.60×", md)

    def test_rr_worded_as_failure_risk_not_prevalence(self):
        # RR = P(fail|behavior)/P(fail|no behavior) = 1.60. The prevalence ratio
        # is 68/53 ≈ 1.28 — the callout must not describe RR as "do this X× as
        # often", which is the prevalence reading.
        md = "\n".join(_discovery_section(_disc()))
        self.assertIn("fail 1.60× as often", md)
        self.assertNotIn("do this 1.60×", md)
        self.assertIn("68% of failed", md)

    def test_fragile_integrity_line_preserves_uncertainty(self):
        md = "\n".join(_discovery_section(_disc())).lower()
        self.assertIn("integrity", md)
        self.assertIn("728/1,815", md)  # thousands separator
        self.assertIn("depends", md)
        self.assertIn("does not establish gaming", md)

    def test_fragile_line_does_not_assume_reset_semantics(self):
        # verified 2026-07-12 against swebench 4.1.0 make_eval_script_list_py:
        # git checkout <base> -- <test files> precedes git apply <test_patch>
        md = "\n".join(_discovery_section(_disc()))
        self.assertIn("recorded reset policy", md)
        self.assertNotIn("swebench 4.1.0", md)

    def test_top_cap_notes_the_remainder(self):
        d = _disc()
        d["cohort"]["associations"] = [
            {
                "feature": f"f{i}",
                "binary": True,
                "diff": 0.2,
                "mean_failed": 0.6,
                "mean_solved": 0.4,
                "risk_ratio": 1.5,
                "odds_ratio": 1.5,
                "p_holm": 0.01,
                "n_failed": 685,
                "n_solved": 1815,
            }
            for i in range(10)
        ]
        md = "\n".join(_discovery_section(d, top=7))
        self.assertIn("+3 more", md)  # 10 significant, 7 shown
        for i in range(7, 10):  # the omitted rows are NAMED inline
            self.assertIn(f"f{i}", md)  # (no dangling pointer)

    def test_empty_associations_renders_gate_note(self):
        md = "\n".join(_discovery_section({"cohort": {"associations": []}, "fragile": {}}))
        self.assertIn("No behavior is significantly associated", md)


class TestDiscoveryBlockGuard(unittest.TestCase):
    """build_insights_md renders discovery via _discovery_block — the guarded
    wrapper that makes 'a discovery failure never blocks INSIGHTS' cover
    RENDERING, not just computation."""

    def test_none_renders_nothing(self):
        self.assertEqual(_discovery_block(None), [])

    def test_good_blob_renders_section(self):
        md = "\n".join(_discovery_block(_disc()))
        self.assertIn("Discovery signals", md)
        self.assertIn("RR=1.60", md)

    def test_malformed_blob_skips_section_instead_of_raising(self):
        # a schema-drifted blob (missing keys) skips the section, not the build
        bad = {"cohort": {"associations": [{"feature": "x"}]}, "fragile": {"n": 1}}
        self.assertEqual(_discovery_block(bad), [])

    def test_build_insights_md_uses_the_guarded_block(self):
        import inspect
        from agent_trajviz.awe._engine import digest

        src = inspect.getsource(digest.build_insights_md)
        self.assertIn("_discovery_block(discovery)", src)
        self.assertNotIn("_discovery_section(discovery)", src)


class TestFragileScanSkips(unittest.TestCase):
    def test_scan_reports_skipped_cases(self):
        # A systematic loader error must be visible, not a silent false null.
        import agent_trajviz.awe._engine.fragile as fragile
        import agent_trajviz.awe._engine.gold as gold
        import agent_trajviz.awe._engine.outcomes as outcomes

        orig_gold, orig_res = gold.build_gold_reference, outcomes.resolved
        gold.build_gold_reference = lambda inst: (_ for _ in ()).throw(OSError("boom"))
        outcomes.resolved = lambda a, i: True
        try:
            skipped = []
            out = fragile.scan("agent_x", ["i1", "i2"], skipped=skipped)
        finally:
            gold.build_gold_reference, outcomes.resolved = orig_gold, orig_res
        self.assertEqual(out, [])
        self.assertEqual(len(skipped), 2)

    def test_report_md_surfaces_skips(self):
        from agent_trajviz.awe._engine.fragile import report_md

        md = report_md([], n_resolved=100, n_skipped=7)
        self.assertIn("7", md)
        self.assertIn("skipped", md.lower())

    def test_report_md_silent_when_no_skips(self):
        from agent_trajviz.awe._engine.fragile import report_md

        md = report_md([], n_resolved=100)
        self.assertNotIn("skipped", md.lower())


if __name__ == "__main__":
    unittest.main()
