"""build_blame records WHICH non-deductive primary the arbiter touched and how —
for both refute (demote) and confirm (corroborate). Pure: synthetic detection +
a fake adjudicate hook.

Run:  python3 -m unittest tests.test_arbiter_provenance -v
"""

import unittest

from agent_trajviz.awe._engine.rootcause import build_blame


def _detection():
    # a single NON-deductive primary (relevant_change_omitted) -> arbiter-eligible
    return {
        "resolved": False,
        "agent": "a",
        "instance_id": "i",
        "loc_flag": "right_place_wrong_change",
        "candidates": [
            {
                "capability": "code_editing",
                "error_type": "relevant_change_omitted",
                "confidence": "high",
                "evidence": {},
            }
        ],
    }


class TestArbiterProvenance(unittest.TestCase):
    def test_refute_demotes_and_records(self):
        def adj(det, cand):
            return {"support": "refute", "confidence": "high", "evidence_steps": [3], "rationale": "handled fine"}

        b = build_blame(_detection(), adjudicate=adj)
        self.assertEqual(b["status"], "unattributed")
        self.assertIsNone(b["primary"])
        self.assertEqual(b["arbiter"]["applied"], "unattributed_after_refute")
        self.assertEqual(b["arbiter"]["target"]["error_type"], "relevant_change_omitted")
        self.assertEqual(b["arbiter"]["support"], "refute")

    def test_confirm_corroborates_and_records(self):
        def adj(det, cand):
            return {"support": "confirm", "confidence": "high", "evidence_steps": [2], "rationale": "yes"}

        b = build_blame(_detection(), adjudicate=adj)
        self.assertEqual(b["status"], "primary")
        self.assertTrue(b["primary"]["arbiter"]["corroborated"])
        self.assertEqual(b["arbiter"]["applied"], "corroborated")

    def test_no_arbiter_key_when_not_consulted(self):
        # no adjudicate hook -> no arbiter provenance, status unchanged
        b = build_blame(_detection())
        self.assertNotIn("arbiter", b)
        self.assertEqual(b["status"], "primary")


class TestRefuteMovesMass(unittest.TestCase):
    """v0.32: an applied refute zeroes the refuted fault's weight and
    renormalizes — the scores can no longer count blame the arbiter rejected."""

    _REFUTE = staticmethod(
        lambda det, cand: {
            "support": "refute",
            "confidence": "high",
            "evidence_steps": [3],
            "rationale": "handled fine",
        }
    )

    def test_refuted_singleton_carries_zero_mass(self):
        b = build_blame(_detection(), adjudicate=self._REFUTE)
        self.assertEqual(b["status"], "unattributed")
        refuted = next(x for x in b["blame"] if x["error_type"] == "relevant_change_omitted")
        self.assertEqual(refuted["weight"], 0.0)
        self.assertTrue(refuted["arbiter_refuted"])
        self.assertEqual(b["capability_blame"].get("code_editing", 0.0), 0.0)

    def test_refute_promotes_deductive_runner_up(self):
        det = {
            "resolved": False,
            "agent": "a",
            "instance_id": "i",
            "loc_flag": "loc_partial",
            "candidates": [
                {
                    "capability": "code_editing",
                    "error_type": "relevant_change_omitted",
                    "confidence": "high",
                    "evidence": {},
                },
                {"capability": "code_editing", "error_type": "incorrect_patch", "confidence": "high", "evidence": {}},
            ],
        }
        # loc_partial -> primary cap is code_localization, absent from candidates
        # -> both editing faults share the capability; the top-weight one
        # (relevant_change_omitted, interpretive) is the arbiter target. After
        # its refute, incorrect_patch (deductive) holds all surviving editing
        # mass and is promoted.
        b = build_blame(det, adjudicate=self._REFUTE)
        self.assertEqual(b["status"], "primary")
        self.assertEqual(b["arbiter"]["applied"], "refuted_reassigned")
        self.assertEqual(b["primary"]["error_type"], "incorrect_patch")
        self.assertTrue(b["primary"]["promoted_after_refute"])
        self.assertAlmostEqual(b["primary"]["weight"], 1.0)

    def test_refute_does_not_promote_interpretive_runner_up(self):
        det = {
            "resolved": False,
            "agent": "a",
            "instance_id": "i",
            "loc_flag": "loc_partial",
            "candidates": [
                {
                    "capability": "code_editing",
                    "error_type": "relevant_change_omitted",
                    "confidence": "high",
                    "evidence": {},
                },
                {"capability": "code_editing", "error_type": "gave_up_early", "confidence": "high", "evidence": {}},
            ],
        }
        b = build_blame(det, adjudicate=self._REFUTE)
        # survivor (gave_up_early) is itself interpretive -> no promotion
        self.assertEqual(b["status"], "conjunctive")
        self.assertEqual(b["arbiter"]["applied"], "demoted_to_conjunctive")


class TestCausalOnlyDominance(unittest.TestCase):
    """v0.32: contributing (verification/self-repair/tool) mass no longer
    dilutes the dominance test — one causal fault vs five symptoms is a
    primary, not 'no dominant cause'."""

    def test_lone_causal_fault_dominates_despite_symptom_count(self):
        cands = [{"capability": "code_editing", "error_type": "incorrect_patch", "confidence": "high", "evidence": {}}]
        for et in (
            "relevant_test_not_run",
            "no_test_after_final_edit",
            "unsupported_success_claim",
            "retry_without_change",
            "repair_not_reverified",
        ):
            cap = "self_repair_loop" if et in ("retry_without_change", "repair_not_reverified") else "code_verification"
            cands.append({"capability": cap, "error_type": et, "confidence": "medium", "evidence": {}})
        b = build_blame({"resolved": False, "loc_flag": "right_place_wrong_change", "candidates": cands})
        self.assertEqual(b["status"], "primary")
        self.assertEqual(b["primary"]["error_type"], "incorrect_patch")

    def test_contributing_only_case_stays_conjunctive(self):
        b = build_blame(
            {
                "resolved": False,
                "loc_flag": None,
                "candidates": [
                    {
                        "capability": "code_verification",
                        "error_type": "relevant_test_not_run",
                        "confidence": "high",
                        "evidence": {},
                    }
                ],
            }
        )
        self.assertEqual(b["status"], "conjunctive")
        self.assertIsNone(b["primary"])

    def test_two_competing_causes_still_conjunctive(self):
        b = build_blame(
            {
                "resolved": False,
                "loc_flag": "loc_partial",
                "candidates": [
                    {
                        "capability": "code_localization",
                        "error_type": "missed_relevant_file",
                        "confidence": "high",
                        "evidence": {},
                    },
                    {
                        "capability": "code_editing",
                        "error_type": "incorrect_patch",
                        "confidence": "high",
                        "evidence": {},
                    },
                ],
            }
        )
        # loc is upstream-primary role (1.0) vs editing secondary (0.4):
        # shares 0.71/0.29 -> dominant. Force a genuine tie instead:
        b2 = build_blame(
            {
                "resolved": False,
                "loc_flag": "loc_partial",
                "candidates": [
                    {
                        "capability": "code_localization",
                        "error_type": "missed_relevant_file",
                        "confidence": "medium",
                        "evidence": {},
                    },
                    {
                        "capability": "code_editing",
                        "error_type": "incorrect_patch",
                        "confidence": "high",
                        "evidence": {},
                    },
                    {
                        "capability": "code_editing",
                        "error_type": "relevant_change_omitted",
                        "confidence": "high",
                        "evidence": {},
                    },
                ],
            }
        )
        # loc: 1.0*0.6=0.6; editing: 0.4+0.4=0.8 -> shares 0.43/0.57:
        # neither clears 0.5 with 0.2 margin -> conjunctive
        self.assertEqual(b2["status"], "conjunctive")
        self.assertEqual(b["status"], "primary")  # sanity: the upstream case


if __name__ == "__main__":
    unittest.main()


class TestPanelAggregatorGuards(unittest.TestCase):
    """v0.32 latent-hole fixes: a plurality label without a strict majority of
    the panel cannot carry blame-eligible confidence; the arbiter's
    representative vote must be a cited one."""

    def test_judge_plurality_label_capped_at_low(self):
        from agent_trajviz.awe._engine.judge import aggregate_verdicts

        def v(et, conf, steps):
            return {
                "requirement_understanding": {
                    "error_type": et,
                    "confidence": conf,
                    "evidence_steps": steps,
                    "rationale": "r",
                },
                "task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": [], "rationale": ""},
            }

        # 3/5 say SOME error (strict majority) but the labels split 1/1/1:
        # winning label holds 20% agreement -> confidence must cap at low.
        out = aggregate_verdicts(
            [
                v("misread_requirement", "high", [1]),
                v("missed_constraint", "low", [2]),
                v("scope_misjudged", "low", [3]),
                v("none", "high", []),
                v("none", "medium", []),
            ]
        )
        ru = out["requirement_understanding"]
        self.assertNotEqual(ru["error_type"], "none")
        self.assertEqual(ru["confidence"], "low")
        self.assertLessEqual(ru["agreement"], 0.2 + 1e-9)

    def test_judge_majority_label_keeps_confidence(self):
        from agent_trajviz.awe._engine.judge import aggregate_verdicts

        def v(et, conf, steps):
            return {
                "requirement_understanding": {
                    "error_type": et,
                    "confidence": conf,
                    "evidence_steps": steps,
                    "rationale": "r",
                },
                "task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": [], "rationale": ""},
            }

        out = aggregate_verdicts(
            [
                v("misread_requirement", "high", [1]),
                v("misread_requirement", "high", [2]),
                v("misread_requirement", "medium", [3]),
                v("none", "high", []),
                v("none", "medium", []),
            ]
        )
        ru = out["requirement_understanding"]
        self.assertEqual(ru["error_type"], "misread_requirement")
        self.assertEqual(ru["confidence"], "high")  # lower median of 3 votes

    def test_arbiter_rep_prefers_cited_vote(self):
        from agent_trajviz.awe._engine.arbiter import aggregate_arbiter

        out = aggregate_arbiter(
            [
                {"support": "refute", "confidence": "high", "evidence_steps": [], "rationale": "uncited"},
                {"support": "refute", "confidence": "medium", "evidence_steps": [7], "rationale": "cited"},
                {"support": "refute", "confidence": "medium", "evidence_steps": [9], "rationale": "cited2"},
            ]
        )
        self.assertEqual(out["support"], "refute")
        self.assertTrue(out["evidence_steps"])  # cited rep chosen
        self.assertEqual(out["confidence"], "medium")  # panel lower median
