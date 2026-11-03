"""Fact-grounding tests for the diagnosis algorithm.

Every test asserts that a root-cause label is backed by a CONCRETE fact
(membership in the gold-set / edit-set / read-set, or a deductive line-overlap),
not by an unreliable heuristic. Pure-function tests — no ARGUS data needed.

Run:  python3 -m unittest tests.test_grounding -v
"""

from __future__ import annotations

import json
import os
import unittest


from agent_trajviz.awe._engine.gold import GoldReference
from agent_trajviz.awe._engine.patches import AgentPatch
from agent_trajviz.awe._engine import signals
from agent_trajviz.awe._engine.detect import objective_candidates
from agent_trajviz.awe._engine.evidence import build_chain
from agent_trajviz.awe._engine.rootcause import build_blame
from agent_trajviz.awe._engine.trajectory import looks_like_test_failure
from agent_trajviz.awe._engine.trajsignals import _gold_test_tokens


def _cand(cap, et, conf, dim="d"):
    return {"capability": cap, "dimension": dim, "error_type": et, "confidence": conf, "evidence": {}}


def gold(files, symbols=(), cross=False, ftp=("pkg/tests/test_x.py::test_x",), loc=10):
    mods = sorted({os.path.dirname(f) for f in files})
    return GoldReference(
        instance_id="i",
        repo="r",
        base_commit="c",
        problem_statement="p",
        gold_files=list(files),
        gold_symbols=list(symbols),
        gold_patch_loc=loc,
        test_patch_files=[],
        fail_to_pass=list(ftp),
        pass_to_pass=[],
        is_multi_file=len(set(files)) > 1,
        is_cross_module=cross or len(mods) > 1,
        modules=mods,
    )


def patch(src=(), loc=10, symbols=(), exists=True):
    src = list(src)
    return AgentPatch(
        agent="a",
        instance_id="i",
        exists=exists,
        edited_files=src,
        edited_source_files=src,
        edited_test_files=[],
        patch_loc=loc,
        symbols=list(symbols),
    )


class TestReadSetGroundsLocalization(unittest.TestCase):
    """A 'localization miss' must mean the gold code was never REACHED.
    A gold file the agent read but did not edit is reached, not missed."""

    def test_missed_gold_file_that_was_read_is_classified_reached_not_unreached(self):
        g = gold(["pkg/a.py"])
        p = patch(["pkg/other.py"])  # edited a non-gold file; gold file not edited
        loc = signals.localization_signals(g, p, resolved=False, read_set={"pkg/a.py"})
        self.assertEqual(loc["missed_reached"], ["pkg/a.py"])
        self.assertEqual(loc["missed_unreached"], [])

    def test_missed_gold_file_never_read_is_unreached(self):
        g = gold(["pkg/a.py"])
        p = patch(["pkg/other.py"])
        loc = signals.localization_signals(g, p, resolved=False, read_set={"pkg/unrelated.py"})
        self.assertEqual(loc["missed_unreached"], ["pkg/a.py"])
        self.assertEqual(loc["missed_reached"], [])

    def test_no_read_set_abstains_from_reachability(self):
        # Missing telemetry cannot establish a search failure.
        g = gold(["pkg/a.py"])
        p = patch(["pkg/other.py"])
        loc = signals.localization_signals(g, p, resolved=False, read_set=None)
        self.assertEqual(loc["missed_unreached"], [])
        self.assertEqual(loc["missed_read_unknown"], ["pkg/a.py"])
        self.assertEqual(loc["missed_reached"], [])


class TestCandidatesGroundedByReadSet(unittest.TestCase):
    """detect's objective candidate logic must attribute by the read/edit/gold
    facts: localization only when a gold file was never reached."""

    def _caps(self, cands):
        return {(c.capability, c.error_type) for c in cands}

    def test_read_but_unedited_gold_is_editing_omission_not_localization(self):
        g = gold(["pkg/a.py"])
        loc = signals.localization_signals(g, patch(["pkg/b.py"]), False, read_set={"pkg/a.py"})
        caps = self._caps(objective_candidates(g, loc, None, False))
        self.assertIn(("code_editing", "relevant_change_omitted"), caps)
        self.assertNotIn("code_localization", {c for c, _ in caps})

    def test_unread_unedited_gold_is_a_localization_miss(self):
        g = gold(["pkg/a.py"])
        loc = signals.localization_signals(g, patch(["pkg/b.py"]), False, read_set={"pkg/zzz.py"})
        caps = self._caps(objective_candidates(g, loc, None, False))
        self.assertIn(("code_localization", "missed_relevant_file"), caps)
        self.assertNotIn(("code_editing", "relevant_change_omitted"), caps)

    def test_no_trajectory_does_not_infer_search_failure(self):
        g = gold(["pkg/a.py"])
        loc = signals.localization_signals(g, patch(["pkg/b.py"]), False, read_set=None)
        self.assertEqual(self._caps(objective_candidates(g, loc, None, False)), set())
        self.assertEqual(loc["missed_read_unknown"], ["pkg/a.py"])


class TestNoEditingToLocalizationInversion(unittest.TestCase):
    """When the agent edited every gold file (recall 1.0), the failure is an
    editing fault — a noisy def/class symbol mismatch must not raise a
    localization fault that outranks it (the 82-case inversion)."""

    def test_right_files_wrong_change_emits_only_incorrect_patch(self):
        g = gold(["pkg/a.py"], symbols=["_convert_data_to_col"])
        p = patch(["pkg/a.py"], symbols=["Table", "test_ndarray_mixin"])  # disjoint symbols
        loc = signals.localization_signals(g, p, resolved=False, read_set={"pkg/a.py"})
        self.assertEqual(loc["flag"], "right_place_wrong_change")
        caps = {(c.capability, c.error_type) for c in objective_candidates(g, loc, None, False)}
        self.assertIn(("code_editing", "incorrect_patch"), caps)
        self.assertNotIn(("code_localization", "missed_relevant_function"), caps)


class TestOmissionEvidenceChain(unittest.TestCase):
    """The new editing-omission label must be auditable: its chain has to cite the
    concrete fact (gold file read but not edited), not be an empty stub."""

    def test_relevant_change_omitted_chain_cites_read_not_edited_fact(self):
        g = gold(["pkg/a.py"])
        cand = {
            "capability": "code_editing",
            "dimension": "patch_correctness",
            "error_type": "relevant_change_omitted",
            "confidence": "high",
            "evidence": {"read_not_edited": ["pkg/a.py"]},
        }
        chain = build_chain(cand, {"signals": {}}, g, {}, {}, 0.7, True)
        self.assertTrue(chain["links"], "evidence chain must not be empty")
        joined = " ".join(evidence_link["statement"] for evidence_link in chain["links"]).lower()
        self.assertIn("pkg/a.py", joined)
        self.assertIn("read", joined)
        self.assertTrue(chain["counterfactual"]["what_would_confirm"])


class TestJudgeEvidenceQuotes(unittest.TestCase):
    """IG idea 4: a judge fault's chain attaches the VERBATIM trajectory text for the
    steps it cites (trace-quote + mechanism) when a step_text map is supplied; absent
    the map it stays quote-free (backward compatible)."""

    def test_judge_chain_attaches_verbatim_step_quotes(self):
        g = gold(["pkg/a.py"])
        cand = {
            "capability": "requirement_understanding",
            "dimension": "d",
            "error_type": "misread_requirement",
            "confidence": "high",
            "evidence": {
                "evidence_steps": [2, 5],
                "judge_rationale": "misread the constraint",
                "judge_model": "deepseek",
            },
        }
        step_text = {2: "I think we just need to handle the None case", 5: "Done, this should fix it"}
        chain = build_chain(cand, {"signals": {}}, g, {}, {}, 0.6, True, step_text=step_text)
        quoted = [evidence_link for evidence_link in chain["links"] if evidence_link.get("quotes")]
        self.assertTrue(quoted, "expected a link carrying verbatim quotes")
        texts = {q["step"]: q["text"] for evidence_link in quoted for q in evidence_link["quotes"]}
        self.assertEqual(texts[2], "I think we just need to handle the None case")
        self.assertIn(5, texts)

    def test_no_step_text_is_quote_free(self):
        g = gold(["pkg/a.py"])
        cand = {
            "capability": "task_planning",
            "dimension": "d",
            "error_type": "flawed_plan",
            "confidence": "high",
            "evidence": {"evidence_steps": [1], "judge_rationale": "no plan"},
        }
        chain = build_chain(cand, {"signals": {}}, g, {}, {}, 0.6, True)  # no step_text
        self.assertFalse(any(evidence_link.get("quotes") for evidence_link in chain["links"]))


class TestBlameRespectsEvidenceStrength(unittest.TestCase):
    """v0.32: a model_inferred (LLM judge) capability can NEVER take primary
    blame — its high-confidence tier measured anti-discriminative on solved
    controls (fires 2.5x more often on solved than failed; see
    scripts/judge_flag_rates.py). Judge verdicts contribute weight only."""

    def _det(self, cands, flag="right_place_wrong_change"):
        return {"resolved": False, "loc_flag": flag, "candidates": cands}

    def test_medium_judge_does_not_override_deductive_editing(self):
        b = build_blame(
            self._det(
                [
                    _cand("code_editing", "incorrect_patch", "high"),
                    _cand("requirement_understanding", "misread_requirement", "medium"),
                ]
            )
        )
        self.assertIsNotNone(b["primary"], "expected a primary")
        self.assertEqual(b["primary"]["capability"], "code_editing")

    def test_high_judge_never_takes_primary_over_deductive(self):
        # v0.32: self-reported HIGH confidence from a measured-near-noise layer
        # is no basis to demote a deductive fact — the deduction stays primary.
        b = build_blame(
            self._det(
                [
                    _cand("code_editing", "incorrect_patch", "high"),
                    _cand("requirement_understanding", "misread_requirement", "high"),
                ]
            )
        )
        self.assertIsNotNone(b["primary"])
        self.assertEqual(b["primary"]["capability"], "code_editing")
        # the judge fault still carries weight in the ledger
        self.assertGreater(b["capability_blame"]["requirement_understanding"], 0.0)

    def test_medium_judge_still_contributes_blame_weight(self):
        b = build_blame(
            self._det(
                [
                    _cand("code_editing", "incorrect_patch", "high"),
                    _cand("requirement_understanding", "misread_requirement", "medium"),
                ]
            )
        )
        caps = b["capability_blame"]
        self.assertIn("requirement_understanding", caps)
        self.assertGreater(caps["requirement_understanding"], 0.0)


class TestPartialMissEditingGroundedInOverlap(unittest.TestCase):
    """On a partial miss (edited some gold files, missed others), editing is no
    longer structurally exonerated: whether the edited gold file's change is also
    wrong is decided by gold-vs-agent changed-line overlap, not assumed either way."""

    def _setup(self, agent_added):
        g = gold(["pkg/a.py", "pkg/b.py"])  # same dir -> not cross-module
        g.gold_changed_lines = {"pkg/a.py": {"added": {"return fixed"}, "removed": {"return broken"}}}
        p = patch(["pkg/a.py"])  # edited a.py, missed b.py
        p.changed_lines = {"pkg/a.py": {"added": set(agent_added), "removed": set()}}
        loc = signals.localization_signals(g, p, resolved=False, read_set={"pkg/zzz.py"})
        return g, loc

    def test_divergent_hit_edit_is_also_blamed_as_editing(self):
        g, loc = self._setup(agent_added=["return something_else"])  # 0% overlap with gold
        self.assertEqual(loc["flag"], "loc_partial")
        caps = {(c.capability, c.error_type) for c in objective_candidates(g, loc, None, False)}
        self.assertIn(("code_localization", "missed_relevant_file"), caps)  # b.py truly missed
        self.assertIn(("code_editing", "incorrect_patch"), caps)  # a.py edit diverges

    def test_matching_hit_edit_is_not_blamed_as_editing(self):
        g, loc = self._setup(agent_added=["return fixed"])  # reproduces gold's added line
        caps = {(c.capability, c.error_type) for c in objective_candidates(g, loc, None, False)}
        self.assertIn(("code_localization", "missed_relevant_file"), caps)
        self.assertNotIn(("code_editing", "incorrect_patch"), caps)  # a.py edit matches gold


class TestEditingSeverityIsAttributeNotType(unittest.TestCase):
    """Right-files-but-failing is a SINGLE deductive editing fault (incorrect_patch).
    How much of the reference change the agent reproduced is a graded SEVERITY
    ATTRIBUTE on it (reproduced_fraction) — not an arbitrary categorical split into a
    separate `incomplete_edit` type (the 0.34 cut had no data-grounded boundary)."""

    def _cands(self, gold_added, agent_added):
        g = gold(["pkg/a.py"])
        g.gold_changed_lines = {"pkg/a.py": {"added": set(gold_added), "removed": set()}}
        p = patch(["pkg/a.py"])
        p.changed_lines = {"pkg/a.py": {"added": set(agent_added), "removed": set()}}
        loc = signals.localization_signals(g, p, resolved=False, read_set={"pkg/a.py"})
        assert loc["flag"] == "right_place_wrong_change", loc["flag"]
        return objective_candidates(g, loc, None, False)

    def test_single_type_with_reproduced_fraction_attribute(self):
        cands = self._cands(["x=1", "y=2", "z=3"], ["x=1", "y=2"])  # 2/3 reproduced
        ets = {c.error_type for c in cands}
        self.assertIn("incorrect_patch", ets)
        self.assertNotIn("incomplete_edit", ets)  # no categorical split
        ip = next(c for c in cands if c.error_type == "incorrect_patch")
        self.assertAlmostEqual(ip.evidence.get("reproduced_fraction"), 2 / 3, places=2)

    def test_zero_reproduced_is_recorded(self):
        ip = next(c for c in self._cands(["x=1"], ["different=1"]) if c.error_type == "incorrect_patch")
        self.assertEqual(ip.evidence.get("reproduced_fraction"), 0.0)

    def test_incomplete_edit_type_is_gone(self):
        cands = self._cands(["a=1", "b=2"], ["a=1"])  # 1/2 reproduced
        self.assertNotIn("incomplete_edit", {c.error_type for c in cands})


class TestVerificationHeuristicsGrounded(unittest.TestCase):
    """Substring heuristics must not misread output. 'no errors' is success;
    only specific failure markers count as a failure."""

    def test_no_errors_is_not_a_failure(self):
        self.assertFalse(looks_like_test_failure("Ran 42 tests, no errors found"))

    def test_all_passed_is_not_a_failure(self):
        self.assertFalse(looks_like_test_failure("===== 5 passed in 0.3s ====="))

    def test_zero_failures_summary_is_not_a_failure(self):
        self.assertFalse(looks_like_test_failure("Ran 10 tests in 1s\n\nOK"))

    def test_real_failure_is_detected(self):
        self.assertTrue(looks_like_test_failure("===== 1 failed, 2 passed ====="))

    def test_assertion_error_is_detected(self):
        self.assertTrue(looks_like_test_failure("E   AssertionError: 1 != 2"))

    def test_empty_output_is_not_a_failure(self):
        self.assertFalse(looks_like_test_failure(""))


class TestGatingTestTokensAreSpecific(unittest.TestCase):
    """'Ran the gating test' must be anchored on the specific test function /
    file, not generic path tokens like 'tests' or the package name."""

    def test_tokens_exclude_generic_path_parts(self):
        g = gold(["pkg/a.py"], ftp=["astropy/timeseries/tests/test_binned.py::test_remove_required_column"])
        g.test_patch_files = ["astropy/timeseries/tests/test_binned.py"]
        toks = _gold_test_tokens(g)
        self.assertIn("test_remove_required_column", toks)
        self.assertNotIn("astropy", toks)
        self.assertNotIn("timeseries", toks)
        self.assertNotIn("tests", toks)


class TestArbiterGatesNonDeductivePrimary(unittest.TestCase):
    """The LLM arbiter complements — never overrides deductive facts. It may demote
    a NON-deductive primary (omission / judge / contributing) when it confidently
    refutes the label, and corroborates it on confirm. Deductive loc/edit facts
    are never sent to the arbiter."""

    def _det(self, cands, flag):
        return {"resolved": False, "loc_flag": flag, "candidates": cands}

    def test_confident_refute_demotes_nondeductive_primary(self):
        det = self._det([_cand("code_editing", "relevant_change_omitted", "high")], "loc_miss")

        def refute(d, c):
            return {"support": "refute", "confidence": "high", "evidence_steps": [3], "rationale": "valid alternative"}

        b = build_blame(det, adjudicate=refute)
        self.assertEqual(b["status"], "unattributed")
        self.assertIsNone(b["primary"])

    def test_deductive_primary_is_never_sent_to_arbiter(self):
        det = self._det([_cand("code_editing", "incorrect_patch", "high")], "right_place_wrong_change")
        calls = []

        def spy(d, c):
            return (
                calls.append(c),
                {"support": "refute", "confidence": "high", "evidence_steps": [1], "rationale": "x"},
            )[1]

        b = build_blame(det, adjudicate=spy)
        self.assertEqual(b["primary"]["capability"], "code_editing")  # deductive stands
        self.assertEqual(calls, [])  # arbiter not consulted

    def test_confirm_keeps_primary_and_marks_corroborated(self):
        det = self._det([_cand("code_editing", "relevant_change_omitted", "high")], "loc_miss")

        def confirm(d, c):
            return {"support": "confirm", "confidence": "high", "evidence_steps": [2], "rationale": "clearly omitted"}

        b = build_blame(det, adjudicate=confirm)
        self.assertEqual(b["primary"]["error_type"], "relevant_change_omitted")
        self.assertTrue(b["primary"].get("arbiter", {}).get("corroborated"))

    def test_low_confidence_refute_does_not_demote(self):
        det = self._det([_cand("code_editing", "relevant_change_omitted", "high")], "loc_miss")

        def weak(d, c):
            return {"support": "refute", "confidence": "low", "evidence_steps": [], "rationale": "not sure"}

        b = build_blame(det, adjudicate=weak)
        self.assertEqual(b["status"], "primary")  # an unconfirmed refute can't demote


class TestDetectorLabelReflectsConfidentPrimary(unittest.TestCase):
    """The construct-validity prediction must report the CONFIDENT primary (so the
    arbiter's demote shows up as an abstention) and surface corroboration — not just
    the largest blame mass."""

    def test_confident_primary_is_reported(self):
        from agent_trajviz.awe._engine.annotation import primary_from_blame

        b = {
            "status": "primary",
            "primary": {"capability": "code_editing", "error_type": "incorrect_patch", "weight": 1.0},
            "capability_blame": {"code_editing": 1.0},
        }
        lbl = primary_from_blame(b)
        self.assertEqual(lbl["primary_capability"], "code_editing")
        self.assertEqual(lbl["primary_error_type"], "incorrect_patch")
        self.assertEqual(lbl["status"], "primary")
        self.assertFalse(lbl["arbiter_corroborated"])

    def test_arbiter_corroboration_is_surfaced(self):
        from agent_trajviz.awe._engine.annotation import primary_from_blame

        b = {
            "status": "primary",
            "primary": {
                "capability": "code_editing",
                "error_type": "relevant_change_omitted",
                "weight": 1.0,
                "arbiter": {"support": "confirm", "corroborated": True},
            },
            "capability_blame": {"code_editing": 1.0},
        }
        self.assertTrue(primary_from_blame(b)["arbiter_corroborated"])

    def test_conjunctive_abstains_on_primary_but_keeps_dominant(self):
        from agent_trajviz.awe._engine.annotation import primary_from_blame

        b = {
            "status": "conjunctive",
            "primary": None,
            "capability_blame": {"code_localization": 0.55, "code_editing": 0.45},
        }
        lbl = primary_from_blame(b)
        self.assertIsNone(lbl["primary_capability"])  # arbiter/conjunctive -> abstain
        self.assertEqual(lbl["status"], "conjunctive")
        self.assertEqual(lbl["dominant_capability"], "code_localization")


class TestArbiterParseAndGate(unittest.TestCase):
    """The arbiter's parser must enforce the bias rules: closed verdict set, and
    a refute/confirm with no cited steps is downgraded so it cannot gate blame."""

    def test_parses_clean_refute(self):
        from agent_trajviz.awe._engine.arbiter import parse_verify

        v = parse_verify('{"support":"refute","confidence":"high","evidence_steps":[3,4],"rationale":"valid alt"}')
        self.assertEqual(v["support"], "refute")
        self.assertEqual(v["confidence"], "high")
        self.assertEqual(v["evidence_steps"], [3, 4])

    def test_invalid_support_coerced_to_unsure(self):
        from agent_trajviz.awe._engine.arbiter import parse_verify

        self.assertEqual(
            parse_verify('{"support":"maybe","confidence":"high","evidence_steps":[1]}')["support"], "unsure"
        )

    def test_refute_without_evidence_is_downgraded(self):
        from agent_trajviz.awe._engine.arbiter import parse_verify

        v = parse_verify('{"support":"refute","confidence":"high","evidence_steps":[]}')
        self.assertEqual(v["confidence"], "low")  # evidence gate -> cannot demote

    def test_garbage_is_safe_unsure(self):
        from agent_trajviz.awe._engine.arbiter import parse_verify

        self.assertEqual(parse_verify("not json")["support"], "unsure")

    def test_mock_client_is_deterministic_and_parseable(self):
        from agent_trajviz.awe._engine.arbiter import mock_client, parse_verify

        msgs = [{"role": "user", "content": "CLAIMED PRIMARY CAUSE: code_editing — relevant_change_omitted"}]
        a = mock_client(msgs)
        b = mock_client(msgs)
        self.assertEqual(a["content"], b["content"])
        self.assertIn(parse_verify(a["content"])["support"], ("confirm", "refute", "unsure"))


class TestStrengthComposition(unittest.TestCase):
    """#2 trust profile: how much of the diagnosed blame rests on deductive vs
    weaker evidence. strength_of maps capability -> tier; compose_strength rolls
    blame weights up into tier shares."""

    def test_strength_of_maps_capabilities(self):
        from agent_trajviz.awe._engine.evidence import strength_of

        self.assertEqual(strength_of("code_localization"), "deductive")
        self.assertEqual(strength_of("code_editing"), "deductive")
        self.assertEqual(strength_of("code_verification"), "associational")
        self.assertEqual(strength_of("self_repair_loop"), "associational")
        self.assertEqual(strength_of("requirement_understanding"), "model_inferred")
        self.assertEqual(strength_of("task_planning"), "model_inferred")

    def test_compose_strength_shares_sum_to_one(self):
        from agent_trajviz.awe._engine.scoring import compose_strength

        blame = [
            {"capability": "code_editing", "weight": 0.6},
            {"capability": "code_verification", "weight": 0.3},
            {"capability": "task_planning", "weight": 0.1},
        ]
        comp = compose_strength(blame)
        self.assertAlmostEqual(comp["deductive"], 0.6, places=3)
        self.assertAlmostEqual(comp["associational"], 0.3, places=3)
        self.assertAlmostEqual(comp["model_inferred"], 0.1, places=3)
        self.assertAlmostEqual(sum(comp.values()), 1.0, places=6)

    def test_compose_strength_empty_is_zero(self):
        from agent_trajviz.awe._engine.scoring import compose_strength

        self.assertEqual(compose_strength([]), {"deductive": 0.0, "associational": 0.0, "model_inferred": 0.0})


class TestPanelAggregation(unittest.TestCase):
    """#3 panel voting: aggregate N independent LLM samples by majority, recording
    an agreement score so single-sample variance is visible/reduced."""

    def test_judge_majority_per_capability(self):
        from agent_trajviz.awe._engine.judge import aggregate_verdicts

        recs = [
            {
                "requirement_understanding": {
                    "error_type": "misread_requirement",
                    "confidence": "high",
                    "evidence_steps": [3],
                    "rationale": "a",
                }
            },
            {
                "requirement_understanding": {
                    "error_type": "misread_requirement",
                    "confidence": "medium",
                    "evidence_steps": [3],
                    "rationale": "b",
                }
            },
            {
                "requirement_understanding": {
                    "error_type": "none",
                    "confidence": "high",
                    "evidence_steps": [],
                    "rationale": "c",
                }
            },
        ]
        agg = aggregate_verdicts(recs)
        self.assertEqual(agg["requirement_understanding"]["error_type"], "misread_requirement")
        self.assertAlmostEqual(agg["requirement_understanding"]["agreement"], 2 / 3, places=2)

    def test_judge_error_majority_beats_label_split(self):
        from agent_trajviz.awe._engine.judge import aggregate_verdicts

        recs = [
            {"task_planning": {"error_type": "flawed_plan", "confidence": "high", "evidence_steps": [1]}},
            {"task_planning": {"error_type": "no_decomposition", "confidence": "high", "evidence_steps": [2]}},
            {"task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": []}},
        ]
        # HIERARCHICAL vote (review fix 1.13): 2 of 3 votes flag SOME error, so the
        # panel must not aggregate to `none` just because the labels split; the
        # winning label is the plurality (tie -> confidence mass -> name).
        agg = aggregate_verdicts(recs)["task_planning"]
        self.assertEqual(agg["error_type"], "flawed_plan")
        self.assertAlmostEqual(agg["agreement"], 1 / 3, places=2)

    def test_judge_none_majority_stays_none_with_derived_confidence(self):
        from agent_trajviz.awe._engine.judge import aggregate_verdicts

        recs = [
            {"task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": []}},
            {"task_planning": {"error_type": "none", "confidence": "high", "evidence_steps": []}},
            {"task_planning": {"error_type": "flawed_plan", "confidence": "high", "evidence_steps": [1]}},
        ]
        agg = aggregate_verdicts(recs)["task_planning"]
        self.assertEqual(agg["error_type"], "none")
        # Confidence is agreement-derived (2/3 <= 0.8 -> medium), never hard-coded high.
        self.assertEqual(agg["confidence"], "medium")

    def test_arbiter_majority_refute(self):
        from agent_trajviz.awe._engine.arbiter import aggregate_arbiter

        vs = [
            {"support": "refute", "confidence": "high", "evidence_steps": [3]},
            {"support": "refute", "confidence": "medium", "evidence_steps": [4]},
            {"support": "confirm", "confidence": "low", "evidence_steps": [1]},
        ]
        agg = aggregate_arbiter(vs)
        self.assertEqual(agg["support"], "refute")
        self.assertAlmostEqual(agg["agreement"], 2 / 3, places=2)

    def test_arbiter_tie_is_unsure(self):
        from agent_trajviz.awe._engine.arbiter import aggregate_arbiter

        vs = [
            {"support": "confirm", "confidence": "high", "evidence_steps": [1]},
            {"support": "refute", "confidence": "high", "evidence_steps": [2]},
        ]
        self.assertEqual(aggregate_arbiter(vs)["support"], "unsure")


class TestNormalizedLineOverlap(unittest.TestCase):
    """reproduced_fraction is robust to INTERNAL whitespace differences, but leading
    indentation is significant (review fix 1.12): in Python, the same statement at a
    different nesting level is a different change — erasing indent inflated the
    fraction for exactly those wrong edits."""

    def test_overlap_ignores_internal_whitespace(self):
        from agent_trajviz.awe._engine.evidence import changed_lines, gold_overlap

        g = changed_lines("diff --git a/x.py b/x.py\n+    x = 1\n-old = 2\n")
        a = changed_lines("diff --git a/x.py b/x.py\n+    x   =   1\n-old  =  2\n")
        ov = gold_overlap(g, a, ["x.py"])
        self.assertEqual(ov["reproduced_fraction"], 1.0)

    def test_overlap_respects_indentation_level(self):
        from agent_trajviz.awe._engine.evidence import changed_lines, gold_overlap

        g = changed_lines("diff --git a/x.py b/x.py\n+        return x\n")
        a = changed_lines("diff --git a/x.py b/x.py\n+    return x\n")
        self.assertEqual(gold_overlap(g, a, ["x.py"])["reproduced_fraction"], 0.0)

    def test_duplicate_lines_counted_as_multiset(self):
        from agent_trajviz.awe._engine.evidence import changed_lines, gold_overlap

        g = changed_lines("diff --git a/x.py b/x.py\n+x += 1\n+x += 1\n")
        a = changed_lines("diff --git a/x.py b/x.py\n+x += 1\n")
        # one agent occurrence must not satisfy two gold occurrences
        self.assertEqual(gold_overlap(g, a, ["x.py"])["reproduced_fraction"], 0.5)

    def test_different_tokens_still_differ(self):
        from agent_trajviz.awe._engine.evidence import changed_lines, gold_overlap

        g = changed_lines("diff --git a/x.py b/x.py\n+x = 1\n")
        a = changed_lines("diff --git a/x.py b/x.py\n+x = 2\n")
        self.assertEqual(gold_overlap(g, a, ["x.py"])["reproduced_fraction"], 0.0)


class TestTrajectoryRoundTrip(unittest.TestCase):
    """The normalized trajectory must serialize/deserialize losslessly so it can be
    disk-cached (regen perf)."""

    def test_to_from_dict_preserves_steps(self):
        from agent_trajviz.awe._engine.trajectory import NormalizedTrajectory, NormalizedStep

        t = NormalizedTrajectory(
            "claude_code",
            "inst",
            [
                NormalizedStep(
                    step_id=0,
                    turn_id=0,
                    actor="agent",
                    type="tool_call",
                    canonical_action="read",
                    files=["pkg/a.py"],
                    call_id="c1",
                ),
                NormalizedStep(
                    step_id=1,
                    turn_id=0,
                    actor="tool",
                    type="tool_result",
                    tool_error=True,
                    output_excerpt="1 failed",
                    call_id="c1",
                ),
            ],
        )
        t2 = NormalizedTrajectory.from_dict(t.to_dict())
        self.assertEqual(t2.agent, "claude_code")
        self.assertEqual(t2.instance_id, "inst")
        self.assertEqual(len(t2.steps), 2)
        self.assertEqual(t2.steps[0].canonical_action, "read")
        self.assertEqual(t2.steps[0].files, ["pkg/a.py"])
        self.assertTrue(t2.steps[1].tool_error)


class TestAdapters(unittest.TestCase):
    """Synthetic-log regression tests for the 3 scaffold adapters (the parsing
    foundation the read-set rides on). Catch format-drift before it silently breaks
    the read-set / test-run / edit signals."""

    @staticmethod
    def _read_set(t):
        from agent_trajviz.awe._engine.trajectory import READ, RUN_COMMAND

        return {f for s in t.steps if s.canonical_action in (READ, RUN_COMMAND) for f in s.files}

    def test_claude_code(self):
        from agent_trajviz.awe._engine.adapters import adapt_claude_code
        from agent_trajviz.awe._engine.trajectory import SEARCH, OBSERVED

        raw = {
            "trajectory": [
                {
                    "role": "assistant",
                    "index": 0,
                    "content": [
                        {"type": "thinking", "thinking": "look at the file"},
                        {
                            "type": "tool_call",
                            "name": "Read",
                            "input": {"file_path": "/x/workdirs/INST/claude_code/pkg/a.py"},
                            "id": "c1",
                        },
                        {"type": "tool_call", "name": "Grep", "input": {"pattern": "foo"}, "id": "c2"},
                        {
                            "type": "tool_call",
                            "name": "Bash",
                            "input": {"command": "python -m pytest pkg/tests/test_a.py"},
                            "id": "c3",
                        },
                        {"type": "text", "text": "done"},
                    ],
                }
            ]
        }
        t = adapt_claude_code(raw, "claude_code", "INST")
        self.assertIn("pkg/a.py", self._read_set(t))  # workdir prefix stripped
        self.assertTrue(any(s.canonical_action == SEARCH for s in t.steps))
        self.assertTrue(any(s.test_run for s in t.steps))
        self.assertTrue(any(s.type == "reasoning" and s.observability == OBSERVED for s in t.steps))

    def test_claude_code_read_span_preserved(self):
        # chunked reads of one large file must stay distinguishable downstream
        # (step_repetition keys READ signatures on path + offset/limit span)
        from agent_trajviz.awe._engine.adapters import adapt_claude_code

        raw = {
            "trajectory": [
                {
                    "role": "assistant",
                    "index": 0,
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "Read",
                            "input": {
                                "file_path": "/x/workdirs/INST/claude_code/big.py",
                                "offset": 2000,
                                "limit": 1000,
                            },
                            "id": "c1",
                        }
                    ],
                }
            ]
        }
        t = adapt_claude_code(raw, "claude_code", "INST")
        step = t.steps[0]
        self.assertEqual(step.args.get("offset"), 2000)
        self.assertEqual(step.args.get("limit"), 1000)

    def test_opencode_read_span_preserved(self):
        from agent_trajviz.awe._engine.adapters import adapt_opencode

        raw = {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "read",
                            "state": {
                                "input": {
                                    "filePath": "/x/workdirs/INST/opencode_glm/big.py",
                                    "offset": 500,
                                    "limit": 250,
                                },
                                "status": "completed",
                                "output": "...",
                            },
                            "callID": "o1",
                        }
                    ],
                }
            ]
        }
        t = adapt_opencode(raw, "opencode_glm", "INST")
        step = t.steps[0]
        self.assertEqual(step.args.get("offset"), 500)
        self.assertEqual(step.args.get("limit"), 250)

    def test_opencode(self):
        from agent_trajviz.awe._engine.adapters import adapt_opencode

        raw = {
            "messages": [
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "read",
                            "state": {
                                "input": {"filePath": "/x/workdirs/INST/opencode_glm/pkg/a.py"},
                                "status": "completed",
                                "output": "...",
                            },
                            "callID": "o1",
                        },
                        {
                            "type": "tool",
                            "tool": "bash",
                            "state": {"input": {"command": "pytest pkg/tests/test_a.py"}, "status": "completed"},
                            "callID": "o2",
                        },
                    ],
                }
            ]
        }
        t = adapt_opencode(raw, "opencode_glm", "INST")
        self.assertIn("pkg/a.py", self._read_set(t))
        self.assertTrue(any(s.test_run for s in t.steps))

    def test_codex(self):
        from agent_trajviz.awe._engine.adapters import adapt_codex
        from agent_trajviz.awe._engine.trajectory import EDIT, HIDDEN

        raw = [
            {"type": "response_item", "payload": {"type": "reasoning"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "cat pkg/a.py"}),
                    "call_id": "x1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "python -m pytest pkg/tests/test_a.py"}),
                    "call_id": "x2",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": json.dumps({"cmd": "*** Update File: pkg/a.py\n@@\n-x\n+y\n"}),
                    "call_id": "x3",
                },
            },
        ]
        t = adapt_codex(raw, "codex", "INST")
        self.assertIn("pkg/a.py", self._read_set(t))  # shell `cat` -> read-set
        self.assertTrue(any(s.test_run for s in t.steps))
        self.assertTrue(any(s.canonical_action == EDIT and "pkg/a.py" in s.files for s in t.steps))
        self.assertTrue(any(s.type == "reasoning" and s.observability == HIDDEN for s in t.steps))


if __name__ == "__main__":
    unittest.main()
