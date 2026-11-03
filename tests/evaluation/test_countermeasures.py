"""Countermeasure catalog: loader, query API, validation.

Run:  python3 -m unittest tests.test_countermeasures -v
"""

import unittest
from agent_trajviz.awe._engine import countermeasures as cm
from agent_trajviz.awe._engine import validate


class TestLoader(unittest.TestCase):
    def test_load_catalog_returns_entries(self):
        entries = cm.load_catalog()
        self.assertTrue(entries)
        for e in entries:
            self.assertTrue(e.id and e.capability and e.error_type)
            self.assertTrue(e.mechanism, f"{e.id}: every entry must carry a non-empty mechanism")
            self.assertIn(e.evidence_signal, cm.known_evidence_signals(), e.id)
            self.assertIn(e.status, cm.VALID_STATUS, e.id)
            self.assertIn(e.intervention_type, cm.VALID_TYPES, e.id)

    def test_recommend_backcompat_shape(self):
        rec = cm.recommend("relevant_change_omitted")
        self.assertEqual(set(rec), {"target", "strategies"})
        self.assertEqual(rec["target"], "code_editing")
        self.assertIsInstance(rec["strategies"], list)
        self.assertTrue(all(isinstance(s, str) for s in rec["strategies"]))

    def test_countermeasures_for_filters(self):
        edits = cm.countermeasures_for(capability="code_editing")
        self.assertTrue(edits)
        self.assertTrue(all(e.capability == "code_editing" for e in edits))
        one = cm.countermeasures_for(error_type="relevant_change_omitted")
        self.assertTrue(all(e.error_type == "relevant_change_omitted" for e in one))
        untested = cm.countermeasures_for(status="untested")
        self.assertTrue(all(e.status == "untested" for e in untested))

    def test_no_verified_entries_ship(self):
        self.assertEqual(cm.countermeasures_for(status="verified"), [])

    def test_to_entry_reads_artifact_fields(self):
        d = {
            "id": "X",
            "root_cause": {"capability": "code_editing", "error_type": "incorrect_patch"},
            "intervention": {"type": "guardrail", "summary": "s", "strategies": ["a"]},
            "mechanism": "m",
            "grounds": {"evidence_signal": "right_place_wrong_change"},
            "provenance": {"kind": "heuristic", "cites": []},
            "artifact": {
                "type": "guardrail",
                "content": "do X",
                "plug_in": "hook",
                "runtime_trigger": "flag",
                "diagnostic_signature": "detector condition",
            },
            "example_case": {"instance_id": "django__django-123", "repo": "django/django", "what_failed": "wrong edit"},
            "feasibility": "needs test results",
            "status": "untested",
            "verification": None,
            "hook_event": "per_edit",
            "expected_effect": ["fix_rate"],
            "contract": {"max_triggers": 2, "escape_hatch": "justify"},
            "calibration": {"rr": 2.10},
        }
        e = cm._to_entry(d)
        self.assertEqual(e.artifact_type, "guardrail")
        self.assertEqual(e.artifact_content, "do X")
        self.assertEqual(e.artifact_plug_in, "hook")
        self.assertEqual(e.runtime_trigger, "flag")
        self.assertEqual(e.diagnostic_signature, "detector condition")
        self.assertEqual(e.artifact_applies_when, "flag")  # back-compat alias -> runtime_trigger
        self.assertEqual(e.hook_event, "per_edit")
        self.assertEqual(e.expected_effect, ["fix_rate"])
        self.assertEqual(e.contract["max_triggers"], 2)
        self.assertEqual(e.calibration["rr"], 2.10)
        self.assertEqual(e.example_case["instance_id"], "django__django-123")
        self.assertEqual(e.feasibility, "needs test results")

    def test_to_entry_reads_legacy_applies_when_as_trigger(self):
        # a v1-shaped catalog (external/legacy) still loads: applies_when -> trigger
        d = {
            "id": "X",
            "root_cause": {"capability": "code_editing", "error_type": "incorrect_patch"},
            "intervention": {"type": "guardrail", "summary": "s"},
            "artifact": {"type": "guardrail", "content": "c", "applies_when": "old-style"},
        }
        e = cm._to_entry(d)
        self.assertEqual(e.runtime_trigger, "old-style")
        self.assertEqual(e.artifact_applies_when, "old-style")

    def test_to_entry_missing_artifact_defaults_empty(self):
        e = cm._to_entry(
            {
                "id": "Y",
                "root_cause": {"capability": "code_editing", "error_type": "incorrect_patch"},
                "intervention": {"type": "guardrail", "summary": "s"},
            }
        )
        self.assertEqual(e.artifact_content, "")
        self.assertIsNone(e.example_case)
        self.assertEqual(e.feasibility, "")

    def test_to_entry_reads_references(self):
        d = {
            "id": "X",
            "root_cause": {"capability": "code_editing", "error_type": "incorrect_patch"},
            "intervention": {"type": "guardrail", "summary": "s"},
            "references": [
                {
                    "title": "T",
                    "authors": "A",
                    "year": 2023,
                    "source_type": "preprint",
                    "id": "arXiv:1234.5678",
                    "supports": "x",
                }
            ],
        }
        e = cm._to_entry(d)
        self.assertEqual(len(e.references), 1)
        self.assertEqual(e.references[0]["id"], "arXiv:1234.5678")

    def test_provenance_kind_for(self):
        self.assertEqual(cm.provenance_kind_for([{"source_type": "preprint"}]), "literature")
        self.assertEqual(cm.provenance_kind_for([{"source_type": "paper"}]), "literature")
        self.assertEqual(cm.provenance_kind_for([{"source_type": "docs"}]), "documented")
        self.assertEqual(cm.provenance_kind_for([{"source_type": "web"}]), "documented")
        self.assertEqual(cm.provenance_kind_for([]), "heuristic")


class TestCatalogValidation(unittest.TestCase):
    def test_references_wellformed_and_provenance_consistent(self):
        for e in cm.load_catalog():
            for r in e.references:
                self.assertTrue(r.get("title") and r.get("id") and r.get("supports"), f"{e.id}: malformed reference")
                self.assertIn(r.get("source_type"), cm.VALID_SOURCE_TYPES, f"{e.id}: bad source_type")
            self.assertEqual(
                e.provenance_kind,
                cm.provenance_kind_for(e.references),
                f"{e.id}: provenance.kind inconsistent with references",
            )

    def test_real_catalog_passes_all_checks(self):
        r = validate.countermeasure_catalog()
        self.assertEqual(r["passed"], r["n"], r["checks"])
        self.assertEqual(r["uncovered"], [], f"uncovered error types: {r['uncovered']}")

    def _entry(self, **over):
        base = dict(
            id="X",
            capability="code_editing",
            error_type="incorrect_patch",
            intervention_type="guardrail",
            summary="s",
            strategies=["a"],
            mechanism="m",
            evidence_signal="hit_overlap",
            provenance_kind="heuristic",
            status="untested",
            verification=None,
            artifact_type="guardrail",
            artifact_content="c",
            runtime_trigger="w",
            diagnostic_signature="d",
            feasibility="f",
            example_case=None,
            references=[],
            scaffold_layer="control_flow",
            hook_event="per_edit",
            expected_effect=["fix_rate"],
            contract={"max_triggers": 2, "escape_hatch": "state why and proceed"},
        )
        base.update(over)
        return cm.Entry(**base)

    def test_references_check_present_and_passes(self):
        r = validate.countermeasure_catalog()
        self.assertIn("references", r["checks"])
        self.assertTrue(r["checks"]["references"])
        self.assertEqual(r["passed"], r["n"])

    def test_bad_reference_fails_check(self):
        e = self._entry()
        e.references = [{"title": "T", "id": "", "supports": "x", "source_type": "preprint"}]  # empty id
        e.provenance_kind = "literature"
        r = validate.countermeasure_catalog([e])
        self.assertFalse(r["checks"]["references"])

    def test_provenance_inconsistent_fails_check(self):
        e = self._entry()
        e.references = []  # no refs ...
        e.provenance_kind = "literature"  # ... but claims literature
        r = validate.countermeasure_catalog([e])
        self.assertFalse(r["checks"]["references"])

    def test_missing_mechanism_fails_schema(self):
        r = validate.countermeasure_catalog([self._entry(mechanism="")])
        self.assertFalse(r["checks"]["schema"])

    def test_unknown_error_type_fails_referential(self):
        r = validate.countermeasure_catalog([self._entry(error_type="made_up")])
        self.assertFalse(r["checks"]["referential_integrity"])

    def test_unknown_signal_fails_referential(self):
        r = validate.countermeasure_catalog([self._entry(evidence_signal="bogus")])
        self.assertFalse(r["checks"]["referential_integrity"])

    def test_valid_but_wrong_signal_fails_referential(self):
        # misread_requirement is judge-grounded ('llm_judge'). A real-but-wrong signal
        # (a deductive editing signal) must NOT pass: the binding is per-error-type, not
        # mere membership in the global signal vocabulary. Guards the auditability chain.
        e = self._entry(
            capability="requirement_understanding",
            error_type="misread_requirement",
            evidence_signal="right_place_wrong_change",
        )
        r = validate.countermeasure_catalog([e])
        self.assertFalse(r["checks"]["referential_integrity"])

    def test_correct_binding_passes_referential(self):
        # the EXACT per-type signal (llm_judge for a judge-produced type) passes.
        e = self._entry(
            capability="requirement_understanding", error_type="misread_requirement", evidence_signal="llm_judge"
        )
        r = validate.countermeasure_catalog([e])
        self.assertTrue(r["checks"]["referential_integrity"])

    def test_verified_without_verification_fails_honesty(self):
        r = validate.countermeasure_catalog([self._entry(status="verified", verification=None)])
        self.assertFalse(r["checks"]["honesty_invariant"])

    def test_capability_mismatch_fails_referential(self):
        # incorrect_patch belongs to code_editing, not code_localization
        r = validate.countermeasure_catalog([self._entry(capability="code_localization")])
        self.assertFalse(r["checks"]["referential_integrity"])

    def test_catalog_covers_exactly_produced_types(self):
        from agent_trajviz.awe._engine.detect import produced_error_types

        entries = cm.load_catalog()
        covered = {e.error_type for e in entries}
        self.assertEqual(
            covered,
            produced_error_types(),
            f"covered={covered - produced_error_types()} extra, missing={produced_error_types() - covered}",
        )

    def test_eval_plus_types_are_produced_and_covered(self):
        # v0.39: the three formerly-orphan eval+ types are live (per-test tier)
        # and therefore MUST be covered; the demoted gating_test_modified must not be.
        from agent_trajviz.awe._engine.detect import produced_error_types

        entries = cm.load_catalog()
        covered = {e.error_type for e in entries}
        live = {"partial_fix_only", "regression_introduced", "regression_test_missing", "test_expectation_rewritten"}
        self.assertTrue(live <= produced_error_types(), live - produced_error_types())
        self.assertTrue(live <= covered, live - covered)
        self.assertNotIn("gating_test_modified", produced_error_types())
        self.assertNotIn("gating_test_modified", covered)

    def test_every_signal_is_real(self):
        from agent_trajviz.awe._engine.detect import grounding_signals

        real = grounding_signals()
        for e in cm.load_catalog():
            self.assertIn(e.evidence_signal, real, f"{e.id}: fictional signal {e.evidence_signal!r}")

    def test_unproduced_type_fails_referential(self):
        # gating_test_modified is no longer produced by the diagnosis layer (v0.39)
        r = validate.countermeasure_catalog(
            [self._entry(error_type="gating_test_modified", capability="code_verification")]
        )
        self.assertFalse(r["checks"]["referential_integrity"])

    def test_every_entry_has_a_deployable_artifact(self):
        for e in cm.load_catalog():
            self.assertTrue(e.artifact_content.strip(), f"{e.id}: empty artifact.content")
            self.assertTrue(e.artifact_applies_when.strip(), f"{e.id}: empty applies_when")
            self.assertTrue(e.feasibility.strip(), f"{e.id}: empty feasibility")
            self.assertEqual(e.artifact_type, e.intervention_type, f"{e.id}: artifact/intervention type mismatch")
            self.assertTrue(
                e.example_case is None or (isinstance(e.example_case, dict) and e.example_case.get("instance_id")),
                f"{e.id}: example_case must be null or have an instance_id",
            )

    def test_artifacts_have_no_vague_verbs(self):
        # the headline artifact content must be specific, not generic advice
        vague = ("broaden", "improve", "consider ", "as appropriate", "etc.")
        for e in cm.load_catalog():
            low = e.artifact_content.lower()
            self.assertFalse(any(v in low for v in vague), f"{e.id}: vague verb in artifact.content")

    def test_artifact_check_present_and_passes(self):
        r = validate.countermeasure_catalog()
        self.assertIn("artifact", r["checks"])
        self.assertTrue(r["checks"]["artifact"])
        self.assertEqual(r["passed"], r["n"])

    def test_scaffold_layer_check_present_and_passes(self):
        r = validate.countermeasure_catalog()
        self.assertIn("scaffold_layer", r["checks"])
        self.assertTrue(r["checks"]["scaffold_layer"])
        self.assertEqual(r["n"], 18)  # legacy structural checks + evidence/maturity/runtime
        self.assertEqual(r["passed"], r["n"])

    def test_bad_scaffold_layer_fails_check(self):
        r = validate.countermeasure_catalog([self._entry(scaffold_layer="bogus")])
        self.assertFalse(r["checks"]["scaffold_layer"])

    def test_missing_scaffold_layer_fails_check(self):
        r = validate.countermeasure_catalog([self._entry(scaffold_layer="")])
        self.assertFalse(r["checks"]["scaffold_layer"])

    def test_all_entries_have_a_valid_scaffold_layer(self):
        for e in cm.load_catalog():
            self.assertIn(
                e.scaffold_layer, cm.VALID_SCAFFOLD_LAYERS, f"{e.id}: bad scaffold_layer {e.scaffold_layer!r}"
            )

    def test_empty_artifact_content_fails_artifact_check(self):
        e = self._entry()
        e.artifact_content = ""
        e.runtime_trigger = "x"
        e.feasibility = "x"
        e.artifact_type = e.intervention_type
        r = validate.countermeasure_catalog([e])
        self.assertFalse(r["checks"]["artifact"])

    def test_artifact_type_mismatch_fails(self):
        e = self._entry()
        e.artifact_content = "c"
        e.runtime_trigger = "x"
        e.feasibility = "x"
        e.artifact_type = "prompt"  # entry is guardrail
        r = validate.countermeasure_catalog([e])
        self.assertFalse(r["checks"]["artifact"])

    def test_every_entry_has_empirical_support(self):
        for e in cm.load_catalog():
            self.assertIsInstance(e.empirical_support, int, f"{e.id}: empirical_support must be int")
            self.assertGreaterEqual(e.empirical_support, 0, f"{e.id}: empirical_support must be >= 0")

    def test_speculative_entries_zero_support(self):
        scope = cm.countermeasures_for(error_type="scope_misjudged")
        self.assertTrue(scope, "no entry for scope_misjudged")
        self.assertEqual(scope[0].empirical_support, 0)
        nodecomp = cm.countermeasures_for(error_type="no_decomposition")
        self.assertTrue(nodecomp, "no entry for no_decomposition")
        self.assertEqual(nodecomp[0].empirical_support, 0)

    def test_incorrect_patch_split_into_content_and_region_variants(self):
        # v2: the biggest fault type carries two attribute-conditioned entries —
        # content-miss (INCORRECT-1, reproduce-and-iterate) and region-miss
        # (INCORRECT-2, within-file anchoring). Both pin the same per-type signal.
        entries = cm.countermeasures_for(error_type="incorrect_patch")
        self.assertEqual([e.id for e in entries], ["CM-EDIT-INCORRECT-1", "CM-EDIT-INCORRECT-2"])
        self.assertTrue(all(e.evidence_signal == "right_place_wrong_change" for e in entries))
        self.assertIn("region_overlap == 0", entries[1].diagnostic_signature)


class TestCatalogV2(unittest.TestCase):
    """The 2026-07 refinement: deployment semantics, composition, honesty caveats."""

    def _entry(self, **over):
        return TestCatalogValidation._entry(self, **over)

    # --- new checks fail on bad synthetic entries ---
    def test_missing_hook_event_fails(self):
        r = validate.countermeasure_catalog([self._entry(hook_event="")])
        self.assertFalse(r["checks"]["hook_event"])

    def test_bogus_hook_event_fails(self):
        r = validate.countermeasure_catalog([self._entry(hook_event="on_vibes")])
        self.assertFalse(r["checks"]["hook_event"])

    def test_empty_expected_effect_fails(self):
        r = validate.countermeasure_catalog([self._entry(expected_effect=[])])
        self.assertFalse(r["checks"]["expected_effect"])

    def test_bogus_expected_effect_fails(self):
        r = validate.countermeasure_catalog([self._entry(expected_effect=["vibes"])])
        self.assertFalse(r["checks"]["expected_effect"])

    def test_gold_in_runtime_trigger_fails_oracle_free(self):
        # the deployable condition must never consume benchmark-oracle data
        r = validate.countermeasure_catalog([self._entry(runtime_trigger="patch touches a gold gating-test file")])
        self.assertFalse(r["checks"]["oracle_free_trigger"])

    def test_gold_in_diagnostic_signature_is_allowed(self):
        # gold vocabulary belongs on the diagnostic side — that must NOT fail
        r = validate.countermeasure_catalog(
            [self._entry(diagnostic_signature="patch modifies a gold gating-test file")]
        )
        self.assertTrue(r["checks"]["oracle_free_trigger"])

    def test_guardrail_without_contract_fails(self):
        r = validate.countermeasure_catalog([self._entry(contract=None)])
        self.assertFalse(r["checks"]["guardrail_contract"])

    def test_guardrail_contract_needs_escape_hatch(self):
        r = validate.countermeasure_catalog([self._entry(contract={"max_triggers": 2, "escape_hatch": ""})])
        self.assertFalse(r["checks"]["guardrail_contract"])

    def test_prompt_entry_needs_no_contract(self):
        e = self._entry(intervention_type="prompt", artifact_type="prompt", contract=None)
        r = validate.countermeasure_catalog([e])
        self.assertTrue(r["checks"]["guardrail_contract"])

    # --- the shipped catalog satisfies all of it ---
    def test_all_new_checks_pass_on_real_catalog(self):
        r = validate.countermeasure_catalog()
        for name in (
            "hook_event",
            "expected_effect",
            "oracle_free_trigger",
            "guardrail_contract",
            "bundles",
            "methodology",
            "annex",
        ):
            self.assertIn(name, r["checks"])
            self.assertTrue(r["checks"][name], name)

    def test_bundles_partition_the_catalog(self):
        members = [m for b in cm.bundles() for m in b.get("members", [])]
        ids = {e.id for e in cm.load_catalog()}
        self.assertEqual(len(members), len(set(members)), "entry in two bundles")
        self.assertEqual(set(members), ids, "bundles must cover every entry exactly once")

    def test_every_guardrail_has_escape_hatch(self):
        for e in cm.load_catalog():
            if e.intervention_type == "guardrail":
                self.assertTrue(
                    e.contract and e.contract.get("escape_hatch"),
                    f"{e.id}: blocking gate without an exit (METH-COMPOSE-1)",
                )

    def test_catalog_role_derived_from_signal(self):
        roles = {e.id: e.catalog_role for e in cm.load_catalog()}
        self.assertEqual(roles["CM-REQ-MISREAD-1"], "completeness")  # llm_judge-grounded
        self.assertEqual(roles["CM-VERIF-TESTREWRITE-1"], "core")  # objective signal
        n_completeness = sum(1 for v in roles.values() if v == "completeness")
        self.assertEqual(n_completeness, 8, "the 8 judge-produced types")

    def test_testrewrite_promoted_to_literature(self):
        g = cm.countermeasures_for(error_type="test_expectation_rewritten")[0]
        self.assertEqual(g.provenance_kind, "literature")
        self.assertTrue(
            any("2510.20270" in r.get("id", "") for r in g.references), "ImpossibleBench must ground the write-guard"
        )
        self.assertEqual(g.evidence_signal, "assert_rewritten_files")
        # the flat pre-v0.39 fact is recorded as the adjacent negative
        self.assertIn("1.01", (g.calibration or {}).get("adjacent_negative", ""))

    def test_calibration_caveats_present(self):
        # INCORRECT-1 must carry the negative reproduce-first result; SELFREVERT-1
        # must carry the protective walk-back caveat. The catalog may not silently
        # contradict the repo's own calibration (DESIGN.md §0y / §0z).
        i1 = cm.countermeasures_for(error_type="incorrect_patch")[0]
        self.assertIn("0.83", (i1.calibration or {}).get("adjacent_negative", ""))
        sr = cm.countermeasures_for(error_type="reverted_own_change")[0]
        self.assertIn("0.39", (sr.calibration or {}).get("adjacent_negative", ""))

    def test_verification_entries_declare_measurable_endpoints(self):
        # CONTRIBUTING is an attribution policy, not a prohibition on measuring
        # downstream resolution improvements from repair or candidate selection.
        for e in cm.load_catalog():
            if e.capability == "code_verification":
                self.assertTrue(e.expected_effect)
                self.assertLessEqual(set(e.expected_effect), cm.VALID_EFFECTS)

    def test_annex_grounds_on_real_trajectory_signal(self):
        import dataclasses
        from agent_trajviz.awe._engine.trajsignals import TrajectorySignals

        fields = {f.name for f in dataclasses.fields(TrajectorySignals)}
        annex = cm.operational_annex()
        self.assertTrue(annex, "annex missing")
        for a in annex:
            self.assertIn(a["grounds"]["trajectory_signal"], fields, a["id"])

    def test_methodology_notes_cover_the_five_caveats(self):
        ids = {m["id"] for m in cm.methodology()}
        self.assertLessEqual({"METH-ORACLE-1", "METH-COMPOSE-1", "METH-OBS-1", "METH-JUDGE-1", "METH-VERIF-1"}, ids)

    # --- v0.29: Jiang et al. (arXiv:2605.12270) crosswalk additions ---

    def test_blindretry_withdrawn_and_recorded_as_adjacent_negative(self):
        # v0.42: retry_without_change calibrates protective (RR 0.40) — the
        # guardrail is withdrawn and the negative is pinned on the sibling entry
        ids = {x.id for x in cm.load_catalog()}
        self.assertNotIn("CM-REPAIR-BLINDRETRY-1", ids)
        import json
        from agent_trajviz.awe._engine import config

        doc = json.loads((config.PROJECT_ROOT / "configs" / "countermeasures.json").read_text())
        raw = next(e for e in doc["entries"] if e["id"] == "CM-REPAIR-REPEAT-1")
        self.assertIn("retry_without_change", raw["calibration"]["adjacent_negative"])
        bundle = next(b for b in cm.bundles() if b["id"] == "retry-policy-pack")
        self.assertNotIn("CM-REPAIR-BLINDRETRY-1", bundle["members"])
        self.assertEqual(len(bundle["members"]), 4)

    def test_meth_harness_note_present_with_primary_citation(self):
        note = next(m for m in cm.methodology() if m["id"] == "METH-HARNESS-1")
        self.assertIn("16.5%", note["note"])
        self.assertTrue(any(r.get("id") == "arXiv:2605.12270" for r in note.get("references", [])))

    def test_benchmark_annex_grounds_on_real_signals_and_stays_manual(self):
        import dataclasses
        from agent_trajviz.awe._engine.trajsignals import TrajectorySignals

        fields = {f.name for f in dataclasses.fields(TrajectorySignals)}
        annex = cm.benchmark_artifact_annex()
        self.assertTrue(annex, "benchmark artifact annex missing")
        for b in annex:
            for s in b["grounds"]["trajectory_signals"]:
                self.assertIn(s, fields, b["id"])
            self.assertTrue(b["grounds"]["patch_condition"].strip(), b["id"])
            self.assertIn("review", b["mechanism"].lower(), b["id"])  # manual queue
            self.assertIsInstance(b["prevalence"], dict, b["id"])

    def test_benchmark_annex_check_fails_on_fake_signal(self):
        import json as _json
        from agent_trajviz.awe._engine import config as _config
        import agent_trajviz.awe._engine.countermeasures as _cm

        raw = _json.loads((_config.PROJECT_ROOT / "configs" / "countermeasures.json").read_text())
        raw["benchmark_artifact_annex"][0]["grounds"]["trajectory_signals"] = ["not_a_signal"]
        old = _cm._DOC_CACHE
        try:
            _cm._DOC_CACHE = raw
            r = validate.countermeasure_catalog()
            self.assertFalse(r["checks"]["benchmark_annex"])
        finally:
            _cm._DOC_CACHE = old

    def test_new_entries_have_real_example_cases(self):
        # six of the v0.22-v0.27 entries had observed instances in the shipped run —
        # they must cite one (the other three genuinely produced zero faults).
        # v0.39: GATETEST was retargeted to TESTREWRITE (same example case).
        for eid in (
            "CM-VERIF-TESTREWRITE-1",
            "CM-REQ-UNRELATED-1",
            "CM-REPAIR-SELFREVERT-1",
            "CM-VERIF-FINALTEST-1",
            "CM-VERIF-AUTHOREDTEST-1",
        ):
            e = next(x for x in cm.load_catalog() if x.id == eid)
            self.assertTrue(e.example_case and e.example_case.get("instance_id"), eid)


class TestSurfaces(unittest.TestCase):
    def test_fault_countermeasures_payload(self):
        from agent_trajviz.awe._engine.countermeasures import countermeasures_for

        cms = countermeasures_for(error_type="relevant_change_omitted")
        payload = [
            {
                "id": c.id,
                "type": c.intervention_type,
                "summary": c.summary,
                "mechanism": c.mechanism,
                "provenance": c.provenance_kind,
                "status": c.status,
            }
            for c in cms
        ]
        self.assertTrue(payload)
        self.assertTrue(all(p["mechanism"] for p in payload))
        self.assertTrue(all(p["status"] in {"untested", "proposed"} for p in payload))

    def test_by_scaffold_layer_groups_all_entries(self):
        groups = cm.by_scaffold_layer()
        total = sum(len(v) for v in groups.values())
        self.assertEqual(total, len(cm.load_catalog()))  # every entry placed
        self.assertNotIn("unassigned", groups)  # all 20 are assigned
        self.assertTrue(set(groups) <= cm.VALID_SCAFFOLD_LAYERS)

    def test_scaffold_view_md_renders_layers_and_types(self):
        md = cm.scaffold_layer_view_md()
        self.assertIn("Control flow", md)
        self.assertIn("Retry policy", md)
        self.assertIn("`tool_failure_not_retried`", md)  # an error type appears
        self.assertIn("untested", md)  # honesty note present


if __name__ == "__main__":
    unittest.main()
