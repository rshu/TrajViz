"""Objective-anchored candidate-error detector.

Maps the gold/patch/outcome/trajectory signals onto taxonomy.slice error types.
This is the RULE layer: objective signals decide *whether* a candidate exists.
A later LLM-explain layer will refine the *why* and the dimension wording — it
must never invent an error the objective signals don't support (see explain())."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# The error types this layer can actually emit, mapped to the objective signal that
# grounds each. This is the SINGLE SOURCE the countermeasure catalog validates against
# (so a remedy can never pin to a signal/type the diagnosis layer does not produce).
OBJECTIVE_ERROR_SIGNALS = {
    "solved_unrelated_problem": "solved_unrelated_candidate",
    "missed_relevant_file": "missed_unreached",
    "wrong_file_localization": "extra_files",
    "cross_module_context_missing": "missed_unreached",
    "relevant_change_omitted": "missed_reached",
    "incorrect_patch": "right_place_wrong_change",
    "excessive_extra_edits": "extra_src_loc",
    "no_patch_produced": "no_source_edit",
    "gave_up_early": "gave_up_early",
    # v0.39 eval+ tier — per-test harness record (agent_trajviz.awe._engine.outcomes.load_pertest)
    "partial_fix_only": "f2p_pass_fraction",
    "regression_introduced": "p2p_broken",
    "regression_test_missing": "p2p_broken",
    # v0.39: replaces gating_test_modified (its fact, gold_tests_edited, calibrated
    # flat — RR 1.01 — and stays a signal; scripts/calibrate_test_rewrite.py)
    "test_expectation_rewritten": "assert_rewritten_files",
    "relevant_test_not_run": "ran_any_test",
    "unsupported_success_claim": "claimed_success_final",
    "test_result_misinterpreted": "claimed_success_final",
    "no_test_after_final_edit": "no_test_after_final_edit",
    "authored_test_never_run": "authored_test_never_run",
    "repair_not_reverified": "repair_not_reverified",
    "repeated_ineffective_attempt": "repeated_ineffective_attempt",
    # v0.42: retry_without_change demoted — its fact calibrates PROTECTIVE at
    # every measurement (RR 0.68 → 0.40 once exit codes reached the signals,
    # §0ax); the trajsignal stays a cohort feature (scripts/calibrate_*.py).
    "reverted_own_change": "edit_reverted_own_change",
    "tool_failure_not_retried": "tool_failure_not_retried",
    "repeated_patch_apply_failure": "repeated_patch_apply_failure",
    "hallucinated_path_unrecovered": "hallucinated_path_unrecovered",
}
LLM_JUDGE_SIGNAL = "llm_judge"  # honest sentinel: judge-produced types have no objective signal


def produced_error_types() -> set:
    """Every error type the diagnosis layer can actually emit (objective + judge)."""
    from .judge import REQUIREMENT_ERRORS, PLANNING_ERRORS

    return set(OBJECTIVE_ERROR_SIGNALS) | set(REQUIREMENT_ERRORS) | set(PLANNING_ERRORS)


def grounding_signals() -> set:
    """The real grounding-signal vocabulary: objective signals + the judge sentinel."""
    return set(OBJECTIVE_ERROR_SIGNALS.values()) | {LLM_JUDGE_SIGNAL}


def expected_evidence_signal(error_type: str) -> str | None:
    """The ONE grounding signal that diagnoses this error type — the per-type binding
    the countermeasure catalog must pin to. Objective types map to their objective
    signal; the judge-produced types ground on the LLM_JUDGE_SIGNAL sentinel; an
    error type the system does not produce has no valid signal (None)."""
    if error_type in OBJECTIVE_ERROR_SIGNALS:
        return OBJECTIVE_ERROR_SIGNALS[error_type]
    if error_type in produced_error_types():  # judge-produced (requirement/planning)
        return LLM_JUDGE_SIGNAL
    return None


from .diffutil import gold_added_coverage
from .evidence import NEAR_GOLD_FRACTION
from .gold import build_gold_reference
from .patches import load_agent_patch
from .outcomes import resolved as outcome_resolved, load_pertest, load_pertest_log
from .signals import localization_signals, OVEREDIT_EXTRA_LOC
from .trajectory import names_any_test
from .trajsignals import compute as compute_tsig, signals_cached
from .judge import judge_candidates, active as judge_active


@dataclass
class CandidateError:
    capability: str
    dimension: str
    error_type: str
    confidence: str  # high | medium  (rule-based prior)
    evidence: dict = field(default_factory=dict)


_UNSET = object()


def _editing_evidence(loc: dict) -> dict:
    """Evidence for a right-files editing fault. Attaches reproduced_fraction — the
    deductive line-overlap with the reference change — as a graded SEVERITY attribute
    on the single incorrect_patch type (0 = wholly different change, →1 = nearly the
    reference). We do NOT bucket it into a separate error type: the distribution is a
    smooth decline with no natural boundary, and the only principled partial/regression
    split needs per-test results (deferred — ARGUS eval is bool-only)."""
    ov = loc.get("hit_overlap")
    ev = {"hit": loc["hit"]}
    if ov is not None:
        ev["reproduced_fraction"] = ov.get("reproduced_fraction")
        ev["per_file"] = ov.get("per_file")
    rg = loc.get("hit_region")
    if rg is not None:
        # Complementary severity attribute: WHERE in the file the edit landed.
        # Low region_overlap = same file, disjoint region (a within-file
        # localization nuance); high region + low reproduced_fraction = right
        # place, wrong content.
        ev["region_overlap"] = rg.get("region_overlap")
        ev["region_per_file"] = rg.get("per_file")
    return ev


# Failure-licensed candidates whose chain rests on "the gating tests still
# fail". When the per-test record shows EVERY gating test passing, that
# premise is false for the case and the finer grader fact (regression_introduced)
# supersedes them all — a localization miss that did not stop the fix from
# working was not the cause (v0.40; previously only incorrect_patch was
# superseded and the surviving localization primary contradicted the
# regression chain on the same case).
_GATING_FAIL_TYPES = (
    "incorrect_patch",
    "relevant_change_omitted",
    "missed_relevant_file",
    "cross_module_context_missing",
    "wrong_file_localization",
)


def objective_candidates(gold, loc: dict, t, res, pertest: dict | None = None) -> list[CandidateError]:
    """Pure: the objective (non-judge) candidate errors implied by the
    gold/patch/read-set/outcome facts plus trajectory signals. Failure errors
    fire only when ``res is False``. ``pertest`` (optional, v0.39) is the
    normalized per-test harness record (agent_trajviz.awe._engine.outcomes.load_pertest) that unlocks
    the eval+ tier: partial_fix_only / regression_introduced /
    regression_test_missing.

    Grounding rule for the localization/editing split: a gold file is a
    *localization* miss ONLY when it was neither edited NOR read — i.e. never
    reached. A gold file the agent READ but did not edit was reached, so its
    omission is an editing/selection fault (``relevant_change_omitted``), not a
    search failure. With no trajectory, missed files have unknown reachability; no
    search failure is inferred from missing telemetry."""
    cands: list[CandidateError] = []
    if res is not False:
        return cands

    flag = loc["flag"]
    unreached = loc.get("missed_unreached", loc.get("missed_gold", []))  # never read, never edited
    reached = loc.get("missed_reached", [])  # read but not edited

    if flag == "loc_miss":
        if unreached:
            cands.append(
                CandidateError(
                    "code_localization",
                    "file_level_localization",
                    "missed_relevant_file",
                    "high",
                    {
                        "missed_gold": unreached,
                        "recall": loc["recall"],
                        "read_set_available": loc.get("read_set_available", False),
                    },
                )
            )
            if loc["extra_files"]:
                cands.append(
                    CandidateError(
                        "code_localization",
                        "file_level_localization",
                        "wrong_file_localization",
                        "medium",
                        {
                            "extra_files": loc["extra_files"],
                            # zero shared identifiers between patch and issue text =
                            # possibly solved a DIFFERENT problem (requirement-level
                            # candidate; adjudication is the judge/arbiter/human's call)
                            "issue_symbol_overlap": loc.get("issue_symbol_overlap", []),
                            "solved_unrelated_candidate": loc.get("solved_unrelated_candidate", False),
                        },
                    )
                )
        if reached:
            cands.append(
                CandidateError(
                    "code_editing", "patch_correctness", "relevant_change_omitted", "high", {"read_not_edited": reached}
                )
            )
    elif flag == "loc_partial":
        if unreached:
            if loc["is_cross_module"]:
                cands.append(
                    CandidateError(
                        "code_localization",
                        "cross_file_cross_module_localization",
                        "cross_module_context_missing",
                        "high",
                        {"missed_gold": unreached, "recall": loc["recall"]},
                    )
                )
            else:
                cands.append(
                    CandidateError(
                        "code_localization",
                        "file_level_localization",
                        "missed_relevant_file",
                        "high",
                        {"missed_gold": unreached, "recall": loc["recall"]},
                    )
                )
        if reached:
            cands.append(
                CandidateError(
                    "code_editing", "patch_correctness", "relevant_change_omitted", "high", {"read_not_edited": reached}
                )
            )
        # Editing is no longer structurally exonerated on a partial miss: if the
        # agent edited a gold file but its change reproduces little of the
        # reference change there, the edit content also diverges (an editing fault),
        # carrying the reproduced_fraction severity attribute.
        ov = loc.get("hit_overlap")
        if loc["hit"] and ov is not None and ov.get("reproduced_fraction", 1.0) < 0.5:
            cands.append(
                CandidateError(
                    "code_editing",
                    "patch_correctness",
                    "incorrect_patch",
                    "high" if ov["reproduced_fraction"] == 0 else "medium",
                    _editing_evidence(loc),
                )
            )
    elif flag == "right_place_wrong_change":
        # Agent edited every gold file (recall 1.0); failure is deductively an
        # editing-correctness fault. The old def/class symbol-overlap heuristic is
        # NOT used to raise a competing localization fault here — it caused the
        # editing->localization inversion. Function-level nuance is reported in the
        # editing evidence chain via gold-vs-agent line overlap instead, surfaced as
        # the reproduced_fraction severity attribute (not a separate error type).
        # v0.32 confidence split: a NEAR-GOLD patch (fraction >= 0.8) that still
        # fails is the weakest content-divergence evidence in the family — the
        # failure may live in the residual delta, extra edits, or the evaluation
        # itself (BA-HARNESS-1 suspects sit here) — so it carries medium, not
        # high. Previously the weakest evidence held the strongest confidence.
        ov = loc.get("hit_overlap")
        near_gold = (
            ov is not None
            and ov.get("reproduced_fraction") is not None
            and ov["reproduced_fraction"] >= NEAR_GOLD_FRACTION
        )
        cands.append(
            CandidateError(
                "code_editing",
                "patch_correctness",
                "incorrect_patch",
                "medium" if near_gold else "high",
                _editing_evidence(loc),
            )
        )
    elif flag == "no_source_edit":
        # Failed without producing a source fix at all. patch_state distinguishes
        # 'missing' / 'empty' / 'unparseable' (a FORMAT failure) / 'test_only'
        # (wrote a repro test but no fix) — previously all four rendered as the
        # same misleading "empty/absent patch" prose.
        #
        # The trajectory further splits WHY: an agent that produced no patch and
        # never investigated (no test run, <3 files read — the complement of the
        # behaviors.analysis_paralysis 'investigated' arm) DISENGAGED —
        # `gave_up_early` — while one that did investigate carries the generic
        # `no_patch_produced` (analysis-paralysis-shaped). Both stay objectively
        # editing/output faults (upstream abandonment is the judge layer's call);
        # the capability assignment for gave_up_early is interpretive enough that
        # it is arbiter-eligible and carries associational strength.
        base_ev = {
            "patch_state": loc.get("patch_state", "missing"),
            "edited_test_files": loc.get("edited_test_files", []),
            "gold_tests_edited": loc.get("gold_tests_edited", []),
        }
        investigated = t is not None and (
            t.ran_any_test or getattr(t, "n_test_attempts", 0) > 0 or len(t.attempted_read_set or t.read_set) >= 3
        )
        # A run KILLED before it could investigate (dangling final tool_call)
        # did not disengage — killed ≠ gave up; it stays no_patch_produced.
        if t is not None and not investigated and not getattr(t, "truncated_run", False):
            cands.append(
                CandidateError(
                    "code_editing",
                    "patch_correctness",
                    "gave_up_early",
                    "high",
                    {
                        **base_ev,
                        "not_investigated": True,
                        "ran_any_test": t.ran_any_test,
                        "files_read": len(t.read_set),
                        "n_steps": t.n_steps,
                    },
                )
            )
        else:
            cands.append(
                CandidateError(
                    "code_editing",
                    "patch_correctness",
                    "no_patch_produced",
                    "high",
                    {"note": "no source patch produced", **base_ev},
                )
            )

    # Absolute over-edit (v0.37, the promoted v0.9 deferral): >= OVEREDIT_EXTRA_LOC
    # changed LOC in non-gold SOURCE files on a failed case (RR 2.38, Holm
    # p=0.005 — scripts/calibrate_overedit.py). MEDIUM: associational, an
    # editing-SCOPE fault that co-exists with whatever the flag-branch blamed
    # (mass, not file spread — the file-count cuts calibrated flat).
    if loc.get("extra_src_loc", 0) >= OVEREDIT_EXTRA_LOC:
        cands.append(
            CandidateError(
                "code_editing",
                "edit_scope",
                "excessive_extra_edits",
                "medium",
                {
                    "extra_src_loc": loc["extra_src_loc"],
                    "extra_files": loc.get("extra_files", []),
                    "threshold": OVEREDIT_EXTRA_LOC,
                },
            )
        )

    # --- eval+ tier (v0.39): the official harness's PER-TEST record -----------
    # Deductive grader facts, present only when data/pertest/<agent>/<id>.json
    # exists (data/discovery/pertest_pilot.md). The boolean outcome already
    # licensed a fault on this case; these refine WHICH editing fact it is — so
    # emission is gated on data availability, not on the cohort RR gate (the
    # solved side is f2p == 1.0 by definition: there is nothing to contrast).
    # A patch the harness could not apply carries no test information.
    if pertest and pertest.get("patch_applied"):
        frac = pertest.get("f2p_pass_fraction")
        broken = list(pertest.get("p2p_broken") or [])
        shape = pertest.get("shape")
        # An explicitly empty F2P set also cannot support gating-failure blame.
        if frac == 1.0 or pertest.get("f2p_total") == 0:
            gone = sorted({c.error_type for c in cands if c.error_type in _GATING_FAIL_TYPES})
            cands = [c for c in cands if c.error_type not in _GATING_FAIL_TYPES]
        else:
            gone = []
        if broken:
            ev = {
                "p2p_broken": broken,
                "n_p2p_broken": len(broken),
                "f2p_pass_fraction": frac,
                "shape": shape,
                "superseded_types": gone,
                "supersedes_incorrect_patch": bool(gone),
            }
            cands.append(CandidateError("code_editing", "regression_risk_control", "regression_introduced", "high", ev))
        if frac is not None and 0 < frac < 1:
            # The per-test record REFINES the coarse editing fact: some gating
            # tests pass, so "content diverges" (incorrect_patch) is replaced
            # by "covers some cases, not all" with the fraction as severity —
            # replaced, never added (v0.40). When no editing-correctness fault
            # was licensed (a localization miss explains the failure), the
            # fraction is an ATTRIBUTE of the licensed candidates, not a new
            # cause: a record's mere existence must not manufacture rival
            # editing mass and flip a verdict the boolean tier had settled.
            if any(c.error_type == "incorrect_patch" for c in cands):
                cands = [c for c in cands if c.error_type != "incorrect_patch"]
                cands.append(
                    CandidateError(
                        "code_editing",
                        "patch_correctness",
                        "partial_fix_only",
                        "high",
                        {
                            "f2p_pass_fraction": frac,
                            "f2p_passed": list(pertest.get("f2p_passed") or []),
                            "f2p_failed": list(pertest.get("f2p_failed") or []),
                            "shape": shape,
                            "supersedes_incorrect_patch": True,
                        },
                    )
                )
            else:
                for c in cands:
                    if c.error_type in _GATING_FAIL_TYPES:
                        c.evidence["f2p_pass_fraction"] = frac
                        c.evidence["shape"] = shape
        # Verification side of a regression: the broken tests EXIST in the repo
        # (PASS_TO_PASS), so they were locally runnable — yet no selector-free
        # suite ran and no test command ever named them. Contributing role
        # (code_verification), medium: the regression surface was never
        # checked. Disjoint from regression_introduced (the editing fact).
        # Gated (v0.40) on ran_any_test — a run that executed NO tests is
        # already blamed once by relevant_test_not_run — and on the named-test
        # question being ANSWERABLE: a docstring-form id names no module, so
        # names_any_test returns None and nothing is emitted (unknown ≠ unnamed).
        if (
            broken
            and t is not None
            and bool(gold.opportunities.get("code_verification"))
            and getattr(t, "ran_any_test", False)
            and not getattr(t, "ran_suite_wide", False)
            and names_any_test(broken, getattr(t, "test_commands", [])) is False
        ):
            cands.append(
                CandidateError(
                    "code_verification",
                    "regression_testing",
                    "regression_test_missing",
                    "medium",
                    {
                        "p2p_broken": broken,
                        "ran_suite_wide": False,
                        "broken_tests_referenced": False,
                        "ran_any_test": True,
                    },
                )
            )

    # --- Trajectory-derived capabilities: Verification, Self-Repair, Tool Use ---
    if t is not None:
        if bool(gold.opportunities.get("code_verification")):
            # Test-EXPECTATION rewrite (v0.39; replaces gating_test_modified):
            # existing assertion lines in a test file were replaced by new
            # assertions — the agent changed what the tests expect rather than
            # the code. Calibrated (scripts/calibrate_test_rewrite.py): the
            # rewrite discriminates failed from solved runs while merely
            # TOUCHING a gold gating test (RR 1.01) or removing existing lines
            # (RR 1.14) does not — those stay signals. Two arms in one type:
            # gold gating files (the harness resets them → a local green was
            # self-deception) and any other test file (NOT reset → the rewritten
            # expectation runs modified at eval and can mask a regression).
            # Medium, contributing: legitimate expectation updates exist. The
            # fact needs only patch+gold, but it is emitted with the verification
            # slice so its blame mass always lands in a SCORED capability.
            if loc.get("assert_rewritten_files"):
                after_fail = sorted(
                    set(loc["assert_rewritten_files"]) & set(getattr(t, "test_files_edited_after_failure", []) or [])
                )
                cands.append(
                    CandidateError(
                        "code_verification",
                        "verification_integrity",
                        "test_expectation_rewritten",
                        "medium",
                        {
                            "assert_rewritten_files": loc["assert_rewritten_files"],
                            "assert_rewritten_gold_tests": loc.get("assert_rewritten_gold_tests", []),
                            "assert_rewritten_other_tests": loc.get("assert_rewritten_other_tests", []),
                            "assert_net_removed_files": loc.get("assert_net_removed_files", []),
                            "after_observed_failure": bool(after_fail),
                            "rewritten_after_failure": after_fail,
                            "gold_tests_edited": loc.get("gold_tests_edited", []),
                        },
                    )
                )
            if not t.ran_any_test:
                cands.append(
                    CandidateError(
                        "code_verification", "test_selection", "relevant_test_not_run", "high", {"ran_any_test": False}
                    )
                )
            elif (
                getattr(gold, "gating_tests_visible", True)
                and not t.fail_to_pass_referenced
                # v0.36 soundness guard (the v0.32 class): a selector-free
                # suite run (bare pytest / tox / make test) ran everything
                # locally available, so a test-SELECTION fault ("did not run
                # the relevant test") is not established — the systematic
                # bias against suite-running agents. Blame elsewhere (e.g.
                # no_test_after_final_edit) is unaffected.
                and not getattr(t, "ran_suite_wide", False)
            ):
                cands.append(
                    CandidateError(
                        "code_verification",
                        "test_selection",
                        "relevant_test_not_run",
                        "medium",
                        {
                            "ran_any_test": True,
                            "fail_to_pass_referenced": False,
                            "ran_suite_wide": False,
                            "n_test_runs": t.n_test_runs,
                        },
                    )
                )
            if t.claimed_success_final and getattr(gold, "gating_tests_visible", True):
                if not t.fail_to_pass_referenced:
                    # NOT suppressed by ran_suite_wide (unlike the selection
                    # arm): per §0ak a green suite run typically executed the
                    # BASE gating file, so it does not SUPPORT the claim w.r.t.
                    # the gating tests — but the fact is carried as evidence.
                    cands.append(
                        CandidateError(
                            "code_verification",
                            "final_claim_consistency",
                            "unsupported_success_claim",
                            "medium",
                            {
                                "claimed_success_final": True,
                                "fail_to_pass_referenced": False,
                                "ran_suite_wide": getattr(t, "ran_suite_wide", False),
                            },
                        )
                    )
                elif getattr(t, "last_test_failed", False):
                    # The taxonomy anchor requires a FAILING output followed by the
                    # claim. The agent's last observed test result was a failure and
                    # it still declared success -> a real misinterpretation.
                    cands.append(
                        CandidateError(
                            "code_verification",
                            "result_interpretation",
                            "test_result_misinterpreted",
                            "medium",
                            {
                                "claimed_success_final": True,
                                "ran_gating_test": True,
                                "last_test_failed": True,
                                # v0.35 specificity attribute (fire condition unchanged —
                                # semantics changes are calibration-gated): whether the
                                # failing-then-claimed shape holds on the GATING-named
                                # run itself, not just the last test of any kind.
                                "gating_run_observed": getattr(t, "gating_run_observed", False),
                                "last_gating_run_failed": getattr(t, "last_gating_run_failed", False),
                            },
                        )
                    )
                # else: gating test referenced, last observed run did NOT fail, agent
                # truthfully reported success, eval still failed -> local-vs-eval
                # divergence, NOT a misinterpretation. No fault. (v0.33: corpus
                # replay showed this shape is 22.8% of failed cases, dominated by
                # under-verification — do NOT read it as environment evidence;
                # the affirmative environment fault_side arm was rejected on it,
                # DESIGN.md §0ai.)
            # "Verified early, edited after": the run DID test at some point but no
            # test followed the LAST source edit, so the submitted state was never
            # verified. Disjoint from relevant_test_not_run/high (requires no tests
            # at all) and deferred to repair_not_reverified when a failure event
            # already covers the same gap (avoid double-blaming one fact).
            if t.ran_any_test and getattr(t, "no_test_after_final_edit", False) and not t.repair_not_reverified:
                cands.append(
                    CandidateError(
                        "code_verification",
                        "final_verification",
                        "no_test_after_final_edit",
                        "medium",
                        {"ran_any_test": True, "no_test_after_final_edit": True, "n_test_runs": t.n_test_runs},
                    )
                )
            # "Wrote the reproduction, never executed it": the agent authored a
            # test version during the run but lacks observed post-edit execution. Gated
            # on ran_any_test — when NO tests ran at all, relevant_test_not_run
            # (high) already owns the whole fact.
            if t.ran_any_test and getattr(t, "authored_test_never_run", False):
                cands.append(
                    CandidateError(
                        "code_verification",
                        "authored_test_execution",
                        "authored_test_never_run",
                        "medium",
                        {
                            "authored_test_files": getattr(t, "authored_test_files", []),
                            "unrun_authored_test_files": (
                                getattr(t, "authored_test_unrun_files", []) or getattr(t, "authored_test_files", [])
                            ),
                            "verification_scope": "observed_post_edit_execution_in_editing_thread",
                            "authored_test_never_run": True,
                        },
                    )
                )
        if t.encountered_failure:  # Self-Repair opportunity only when a failure arose
            if t.repair_not_reverified:
                cands.append(
                    CandidateError(
                        "self_repair_loop",
                        "re_verification_after_repair",
                        "repair_not_reverified",
                        "high",
                        {"edited_after_failure_without_retest": True},
                    )
                )
            if t.repeated_ineffective_attempt:
                cands.append(
                    CandidateError(
                        "self_repair_loop",
                        "avoid_repeated_ineffective_attempt",
                        "repeated_ineffective_attempt",
                        "medium",
                        {"same_file_re_edited_after_failure": True, "n_failure_events": t.n_failure_events},
                    )
                )
            if getattr(t, "edit_reverted_own_change", False):
                # The agent swapped an edit's replacement content back to that
                # edit's ORIGINAL content on the same file: it undid its own
                # change — repair thrash (calibrated v0.27: 5.5% of failed vs
                # 2.6% of solved runs, RR 2.10 at full content coverage).
                cands.append(
                    CandidateError(
                        "self_repair_loop",
                        "avoid_self_reversion",
                        "reverted_own_change",
                        "medium",
                        {"edit_reverted_own_change": True, "n_failure_events": t.n_failure_events},
                    )
                )
        if t.tool_failure_not_retried:  # Tool Use mechanics
            cands.append(
                CandidateError(
                    "tool_use",
                    "tool_failure_recovery",
                    "tool_failure_not_retried",
                    "medium",
                    {"tool_error_without_followup": True},
                )
            )
        if getattr(t, "repeated_patch_apply_failure", False):
            # >=2 edit/write results with a patch-failure signature: the edit
            # mechanics are looping on a diff that does not apply (the
            # Codex-shaped apply_patch loop).
            cands.append(
                CandidateError(
                    "tool_use",
                    "edit_mechanics",
                    "repeated_patch_apply_failure",
                    "medium",
                    {"repeated_patch_apply_failure": True, "tool_error_subtypes": t.tool_error_subtypes},
                )
            )
        if getattr(t, "hallucinated_path_unrecovered", False):
            # The same nonexistent path drew an invalid_target error twice or
            # more: the first "no such file" did not update the agent's world
            # model. A one-off typo (recovered on the next call) is NOT this.
            cands.append(
                CandidateError(
                    "tool_use",
                    "world_model_update",
                    "hallucinated_path_unrecovered",
                    "medium",
                    {"hallucinated_path_unrecovered": True, "tool_error_subtypes": t.tool_error_subtypes},
                )
            )

    return cands


def detect(
    instance_id: str,
    agent: str,
    use_trajectory: bool = True,
    patch_override=None,
    resolved_override=_UNSET,
    trajectory_override=None,
    use_judge: bool | None = None,
    gold_override=None,
    pertest_override=_UNSET,
    pertest_log_override=_UNSET,
) -> dict:
    """Detect candidate errors. The *_override params let validation harnesses
    supply isolated inputs, including synthetic metadata and executed fixtures.

    use_judge: None -> auto (on when judge verdicts exist for this agent, off for
    validation injections); True/False to force. The judge adds the two
    judgment-heavy capabilities (requirement_understanding, task_planning) from
    cached verdicts — no API key needed to read them back."""
    gold = gold_override if gold_override is not None else build_gold_reference(instance_id)
    patch = patch_override if patch_override is not None else load_agent_patch(agent, instance_id)
    # Repartition injected and loaded patches against this task's explicit policy.
    from dataclasses import replace
    from .diffutil import is_test_file

    if patch is not None:
        profile = gold.benchmark_profile or None
        patch = replace(
            patch,
            edited_source_files=[f for f in patch.edited_files if not is_test_file(f, profile)],
            edited_test_files=[f for f in patch.edited_files if is_test_file(f, profile)],
        )
    res = outcome_resolved(agent, instance_id) if resolved_override is _UNSET else resolved_override

    # Trajectory signals come first now: the read-set grounds the localization vs
    # editing-omission split (a gold file the agent READ but did not edit is not a
    # search miss). Without a trajectory reachability stays unknown.
    t = None
    if use_trajectory or trajectory_override is not None:
        t = (
            compute_tsig(trajectory_override, gold)
            if trajectory_override is not None
            else signals_cached(agent, instance_id, gold)
        )
    read_set = set(t.read_set) if t is not None else None

    loc = localization_signals(
        gold, patch, res, read_set=read_set, unobserved_read_set=t.unobserved_read_set if t else None
    )
    flag = loc["flag"]

    # eval+ tier (v0.39): the per-test harness record, when one exists. Planted /
    # synthetic inputs (any *_override) never read the real per-test data — a
    # mutated patch's grader record would describe the UNMUTATED patch.
    overridden = patch_override is not None or resolved_override is not _UNSET or trajectory_override is not None
    if pertest_override is not _UNSET:
        pt = pertest_override
    elif overridden:
        pt = None
    else:
        pt = load_pertest(agent, instance_id)
    cands = objective_candidates(gold, loc, t, res, pertest=pt)

    # NOTE: over_editing is intentionally NOT emitted as a failure error.
    # Calibration showed edit_size_ratio does not discriminate solved vs failed
    # (median 6.7x vs 7.5x; identical fire rates). It is kept as a behavioral
    # annotation in signals["over_editing"] only. See awe/calibrate.py.
    signals = {k: loc[k] for k in ("recall", "precision", "edit_size_ratio")}
    signals["over_editing_behavioral"] = loc["over_editing"]
    signals["extra_src_loc"] = loc.get("extra_src_loc", 0)
    signals["read_set_available"] = loc.get("read_set_available", False)
    signals["missed_read_unknown"] = loc.get("missed_read_unknown", [])
    signals["patch_state"] = loc.get("patch_state")
    signals["benchmark_profile"] = gold.benchmark_profile
    # Gold gating-test files the agent's patch touches — the oracle-touch
    # surface (verifying against a self-modified gating test). A SIGNAL on every
    # case (v0.39: no longer a blamed fault — its fact calibrated flat, RR 1.01);
    # the calibrated rewrite form below grounds `test_expectation_rewritten`.
    signals["gold_tests_edited"] = loc.get("gold_tests_edited", [])
    signals["assert_rewritten_files"] = loc.get("assert_rewritten_files", [])
    # eval+ tier: per-test harness record (None/absent → boolean tier only).
    signals["pertest_available"] = bool(pt)
    if pt:
        signals["eval_status"] = pt.get("eval_status")
        signals["f2p_pass_fraction"] = pt.get("f2p_pass_fraction")
        signals["p2p_broken"] = list(pt.get("p2p_broken") or [])
        signals["pertest_shape"] = pt.get("shape")
        signals["pertest_outcome_conflict"] = pt.get("outcome_conflict", False)
        # v0.42 (§0ba): the grader-log audit refines eval_status from the
        # harness's own output — a test-patch apply failure or a crash before
        # the runner summary is invisible in report.json. Only real cases have
        # one (synthetic pertest overrides never reach the disk).
        if pertest_log_override is not _UNSET:
            lg = pertest_log_override
        elif pertest_override is not _UNSET or overridden:
            lg = None
        else:
            lg = load_pertest_log(agent, instance_id)
        if lg:
            if lg.get("eval_status"):
                signals["eval_status"] = lg["eval_status"]
            signals["eval_attribution"] = lg.get("attribution")
            signals["eval_crash_kind"] = lg.get("crash_kind")
            signals["test_patch_apply_error"] = lg.get("test_patch_apply_error")
    signals["gave_up_early"] = any(c.error_type == "gave_up_early" for c in cands)
    signals["issue_symbol_overlap"] = loc.get("issue_symbol_overlap", [])
    signals["solved_unrelated_candidate"] = loc.get("solved_unrelated_candidate", False)
    if loc.get("hit_region") is not None:
        signals["region_overlap"] = loc["hit_region"].get("region_overlap")
    if t is not None:
        signals["no_test_after_final_edit"] = getattr(t, "no_test_after_final_edit", False)
        signals["exploration_loop"] = getattr(t, "exploration_loop", False)
        signals["truncated_run"] = getattr(t, "truncated_run", False)

    opportunities = {
        "code_localization": True,
        "code_editing": True,
        # v0.32: trajectory-gated like its siblings — every verification
        # detector needs a trajectory, so a case without one must not open the
        # opportunity (it would contribute a structural zero to the verification
        # rate, silently inflating the score of any agent with partial
        # trajectory coverage; latent on the shipped dataset, 0 missing).
        "code_verification": bool(t) and bool(gold.opportunities.get("code_verification")),
        "self_repair_loop": bool(t and t.encountered_failure),  # encountered opportunity
        "tool_use": t is not None,
    }

    # The BA-HARNESS-1 metric (added source lines only), surfaced on every case
    # so the fault_side grader arm and the manual-review queue share one number.
    signals["gold_added_coverage"] = gold_added_coverage(gold, patch)
    if t is not None:
        signals.update(
            {
                "ran_any_test": t.ran_any_test,
                "last_test_failed": t.last_test_failed,
                "gating_run_observed": getattr(t, "gating_run_observed", False),
                "last_gating_run_failed": getattr(t, "last_gating_run_failed", False),
                "gating_exec_observed": getattr(t, "gating_exec_observed", False),
                "last_gating_exec_failed": getattr(t, "last_gating_exec_failed", False),
                "ran_suite_wide": getattr(t, "ran_suite_wide", False),
                "n_delegations": getattr(t, "n_delegations", 0),
                "delegate_error_result": getattr(t, "delegate_error_result", False),
                "delegate_no_result": getattr(t, "delegate_no_result", False),
                "authored_test_ran_never_failed": getattr(t, "authored_test_ran_never_failed", False),
                "fail_to_pass_referenced": t.fail_to_pass_referenced,
                "claimed_success_final": t.claimed_success_final,
                "hidden_reasoning": t.hidden_reasoning,
                "encountered_failure": t.encountered_failure,
                "encountered_tool_error": t.encountered_tool_error,
                "n_tool_calls": t.n_tool_calls,
                "n_tool_call_errors": t.n_tool_call_errors,
                "tool_error_subtypes": t.tool_error_subtypes,
            }
        )

    # --- Judgment-heavy capabilities (LLM judge): Requirement Understanding,
    #     Task Planning. Read-only here — the verdict was produced by run_judge.
    judge_on = use_judge if use_judge is not None else (not overridden and judge_active(agent))
    if judge_on:
        specs, available = judge_candidates(agent, instance_id)
        signals["judge_available"] = available
        # Opportunity opens ONLY when this case actually has a verdict — an
        # unjudged failure must not count as a 0-blame requirement/planning
        # opportunity (that would deflate the rate and break normalization).
        opportunities["requirement_understanding"] = available
        opportunities["task_planning"] = available and bool(gold.opportunities.get("task_planning"))
        if res is False:
            for spec in specs:
                # A verdict whose capability has NO opportunity on this task (e.g.
                # flawed_plan on a single-file, small-patch task) must not enter
                # the blame distribution: its weight would deflate every other
                # candidate's share during normalization and then vanish from the
                # capability vectors (scoring only appends shares for open
                # opportunities) — silently dropped mass.
                if opportunities.get(spec["capability"], True):
                    cands.append(CandidateError(**spec))

    # Objective requirement-understanding anchor (v0.26, calibration RR=7.95):
    # a failed total localization miss whose patch shares ZERO code identifiers
    # with the issue text — the agent plausibly solved a DIFFERENT problem than
    # the one reported. Emitted only when the requirement capability is being
    # scored (judge coverage opens the opportunity), so its mass always lands in
    # a scored vector; in objective-only runs it stays the
    # `solved_unrelated_candidate` signal. MEDIUM confidence (the token
    # extraction is best-effort), so it can never demote a deductive fact to
    # secondary — and it is arbiter-eligible like every interpretive label.
    if res is False and opportunities.get("requirement_understanding") and loc.get("solved_unrelated_candidate"):
        cands.append(
            CandidateError(
                "requirement_understanding",
                "requirement_comprehension",
                "solved_unrelated_problem",
                "medium",
                {
                    "solved_unrelated_candidate": True,
                    "extra_files": loc.get("extra_files", []),
                    "patch_symbols": sorted(patch.symbols)[:8] if patch and patch.exists else [],
                    "issue_symbol_overlap": loc.get("issue_symbol_overlap", []),
                },
            )
        )

    return {
        "instance_id": instance_id,
        "agent": agent,
        "resolved": res,
        "loc_flag": flag,
        "candidates": [asdict(c) for c in cands],
        "signals": signals,
        "opportunities": opportunities,
    }


def explain(detection: dict) -> dict:
    """STUB for the LLM-explain layer (Phase next). It will, per candidate:
    pick the precise Level-2 dimension, attach trajectory evidence step_ids, and
    rate evidential support — constrained to the rule-emitted error_type set."""
    raise NotImplementedError("LLM-explain layer not yet implemented")
