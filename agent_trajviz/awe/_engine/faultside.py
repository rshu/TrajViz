"""Case-level fault-side assessment (v0.33).

Raj et al. (arXiv:2607.28802) organize agent failures by interaction edge plus
a *fault side* — which component the repair belongs to. AWE's produced error
types are all agent-capability faults (fault side: model) by construction; the
non-model evidence has lived in annex signatures. This module surfaces those
signatures as one first-class attribute on the case record.

Vocabulary note (the two "harness"es): AWE's BA-HARNESS-1 annex says "harness"
for the *benchmark evaluation apparatus*; in Raj et al.'s component vocabulary
that endpoint is the GRADER, while their "harness" is the agent scaffold
(context/memory/tool plumbing). ``fault_side`` uses the Raj vocabulary:
``grader`` for evaluation-apparatus suspects (the BA-HARNESS-1 queue),
``harness`` for scaffold-side truncation (OP-TRUNC-1).

The grader arm IS the annex predicate — same metric
(``diffutil.gold_added_coverage``, added source lines only), same threshold
(``BA_HARNESS_COVERAGE``), same green gate — shared with
``scripts/harness_suspects.py`` via :func:`is_ba_harness_suspect`, so the
in-record suspects and the manual-review queue are one set by construction.
(The first draft re-derived the shape from ``reproduced_fraction >= 0.8``; the
review replay showed the two sets diverge in both directions — 12 of the 22
queue suspects missed, 5 non-queue cases added via matched *removed* lines —
so the re-implementation was replaced by the shared predicate. The inherited
looseness of the green gate — the last test of ANY kind, not provably the
gating run — is the annex's own documented contract: suspects are queued for
manual review, never auto-exonerated.)

Sides: ``model | harness | grader | environment | owner``. Two are reserved
(in the enum, never assigned today):

- ``owner`` — no detector separates instruction-vs-intent divergence
  (Instruction–Grader Mismatch in Raj et al.; the spec-oracle gap of Jiang V1).
- ``environment`` — an affirmative divergence signature (gating referenced +
  last test green + success claimed, eval failed) was implemented and REJECTED
  at this pass: it fires on 156/685 shipped failed cases (22.8%), because
  ``last_test_failed`` covers the last test of ANY kind, so the shape is
  dominated by under-verification (green subset run), a model-side fault
  (DESIGN.md §0ai). The TIGHT version §0ai asked for — the per-gating-run
  ``gating_run_observed``/``last_gating_run_failed`` trajsignal (schema 14,
  v0.35) — was then built and ALSO rejected by corpus replay: gating-green +
  claim + untruncated fires on 150/685 failed (21.9%, vs the loose 163) and is
  MORE common on solved runs (562/1815, 31%), because a green "gating-named"
  run usually executed the BASE version of the gating file (F2P tests are
  added by the eval's test_patch), so the shape is still under-verification.
  An affirmative environment signature would need proof the gating test
  FUNCTION executed (e.g. the run output naming the test id), not a command
  token match. See DESIGN.md §0ak. The divergence facts stay visible in the
  ``signals`` payload.

Honesty contract: non-model sides annotate, never exonerate — no fault leaves
any count (the BA-HARNESS-1 annex contract). Every gate is fail-closed:
missing or non-strict-bool signal values fail the condition, so a case without
a trajectory lands on the conservative ``model`` default and the payload keeps
unknowns as ``None`` rather than asserted ``False``.
"""

from __future__ import annotations

# The annex's coverage threshold (scripts/harness_suspects.py --cov default
# imports this — one number, both consumers).
BA_HARNESS_COVERAGE = 0.7

SIDES = ("model", "harness", "grader", "environment", "owner")

# Payload keys: every signal a signature reads plus the divergence facts kept
# visible for analysts. Bare .get() throughout — None means unmeasured, never
# asserted-False (a no-trajectory case must not claim "not truncated").
_TRIGGER_KEYS = (
    "truncated_run",
    "ran_any_test",
    "last_test_failed",
    "claimed_success_final",
    "fail_to_pass_referenced",
    "no_test_after_final_edit",
    "gold_added_coverage",
    # v0.35: the tight per-gating-run outcome §0ai asked for —
    # carried in the payload for the manual-review queue and the
    # environment-arm decision; not yet a trigger.
    "gating_run_observed",
    "last_gating_run_failed",
    # v0.42 (§0ba): the grader-log audit's refinement of the
    # per-test eval_status — GRADER-APPLY-1 / GRADER-GOLDDEP-1
    "eval_status",
    "eval_attribution",
)

# eval_status values (scripts/grader_log_audit.py) that the harness's own log
# affirms as evaluation-apparatus failures: the gating tests never existed in
# the run (the official test patch did not apply over the agent's files), or
# could not be imported for a reason only the gold's dependency choice
# explains. Decidable per case from the log — not a trajectory-shaped guess.
GRADER_LOG_STATUSES = {
    "test_patch_apply_failed": (
        "GRADER-APPLY-1",
        "the official TEST patch failed to apply over the "
        "agent's files (git apply is atomic), so the gating tests never "
        "existed in the evaluation run — read off the harness's own log; "
        "adjudicable by a source-only re-evaluation (pertest_harness.py "
        "--source-only)",
    ),
}
GRADER_LOG_ATTRIBUTIONS = {
    "gold_dependency_unverified": (
        "GRADER-GOLDDEP-1",
        "the test module imports a third-party module "
        "that only the gold patch's requirements introduce, so an agent "
        "that solved the issue another way cannot even import the "
        "tests; the fix is unverified, not shown wrong",
    ),
}


def is_ba_harness_suspect(
    ran_any_test, last_test_failed, gold_added_cov, truncated_run, threshold: float = BA_HARNESS_COVERAGE
) -> bool:
    """THE BA-HARNESS-1 suspect predicate, shared by fault_side and the
    manual-review queue generator. A failed case qualifies when the agent ran
    tests, its LAST observed test run was green, the run was not truncated
    (a dangling final test call means the "green" was never observed), and the
    final patch's added-line content covers >= threshold of the gold patch's
    added source lines. Fail-closed: every gate needs the strict bool in the
    qualifying state — 1/'true'/None all disqualify."""
    return (
        ran_any_test is True
        and last_test_failed is False
        and truncated_run is False
        and gold_added_cov is not None
        and gold_added_cov >= threshold
    )


def assess(signals: dict | None) -> dict:
    """Pure: the case-level fault-side assessment for a FAILED case, from the
    detect() signals dict (which carries ``gold_added_coverage`` since v0.33).

    Precedence: truncation first. A truncated run's last-test observation is
    unreliable by construction (the run may have been killed mid-test), so it
    must not mint a grader suspect on an unobserved green — this REVERSES the
    draft's most-specific-first ordering; the shared predicate also excludes
    truncated runs outright, making the two arms disjoint."""
    signals = signals or {}

    if signals.get("truncated_run") is True:
        side, certainty = "harness", "suspect"
        basis = (
            "OP-TRUNC-1: run truncated by context/step budget before "
            "completion (the kill itself is deductive: dangling final "
            "tool_call). The repair-locus attribution is suspect, not "
            "proven — model-side faults may exist independently; scaffold "
            "remedies (compaction/checkpointing) apply regardless"
        )
    elif signals.get("eval_status") in GRADER_LOG_STATUSES:
        tag, why = GRADER_LOG_STATUSES[signals["eval_status"]]
        side, certainty = "grader", "suspect"
        basis = f"{tag}: {why}; queued for adjudication (data/discovery/grader_log_audit.md), faults stay counted"
    elif (
        signals.get("eval_status") == "crash_before_summary"
        and signals.get("eval_attribution") in GRADER_LOG_ATTRIBUTIONS
    ):
        tag, why = GRADER_LOG_ATTRIBUTIONS[signals["eval_attribution"]]
        side, certainty = "grader", "suspect"
        basis = (
            f"{tag}: {why}; gold-sanity decides whether the image lacks the "
            "module (instance unresolvable) or the agent's run lost it "
            "(data/discovery/grader_log_audit.md), faults stay counted"
        )
    elif is_ba_harness_suspect(
        signals.get("ran_any_test"),
        signals.get("last_test_failed"),
        signals.get("gold_added_coverage"),
        truncated_run=signals.get("truncated_run"),
    ):
        cov = signals["gold_added_coverage"]
        side, certainty = "grader", "suspect"
        basis = (
            "BA-HARNESS-1: final patch covers "
            f"{cov:.0%} >= {BA_HARNESS_COVERAGE:.0%} of the gold patch's "
            "added source lines and the agent's own last test run was "
            "green, yet the eval failed — evaluation-apparatus (grader) "
            "suspect; queued for manual review "
            "(data/discovery/harness_suspects.md), faults stay counted"
        )
    else:
        side, certainty = "model", "default"
        basis = "no non-model signature fired; all blamed faults are agent-capability faults"

    return {"side": side, "certainty": certainty, "basis": basis, "signals": {k: signals.get(k) for k in _TRIGGER_KEYS}}
