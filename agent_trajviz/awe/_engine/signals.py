"""Gold-anchored localization / edit signals.

Every signal is conditioned on the objective outcome: divergence from the gold
patch is only evidence of an error when the tests also fail (the gold patch is
one valid solution, not the only one). These flags are CANDIDATES for the
taxonomy detector to explain — they are not final diagnoses."""

from __future__ import annotations

from .gold import GoldReference
from .patches import AgentPatch
from .evidence import (
    gold_overlap,
    region_overlap,
    issue_has_repro,
    issue_code_tokens as _issue_code_tokens,
    is_assertion_line,
)

# Behavioral-only threshold (~75th pctile). Calibration (agent_trajviz.awe._engine.calibrate) found
# edit_size_ratio does NOT discriminate solved vs failed, so over_editing is no
# longer a failure error — just a 'notably large patch' behavioral flag. The
# ratio-to-gold metric is unstable for tiny gold patches.
OVER_EDIT_THRESHOLD = 50.0

# The v0.37 ROBUST over-edit metric — the v0.9 deferral executed: ABSOLUTE
# changed LOC in non-gold SOURCE files. Calibrated (scripts/calibrate_overedit.py,
# 2,498 patched cases): 12.4% of failed vs 3.1% of solved runs at >=50, RR 2.38,
# Holm p=0.005, threshold-stable (RR 2.14–2.38 across 50/100/200); the file-COUNT
# cuts stayed flat — mass discriminates, spread does not. Grounds the
# `excessive_extra_edits` error type. The constant lives in agent_trajviz.awe._engine.evidence (the
# shared-primitives module) so the audit re-deriver and the calibration script
# consume the same number; re-exported here for the detector layer.
from .evidence import OVEREDIT_EXTRA_LOC  # noqa: F401  (re-export)

# issue_has_repro / _issue_code_tokens live in agent_trajviz.awe._engine.evidence (shared primitives —
# the audit re-derives the solved_unrelated anchor from the same functions).


def _n_assertions(lines) -> int:
    """Count assertion lines in a changed-line multiset (Counter/dict: sum the
    counts; legacy set/list fixtures: count members)."""
    if isinstance(lines, dict):
        return sum(n for ln, n in lines.items() if is_assertion_line(ln))
    return sum(1 for ln in (lines or []) if is_assertion_line(ln))


def localization_signals(
    gold: GoldReference, patch: AgentPatch | None, resolved: bool | None, read_set=None, unobserved_read_set=None
) -> dict:
    gold_src = set(gold.gold_files)
    edited_src = set(patch.edited_source_files) if patch and patch.exists else set()

    hit = edited_src & gold_src
    missed_gold = gold_src - edited_src
    extra = edited_src - gold_src

    # Read-set grounding: a missed gold file the agent demonstrably READ was
    # reached — its omission is a selection/editing decision, NOT a search miss.
    # Only gold files neither edited nor read are true localization misses. With
    # no trajectory (read_set is None), reachability is unknown. Reference
    # divergence remains visible but cannot establish a search failure.
    read_set = set(read_set) if read_set is not None else None
    read_set_available = read_set is not None
    if read_set_available:
        missed_reached = missed_gold & read_set  # found but not edited
        missed_unreached = missed_gold - read_set - set(unobserved_read_set or [])  # never reached -> real loc miss
    else:
        missed_reached = set()
        missed_unreached = set()

    recall = len(hit) / len(gold_src) if gold_src else None
    precision = len(hit) / len(edited_src) if edited_src else None
    ratio = patch.patch_loc / gold.gold_patch_loc if patch and patch.exists and gold.gold_patch_loc else None
    # Absolute changed LOC in non-gold source files (v0.37 robust over-edit
    # metric; per-file mass from the parsed diff, 0 when unavailable).
    _floc = getattr(patch, "file_loc", {}) if patch and patch.exists else {}
    extra_src_loc = sum(_floc.get(f, 0) for f in extra)

    # For gold files the agent DID edit, how much of the reference change it
    # reproduced (same added/removed line content). Grounds the editing decision on
    # a partial localization: a low fraction => the edit content also diverges, so
    # editing shares blame; a high fraction => the edit matches and only the miss
    # is at fault. None when we lack the line maps (e.g. synthetic/legacy inputs).
    hit_overlap = None
    if hit and gold.gold_changed_lines and patch and getattr(patch, "changed_lines", None):
        hit_overlap = gold_overlap(gold.gold_changed_lines, patch.changed_lines, sorted(hit))

    # Hunk-RANGE overlap on hit files (old-file coordinates): low region_overlap
    # with a real edit = "same file, DISJOINT region" — localization-within-file
    # nuance as a graded attribute (recovers what the pruned def/class regex tried
    # to capture, deductively).
    hit_region = None
    if hit and getattr(gold, "gold_hunks", None) and patch and getattr(patch, "hunks", None):
        hit_region = region_overlap(gold.gold_hunks, patch.hunks, sorted(hit))

    # How the patch degenerated, when it did — 'missing' (no patch artifact),
    # 'unparseable' (text present, zero files parsed: a FORMAT failure, not an
    # empty submission), 'test_only' (edited only test files — e.g. wrote a repro
    # but no fix), or 'ok'. The old single no_source_edit flag collapsed all
    # three degenerate states into one misleading "empty/absent patch" diagnosis.
    if patch is None or not patch.exists:
        patch_state = "missing"
    elif getattr(patch, "unparseable", False):
        patch_state = "unparseable"
    elif not edited_src and patch.edited_test_files:
        patch_state = "test_only"
    elif not edited_src:
        patch_state = "empty"
    else:
        patch_state = "ok"

    # Agent edits to the GOLD GATING-TEST files — the oracle-touch surface. A
    # SIGNAL on every case (v0.39: no longer a blamed fault — the fact calibrated
    # flat, RR 1.01; the calibrated REWRITE form below is the fault); on a
    # resolved case it feeds the benchmark-integrity audit (awe/fragile.py): the
    # effect on grading depends on the benchmark reset policy; a name alone
    # cannot establish evaluator behavior.
    edited_all = set(patch.edited_files) if patch and patch.exists else set()
    gold_tests_edited = sorted(edited_all & set(gold.test_patch_files))
    # Destructiveness split: REMOVED lines in a gold gating-test file mean the
    # agent altered/deleted EXISTING test content (assertions changed — strong
    # tamper evidence); a file with only ADDED lines is likely a repro appended
    # to the gating file (benign-shaped). Graded attribute, not a separate type.
    _cl = getattr(patch, "changed_lines", {}) if patch and patch.exists else {}
    gold_tests_modified_existing = sorted(f for f in gold_tests_edited if (_cl.get(f) or {}).get("removed"))

    # Test-EXPECTATION rewrite (v0.39; calibrated — scripts/calibrate_test_rewrite.py):
    # per edited TEST file, existing assertion lines were removed AND assertion
    # lines were added — the agent changed what a test expects rather than
    # merely appending a repro. Split by whether the file is a gold gating-test
    # file (the harness resets those → a local green is self-deception) or any
    # other test file (NOT reset → the rewritten expectation runs modified at
    # eval and can mask a regression). The two facts above stay SIGNALS only:
    # touching a gold test at all (RR 1.01) and removing existing lines
    # (RR 1.14) calibrate flat; rewriting assertions discriminates.
    edited_tests = set(patch.edited_test_files) if patch and patch.exists else set()
    assert_rewritten, assert_net_removed = [], []
    for f in sorted(edited_tests):
        cl = _cl.get(f) or {}
        rem = _n_assertions(cl.get("removed"))
        add = _n_assertions(cl.get("added"))
        if rem and add:
            assert_rewritten.append(f)
        if rem > add:
            assert_net_removed.append(f)
    _gold_test_set = set(gold.test_patch_files)

    # Objective requirement-understanding ANCHOR (signal, not yet a blamed error
    # type — promotion needs the judge-cap denominator redesign): on a total
    # localization miss, does the patch share ANY code identifier with the issue
    # text? Zero overlap while the patch defines symbols = the agent plausibly
    # solved a DIFFERENT problem than the one reported. Surfaced on the loc_miss
    # faults and left to the judge/arbiter/human to adjudicate.
    issue_tokens = _issue_code_tokens(gold.problem_statement)
    patch_syms = set(getattr(patch, "symbols", []) or []) if patch and patch.exists else set()
    edited_names = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in edited_src}
    issue_symbol_overlap = sorted((patch_syms | edited_names) & issue_tokens)

    # Outcome-conditioned candidate flag.
    if patch is None or not patch.exists or not edited_src:
        flag = "no_source_edit"
    elif resolved is None:
        flag = "unknown_outcome"
    elif resolved:
        flag = "resolved_exact" if not missed_gold else "resolved_alt_solution"
    else:
        if not gold_src:
            # Reference change touches no source file (gold test-only): the
            # loc-flag vocabulary doesn't apply — `not hit` would be vacuously
            # true and mint a phantom loc_miss (v0.32 guard; no such instance
            # exists in the shipped dataset, this is a forward guard).
            flag = "gold_test_only"
        elif not hit:
            flag = "loc_miss"  # found none of the gold files
        elif missed_gold:
            flag = "loc_partial"  # found some, missed some (cross-module candidate)
        else:
            flag = "right_place_wrong_change"  # right files, tests still fail (editing candidate)

    return {
        "flag": flag,
        "resolved": resolved,
        "recall": recall,
        "precision": precision,
        "edit_size_ratio": ratio,
        "over_editing": bool(ratio and ratio > OVER_EDIT_THRESHOLD),
        "extra_src_loc": extra_src_loc,
        "gold_files": sorted(gold_src),
        "edited_source_files": sorted(edited_src),
        "hit": sorted(hit),
        "hit_overlap": hit_overlap,
        "hit_region": hit_region,
        "issue_symbol_overlap": issue_symbol_overlap,
        "issue_has_repro": issue_has_repro(gold.problem_statement),
        # candidate flag for human/judge review — NOT a blamed error type.
        # v0.32 guards: the issue must CONTAIN code identifiers (issue_tokens
        # non-empty) — on an identifier-free issue "zero overlap" is vacuously
        # true and the anchor's premise ('a real fix touches an entity the
        # issue names') is empty; it fired that way on sphinx-doc__sphinx-7757
        # (0 extractable tokens). gold_src must be non-empty too: with a
        # test-only reference change, `not hit` is vacuous.
        "solved_unrelated_candidate": bool(
            resolved is False
            and not hit
            and gold_src
            and edited_src
            and patch_syms
            and issue_tokens
            and not issue_symbol_overlap
        ),
        "missed_gold": sorted(missed_gold),
        "missed_reached": sorted(missed_reached),
        "missed_unreached": sorted(missed_unreached),
        "read_set_available": read_set_available,
        "missed_read_unknown": sorted(missed_gold & set(unobserved_read_set or []))
        if read_set_available
        else sorted(missed_gold),
        "extra_files": sorted(extra),
        "is_cross_module": gold.is_cross_module,
        "patch_state": patch_state,
        "edited_test_files": sorted(patch.edited_test_files) if patch and patch.exists else [],
        "evaluator_test_reset": (gold.benchmark_profile or {}).get("evaluator_test_reset", "unknown"),
        "gold_tests_edited": gold_tests_edited,
        "gold_tests_modified_existing": gold_tests_modified_existing,
        # v0.39 test-expectation rewrite (grounds `test_expectation_rewritten`)
        "assert_rewritten_files": assert_rewritten,
        "assert_rewritten_gold_tests": [f for f in assert_rewritten if f in _gold_test_set],
        "assert_rewritten_other_tests": [f for f in assert_rewritten if f not in _gold_test_set],
        "assert_net_removed_files": assert_net_removed,
    }


# Human-readable note per flag (for reports / sanity).
FLAG_NOTES = {
    "resolved_exact": "solved; edited exactly the gold source files",
    "resolved_alt_solution": "solved a different valid way (not an error)",
    "loc_miss": "FAILED; touched none of the gold files -> localization miss candidate",
    "loc_partial": "FAILED; missed some gold files -> partial/cross-module localization candidate",
    "right_place_wrong_change": "FAILED; right files, wrong change -> editing-correctness candidate",
    "no_source_edit": "no source edit recorded (empty/missing patch)",
    "unknown_outcome": "instance missing from eval set",
}
