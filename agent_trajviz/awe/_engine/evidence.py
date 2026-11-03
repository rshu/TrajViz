"""Evidence chains for diagnosed faults.

Each fault label is turned into an auditable *chain*: an ordered list of
observation -> inference -> conclusion links, each pointing at a concrete source
(the gold patch, the agent patch, the test outcome, or a trajectory step), plus
an honest STRENGTH tier so a reader knows whether a label is a deduction, an
association, or a model inference.

Three tiers (the strength of the CAUSAL claim, not merely 'an error exists'):

  - deductive      loc/edit faults grounded in set arithmetic over gold ∩ patch
                   conditioned on the test outcome. 'Edited the right files, tests
                   still fail -> the edit content is wrong' is a logical deduction.
  - associational  verification/self-repair/tool faults: objective trajectory
                   facts, but they failed to *catch/recover*, they did not *cause*
                   the wrong patch -> contributing, association not deduction.
  - model_inferred requirement/planning faults from the LLM judge: evidence-cited
                   and confidence-gated, but an interpretation, not verified.

The `counterfactual` field is the hook for the (deferred) interventional probe:
status is always "not_tested" here; a future execution harness flips it to
"confirmed"/"refuted" by applying the gold change and re-running the tests."""

from __future__ import annotations

import re

# The v0.32 near-gold split. Single home (this module has no internal imports,
# so both detect.py's incorrect_patch confidence rule and the NEAR-GOLD prose
# branch below can import it without cycles) — three drifting hardcoded 0.8s
# was a review finding.
NEAR_GOLD_FRACTION = 0.8

# The v0.37 robust over-edit threshold (absolute changed LOC in non-gold source
# files). Lives here — the shared-primitives module — so signals (detector),
# audit (independent re-deriver), and the calibration script consume the SAME
# constant without audit importing the detector layer.
OVEREDIT_EXTRA_LOC = 50

# Assertion-line classifier (shared primitive, v0.39). The test-expectation
# rewrite signal (signals.py) and its independent audit re-deriver (audit.py)
# must agree on WHAT counts as an assertion line, so the classifier lives here;
# the per-file rewrite STRUCTURE (removed ∧ added assertions in one test file)
# is implemented separately in each layer. Accepts a raw diff body line or the
# `indent|body` norm_line form. Covers assert statements, unittest
# self.assert*, bare assertX( aliases (assertEqual / assertFieldType imported
# or mixin-defined), numpy.testing.assert_*, pytest.raises / bare raises(
# (sympy's idiom) / assertRaises, and the JS-style expect( idiom (Harbor
# multi-language inputs). v0.40 added the bare forms after the PR-56 review
# found sympy rewrites classified as non-assertions; re-calibrated (DESIGN §0aw).
ASSERTION_RE = re.compile(
    r"^(?:\d+\|)?\s*(?:with\s+)?(?:assert\b|self\.assert\w*\(|assert\w+\(|"
    r"(?:np|numpy)\.testing\.assert\w*\(|pytest\.raises\(|raises\(|"
    r"self\.assertRaises\w*\(|expect\()"
)


def is_assertion_line(line: str) -> bool:
    """True iff a changed line (raw diff body or norm_line form) is an assertion."""
    return bool(ASSERTION_RE.match(line or ""))


_GIT = re.compile(r"diff --git a/(.+?) b/(.+)$")

DEDUCTIVE = "deductive"
ASSOCIATIONAL = "associational"
MODEL_INFERRED = "model_inferred"

_STRENGTH_BY_CAP = {
    "code_localization": DEDUCTIVE,
    "code_editing": DEDUCTIVE,
    "code_verification": ASSOCIATIONAL,
    "self_repair_loop": ASSOCIATIONAL,
    "tool_use": ASSOCIATIONAL,
    "requirement_understanding": MODEL_INFERRED,
    "task_planning": MODEL_INFERRED,
}


def strength_of(capability: str, error_type: str | None = None, *, default=MODEL_INFERRED):
    """The evidence-strength tier a fault carries (deductive / associational /
    model_inferred). Unknown capabilities default to the most cautious tier.
    Pass ``default=None`` to distinguish an unknown taxonomy entry from a known
    model-inferred tier, e.g. when rendering assessment provenance.

    ``relevant_change_omitted`` and ``gave_up_early`` are the editing faults that
    are NOT deductive: the underlying facts (read-but-not-edited; no patch and no
    investigation) are objective, but assigning them to code_editing (vs
    comprehension/planning/engagement) is an interpretation — both are
    arbiter-eligible in rootcause for exactly that reason, so the trust profile
    must not count them in the deductive share.

    ``solved_unrelated_problem`` is the converse case: an OBJECTIVE anchor
    (zero patch∩issue identifier overlap on a total miss; calibrated RR=7.95)
    living in a judge capability — associational, NOT model_inferred: no model
    decided it."""
    if error_type in ("relevant_change_omitted", "gave_up_early", "solved_unrelated_problem"):
        return ASSOCIATIONAL
    # excessive_extra_edits (v0.37) is associational for a THIRD reason: the
    # fact (extra source LOC >= threshold) is deductive, but its status as a
    # FAULT rests on the cohort calibration (RR 2.38) — a correlational
    # license, not a per-case deduction. It must not inherit code_editing's
    # deductive tier (v0.37 review fix: it did, contradicting its own audit
    # record's associational tier).
    if error_type == "excessive_extra_edits":
        return ASSOCIATIONAL
    # test_expectation_rewritten (v0.39) is code_verification (associational
    # by capability) and, like excessive_extra_edits, licensed by calibration.
    return _STRENGTH_BY_CAP.get(capability, default)


_STRENGTH_BASIS = {
    DEDUCTIVE: "set arithmetic over gold ∩ agent patch, conditioned on the test "
    "outcome; reproducible with no model and no heuristic.",
    ASSOCIATIONAL: "an objective trajectory fact, but a contributing 'failed to "
    "catch/recover' role — associated with the failure, not its upstream cause.",
    MODEL_INFERRED: "an outcome-blinded LLM judgment over a closed taxonomy with cited "
    "steps; evidence-grounded and confidence-gated, but an interpretation.",
}

# The three types strength_of() special-cases into ASSOCIATIONAL are there for a
# DIFFERENT reason than the failed-to-catch role: the underlying fact is objective
# (and may be the upstream cause), but WHICH capability it indicts is interpretive.
# Giving them the generic associational basis made 41 shipped chains claim
# "primary cause" while their own basis said "not its upstream cause" (v0.31).
_INTERPRETIVE_ETS = ("relevant_change_omitted", "gave_up_early", "solved_unrelated_problem")
_INTERPRETIVE_BASIS = (
    "an objective, re-derivable fact counted associational because the CAPABILITY "
    "assignment is interpretive (arbiter-eligible), not because the fact is "
    "correlational — it may well be the upstream cause."
)


_CALIBRATED_BASIS = (
    "a deductively re-derivable fact whose FAULT status rests on cohort "
    "calibration (solved-vs-failed risk ratio, Holm-gated) — associated with "
    "failure across the corpus, not proof of cause on any single case."
)


# eval+ tier (v0.39): the fact is the official harness's per-test record, not
# gold∩patch set arithmetic — deductive, but from a different raw input.
_GRADER_BASIS = (
    "per-test outcome from the official evaluation harness report (FAIL_TO_PASS / "
    "PASS_TO_PASS status per test id) joined to the agent's final patch; "
    "reproducible with no model and no heuristic."
)

_CALIBRATED_ETS = ("excessive_extra_edits", "test_expectation_rewritten")
_GRADER_ETS = ("partial_fix_only", "regression_introduced")


def strength_basis_for(strength: str, error_type: str | None = None) -> str:
    if error_type in _INTERPRETIVE_ETS:
        return _INTERPRETIVE_BASIS
    if error_type in _CALIBRATED_ETS:
        return _CALIBRATED_BASIS
    if error_type in _GRADER_ETS:
        return _GRADER_BASIS
    return _STRENGTH_BASIS[strength]


def changed_lines(diff_text: str) -> dict:
    """{file: {"added": Counter[str], "removed": Counter[str]}} of changed line
    *content* from a unified diff. MULTISETS, not sets: repeated identical lines
    (common in real patches) must each count, or the overlap denominator
    undercounts and one agent line can satisfy N gold occurrences.

    Built on diffutil.parse_unified_diff — the ONE hardened parser (hunk-count
    budgets + header-pair recovery). The previous standalone scan here treated
    added '++ ...' body lines as file headers (misattributing what followed)
    and dropped removed '-- ...' lines (SQL/Lua comments) entirely."""
    from collections import Counter
    from .diffutil import parse_unified_diff

    out: dict = {}
    for fd in parse_unified_diff(diff_text or ""):
        entry = out.setdefault(fd.path, {"added": Counter(), "removed": Counter()})
        for ln in fd.added_lines:
            s = _norm_line(ln)
            if s:
                entry["added"][s] += 1
        for ln in fd.removed_lines:
            s = _norm_line(ln)
            if s:
                entry["removed"][s] += 1
    return out


def norm_line(s: str) -> str:
    """Public alias of the changed-line normalization — adapters use it to store
    edit content in the SAME key space the gold/patch changed-line maps use, so
    trajectory-edit vs gold/final-patch comparisons are apples-to-apples."""
    return _norm_line(s)


def _norm_line(s: str) -> str:
    """Normalize a changed line for overlap comparison: collapse INTERNAL
    whitespace but keep the leading-indent depth as part of the key — in Python,
    `return x` at the wrong nesting level is a different change, and erasing
    indentation inflated reproduced_fraction for exactly those wrong edits.
    (Tabs expand to 4 spaces so a tab/space convention mismatch, rare within one
    file, does not spuriously break matches.)"""
    body = re.sub(r"\s+", " ", s.strip())
    if not body:
        return ""
    indent = len(s[: len(s) - len(s.lstrip("\t "))].expandtabs(4))
    return f"{indent}|{body}"


def gold_overlap(gold_changed: dict, agent_changed: dict, files) -> dict | None:
    """For files the agent also edited, what fraction of gold's changed lines the
    agent reproduced (same added line, same removed line, multiset-min counted).
    Low overlap on a file the agent *did* touch = same place, materially
    different change. reproduced_fraction is rounded to 4 decimals so a small
    non-zero reproduction (>= 0.005%, i.e. any realistic SWE-bench patch) does
    not round to the exact-0.0 that detect treats as 'wholly different change,
    high confidence'."""
    from collections import Counter

    def _c(x) -> Counter:
        # Tolerate legacy set/list inputs (synthetic fixtures, injected patches).
        return x if isinstance(x, Counter) else Counter(x or [])

    g_total = shared = 0
    per_file = {}
    for f in files:
        g = gold_changed.get(f)
        a = agent_changed.get(f)
        if not g:
            continue
        g_add, g_rem = _c(g["added"]), _c(g["removed"])
        gt = sum(g_add.values()) + sum(g_rem.values())
        if gt == 0:
            continue
        a_add = _c(a["added"]) if a else Counter()
        a_rem = _c(a["removed"]) if a else Counter()
        sh = sum((g_add & a_add).values()) + sum((g_rem & a_rem).values())
        per_file[f] = round(sh / gt, 4)
        g_total += gt
        shared += sh
    if g_total == 0:
        return None
    return {"reproduced_fraction": round(shared / g_total, 4), "per_file": per_file}


# --- issue-text primitives (shared: signals uses them to build the anchor, the
# --- audit re-derives from them; evidence.py is the documented shared-primitive
# --- module, like _norm_line / changed_lines)

_REPRO_RES = (
    re.compile(r"```"),
    re.compile(r"^\s*>>> ", re.MULTILINE),
    re.compile(r"steps to reproduce|how to reproduce|to reproduce\b", re.IGNORECASE),
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"minimal (?:working |reproducible )?example|\bmcve\b|\bmwe\b", re.IGNORECASE),
)

# Code-like identifiers in issue text: backticked spans, snake_case, CamelCase,
# dotted paths, call-sites. Plain English words excluded by construction.
_ISSUE_TOKEN_RES = (
    re.compile(r"`([^`\n]{3,60})`"),
    re.compile(r"\b([a-z_][a-z0-9]*_[a-z0-9_]+)\b"),
    re.compile(r"\b([A-Z][a-z0-9]+[A-Z]\w*)\b"),
    re.compile(r"\b([\w.]+\.[a-z_]\w*)\s*\("),
    re.compile(r"\b([a-zA-Z_]\w{2,})\s*\("),
)


def issue_has_repro(text: str) -> bool:
    """Does the issue PROVIDE reproduction material (fenced code, doctest lines,
    explicit repro instructions, a traceback)? High-precision cues only."""
    return any(rx.search(text or "") for rx in _REPRO_RES)


def issue_code_tokens(text: str) -> set:
    """Best-effort code identifiers mentioned in the problem statement; tokens
    >=4 chars so short generic words never match."""
    toks: set = set()
    for rx in _ISSUE_TOKEN_RES:
        for m in rx.findall(text or ""):
            for t in re.findall(r"[A-Za-z_]\w{3,}", m):
                toks.add(t)
    return toks


def region_overlap(gold_hunks: dict, agent_hunks: dict, files, margin: int = 3) -> dict | None:
    """Hunk-RANGE overlap in old-file coordinates (both diffs share the base
    commit): what fraction of the gold change's line span falls inside a region
    the agent also touched (± ``margin`` context lines)?

    Complements reproduced_fraction (content): low region_overlap on a hit file
    means "same file, DISJOINT region" — the sharper localization-within-file
    nuance the pruned def/class-regex heuristic tried (and failed) to capture.
    A graded severity attribute, not an error type (v0.9 discipline)."""
    g_total = covered = 0
    per_file = {}
    for f in files:
        gh = gold_hunks.get(f) or []
        ah = agent_hunks.get(f) or []
        g_lines = {ln for start, cnt in gh for ln in range(start, start + max(cnt, 1))}
        if not g_lines:
            continue
        a_lines = set()
        for start, cnt in ah:
            a_lines.update(range(max(1, start - margin), start + max(cnt, 1) + margin))
        cov = len(g_lines & a_lines)
        per_file[f] = round(cov / len(g_lines), 4)
        g_total += len(g_lines)
        covered += cov
    if g_total == 0:
        return None
    return {"region_overlap": round(covered / g_total, 4), "per_file": per_file}


def _link(kind: str, statement: str, source: str, quotes: list | None = None) -> dict:
    d = {"kind": kind, "statement": statement, "source": source}
    if quotes:
        d["quotes"] = quotes
    return d


def build_chain(
    cand: dict,
    detection: dict,
    gold,
    gold_changed: dict,
    agent_changed: dict,
    blame_weight: float,
    is_primary: bool,
    step_text: dict | None = None,
) -> dict:
    """Construct the evidence chain for one candidate fault. ``step_text`` (optional)
    maps trajectory step_id -> verbatim text; when present, links that cite specific
    steps carry a ``quotes`` field (the IG-style trace-quote + mechanism), so a reader
    can see the exact trajectory text the judgment rests on."""
    cap = cand["capability"]
    et = cand["error_type"]
    ev = cand.get("evidence", {})
    strength = strength_of(cap, et)
    role = (
        "primary cause"
        if is_primary
        else (
            "contributing factor" if cap in ("code_verification", "self_repair_loop", "tool_use") else "secondary cause"
        )
    )
    links: list[dict] = []
    counterfactual = {"status": "not_tested", "what_would_confirm": ""}

    if cap == "code_localization":
        missed = ev.get("missed_gold") or gold.gold_files
        recall = ev.get("recall")
        links.append(_link("observation", f"the reference fix changes {gold.gold_files}", "gold_patch"))
        links.append(
            _link(
                "observation",
                f"the agent never edited {missed}"
                + (f" (file recall {recall:.2f})" if isinstance(recall, (int, float)) else ""),
                "agent_patch ∩ gold",
            )
        )
        if et == "cross_module_context_missing":
            links.append(
                _link(
                    "inference",
                    "the missed code lives in a different module than what the "
                    "agent did touch — cross-module context was not assembled",
                    "rule:loc_partial+cross_module",
                )
            )
        elif et == "wrong_file_localization":
            links.append(
                _link(
                    "observation",
                    f"the agent instead edited non-gold files {ev.get('extra_files')}",
                    "agent_patch − gold",
                )
            )
            if ev.get("solved_unrelated_candidate"):
                links.append(
                    _link(
                        "observation",
                        "the patch shares ZERO code identifiers with the issue text "
                        "— the agent may have solved a DIFFERENT problem than the "
                        "one reported (candidate for judge/human review, not a "
                        "blamed requirement fault)",
                        "patch symbols ∩ issue identifiers",
                    )
                )
        links.append(_link("observation", "the gating tests still fail", "eval_outcome"))
        links.append(
            _link(
                "conclusion",
                "code that had to change was never reached, so the fix could not be made → localization fault",
                "rule:deduction",
            )
        )
        counterfactual["what_would_confirm"] = (
            f"locate and correctly edit {missed}, then re-run the gating tests → expect pass"
        )

    elif cap == "code_editing" and et == "incorrect_patch":
        hit = ev.get("hit") or gold.gold_files
        full_hit = set(hit) >= set(gold.gold_files)
        links.append(_link("observation", f"the reference fix changes exactly {gold.gold_files}", "gold_patch"))
        if full_hit:
            links.append(
                _link(
                    "observation",
                    f"the agent edited the right files {hit} (recall 1.0, none missed)",
                    "agent_patch ∩ gold",
                )
            )
            links.append(
                _link(
                    "inference",
                    "localization is exonerated — the agent reached the files the fix lives in",
                    "rule:right_place_wrong_change",
                )
            )
        else:
            # Also emitted on a PARTIAL localization miss (loc_partial + low
            # overlap): the prose must not claim recall 1.0 / full exoneration —
            # the missed files carry their own localization/omission fault.
            links.append(
                _link(
                    "observation",
                    f"the agent edited gold file(s) {hit}, but its change content "
                    "there diverges from the reference (other gold files it missed "
                    "are covered by a separate localization/omission fault on this "
                    "case)",
                    "agent_patch ∩ gold",
                )
            )
            links.append(
                _link(
                    "inference",
                    "for THESE files localization succeeded — the fault here is the edit content, not the search",
                    "rule:loc_partial_low_overlap",
                )
            )
        ov = gold_overlap(gold_changed, agent_changed, hit)
        frac = ov["reproduced_fraction"] if ov is not None else None
        if ov is not None:
            # v0.32: the fraction proves DIVERGENCE FROM THE REFERENCE, not
            # wrongness — a different-but-correct fix diverges too (PatchDiff:
            # ~30% of plausible patches do), and at the top of the range the
            # old template produced absurdities ("reproduce only 100% … same
            # place, different change") on near-gold patches.
            if frac >= NEAR_GOLD_FRACTION:
                links.append(
                    _link(
                        "observation",
                        f"the agent's changed lines reproduce "
                        f"{int(100 * frac)}% of the reference change at those files "
                        f"(per-file {ov['per_file']}) — a NEAR-GOLD patch; the failure "
                        "signal must come from the residual delta, extra edits, or the "
                        "evaluation itself (a BA-HARNESS-1 suspect shape)",
                        "diff(agent) vs diff(gold)",
                    )
                )
            else:
                links.append(
                    _link(
                        "observation",
                        f"the agent's changed lines reproduce only "
                        f"{int(100 * frac)}% of the reference change at those files "
                        f"(per-file {ov['per_file']}) — same place, the change DIVERGES "
                        "from the reference",
                        "diff(agent) vs diff(gold)",
                    )
                )
        rg = ev.get("region_overlap")
        if isinstance(rg, (int, float)):
            links.append(
                _link(
                    "observation",
                    (
                        f"the agent's hunks cover {int(100 * rg)}% of the reference "
                        "change's line REGIONS — "
                        + (
                            "same region, divergent content"
                            if rg >= 0.5
                            else "largely a DISJOINT region of the same file (a within-file localization nuance)"
                        )
                    ),
                    "hunk ranges (old-file coords)",
                )
            )
        links.append(_link("observation", "the gating tests still fail", "eval_outcome"))
        if full_hit:
            links.append(
                _link(
                    "conclusion",
                    "right files + failing tests → the edit content diverges from a "
                    "working fix → editing-correctness fault (divergence from the "
                    "reference is proven; a different-but-correct fix rejected by the "
                    "harness is the documented alternative — METH-HARNESS-1)",
                    "rule:deduction",
                )
            )
            counterfactual["what_would_confirm"] = (
                f"replace the agent's hunks in {hit} with the reference hunks, then re-run the "
                "gating tests → expect pass"
            )
        else:
            # loc_partial: the unedited gold file(s) already explain the failure;
            # divergence at the reached files is an editing fault under the
            # reference-anchored view, but wrongness there is NOT deduced.
            links.append(
                _link(
                    "conclusion",
                    "the edit content at the reached files diverges from the reference "
                    "→ editing shares blame (divergence, not proven wrongness — the "
                    "unedited gold file(s) independently explain the failing tests)",
                    "rule:association",
                )
            )
            counterfactual["what_would_confirm"] = (
                f"replace the agent's hunks in {hit} with the reference hunks AND apply the "
                "reference change to the missed gold file(s), then re-run the gating tests → "
                "expect pass (the joint intervention; hunk replacement alone cannot pass while "
                "a gold file is unedited)"
            )

    elif cap == "code_editing" and et == "relevant_change_omitted":
        read_not_edited = ev.get("read_not_edited") or []
        links.append(_link("observation", f"the reference fix changes {gold.gold_files}", "gold_patch"))
        links.append(
            _link(
                "observation",
                f"the agent READ {read_not_edited} during the run but never edited those files",
                "trajectory:read_set ∩ gold − edit_set",
            )
        )
        links.append(
            _link(
                "inference",
                "localization is exonerated — the agent reached this code, so the "
                "omission is a selection/editing decision, not a search failure",
                "rule:reached_not_edited",
            )
        )
        links.append(_link("observation", "the gating tests still fail", "eval_outcome"))
        links.append(
            _link(
                "conclusion",
                "a necessary change at a file the agent had open was omitted → editing/"
                "selection fault (whether the cause is comprehension or planning is "
                "adjudicated by the judge layer)",
                "rule:association",
            )
        )
        counterfactual["what_would_confirm"] = (
            f"apply the reference change to {read_not_edited}, then re-run the gating tests → expect pass"
        )

    elif cap == "code_editing" and et == "gave_up_early":
        links.append(
            _link(
                "observation",
                f"the agent produced no source fix (patch_state: {ev.get('patch_state', 'missing')})",
                "agent_patch",
            )
        )
        links.append(
            _link(
                "observation",
                f"and it never investigated: ran_any_test={ev.get('ran_any_test')}, "
                f"files_read={ev.get('files_read')} (<3), "
                f"n_steps={ev.get('n_steps')}",
                "trajectory:activity",
            )
        )
        links.append(
            _link(
                "inference",
                "no investigation + no output = early disengagement, not a failed "
                "editing attempt (the investigated-but-no-patch case stays "
                "no_patch_produced / analysis-paralysis-shaped)",
                "rule:not_investigated",
            )
        )
        links.append(_link("observation", "the task is unresolved", "eval_outcome"))
        links.append(
            _link(
                "conclusion",
                "the run disengaged before attempting a fix → engagement/output fault "
                "(the deeper cause — comprehension, planning, or environment — is "
                "interpretive; this label is arbiter-eligible)",
                "rule:association",
            )
        )
        counterfactual["what_would_confirm"] = (
            "re-run with a minimum-investigation scaffold gate (must run a test or read the "
            "implicated files before finishing) → expect an actual fix attempt"
        )

    elif cap == "code_editing" and et == "no_patch_produced":
        state = ev.get("patch_state") or "missing"
        state_prose = {
            "missing": "the agent produced no patch artifact at all",
            "empty": "the agent's patch is empty (no file changes)",
            "unparseable": "the agent emitted patch text, but it parses to zero files "
            "(a format failure, not an empty submission)",
            "test_only": "the agent edited only test files "
            f"({ev.get('edited_test_files') or []}) and produced no source fix",
        }.get(state, "the agent produced no source edit")
        links.append(_link("observation", state_prose, "agent_patch"))
        if ev.get("gold_tests_edited"):
            links.append(
                _link(
                    "observation",
                    f"⚠ the edited test files include the GOLD GATING-TEST file(s) "
                    f"{ev['gold_tests_edited']} — the agent modified the very tests "
                    "that gate success",
                    "agent_patch ∩ gold_tests",
                )
            )
        links.append(_link("observation", "the task is unresolved", "eval_outcome"))
        links.append(
            _link(
                "conclusion",
                "no fix was emitted at all → editing/output fault (may be upstream "
                "abandonment; see any requirement/planning fault on this case)",
                "rule:deduction",
            )
        )
        counterfactual["what_would_confirm"] = "apply the reference fix, then re-run the gating tests → expect pass"

    elif cap == "code_editing" and et == "excessive_extra_edits":
        links.append(_link("observation", f"the reference fix changes only {gold.gold_files}", "gold_patch"))
        links.append(
            _link(
                "observation",
                f"the agent's patch changes {ev.get('extra_src_loc')} LOC in "
                f"non-gold source files {ev.get('extra_files')}",
                "agent_patch − gold",
            )
        )
        links.append(_link("observation", "the gating tests still fail", "eval_outcome"))
        links.append(
            _link(
                "inference",
                "editing mass off the reference path at this scale is "
                "failure-associated (12.4% of failed vs 3.1% of solved runs, "
                "RR 2.38, Holm p=0.005) — scope creep multiplies the surface "
                "where a regression or unrelated change can sink the eval; "
                "large exploratory edits are sometimes legitimate, hence "
                "medium confidence",
                "rule:calibrated_anchor",
            )
        )
        counterfactual["what_would_confirm"] = (
            "revert the off-path edits, keep the on-path fix, re-run the gating "
            "tests → an unchanged failure would exonerate the extra edits"
        )

    elif cap == "code_editing" and et in ("partial_fix_only", "regression_introduced"):
        # eval+ tier (v0.39): the grader's per-test record is the raw input.
        frac = ev.get("f2p_pass_fraction")
        passed = list(ev.get("f2p_passed") or [])
        failed = list(ev.get("f2p_failed") or [])
        broken = list(ev.get("p2p_broken") or [])
        links.append(
            _link(
                "observation",
                "the official harness reports per-test status for this patch "
                f"(FAIL_TO_PASS: {len(passed)} pass / {len(failed)} fail; "
                f"PASS_TO_PASS broken: {len(broken)})",
                "eval_report:tests_status",
            )
        )
        if et == "partial_fix_only":
            links.append(
                _link(
                    "observation",
                    f"gating tests now passing: {passed[:6]}; still failing: {failed[:6]}",
                    "eval_report:FAIL_TO_PASS",
                )
            )
            links.append(
                _link(
                    "inference",
                    "the change fixes some of the specified cases but not all — "
                    "a coverage / corner-case miss, not a wholly wrong change",
                    "rule:partial_f2p",
                )
            )
            pct = f"{frac:.0%} of" if isinstance(frac, (int, float)) else "some but not all"
            links.append(
                _link(
                    "conclusion",
                    f"{pct} the gating tests pass → partial fix (deductive from "
                    "the grader's per-test record; severity = pass fraction)",
                    "rule:deduction",
                )
            )
            counterfactual["what_would_confirm"] = (
                f"extend the change to cover the still-failing gating cases {failed[:6]}, "
                "re-run the gating tests → expect pass"
            )
        else:
            links.append(
                _link(
                    "observation",
                    f"PASS_TO_PASS tests that passed before the patch now fail: {broken[:8]}",
                    "eval_report:PASS_TO_PASS",
                )
            )
            if frac == 1.0:
                links.append(
                    _link(
                        "inference",
                        "every gating test passes — the fix itself worked; the "
                        "evaluation failed ONLY because of the regression "
                        "(supersedes incorrect_patch for this case)",
                        "rule:regression_only",
                    )
                )
            links.append(
                _link(
                    "conclusion",
                    "the patch broke previously-passing behavior → regression "
                    "introduced (deductive from the grader's per-test record)",
                    "rule:deduction",
                )
            )
            counterfactual["what_would_confirm"] = (
                f"narrow or revert the change so {broken[:4]} pass again while the gating "
                "tests stay green → expect resolved"
            )

    elif cap == "code_verification":
        links.append(
            _link(
                "observation",
                "the benchmark declares F2P/P2P verification targets; local visibility is recorded separately",
                "gold_tests",
            )
        )
        if et == "relevant_test_not_run":
            ran = ev.get("ran_any_test")
            # v0.32 honest wording: the condition establishes that no test
            # command NAMED the gating tests — a bare `pytest`/directory run
            # can execute them without naming them (token-match limit).
            links.append(
                _link(
                    "observation",
                    "no test-shaped command was observed in the run"
                    if ran is False
                    else "the agent ran tests but no command ever NAMED the fail_to_pass "
                    "gating tests (a bare suite-wide run may still have executed "
                    "them — the token match cannot see that)",
                    "trajectory:test_runs",
                )
            )
            links.append(
                _link(
                    "inference", "no observed, targeted gating signal preceded submission", "rule:no_gating_observation"
                )
            )
        elif et == "unsupported_success_claim":
            links.append(
                _link(
                    "observation",
                    "the agent declared success in its final message while no test "
                    "command ever NAMED the gating tests (suite-wide runs are not "
                    "visible to the token match)",
                    "trajectory:final_claim",
                )
            )
        elif et == "test_result_misinterpreted":
            links.append(
                _link(
                    "observation",
                    "the agent's LAST observed test run failed, yet its final message declared success",
                    "trajectory:last_test_result + final_claim",
                )
            )
        elif et == "no_test_after_final_edit":
            links.append(
                _link(
                    "observation",
                    f"the agent ran tests ({ev.get('n_test_runs')} run(s)) but none "
                    "AFTER its last source edit — the submitted state was never "
                    "verified",
                    "trajectory:edit/test ordering",
                )
            )
            links.append(
                _link(
                    "inference",
                    "whatever the earlier runs showed, the final patch is not the "
                    "code that was tested ('verified early, edited after')",
                    "rule:no_final_verification",
                )
            )
        elif et == "test_expectation_rewritten":
            g_files = list(ev.get("assert_rewritten_gold_tests") or [])
            o_files = list(ev.get("assert_rewritten_other_tests") or [])
            links.append(
                _link(
                    "observation",
                    "assertion lines were removed and other assertion lines added in "
                    f"test file(s) {ev.get('assert_rewritten_files')} — what the "
                    "test expectations may have changed (a file-level association)",
                    "diff: removed ∩ added assertion lines per test file",
                )
            )
            if g_files or o_files:
                reset = (gold.benchmark_profile or {}).get("evaluator_test_reset", "unknown")
                links.append(
                    _link(
                        "observation",
                        f"reference test files: {g_files}; other test files: {o_files}. "
                        f"Recorded evaluator test reset policy: {reset}. "
                        "Oracle edits alone do not establish which assertions graded the patch.",
                        "benchmark_profile:evaluator_test_reset",
                    )
                )
            if ev.get("after_observed_failure"):
                links.append(
                    _link(
                        "observation",
                        "the rewrite followed an OBSERVED test failure — the "
                        "expectation was changed in a repair step rather than the "
                        "code",
                        "trajectory: failure → test-edit ordering",
                    )
                )
            links.append(
                _link(
                    "inference",
                    "rewriting test expectations is failure-associated (v0.39 "
                    "calibration, scripts/calibrate_test_rewrite.py) while merely "
                    "TOUCHING a gating test is not (RR 1.01) — the counter-productive "
                    "shape is 'change what the test expects', not 'edit a test "
                    "file'; large legitimate expectation updates exist, hence "
                    "medium confidence",
                    "rule:calibrated_anchor",
                )
            )
        elif et == "regression_test_missing":
            links.append(
                _link(
                    "observation",
                    f"PASS_TO_PASS tests {list(ev.get('p2p_broken') or [])[:6]} broke "
                    "under the patch (official per-test record)",
                    "eval_report:PASS_TO_PASS",
                )
            )
            links.append(
                _link(
                    "observation",
                    "no selector-free suite run and no test command ever NAMED the "
                    "broken tests' files — the regression was locally detectable "
                    "but never checked",
                    "trajectory: test_cmds vs broken tests",
                )
            )
            links.append(
                _link(
                    "inference",
                    "the verification loop scoped itself to the gating tests and skipped the regression surface",
                    "rule:no_regression_check",
                )
            )
        elif et == "authored_test_never_run":
            missing = ev.get("unrun_authored_test_files", ev.get("authored_test_files"))
            links.append(
                _link(
                    "observation",
                    f"authored test file(s) {missing} lack observed execution "
                    "after their latest edit in the thread that edited them",
                    "trajectory:per_file_edit → observed_test_result",
                )
            )
            links.append(
                _link(
                    "inference",
                    "the latest prepared test versions have no established "
                    "execution evidence; a broad suite without visible file "
                    "coverage or an unordered cross-thread run remains uncertain",
                    "rule:authored_post_edit_execution",
                )
            )

        links.append(
            _link(
                "conclusion",
                "verification failed to catch the wrong patch — a contributing gap, not "
                "the upstream cause of the wrong patch",
                "rule:contributing",
            )
        )
        counterfactual["what_would_confirm"] = (
            "verification is downstream; its effect is tested by whether running the gating "
            "tests would have surfaced the already-present edit/localization fault"
        )

    elif cap == "self_repair_loop":
        n = ev.get("n_failure_events")
        links.append(
            _link(
                "observation",
                f"the agent hit {n if n is not None else 'a'} failure event(s) mid-run",
                "trajectory:failure_events",
            )
        )
        if et == "repair_not_reverified":
            links.append(
                _link(
                    "observation",
                    "it edited again after a failure without re-running the test",
                    "trajectory:edit_after_failure",
                )
            )
        elif et == "repeated_ineffective_attempt":
            links.append(
                _link(
                    "observation",
                    "it re-edited the same file after a failure with no progress",
                    "trajectory:repeated_edit",
                )
            )
        elif et == "reverted_own_change":
            links.append(
                _link(
                    "observation",
                    "a later edit on the same file swapped the content back to an "
                    "earlier edit's original — the agent undid its own change",
                    "trajectory:edit_content (old/new line sets)",
                )
            )
            links.append(
                _link(
                    "inference",
                    "apply-then-revert cycles are repair thrash: calibrated at 5.5% "
                    "of failed vs 2.6% of solved runs (RR 2.10)",
                    "rule:self_reversion (calibrated)",
                )
            )
        links.append(
            _link(
                "conclusion",
                "the repair loop was inefficient — contributing wasted effort, rarely the root cause",
                "rule:contributing",
            )
        )
        counterfactual["what_would_confirm"] = (
            "re-verifying after each repair would have exposed, not removed, the underlying fault"
        )

    elif cap == "tool_use":
        if et == "hallucinated_path_unrecovered":
            links.append(
                _link(
                    "observation",
                    "the same nonexistent path drew a 'no such file' error two or "
                    "more times — the first error did not update the agent's model "
                    "of the repository",
                    "trajectory:repeated_invalid_target",
                )
            )
            links.append(
                _link(
                    "inference",
                    "a one-off typo recovers on the next call; repeating the exact "
                    "phantom target is a world-model update failure",
                    "rule:unrecovered_hallucination",
                )
            )
        elif et == "repeated_patch_apply_failure":
            links.append(
                _link(
                    "observation",
                    "two or more edit/apply attempts failed with patch-does-not-apply "
                    "signatures — the edit mechanics looped on a diff that cannot land",
                    "trajectory:patch_apply_errors",
                )
            )
            links.append(
                _link(
                    "inference",
                    "the agent kept re-emitting a stale diff instead of re-reading the file to rebase its edit",
                    "rule:stale_edit_loop",
                )
            )
        else:
            links.append(
                _link(
                    "observation",
                    "a tool call errored and the agent took no recovering follow-up",
                    "trajectory:tool_error",
                )
            )
        links.append(
            _link(
                "conclusion",
                "tool-failure recovery gap — contributing mechanics, not the cause of a wrong patch",
                "rule:contributing",
            )
        )
        # Per-type counterfactuals: for the phantom path, RETRYING is the failure
        # mode itself (the chain's own inference) — the probe is re-grounding.
        if et == "hallucinated_path_unrecovered":
            counterfactual["what_would_confirm"] = (
                "listing the directory / re-reading the tree and targeting a real path would have unblocked the step"
            )
        elif et == "repeated_patch_apply_failure":
            counterfactual["what_would_confirm"] = (
                "re-reading the file and re-basing the edit on its current content would have let the patch land"
            )
        else:
            counterfactual["what_would_confirm"] = "retrying the failed tool call would have unblocked the step"

    elif cap == "requirement_understanding" and et == "solved_unrelated_problem":
        links.append(
            _link(
                "observation",
                f"the patch edits only non-gold files {ev.get('extra_files')} (a total localization miss)",
                "agent_patch − gold",
            )
        )
        links.append(
            _link(
                "observation",
                f"and it shares ZERO code identifiers with the issue text — the "
                f"symbols it defines/touches ({ev.get('patch_symbols')}) never "
                "appear in the problem statement",
                "patch symbols ∩ issue identifiers",
            )
        )
        links.append(
            _link(
                "inference",
                "a fix for the reported problem almost always touches at least one "
                "entity the issue names (only 1.0% of SOLVED runs violate this; "
                "8.32% of failed runs do — calibrated RR 7.95): the agent plausibly "
                "solved a DIFFERENT problem than the one reported",
                "rule:zero_issue_overlap (calibrated)",
            )
        )
        links.append(_link("observation", "the gating tests still fail", "eval_outcome"))
        links.append(
            _link(
                "conclusion",
                "objective requirement-understanding anchor at medium confidence "
                "(token extraction is best-effort; arbiter-eligible; can never "
                "demote a deductive fact)",
                "rule:association",
            )
        )
        counterfactual["what_would_confirm"] = (
            "human review of the issue vs the patch: does the change address the reported "
            "problem at all? (also the judge/arbiter's adjudication target)"
        )

    elif cap in ("requirement_understanding", "task_planning"):
        steps = ev.get("evidence_steps") or []
        rationale = ev.get("judge_rationale") or ""
        model = ev.get("judge_model") or "judge"
        links.append(
            _link(
                "observation",
                "the judge saw the problem statement + an outcome-blinded view of the "
                "agent's process (no gold patch, no pass/fail)",
                "judge_input",
            )
        )
        if steps:
            quotes = [{"step": s, "text": step_text[s]} for s in steps if s in step_text] if step_text else None
            links.append(
                _link(
                    "observation",
                    f"cited trajectory steps {steps}: {rationale}".strip(),
                    f"trajectory:steps {steps}",
                    quotes=quotes or None,
                )
            )
        elif rationale:
            links.append(_link("observation", rationale, "judge_rationale"))
        links.append(
            _link(
                "conclusion",
                f"{model} labels `{et}` at {cand.get('confidence')} confidence; this is "
                "a model-inferred candidate with a secondary role; judge capabilities "
                "are not eligible for primary attribution",
                "rule:judge_secondary",
            )
        )
        counterfactual["what_would_confirm"] = (
            "judge-vs-human agreement (Cohen's κ) on the calibration set, and a planted-misread "
            "control that should score this label"
        )

    return {
        "claim": f"{cap} → {et} is a {role} of this failure",
        "strength": strength,
        "observation_strength": strength,
        "causal_support": {
            "status": "not_tested",
            "basis": "No intervention or controlled replay supplied for this claim.",
        },
        "strength_basis": strength_basis_for(strength, et),
        "blame_weight": round(blame_weight, 2),
        "links": links,
        "counterfactual": counterfactual,
    }
