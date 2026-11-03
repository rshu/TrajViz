# awe/audit.py
"""Internal-consistency / tamper-evidence check via independent re-derivation.

Re-derives each fault's factual atoms from the RAW inputs (gold patch, agent diff,
outcome, trajectory) and compares to the persisted chain. It must never import
signals/detect/rootcause/trajsignals, or the check would be vacuous (the PR #7
trap) — the self-repair/tool ordering atoms are re-implemented here from the
shared trajectory primitives, not borrowed from trajsignals.

INDEPENDENCE SCOPE (stated honestly): the STRUCTURE of every check — set
arithmetic, summation, the failure→edit→retest / tool-error→no-retry ordering —
is re-implemented inline. The low-level PRIMITIVES are shared with the data
layer: unified-diff parsing + line normalization (diffutil/evidence) and the
text classifiers (trajectory). A bug inside a shared primitive produces the same
wrong value on both sides and is NOT detectable here; transcription errors,
tampering, and any drift in the derived logic ARE. Each fault must also expose
its type-specific atoms — a fault whose atoms were deleted (or whose error type
matches no known branch) audits as `unverifiable`, never as `re_derived`.

Model-inferred (judge) faults can never be re_derived; their structural audit
additionally cross-checks the persisted fault against the CACHED judge verdict
(label/confidence/cited steps), so relabeling a judge fault after the fact is
detectable when the verdict cache is present.

This establishes **internal consistency** and **tamper-evidence** (drift or
transcription error is detectable) — it does NOT establish that a diagnosis matches
ground truth (construct validity requires the human-κ study / interventional loop,
not yet run)."""

from __future__ import annotations

from .diffutil import parse_unified_diff, is_test_file
from .evidence import (
    changed_lines,
    issue_code_tokens,
    strength_of,
    DEDUCTIVE,
    ASSOCIATIONAL,
    MODEL_INFERRED,
    OVEREDIT_EXTRA_LOC,
    is_assertion_line,
)
from .verification import known_zero_test_calls, tool_observations
from .trajectory import (
    RUN_COMMAND,
    RUN_TEST,
    EDIT,
    WRITE,
    looks_like_success_claim,
    looks_like_test_failure,
    classify_tool_error,
    is_suite_wide_test_command,
    names_any_test,
    read_observations,
    observed_test_runs,
)

RE_DERIVED = "re_derived"
STRUCTURAL_OK = "structural_ok"
MISMATCH = "mismatch"
UNVERIFIABLE = "unverifiable"

_TOL = 0.01


def _src_files(text: str) -> set:
    return {fd.path for fd in parse_unified_diff(text or "") if not is_test_file(fd.path)}


def _read_set(traj) -> set | None:
    if traj is None:
        return None
    return read_observations(traj)[0]


def _reproduced_fraction(gold_text: str, agent_text: str, files) -> float | None:
    from collections import Counter

    g, a = changed_lines(gold_text), changed_lines(agent_text)
    shared = denom = 0
    for f in files:
        gf = g.get(f)
        if not gf:
            continue
        af = a.get(f, {"added": Counter(), "removed": Counter()})
        # multiset-min counting, mirroring evidence.gold_overlap (re-summed here)
        shared += sum((gf["added"] & af["added"]).values()) + sum((gf["removed"] & af["removed"]).values())
        denom += sum(gf["added"].values()) + sum(gf["removed"].values())
    return round(shared / denom, 4) if denom else None


def _region_overlap(gold_text: str, agent_text: str, files, margin: int = 3) -> float | None:
    """Hunk-range overlap re-summed inline from both raw diffs (old-file
    coordinates; mirrors evidence.region_overlap's structure)."""
    g = {fd.path: fd.hunks for fd in parse_unified_diff(gold_text or "")}
    a = {fd.path: fd.hunks for fd in parse_unified_diff(agent_text or "")}
    total = covered = 0
    for f in files:
        g_lines = {ln for st, c in (g.get(f) or []) for ln in range(st, st + max(c, 1))}
        if not g_lines:
            continue
        a_lines = set()
        for st, c in a.get(f) or []:
            a_lines.update(range(max(1, st - margin), st + max(c, 1) + margin))
        total += len(g_lines)
        covered += len(g_lines & a_lines)
    return round(covered / total, 4) if total else None


def _eq(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= _TOL
    if isinstance(expected, list) or isinstance(actual, list):
        return sorted(expected or []) == sorted(actual or [])
    return expected == actual


def _chk(atom, expected, actual) -> dict:
    return {"atom": atom, "expected": expected, "actual": actual, "ok": _eq(expected, actual)}


def _finish(tier: str, checks: list, *, unverifiable: bool = False) -> dict:
    mism = [c for c in checks if not c["ok"]]
    if mism:
        verdict = MISMATCH
    elif unverifiable:
        verdict = UNVERIFIABLE
    else:
        verdict = RE_DERIVED if tier in (DEDUCTIVE, ASSOCIATIONAL) else STRUCTURAL_OK
    return {"tier": tier, "verdict": verdict, "checks": checks, "mismatches": mism}


_DEDUCTIVE_ETS = {
    "incorrect_patch",
    "missed_relevant_file",
    "cross_module_context_missing",
    "wrong_file_localization",
    "relevant_change_omitted",
    "no_patch_produced",
    "gave_up_early",
}


def _audit_deductive(fault, gold_text, agent_text, resolved, traj, tier: str = DEDUCTIVE) -> dict:
    ev = fault.get("detector_evidence", {})
    et = fault["error_type"]
    gold_src, agent_src = _src_files(gold_text), _src_files(agent_text)
    read = _read_set(traj)
    checks = []
    unverifiable = False

    # A missing outcome record cannot certify anything (and must not read as a
    # tamper mismatch); a fault on a case that did not FAIL is a real mismatch.
    if resolved is None:
        unverifiable = True
    else:
        checks.append(_chk("outcome", False, resolved))

    if "recall" in ev:
        rec = round(len(gold_src & agent_src) / len(gold_src), 2) if gold_src else None
        checks.append(_chk("recall", ev["recall"], rec))

    if et == "incorrect_patch":
        hit = ev.get("hit") or sorted(gold_src)
        if "reproduced_fraction" in ev:
            checks.append(
                _chk("reproduced_fraction", ev["reproduced_fraction"], _reproduced_fraction(gold_text, agent_text, hit))
            )
        if "region_overlap" in ev:
            checks.append(_chk("region_overlap", ev["region_overlap"], _region_overlap(gold_text, agent_text, hit)))
    elif et in ("missed_relevant_file", "cross_module_context_missing"):
        missed_all = gold_src - agent_src
        unknown = read_observations(traj)[2] if traj is not None else set()
        if read is None:
            unverifiable = True
        missed = sorted(missed_all - read - unknown) if read is not None else None
        if "missed_gold" in ev and missed is not None:
            checks.append(_chk("missed_gold", ev["missed_gold"], missed))
    elif et == "wrong_file_localization":
        if read is None:
            unverifiable = True
        else:
            unknown = read_observations(traj)[2]
            checks.append(_chk("unreached_reference_file", True, bool(gold_src - agent_src - read - unknown)))
        if "extra_files" in ev:
            checks.append(_chk("extra_files", ev["extra_files"], sorted(agent_src - gold_src)))
    elif et == "relevant_change_omitted":
        if read is None:
            unverifiable = True
        elif "read_not_edited" in ev:
            rne = sorted((read & gold_src) - agent_src)
            checks.append(_chk("read_not_edited", ev["read_not_edited"], rne))
    elif et == "no_patch_produced":
        checks.append(_chk("no_source_edit", True, not agent_src))
    elif et == "gave_up_early":
        checks.append(_chk("no_source_edit", True, not agent_src))
        if traj is None:
            unverifiable = True
        else:
            ran = any(s.test_run or s.canonical_action == RUN_TEST for s in traj.steps)
            reads = read_observations(traj)[1]
            main = _main_steps(traj)
            truncated = bool(main) and main[-1].type == "tool_call"
            # killed ≠ gave up: mirrors the detector's truncation gate
            checks.append(_chk("not_investigated", True, (not ran) and len(reads) < 3 and not truncated))

    # MINIMUM-ATOM RULE: the strongest verdict requires at least one TYPE-SPECIFIC
    # atom to have been re-derived. A fault whose detector_evidence keys were
    # stripped, or whose error_type matches no branch above, degenerates to the
    # outcome check alone — that is `unverifiable`, not `re_derived` (the old
    # behavior let tamper-by-deletion and tamper-by-relabel audit clean).
    # `recall` is capability-GENERIC (any loc/edit fault may carry it), so it
    # cannot serve as the type-specific atom either: relabeling e.g. a
    # missed_relevant_file fault to incorrect_patch used to audit clean on
    # outcome+recall alone (v0.31). Residual relabel surface, documented:
    # missed_relevant_file <-> cross_module_context_missing share `missed_gold`
    # (the cross-module attribute itself is not re-derived), and gave_up_early
    # -> no_patch_produced shares `no_source_edit` (the reverse direction is
    # caught by gave_up_early's extra not_investigated atom).
    _GENERIC_ATOMS = ("outcome", "recall")
    if et not in _DEDUCTIVE_ETS or not any(c["atom"] not in _GENERIC_ATOMS for c in checks):
        unverifiable = True

    return _finish(tier, checks, unverifiable=unverifiable)


# --- Independent re-derivation of the self-repair / tool-mechanic ordering atoms.
# These mirror trajsignals' logic but are re-implemented inline from the raw steps
# (audit.py must NOT import trajsignals, or the check would just re-run the detector).
# They use only the shared trajectory PRIMITIVES (RUN_TEST/EDIT/WRITE + the failure /
# tool-error text classifiers) — the same low-level facts both layers agree on — while
# re-deriving the STRUCTURE (failure→edit→retest ordering) independently.


def _main_steps(traj) -> list:
    """Ordering analysis is main-thread only: the CC adapter appends sub-agent
    steps after the main trajectory, so cross-thread step-id order is meaningless
    (mirrors the detector's contract; re-derived independently)."""
    return [s for s in traj.steps if getattr(s, "subagent_id", None) is None]


def _result_failed(r) -> bool:
    """Recorded exit code wins when present (0 vetoes echoed failure text,
    non-0 catches verbose failures); text/flag heuristics are the fallback."""
    ec = r.args.get("exit_code") if isinstance(getattr(r, "args", None), dict) else None
    if isinstance(ec, int):
        return ec != 0
    return bool(r.tool_error) or looks_like_test_failure(r.output_excerpt)


def _failure_event_steps(traj) -> list:
    """Step ids where a gating test ran and its result indicated failure — the events
    that open a self-repair window."""
    main = _main_steps(traj)
    results = {id(call): result for _, call, _, result, _ in tool_observations(traj) if result is not None}
    out = []
    for i, s in enumerate(main):
        if s.type == "tool_call" and s.test_run:
            r = results.get(id(s))
            if r is not None and _result_failed(r):
                aggregation = (r.args or {}).get("output_aggregation") or {}
                out.append(aggregation.get("completion_step", s.step_id))
    return out


def _redrv_repair_not_reverified(traj) -> bool:
    zero = known_zero_test_calls(traj)
    fails = _failure_event_steps(traj)
    if not fails:
        return False
    main = _main_steps(traj)
    edits_after = [s.step_id for s in main if s.canonical_action in (EDIT, WRITE) and s.step_id > min(fails)]
    if not edits_after:
        return False
    last_edit = max(edits_after)
    if any(s.step_id > last_edit for s in main if s.test_run and id(s) not in zero):
        return False
    # killed ≠ gave up: run truncated right after the edit (mirrors the detector)
    if main and main[-1].type == "tool_call" and last_edit >= main[-1].step_id - 3:
        return False
    return True


def _redrv_repeated_ineffective(traj) -> bool:
    fails = _failure_event_steps(traj)
    if not fails:
        return False
    first, counts = min(fails), {}
    for s in _main_steps(traj):
        if s.canonical_action in (EDIT, WRITE) and s.step_id > first:
            for f in s.files or []:
                if f:
                    counts[f] = counts.get(f, 0) + 1
    return any(c >= 2 for c in counts.values())


def _redrv_tool_failure_not_retried(traj) -> bool:
    # Text-signature fallback applies to command/edit/write results only —
    # READ/SEARCH/DELEGATE results echo file content (mirrors the detector's
    # scoping; re-derived independently). Retry window stays within a thread.
    threads: dict = {}
    for s in traj.steps:
        threads.setdefault(getattr(s, "subagent_id", None), []).append(s)
    call_by_id = {(s.subagent_id, s.call_id): s for s in traj.steps if s.type == "tool_call" and s.call_id}
    for thread in threads.values():
        thread_truncated = bool(thread) and thread[-1].type == "tool_call"
        for i, s in enumerate(thread):
            if s.type != "tool_result":
                continue
            call = call_by_id.get((s.subagent_id, s.call_id))
            if call is not None and call.canonical_action == RUN_TEST:
                continue
            action = call.canonical_action if call is not None else None
            is_err = bool(s.tool_error) or (
                action in (RUN_COMMAND, EDIT, WRITE) and classify_tool_error(s.output_excerpt) is not None
            )
            if (
                is_err
                and not any(t.type == "tool_call" for t in thread[i + 1 : i + 6])
                and not (thread_truncated and i >= len(thread) - 6)
            ):
                return True
    return False


def _redrv_retry_without_change(traj) -> bool:
    zero = known_zero_test_calls(traj)
    import re as _re

    # mirrors trajsignals' env-repair suppression (re-implemented inline; the
    # make branch must track the detector's — v0.37 review fix: the mirror kept
    # the pre-v0.36 `make\b` and flagged the detector's honest faults as forged)
    env_re = _re.compile(
        r"\b(?:pip3?\s+install|conda\s+install|apt(?:-get)?\s+install|"
        r"npm\s+(?:install|ci)|python[\w.]*\s+setup\.py\s+(?:install|develop|build)|"
        r"pip3?\s+uninstall|make\b(?![^&|;]*\b(?:test|tests|check|lint)\b))"
    )
    fails = _failure_event_steps(traj)
    if not fails:
        return False
    main = _main_steps(traj)
    cmd_of = {
        s.step_id: (s.args or {}).get("command", "").strip()
        for s in main
        if s.type == "tool_call" and s.test_run and id(s) not in zero
    }
    change_ids = sorted(
        {s.step_id for s in main if s.canonical_action in (EDIT, WRITE)}
        | {
            s.step_id
            for s in main
            if s.type == "tool_call"
            and s.canonical_action == RUN_COMMAND
            and env_re.search((s.args or {}).get("command", "") or "")
        }
    )
    failure_commands = {}
    for _, call, _, result, _ in tool_observations(traj):
        if result is not None and call.subagent_id is None and call.test_run and id(call) not in zero:
            aggregation = (result.args or {}).get("output_aggregation") or {}
            failure_commands[aggregation.get("completion_step", call.step_id)] = (
                (call.args or {}).get("command", "").strip()
            )
    for fid in fails:
        fcmd = failure_commands.get(fid, "")
        if not fcmd:
            continue
        for tid, tcmd in cmd_of.items():
            if tid > fid and tcmd == fcmd and not any(fid < e < tid for e in change_ids):
                return True
    return False


def _redrv_repeated_patch_apply_failure(traj) -> bool:
    call_by_id = {(s.subagent_id, s.call_id): s for s in traj.steps if s.type == "tool_call" and s.call_id}
    n = 0
    for s in traj.steps:
        if s.type != "tool_result":
            continue
        call = call_by_id.get((s.subagent_id, s.call_id))
        if call is None or call.canonical_action not in (EDIT, WRITE):
            continue
        if classify_tool_error(s.output_excerpt) == "bad_arguments":
            n += 1
    return n >= 2


def _redrv_hallucinated_path_unrecovered(traj) -> bool:
    call_by_id = {(s.subagent_id, s.call_id): s for s in traj.steps if s.type == "tool_call" and s.call_id}
    counts: dict = {}
    for s in traj.steps:
        if s.type != "tool_result":
            continue
        call = call_by_id.get((s.subagent_id, s.call_id))
        if call is None or call.canonical_action == RUN_TEST:
            continue
        action = call.canonical_action
        is_err = bool(s.tool_error) or (
            action in (RUN_COMMAND, EDIT, WRITE) and classify_tool_error(s.output_excerpt) is not None
        )
        if is_err and classify_tool_error(s.output_excerpt) == "invalid_target":
            for f in call.files or []:
                if f:
                    counts[f] = counts.get(f, 0) + 1
    return any(c >= 2 for c in counts.values())


def _unrun_authored_test_files(traj) -> list:
    missing = set()
    latest = {}
    for i, s in enumerate(traj.steps):
        if s.canonical_action in (EDIT, WRITE):
            for f in s.files:
                if is_test_file(f):
                    latest[(s.subagent_id, f)] = i
    positions = {id(s): i for i, s in enumerate(traj.steps)}
    runs = observed_test_runs(traj)
    for (thread, f), pos in latest.items():
        stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        toks = {t for t in (f, f.rsplit("/", 1)[-1], stem) if len(t) >= 5}
        exercised = False
        for call, result in runs:
            if call.subagent_id != thread or positions[id(call)] <= pos:
                continue
            evidence = (call.args or {}).get("command", "") + "\n" + (result.output_excerpt or "")
            if any(tok in evidence for tok in toks):
                exercised = True
                break
        if not exercised:
            missing.add(f)
    return sorted(missing)


def _redrv_authored_test_never_run(traj) -> bool:
    return bool(_unrun_authored_test_files(traj))


def _redrv_reverted_own_change(traj) -> bool:
    """Re-derive observed directed reversals, gated on a failure event."""
    from .trajectory import edit_reversals

    return bool(_failure_event_steps(traj) and edit_reversals(traj))


def _redrv_no_test_after_final_edit(traj) -> bool:
    zero = known_zero_test_calls(traj)
    main = _main_steps(traj)
    # v0.32 mirror: last SOURCE edit only (test files and .md/.rst/.txt notes
    # do not arm the flag) — same filter as the detector.
    nonsource = (".md", ".rst", ".txt")
    edits = [
        s.step_id
        for s in main
        if s.canonical_action in (EDIT, WRITE)
        and (not s.files or any(not is_test_file(f) and not f.endswith(nonsource) for f in s.files))
    ]
    if not edits:
        return False
    last_edit = max(edits)
    if any(s.step_id > last_edit for s in main if s.test_run and id(s) not in zero):
        return False
    # killed ≠ gave up (mirrors the detector's truncation guard)
    if main and main[-1].type == "tool_call" and last_edit >= main[-1].step_id - 3:
        return False
    return True


def _gating_test_tokens(test_files, fail_to_pass) -> set:
    """Mirror of the detector's gold-test token builder, re-implemented inline
    (audit must not import trajsignals): gating test function names from the
    FAIL_TO_PASS ids + test_patch file paths/basenames."""
    import re

    toks: set = set()
    for tid in fail_to_pass or []:
        toks.update(re.findall(r"test_\w+", tid))
        toks.update(re.findall(r"[\w./-]+_test\.py", tid))
    for f in test_files or []:
        toks.add(f)
        toks.add(f.split("/")[-1].rsplit(".", 1)[0])
    # Same specificity bar as the detector (trajsignals._gold_test_tokens):
    # short tokens like a bare 'test' stem would substring-match every test
    # command and mint false MISMATCHes (v0.32 fix — the mirror lacked this).
    return {t for t in toks if len(t) >= 5}


def _redrv_gating_referenced(traj, tokens) -> bool:
    """Did any test command name a gating test/file? (defining atom of the
    medium relevant_test_not_run and of unsupported_success_claim)."""
    zero = known_zero_test_calls(traj)
    cmds = [(s.args or {}).get("command", "") or "" for s in traj.steps if s.test_run and id(s) not in zero]
    return any(tok in cmd for cmd in cmds for tok in tokens)


def _redrv_last_test_failed(traj) -> bool:
    """Was the LAST observed main-thread test result a failure? (defining atom
    of test_result_misinterpreted). Exit code is ground truth when recorded;
    text classification is the fallback — mirroring the detector's semantics
    from the shared primitives only."""
    zero = known_zero_test_calls(traj)
    main = _main_steps(traj)
    results = {id(call): result for _, call, _, result, _ in tool_observations(traj) if result is not None}
    last = None
    for i, s in enumerate(main):
        if s.type == "tool_call" and s.test_run and id(s) not in zero:
            r = results.get(id(s))
            if r is not None:
                last = r
    if last is None:
        return False
    ec = last.args.get("exit_code") if isinstance(last.args, dict) else None
    if isinstance(ec, int):
        return ec != 0
    return last.tool_error or looks_like_test_failure(last.output_excerpt)


def _redrv_last_gating_run(traj, tokens) -> tuple:
    """(observed, last_failed) for MAIN-thread test runs whose command names a
    gating test/file — mirror of trajsignals' gating_outcomes accounting,
    re-implemented from the shared primitives (defining atoms of the v0.35
    gating-specificity evidence on test_result_misinterpreted)."""
    zero = known_zero_test_calls(traj)
    main = _main_steps(traj)
    results = {id(call): result for _, call, _, result, _ in tool_observations(traj) if result is not None}
    observed, last_failed = False, False
    for i, s in enumerate(main):
        if s.type == "tool_call" and s.test_run and id(s) not in zero:
            cmd = (s.args or {}).get("command", "") or ""
            if not any(tok in cmd for tok in tokens):
                continue
            r = results.get(id(s))
            if r is not None:
                observed, last_failed = True, _result_failed(r)
    return observed, last_failed


_REDERIVE_ATOM = {
    "repair_not_reverified": _redrv_repair_not_reverified,
    "repeated_ineffective_attempt": _redrv_repeated_ineffective,
    "retry_without_change": _redrv_retry_without_change,  # legacy (pre-v0.42 persisted records)
    "reverted_own_change": _redrv_reverted_own_change,
    "tool_failure_not_retried": _redrv_tool_failure_not_retried,
    "repeated_patch_apply_failure": _redrv_repeated_patch_apply_failure,
    "hallucinated_path_unrecovered": _redrv_hallucinated_path_unrecovered,
    "no_test_after_final_edit": _redrv_no_test_after_final_edit,
    "authored_test_never_run": _redrv_authored_test_never_run,
}


def _audit_associational(fault, traj, test_files=None, fail_to_pass=None) -> dict:
    zero = known_zero_test_calls(traj)
    ev = fault.get("detector_evidence", {})
    et = fault["error_type"]
    if traj is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    checks = []
    defining_missing = False
    # Defining atoms of the verification faults (v0.31 — previously never
    # re-derived, so a forged fault_to_pass_referenced/last_test_failed could
    # audit clean on the generic ran_any_test/claimed_success atoms alone).
    if "fail_to_pass_referenced" in ev:
        if test_files is None and fail_to_pass is None:
            defining_missing = True  # inputs unavailable -> honest downgrade
        else:
            toks = _gating_test_tokens(test_files, fail_to_pass)
            checks.append(
                _chk("fail_to_pass_referenced", ev["fail_to_pass_referenced"], _redrv_gating_referenced(traj, toks))
            )
    if "last_test_failed" in ev:
        checks.append(_chk("last_test_failed", ev["last_test_failed"], _redrv_last_test_failed(traj)))
    if "last_gating_run_failed" in ev or "gating_run_observed" in ev:
        if test_files is None and fail_to_pass is None:
            defining_missing = True  # inputs unavailable -> honest downgrade
        else:
            toks = _gating_test_tokens(test_files, fail_to_pass)
            obs, gfailed = _redrv_last_gating_run(traj, toks)
            if "gating_run_observed" in ev:
                checks.append(_chk("gating_run_observed", ev["gating_run_observed"], obs))
            if "last_gating_run_failed" in ev:
                checks.append(_chk("last_gating_run_failed", ev["last_gating_run_failed"], gfailed))
    if "ran_suite_wide" in ev:
        # Defining atom of the v0.36 suite-wide guard on the medium
        # relevant_test_not_run arm: the fault asserts NO selector-free suite
        # run happened; re-derived via the shared classifier primitive.
        cmds = [(s.args or {}).get("command", "") or "" for s in traj.steps if s.test_run and id(s) not in zero]
        checks.append(_chk("ran_suite_wide", ev["ran_suite_wide"], any(is_suite_wide_test_command(c) for c in cmds)))
    if "ran_any_test" in ev:
        # test_run flag OR canonical RUN_TEST: adapters may set either; requiring
        # exactly one of them would mint spurious mismatches on the other.
        ran = any((s.test_run or s.canonical_action == RUN_TEST) and id(s) not in zero for s in traj.steps)
        checks.append(_chk("ran_any_test", ev["ran_any_test"], ran))
    # best-effort: re-derived via looks_like_success_claim over the MAIN thread's
    # last assistant texts, falling back to ALL texts when the main thread has
    # none — mirroring the detector exactly (a delegate-heavy run whose final
    # summary comes from a sub-agent must not audit as a mismatch).
    if ev.get("claimed_success_final") is True:
        all_texts = [s for s in traj.assistant_texts() if s.text]
        main_texts = [s for s in all_texts if getattr(s, "subagent_id", None) is None]
        texts = [s.text for s in (main_texts or all_texts)][-1:]
        checks.append(_chk("claimed_success_final", True, looks_like_success_claim(" ".join(texts))))
    # Self-repair + tool mechanics: re-derive the ordering condition the fault asserts
    # (the fault's existence claims it holds -> expect True). These were previously
    # blanket-`unverifiable`; they are now re-derived from the raw trajectory.
    rederive = _REDERIVE_ATOM.get(et)
    if rederive is not None:
        checks.append(_chk(et, True, rederive(traj)))
        if et == "repeated_ineffective_attempt" and "n_failure_events" in ev:
            checks.append(_chk("n_failure_events", ev["n_failure_events"], len(_failure_event_steps(traj))))
    if et == "authored_test_never_run" and "unrun_authored_test_files" in ev:
        checks.append(
            _chk("unrun_authored_test_files", ev["unrun_authored_test_files"], _unrun_authored_test_files(traj))
        )
    # No directly re-derivable atom present -> structural presence check, not a
    # failure; a fault whose DEFINING atom could not be re-derived for lack of
    # inputs is unverifiable even when generic atoms checked out.
    return _finish(ASSOCIATIONAL, checks, unverifiable=(not checks) or defining_missing)


def _audit_solved_unrelated(fault, gold_text, agent_text, problem_statement, resolved) -> dict:
    """Re-derive the solved_unrelated anchor from raw inputs: edited source files
    ∩ gold source files must be empty, and the patch's symbols/file basenames
    must share nothing with the issue's code identifiers. Includes the OUTCOME
    atom (fault valid only on failed cases — a fault forged onto a solved case
    must not audit clean). Unverifiable when the problem statement or outcome
    was not available to the auditor."""
    ev = fault.get("detector_evidence", {})
    if problem_statement is None or resolved is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    gold_src, agent_src = _src_files(gold_text), _src_files(agent_text)
    diffs = parse_unified_diff(agent_text or "")
    syms = {s for fd in diffs for s in fd.symbols}
    names = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in agent_src}
    overlap = sorted((syms | names) & issue_code_tokens(problem_statement))
    checks = [
        _chk("outcome", False, resolved),
        _chk("no_gold_overlap", True, bool(agent_src) and not (agent_src & gold_src)),
        _chk("issue_symbol_overlap", ev.get("issue_symbol_overlap", []), overlap),
        _chk("has_symbols", True, bool(syms)),
    ]
    return _finish(ASSOCIATIONAL, checks)


def _audit_extra_edits(fault, gold_text, agent_text, resolved) -> dict:
    """Re-derive `excessive_extra_edits` from raw inputs: changed LOC summed
    over agent-diff SOURCE files absent from the gold source set, compared to
    the evidence value and its threshold condition. Includes the OUTCOME atom
    (failed-only fault). Unverifiable when the raw texts are unavailable."""
    ev = fault.get("detector_evidence", {})
    if agent_text is None or gold_text is None or resolved is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    gold_src = _src_files(gold_text)
    diffs = parse_unified_diff(agent_text or "")
    extra_loc = sum(fd.loc for fd in diffs if not is_test_file(fd.path) and fd.path not in gold_src)
    threshold = ev.get("threshold", OVEREDIT_EXTRA_LOC)
    checks = [
        _chk("outcome", False, resolved),
        _chk("extra_src_loc", ev.get("extra_src_loc"), extra_loc),
        _chk("meets_threshold", True, extra_loc >= threshold),
    ]
    return _finish(ASSOCIATIONAL, checks)


def _audit_gating_test(fault, agent_text, test_files, resolved) -> dict:
    """Re-derive `gating_test_modified`: agent-patch files ∩ the gold test_patch
    files, both recomputed from raw inputs; the destructiveness atom (existing
    test lines removed) re-derives from the agent diff's removed-line sets.
    Includes the OUTCOME atom (fault valid only on failed cases). Unverifiable
    when the requirements record (test_patch) or outcome was not available."""
    ev = fault.get("detector_evidence", {})
    if test_files is None or resolved is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    touched = sorted({fd.path for fd in parse_unified_diff(agent_text or "")} & set(test_files))
    checks = [
        _chk("outcome", False, resolved),
        _chk("gold_tests_edited", ev.get("gold_tests_edited"), touched),
        _chk("non_empty", True, bool(touched)),
    ]
    if "gold_tests_modified_existing" in ev:
        cl = changed_lines(agent_text or "")
        destructive = sorted(f for f in touched if (cl.get(f) or {}).get("removed"))
        checks.append(_chk("gold_tests_modified_existing", ev["gold_tests_modified_existing"], destructive))
    return _finish(ASSOCIATIONAL, checks)


def _audit_test_rewrite(fault, agent_text, test_files, resolved) -> dict:
    """Re-derive `test_expectation_rewritten` (v0.39) from the raw agent diff:
    per TEST file (all diff sections for the path summed), >=1 removed assertion
    line AND >=1 added assertion line — the shared `is_assertion_line`
    classifier, the per-file structure re-implemented here. The gold-gating
    split re-derives against the requirements' test_patch file list (absent →
    that atom is unverifiable, never a silent pass). Includes the OUTCOME atom."""
    ev = fault.get("detector_evidence", {})
    if agent_text is None or resolved is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    per_path: dict = {}
    for fd in parse_unified_diff(agent_text or ""):
        if not is_test_file(fd.path):
            continue
        rem, add = per_path.get(fd.path, (0, 0))
        rem += sum(1 for ln in fd.removed_lines if is_assertion_line(ln))
        add += sum(1 for ln in fd.added_lines if is_assertion_line(ln))
        per_path[fd.path] = (rem, add)
    rewritten = sorted(p for p, (rem, add) in per_path.items() if rem and add)
    checks = [
        _chk("outcome", False, resolved),
        _chk("assert_rewritten_files", ev.get("assert_rewritten_files"), rewritten),
        _chk("non_empty", True, bool(rewritten)),
    ]
    unverifiable = False
    if "assert_rewritten_gold_tests" in ev:
        if test_files is None:
            unverifiable = True
        else:
            gold_t = set(test_files)
            checks.append(
                _chk(
                    "assert_rewritten_gold_tests",
                    ev["assert_rewritten_gold_tests"],
                    [f for f in rewritten if f in gold_t],
                )
            )
    return _finish(ASSOCIATIONAL, checks, unverifiable=unverifiable)


def _audit_pertest(fault, pertest, resolved, traj=None, tier: str = DEDUCTIVE) -> dict:
    """Re-derive the eval+ faults (v0.39) from the RAW per-test harness record
    (agent_trajviz.awe._engine.outcomes.load_pertest — a normalized swebench report.json, itself an
    input, never produced by the detector): the FAIL_TO_PASS pass fraction for
    `partial_fix_only`, the PASS_TO_PASS failure list for `regression_introduced`
    / `regression_test_missing`, plus — for the latter — the trajectory's
    suite-wide / named-test facts re-derived from the raw steps. Includes the
    OUTCOME atom. Unverifiable without a record or when the harness could not
    apply the patch (no test information exists)."""
    zero = known_zero_test_calls(traj)
    ev = fault.get("detector_evidence", {})
    et = fault["error_type"]
    if (
        pertest is None
        or resolved is None
        or not pertest.get("patch_applied")
        or (et == "partial_fix_only" and not pertest.get("f2p_total"))
    ):
        # no record / patch not applied / no gating tests: no test information
        # exists, so nothing the detector could have emitted is re-derivable
        return _finish(tier, [], unverifiable=True)
    frac = pertest.get("f2p_pass_fraction")
    broken = sorted(pertest.get("p2p_broken") or [])
    checks = [_chk("outcome", False, resolved)]
    if et == "partial_fix_only":
        checks.append(_chk("f2p_pass_fraction", ev.get("f2p_pass_fraction"), frac))
        checks.append(_chk("partial", True, frac is not None and 0 < frac < 1))
        return _finish(tier, checks)
    checks.append(_chk("p2p_broken", sorted(ev.get("p2p_broken") or []), broken))
    checks.append(_chk("non_empty", True, bool(broken)))
    if et == "regression_test_missing":
        if traj is None:
            return _finish(tier, checks, unverifiable=True)
        cmds = [(s.args or {}).get("command", "") or "" for s in traj.steps if s.test_run and id(s) not in zero]
        suite = any(is_suite_wide_test_command(c) for c in cmds)
        # None (an un-nameable broken id) must never re-derive as "not named"
        named = names_any_test(broken, cmds)
        checks.append(_chk("ran_any_test", True, bool(cmds)))
        checks.append(_chk("ran_suite_wide", False, suite))
        checks.append(_chk("broken_tests_referenced", False, named))
    return _finish(tier, checks)


def audit_fragile_solve(record: dict, agent_text: str, test_files, resolved) -> dict:
    """Re-derive a fragile-success oracle-touch record (awe/fragile.py — the
    §0ag deferral executed, v0.37): edited gold-test files, destructiveness,
    and severity, all recomputed from the raw agent diff + the requirements
    record's test_patch files — without importing fragile. Includes the OUTCOME
    atom (records are valid only on RESOLVED cases — the mirror of
    _audit_gating_test's failed-only atom)."""
    if test_files is None or resolved is None or agent_text is None:
        return _finish(ASSOCIATIONAL, [], unverifiable=True)
    touched = sorted({fd.path for fd in parse_unified_diff(agent_text or "")} & set(test_files))
    cl = changed_lines(agent_text or "")
    destructive = sorted(f for f in touched if (cl.get(f) or {}).get("removed"))
    checks = [
        _chk("outcome", True, resolved),
        _chk("edited_test_files", record.get("edited_test_files"), touched),
        _chk("non_empty", True, bool(touched)),
        _chk("modified_existing", record.get("modified_existing"), destructive),
        _chk("severity", record.get("severity"), "high" if destructive else "medium"),
    ]
    return _finish(ASSOCIATIONAL, checks)


def _audit_model_inferred(fault, agent, inst, traj) -> dict:
    from .judge import REQUIREMENT_ERRORS, PLANNING_ERRORS, verdict_cached

    ev = fault.get("detector_evidence", {})
    et, cap = fault["error_type"], fault["capability"]
    valid_ids = {s.step_id for s in traj.steps} if traj is not None else set()
    # error type must belong to THIS capability's closed set — the old union test
    # let a capability/error-type cross-swap audit clean.
    cap_errors = (
        REQUIREMENT_ERRORS if cap == "requirement_understanding" else PLANNING_ERRORS if cap == "task_planning" else {}
    )
    steps = ev.get("evidence_steps") or []
    checks = [
        _chk("error_type_in_capability", True, et in cap_errors),
        _chk("confidence_gated", True, fault.get("confidence") in ("high", "medium")),
        # a BLAMED judge fault must cite evidence (the gate that admitted it
        # required steps) — deleting the citations from a persisted fault is a
        # tamper, not a pass. NB: all([]) is True, hence the explicit bool.
        _chk("evidence_steps_cited", True, bool(steps)),
    ]
    if traj is not None:
        checks.append(_chk("evidence_steps_valid", True, all(s in valid_ids for s in steps)))
    # Cross-check against the CACHED judge verdict when it is readable (same
    # model env as the reports): relabeling the fault, rewriting its citations,
    # or upgrading its confidence after the fact is then detectable.
    rec = None
    try:
        rec = verdict_cached(agent, inst)
    except Exception:
        rec = None
    v = ((rec or {}).get("verdict") or {}).get(cap)
    if v:
        checks.append(_chk("matches_cached_verdict_label", v.get("error_type"), et))
        checks.append(_chk("matches_cached_verdict_confidence", v.get("confidence"), fault.get("confidence")))
        checks.append(_chk("matches_cached_verdict_steps", v.get("evidence_steps") or [], steps))
    return _finish(MODEL_INFERRED, checks, unverifiable=traj is None)


def _load_inputs(agent: str, inst: str):
    from . import config
    from .outcomes import resolved as _resolved
    from .adapters import load_normalized

    gold_text, test_files, problem_statement, fail_to_pass = "", None, None, None
    rp = config.requirements_path(inst)
    if rp.exists():
        import json

        req = json.loads(rp.read_text(encoding="utf-8"))
        gold_text = req.get("gold_patch") or ""
        problem_statement = req.get("problem_statement") or ""
        test_files = sorted({fd.path for fd in parse_unified_diff(req.get("test_patch") or "")})
        ftp = req.get("FAIL_TO_PASS")
        if isinstance(ftp, str):  # stored as a JSON-encoded string
            try:
                ftp = json.loads(ftp)
            except ValueError:
                ftp = None
        fail_to_pass = list(ftp) if isinstance(ftp, list) else None
    from .benchmark import resolve_profile

    profile = resolve_profile(req if rp.exists() else {})
    pp = config.patch_path(agent, inst)
    agent_text = pp.read_text(errors="replace", encoding="utf-8") if pp.exists() else ""
    from .outcomes import load_pertest

    return (
        gold_text,
        agent_text,
        _resolved(agent, inst),
        load_normalized(agent, inst),
        test_files,
        problem_statement,
        fail_to_pass,
        load_pertest(agent, inst),
        profile,
    )


def audit_case(agent: str, inst: str, persisted: dict | None = None, _inputs=None) -> dict:
    """Audit every fault in a case. `persisted` defaults to the matching record in
    data/dossier/faults.json, falling back to a fresh dossier.case_record. `_inputs`
    is a test seam (gold_text, agent_text, resolved, traj)."""
    if persisted is None:
        from .dossier import case_record

        persisted = case_record(agent, inst) or {
            "case_id": f"{agent}__{inst}",
            "agent": agent,
            "instance_id": inst,
            "faults": [],
        }
    inputs = _inputs or _load_inputs(agent, inst)
    problem_statement, fail_to_pass = None, None
    pertest = inputs[7] if len(inputs) >= 8 else None
    profile = inputs[8] if len(inputs) >= 9 else None
    if len(inputs) >= 7:
        (gold_text, agent_text, resolved, traj, test_files, problem_statement, fail_to_pass) = inputs[:7]
    elif len(inputs) == 6:
        gold_text, agent_text, resolved, traj, test_files, problem_statement = inputs
    elif len(inputs) == 5:
        gold_text, agent_text, resolved, traj, test_files = inputs
    else:  # legacy 4-tuple test seam
        gold_text, agent_text, resolved, traj = inputs
        test_files = None
    results, counts = [], {RE_DERIVED: 0, STRUCTURAL_OK: 0, UNVERIFIABLE: 0, MISMATCH: 0}
    for f in persisted.get("faults", []):
        a = audit_fault(
            f,
            agent=agent,
            inst=inst,
            gold_text=gold_text,
            agent_text=agent_text,
            resolved=resolved,
            traj=traj,
            test_files=test_files,
            problem_statement=problem_statement,
            fail_to_pass=fail_to_pass,
            pertest=pertest,
            profile=profile,
        )
        results.append(a)
        counts[a["verdict"]] = counts.get(a["verdict"], 0) + 1
    return {"case_id": persisted.get("case_id", f"{agent}__{inst}"), "faults": results, "verdict_counts": counts}


def audit_fault(fault: dict, *, profile=None, **kwargs) -> dict:
    from .benchmark import using_profile, resolve_profile

    with using_profile(profile or resolve_profile()):
        return _audit_fault_impl(fault, **kwargs)


def _audit_fault_impl(
    fault: dict,
    *,
    agent: str,
    inst: str,
    gold_text: str,
    agent_text: str,
    resolved,
    traj,
    test_files=None,
    problem_statement=None,
    fail_to_pass=None,
    pertest=None,
) -> dict:
    tier = strength_of(fault["capability"], fault.get("error_type"))
    et = fault.get("error_type")
    # v0.39 types
    if et == "test_expectation_rewritten":
        return _audit_test_rewrite(fault, agent_text, test_files, resolved)
    if et in ("partial_fix_only", "regression_introduced"):
        return _audit_pertest(fault, pertest, resolved)
    if et == "regression_test_missing":
        return _audit_pertest(fault, pertest, resolved, traj=traj, tier=ASSOCIATIONAL)
    # relevant_change_omitted / gave_up_early carry the ASSOCIATIONAL tier (their
    # capability assignment is interpretive) but their FACTS — read-not-edited;
    # no patch + no investigation — are fully re-derivable from raw inputs, so
    # they go through the deductive auditor with their honest tier label.
    if et in ("relevant_change_omitted", "gave_up_early"):
        return _audit_deductive(fault, gold_text, agent_text, resolved, traj, tier=tier)
    if et == "gating_test_modified":  # legacy (pre-v0.39 persisted records)
        return _audit_gating_test(fault, agent_text, test_files, resolved)
    if et == "solved_unrelated_problem":
        return _audit_solved_unrelated(fault, gold_text, agent_text, problem_statement, resolved)
    if et == "excessive_extra_edits":
        return _audit_extra_edits(fault, gold_text, agent_text, resolved)
    if tier == DEDUCTIVE:
        return _audit_deductive(fault, gold_text, agent_text, resolved, traj)
    if tier == ASSOCIATIONAL:
        return _audit_associational(fault, traj, test_files=test_files, fail_to_pass=fail_to_pass)
    return _audit_model_inferred(fault, agent, inst, traj)
