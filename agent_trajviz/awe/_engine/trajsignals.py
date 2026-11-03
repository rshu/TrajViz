"""Signals derived from the normalized trajectory: read-set, test execution,
success claims, and repair-loop edits. Combined with gold + patch + outcome,
these complete the read/edit/gold triangulation and enable verification errors."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from . import config
from .diffutil import is_test_file
from .gold import GoldReference
from .verification import known_zero_test_calls, tool_observations
from .trajectory import (
    NormalizedTrajectory,
    looks_like_success_claim,
    looks_like_test_failure,
    classify_tool_error,
    looks_like_environment_error,
    SCHEMA_VERSION,
    read_observations,
    is_suite_wide_test_command,
    observed_test_runs,
    READ,
    EDIT,
    WRITE,
    RUN_COMMAND,
    RUN_TEST,
    SEARCH,
    DELEGATE,
    PLAN,
    PLAN_OPEN,
)


# Environment-repair commands that legitimately change the world between two
# identical test invocations (v0.32 — suppresses false retry_without_change):
# installs/builds only, deliberately narrow so an unrelated `ls` never counts.
_ENV_REPAIR_RE = re.compile(
    r"\b(?:pip3?\s+install|conda\s+install|apt(?:-get)?\s+install|"
    r"npm\s+(?:install|ci)|python[\w.]*\s+setup\.py\s+(?:install|develop|build)|"
    # bare `make` is a build; a make invocation whose SEGMENT names a
    # test/check/lint target runs checks against an UNCHANGED world and must
    # not excuse a blind retry. The lookahead scans to the end of the shell
    # segment (v0.37 review fix — the v0.36 single-token lookahead let
    # `make -j4 test` / `make -C src check` / `make V=1 lint` through).
    r"pip3?\s+uninstall|make\b(?![^&|;]*\b(?:test|tests|check|lint)\b))"
)


def _gold_test_tokens(gold: GoldReference) -> set[str]:
    """SPECIFIC anchors for 'did the command run the gating test': the test
    function name(s) and the test file (full path + basename). Generic path
    components (the package dir, 'tests', the module) are deliberately excluded —
    matching them over-credits any command that merely mentions the package, so a
    command had to name the actual gating test/file to count as referencing it."""
    toks: set[str] = set()
    for tid in gold.fail_to_pass:
        toks.update(re.findall(r"test_\w+", tid))  # gating test function(s)
        toks.update(re.findall(r"[\w./-]+_test\.py", tid))  # *_test.py style
    for f in gold.test_patch_files:
        toks.add(f)  # full repo-relative path
        toks.add(f.split("/")[-1].rsplit(".", 1)[0])  # file basename, e.g. test_binned
    return {t for t in toks if len(t) >= 5}


@dataclass
class TrajectorySignals:
    n_steps: int
    read_set: list[str]
    edit_set_traj: list[str]
    ran_any_test: bool
    fail_to_pass_referenced: bool
    claimed_success_final: bool
    claimed_success_any: bool
    repeated_edit_files: list[str]
    edits_per_file: dict
    n_tool_errors: int
    n_test_runs: int
    hidden_reasoning: bool  # e.g. Codex: cannot assert "ignored X" from absence
    n_test_attempts: int = 0
    attempted_read_set: list[str] = field(default_factory=list)
    unobserved_read_set: list[str] = field(default_factory=list)
    # Self-Repair Loop (ordering-aware; encountered_failure is the OPPORTUNITY)
    encountered_failure: bool = False
    n_failure_events: int = 0
    repair_not_reverified: bool = False
    repeated_ineffective_attempt: bool = False
    # The LAST observed (main-thread) test result was a failure — grounds the
    # ordering-aware test_result_misinterpreted anchor ("failing output followed
    # by a success claim"): a final claim after a last-run PASS is a local-vs-eval
    # divergence, not a misinterpretation.
    last_test_failed: bool = False
    # Gating-run-specific outcome (v0.35, schema 14): among MAIN-thread test runs
    # whose command NAMES a gating test/file (the same >=5-char token bar as
    # fail_to_pass_referenced), was at least one result observed — and did the
    # LAST one fail? Splits last_test_failed's "last test of ANY kind" looseness
    # (the documented reason DESIGN.md §0ai rejected the environment fault_side
    # arm). CAVEAT the consumer must respect: F2P tests are usually ADDED by the
    # eval's test_patch, so a green "gating-referenced" run may have executed the
    # BASE version of the gating file — the failed direction is strong evidence,
    # the green direction is not proof the gating tests passed locally.
    gating_run_observed: bool = False
    last_gating_run_failed: bool = False
    # Execution PROOF, one level stronger than command intent (v0.36, schema 15):
    # a gating test FUNCTION name (from the FAIL_TO_PASS ids) appeared in a
    # main-thread test run's OUTPUT — the test demonstrably ran. Asymmetric by
    # nature: failing runs name the test (FAILED tests/x.py::test_y) while quiet
    # green runs often don't, so absence proves nothing; the observed-green form
    # is the affirmative signature the environment fault_side arm would need.
    gating_exec_observed: bool = False
    last_gating_exec_failed: bool = False
    # A test invocation with NO selector (bare pytest / tox / make test /
    # unittest discover) ran during the run: the agent ran everything locally
    # available, so "did not run the relevant test" (the medium
    # relevant_test_not_run arm) is not established. All-thread, like
    # ran_any_test.
    ran_suite_wide: bool = False
    # Delegation accounting (Raj model—subagent edge; 106/500 claude_code runs
    # carry sub-agent threads no signal consumed before v0.36). BEHAVIORAL until
    # calibration says otherwise: main-thread DELEGATE calls, whether any linked
    # result carries the scaffold error flag, and whether a DELEGATE call has NO
    # result at all (truncation-guarded — a cut-edge dangling call is the
    # harness's kill, not a lost delegation).
    # v20 (schema 20): the scaffold's explicit plan tool (PLAN steps, main
    # thread). Behavioral cohort features first (DESIGN §0ay) — no blame path
    # until they clear the calibration gate.
    n_plan_updates: int = 0
    plan_declared: bool = False
    plan_before_first_edit: bool = False
    plan_items_total: int = 0  # items in the LAST plan update
    plan_items_open_at_end: int = 0  # of those, still pending / in_progress
    n_delegations: int = 0
    delegate_error_result: bool = False
    delegate_no_result: bool = False
    # Weak-oracle repro (SWT-Bench lens): the agent authored test file(s), ran
    # them, and NO referencing run was ever observed to fail — a reproduction
    # that never reproduced. False when authored tests were never run (that is
    # authored_test_never_run's fact) or when any referencing run failed.
    authored_test_ran_never_failed: bool = False
    # The run edited source AFTER its last test run (or never tested around an
    # edit at all): nothing verified the final state of the patch. Grounds the
    # code_verification error `no_test_after_final_edit` — the "verified early,
    # edited after" hole that repair_not_reverified (failure-observed variant)
    # cannot see. Main-thread ordering; False when the run made no edits.
    no_test_after_final_edit: bool = False
    # After a failure event, the SAME test command was re-run with NO source edit
    # in between: a blind retry adds no new information (grounds the
    # `retry_without_change` self-repair error).
    retry_without_change: bool = False
    # Tool Use (mechanics)
    encountered_tool_error: bool = False
    tool_failure_not_retried: bool = False
    # >=2 EDIT/WRITE tool results with a patch-failure signature (bad_arguments):
    # the agent's edit mechanics are looping on a diff that does not apply
    # (grounds the `repeated_patch_apply_failure` tool-use error).
    repeated_patch_apply_failure: bool = False
    # The SAME target path drew an invalid_target error >=2 times: the agent's
    # world model was not updated by the first "no such file" (grounds the
    # `hallucinated_path_unrecovered` tool-use error; a one-off typo is not it).
    hallucinated_path_unrecovered: bool = False
    # The run ended on a dangling tool_call (no result): killed/truncated, NOT
    # "gave up" — used to suppress false tool_failure_not_retried /
    # repair_not_reverified at the cut edge.
    truncated_run: bool = False
    # Behavioral tool-error accounting (NOT blame): every mechanical tool failure,
    # whether or not the agent recovered. n_tool_calls is the non-test denominator;
    # tool_error_subtypes maps subtype -> count (see trajectory.classify_tool_error).
    n_tool_calls: int = 0
    n_tool_call_errors: int = 0
    tool_error_subtypes: dict = field(default_factory=dict)
    # An environment / dependency error surfaced in a command/test-run result (the
    # scout-discovered 'environment-failure' mode). File-read/search results are
    # never scanned — they echo file content.
    # NB: "encountered", not "blocked" — often transient/recovered (see cohort test).
    environment_error: bool = False
    # Test files the agent itself created/edited during the run, and whether NONE
    # of them lacks an observed post-edit execution ("wrote the repro, never ran
    # it" — grounds the `authored_test_never_run` verification error).
    authored_test_files: list = field(default_factory=list)
    authored_test_never_run: bool = False
    authored_test_unrun_files: list = field(default_factory=list)
    # v0.39 (schema 17): main-thread TEST files edited after the FIRST observed
    # failure event — grounds the `after_observed_failure` attribute of
    # test_expectation_rewritten (a rewrite in a repair step, not an up-front
    # repro). Empty when no failure was observed.
    test_files_edited_after_failure: list = field(default_factory=list)
    # v0.39 (schema 17): the run's test commands (all threads, deduped, each
    # truncated to 400 chars) — lets the eval+ tier ask "did any command NAME
    # this test file" for tests the harness reports broken
    # (regression_test_missing) without re-reading the trajectory.
    test_commands: list = field(default_factory=list)
    # BEHAVIORAL (cohort features, no blame path): the max number of READs of a
    # single file the run never edited, and the >=4 threshold flag — the
    # objective form of the scout's "analysis-loop" cluster (re-reading the same
    # code without acting on it).
    max_reads_single_file: int = 0
    exploration_loop: bool = False
    # The run's FIRST source edit precedes any command/test execution (main
    # thread): nothing was ever run — no repro, no test, no probe — before code
    # was changed. Calibration v0.26: NOT failure-associated (cohort feature only).
    edited_before_any_run: bool = False
    # BEHAVIORAL (cohort feature, no blame path): MAST-3 "step repetition". A
    # non-edit tool-call signature (canonical_action, key) that fires again with
    # NO EDIT/WRITE strictly between it and its previous firing is a blind repeat —
    # the same action against an unchanged codebase (generalizes retry_without_change).
    # n_unproductive_repeats is the graded count; step_repetition is the >=1 presence
    # flag (threshold-free). Edit/write repetition stays with repeated_ineffective_
    # attempt / repeated_patch_apply_failure. Sourced from ATLAS's MAST seed.
    n_unproductive_repeats: int = 0
    step_repetition: bool = False
    # Edit-content signals (v0.27; need adapters that capture old/new lines —
    # unverifiable-shaped False when content is absent, e.g. legacy logs):
    # an edit's replacement content was later swapped back to that edit's
    # ORIGINAL content on the same file — the agent undid its own change.
    edit_reverted_own_change: bool = False
    # {file: n} — max count of gold-ADDED lines any single trajectory edit
    # matched (same normalization as the gold map). SIGNAL ONLY: the derived
    # "walked back from a near-gold edit" condition calibrated PROTECTIVE
    # (RR 0.39 — solved runs iterate away from near-gold edits more often), so
    # no detector consumes this; kept for calibration/discovery work.
    traj_matched_gold_lines: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


# Actions whose result text may be scanned for mechanical-error signatures.
# READ, SEARCH, and DELEGATE results ECHO file content / sub-agent prose (a grep
# for the issue's error message returns lines containing 'No such file or
# directory'), so they are never text-scanned; nor is a result whose call linkage
# is missing (we cannot prove it isn't a read). An explicit scaffold error flag
# still counts for any action.
_TEXT_SCAN_ACTIONS = (RUN_COMMAND, EDIT, WRITE)


def _effective_tool_error(result, call) -> tuple:
    """(is_error, subtype) for a NON-test tool result. The explicit scaffold error
    flag wins (subtype from the text if recognizable, else 'other'); otherwise —
    for command/edit/write results only — fall back to the high-precision text
    signature, which recovers mechanical errors for scaffolds that emit no flag
    (Codex)."""
    action = call.canonical_action if call is not None else None
    if result.tool_error:
        return True, (classify_tool_error(result.output_excerpt) or "other")
    if action in _TEXT_SCAN_ACTIONS:
        sub = classify_tool_error(result.output_excerpt)
        if sub:
            return True, sub
    return False, None


def _run_failed(result) -> bool:
    """Did a test-run result indicate failure? The recorded EXIT CODE (Codex
    metadata) is ground truth when present — 0 vetoes text markers echoed in
    passing output, non-0 catches verbose failures whose summary the excerpt
    missed. Text heuristics are the fallback for scaffolds without exit codes."""
    ec = result.args.get("exit_code") if isinstance(result.args, dict) else None
    if isinstance(ec, int):
        return ec != 0
    return result.tool_error or looks_like_test_failure(result.output_excerpt)


def _repair_and_tool_signals(traj, profile=None) -> dict:
    """Step-ordering analysis for Self-Repair Loop + Tool Use mechanics.

    ORDERING signals use the MAIN thread only: the Claude Code adapter appends
    sub-agent steps after the main trajectory, so cross-thread step-id
    comparisons are meaningless (a mid-run sub-agent edit would otherwise look
    like it happened after the final test run). Tool mechanics are counted over
    every thread, with the retry window kept within its own thread."""
    zero = known_zero_test_calls(traj)
    steps = traj.steps
    call_by_id = {(s.subagent_id, s.call_id): s for s in steps if s.type == "tool_call" and s.call_id}
    main = [s for s in steps if s.subagent_id is None]
    # Ordering analysis resolves results within the MAIN thread only: call_ids
    # are adapter-passthrough with no uniqueness guarantee, so a global
    # last-wins index could hand a main-thread test call an appended
    # sub-agent's result (v0.37 review fix — mirrors the audit's convention).
    result_by_id = {s.call_id: s for s in main if s.type == "tool_result" and s.call_id}

    test_results = {id(call): result for _, call, _, result, _ in tool_observations(traj) if result is not None}
    fail_event_steps = []
    failure_commands = {}
    test_outcomes = []  # (step_id, failed) in main-thread order
    test_records = []  # (step_id, cmd, failed, output_excerpt)
    for i, s in enumerate(main):
        if s.type == "tool_call" and s.test_run:
            r = test_results.get(id(s))
            if r is None:
                continue
            failed = _run_failed(r)
            if id(s) not in zero:
                test_outcomes.append((s.step_id, failed))
                test_records.append(
                    (s.step_id, (s.args or {}).get("command", "") or "", failed, r.output_excerpt or "")
                )
            if failed:
                aggregation = (r.args or {}).get("output_aggregation") or {}
                event = aggregation.get("completion_step", s.step_id)
                fail_event_steps.append(event)
                failure_commands[event] = (s.args or {}).get("command", "").strip()

    edits = [(s.step_id, f) for s in main if s.canonical_action in (EDIT, WRITE) for f in (s.files or [None])]
    test_ids = [s.step_id for s in main if s.test_run and id(s) not in zero]
    _truncated_early = bool(main) and main[-1].type == "tool_call"
    # v0.32: the anchor says "no test after the last SOURCE edit" — a trailing
    # write to a test file, summary .md, or scratch note is not an unverified
    # source change, so it must not arm the flag. Files the parser can't name
    # (None) stay counted conservatively.
    _NONSOURCE_SUFFIXES = (".md", ".rst", ".txt")
    src_edits = [
        (sid, f)
        for sid, f in edits
        if f is None or (not is_test_file(f, profile) and not f.endswith(_NONSOURCE_SUFFIXES))
    ]
    last_edit_any = max((sid for sid, _ in src_edits), default=None)
    no_test_after_final_edit = (
        last_edit_any is not None
        and not any(t > last_edit_any for t in test_ids)
        # killed ≠ gave up: an edit at the cut edge of a truncated run had no
        # opportunity to be re-verified (mirrors the repair_not_reverified guard)
        and not (_truncated_early and last_edit_any >= main[-1].step_id - 3)
    )
    first_edit = min((sid for sid, _ in edits), default=None)
    run_ids = [s.step_id for s in main if s.canonical_action in (RUN_COMMAND, RUN_TEST)]
    edited_before_any_run = first_edit is not None and not any(r < first_edit for r in run_ids)

    # Killed vs gave up: a run whose main thread ends on a dangling tool_call
    # (a result, had one arrived, would be a later step) was truncated by the
    # harness — absence of a follow-up action at the cut edge is not evidence
    # the agent chose to stop.
    truncated = bool(main) and main[-1].type == "tool_call"

    # Delegation accounting (main-thread DELEGATE calls; results may live in
    # the global result index — linkage, not ordering, is what matters here).
    delegate_calls = [s for s in main if s.type == "tool_call" and s.canonical_action == DELEGATE]
    delegate_error = any(
        s.call_id and (r := result_by_id.get(s.call_id)) is not None and r.tool_error for s in delegate_calls
    )
    delegate_no_result = any(
        s.call_id
        and s.call_id not in result_by_id
        # cut-edge dangling call = harness kill, not a lost delegation
        and not (truncated and s.step_id >= main[-1].step_id - 3)
        for s in delegate_calls
    )

    encountered = bool(fail_event_steps)
    repair_not_reverified = repeated = retry_without_change = False
    test_edits_after_failure: list = []
    if encountered:
        first_fail = min(fail_event_steps)
        edits_after = [(sid, f) for sid, f in edits if sid > first_fail]
        # v0.39: which TEST files the agent edited once it had seen a failure —
        # the ordering attribute of test_expectation_rewritten.
        test_edits_after_failure = sorted({f for _sid, f in edits_after if f and is_test_file(f, profile)})
        if edits_after:
            last_edit = max(sid for sid, _ in edits_after)
            repair_not_reverified = not any(t > last_edit for t in test_ids)
            # ...unless the run was cut right after the edit — the agent never
            # got the chance to re-verify (killed ≠ gave up).
            if truncated and last_edit >= main[-1].step_id - 3:
                repair_not_reverified = False
            fc = Counter(f for _, f in edits_after if f)
            repeated = any(c >= 2 for c in fc.values())
        # Blind retry: the SAME test command re-run after a failure with no
        # source edit AND no environment-repair command in between — no new
        # information was added. v0.32: `pip install x` (or conda/apt/npm/
        # setup.py) between identical runs is a rational environment fix, not
        # a blind retry — "an unchanged input cannot produce a different
        # outcome" is false when the environment changed.
        cmd_of = {
            s.step_id: (s.args or {}).get("command", "").strip()
            for s in main
            if s.type == "tool_call" and s.test_run and id(s) not in zero
        }
        env_repair_ids = sorted(
            s.step_id
            for s in main
            if s.type == "tool_call"
            and s.canonical_action == RUN_COMMAND
            and _ENV_REPAIR_RE.search((s.args or {}).get("command", "") or "")
        )
        edit_ids = sorted(sid for sid, _ in edits)
        change_ids = sorted(set(edit_ids) | set(env_repair_ids))
        for fid in fail_event_steps:
            fcmd = failure_commands.get(fid, "")
            if not fcmd:
                continue
            for tid, tcmd in cmd_of.items():
                if tid > fid and tcmd == fcmd and not any(fid < e < tid for e in change_ids):
                    retry_without_change = True
                    break
            if retry_without_change:
                break

    # Tool mechanics: every NON-test mechanical tool error (effective = flag OR text
    # signature) is counted for the behavioral rate; tool_failure_not_retried fires
    # only when the agent then issued no further tool call (gave up). RUN_TEST results
    # are excluded — a failing test is a test failure, not a tool-call failure.
    n_tool_calls = sum(1 for s in steps if s.type == "tool_call" and s.canonical_action != RUN_TEST)
    n_errors = 0
    subtypes = Counter()
    n_patch_apply_failures = 0
    invalid_target_paths = Counter()  # path -> invalid_target error count
    enc_tool_err = tool_not_retried = False
    threads: dict = {}
    for s in steps:
        threads.setdefault(s.subagent_id, []).append(s)
    for thread in threads.values():
        thread_truncated = bool(thread) and thread[-1].type == "tool_call"
        for i, s in enumerate(thread):
            if s.type != "tool_result":
                continue
            call = call_by_id.get((s.subagent_id, s.call_id))
            if call is not None and call.canonical_action == RUN_TEST:
                continue
            is_err, subtype = _effective_tool_error(s, call)
            if not is_err:
                continue
            n_errors += 1
            subtypes[subtype] += 1
            enc_tool_err = True
            if call is not None and call.canonical_action in (EDIT, WRITE) and subtype == "bad_arguments":
                n_patch_apply_failures += 1
            if subtype == "invalid_target" and call is not None:
                for f in call.files or []:
                    if f:
                        invalid_target_paths[f] += 1
            if not any(t.type == "tool_call" for t in thread[i + 1 : i + 6]):
                # Not-retried needs an OPPORTUNITY to retry: if the thread was
                # cut (dangling final call) and this error sits at the cut edge,
                # the missing follow-up is truncation, not giving up.
                if not (thread_truncated and i >= len(thread) - 6):
                    tool_not_retried = True

    return {
        "encountered_failure": encountered,
        "n_failure_events": len(fail_event_steps),
        "last_test_failed": bool(test_outcomes) and test_outcomes[-1][1],
        "n_delegations": len(delegate_calls),
        "delegate_error_result": delegate_error,
        "delegate_no_result": delegate_no_result,
        # private: consumed (popped) by compute() for the gating / authored-test
        # outcome scans — never a TrajectorySignals field.
        "_test_records": test_records,
        "no_test_after_final_edit": no_test_after_final_edit,
        "test_files_edited_after_failure": test_edits_after_failure,
        "edited_before_any_run": edited_before_any_run,
        "repair_not_reverified": repair_not_reverified,
        "repeated_ineffective_attempt": repeated,
        "retry_without_change": retry_without_change,
        "encountered_tool_error": enc_tool_err,
        "tool_failure_not_retried": tool_not_retried,
        "repeated_patch_apply_failure": n_patch_apply_failures >= 2,
        "hallucinated_path_unrecovered": any(c >= 2 for c in invalid_target_paths.values()),
        "truncated_run": truncated,
        "n_tool_calls": n_tool_calls,
        "n_tool_call_errors": n_errors,
        "tool_error_subtypes": dict(subtypes),
    }


def _step_repetition(traj: NormalizedTrajectory) -> int:
    """MAST-3 (deductive): count unproductive repeats — a NON-edit tool-call
    signature that fires again with NO EDIT/WRITE action strictly between it and its
    previous firing. World change = EDIT/WRITE or an environment-repair COMMAND
    (_ENV_REPAIR_RE on RUN_COMMAND only, matching retry_without_change, so a re-test
    after an install is not counted — but a blind re-run of `make test` is).
    Generalizes retry_without_change. Main thread only."""
    progress = 0
    last_progress_at: dict = {}  # signature -> progress value at its previous firing
    n_unproductive = 0
    for s in traj.steps:
        if s.subagent_id is not None or s.type != "tool_call":
            continue
        act = s.canonical_action
        if act in (EDIT, WRITE):
            progress += 1
            continue
        a = s.args or {}
        cmd = a.get("command") or ""
        if act in (RUN_COMMAND, RUN_TEST):
            key = re.sub(r"\s+", " ", cmd).strip()
        elif act == SEARCH:
            # native search tools carry no files: identity = pattern + scope
            # (claude_code {pattern,path}, opencode {pattern,include})
            what = (a.get("pattern") or a.get("query") or cmd).strip()
            where = a.get("path") or a.get("include") or "|".join(sorted(s.files or []))
            key = f"{what}@{where}" if what else ""
        elif act == READ:
            key = "|".join(sorted(s.files or [])).strip()
            # a paginated read of one large file is one logical read, not thrash
            if key and (a.get("offset") is not None or a.get("limit") is not None):
                key += f"@{a.get('offset')}+{a.get('limit')}"
        else:
            key = ""
        if not key:
            continue
        sig = (act, key)
        if last_progress_at.get(sig) == progress:  # seen before AND no world change since
            n_unproductive += 1
        last_progress_at[sig] = progress
        if act == RUN_COMMAND and _ENV_REPAIR_RE.search(cmd):
            progress += 1  # install/build changed the world (retry_without_change parity)
    return n_unproductive


def _plan_signals(traj) -> dict:
    """Main-thread PLAN steps (v20): how many, whether one preceded the first
    source-or-test edit, and the end state of the LAST plan (items left
    pending / in_progress when the run ended)."""
    main = [s for s in traj.steps if s.subagent_id is None]
    plans = [s for s in main if s.type == "tool_call" and s.canonical_action == PLAN]
    first_edit = min((s.step_id for s in main if s.canonical_action in (EDIT, WRITE)), default=None)
    items = list((plans[-1].args or {}).get("items") or []) if plans else []
    return {
        "n_plan_updates": len(plans),
        "plan_declared": bool(plans),
        "plan_before_first_edit": bool(plans) and (first_edit is None or plans[0].step_id < first_edit),
        "plan_items_total": len(items),
        "plan_items_open_at_end": sum(1 for it in items if isinstance(it, dict) and it.get("status") in PLAN_OPEN),
    }


def compute(traj: NormalizedTrajectory, gold: GoldReference) -> TrajectorySignals:
    zero = known_zero_test_calls(traj)
    read_set, attempted_reads, unknown_reads = read_observations(traj)
    edit_traj = set()
    test_cmds = []
    edits = Counter()
    n_tool_errors = 0
    env_blocked = False
    call_by_id = {(s.subagent_id, s.call_id): s for s in traj.steps if s.type == "tool_call" and s.call_id}

    for s in traj.steps:
        if s.canonical_action in (EDIT, WRITE):
            edit_traj.update(f for f in s.files if f)
            for f in s.files:
                if f:
                    edits[f] += 1
        if s.test_run and id(s) not in zero:
            test_cmds.append(s.args.get("command", "") or "")
        if s.type == "tool_result":
            if s.tool_error:
                n_tool_errors += 1
            # Env-error scan runs on command/test-run output ONLY: file reads and
            # search results echo file content ('except ImportError:' in a compat
            # shim would otherwise flag the whole run as env-blocked).
            call = call_by_id.get((s.subagent_id, s.call_id))
            if (
                not env_blocked
                and call is not None
                and call.canonical_action in (RUN_COMMAND, RUN_TEST)
                and looks_like_environment_error(s.output_excerpt)
            ):
                env_blocked = True

    tokens = _gold_test_tokens(gold)
    ftp_ref = any(any(tok in cmd for tok in tokens) for cmd in test_cmds)

    # Tests the agent AUTHORED (created/edited during the run) and never ran:
    # "wrote the reproduction, never executed it". A file counts as run when a
    # test command names its path / basename / basename stem (>=5 chars, same
    # specificity bar as the gold-test tokens).
    # Re-read-without-edit loop (behavioral): count READs per file; the flag
    # fires when a file the run never edited was read >=4 times.
    reads_per_file = Counter(f for s in traj.steps if s.canonical_action == READ for f in s.files if f)
    unedited_reads = {f: c for f, c in reads_per_file.items() if f not in edit_traj}
    max_reads = max(unedited_reads.values(), default=0)

    # Edit-content analysis (adapters that capture old/new lines; else no-op).
    # (a) self-revert: an observed successful edit exactly reverses the last
    #     directed edit on the same file/thread; legacy line sets abstain.
    # (b) per-file max overlap between any single edit's NEW lines and the
    #     gold-ADDED lines (same normalization) — the "had it in hand" measure.
    edits_content = []
    for s in traj.steps:
        if s.canonical_action not in (EDIT, WRITE):
            continue
        per = (s.args or {}).get("per_file_lines") or {}
        for f in s.files or []:
            if not f:
                continue
            src = per.get(f, {} if per else (s.args or {}))
            edits_content.append((s.step_id, f, set(src.get("old_lines") or []), set(src.get("new_lines") or [])))
    from .trajectory import edit_reversals

    reverted_own = bool(edit_reversals(traj))
    by_file: dict = {}
    for _sid, f, ol, nl in edits_content:
        by_file.setdefault(f, []).append((ol, nl))
    traj_matched = {}
    for f, seq in by_file.items():
        g = gold.gold_changed_lines.get(f)
        if not g:
            continue
        g_add = set(g.get("added") or [])
        if not g_add:
            continue
        best = max((len(nl & g_add) for _, nl in seq), default=0)
        if best:
            traj_matched[f] = best

    authored_tests = sorted(f for f in edit_traj if is_test_file(f, gold.benchmark_profile or None))
    # Each authored file must be exercised after its latest edit, in the same
    # thread where ordering is observable. Commands show selection; observed
    # suite output may establish coverage without a filename in the command.
    last_auth_edit = {
        (s.subagent_id, f): i
        for i, s in enumerate(traj.steps)
        if s.canonical_action in (EDIT, WRITE)
        for f in s.files
        if is_test_file(f, gold.benchmark_profile or None)
    }
    positions = {id(s): i for i, s in enumerate(traj.steps)}
    auth_outcomes, unrun = [], set()
    observed_runs = observed_test_runs(traj)
    for (thread, f), edit_pos in last_auth_edit.items():
        stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        toks = {t for t in (f, f.rsplit("/", 1)[-1], stem) if len(t) >= 5}
        matches = [
            r
            for call, r in observed_runs
            if call.subagent_id == thread
            and positions[id(call)] > edit_pos
            and any(tok in ((call.args or {}).get("command", "") + "\n" + (r.output_excerpt or "")) for tok in toks)
        ]
        if not matches:
            unrun.add(f)
        auth_outcomes.extend(_run_failed(r) for r in matches)
    authored_never_run = bool(unrun)

    # Final-claim detection reads the MAIN thread's closing messages: the CC
    # adapter appends sub-agent steps after the main trajectory, so an appended
    # sub-agent's mid-run report must not pass for the run's final message.
    all_texts = traj.assistant_texts()
    texts = [s for s in all_texts if s.subagent_id is None] or all_texts
    tail = texts[-1].text if texts else ""
    claimed_final = looks_like_success_claim(tail)
    claimed_any = any(looks_like_success_claim(s.text) for s in all_texts)

    hidden = any(s.observability == "hidden" for s in traj.steps)
    rt = _repair_and_tool_signals(traj, gold.benchmark_profile or None)
    records = rt.pop("_test_records")
    n_unproductive = _step_repetition(traj)

    # Gating outcome, two evidence tiers from the same records:
    # (a) command INTENT — the run's command names a gating test/file;
    # (b) execution PROOF — the run's OUTPUT names a gating test function.
    gating_cmd = [(sid, failed) for sid, cmd, failed, _out in records if any(tok in cmd for tok in tokens)]
    # Function tokens come from the part AFTER '::' only: a bare findall over
    # the full id also captured the FILE stem (test_core from
    # tests/test_core.py::test_x), so a plain pytest progress line
    # ('tests/test_core.py ....') passed for execution proof (v0.37 review
    # fix). An id with no '::' carries no function name — it contributes none.
    fn_tokens = {
        t
        for tid in gold.fail_to_pass
        if "::" in tid
        for t in re.findall(r"test_\w+", tid.split("::", 1)[1])
        if len(t) >= 5
    }
    gating_exec = [(sid, failed) for sid, _cmd, failed, out in records if any(tok in out for tok in fn_tokens)]
    ran_suite_wide = any(is_suite_wide_test_command(c) for c in test_cmds)

    # Weak-oracle repro: authored tests were run, and no referencing run was
    # ever observed failing (main-thread observed outcomes; same token bar as
    # authored_test_never_run).
    authored_ran_never_failed = bool(auth_outcomes) and not unrun and not any(auth_outcomes)

    # v0.39: deduped test invocations (all threads — same scope as
    # ran_any_test / fail_to_pass_referenced). NOT truncated (v0.40): the
    # audit re-derives the named-test atom from the full raw command, so a
    # cached prefix would make an honest fault audit as a mismatch.
    _seen: set = set()
    test_commands: list = []
    for c in test_cmds:
        c = (c or "").strip()
        if c and c not in _seen:
            _seen.add(c)
            test_commands.append(c)

    plan = _plan_signals(traj)

    return TrajectorySignals(
        **rt,
        **plan,
        test_commands=test_commands,
        gating_run_observed=bool(gating_cmd),
        last_gating_run_failed=bool(gating_cmd) and gating_cmd[-1][1],
        gating_exec_observed=bool(gating_exec),
        last_gating_exec_failed=bool(gating_exec) and gating_exec[-1][1],
        ran_suite_wide=ran_suite_wide,
        authored_test_ran_never_failed=authored_ran_never_failed,
        n_steps=len(traj.steps),
        n_unproductive_repeats=n_unproductive,
        step_repetition=n_unproductive >= 1,
        read_set=sorted(read_set),
        attempted_read_set=sorted(attempted_reads),
        unobserved_read_set=sorted(unknown_reads),
        edit_set_traj=sorted(edit_traj),
        ran_any_test=bool(test_cmds),
        fail_to_pass_referenced=ftp_ref,
        claimed_success_final=claimed_final,
        claimed_success_any=claimed_any,
        repeated_edit_files=sorted(f for f, c in edits.items() if c >= 2),
        edits_per_file=dict(edits),
        n_tool_errors=n_tool_errors,
        n_test_runs=len(test_cmds),
        n_test_attempts=sum(s.type == "tool_call" and s.test_run for s in traj.steps),
        hidden_reasoning=hidden,
        environment_error=env_blocked,
        authored_test_files=authored_tests,
        authored_test_never_run=authored_never_run,
        authored_test_unrun_files=sorted(unrun),
        max_reads_single_file=max_reads,
        exploration_loop=max_reads >= 4,
        edit_reverted_own_change=reverted_own,
        traj_matched_gold_lines=traj_matched,
    )


_CACHE = config.PROJECT_DATA / "cache" / "trajsig"


def signals_cached(agent: str, instance_id: str, gold: GoldReference):
    """compute() with a disk cache — the trajectory parse is the expensive step,
    so the full report runs in seconds after the first pass. Entries embed
    trajectory.SCHEMA_VERSION: caches written by older signal/heuristic logic are
    recomputed, not silently served (dataclass defaults would otherwise mask
    missing/changed fields forever)."""
    from .adapters import load_normalized  # local import to avoid import cycle
    from .provenance import fingerprint, trajectory_fingerprint

    source = fingerprint({"trajectory": trajectory_fingerprint(agent, instance_id), "gold": asdict(gold)})
    p = _CACHE / agent / f"{instance_id}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("schema_version") == SCHEMA_VERSION and d.get("source_fingerprint") == source:
                return TrajectorySignals(**d["signals"])
        except Exception:
            pass
    traj = load_normalized(agent, instance_id)
    if traj is None:
        return None
    sig = compute(traj, gold)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "source_fingerprint": source, "signals": sig.to_dict()}), encoding="utf-8"
        )
    except Exception:
        pass
    return sig


def tool_behavior(agent: str, instances) -> dict:
    """Aggregate mechanical tool-call behavior over ALL of an agent's trajectories
    (resolved + failed) — a behavioral signal, not blame. Reports the non-test
    tool-call count, how many errored, the rate, how many tasks were affected, and
    the subtype breakdown. This is also the cross-agent CAPTURE-FIDELITY surface: a
    structurally-zero agent (e.g. pre-fix Codex, whose errors were never parsed) is
    visible here as 0 against non-zero peers."""
    from .gold import build_gold_reference
    from .outcomes import resolved

    n_traj = calls = errs = cases_err = 0
    subtypes: Counter = Counter()
    for inst in instances:
        if resolved(agent, inst) is None:
            continue
        try:
            gold = build_gold_reference(inst)
        except Exception:
            continue
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        n_traj += 1
        calls += sig.n_tool_calls
        errs += sig.n_tool_call_errors
        if sig.n_tool_call_errors:
            cases_err += 1
        for k, v in (sig.tool_error_subtypes or {}).items():
            subtypes[k] += v
    return {
        "agent": agent,
        "n_traj": n_traj,
        "n_tool_calls": calls,
        "n_tool_call_errors": errs,
        "tool_error_rate": round(errs / calls, 4) if calls else 0.0,
        "cases_with_error": cases_err,
        "case_rate": round(cases_err / n_traj, 4) if n_traj else 0.0,
        "subtypes": dict(subtypes),
    }
