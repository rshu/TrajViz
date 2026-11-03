"""Normalized trajectory model + cross-scaffold helpers.

A NormalizedStep is one atomic event. Adapters map each scaffold's native log
onto this so localization/verification signals and evidence citation work
identically across Claude Code, OpenCode, and Codex."""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import asdict, dataclass, field

# Bump when NormalizedStep fields or any parsing/heuristic below changes: the
# normtraj/trajsig disk caches embed this so stale entries are recomputed instead
# of silently serving pre-fix results (dataclass defaults would otherwise mask
# missing/changed fields forever).
# v3: + trajsignals.no_test_after_final_edit
# v4: + trajsignals retry_without_change / repeated_patch_apply_failure /
#       authored_test_never_run / truncated_run; timeout + unknown_tool subtypes
# v5: + trajsignals hallucinated_path_unrecovered / exploration_loop /
#       max_reads_single_file
# v6: + trajsignals edited_before_any_run
# v7: + edit steps carry normalized old_lines/new_lines content (CC/OC inputs,
#       codex apply_patch body); codex cwd tracking for relative read targets;
#       trajsignals edit_reverted_own_change / traj_matched_gold_lines
# v8: trajsignals soundness guards (v0.32): retry_without_change suppressed by
#       environment-repair commands between identical runs; no_test_after_final_edit
#       keyed to the last SOURCE edit (test/.md/.rst/.txt writes no longer arm it)
# v9: MAST-3 step_repetition / n_unproductive_repeats trajsignals fields added —
#       bump invalidates stale trajsig caches so the new signal actually recomputes.
# v10: step_repetition review fixes — SEARCH signatures keyed on pattern+scope
#       (was dead code: no adapter emits args["query"]); env-repair progress bump
#       restricted to RUN_COMMAND (blind `make test` re-runs now count); READ
#       signatures carry offset/limit spans (chunked reads of one large file are
#       one logical read); command signatures collapse internal whitespace.
#       Adapters now preserve Read offset/limit in step args.
# v11: Harbor ATIF-v1.x trajectories are normalized directly.
# v12: recognize multi-language Harbor test runners (Go, Maven, Gradle, Rust,
#      Node, .NET, Bazel, and CPython's ``-m test``) and timeout wrappers.
# v13: normalize Pi's v3 JSONL session transcript (Harbor agent/pi.txt).
# v14: trajsignals gating_run_observed / last_gating_run_failed — the outcome of
#      the last main-thread test run whose command NAMES a gating test/file
#      (the tight per-gating-run signal DESIGN.md §0ai reserved the environment
#      fault_side arm on).
# v15: batch-8 trajsignals — ran_suite_wide (selector-free test invocation),
#      gating_exec_observed / last_gating_exec_failed (gating test FUNCTION named
#      in a run's OUTPUT — execution proof, not command intent), delegation
#      accounting (n_delegations / delegate_error_result / delegate_no_result),
#      authored_test_ran_never_failed (weak-oracle repro); _ENV_REPAIR_RE no
#      longer excuses `make test|check|lint` as an environment repair.
# v16: PR-47 review fixes — is_suite_wide_test_command rejects no-run/subset
#      invocations (--collect-only/--lf/--ignore/--deselect, tox -e); fn_tokens
#      for gating_exec come only from the part after '::' (file stems leaked);
#      _ENV_REPAIR_RE's make lookahead scans the whole segment (flags/vars no
#      longer defeat it); main-thread ordering uses a main-thread result index
#      (cross-thread call_id collisions).
# v17: trajsignals test_files_edited_after_failure (test files edited after the
#      first observed failure — the after_observed_failure attribute of
#      test_expectation_rewritten) and test_commands (deduped test invocations,
#      for the eval+ regression_test_missing reference check).
# v19: adapters keep the recorded EXIT CODE on every shell result that has one
#      (OpenCode metadata.exit, Codex 'Process exited with code N' header,
#      Claude Code 'Exit code N' text); Codex write_stdin continuation chunks
#      are merged into the originating exec_command result; unittest's bare
#      'OK' verdict joins the clean-summary pattern (DESIGN §0ax).
# v20: PLAN canonical action — the scaffolds' plan tools (Claude Code
#      TodoWrite, OpenCode todowrite, Codex update_plan) keep their items and
#      statuses; trajsignals plan_* fields (DESIGN §0ay).
# v29: non-Python subcommand positions, compile/list/skip modes and suite-scope consistency.
SCHEMA_VERSION = 32

# Canonical action ontology (native tool name kept separately for fairness).
SEARCH, READ, EDIT, WRITE = "search", "read", "edit", "write"
RUN_COMMAND, RUN_TEST = "run_command", "run_test"
DELEGATE, OTHER, NONE = "delegate", "other", "none"
PLAN = "plan"  # v20: an explicit plan / todo-list update
PLAN_OPEN = ("pending", "in_progress")


def plan_items(raw) -> list:
    """Normalize a scaffold's plan payload to ``[{content, status}]``. Accepts
    Claude Code / OpenCode todo items (``content`` + ``status``), Codex plan
    steps (``step`` + ``status``) and bare strings; unknown statuses are kept
    verbatim, missing ones are ``"unknown"``. Shared by the adapters and any
    consumer that reads a PLAN step."""
    out = []
    for it in raw or []:
        if isinstance(it, str):
            out.append({"content": it.strip(), "status": "unknown"})
        elif isinstance(it, dict):
            content = it.get("content") or it.get("step") or it.get("title") or it.get("task") or ""
            status = str(it.get("status") or "unknown").strip().lower()
            out.append({"content": str(content).strip(), "status": status})
    return out


# Observability of a step's content.
OBSERVED, HIDDEN, INFERRED = "observed", "hidden", "inferred"


@dataclass
class NormalizedStep:
    step_id: int
    turn_id: int
    actor: str  # agent | tool | user | system
    type: str  # reasoning | message | tool_call | tool_result | patch | final_answer
    canonical_action: str = NONE
    native_tool: str = ""
    text: str = ""
    args: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)  # repo-relative paths touched
    test_run: bool = False
    tool_error: bool = False
    output_excerpt: str = ""
    call_id: str | None = None
    observability: str = OBSERVED
    subagent_id: str | None = None
    source_ref: dict = field(default_factory=dict)
    native_input: dict = field(default_factory=dict)
    status: str = ""
    timestamp: object = None


@dataclass
class NormalizedTrajectory:
    agent: str
    instance_id: str
    steps: list[NormalizedStep] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    def tool_calls(self) -> list[NormalizedStep]:
        return [s for s in self.steps if s.type == "tool_call"]

    def by_action(self, action: str) -> list[NormalizedStep]:
        return [s for s in self.steps if s.canonical_action == action]

    def assistant_texts(self) -> list[NormalizedStep]:
        return [s for s in self.steps if s.type in ("message", "final_answer") and s.actor == "agent" and s.text]

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "instance_id": self.instance_id,
            "steps": [asdict(s) for s in self.steps],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NormalizedTrajectory:
        return cls(d["agent"], d["instance_id"], [NormalizedStep(**s) for s in d.get("steps", [])], d.get("source", {}))


# ---------------------------------------------------------------- path helpers


def repo_relative(path: str | None, instance_id: str = "", agent: str = "") -> str | None:
    """Strip the '.../workdirs/<instance>/<agent>/' sandbox prefix to a
    repo-relative path so it can be matched against gold files. Also normalizes
    './'-style relative paths ('./a/b.py', 'a/./b.py') which otherwise never
    intersect the gold set."""
    if not path:
        return None
    if instance_id and agent:
        marker = f"/workdirs/{instance_id}/{agent}/"
        i = path.find(marker)
        if i >= 0:
            path = path[i + len(marker) :]
    if "/workdirs/" in path:  # generic fallback: drop <instance>/<agent>/
        tail = path.split("/workdirs/", 1)[1].split("/")
        if len(tail) >= 3:
            path = "/".join(tail[2:])
    if path.startswith("./") or "/./" in path:
        norm = posixpath.normpath(path)
        if norm not in (".", ".."):
            path = norm
    return path


def excerpt(text: str, head: int = 200, tail: int = 380) -> str:
    """Bounded excerpt that keeps the TAIL as well as the head: test runners print
    the pass/fail verdict LAST, so a head-only excerpt of verbose output hides the
    summary from the failure/clean classifiers."""
    t = text or ""
    if len(t) <= head + tail + 20:
        return t
    return t[:head] + " …[truncated]… " + t[-tail:]


# ------------------------------------------------------------- command parsing

# Multi-word invocations that mark a test run wherever they appear in a command.
_TEST_PHRASES = (
    "manage.py test",
    "django-admin test",
    "bin/test",
    "make test",
    "setup.py test",
    "hatch test",
    "hatch run test",
)
# Runners that mark a test run only as the COMMAND HEAD of a shell segment —
# substring matching false-fired on `pip install pytest`, `cat tox.ini`,
# `git commit -m "fix pytest issue"`, `grep -rn unittest .`, etc.
_TEST_RUNNER_HEADS = {"pytest", "py.test", "nosetests", "tox", "runtests.py"}
_TEST_MODULES = {"pytest", "unittest", "nose", "nose2", "tox", "test"}
_TEST_SCRIPT_RE = re.compile(r"(?:test_[\w.]*\.py|[\w-]*_test\.py|runtests\.py)$")
_ENV_ASSIGN_RE = re.compile(r"^\w+=")
_WRAPPERS = {"time", "timeout", "sudo", "xvfb-run", "nice", "env"}
_READ_CMDS = {"cat", "head", "tail", "less", "more", "sed", "bat", "nl", "view"}
_FILE_RE = re.compile(r"[\w./-]+\.(?:py|txt|cfg|ini|rst|md|toml|yaml|yml|json)$")


def _segments(cmd: str) -> list[str]:
    """Split shell operators outside quotes; quoted regex/code is an argument.

    Preserve quotes until tokenization so spaces and escaped operators stay
    inside their arguments. This recognizes simple shell lists and pipelines,
    not arbitrary shell programs; malformed quotes are rejected by _tokens.
    """
    segments, start, quote, escaped = [], 0, None, False
    for i, char in enumerate(cmd or ""):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in ";|&\n":
            if (cmd[start:i]).strip():
                segments.append(cmd[start:i])
            start = i + 1
    if (cmd or "")[start:].strip():
        segments.append(cmd[start:])
    return segments


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return []  # incomplete input is not execution evidence


def _segment_head(toks: list[str]) -> tuple[str, list[str]]:
    """The effective command word of a segment (env assignments and benign
    wrappers skipped), plus the remaining argv."""
    i = 0
    while i < len(toks):
        token = toks[i].rsplit("/", 1)[-1]
        if _ENV_ASSIGN_RE.match(toks[i]):
            i += 1
            continue
        if token not in _WRAPPERS:
            break
        i += 1
        # `timeout 600 go test` and `nice -n 5 pytest` have wrapper-specific
        # operands before the command.  Strip only their well-defined prefix.
        if token == "timeout":
            while i < len(toks) and toks[i].startswith("-"):
                i += 1
            if i < len(toks):  # duration (600, 10m, ...)
                i += 1
        elif token == "nice":
            if i < len(toks) and toks[i] in ("-n", "--adjustment"):
                i += 2
        elif token == "env":
            while i < len(toks) and (toks[i].startswith("-") or _ENV_ASSIGN_RE.match(toks[i])):
                i += 1
    if i >= len(toks):
        return "", []
    return toks[i].rsplit("/", 1)[-1].strip("'\""), toks[i + 1 :]


def is_test_command(cmd: str) -> bool:
    c = cmd or ""
    for seg in _segments(c):
        head, rest = _segment_head(_tokens(seg))
        head = head.lower()
        if not head:
            continue
        # Introspection/collection can name a runner and gating files without
        # executing any test. Do not let them close verification/repair gaps.
        if any(a.split("=", 1)[0] in _NO_EXECUTION_FLAGS for a in rest):
            continue
        if head in _TEST_RUNNER_HEADS:
            return True
        args = [a.strip("'\"").lower() for a in rest]
        if head in ("manage.py", "django-admin") and args[:1] == ["test"]:
            return True
        if head == "hatch" and (args[:1] == ["test"] or args[:2] == ["run", "test"]):
            return True
        if _tokens(seg)[:1] == ["bin/test"]:
            return True
        # Runner-specific subcommands and nonexecution modes share one home.
        from .testcommands import nonpython_test_command

        if nonpython_test_command(head, rest):
            return True
        if head in ("python", "python2", "python3") or head.startswith("python3."):
            if "-m" in rest:
                mod = rest[rest.index("-m") + 1] if rest.index("-m") + 1 < len(rest) else ""
                if mod.split(".")[0] in _TEST_MODULES:
                    return True
            for a in rest:  # first non-flag arg = the script
                if a.startswith("-"):
                    continue
                if _TEST_SCRIPT_RE.search(a.rsplit("/", 1)[-1].strip("'\"")):
                    return True
                if a.rsplit("/", 1)[-1] in ("setup.py", "manage.py"):
                    if rest[rest.index(a) + 1 :][:1] == ["test"]:
                        return True
                break
        if head in ("x.py", "run-tests.php", "runtests", "runtests.sh") and "test" in c:
            return True
    return False


def classify_command(cmd: str) -> str:
    return RUN_TEST if is_test_command(cmd) else RUN_COMMAND


_SUITE_PY_MODULES = {"pytest", "unittest", "nose", "nose2"}
_NO_EXECUTION_FLAGS = {
    "--version",
    "--help",
    "-h",
    "-V",
    "--collect-only",
    "--co",
    "--list-tests",
    "--listTests",
    "--dry-run",
}


def is_suite_wide_test_command(cmd: str) -> bool:
    """A test invocation with NO test selector — it runs whatever suite is
    locally present rather than a named subset (bare ``pytest``, ``tox``,
    ``make test``, ``python -m unittest [discover]``, ``python setup.py test``).

    Shared primitive (trajsignals + audit): grounds the v0.36 suppression of the
    medium ``relevant_test_not_run`` arm — "did not run the relevant test" is
    not established against an agent that ran everything available. Conservative
    by construction: any positional argument or ``-k``/``-m`` selector means NOT
    suite-wide (an option value like ``-n auto`` therefore also disqualifies —
    a false negative here only declines to suppress)."""
    for seg in _segments(cmd or ""):
        if not is_test_command(seg):
            continue
        head, rest = _segment_head(_tokens(seg))
        head = head.lower()
        if not head:
            continue
        if any(a.split("=", 1)[0] in _NO_EXECUTION_FLAGS for a in rest):
            continue
        if head == "tox":
            # `tox -e <env>` runs a NAMED env (possibly lint/format, no tests
            # at all) — only a bare/flag-generic tox counts (v0.37 review fix).
            if not any(a == "-e" or a.startswith("-e") or a.startswith("--env") for a in rest):
                return True
            continue
        if head in ("make", "gmake") and any(a in ("test", "tests", "check") for a in rest):
            return True
        runner_rest = None
        if head in ("pytest", "py.test", "nosetests", "nose2"):
            runner_rest = rest
        elif head in ("python", "python2", "python3") or head.startswith("python3."):
            if "-m" in rest:
                i = rest.index("-m")
                mod = (rest[i + 1] if i + 1 < len(rest) else "").split(".")[0]
                if mod in _SUITE_PY_MODULES:
                    runner_rest = rest[i + 2 :]
                    if mod == "unittest" and runner_rest[:1] == ["discover"]:
                        return True
            elif rest and rest[0].rsplit("/", 1)[-1] == "setup.py" and rest[1:2] == ["test"]:
                return True
        if runner_rest is None:
            continue
        # Flags that mean the invocation ran a SUBSET or NO tests at all —
        # each one disqualifies (v0.37 review fix: --collect-only/--co run
        # nothing, --lf/--ff re-run prior failures, --ignore/--deselect can
        # exclude the very gating file; all previously passed for suite-wide
        # and wrongly SUPPRESSED relevant_test_not_run).
        if any(
            a in ("-k", "-m", "--co", "--collect-only", "--lf", "--last-failed", "--ff", "--failed-first")
            or a.split("=", 1)[0] in ("--ignore", "--ignore-glob", "--deselect")
            for a in runner_rest
        ):
            continue
        if any(not a.startswith("-") for a in runner_rest):
            continue  # positional target (path / node id)
        return True
    return False


def command_read_targets(cmd: str) -> list[str]:
    """Best-effort file reads from a shell command (cat/sed/head/...). Used to
    recover read-set for shell-driven agents (Codex). Approximate by design.

    Only the ARGUMENTS OF THE READ COMMAND ITSELF count, and everything from the
    first redirect on is dropped: redirect/heredoc targets are WRITES, and files
    named by sibling segments (`cat a.py && python b.py`) are not reads. `sed -i`
    is an in-place EDIT, not a read. The read-set feeds the localization-vs-
    editing blame split, so contamination here flips diagnoses."""
    out = []
    for seg in _segments(cmd or ""):
        seg = re.split(r"[<>]", seg, maxsplit=1)[0]  # cut at first redirect/heredoc
        head, rest = _segment_head(_tokens(seg))
        if head not in _READ_CMDS:
            continue
        if head == "sed" and any(t == "-i" or t.startswith("-i") for t in rest):
            continue  # in-place edit, not a read
        for t in rest:
            t = t.strip("'\"")
            if t and not t.startswith("-") and "*" not in t and _FILE_RE.search(t):
                out.append(t)
    return out


_CLAIM_PATTERNS = (
    "all tests pass",
    "all tests passed",
    "tests passed",
    "tests now passed",
    "tests pass",
    "tests now pass",
    "test passes",
    "all passing",
    "fix is complete",
    "issue is resolved",
    "this resolves",
    "resolved the issue",
    "successfully fixed",
    "have fixed",
    "i've fixed",
    "the fix works",
    "bug is fixed",
    "should now work",
    "problem is solved",
    "change is complete",
    "works as expected",
)


def looks_like_success_claim(text: str) -> bool:
    # Quoted examples and negated/conditional reports are not assertions.
    t = re.sub(r'```[\s\S]*?```|`[^`]*`|"[^"\n]*"|“[^”]*”', "", text.lower())
    for clause in re.split(r"[.;!\n]|\b(?:but|however)\b", t):
        if "?" in clause:
            continue
        for phrase in _CLAIM_PATTERNS:
            for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", clause):
                prefix = clause[: match.start()]
                suffix = clause[match.end() :]
                if re.search(
                    r"\b(?:not|never|cannot|can't|couldn't|don't|doesn't|haven't|"
                    r"hasn't|unable|without|whether|if|unless|previously|earlier|"
                    r"initially|thought|believed)\b",
                    prefix,
                ):
                    continue
                if re.match(r"\s*(?:is |are |was |were )?(?:not true|false|incorrect|unverified|wrong)\b", suffix):
                    continue
                return True
    return False


# Grounded failure detection: match SPECIFIC failure markers, and first rule out
# explicit "clean" summaries so that benign phrases like "no errors" / "0 failed"
# / "all passed" are never read as failures (the old substring scan flagged any
# occurrence of the word "error").
_CLEAN_RE = re.compile(
    r"\b(?:no (?:errors?|failures?)|0 (?:failed|failures?|errors?)|failed=0|errors?=0|"
    r"all (?:tests )?passed|\d+ passed(?:,?\s*0 (?:failed|error))?)\b"
    # unittest / Django runner verdict line: a bare OK (optionally with skips)
    r"|^ok(?: \(skipped=\d+\))?\s*$",
    re.MULTILINE,
)
_FAIL_RE = re.compile(
    r"(?:\b[1-9]\d* (?:failed|errors?)\b|\bfailed=[1-9]|\berrors?=[1-9]|"
    r"\bassertion\s?error\b|traceback \(most recent call last\)|"
    r"\bfailed\b|^FAILED\b|\bE\s{2,})",
    re.IGNORECASE | re.MULTILINE,
)


# Exit-code text forms the scaffolds print instead of a structured field.
_CODEX_EXIT_RE = re.compile(r"^Process exited with code (-?\d+)\s*$", re.MULTILINE)
_CODEX_SESSION_RE = re.compile(r"^Process running with session ID (\d+)\s*$", re.MULTILINE)
_CC_EXIT_RE = re.compile(r"^Exit code (\d+)\s*$", re.MULTILINE)


def codex_result_header(output: str) -> tuple:
    """(exit_code | None, session_id | None, body) of a Codex exec_command /
    write_stdin output. The header ends at the first 'Output:' line; the body
    is the command's own text. Shared by the adapter and any re-deriver."""
    o = output or ""
    m = _CODEX_EXIT_RE.search(o)
    s = _CODEX_SESSION_RE.search(o)
    i = o.find("\nOutput:\n")
    body = o[i + len("\nOutput:\n") :] if i >= 0 else o
    return (int(m.group(1)) if m else None, int(s.group(1)) if s else None, body)


def claude_code_exit_code(output: str) -> int | None:
    """Claude Code's Bash tool prints 'Exit code N' as the first line of a
    non-zero result; None when absent (a zero exit prints nothing)."""
    m = _CC_EXIT_RE.match((output or "").lstrip())
    return int(m.group(1)) if m else None


def looks_like_test_failure(output: str) -> bool:
    """Heuristic: does a test/command result indicate failure? Used to find the
    failure events that open a self-repair opportunity. Clean summaries are
    REMOVED before scanning so their own tokens ('0 failed', 'no errors',
    '5 passed, 0 failed') can never double as failure markers — the old
    both-match guard fell through exactly there and minted phantom failures."""
    o = (output or "").lower()
    if not o:
        return False
    return bool(_FAIL_RE.search(_CLEAN_RE.sub(" ", o)))


# Mechanical tool-call failures (NOT test failures): a tool/shell call that could
# not execute — most importantly a HALLUCINATED PATH/TARGET (`No such file`), a bad
# command, malformed args, or a failed patch. Patterns are deliberately HIGH-
# PRECISION shell/tool error signatures that essentially never appear as the benign
# output of a successful call, so the text fallback can recover errors for scaffolds
# that don't set an explicit error flag (e.g. Codex) without false-positiving on file
# content. The caller must NOT run this over file-READ output (a successful read can
# legitimately contain error-like text); it is for command/edit/search results.
_TOOL_ERR_PATTERNS = (
    (
        "invalid_target",
        re.compile(
            r"no such file or directory|cannot find the (?:file|path)|"
            r"\bno such file\b|is not a directory|fatal: pathspec|"
            r"did not match any file|could not (?:open|find) (?:file|the file)|"
            r"\bfile not found\b|string not found|could not find the (?:string|text)|"
            r"no replacement was performed|old_string not found",
            re.IGNORECASE,
        ),
    ),
    (
        "command_not_found",
        re.compile(
            r"command not found|not recognized as an internal or external command|"
            r"executable file not found|no such command",
            re.IGNORECASE,
        ),
    ),
    (
        "bad_arguments",
        re.compile(
            r"unrecognized (?:argument|option)|invalid option|invalid choice|"
            r"the following arguments are required|unexpected argument|"
            r"patch does not apply|malformed patch|patch failed|invalid patch|"
            r"corrupt patch|error: while searching for",
            re.IGNORECASE,
        ),
    ),
    ("permission_denied", re.compile(r"permission denied|operation not permitted", re.IGNORECASE)),
    (
        "timeout",
        re.compile(
            r"command timed out|timed out after|deadline exceeded|"
            r"execution timed out",
            re.IGNORECASE,
        ),
    ),
    (
        "unknown_tool",
        re.compile(
            r"unknown tool|tool not found|no such tool|is not a valid tool|"
            r"unrecognized tool",
            re.IGNORECASE,
        ),
    ),
)


def classify_tool_error(output: str) -> str | None:
    """Return the tool-error SUBTYPE for a command/edit/search result, or None if no
    mechanical-error signature is present. Behavioral subtypes (not blame error
    types): invalid_target (hallucinated path / unfindable edit target),
    command_not_found, bad_arguments (incl. failed patch), permission_denied,
    timeout, unknown_tool (hallucinated tool name)."""
    text = output or ""
    for name, rx in _TOOL_ERR_PATTERNS:
        if rx.search(text):
            return name
    return None


def looks_like_tool_error(output: str) -> bool:
    """True iff the result text shows a mechanical tool/shell failure. See
    classify_tool_error for the scope/precision contract."""
    return classify_tool_error(output) is not None


# Environment / dependency setup failures: the run could not import, build, or invoke
# what it needed (missing module, unbuildable wheel, absent runner, version conflict) —
# the objective signature of the scout-discovered 'environment-failure' mode. Distinct
# from a code bug (a real test failure) and from a tool-call mistake; scanned over
# command AND test-run outputs (import/build errors surface in test runs), but never
# over file-read/search results (they echo file content). High-precision.
_ENV_ERR_RE = re.compile(
    r"modulenotfounderror|no module named|\bimporterror\b|cannot import name|"
    r"failed building wheel|subprocess-exited-with-error|could not find a version|"
    r"no matching distribution|error: command '[\w.+-]+' failed|microsoft visual c\+\+|"
    r"\bcommand not found\b|\bdistributionnotfound\b|version conflict",
    re.IGNORECASE,
)


def looks_like_environment_error(output: str) -> bool:
    """True iff the result text shows an ENVIRONMENT / dependency setup failure — the
    run was blocked by its environment, not by a code bug. High-precision for
    COMMAND/TEST output. Do NOT run it over file-READ or SEARCH results: source files
    and grep output legitimately contain these signatures ('except ImportError:',
    quoted 'No module named ...') — the caller must filter by the linked call's
    canonical action (see trajsignals.compute)."""
    return bool(_ENV_ERR_RE.search(output or ""))


def test_id_tokens(test_id: str) -> set:
    """SPECIFIC tokens that show a command NAMED a given test id (the swebench
    ``path/to/test_file.py::Class::test_fn`` form): the file path, its basename,
    the basename stem, and each ``test_*`` function name after ``::`` — all at
    the >=5-char specificity bar the gating-test token match uses, so a bare
    'test' never matches. Shared primitive (v0.39): the eval+ detector and its
    independent audit re-deriver must agree on the token vocabulary."""
    tid = (test_id or "").strip()
    if not tid:
        return set()
    if "::" in tid:
        path, _, rest = tid.partition("::")
        base = path.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        toks = {t for t in (path, base, stem) if len(t) >= 5}
        toks |= {t for t in re.findall(r"test_\w+", rest) if len(t) >= 5}
        return toks
    # Django / unittest form ``test_fn (pkg.module.Class)`` (the swebench log
    # parser's id for Django's runner): the dotted module path, its slash
    # form, the module stem and every app-label prefix name the test — Django's
    # ``runtests.py`` takes any of ``label``, ``label.module``,
    # ``label.module.Class``; pytest takes the slash path.
    m = _DJANGO_ID.match(tid)
    if m:
        fn, dotted = m.group(1), m.group(2)
        parts = dotted.split(".")
        # the trailing CamelCase segment(s) are the TestCase class, not a module
        while len(parts) > 1 and parts[-1][:1].isupper():
            parts.pop()
        toks = {t for t in (fn, dotted) if len(t) >= 5}
        for k in range(1, len(parts) + 1):
            prefix = ".".join(parts[:k])
            toks |= {t for t in (prefix, prefix.replace(".", "/")) if len(t) >= 5}
        if len(parts[-1]) >= 5:
            toks.add(parts[-1])
        return toks
    # Docstring-form ids (the first line of the test's docstring, no module
    # information) carry nothing a command could name: empty — callers treat
    # an un-nameable id as UNKNOWN, never as "not named".
    return set()


_DJANGO_ID = re.compile(r"^(\w+) \(([\w.]+)\)$")


def names_any_test(test_ids, commands) -> bool | None:
    """Did any command NAME one of these test ids? ``None`` when some id is
    un-nameable (no tokens) and none of the nameable ones was named — the
    question cannot be answered, and the caller must not blame. Shared
    primitive (v0.39/v0.40): the eval+ detector and its audit re-deriver must
    agree on both the vocabulary and the unknown case."""
    cmds = [c or "" for c in (commands or [])]
    unknown = False
    for tid in test_ids:
        toks = test_id_tokens(tid)
        if not toks:
            unknown = True
            continue
        if any(tok in cmd for tok in toks for cmd in cmds):
            return True
    return None if unknown else False


def read_observations(traj):
    """Successful, attempted, and unanswered read paths from tool linkage.

    Missing results are unknown; explicit failures cannot demonstrate a read.
    A zero exit status overrides command-output error words, which can be file
    content. READ results are never scanned for error words in source content.
    """
    from .verification import tool_observations

    successful, attempted, unanswered = set(), set(), set()
    for _, s, _, r, _ in tool_observations(traj):
        if s.canonical_action not in (READ, RUN_COMMAND):
            continue
        files = set(filter(None, s.files))
        attempted.update(files)
        if r is None:
            unanswered.update(files)
            continue
        ec = (r.args or {}).get("exit_code")
        failed = r.tool_error or (type(ec) is int and ec != 0)
        if ec is None and s.canonical_action == RUN_COMMAND:
            failed = failed or classify_tool_error(r.output_excerpt) is not None
        if not failed:
            successful.update(files)
    return successful, attempted, unanswered - successful


def observed_test_runs(traj):
    """Pairs of test calls and execution-bearing observed results, per thread.

    Collection/infrastructure failures and dangling calls do not prove that a
    test ran. Exit codes 0/1 permit passing/failing runs; other codes abstain.
    Output naming a file can establish a selector-free suite's coverage.
    """
    from .verification import tool_observations, known_zero_test_calls

    zero = known_zero_test_calls(traj)
    runs = []
    for _, call, _, r, _ in tool_observations(traj):
        if not call.test_run or id(call) in zero or r is None:
            continue
        ec = (r.args or {}).get("exit_code")
        out = (r.output_excerpt or "").lower()
        if re.search(r"error collecting|errors? during collection|no tests (?:ran|collected)", out):
            continue
        if type(ec) is int:
            if ec not in (0, 1):
                continue
        elif not out.strip() or (r.tool_error and not looks_like_test_failure(out)):
            continue
        if looks_like_environment_error(out) and ec != 0:
            continue
        runs.append((call, r))
    return runs


def edit_reversals(traj):
    """Observed successful edits that exactly reverse prior directed content.

    Full ordered text is required. Set similarity, capped legacy line maps,
    failed attempts and missing results cannot establish an applied reversal.
    Comparisons stay within an observed thread and file.
    """
    from .verification import tool_observations

    by_target, pairs = {}, []
    for _, call, _, result, _ in tool_observations(traj):
        if call.canonical_action not in (EDIT, WRITE) or result is None:
            continue
        ec = (result.args or {}).get("exit_code")
        if result.tool_error or (type(ec) is int and ec != 0):
            continue
        per = (call.args or {}).get("per_file_lines") or {}
        for path in filter(None, call.files):
            content = per.get(path, {} if per else call.args)
            old, new = content.get("old_content"), content.get("new_content")
            if not isinstance(old, str) or not isinstance(new, str) or old == new:
                continue
            history = by_target.setdefault((call.subagent_id, path), [])
            for prior, before, after in history[-1:]:
                if old == after and new == before:
                    pairs.append({"file": path, "thread": call.subagent_id, "steps": [prior, call.step_id]})
            history.append((call.step_id, old, new))
    return pairs
