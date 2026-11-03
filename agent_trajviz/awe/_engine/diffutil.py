"""Minimal unified-diff parser. Recovers the set of touched files, the changed
line count, and best-effort def/class symbol hints — enough to anchor
localization and edit-size signals against the SWE-bench gold patch."""

from __future__ import annotations

import re
import fnmatch
from dataclasses import dataclass, field

_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@(.*)$")
_DEFCLASS_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)")


@dataclass
class FileDiff:
    path: str
    added: int = 0
    removed: int = 0
    symbols: set[str] = field(default_factory=set)
    is_new: bool = False
    is_deleted: bool = False
    # Raw changed-line CONTENT (un-normalized). evidence.changed_lines builds its
    # per-file normalized multisets from these, so there is exactly ONE hardened
    # diff parser (the old second scanner in evidence.py mis-handled '-- '/'++ '
    # body lines the way this parser was specifically hardened against).
    added_lines: list = field(default_factory=list)
    removed_lines: list = field(default_factory=list)
    # (old_start, old_count) per hunk, in OLD-file coordinates. Both the gold and
    # the agent patch diff against the same base commit, so old-coordinate ranges
    # are directly comparable — the basis for the region_overlap severity
    # attribute ("edited the same file but a disjoint region").
    hunks: list = field(default_factory=list)
    # Count-bounded context for conservative AST review of test-oracle edits.
    hunk_contents: list = field(default_factory=list)

    unbounded_lines: int = 0
    invalid_hunks: int = 0

    @property
    def loc(self) -> int:
        return self.added + self.removed


def _strip_prefix(p: str, prefix: str) -> str:
    return p[2:] if p.startswith(prefix) else p


def parse_unified_diff(text: str) -> list[FileDiff]:
    """Parse git-format AND plain (headerless) unified diffs.

    Two robustness properties the naive line scan lacked:
    * Hunk line counts from ``@@ -a,b +c,d @@`` bound body consumption, so a
      removed line whose content starts with ``-- `` (SQL/Lua comments) or an
      added ``++ …`` line can never be mistaken for a file header.
    * A ``--- old / +++ new`` pair outside a git section opens a file, so plain
      unified diffs (the format a new benchmark's scaffold may emit) parse to
      their files instead of silently yielding [] (which downstream read as
      "no patch produced").
    """
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    old_path: str | None = None
    expect_headers = False  # between 'diff --git' and its first hunk
    pending_old: str | None = None  # '---' seen with no git section open
    pending_seen = pending_new_file = False
    hunk = None
    rem_old = rem_new = 0  # unconsumed hunk-body line budget

    lines = text.splitlines()
    i = -1
    while i + 1 < len(lines):
        i += 1
        line = lines[i]
        if rem_old > 0 or rem_new > 0:  # inside a hunk body
            # A hunk header may OVERSTATE its counts (hand/LLM-authored plain
            # diffs). Without an escape, the budget would consume the next
            # file's headers as context and whole files would silently vanish.
            # Real file starts terminate a hunk early: 'diff --git' always (a
            # body line cannot legitimately match it), and a strict
            # '--- a/|/dev/null' + '+++ b/|/dev/null' header PAIR (a lone
            # '--- '/'+++ ' body line — removed SQL comment, added '++ ' — is
            # still consumed as body, which the budget exists to protect).
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _GIT_RE.match(line) or (
                re.match(r"^--- (?:a/|/dev/null)", line) and re.match(r"^\+\+\+ (?:b/|/dev/null)", nxt)
            ):
                rem_old = rem_new = 0
                i -= 1  # reprocess as a header line
                continue
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            if hunk is not None:
                prefix = line[:1]
                hunk["valid"] &= prefix in ("+", "-", " ")
                if prefix in ("-", " "):
                    hunk["old"].append(line[1:])
                if prefix in ("+", " "):
                    hunk["new"].append(line[1:])
            if line.startswith("+"):
                cur.added += 1
                rem_new -= 1
                cur.added_lines.append(line[1:])
                cur.symbols.update(_DEFCLASS_RE.findall(line[1:]))
            elif line.startswith("-"):
                cur.removed += 1
                rem_old -= 1
                cur.removed_lines.append(line[1:])
                cur.symbols.update(_DEFCLASS_RE.findall(line[1:]))
            else:  # context line
                rem_old -= 1
                rem_new -= 1
            if hunk is not None:
                hunk["complete"] = rem_old == rem_new == 0 and hunk["valid"]
            continue

        m = _GIT_RE.match(line)
        if m:
            cur = FileDiff(path=m.group(2))
            old_path = m.group(1)
            files.append(cur)
            expect_headers = True
            pending_seen = False
            continue
        if line.startswith("--- ") and not line.startswith("----"):
            p = line[4:].strip().split("\t")[0]
            if cur is not None and expect_headers:
                if p == "/dev/null":
                    cur.is_new = True
                else:
                    old_path = _strip_prefix(p, "a/")
            else:  # plain diff: opens a file section
                pending_seen = True
                pending_new_file = p == "/dev/null"
                pending_old = None if pending_new_file else _strip_prefix(p, "a/")
            continue
        if line.startswith("+++ ") and not line.startswith("++++"):
            p = line[4:].strip().split("\t")[0]
            if cur is not None and expect_headers:
                if p == "/dev/null":
                    cur.is_deleted = True
                    if old_path:
                        cur.path = old_path
                else:
                    cur.path = _strip_prefix(p, "b/")
                expect_headers = False  # header block done; hunk-less +/- may follow
            elif pending_seen:
                if p == "/dev/null":
                    cur = FileDiff(path=pending_old or "", is_deleted=True)
                else:
                    cur = FileDiff(path=_strip_prefix(p, "b/"), is_new=pending_new_file)
                files.append(cur)
                pending_seen = False
            continue
        if line.startswith("@@") and cur is not None:
            mh = _HUNK_RE.match(line)
            if mh:
                old_start = int(mh.group(1))
                rem_old = int(mh.group(2)) if mh.group(2) is not None else 1
                rem_new = int(mh.group(3)) if mh.group(3) is not None else 1
                cur.hunks.append((old_start, rem_old))
                hunk = {"old_start": old_start, "old": [], "new": [], "complete": False, "valid": True}
                cur.hunk_contents.append(hunk)
                cur.symbols.update(_DEFCLASS_RE.findall(mh.group(4)))
                expect_headers = False
            else:
                cur.invalid_hunks += 1
            continue
        # HUNK-LESS fallback: a +/- body line with no declared @@ hunk (sloppy
        # hand/LLM-authored diffs). Counted naively — the header-lookalike
        # hardening only applies inside declared hunks, which is the best a
        # hunk-less diff permits. Header checks above still win, and real
        # header-block lines (index/mode/---/+++) never start with a bare +/-,
        # so this is safe even directly after a `diff --git` line.
        if cur is not None:
            if line.startswith(("+", "-")):
                cur.unbounded_lines += 1
                if cur.hunks:
                    continue  # never add ignored trailing payload to valid hunks
            if line.startswith("+") and not line.startswith("+++"):
                cur.added += 1
                cur.added_lines.append(line[1:])
                cur.symbols.update(_DEFCLASS_RE.findall(line[1:]))
            elif line.startswith("-") and not line.startswith("---"):
                cur.removed += 1
                cur.removed_lines.append(line[1:])
                cur.symbols.update(_DEFCLASS_RE.findall(line[1:]))
    return files


def changed_files(text: str) -> set[str]:
    return {fd.path for fd in parse_unified_diff(text)}


def total_loc(text: str) -> int:
    return sum(fd.loc for fd in parse_unified_diff(text))


def is_test_file(path: str, profile=None) -> bool:
    """Heuristic: is this a test file rather than source under change?
    Deliberately strict to avoid false positives like 'latest.py'. A NESTED
    'test' segment is NOT enough on its own: django ships its testing
    FRAMEWORK as the source package `django/test/` (editing
    django/test/testcases.py is a source edit, not a test edit). ROOT-level
    `test/` and `testing/` directories DO count — those are test-tree layout
    conventions (CPython-style test/, pytest-repo testing/); nested
    '/testing/' stays source (numpy/testing, pandas/testing are shipped APIs)."""
    if profile is None:
        from .benchmark import resolve_profile

        profile = resolve_profile()
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in profile.get("test_file_exclude", [])):
        return False
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in profile.get("test_file_include", [])):
        return True
    p = path.lower()
    fname = p.rsplit("/", 1)[-1]
    test_name = (
        fname in ("tests.py", "conftest.py")
        or fname.startswith("test_")
        or fname.endswith("_test.py")
        # JS / TS / Go conventions (SWE-bench Pro repositories,
        # ADDING_A_BENCHMARK.md §4.2): foo.test.ts, foo.spec.js, foo_test.go
        or fname.endswith(
            (
                ".test.js",
                ".test.ts",
                ".test.jsx",
                ".test.tsx",
                ".spec.js",
                ".spec.ts",
                ".spec.jsx",
                ".spec.tsx",
                "_test.go",
            )
        )
    )
    return (
        test_name or "/tests/" in p or "/__tests__/" in p or p.startswith(("tests/", "test/", "testing/", "__tests__/"))
    )


def gold_added_coverage(gold, patch) -> float:
    """Fraction of the gold patch's added SOURCE lines (normalized content)
    present among the agent patch's added lines. THE BA-HARNESS-1 metric —
    added lines only, deliberately not reproduced_fraction (which also counts
    matched REMOVED lines, so a deletion-dominated patch that deletes the right
    code but writes the wrong replacement scores near-gold). Single home for
    both consumers: agent_trajviz.awe._engine.faultside and scripts/harness_suspects.py."""
    g = set()
    for f, ch in (gold.gold_changed_lines or {}).items():
        if not is_test_file(f, gold.benchmark_profile or None):
            g |= set(ch.get("added") or [])
    if not g:
        return 0.0
    a = set()
    for ch in (patch.changed_lines or {}).values():
        a |= set(ch.get("added") or [])
    return len(g & a) / len(g)
