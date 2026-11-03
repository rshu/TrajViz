"""Load an agent's final patch (the authoritative edit-set).

Reading the final diff sidesteps per-scaffold trajectory differences entirely
— in particular it recovers Codex's edits without parsing shell commands."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .diffutil import parse_unified_diff, is_test_file
from .evidence import changed_lines as _changed_lines


@dataclass
class AgentPatch:
    agent: str
    instance_id: str
    exists: bool
    edited_files: list[str] = field(default_factory=list)  # all touched files
    edited_source_files: list[str] = field(default_factory=list)  # tests excluded
    edited_test_files: list[str] = field(default_factory=list)
    # Non-empty patch text that parsed to ZERO files — a format failure (a diff
    # dialect the parser missed), which must be distinguishable from "agent
    # emitted nothing".
    unparseable: bool = False
    patch_loc: int = 0
    symbols: list[str] = field(default_factory=list)
    # {file: {"added": set, "removed": set}} of changed line CONTENT — lets the
    # diagnosis check whether an edited gold file's change matches the reference,
    # rather than only whether the file was touched.
    changed_lines: dict = field(default_factory=dict)
    # {file: [(old_start, old_count), ...]} hunk ranges (old-file coordinates).
    hunks: dict = field(default_factory=dict)
    # {file: changed-LOC} — grounds the absolute over-edit metric (v0.37):
    # the extra-source-LOC sum needs per-file mass, not just the total.
    file_loc: dict = field(default_factory=dict)


def load_agent_patch(agent: str, instance_id: str) -> AgentPatch:
    path = config.patch_path(agent, instance_id)
    if not path.exists():
        return AgentPatch(agent=agent, instance_id=instance_id, exists=False)
    text = path.read_text(errors="replace", encoding="utf-8")
    return patch_from_text(agent, instance_id, text)


def patch_from_text(agent: str, instance_id: str, text: str, profile=None) -> AgentPatch:
    """Parse an observed final diff, including an intentionally empty patch."""
    diffs = parse_unified_diff(text)
    edited = [fd.path for fd in diffs]
    src = [p for p in edited if not is_test_file(p, profile)]
    tst = [p for p in edited if is_test_file(p, profile)]
    return AgentPatch(
        agent=agent,
        instance_id=instance_id,
        exists=True,
        edited_files=edited,
        edited_source_files=src,
        edited_test_files=tst,
        unparseable=bool(text.strip())
        and (
            not diffs
            or any(
                fd.invalid_hunks
                or (fd.hunks and (fd.unbounded_lines or any(not h["complete"] for h in fd.hunk_contents)))
                for fd in diffs
            )
        ),
        patch_loc=sum(fd.loc for fd in diffs),
        symbols=sorted({s for fd in diffs for s in fd.symbols}),
        changed_lines=_changed_lines(text),
        hunks={fd.path: list(fd.hunks) for fd in diffs if fd.hunks},
        # summed, not last-wins: a concatenated patch can carry several diff
        # sections for one file, and the audit/calibration both sum all
        # sections (v0.37 review fix — a dict comprehension kept only the last)
        file_loc=_summed_loc(diffs),
    )


def _summed_loc(diffs) -> dict:
    out: dict = {}
    for fd in diffs:
        out[fd.path] = out.get(fd.path, 0) + fd.loc
    return out
