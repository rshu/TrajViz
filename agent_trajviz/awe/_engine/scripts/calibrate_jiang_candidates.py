"""Batch 7 cohort calibration: the Jiang et al. (arXiv:2605.12270) candidates.

Computes solved-vs-failed risk ratios for the three computable proxies of that
study's failure categories that this project did not already cover, plus the
BA-HARNESS-1 suspect counts. Verdicts recorded in DESIGN.md §0ab (both fault
candidates REJECTED; the harness-suspect signature grounds the benchmark-artifact
annex). Re-run to reproduce the table:

    python3 scripts/calibrate_jiang_candidates.py

Candidates:
  hint_echo            — speculative-hint sentence in the issue whose hint-only
                         code tokens appear in the agent's added lines but not in
                         gold's (Jiang P2 "Distracted by Hints" proxy).
  traj_edit_not_landed — source file edited via EDIT/WRITE steps absent from the
                         final diff (Jiang I1 family).
  shell_noop_edit      — sed -i / perl -i target never appears in the final diff
                         (Jiang I1 exact mechanism: silent no-op edit).
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.cohort import binary_association, contrast
from agent_trajviz.awe._engine.diffutil import is_test_file
from agent_trajviz.awe._engine.evidence import issue_code_tokens
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.patches import load_agent_patch
from agent_trajviz.awe._engine.significance import holm
from agent_trajviz.awe._engine.trajectory import RUN_COMMAND, RUN_TEST
from agent_trajviz.awe._engine.trajsignals import signals_cached

HINT_RE = re.compile(
    r"\b(?:workaround|possible (?:fix|solution)|potential(?:ly)? (?:fix|solution|workaround)?|"
    r"maybe (?:we|you|it|the)|perhaps|i (?:think|guess|suppose)|"
    r"could be (?:fixed|solved|to)|would be to|should (?:probably|just)|"
    r"quick fix|simplest (?:fix|solution)|one (?:fix|solution|option))\b",
    re.IGNORECASE,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")
INPLACE_RE = re.compile(r"\b(?:sed\s+(?:-[a-zA-Z]*i[a-zA-Z]*\b|--in-place)|perl\s+-[a-zA-Z]*i)")
PATHISH = re.compile(r"(?<!-)\b([\w./-]+\.(?:py|pyx|c|h|cfg|ini|rst|txt|toml|yaml|yml))\b")

CANDS = ("hint_echo", "traj_edit_not_landed", "shell_noop_edit")


def hint_tokens_of(statement: str) -> set:
    """Code tokens appearing ONLY in speculative-hint sentences of the issue."""
    sents = SENT_SPLIT.split(statement or "")
    hint_s = [s for s in sents if HINT_RE.search(s)]
    if not hint_s:
        return set()
    other = "\n".join(s for s in sents if not HINT_RE.search(s))
    return issue_code_tokens("\n".join(hint_s)) - issue_code_tokens(other)


def added_text(changed_lines: dict) -> str:
    out = []
    for ch in (changed_lines or {}).values():
        out.extend(ch.get("added") or [])
    return "\n".join(out)


def case_signals(agent: str, inst: str) -> dict | None:
    res = resolved(agent, inst)
    if res is None:
        return None
    gold = build_gold_reference(inst)
    patch = load_agent_patch(agent, inst)
    sig = signals_cached(agent, inst, gold)
    traj = load_normalized(agent, inst)
    patch_files = set(patch.edited_files or [])

    htoks = hint_tokens_of(gold.problem_statement)
    hint_echo = 0
    if htoks and patch.changed_lines:
        a_add = added_text(patch.changed_lines)
        g_add = added_text(gold.gold_changed_lines)
        hint_echo = 1 if any(t in a_add and t not in g_add for t in htoks) else 0

    traj_src = {f for f in (sig.edit_set_traj or []) if f and not is_test_file(f)}
    traj_edit_not_landed = 1 if any(f not in patch_files for f in traj_src) else 0

    targets: set = set()
    if traj is not None:
        for s in traj.steps:
            if s.canonical_action in (RUN_COMMAND, RUN_TEST):
                cmd = (s.args or {}).get("command") or ""
                if INPLACE_RE.search(cmd):
                    targets.update(PATHISH.findall(cmd))
    norm = {t.lstrip("./") for t in targets}
    shell_noop = (
        1 if norm and not any(t in patch_files or any(pf.endswith("/" + t) for pf in patch_files) for t in norm) else 0
    )

    return {
        "agent": agent,
        "inst": inst,
        "failed": 0 if res else 1,
        "hint_echo": hint_echo,
        "traj_edit_not_landed": traj_edit_not_landed,
        "shell_noop_edit": shell_noop,
    }


def main() -> None:
    rows = []
    for agent in config.AGENTS:
        for inst in config.list_instances():
            r = case_signals(agent, inst)
            if r is not None:
                rows.append(r)
    n_failed = sum(r["failed"] for r in rows)
    print(f"cases={len(rows)} failed={n_failed}")

    rng = random.Random(29)
    stats, pvals = {}, []
    for c in CANDS:
        fv = [r[c] for r in rows if r["failed"]]
        sv = [r[c] for r in rows if not r["failed"]]
        assoc, ct = binary_association(fv, sv), contrast(fv, sv, rng)
        stats[c] = (assoc, ct)
        pvals.append(ct["p"])
    for c, p_h in zip(CANDS, holm(pvals), strict=False):
        a, ct = stats[c]
        rr = a["risk_ratio"]
        rr_s = "inf" if rr == float("inf") else ("NA" if rr is None else f"{rr:.2f}")
        print(f"{c:22s} n_present={a['n_present']:4d}  RR={rr_s:>5s}  p={ct['p']:.4f}  p_holm={p_h:.4f}")


if __name__ == "__main__":
    main()
