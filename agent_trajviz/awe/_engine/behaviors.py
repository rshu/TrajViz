"""Objective detectors for scout-DISCOVERED failure modes (the IG idea-2 follow-up).

The scout (`agent_trajviz.awe._engine.scout`) surfaced recurring modes the fixed 7-capability taxonomy does
not name; the dominant ones were **analysis-paralysis** (the agent investigates but
never commits to a patch) and **environment-failure** (setup/build/import errors block
the run). This module turns those LLM HYPOTHESES into OBJECTIVE behavioral detectors
over the trajectory signals, so the cohort-contrast layer can test whether they are
actually associated with failure — closing the discover → detect → validate loop.

Honesty: these are BEHAVIORAL / associational signals (a pattern in the run), NOT blame
faults and NOT (yet) taxonomy error types. Promoting one to a diagnosed fault with a
countermeasure is a separate, human-reviewed step (and would need its own grounding +
catalog entry). Pure over a TrajectorySignals; imports nothing from detect/rootcause.
"""

from __future__ import annotations

# "Investigated" = the run did real work before (not) editing. A run that read several
# files or executed a gating test clearly engaged with the problem; one that did almost
# nothing is "gave up early", a different mode.
_MIN_READS_FOR_INVESTIGATION = 3


def analysis_paralysis(sig, produced_source_patch: bool) -> bool:
    """Investigated but never produced a fix: the run made NO source patch despite real
    investigation (ran a gating test, or read >= 3 files). The objective signature of
    'analysis-paralysis' — distinct from giving up early (little activity) and from
    editing-then-failing (a patch exists).

    ``produced_source_patch`` MUST come from the authoritative final patch
    (agent_trajviz.awe._engine.patches), not the trajectory edit-set: some scaffolds (Codex apply_patch,
    opencode_gpt) don't reliably surface edit actions in the trajectory, so
    sig.edit_set_traj would false-fire on nearly every run."""
    investigated = sig.n_test_runs >= 1 or len(sig.read_set) >= _MIN_READS_FOR_INVESTIGATION
    return investigated and not produced_source_patch


def environment_error(sig) -> bool:
    """The run hit an environment / dependency setup failure (missing module,
    unbuildable wheel, absent runner, version conflict) — the 'environment-failure'
    mode. Reads the env-error scan computed in trajsignals."""
    return bool(getattr(sig, "environment_error", False))


def behavior_flags(sig, produced_source_patch: bool) -> dict:
    """All discovered-mode behavioral flags for one run (mode -> bool)."""
    return {
        "analysis_paralysis": analysis_paralysis(sig, produced_source_patch),
        "environment_error": environment_error(sig),
    }


def _produced_source_patch(agent: str, inst: str) -> bool:
    """The authoritative 'did the agent emit a source fix' signal — from the final
    patch (agent_trajviz.awe._engine.patches), NOT the trajectory edit-set."""
    from .patches import load_agent_patch

    p = load_agent_patch(agent, inst)
    return bool(p and p.exists and p.edited_source_files)


MODES = ("analysis_paralysis", "environment_error")
MODE_DESC = {
    "analysis_paralysis": "investigated (read files / ran tests) but never produced a patch",
    "environment_error": "encountered an environment / dependency error (often recovered)",
}


def behaviors_summary(agent: str, instances) -> dict:
    """Per-agent prevalence of each discovered mode + the failure rate GIVEN the mode
    fired — the objective validation of the scout's discovery. Needs ARGUS data."""
    from .gold import build_gold_reference
    from .outcomes import resolved as _resolved
    from .trajsignals import signals_cached

    rows = []
    for inst in instances:
        r = _resolved(agent, inst)
        if r is None:
            continue
        try:
            gold = build_gold_reference(inst)
        except Exception:
            continue
        sig = signals_cached(agent, inst, gold)
        if sig is None:
            continue
        rows.append((r, behavior_flags(sig, _produced_source_patch(agent, inst))))
    out = {"agent": agent, "n": len(rows)}
    for mode in MODES:
        fired = [r for r, f in rows if f[mode]]
        n_failed = sum(1 for r in fired if r is False)
        out[mode] = {
            "n": len(fired),
            "rate": round(len(fired) / len(rows), 4) if rows else 0.0,
            "fail_given_mode": round(n_failed / len(fired), 4) if fired else None,
        }
    return out


def behaviors_report_md(summaries: list) -> str:
    """Render per-agent discovered-mode prevalence as markdown."""
    L = [
        "# Discovered-mode behaviors (objective detectors)",
        "_Objective/behavioral detectors for the modes the scout surfaced "
        "(`awe/scout.py`) — **associational, not blame faults**. `rate` = share of an "
        "agent's runs exhibiting the mode; `fail%|mode` = failure rate *given* the mode "
        "fired (the objective check that the discovered mode tracks failure). The "
        "cohort-contrast report tests significance._",
        "",
    ]
    for mode in MODES:
        L += [
            f"## `{mode}` — {MODE_DESC[mode]}",
            "",
            "| Agent | runs | fired | rate | fail% \\| mode |",
            "|---|---|---|---|---|",
        ]
        for s in summaries:
            m = s.get(mode, {})
            fg = m.get("fail_given_mode")
            fg_s = f"{100 * fg:.0f}%" if fg is not None else "—"
            L.append(f"| {s['agent']} | {s['n']} | {m.get('n', 0)} | {100 * m.get('rate', 0):.1f}% | {fg_s} |")
        L.append("")
    return "\n".join(L)
