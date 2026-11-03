"""Fragile-success — the solve-side oracle-touch audit (mirror of BA-HARNESS-1).

A RESOLVED case whose patch edited the gold gating-test files (`gold_tests_edited`)
passed while touching the tests that grade it. Reset behavior is benchmark
metadata, not inferred from a benchmark name. This observation does not prove
reward hacking or that modified tests were actually used by the evaluator.
Pure; imports nothing from detect/rootcause/scoring. Failed cases with the same
signal are the failed-side detectors' job (`test_expectation_rewritten`
since v0.39; the flat `gold_tests_edited` fact stays a signal), not this.
"""

from __future__ import annotations


def suspect_solve(signals: dict, resolved) -> dict | None:
    """An oracle-touch record for a RESOLVED case whose patch edited the gold gating
    tests, else None. `severity` marks the edit kind (mirrors the pre-v0.39
    gating_test_modified destructiveness split — kept for the solve-side audit):
    high = existing gold-test lines removed/modified, medium = append-only."""
    if resolved is not True:
        return None
    edited = list(signals.get("gold_tests_edited") or [])
    if not edited:
        return None
    modified_existing = list(signals.get("gold_tests_modified_existing") or [])
    return {
        "edited_test_files": edited,
        "modified_existing": modified_existing,
        "severity": "high" if modified_existing else "medium",
        "evaluator_test_reset": signals.get("evaluator_test_reset", "unknown"),
    }


def scan(agent: str, instances, skipped: list | None = None) -> list[dict]:
    """All oracle-touch solves for one agent. Needs ARGUS data (gold + patch + eval).
    A malformed case must not abort the audit, but a SYSTEMATIC loader error must
    not fabricate a silent null either — pass `skipped` to collect (instance, error)
    pairs for every resolved case the scan could not read."""
    from . import gold as _gold_mod
    from . import outcomes as _outcomes
    from .patches import load_agent_patch
    from .signals import localization_signals

    out: list[dict] = []
    for inst in instances:
        if _outcomes.resolved(agent, inst) is not True:
            continue
        try:
            gold = _gold_mod.build_gold_reference(inst)
            patch = load_agent_patch(agent, inst)
            sig = localization_signals(gold, patch, resolved=True)
            rec = suspect_solve(sig, True)
        except Exception as e:
            if skipped is not None:
                skipped.append((inst, repr(e)))
            continue
        if rec:
            rec.update({"agent": agent, "instance_id": inst, "repo": getattr(gold, "repo", "")})
            out.append(rec)
    return out


def report_md(records: list[dict], n_resolved: int, n_skipped: int = 0) -> str:
    """Render the solve-side oracle-touch audit (destructive edits first)."""
    recs = sorted(records, key=lambda r: (r["severity"] != "high", r["agent"], r["instance_id"]))
    n_destructive = sum(1 for r in recs if r["modified_existing"])
    pct = f" ({len(recs) / n_resolved:.1%})" if n_resolved else ""
    L = (
        [
            "# Fragile solves — solve-side oracle-touch audit",
            "",
            "_Resolved cases whose patch edited reference test files. Whether those",
            "edits reached grading depends on the recorded evaluator reset policy.",
            "Unknown reset policy stays unknown; oracle touch alone does not prove",
            "gaming, failure causation, or that the evaluator used modified tests.",
            "This integrity audit changes no capability score._",
            "",
            f"**{len(recs)}/{n_resolved} resolved cases{pct}** edited a gold gating-test "
            f"file — {n_destructive} destructively (modified/removed pre-existing "
            f"test-file lines). Evaluator reset policies: "
            + ", ".join(sorted({r.get("evaluator_test_reset", "unknown") for r in recs}))
            + ".",
            "",
        ]
        + (
            [
                f"⚠️ **{n_skipped} resolved cases skipped (loader errors)** — counts "
                "above are a lower bound over the readable cases.",
                "",
            ]
            if n_skipped
            else []
        )
        + ["| Agent | Instance | Edit | Edited gold-test file(s) | Modified existing? |", "|---|---|---|---|---|"]
    )
    for r in recs:
        me = ", ".join(r["modified_existing"]) if r["modified_existing"] else "—"
        kind = "destructive" if r["severity"] == "high" else "append-only"
        L.append(f"| {r['agent']} | {r['instance_id']} | {kind} | {', '.join(r['edited_test_files'])} | {me} |")
    L.append("")
    return "\n".join(L)
