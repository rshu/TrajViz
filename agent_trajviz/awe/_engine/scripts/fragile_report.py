"""Generate the solve-side oracle-touch audit — the mirror of BA-HARNESS-1.

A RESOLVED case whose patch edited the gold gating-test files (`gold_tests_edited`)
passed while touching the tests that grade it. The harness resets test-patch files
to base before applying the gold test patch at eval (verified: swebench 4.1.0
`make_eval_script_list_py`), so these still passed the REAL tests: the output
quantifies oracle-touching and confirms grading integrity — not gaming. It
changes no score or count.

Usage:  python3 scripts/fragile_report.py [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config, fragile
from agent_trajviz.awe._engine.outcomes import resolved

OUT = config.PROJECT_DATA / "discovery" / "fragile_solves.md"


def collect() -> tuple[list[dict], int, list]:
    records, n_resolved, skipped = [], 0, []
    insts = config.list_instances()
    for agent in config.AGENTS:
        n_resolved += sum(1 for i in insts if resolved(agent, i) is True)
        records.extend(fragile.scan(agent, insts, skipped=skipped))
    return records, n_resolved, skipped


def audit_records(records: list[dict]) -> dict:
    """Independently re-derive every oracle-touch record from the raw agent
    diff + requirements test_patch (audit.audit_fragile_solve — the §0ag
    audit-re-derivation deferral, v0.37). Returns verdict counts; mismatches
    are printed so a drifted record can never ship silently."""
    from agent_trajviz.awe._engine.audit import audit_fragile_solve

    counts: dict = {}
    for rec in records:
        agent, inst = rec["agent"], rec["instance_id"]
        p = config.patch_path(agent, inst)
        agent_text = p.read_text(errors="replace", encoding="utf-8") if p.exists() else None
        try:
            import json

            req = json.loads(config.requirements_path(inst).read_text(encoding="utf-8"))
            from agent_trajviz.awe._engine.diffutil import parse_unified_diff

            test_files = sorted({fd.path for fd in parse_unified_diff(req.get("test_patch", "") or "")})
        except Exception:
            test_files = None
        a = audit_fragile_solve(rec, agent_text, test_files, resolved(agent, inst))
        counts[a["verdict"]] = counts.get(a["verdict"], 0) + 1
        if a["verdict"] == "mismatch":
            print(f"AUDIT MISMATCH {agent}/{inst}: {a['mismatches']}", file=sys.stderr)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help=f"write to {OUT} instead of stdout")
    args = ap.parse_args()
    records, n_resolved, skipped = collect()
    if skipped:
        print(f"warning: {len(skipped)} resolved cases skipped (loader errors), e.g. {skipped[0]}", file=sys.stderr)
    counts = audit_records(records)
    print(f"audit re-derivation: {counts}", file=sys.stderr)
    md = fragile.report_md(records, n_resolved, n_skipped=len(skipped))
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(md, encoding="utf-8")
        print(f"wrote {OUT} ({len(records)}/{n_resolved} resolved edited gold tests)")
    else:
        print(md)


if __name__ == "__main__":
    main()
