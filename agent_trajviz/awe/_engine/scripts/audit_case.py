#!/usr/bin/env python3
"""Interactive evidence-chain audit for one case.

  python3 scripts/audit_case.py <agent> <instance> [--json]

Prints each diagnosed fault's evidence-chain claim and a re-derivation table
(atom | detector said | audit re-derived | OK?) plus the per-fault verdict.
Requires AWE_ARGUS_ROOT set (it re-loads the raw inputs)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import audit
from agent_trajviz.awe._engine.dossier import case_record

_BADGE = {
    audit.RE_DERIVED: "OK  re-derived",
    audit.STRUCTURAL_OK: "OK  structural",
    audit.MISMATCH: "!!  MISMATCH",
    audit.UNVERIFIABLE: "--  unverifiable",
}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    if len(args) != 2:
        print("usage: audit_case.py <agent> <instance> [--json]")
        sys.exit(2)
    agent, inst = args
    try:
        rec = case_record(agent, inst)
    except FileNotFoundError:
        rec = None
    if rec is None:
        print(f"no diagnosed failure for {agent}__{inst}")
        sys.exit(1)
    out = audit.audit_case(agent, inst, persisted=rec)
    if as_json:
        print(json.dumps(out, indent=2, default=list))
        return
    print(f"# audit {out['case_id']}   {out['verdict_counts']}\n")
    for fault, a in zip(rec["faults"], out["faults"], strict=False):
        print(f"## {fault['label']}   [{_BADGE.get(a['verdict'], a['verdict'])}]  ({a['tier']})")
        if a["checks"]:
            for c in a["checks"]:
                flag = "OK" if c["ok"] else "!!"
                print(f"   {flag}  {c['atom']}: detector={c['expected']!r}  audit={c['actual']!r}")
        else:
            print("   (no re-derivable atoms)")
        print()


if __name__ == "__main__":
    main()
