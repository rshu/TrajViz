#!/usr/bin/env python3
"""Run the Track-C validation harness and print a reliability report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import validate
from agent_trajviz.awe._engine.validate import (
    false_positive_control,
    patch_mutation,
    synthetic_trajectory,
    judge_parse,
    judge_blame_upstream,
    arbiter_wiring,
    countermeasure_catalog,
)


def pct(num, den):
    return f"{100 * num / den:.0f}%" if den else "-"


def main() -> None:
    print("\n=== 1. False-positive control (real resolved cases) ===")
    fp = false_positive_control()
    print(f"  resolved cases checked : {fp['n_resolved']}")
    print(
        f"  failure-errors emitted : {fp['failure_errors_emitted']}  "
        f"(false-positive rate {100 * fp['false_positive_rate']:.2f}%) — expect 0"
    )
    print(f"  over_editing on success: {fp['over_editing_on_success']} (quality flag, not a false positive)")

    print("\n=== 2. Patch-mutation recall (planted defect -> expected error) ===")
    pm = patch_mutation(limit=150)
    print(f"  {'mutation':<18}{'n':>5}{'recall':>9}{'primary_cap':>13}")
    print("  " + "-" * 43)
    for mut, s in pm["per_mutation"].items():
        print(f"  {mut:<18}{s['n']:>5}{pct(s['recall'], s['n']):>9}{pct(s['cap_ok'], s['n']):>13}")
    print("\n  primary-capability confusion (planted -> detected top capability):")
    for mut, conf in pm["cap_confusion"].items():
        print(f"    {mut:<18} {conf}")

    print("\n=== 3. Synthetic-trajectory verification rules ===")
    st = synthetic_trajectory(limit=80)
    print(f"  {'case':<24}{'n':>5}{'correct':>9}")
    print("  " + "-" * 38)
    for name, s in st.items():
        print(f"  {name:<24}{s['n']:>5}{pct(s['ok'], s['n']):>9}")

    print("\n=== 4. LLM-judge wiring (deterministic; no model call) ===")
    jp = judge_parse()
    print(f"  parse/gate checks      : {jp['passed']}/{jp['n']} passed")
    for name, ok in jp["checks"].items():
        print(f"      {'ok ' if ok else 'FAIL'} {name}")
    jb = judge_blame_upstream()
    print(f"  upstream-blame checks  : {jb['passed']}/{jb['n']} passed")
    for name, ok in jb["checks"].items():
        print(f"      {'ok ' if ok else 'FAIL'} {name}")
    print(
        "\n  (prompt-level judge quality is validated separately against the human-cal "
        "set + planted misreads with a REAL model — see LLM_JUDGE.md.)"
    )

    print("\n=== 5. LLM-arbiter wiring (deterministic; no model call) ===")
    aw = arbiter_wiring()
    print(f"  parse/gate + blame-integration : {aw['passed']}/{aw['n']} passed")
    for name, ok in aw["checks"].items():
        print(f"      {'ok ' if ok else 'FAIL'} {name}")
    print(
        "\n  (the arbiter only adjudicates NON-deductive primaries; deductive loc/edit facts "
        "are never sent. Prompt-level quality needs a REAL model run — see DESIGN.md.)"
    )

    print("\n=== 6. Countermeasure catalog (schema / grounding / coverage / honesty) ===")
    cc = countermeasure_catalog()
    print(f"  checks passed : {cc['passed']}/{cc['n']}")
    for name, ok in cc["checks"].items():
        print(f"      {'ok ' if ok else 'FAIL'} {name}")
    if cc["uncovered"]:
        print(f"  UNCOVERED error types (no countermeasure): {cc['uncovered']}")

    ec = validate.evidence_chain_audit()
    print("\n=== 7. Evidence-chain audit (re-derivation vs. dossier) ===")
    print(f"  audit counts : {ec['counts']}")
    print(f"  result       : {'PASS' if ec['checks']['no_mismatches'] else 'FAIL (mismatches!)'}")
    # Printed failures must fail the command; absent corpus controls are not a
    # successful validation of a publication either.
    passed = (
        fp["n_resolved"] > 0
        and fp["failure_errors_emitted"] == 0
        and bool(pm["per_mutation"])
        and all(s["n"] > 0 and s["recall"] == s["n"] and s["cap_ok"] == s["n"] for s in pm["per_mutation"].values())
        and bool(st)
        and all(s["n"] > 0 and s["ok"] == s["n"] for s in st.values())
        and all(r["passed"] == r["n"] for r in (jp, jb, aw, cc))
        and ec["n"] > 0
        and ec["checks"]["no_mismatches"]
    )
    if not passed:
        raise SystemExit("Validation failed or required corpus evidence was unavailable.")


if __name__ == "__main__":
    main()
