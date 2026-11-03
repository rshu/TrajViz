"""Presentation summaries from measured validation results, without fallback scores."""


def countermeasure_catalog_payload():
    """Reuse the same evidence/readiness fields as case and RSI exports."""
    from .countermeasures import load_catalog
    from .remedies import entry_payload

    return [
        {
            **entry_payload(e),
            "capability": e.capability,
            "error_type": e.error_type,
            "empirical_support": e.empirical_support,
        }
        for e in load_catalog()
    ]


def arbiter_refutation_target(case):
    """Select an actual applied demotion, rather than any raw refutation."""
    applied = case.get("arbiter") or {}
    target = applied.get("target") or {}
    if (
        applied.get("support") == "refute"
        and applied.get("applied") == "demoted_to_conjunctive"
        and target.get("capability")
        and target.get("error_type")
    ):
        return target
    return None


def validation_summary(fp, pm, st, jp, jb, aw, cat, ec, n_tests):
    def ratio(hits, n):
        return f"{hits}/{n}" if n else "not measured"

    mutations = list(pm.get("per_mutation", {}).values())
    synthetic = list(st.values())
    mn = sum(s["n"] for s in mutations)
    mh = sum(s["recall"] for s in mutations)
    sn = sum(s["n"] for s in synthetic)
    sh = sum(s["ok"] for s in synthetic)
    counts = ec["counts"]
    return {
        "fp": ratio(fp["failure_errors_emitted"], fp["n_resolved"]),
        "fp_scope": "outcome_gate_control",
        "mutation": f"{100 * mh / mn:.1f}%" if mn else "not measured",
        "synthetic": f"{100 * sh / sn:.1f}%" if sn else "not measured",
        "judge_wiring": ratio(jp["passed"] + jb["passed"], jp["n"] + jb["n"]),
        "arbiter_wiring": ratio(aw["passed"], aw["n"]),
        "catalog": ratio(cat["passed"], cat["n"]),
        "unit_tests": n_tests,
        "evidence_audit": ratio(counts.get("re_derived", 0) + counts.get("structural_ok", 0), ec["n"]),
        "evidence_mismatches": counts.get("mismatch", 0),
        "evidence_audit_counts": counts,
        "sample_counts": {
            "mutation": {"hits": mh, "n": mn},
            "synthetic": {"hits": sh, "n": sn},
            "solved_runs": fp["n_resolved"],
        },
    }
