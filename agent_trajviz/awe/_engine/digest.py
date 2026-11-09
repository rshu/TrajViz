"""Cross-agent insights digest — the one-page 'at a glance' entry point.

Root-cause view: counts PRIMARY faults only (the dominant cause per failure),
so the headline reflects what actually breaks each agent, not co-occurring
symptoms. A separate table surfaces the contributing/process faults."""

from __future__ import annotations

import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from . import config
from .outcomes import resolved, load_pertest
from .detect import detect
from .faultside import assess as fault_side_assess
from .rootcause import build_blame
from .arbiter import auto_adjudicate
from .evidence import strength_of
from .scoring import agent_capability_scores
from .report import CAP_LABEL
from .annotation import ERRORS_BY_CAPABILITY
from .significance import rank_with_significance, paired_bootstrap_diff, holm
from .trajsignals import tool_behavior

LOC_ERRORS = ("missed_relevant_file", "cross_module_context_missing", "wrong_file_localization")
PROCESS_COLS = (
    "test_result_misinterpreted",
    "relevant_test_not_run",
    "unsupported_success_claim",
    "no_test_after_final_edit",
    "test_expectation_rewritten",
    "authored_test_never_run",
    "regression_test_missing",
    "repeated_ineffective_attempt",
    "repair_not_reverified",
    "reverted_own_change",
    "tool_failure_not_retried",
    "repeated_patch_apply_failure",
    "hallucinated_path_unrecovered",
)

# The LLM judge's closed label sets — NOT the full taxonomy lists: requirement/
# planning now also carry OBJECTIVE anchors (solved_unrelated_problem), and the
# "judge yield" stat must count only what the judge itself diagnosed.
from .judge import REQUIREMENT_ERRORS as _REQ_JUDGE, PLANNING_ERRORS as _PLAN_JUDGE

_JUDGE_ETS = set(_REQ_JUDGE) | set(_PLAN_JUDGE)


def agent_digest(agent: str, instances) -> dict:
    primary, allf, ex = Counter(), Counter(), {}
    cap_mass = Counter()  # capability blame mass over failures (sums ≈ n_failed)
    mode_mass: dict[str, Counter] = {}  # full-precision mass by capability + error type
    mode_count: dict[str, Counter] = {}  # unweighted fault-entry count for CSV auditing
    unattributed_mass = 0.0  # explicit gaps, including an arbiter-refuted singleton
    strength_mass = Counter()  # blame mass by evidence-strength tier (error-type-aware)
    fside = Counter()  # v0.33 case-level repair locus (side, over failures)
    pertest_cov = 0  # v0.39 eval+ tier: failed cases with a GRADABLE per-test record
    adjudicate = auto_adjudicate(agent)  # LLM arbiter (no-op unless verdicts cached)
    n = nf = conj = unknown = judge_supported = 0
    attribution_statuses = Counter()
    arbiter_applied = Counter()
    adj_total = adj_refute = 0
    hidden = False
    for inst in instances:
        r = resolved(agent, inst)
        if r is None:
            continue
        n += 1
        if r is not False:
            continue
        nf += 1
        pt = load_pertest(agent, inst)
        if pt is not None and pt.get("patch_applied") and pt.get("f2p_total"):
            pertest_cov += 1  # gradable: the eval+ tier can actually fire
        d = detect(inst, agent, use_trajectory=True)
        if d["signals"].get("hidden_reasoning"):
            hidden = True
        fside[fault_side_assess(d.get("signals"))["side"]] += 1
        b = build_blame(d, adjudicate=adjudicate)
        attribution_statuses[b["status"]] += 1
        conj += int(b["status"] == "conjunctive")
        unknown += int(b["status"] == "unattributed")
        arb = b.get("arbiter")
        if arb:
            adj_total += 1
            arbiter_applied[arb.get("applied", "noted")] += 1
            if arb.get("applied") in ("demoted_to_conjunctive", "refuted_reassigned", "unattributed_after_refute"):
                adj_refute += 1
        for f in b.get("blame", []):
            allf[f["error_type"]] += 1
            judge_supported += int(f["error_type"] in _JUDGE_ETS and f["weight"] > 0)
            mode_mass.setdefault(f["capability"], Counter())[f["error_type"]] += f["weight"]
            mode_count.setdefault(f["capability"], Counter())[f["error_type"]] += 1
            # strength tier is ERROR-TYPE-aware (relevant_change_omitted /
            # gave_up_early are associational, solved_unrelated_problem is an
            # objective anchor in a judge cap) — aggregating by capability alone
            # would contradict scoring.compose_strength for the same faults.
            strength_mass[strength_of(f["capability"], f.get("error_type"))] += f["weight"]
        for cap, w in b.get("capability_blame", {}).items():
            cap_mass[cap] += w
        case_cap_mass = b.get("capability_blame", {})
        explicit_unattributed = case_cap_mass.get("unattributed", 0.0)
        assigned = math.fsum(w for cap, w in case_cap_mass.items() if cap != "unattributed")
        if explicit_unattributed:
            unattributed_mass += explicit_unattributed
        elif not math.isclose(assigned, 1.0, rel_tol=0.0, abs_tol=1e-12):
            # A refuted singleton has assigned == 0 after its mass is zeroed;
            # preserve the resulting whole unit as unattributed. Near-one
            # deviations are binary summation noise, not missing diagnosis.
            unattributed_mass += max(0.0, 1.0 - assigned)
        p = b.get("primary")
        if p:
            primary[p["error_type"]] += 1
            if p["weight"] >= ex.get(p["error_type"], (None, -1))[1]:
                ex[p["error_type"]] = (inst, p["weight"])
    sc = agent_capability_scores(agent, instances, use_trajectory=True, bootstrap=200, return_vectors=True)
    scores = {c: round(v["score"], 1) for c, v in sc["capabilities"].items()}
    bottleneck = min(scores, key=scores.get)
    # diagnosis trust profile: share of blame mass by evidence-strength tier,
    # accumulated per blame ENTRY (error-type-aware) — see the loop above.
    stot = sum(strength_mass.values()) or 1.0
    strength = {k: round(strength_mass.get(k, 0.0) / stot, 3) for k in ("deductive", "associational", "model_inferred")}
    # Make the detailed mode ledger canonical for attributed phase mass. This
    # prevents summation-order drift between phase and within-phase totals.
    cap_mass = Counter({cap: math.fsum(masses.values()) for cap, masses in mode_mass.items()})
    if unattributed_mass:
        cap_mass["unattributed"] = unattributed_mass
    return {
        "agent": agent,
        "n": n,
        "n_failed": nf,
        "failure_rate": (nf / n if n else 0),
        "hidden": hidden,
        "bottleneck": bottleneck,
        "scores": scores,
        "strength": strength,
        "vectors": sc.get("vectors", {}),
        "fail_vector": sc.get("fail_vector", {}),
        "primary": primary,
        "allf": allf,
        "cap_mass": cap_mass,
        "mode_mass": {cap: dict(mass) for cap, mass in mode_mass.items()},
        "mode_count": {cap: dict(counts) for cap, counts in mode_count.items()},
        "examples": ex,
        "conjunctive": conj,
        "unattributed": unknown,
        "attribution_statuses": dict(attribution_statuses),
        "judge_supported_faults": judge_supported,
        "judge_opportunities": {
            cap: v["opportunity"]
            for cap, v in sc["capabilities"].items()
            if cap in ("requirement_understanding", "task_planning")
        },
        "adj_total": adj_total,
        "adj_refute": adj_refute,
        "arbiter_applied": dict(arbiter_applied),
        "fault_side": fside,
        "pertest_cov": pertest_cov,
        "tool_behavior": tool_behavior(agent, instances),
    }


def _agent_digest_payload(payload):
    agent, instances = payload
    return agent, agent_digest(agent, instances)


def build_digest(instances=None, workers: int = 1) -> dict:
    instances = config.require_instances(instances if instances is not None else config.list_instances())
    payloads = [(agent, instances) for agent in config.AGENTS]
    if workers <= 1 or len(payloads) <= 1:
        return {agent: agent_digest(agent, instances) for agent in config.AGENTS}
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
        return dict(pool.map(_agent_digest_payload, payloads))


# Workflow phases in causal order. The first two are the LLM-judge capabilities;
# they appear only when judge verdicts exist (else the row is omitted, not faked).
_BASE_PROC = [
    ("requirement_understanding", "Requirement Understanding"),
    ("task_planning", "Task Planning"),
    ("code_localization", "Localization"),
    ("code_editing", "Editing"),
    ("code_verification", "Verification"),
    ("self_repair_loop", "Self-Repair"),
    ("tool_use", "Tool Use"),
]


def _judge_present(D: dict) -> bool:
    return any("requirement_understanding" in D[a]["scores"] for a in config.AGENTS)


_EVALPLUS_TYPES = ("partial_fix_only", "regression_introduced", "regression_test_missing")


def _evalplus_coverage_note(D: dict) -> str | None:
    """v0.39: the eval+ tier fires only where a per-test harness record exists
    (`data/pertest/<agent>/`). When SOME agents carry records and others none,
    the cross-agent Editing / Verification numbers mix coverages, so the digest
    says so — computed from the per-agent `pertest_cov` count, never
    hand-written. None when every agent is covered, none is, or the digest
    predates the field."""
    A = config.AGENTS
    cov = {a: D[a].get("pertest_cov") for a in A}
    if any(c is None for c in cov.values()):
        return None
    have = [a for a in A if cov[a]]
    if not have or len(have) == len(A):
        return None
    covered = ", ".join(f"`{a}` ({cov[a]}/{D[a]['n_failed']} failed)" for a in have)
    missing = ", ".join(f"`{a}`" for a in A if not cov[a])
    types = ", ".join(f"`{t}`" for t in _EVALPLUS_TYPES)
    return (
        f"- **eval+ coverage is partial:** per-test harness records (the `eval+` tier — "
        f"{types}) exist for {covered} only; {missing} carry none. Those fault types "
        f"can only fire for the covered agents, so cross-agent Editing / Verification "
        f"comparisons mix coverages — read them as within-agent detail, not as a "
        f"ranking, until the other agents are re-evaluated (DESIGN §0av)."
    )


def _procs(D: dict) -> list[tuple]:
    return [(c, l) for c, l in _BASE_PROC if any(c in D[a]["scores"] for a in config.AGENTS)]


def _exact_pct(numerator: float, denominator: float) -> str:
    """Percentage with no presentation rounding (round-trip float spelling)."""
    return f"{100 * numerator / denominator!r}%" if denominator else "0.0%"


def _judge_yield_note(D: dict) -> str:
    """Honest statement of the LLM judge's actual yield — it scores Requirement
    Understanding & Task Planning, but was the diagnosed PRIMARY cause in only a
    small fraction of failures. Prevents the '7 of 7' framing from overselling."""
    A = config.AGENTS
    nf = sum(D[a].get("n_failed", 0) for a in A) or 1
    jp = sum(D[a]["primary"].get(et, 0) for a in A for et in _JUDGE_ETS)
    jc = sum(D[a].get("judge_supported_faults", sum(D[a]["allf"].get(et, 0) for et in _JUDGE_ETS)) for a in A)
    return (
        f"Judge capabilities cannot take primary attribution by policy "
        f"(**{jp} of {nf}** failures); this is not evidence that they never matter. "
        f"There are {jc} positive-weight judge fault entries after all gates and "
        f"adjudication. These labels describe contributing evidence; causal accuracy "
        f"has not been measured."
    )


def _arbiter_summary_line(D: dict) -> str:
    """Surface how often the LLM arbiter REFUTED the non-deductive primaries it was
    asked to corroborate. Empty when no arbiter verdicts were applied."""
    A = config.AGENTS
    tot = sum(D[a].get("adj_total", 0) for a in A)
    ref = sum(D[a].get("adj_refute", 0) for a in A)
    if tot == 0:
        return ""
    applied = Counter()
    for a in A:
        applied.update(D[a].get("arbiter_applied", {}))
    detail = (
        f" Outcomes: {applied['demoted_to_conjunctive']} no dominant primary, "
        f"{applied['refuted_reassigned']} reassigned, "
        f"{applied['unattributed_after_refute']} unattributed. "
        if applied
        else " "
    )
    return (
        f"- **Arbiter check:** an outcome-blinded LLM second opinion adjudicated "
        f"{tot} interpretive primary candidates and **applied {ref} refutes "
        f"({100 * ref / tot:.0f}%)**."
        + detail
        + "Deductive Localization/Editing primary candidates are never sent to the arbiter."
    )


def _significance(D: dict, seed: int = 0, B: int = 2000, alpha: float = 0.05) -> dict:
    """Decide, with paired bootstrap + Holm over the whole family of binding
    comparisons, which row best/worst claims (and the overall strongest/weakest by
    failure rate) are statistically separated. Pure over the digest's D dict."""
    rng = random.Random(seed)
    A = config.AGENTS
    procs = _procs(D)
    verdicts: dict = {}
    comps: list = []  # (key, role, p)
    for cap, _ in procs:
        have = [a for a in A if cap in D[a]["scores"]]
        scores = {a: D[a]["scores"][cap] for a in have}
        vectors = {a: D[a].get("vectors", {}).get(cap, {}) for a in have}
        r = rank_with_significance(scores, vectors, rng, B)
        verdicts[cap] = r
        if len(have) == 2:
            # best-vs-runner and worst-vs-next are the SAME unordered pair here:
            # one comparison, one family entry (two independently-resampled
            # entries double-counted the Holm family and could disagree).
            comps.append((cap, "pair", r["p_best"]))
        else:
            comps.append((cap, "best", r["p_best"]))
            comps.append((cap, "worst", r["p_worst"]))
    # overall strongest/weakest by failure rate (lower rate = stronger)
    order = sorted(A, key=lambda a: D[a].get("failure_rate", 0.0))
    fv = {a: D[a].get("fail_vector", {}) for a in A}
    if len(order) > 1:
        ps = paired_bootstrap_diff(fv[order[0]], fv[order[1]], rng, B)["p_two_sided"]
        pw = paired_bootstrap_diff(fv[order[-1]], fv[order[-2]], rng, B)["p_two_sided"]
    else:
        ps = pw = 1.0
    if len(order) == 2:
        comps.append(("_overall", "pair", ps))
    else:
        comps.append(("_overall", "strong", ps))
        comps.append(("_overall", "weak", pw))
    adj = holm([p for _, _, p in comps])
    sigmap = {(k, role): a < alpha for (k, role, _), a in zip(comps, adj, strict=False)}

    def _sig(key, role):
        if (key, "pair") in sigmap:
            return sigmap[(key, "pair")]
        return sigmap[(key, role)]

    for cap, _ in procs:
        v = verdicts[cap]
        v["best_sig"] = _sig(cap, "best") and v["best"] is not None
        v["worst_sig"] = _sig(cap, "worst") and v["worst"] is not None
        v["sig"] = v["best_sig"] or v["worst_sig"]
    verdicts["_overall"] = {
        "strong": order[0],
        "weak": order[-1],
        "strong_sig": _sig("_overall", "strong"),
        "weak_sig": _sig("_overall", "weak"),
    }
    return verdicts


def _performance_by_process(D: dict, verdicts: dict | None = None) -> list[str]:
    """Every agent across every measured phase. Best/worst are marked ONLY when the
    paired, Holm-corrected comparison is significant; a statistically tied row is
    tagged `ns`. Score = 100 − blame-rate; higher is better."""
    A = config.AGENTS
    procs = _procs(D)
    verdicts = verdicts or _significance(D)
    short = [a.replace("opencode_", "oc_") for a in A]
    err2cap = {e: c for c, es in ERRORS_BY_CAPABILITY.items() for e in es}
    cap_err = {cap: Counter() for cap, _ in procs}
    for a in A:
        for et, c in D[a]["allf"].items():
            if err2cap.get(et) in cap_err:
                cap_err[err2cap[et]][et] += c

    L = [
        "## Performance by workflow process",
        "",
        "Every agent across every measured phase (score = 100 − blame-rate; higher is "
        "better). **Bold** = best, _italic_ = weakest — shown **only when the paired, "
        "Holm-corrected difference is significant**; `ns` = no significant separation; "
        "n/a = not measured for that agent.",
        "",
        "| Workflow process | " + " | ".join(short) + " | sig |",
        "|---|" + "|".join("---" for _ in A) + "|---|",
    ]
    for cap, label in procs:
        have = [a for a in A if cap in D[a]["scores"]]
        sc = {a: D[a]["scores"][cap] for a in have}
        v = verdicts.get(cap, {})
        best = v.get("best") if v.get("best_sig") else None
        worst = v.get("worst") if v.get("worst_sig") else None
        cells = []
        for a in A:
            if a not in sc:
                cells.append("n/a")
            elif a == best:
                cells.append(f"**{sc[a]:.1f}**")
            elif a == worst:
                cells.append(f"_{sc[a]:.1f}_")
            else:
                cells.append(f"{sc[a]:.1f}")
        sig_tag = "yes" if v.get("sig") else "ns"
        L.append(f"| {label} | " + " | ".join(cells) + f" | {sig_tag} |")

    L += ["", "### Per-process read", ""]
    for cap, label in procs:
        have = [a for a in A if cap in D[a]["scores"]]
        if not have:
            continue
        sc = {a: D[a]["scores"][cap] for a in have}
        ranked = sorted(have, key=lambda a: -sc[a])
        te = cap_err[cap].most_common(1)
        tetxt = f"; dominant fault `{te[0][0]}` (×{te[0][1]})" if te else ""
        v = verdicts.get(cap, {})
        if v.get("sig"):
            L.append(
                f"- **{label}** — strongest `{ranked[0]}` ({sc[ranked[0]]:.1f}), "
                f"weakest `{ranked[-1]}` ({sc[ranked[-1]]:.1f}), "
                f"spread {sc[ranked[0]] - sc[ranked[-1]]:.1f}{tetxt}."
            )
        else:
            L.append(
                f"- **{label}** — no significant separation across agents "
                f"(scores {sc[ranked[0]]:.1f}–{sc[ranked[-1]]:.1f}, ns){tetxt}."
            )
    L.append("")
    return L


def _failure_share_section(D: dict) -> list[str]:
    """Comparative view: of each agent's FAILURES, what share each capability is
    to blame for (denominator = that agent's failures, not all tasks). Unlike the
    absolute score it is not diluted by the solved majority, so it uses the full
    range and discriminates agents. Shares per agent sum to ≈100% (remainder =
    unattributed failures with no confident cause)."""
    A = config.AGENTS
    procs = _procs(D)
    short = [a.replace("opencode_", "oc_") for a in A]
    L = [
        "## Failure composition",
        "",
        "### Share of each agent's failures by phase",
        "",
        "Read **down** a column: where does *this* agent's failure mass concentrate. "
        "Denominator is the agent's own failures, so this is the comparative lens the "
        "absolute scores compress. **Bold** = that agent's largest share. Values are "
        "emitted without presentation rounding; exact masses and denominators are in "
        "the detailed CSV. Shares decompose each column's failure mass (remainder = "
        "unattributed).",
        "",
        "| Phase \\ agent | " + " | ".join(short) + " |",
        "|---|" + "|".join("---" for _ in A) + "|",
    ]
    # largest share per agent (for bolding the column max)
    col_max = {}
    for a in A:
        nf = D[a]["n_failed"] or 1
        shares = {c: D[a]["cap_mass"].get(c, 0.0) / nf for c, _ in procs}
        col_max[a] = max(shares, key=shares.get) if shares else None
    for cap, label in procs:
        cells = []
        for a in A:
            nf = D[a]["n_failed"] or 1
            s = _exact_pct(D[a]["cap_mass"].get(cap, 0.0), nf)
            cells.append(f"**{s}**" if col_max[a] == cap else s)
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    # unattributed remainder row
    rem = []
    for a in A:
        nf = D[a]["n_failed"] or 1
        rem.append(_exact_pct(D[a]["cap_mass"].get("unattributed", 0.0), nf))
    L.append("| _unattributed_ | " + " | ".join(rem) + " |")
    L.append("")
    return L


def _modes_for_capability(D: dict, capability: str) -> list[str]:
    """Observed modes for one capability, ordered by aggregate blame mass."""
    masses = Counter()
    for agent in config.AGENTS:
        masses.update((D[agent].get("mode_mass") or {}).get(capability, {}))
    return [mode for mode, mass in sorted(masses.items(), key=lambda item: (-item[1], item[0])) if mass > 0]


def failure_mode_rows(D: dict) -> list[dict]:
    """Long-form exact data behind the within-phase mode tables.

    Each observed ``(phase, mode)`` is emitted for every agent plus one
    ``__all_agents__`` aggregate row. The two percentages expose both useful
    denominators: attributed phase mass and all failed tasks.
    """
    rows = []
    total_failed = sum(D[a].get("n_failed", 0) for a in config.AGENTS)
    for capability, label in _procs(D):
        modes = _modes_for_capability(D, capability)
        if not modes:
            continue
        aggregate_mode_masses = {
            mode: math.fsum((D[a].get("mode_mass") or {}).get(capability, {}).get(mode, 0.0) for a in config.AGENTS)
            for mode in modes
        }
        aggregate_phase_mass = math.fsum(aggregate_mode_masses.values())
        for mode in modes:
            aggregate_mode_mass = aggregate_mode_masses[mode]
            aggregate_count = sum(
                (D[a].get("mode_count") or {}).get(capability, {}).get(mode, 0) for a in config.AGENTS
            )
            scopes = [("__all_agents__", aggregate_mode_mass, aggregate_phase_mass, total_failed, aggregate_count)]
            scopes.extend(
                (
                    a,
                    (D[a].get("mode_mass") or {}).get(capability, {}).get(mode, 0.0),
                    D[a].get("cap_mass", {}).get(capability, 0.0),
                    D[a].get("n_failed", 0),
                    (D[a].get("mode_count") or {}).get(capability, {}).get(mode, 0),
                )
                for a in config.AGENTS
            )
            for agent, mode_mass_value, phase_mass, failed, count in scopes:
                rows.append(
                    {
                        "phase": capability,
                        "phase_label": label,
                        "root_cause_mode": mode,
                        "agent": agent,
                        "fault_entry_count": count,
                        "attributed_blame_mass": mode_mass_value,
                        "phase_blame_mass": phase_mass,
                        "failed_tasks": failed,
                        "share_within_phase_pct": (100 * mode_mass_value / phase_mass if phase_mass else None),
                        "share_of_agent_failures_pct": (100 * mode_mass_value / failed if failed else None),
                    }
                )
    return rows


def _failure_mode_share_section(D: dict) -> list[str]:
    """Decompose every capability's blame mass into its observed error modes."""
    if not any(D[a].get("mode_mass") for a in config.AGENTS):
        return []
    agents = config.AGENTS
    short = [a.replace("opencode_", "oc_") for a in agents]
    lines = [
        "### Root-cause mode distribution within each phase",
        "",
        "Within each phase, read **down** an agent column: each cell is the "
        "mode's share of the blame mass attributed to that phase for that agent. "
        "Values use the full round-trip representation of the computed float, with "
        "no presentation rounding. These are weighted primary + contributing "
        "attributions, not raw fault counts. **Overall** pools blame mass across "
        "all agents; **—** means the agent has no attributed mass in that phase. "
        "Exact masses, fault-entry counts, and both denominators are in "
        "[`failure_composition_by_mode.csv`](failure_composition_by_mode.csv).",
        "",
    ]

    for capability, label in _procs(D):
        modes = _modes_for_capability(D, capability)
        if not modes:
            continue
        overall_masses = {
            mode: math.fsum((D[a].get("mode_mass") or {}).get(capability, {}).get(mode, 0.0) for a in agents)
            for mode in modes
        }
        overall_den = math.fsum(overall_masses.values())
        overall_max = max(overall_masses, key=overall_masses.get)
        col_max = {}
        for agent in agents:
            masses = (D[agent].get("mode_mass") or {}).get(capability, {})
            col_max[agent] = (
                max(modes, key=lambda mode: masses.get(mode, 0.0))
                if D[agent].get("cap_mass", {}).get(capability, 0.0)
                else None
            )

        lines += [
            f"#### {label}",
            "",
            "| Root-cause mode | Overall | " + " | ".join(short) + " |",
            "|---|---|" + "|".join("---" for _ in agents) + "|",
        ]
        for mode in modes:
            overall = _exact_pct(overall_masses[mode], overall_den)
            overall_cell = f"**{overall}**" if mode == overall_max else overall
            cells = []
            for agent in agents:
                denominator = D[agent].get("cap_mass", {}).get(capability, 0.0)
                if not denominator:
                    cells.append("—")
                    continue
                mass = (D[agent].get("mode_mass") or {}).get(capability, {}).get(mode, 0.0)
                share = _exact_pct(mass, denominator)
                cells.append(f"**{share}**" if col_max[agent] == mode else share)
            lines.append(f"| `{mode}` | {overall_cell} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _coverage_section(D: dict) -> list[str]:
    """Trajectory accounting: how many trajectories were analyzed in total, how
    many failed, and that every failure was diagnosed (the analyzed set)."""
    A = config.AGENTS
    total = sum(D[a]["n"] for a in A)
    failed = sum(D[a]["n_failed"] for a in A)
    resolved_n = total - failed
    judged = _judge_present(D)
    L = [
        "## Trajectory coverage",
        "",
        f"The selected cohort contains **{total:,} scored runs** "
        f"across {len(A)} agent configurations; the {resolved_n:,} resolved cases serve as the "
        f"negative control for the outcome gate. Failure labels are suppressed on solved "
        f"runs by construction; this does not measure causal precision. The **{failed:,} failed** "
        f"runs are analyzed, including cases with no supported primary attribution. "
        f"Trajectory-dependent evidence requires an available trace; judge coverage "
        f"requires a usable cached verdict.",
        "",
        "| Agent | Scored runs | Resolved | Failed (analyzed for faults) |",
        "|---|---|---|---|",
    ]
    for a in A:
        d = D[a]
        L.append(f"| {a} | {d['n']} | {d['n'] - d['n_failed']} | {d['n_failed']} ({100 * d['failure_rate']:.0f}%) |")
    L.append(f"| **Total** | **{total:,}** | **{resolved_n:,}** | **{failed:,} ({100 * failed / total:.0f}%)** |")
    L.append("")
    if judged:
        L += [
            "Judge capability opportunities (availability plus task eligibility):",
            "",
            "| Agent | Requirement Understanding | Task Planning |",
            "|---|---|---|",
        ]
        for a in A:
            opportunities = D[a].get("judge_opportunities", {})
            L.append(
                f"| {a} | {opportunities.get('requirement_understanding', 'not recorded')} | "
                f"{opportunities.get('task_planning', 'not recorded')} |"
            )
        L.append("")
    return L


def _trust_profile_section(D: dict) -> list[str]:
    """How much of each agent's diagnosis rests on deductive facts vs weaker
    evidence — the headline's trust profile."""
    A = config.AGENTS
    short = [a.replace("opencode_", "oc_") for a in A]
    L = [
        "## Diagnosis trust profile",
        "",
        "Share of each agent's blame mass by evidence-strength tier — how much of the "
        "diagnosis is hard fact. **deductive** = set arithmetic over gold ∩ patch + outcome "
        "(reproducible, no model); **associational** = an objective trajectory fact in a "
        "contributing role; **model-inferred** = the outcome-blinded LLM judge. The "
        "editing/localization headline rests on reproducible observations. "
        "**These tiers describe evidence, not validated causality:** reference-patch "
        "differences and arbiter agreement do not establish that repairing a labelled "
        "fault changes the task outcome. Controlled interventions remain untested.",
        "",
        "| Evidence tier | " + " | ".join(short) + " |",
        "|---|" + "|".join("---" for _ in A) + "|",
    ]
    for tier, label in [
        ("deductive", "Deductive"),
        ("associational", "Associational"),
        ("model_inferred", "Model-inferred"),
    ]:
        cells = [f"{100 * D[a].get('strength', {}).get(tier, 0):.0f}%" for a in A]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def _attribution_section(D: dict) -> list[str]:
    """Partition failed runs by attribution status, rather than fault-entry count."""
    L = [
        "## Primary attribution coverage",
        "",
        "| Agent | Assigned primary | No dominant primary / conjunctive | Unattributed | Failed |",
        "|---|---|---|---|---|",
    ]
    totals = Counter()
    for a in config.AGENTS:
        d = D[a]
        counts = d.get("attribution_statuses")
        if counts is None:
            counts = {
                "primary": sum(d["primary"].values()),
                "conjunctive": d.get("conjunctive", 0),
                "unattributed": d.get("unattributed", 0),
            }
        totals.update(counts)
        L.append(
            f"| {a} | {counts.get('primary', 0)} | {counts.get('conjunctive', 0)} | "
            f"{counts.get('unattributed', 0)} | {d['n_failed']} |"
        )
    L.append(
        f"| **Total** | **{totals['primary']}** | **{totals['conjunctive']}** | "
        f"**{totals['unattributed']}** | **{sum(D[a]['n_failed'] for a in config.AGENTS)}** |"
    )
    L += [
        "",
        "Conjunctive means supported candidates remain without an eligible dominant primary; "
        "it does not prove multiple causal mechanisms. Only Localization and Editing can take "
        "primary attribution. Other capabilities can contribute, so zero primary counts for them "
        "reflect policy. Fault ledgers retain zero-weight refuted entries for audit; their counts "
        "are not counts of supported causes.",
        "",
    ]
    return L


def _headline_section(D: dict) -> list[str]:
    """The executive read at the very top: run scale plus three data-derived
    bullets — bottleneck, dominant root cause, repair locus. Every number is
    computed from D (never hand-written) so the headline cannot drift from the
    body; each bullet is independently guarded and simply absent when its data
    is (old digest dicts, judge-free runs, pre-v0.33 fault_side)."""
    A = config.AGENTS
    task_counts = [D[a].get("n", 0) for a in A]
    scope = (
        f"{len(A)} agents × {task_counts[0]} tasks"
        if len(set(task_counts)) == 1
        else f"{len(A)} agents, {sum(task_counts):,} scored runs "
        f"({min(task_counts)}–{max(task_counts)} tasks per agent)"
    )
    n_failed = sum(D[a].get("n_failed") or 0 for a in A)
    L = [
        "## Headline",
        "",
        f"_{scope} — {n_failed} failed cases diagnosed"
        + (f"; judge: {config.JUDGE_MODEL} (coverage-only)" if _judge_present(D) else "")
        + "._",
        "",
    ]
    bns = Counter(D[a]["bottleneck"] for a in A if D[a].get("bottleneck"))
    if bns:
        top_bn, k = bns.most_common(1)[0]
        scores = sorted(D[a]["scores"][top_bn] for a in A if top_bn in D[a].get("scores", {}))
        label = CAP_LABEL.get(top_bn, top_bn)
        who = (
            "every agent's weakest capability" if k == len(A) else f"the weakest capability for {k} of {len(A)} agents"
        )
        L.append(f"- **Bottleneck:** {label} is {who} (scores {scores[0]:.1f}–{scores[-1]:.1f}).")
    overall = Counter()
    for a in A:
        overall.update(D[a].get("primary") or {})
    if overall and n_failed:
        (et, c) = overall.most_common(1)[0]
        line = f"- **Dominant root cause:** `{et}` — {c} primary cases ({100 * c / n_failed:.0f}% of failures)"
        rest = overall.most_common(2)
        if len(rest) > 1:
            line += f"; next: `{rest[1][0]}` ({rest[1][1]})"
        L.append(line + ".")
    fs = Counter()
    for a in A:
        fs.update(D[a].get("fault_side") or {})
    note = _evalplus_coverage_note(D)
    if note:
        L.append(note)
    if fs:
        n = sum(fs.values()) or 1
        L.append(
            f"- **Repair locus:** {fs.get('model', 0)} of {n} failures are "
            f"model-side; {fs.get('grader', 0)} "
            f"({100 * fs.get('grader', 0) / n:.1f}%) are grader suspects "
            "(BA-HARNESS-1 queue, manual review pending); "
            f"{fs.get('harness', 0)} harness truncations."
        )
    L.append("")
    return L


def _fault_side_section(D: dict) -> list[str]:
    """v0.33 repair-locus distribution. Guarded like every additive section: an
    old digest dict without the key skips the section, not the build."""
    if not any(D[a].get("fault_side") for a in config.AGENTS):
        return []
    sides = ("model", "grader", "harness")
    L = [
        "## Fault side — where does the repair belong?",
        "_Case-level repair locus from the bundled fault-side assessment. `grader` is "
        "the BA-HARNESS-1 suspect queue (evaluation-apparatus suspects, manual review "
        "pending — faults stay counted); `harness` is scaffold truncation (OP-TRUNC-1). "
        "Suspects annotate, never exonerate._",
        "",
        "| Agent | model (default) | grader (suspect) | harness (suspect) |",
        "|---|---|---|---|",
    ]
    for a in config.AGENTS:
        fs = D[a].get("fault_side") or {}
        L.append("| " + a + " | " + " | ".join(str(fs.get(s, 0)) for s in sides) + " |")
    tot = Counter()
    for a in config.AGENTS:
        tot.update(D[a].get("fault_side") or {})
    n = sum(tot.values()) or 1
    L += [
        "",
        f"Across agents: {tot.get('model', 0)} model ({100 * tot.get('model', 0) / n:.0f}%), "
        f"{tot.get('grader', 0)} grader-suspect ({100 * tot.get('grader', 0) / n:.0f}%), "
        f"{tot.get('harness', 0)} harness ({100 * tot.get('harness', 0) / n:.0f}%) of "
        f"{n} failed cases. Grader suspects mirror "
        "`data/discovery/harness_suspects.md` (same predicate, shared code).",
        "",
    ]
    return L


def _tool_behavior_section(D: dict) -> list[str]:
    """Cross-agent mechanical tool-call behavior (behavioral, NOT blame). Also the
    capture-fidelity surface: before the error-flag fix, Codex read 0 here purely
    because its tool results were never parsed for errors — a measurement gap, not a
    fact about Codex. Errors are now recovered uniformly across scaffolds (explicit
    flag + high-precision result-text signature)."""
    A = config.AGENTS
    L = [
        "## Tool-use behavior (behavioral signal — every mechanical tool error)",
        "",
        "Rate at which each agent's non-test tool/shell calls fail mechanically — a "
        "hallucinated path (`invalid_target`), unknown command, malformed args, failed "
        "patch — **whether or not the agent recovered**. This is distinct from the "
        "`tool_failure_not_retried` blame, which fires only when the agent gave up after "
        "an error; most tool errors are recovered, so the rate below is behavior, not a "
        "failure cause.",
        "",
        "| Agent | Tool calls | Errored | Error rate | Tasks w/ ≥1 error | Dominant type |",
        "|---|---|---|---|---|---|",
    ]
    for a in A:
        tb = D[a].get("tool_behavior", {})
        if not tb:
            continue
        subs = tb.get("subtypes", {})
        dom = max(subs, key=subs.get) if subs else "—"
        L.append(
            f"| {a} | {tb.get('n_tool_calls', 0):,} | {tb.get('n_tool_call_errors', 0):,} "
            f"| {100 * tb.get('tool_error_rate', 0):.1f}% | "
            f"{tb.get('cases_with_error', 0)} ({100 * tb.get('case_rate', 0):.0f}%) "
            f"| `{dom}` |"
        )
    L.append("")
    return L


def discovery_summary(instances=None) -> dict:
    """Corpus-level discovery signals for the digest: the significance-gated cohort
    contrast (pooled across all agents) and the solve-side integrity check. Reads
    trajectory signals + gold/patch, so it needs ARGUS data; returns structured
    numbers only (rendering is `_discovery_section`)."""
    from . import cohort, fragile
    from .outcomes import resolved as _resolved

    instances = config.require_instances(instances if instances is not None else config.list_instances())
    rows: list = []
    for a in config.AGENTS:
        rows += cohort.feature_rows(a, instances)
    disc = cohort.discover(rows)
    frag: list = []
    skipped: list = []
    n_resolved = 0
    for a in config.AGENTS:
        n_resolved += sum(1 for i in instances if _resolved(a, i) is True)
        frag += fragile.scan(a, instances, skipped=skipped)
    n_destructive = sum(1 for r in frag if r.get("modified_existing"))
    return {
        "cohort": disc,
        "fragile": {
            "n": len(frag),
            "n_resolved": n_resolved,
            "n_destructive": n_destructive,
            "n_skipped": len(skipped),
        },
    }


def _discovery_section(disc: dict, alpha: float = 0.05, top: int = 7) -> list[str]:
    """Render the corpus-level discovery signals: the significance-gated cohort
    contrast (associational, not causal) + the solve-side integrity check. Pure."""
    cohort_d = disc.get("cohort") or {}
    assoc = cohort_d.get("associations", [])
    sig = [a for a in assoc if a["p_holm"] < alpha and a.get("n_failed") and a.get("n_solved")]
    L = [
        "## Discovery signals (corpus-level, significance-gated)",
        "",
        "Behaviors whose prevalence differs between the pooled failed and solved "
        "cohorts (unpaired bootstrap, Holm-corrected). **Associational, not "
        "causal** — a difference is not proof of cause. Only rows that survive the "
        "gate are shown, strongest first:",
        "",
    ]
    if not sig:
        L += [
            "_No behavior is significantly associated with failure after "
            "correction — the validation gate working, not a null to hide._",
            "",
        ]
    else:
        L += ["| Behavior | more common in | failed | solved | effect | p (Holm) |", "|---|---|---|---|---|---|"]
        for a in sig[:top]:
            direction = "**failures**" if a["diff"] > 0 else "successes"
            if a.get("binary"):
                rr = a.get("risk_ratio")
                if rr is None:
                    eff = f"OR={a['odds_ratio']:.2f}" if a.get("odds_ratio") else "—"
                elif rr == float("inf"):
                    eff = "RR=∞"
                else:
                    eff = f"RR={rr:.2f}"
                fv, sv = f"{100 * a['mean_failed']:.0f}%", f"{100 * a['mean_solved']:.0f}%"
            else:
                eff = f"Δmean={a['diff']:+.2f}"
                fv, sv = f"{a['mean_failed']:.2f}", f"{a['mean_solved']:.2f}"
            L.append(f"| `{a['feature']}` | {direction} | {fv} | {sv} | {eff} | {a['p_holm']:.3f} |")
        if len(sig) > top:
            rest = ", ".join(f"`{a['feature']}`" for a in sig[top:])
            L.append(f"| _+{len(sig) - top} more: {rest}_ | | | | | |")
        L.append("")
        sr = next(
            (a for a in sig if a["feature"] == "step_repetition" and a.get("risk_ratio") not in (None, float("inf"))),
            None,
        )
        if sr:
            # RR = P(fail|behavior)/P(fail|no behavior) — a failure-risk ratio,
            # NOT the prevalence ratio (that is mean_failed/mean_solved).
            L += [
                f"**`step_repetition` (MAST-3):** a non-edit action re-issued with no "
                f"world-change between tries — runs showing it fail "
                f"{sr['risk_ratio']:.2f}× as often (it appears in "
                f"{100 * sr['mean_failed']:.0f}% of failed vs "
                f"{100 * sr['mean_solved']:.0f}% of solved runs). A loop-thrash signal "
                f"the fixed taxonomy had no slot for.",
                "",
            ]
    f = disc.get("fragile") or {}
    if f.get("n_resolved"):
        pct = 100 * f["n"] / f["n_resolved"]
        L += [
            f"**Benchmark-integrity check (solve side):** {f['n']:,}/{f['n_resolved']:,} "
            f"resolved cases ({pct:.0f}%) edited a gold gating-test file "
            f"({f['n_destructive']:,} destructively). The effect on grading depends "
            f"on the benchmark's recorded reset policy. Oracle touch alone does "
            f"not establish gaming or evaluator integrity "
            f"(`data/discovery/fragile_solves.md`).",
            "",
        ]
    return L


def _discovery_block(discovery: dict | None) -> list[str]:
    """_discovery_section, guarded: discovery is an additive layer, so neither a
    missing blob nor a schema-drifted one may ever block the INSIGHTS build."""
    if not discovery:
        return []
    try:
        return _discovery_section(discovery)
    except Exception:
        return []


def build_insights_md(
    D: dict, discovery: dict | None = None, recommendations: dict | None = None, attribution_graphs: dict | None = None
) -> str:
    A = config.AGENTS
    judged = _judge_present(D)
    n_cov = 7 if judged else 5
    task_counts = {D[a].get("n", 0) for a in A}
    if len(A) == 1:
        scope = f"{D[A[0]].get('n', 0):,} evaluated tasks"
    elif len(task_counts) == 1:
        scope = f"{next(iter(task_counts)):,} tasks/agent"
    else:
        scope = f"{sum(D[a].get('n', 0) for a in A):,} agent-task evaluations"
    L = [
        "# Cross-Agent Insights Digest",
        "_Root-cause view: **primary** faults only (dominant cause per failure). "
        f"{config.BENCHMARK_NAME}, {scope}. Read top-to-bottom; drill into the named "
        "example cases in each `<agent>.md`._",
        "",
    ]
    L += _headline_section(D)
    if attribution_graphs is not None:
        if attribution_graphs["scope"]["agents"] != sorted(D) or attribution_graphs["scope"][
            "diagnosed_failed_cases"
        ] != sum(d.get("n_failed", 0) for d in D.values()):
            raise ValueError("Attribution graph scope differs from the insights digest")
        L += [
            "## Follow the attribution",
            "",
            "[Open ATTRIBUTION.html](ATTRIBUTION.html) for the numbered evidence DAG of every diagnosed case. "
            "Select a node to inspect source observations, assessments, exact recorded weights, "
            "arbiter changes and selection reasons. Proposed causal checks remain separate from observed evidence.",
            "",
        ]
    if recommendations is not None:
        from .remedy_plan import insights_section

        if recommendations["scope"]["agents"] != sorted(D) or recommendations["scope"]["diagnosed_failed_cases"] != sum(
            d.get("n_failed", 0) for d in D.values()
        ):
            raise ValueError("Recommendation plan scope differs from the insights digest")
        L += insights_section(recommendations)
    L += [f"## Coverage — {n_cov} of 7 workflow phases", ""]
    if judged:
        L += [
            "Seven phases are reported (five objectively anchored + two LLM-judge — see yield below). "
            "The five objectively-anchored phases use the gold "
            "patch + tests + trajectory structure; the two judgment-heavy phases "
            "(**Requirement Understanding, Task Planning**) are scored by a constrained, "
            "outcome-blinded LLM judge over a closed error taxonomy (see LLM_JUDGE.md). "
            "Judge verdicts contribute blame weight but can never take primary blame "
            "(v0.32: the judge's confidence tiers select no reliable subset — flag-rate "
            "direction is unstable across judge models at sub-1% yields — so the two "
            "judge phases are coverage-only).",
            "",
            "| Workflow phase / capability | Status |",
            "|---|---|",
            "| Requirement Understanding | ✅ measured (LLM judge) |",
            "| Task Planning | ✅ measured (LLM judge) |",
            "| Code Localization | ✅ measured |",
            "| Code Editing | ✅ measured |",
            "| Tool Use | ✅ measured |",
            "| Code Verification | ✅ measured |",
            "| Self-Repair Loop | ✅ measured |",
            "",
        ]
        L += [_judge_yield_note(D), ""]
    else:
        L += [
            "Only **objectively-anchored** phases are scored (gold patch + tests + trajectory "
            "structure). The two judgment-heavy phases are **pending the LLM-judge layer** "
            "(run `scripts/run_judge.py`): pass/fail and a gold patch cannot verify an agent's "
            "*reasoning*, and such failures have no objective signature — they propagate "
            "downstream and currently surface here as Localization/Editing faults.",
            "",
            "| Workflow phase / capability | Status |",
            "|---|---|",
            "| Requirement Understanding | ⏳ pending (LLM judge) |",
            "| Task Planning | ⏳ pending (LLM judge) |",
            "| Code Localization | ✅ measured |",
            "| Code Editing | ✅ measured |",
            "| Tool Use | ✅ measured |",
            "| Code Verification | ✅ measured |",
            "| Self-Repair Loop | ✅ measured |",
            "",
        ]
    L += _coverage_section(D)
    L += _attribution_section(D)
    verdicts = _significance(D)
    L += _performance_by_process(D, verdicts)
    L += _trust_profile_section(D)
    L += _fault_side_section(D)
    L += _tool_behavior_section(D)
    L += _discovery_block(discovery)
    L += _failure_share_section(D)
    L += _failure_mode_share_section(D)
    L += [
        "## At a glance — root-cause profile",
        "_The single weakest point per agent (complements the full table above)._",
        "",
        "| Agent | Fail% | Bottleneck (score) | Top primary root causes |",
        "|---|---|---|---|",
    ]
    for a in A:
        d = D[a]
        top = ", ".join(f"`{et}` ({c})" for et, c in d["primary"].most_common(3))
        L.append(
            f"| {a} | {100 * d['failure_rate']:.0f}% | "
            f"{CAP_LABEL.get(d['bottleneck'], d['bottleneck'])} "
            f"({d['scores'][d['bottleneck']]}) | {top} |"
        )

    overall = Counter()
    for a in A:
        overall += D[a]["primary"]
    top1 = overall.most_common(1)[0] if overall else None
    loc_total = sum(D[a]["primary"][e] for a in A for e in LOC_ERRORS)
    weak = max(A, key=lambda a: D[a]["failure_rate"])
    strong = min(A, key=lambda a: D[a]["failure_rate"])
    sr = max(A, key=lambda a: D[a]["allf"]["repeated_ineffective_attempt"])
    hid = [a for a in A if D[a]["hidden"]]

    L += ["", "## Key patterns", ""]
    if all(D[a]["bottleneck"] == "code_editing" for a in A):
        L.append(
            "- **Editing is the universal bottleneck.** Every agent's lowest capability "
            "score is Code Editing — they locate the code better than they correctly change it."
        )
    if top1 is not None:
        top_explanation = {
            "cross_module_context_missing": (
                "on a cross-module task, the agent reaches at least one reference file "
                "but leaves another required module unreached"
            ),
            "incorrect_patch": "the reference region is reached, but the submitted change is incorrect",
        }.get(top1[0], "see the detector definition and case evidence for the exact predicate")
        L.append(
            f"- **`{top1[0]}` is the dominant root cause across all agents** "
            f"({top1[1]} primary cases): {top_explanation}."
        )
        loc_rank = "dominates overall" if top1[0] in LOC_ERRORS else "is the #2 root-cause family"
        L.append(
            f"- **Localization {loc_rank}** ({loc_total} primary cases) — "
            f"led by cross-module misses and missed functions."
        )
    else:
        L.append("- No primary root causes were diagnosed; no dominant family is inferred.")
    ov = verdicts["_overall"]
    if ov["strong_sig"] or ov["weak_sig"]:
        L.append(
            f"- **Lowest failure rate:** `{strong}` "
            f"({100 * D[strong]['failure_rate']:.0f}%) · **Highest:** `{weak}` "
            f"({100 * D[weak]['failure_rate']:.0f}%) — difference is statistically "
            f"significant (paired, Holm-corrected)."
        )
    else:
        L.append(
            f"- **Failure rates are statistically indistinguishable** across "
            f"agents (range {100 * min(D[a]['failure_rate'] for a in A):.0f}–"
            f"{100 * max(D[a]['failure_rate'] for a in A):.0f}%); no agent is "
            f"significantly best or worst overall."
        )
    _arb = _arbiter_summary_line(D)
    if _arb:
        L.append(_arb)
    if D[sr]["allf"]["repeated_ineffective_attempt"]:
        L.append(
            f"- **Repair loops:** `{sr}` most often re-edits the same file after a failure "
            f"without progress (`repeated_ineffective_attempt` ×{D[sr]['allf']['repeated_ineffective_attempt']}). "
            "Contributing fault — wasted effort, rarely the root cause."
        )
    if hid:
        L.append(
            f"- **Observability caveat:** {', '.join(f'`{a}`' for a in hid)} hide reasoning "
            "(encrypted), so 'thinking' evidence is partial — those diagnoses lean on actions + messages."
        )

    L += [
        "",
        "## Process weaknesses (contributing faults — symptoms, not root causes)",
        "",
        "| Agent | " + " | ".join(c.replace("_", " ") for c in PROCESS_COLS) + " |",
        "|---|" + "|".join("---" for _ in PROCESS_COLS) + "|",
    ]
    for a in A:
        L.append(f"| {a} | " + " | ".join(str(D[a]["allf"][c]) for c in PROCESS_COLS) + " |")

    L += ["", "## Per agent — top root causes + where to look", ""]
    for a in A:
        d = D[a]
        L.append(
            f"### `{a}` — {100 * d['failure_rate']:.0f}% fail · "
            f"bottleneck {CAP_LABEL.get(d['bottleneck'], d['bottleneck'])} ({d['scores'][d['bottleneck']]})"
        )
        for et, c in d["primary"].most_common(3):
            inst = d["examples"].get(et, (None,))[0]
            tail = f" — read `{inst}` in {a}.md" if inst else ""
            L.append(f"- **{et}** — {c} cases{tail}")
        L.append("")
    return "\n".join(L) + "\n"
