"""Developer-facing markdown diagnosis report — the framework's actual output:
capability scores (with CI), the dominant error types, representative failing
cases with evidence, and mapped improvement recommendations."""

from __future__ import annotations

from collections import Counter, defaultdict

from . import config
from .outcomes import resolved
from .scoring import agent_capability_scores, strength_composition
from .detect import detect
from .countermeasures import countermeasures_for
from .remedies import select_remedies
from .rootcause import build_blame
from .arbiter import auto_adjudicate
from .trajsignals import tool_behavior

CAP_LABEL = {
    "requirement_understanding": "Requirement Understanding",
    "task_planning": "Task Planning",
    "code_localization": "Code Localization",
    "code_editing": "Code Editing",
    "code_verification": "Code Verification",
    "self_repair_loop": "Self-Repair Loop",
    "tool_use": "Tool Use",
}

# Capabilities scored over EVERY task (denominator = all tasks). Their blame-rates
# approximately decompose the failure rate. The rest (including the two judge caps,
# scored only over judged tasks) are opportunity/availability-gated.
ALL_TASK_CAPS = ("code_localization", "code_editing")


def remedy_lines(cmeasure, match_count):
    """Render current advice independently of scores and diagnostic examples."""
    L = []
    L.append(
        f"- **{cmeasure.summary}** _({cmeasure.intervention_type} · "
        f"{cmeasure.scaffold_layer}; recipe {cmeasure.id}; diagnostic match in {match_count} "
        f"cases, prerequisites unverified; applies when {cmeasure.artifact_applies_when})_"
    )
    if cmeasure.strategies:
        L.append("  - _steps:_ " + " ".join(f"({i + 1}) {s}" for i, s in enumerate(cmeasure.strategies)))
    L.append("  ```")
    for line in cmeasure.artifact_content.splitlines():
        L.append("  " + line)
    L.append("  ```")
    L.append(f"  - _plug in:_ {cmeasure.artifact_plug_in}  ·  _feasibility:_ {cmeasure.feasibility}")
    if cmeasure.example_case:
        L.append(
            f"  - _real case:_ `{cmeasure.example_case.get('instance_id')}` — "
            f"{cmeasure.example_case.get('what_failed', '')}"
        )
    L.append(f"  - _why:_ {cmeasure.mechanism}  ·  _status:_ {cmeasure.status}")
    if cmeasure.calibration:
        L.append("  - _calibration limitations:_ " + str(cmeasure.calibration))
    if cmeasure.contract:
        L.append("  - _gate limits and escape path:_ " + str(cmeasure.contract))
    L.append(
        f"  - _maturity:_ {cmeasure.artifact_maturity}; exact recipe efficacy: "
        f"{cmeasure.evidence_assessment.get('exact_recipe_efficacy', 'unspecified')}"
    )
    if cmeasure.runtime_contract:
        L.append("  - _bounded runtime:_ " + str(cmeasure.runtime_contract))
    if cmeasure.references:
        ref_parts = []
        for r in cmeasure.references:
            part = f"{r.get('title')} ({r.get('id')})"
            part += (
                f" — role: {r.get('evidence_role', 'unspecified')}; "
                f"scope reviewed: {r.get('scope_reviewed', False)}; "
                f"{r.get('study_scope', 'scope unspecified')}"
            )
            ref_parts.append(part)
        L.append(f"  - _refs ({cmeasure.provenance_kind}):_ " + "; ".join(ref_parts))
    return L


def _examples(agent, instances, wanted_errors, per=2):
    found = defaultdict(list)
    selected_counts = Counter()
    adjudicate = auto_adjudicate(agent)
    for inst in instances:
        if not wanted_errors:
            break
        if resolved(agent, inst) is None:
            continue
        d = detect(inst, agent, use_trajectory=True)
        if d["resolved"] is not False:
            continue
        blame = build_blame(d, adjudicate=adjudicate)
        weights = {(b["capability"], b["error_type"]): b["weight"] for b in blame["blame"]}
        for c in d["candidates"]:
            et = c["error_type"]
            weight = weights.get((c["capability"], et), 0)
            if et not in wanted_errors or weight <= 0:
                continue
            choice = select_remedies(
                {"agent": agent, "arbiter": blame.get("arbiter")},
                {"capability": c["capability"], "error_type": et, "detector_evidence": c["evidence"]},
                weight=weight,
            )
            selected_counts.update(r["id"] for r in choice["countermeasures"])
            if len(found[et]) < per:
                ev = ", ".join(f"{k}={v}" for k, v in c["evidence"].items())
                found[et].append((inst, ev[:140]))
    return found, selected_counts


def agent_report_md(agent: str, instances=None) -> str:
    instances = config.require_instances(instances if instances is not None else config.list_instances())
    sc = agent_capability_scores(agent, instances, use_trajectory=True)
    caps = list(sc["capabilities"])

    wanted = {et for c in caps for et, _ in sc["capabilities"][c]["top_errors"]}
    ex, selected_counts = _examples(agent, instances, wanted)

    L = [
        f"# Capability Diagnosis — `{agent}`",
        f"_{config.BENCHMARK_NAME} · {sc['n_tasks']} tasks · "
        f"{100 * sc['failure_rate']:.1f}% failed ({sc['n_failed']})_",
        "",
    ]

    # scores table
    bottleneck = min(caps, key=lambda c: sc["capabilities"][c]["score"])
    L += [
        "## Capability scores",
        "",
        "| Capability | Score (95% CI) | Failure blame-rate | Opportunities |",
        "|---|---|---|---|",
    ]
    for c in caps:
        cc = sc["capabilities"][c]
        lo, hi = cc["score_ci"]
        mark = "  ⬅ **bottleneck**" if c == bottleneck else ""
        L.append(
            f"| {CAP_LABEL.get(c, c)} | {cc['score']:.1f} [{lo:.0f}, {hi:.0f}] "
            f"| {cc['rate']:.3f}{mark} | {cc['opportunity']} |"
        )
    base_sum = sum(sc["capabilities"][c]["rate"] for c in caps if c in ALL_TASK_CAPS)
    L += [
        "",
        f"_Score = 100×(1 − blame-rate). Localization & Editing are scored over all tasks "
        f"and account for {base_sum:.3f} of the {sc['failure_rate']:.3f} failure rate; "
        f"Requirement Understanding, Task Planning, Verification, Self-Repair & Tool Use are "
        f"opportunity/availability-aware (scored only over the tasks that exercised them, or "
        f"that the judge reviewed), so they're conditional and not part of that sum. "
        f"Scores measure attributed blame intensity, not independently measured competence._",
        "",
        f"**Attribution gaps:** {sc['unattributed']} failed runs are unattributed. "
        "The identification bounds below allocate their opportunity-specific mass to each "
        "capability in turn; they describe attribution uncertainty and are separate from bootstrap CIs.",
        "",
        "| Capability | Score identification bounds | Unattributed opportunity mass |",
        "|---|---|---|",
    ]
    for c in caps:
        cc = sc["capabilities"][c]
        lo, hi = cc["score_identification_bounds"]
        L.append(f"| {CAP_LABEL.get(c, c)} | [{lo:.2f}, {hi:.2f}] | {cc['unattributed_opportunity_mass']:.2f} |")
    L.append("")

    # diagnosis trust profile — how much of this agent's blame rests on hard fact
    comp = strength_composition(agent, instances)
    bm = comp["by_mass"]
    L += [
        "## Diagnosis trust profile",
        "",
        f"Of this agent's diagnosed blame mass: **{100 * bm['deductive']:.0f}% deductive** "
        f"(set arithmetic over gold ∩ patch + outcome — reproducible, no model), "
        f"{100 * bm['associational']:.0f}% associational (objective trajectory facts in a "
        f"contributing role), {100 * bm['model_inferred']:.0f}% model-inferred (the "
        f"outcome-blinded LLM judge). The bottleneck RANKING is invariant to the "
        f"uncalibrated blame role-multipliers (`agent_trajviz.awe._engine.calibrate.blame_sensitivity`); "
        f"since v0.32 the attribution COUNTS are checked separately on the full "
        f"trajectory pipeline (`agent_trajviz.awe._engine.calibrate.attribution_sensitivity`) — "
        f"causal-only dominance removes the verification-multiplier instability "
        f"the earlier check could not see.",
        "",
    ]

    # tool-use behavior — a BEHAVIORAL signal (every mechanical tool error, recovered
    # or not), distinct from the tool_failure_not_retried blame above. Most tool
    # errors are recovered, so this captures the rate at which the agent fumbles tool
    # calls (e.g. hallucinated paths) without charging it for failures it fixed.
    tb = tool_behavior(agent, instances)
    if tb["n_tool_calls"]:
        L += [
            "## Tool-use behavior (behavioral — not blamed)",
            "",
            f"Across {tb['n_traj']} trajectories the agent issued "
            f"**{tb['n_tool_calls']:,}** non-test tool calls; "
            f"**{tb['n_tool_call_errors']:,}** errored "
            f"(**{100 * tb['tool_error_rate']:.1f}%**), affecting "
            f"{tb['cases_with_error']} tasks ({100 * tb['case_rate']:.1f}%). A tool error "
            f"becomes a *failure cause* (`tool_failure_not_retried`) only when the agent "
            f"then gave up; most are recovered, so the rate is reported as behavior, not "
            f"blame.",
            "",
        ]
        if tb["subtypes"]:
            L += ["| Tool-error type | count |", "|---|---|"]
            for k, v in sorted(tb["subtypes"].items(), key=lambda kv: -kv[1]):
                L.append(f"| `{k}` | {v} |")
            L.append("")

    # per-capability detail
    for c in caps:
        cc = sc["capabilities"][c]
        if not cc["top_errors"]:
            continue
        L += [f"## {CAP_LABEL.get(c, c)} — score {cc['score']:.1f}", ""]
        for et, mass in cc["top_errors"]:
            L.append(f"### `{et}`  _(blame mass {mass:.1f})_")
            # Single source: the grounded catalog. Strategies render once, inside each
            # remedy block they belong to — no separate, duplicate "Fix:" summary line.
            for cmeasure in countermeasures_for(error_type=et):
                if not selected_counts[cmeasure.id]:
                    continue
                L.extend(remedy_lines(cmeasure, selected_counts[cmeasure.id]))
            for inst, ev in ex.get(et, []):
                L.append(f"- `{inst}` — {ev}")
            L.append("")

    # ranked focus
    ranked = sorted(((sc["capabilities"][c]["rate"], c) for c in caps), reverse=True)
    L += [
        "## Diagnostic focus (ranked by blame-rate)",
        "",
        "These are conditional diagnostic intensities, not expected intervention benefits. "
        "Consider case-selected remedies as experiments; deployment prerequisites require checking.",
        "",
    ]
    for i, (rate, c) in enumerate(ranked, 1):
        top = sc["capabilities"][c]["top_errors"]
        lead = top[0][0] if top else "—"
        lead_cms = [e for e in countermeasures_for(error_type=lead) if selected_counts[e.id]]
        hint = (
            "; ".join(e.summary for e in lead_cms)
            if lead_cms
            else "no selected remedy; inspect evidence gaps"
            if top
            else "no supported fault mass; no remedy indicated"
        )
        L.append(f"{i}. **{CAP_LABEL.get(c, c)}** (rate {rate:.3f}) — start with `{lead}`: {hint}")
    return "\n".join(L) + "\n"
