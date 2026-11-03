"""Comprehensive per-fault diagnosis dossier.

For each failed run, assemble what an agent developer needs to pinpoint the
problem: the fault LABEL (capability → error_type), the TASK, the CODE evidence
(the gold code the agent missed, or its actual change vs the reference), and the
agent's THINKING PROCESS (trajectory reasoning + the decisive tool actions)."""

from __future__ import annotations

import json
import re
from urllib.parse import quote
from concurrent.futures import ProcessPoolExecutor

from . import config
from .gold import build_gold_reference
from .outcomes import resolved, load_pertest
from .detect import detect
from .rootcause import build_blame
from .arbiter import auto_adjudicate
from .faultside import assess as fault_side_assess
from .adapters import load_normalized
from .eventledger import event_ledger
from .symptoms import analyze as analyze_symptoms
from .symptom_exports import compact as compact_symptoms
from .execution_context import for_case
from .remedies import select_remedies
from .audit import audit_fault
from .report import CAP_LABEL
from .evidence import build_chain, changed_lines
from .trajectory import SEARCH, READ, EDIT, WRITE, RUN_TEST, RUN_COMMAND, PLAN

_GIT = re.compile(r"diff --git a/(.+?) b/(.+)$")
_ACTIONS = (SEARCH, READ, EDIT, WRITE, RUN_TEST, RUN_COMMAND)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _clean(text: str) -> str:
    """Strip GitHub-issue HTML boilerplate and collapse blank lines."""
    t = _HTML_COMMENT.sub("", text or "")
    t = re.sub(r"[ \t]+\n", "\n", t)  # drop trailing whitespace (else blank lines survive)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def file_hunks(diff_text: str, max_lines: int = 28, max_hunks: int = 2) -> dict:
    """{file: [hunk_text, ...]} from a unified diff (the code that changed)."""
    files: dict = {}
    cur = None
    hunk = None
    for line in (diff_text or "").splitlines():
        m = _GIT.match(line)
        if m:
            cur = m.group(2)
            files.setdefault(cur, [])
            hunk = None
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                cur = p[2:] if p.startswith("b/") else p
                files.setdefault(cur, [])
            continue
        if line.startswith("@@"):
            hunk = [line]
            if cur is not None and len(files.get(cur, [])) < max_hunks:
                files.setdefault(cur, []).append(hunk)
            continue
        if hunk is not None and line[:1] in "+- " and not line.startswith(("+++", "---")):
            if len(hunk) < max_lines:
                hunk.append(line)
    return {f: ["\n".join(h) for h in hs] for f, hs in files.items() if hs}


def thinking_excerpt(traj, max_steps: int = 22) -> list[dict]:
    """Trajectory reasoning + decisive actions — the agent's thought process."""
    rows = []
    for position, s in enumerate(traj.steps):
        previous = len(rows)
        if s.type == "reasoning" and s.text:
            rows.append({"step": s.step_id, "kind": "thinks", "text": s.text[:240], "observability": s.observability})
        elif s.type in ("message", "final_answer") and s.actor == "agent" and s.text:
            rows.append({"step": s.step_id, "kind": "says", "text": s.text[:240]})
        elif s.type == "tool_call" and s.canonical_action == PLAN:
            items = (s.args or {}).get("items") or []
            rows.append(
                {
                    "step": s.step_id,
                    "kind": "plans",
                    "tool": s.native_tool,
                    "text": "; ".join(
                        f"[{it.get('status')}] {str(it.get('content'))[:60]}" for it in items if isinstance(it, dict)
                    )[:480],
                }
            )
        elif s.type == "tool_call" and s.canonical_action in _ACTIONS:
            a = {k: v for k, v in (s.args or {}).items() if k in ("query", "pattern", "command", "path") and v}
            rows.append(
                {"step": s.step_id, "kind": s.canonical_action, "tool": s.native_tool, "files": s.files, "args": a}
            )
        if len(rows) > previous:
            rows[-1]["source_ref"] = {
                "position": position,
                "step_id": s.step_id,
                "thread": s.subagent_id,
                "call_id": s.call_id,
            }
    if len(rows) > max_steps:
        head, tail = rows[: max_steps * 2 // 3], rows[-max_steps // 3 :]
        rows = head + [{"step": "…", "kind": f"({len(rows) - len(head) - len(tail)} steps omitted)"}] + tail
    return rows


def _gold_patch_text(inst: str) -> str:
    try:
        return json.loads(config.requirements_path(inst).read_text(encoding="utf-8")).get("gold_patch", "")
    except Exception:
        return ""


def _agent_patch_text(agent: str, inst: str) -> str:
    p = config.patch_path(agent, inst)
    return p.read_text(errors="replace", encoding="utf-8") if p.exists() else ""


def blame_status_note(status: str) -> str:
    """Human-readable gloss for a case's blame status — so a 'conjunctive' case
    (no single dominant cause; or a primary the arbiter refuted) is explicit rather
    than implied only by per-fault weights."""
    return {
        "primary": "a single dominant cause was identified",
        "conjunctive": "conjunctive — supported candidates remain, no eligible dominant cause",
        "unattributed": "unattributed — no supported attribution remains (coverage gap)",
    }.get(status, status)


def arbiter_note(arb: dict | None) -> str:
    """One-line gloss of the LLM arbiter's verdict on a non-deductive primary, for
    the case Diagnosis line. Empty when the arbiter was not consulted."""
    if not arb:
        return ""
    tgt = f"`{arb.get('target', {}).get('error_type', '?')}`"
    conf = arb.get("confidence", "?")
    if arb.get("applied") == "unattributed_after_refute":
        return f" — arbiter **refuted** the only supported attribution {tgt} ({conf}); failure remains unattributed"
    if arb.get("applied") == "demoted_to_conjunctive":
        why = arb.get("rationale", "")
        return f" — arbiter **refuted** the candidate primary {tgt} ({conf})" + (f": {why}" if why else "")
    if arb.get("applied") == "refuted_reassigned":
        why = arb.get("rationale", "")
        return (
            f" — arbiter **refuted** the candidate primary {tgt} ({conf}); "
            "blame reassigned to the surviving deductive fault" + (f": {why}" if why else "")
        )
    if arb.get("applied") == "corroborated":
        return f" — arbiter **corroborated** the primary {tgt} ({conf})"
    return f" — arbiter {arb.get('support', '?')} ({conf}) on {tgt}"


def fix_label(is_primary: bool) -> str:
    """Gate the prescriptive emphasis to the fault's causal role: the primary cause
    gets the authoritative fix; a contributing/secondary fault's remedy is flagged as
    such so it is not read as THE fix."""
    # "secondary" not "contributing": the blame model reserves *contributing*
    # for the verification/self-repair/tool capabilities; a non-primary
    # localization fault is secondary, not "contributing" in that sense.
    return "Recommended fix (primary cause)" if is_primary else "Secondary-fault fix"


def _arbiter_support_for(arb: dict | None, fault: dict) -> str:
    """The arbiter verdict belongs on its TARGET fault's row only (capability
    matched when the target names one; error_type always)."""
    if not arb:
        return ""
    tgt = arb.get("target") or {}
    if tgt.get("error_type") != fault["error_type"]:
        return ""
    if tgt.get("capability") and tgt["capability"] != fault["capability"]:
        return ""
    return arb.get("support", "")


def case_record(agent: str, inst: str) -> dict | None:
    d = detect(inst, agent, use_trajectory=True)
    if d["resolved"] is not False:
        return None
    # Computed once for both return paths (identical inputs either way).
    fault_side = fault_side_assess(d.get("signals"))
    traj = load_normalized(agent, inst)
    ledger = event_ledger(traj)
    symptom_analysis = compact_symptoms(analyze_symptoms(traj, _agent_patch_text(agent, inst)))
    symptom_analysis["patch_available"] = config.patch_path(agent, inst).exists()
    if not d["candidates"]:
        # A failed case with NO candidate error is the coverage gap the
        # 'unattributed' status names — it must be VISIBLE in the dossier
        # artifact, not silently dropped (it previously never produced a record,
        # so the gap the status note documents could not be counted). The record
        # carries every key the renderers read unconditionally (task / repo /
        # reasoning_observability / thinking_process) so to_markdown does not
        # KeyError on it.
        g = build_gold_reference(inst)
        # fault_side matters MOST here: an unattributed failed case that was
        # truncated is the informative non-model shape (no candidate to blame
        # because the scaffold killed the run).
        return {
            "case_id": f"{agent}__{inst}",
            "agent": agent,
            "instance_id": inst,
            "repo": g.repo,
            **for_case(agent, inst, getattr(g, "benchmark_profile", {})),
            "blame_status": "unattributed",
            "attribution_trace": build_blame({**d, "loc_flag": d.get("loc_flag", "clean")})["decision_trace"],
            "blame_status_note": blame_status_note("unattributed"),
            "fault_side": fault_side,
            "task": {
                "problem_statement": _clean(g.problem_statement),
                "gold_files": g.gold_files,
                "multi_file": g.is_multi_file,
                "cross_module": g.is_cross_module,
            },
            "reasoning_observability": "n/a (no fault diagnosed)",
            "thinking_process": [],
            "event_ledger": ledger,
            "symptom_analysis": symptom_analysis,
            "faults": [],
        }
    gold = build_gold_reference(inst)
    blame = build_blame(d, adjudicate=auto_adjudicate(agent))
    primary = blame.get("primary")
    weights = {(b["capability"], b["error_type"]): b["weight"] for b in blame["blame"]}
    gold_text = _gold_patch_text(inst)
    agent_text = _agent_patch_text(agent, inst)
    gold_h = file_hunks(gold_text)
    agent_h = file_hunks(agent_text)
    gold_changed = changed_lines(gold_text)
    agent_changed = changed_lines(agent_text)
    pertest = load_pertest(agent, inst)  # eval+ tier raw input (v0.39), may be None
    # IG-style verbatim quotes: step_id -> trace text, so judge-cited steps in the
    # evidence chain carry the exact trajectory text (trace-quote + mechanism).
    step_text = (
        {s.step_id: (s.text or s.output_excerpt or "")[:200] for s in traj.steps if (s.text or s.output_excerpt)}
        if traj
        else None
    )

    faults = []
    for c in d["candidates"]:
        cap, et = c["capability"], c["error_type"]
        weight = weights.get((cap, et), 0.0)
        is_primary = bool(primary and primary["capability"] == cap and primary["error_type"] == et)
        code: dict = {}
        if cap == "code_localization":
            missed = c["evidence"].get("missed_gold") or gold.gold_files
            code["code_that_needed_changing_but_was_missed"] = {f: gold_h[f] for f in missed if f in gold_h}
        elif cap == "code_editing":
            # side-by-side: the agent's actual change vs the reference fix, same files
            code["what_the_agent_changed"] = agent_h
            code["reference_fix"] = gold_h
        faults.append(
            {
                "label": f"{CAP_LABEL.get(cap, cap)} → {et}",
                "capability": cap,
                "error_type": et,
                "confidence": c["confidence"],
                "blame_weight": round(weight, 2),
                "is_primary": is_primary,
                "detector_evidence": c["evidence"],
                "code_evidence": code,
                "evidence_chain": build_chain(
                    c, d, gold, gold_changed, agent_changed, weight, is_primary, step_text=step_text
                ),
                # Single source: the grounded catalog. The compact strategy list and the
                # rich countermeasure blocks below are two views of the SAME entries
                # (fetched once) — no second, redundant query via the deprecated shim.
                # Full input set: test_files + problem_statement are required by the
                # gating-test and solved-unrelated auditors — omitting them embedded
                # `unverifiable` for all 328 such faults while §7's batch re-audit
                # (which loads them) re-derived the same faults; the two surfaces
                # must agree (v0.31).
                "audit": (lambda a: {"verdict": a["verdict"], "mismatches": a["mismatches"]})(
                    audit_fault(
                        {
                            "capability": cap,
                            "error_type": et,
                            "confidence": c["confidence"],
                            "detector_evidence": c["evidence"],
                        },
                        agent=agent,
                        inst=inst,
                        gold_text=gold_text,
                        agent_text=agent_text,
                        resolved=d["resolved"],
                        traj=traj,
                        test_files=gold.test_patch_files,
                        problem_statement=gold.problem_statement,
                        fail_to_pass=gold.fail_to_pass,
                        pertest=pertest,
                        profile=gold.benchmark_profile,
                    )
                ),
            }
        )
        # Exact attribution is checked before the displayed weight is rounded.
        faults[-1].update(select_remedies({"agent": agent, "arbiter": blame.get("arbiter")}, faults[-1], weight=weight))
    faults.sort(key=lambda f: -f["blame_weight"])

    return {
        "case_id": f"{agent}__{inst}",
        "agent": agent,
        "instance_id": inst,
        "repo": gold.repo,
        **for_case(agent, inst, getattr(gold, "benchmark_profile", {})),
        "blame_status": blame["status"],
        "attribution_trace": blame["decision_trace"],
        "fault_side": fault_side,
        "arbiter": blame.get("arbiter"),
        "task": {
            "problem_statement": _clean(gold.problem_statement),
            "gold_files": gold.gold_files,
            "multi_file": gold.is_multi_file,
            "cross_module": gold.is_cross_module,
        },
        "reasoning_observability": (
            "partly hidden (encrypted)" if traj and any(s.observability == "hidden" for s in traj.steps) else "observed"
        ),
        "thinking_process": thinking_excerpt(traj) if traj else [],
        "event_ledger": ledger,
        "symptom_analysis": symptom_analysis,
        "faults": faults,
    }


def _collect_agent(payload) -> list[dict]:
    agent, instances = payload
    out = []
    for inst in instances:
        # In aggregate exports most namespaced instances belong to another
        # agent.  Filter those before case_record parses their large evidence
        # payloads.  Keep case_record's standalone/mocked behavior unchanged.
        if resolved(agent, inst) is not False:
            continue
        try:
            r = case_record(agent, inst)
        except Exception:
            r = None
        if r:
            out.append(r)
    return out


def collect(agents, instances, workers: int = 1) -> list[dict]:
    instances = config.require_instances(instances)
    payloads = [(agent, instances) for agent in agents]
    if workers <= 1 or len(payloads) <= 1:
        return [case for payload in payloads for case in _collect_agent(payload)]
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
        chunks = pool.map(_collect_agent, payloads)
        return [case for chunk in chunks for case in chunk]


# ----------------------------------------------------------------- renderers


def to_markdown(cases: list[dict], max_cases: int = 30) -> str:
    from collections import Counter

    # cluster cases by their dominant fault so same faults read together
    cases = sorted(
        cases, key=lambda c: (c["faults"][0]["capability"], c["faults"][0]["error_type"]) if c["faults"] else ("", "")
    )
    counts = Counter(f["label"] for c in cases for f in c["faults"])
    L = [
        "# Agent Fault Dossier",
        f"_{len(cases)} failed cases with diagnosed faults (showing up to {max_cases}; full data in faults.json)_",
        "",
        "[Open the numbered attribution graphs](ATTRIBUTION.html). "
        "They show evidence and policy decisions, not experimentally proven causes.",
        "",
        "## Fault summary",
        "",
        "| Fault | Count |",
        "|---|---|",
    ]
    L += [f"| {lab} | {n} |" for lab, n in counts.most_common()]
    L.append("")
    for case in cases[:max_cases]:
        t = case["task"]
        ps = t["problem_statement"]
        L += [
            f"## `{case['case_id']}` — {case['repo']}",
            "",
            f"[Follow this case's numbered attribution DAG](ATTRIBUTION.html#case={quote(case['case_id'], safe='')})",
            "",
            f"**Task:** {ps[:600]}" + ("…" if len(ps) > 600 else ""),
            "",
            f"**Gold files:** `{', '.join(t['gold_files']) or '—'}` · "
            f"multi_file={t['multi_file']} cross_module={t['cross_module']} · "
            f"reasoning {case['reasoning_observability']}",
            "",
            f"**Diagnosis:** {blame_status_note(case['blame_status'])}{arbiter_note(case.get('arbiter'))}",
            "",
        ]
        # v0.33: the repair-locus attribute must reach the manual-review
        # surfaces, not just faults.json — a grader-suspect marking invisible
        # in the rendered dossier defeats its purpose. Non-model sides carry
        # their basis; the model default stays a single unobtrusive line.
        fsd = case.get("fault_side")
        if fsd:
            line = f"**Fault side:** {fsd['side']} ({fsd['certainty']})"
            if fsd["side"] != "model":
                line += f" — {fsd['basis']}"
            L += [line, ""]
        analysis = case.get("symptom_analysis") or {}
        symptoms = analysis.get("symptoms") or []
        if symptoms:
            L += ["**Behavioral symptoms (unscored review evidence):**", ""]
            for symptom in symptoms:
                ev = symptom.get("evidence") or {}
                anchor = ev.get("file") or str((symptom.get("call") or {}).get("step_id", "?"))
                L.append(
                    f"- `{symptom['kind']}` at `{anchor}`; {symptom['lifecycle']}; "
                    "terminal relevance unknown; causality not tested."
                )
            L.append("")
        ledger = case.get("event_ledger") or {}
        if ledger.get("available"):
            counts = ledger.get("counts") or {}
            L.append(
                f"**Observed failure episodes:** {len(ledger.get('events', []))}; "
                f"later success on the same action: {counts.get('later_success_observed', 0)}; "
                f"no recovery observed: {counts.get('no_recovery_observed', 0)}; "
                f"missing tool results: {ledger.get('missing_result_calls', 0)}. "
                "Terminal relevance is unknown; complete links are in faults.json."
            )
        for f in case["faults"]:
            tag = " · **PRIMARY**" if f["is_primary"] else ""
            L.append(f"### {f['label']} _(blame {f['blame_weight']}, {f['confidence']} conf{tag})_")
            _av = (f.get("audit") or {}).get("verdict")
            if _av:
                _badge = {
                    "re_derived": "✓ independently re-derived from raw inputs",
                    "structural_ok": "✓ structural audit (judge label)",
                    "unverifiable": "— unverifiable (input unavailable)",
                    "mismatch": "⚠ AUDIT MISMATCH",
                }.get(_av, _av)
                L.append(f"_audit: {_badge}_")
            ev = ", ".join(f"{k}={v}" for k, v in f["detector_evidence"].items())
            if ev:
                L.append(f"**Why flagged:** {ev[:300]}")
            chain = f.get("evidence_chain")
            if chain:
                _ICON = {"observation": "•", "inference": "↳", "conclusion": "⇒"}
                L.append(f"**Evidence chain** _(strength: {chain['strength']})_ — {chain['strength_basis']}")
                causal = (chain.get("causal_support") or {}).get(
                    "status", (chain.get("counterfactual") or {}).get("status", "not_recorded")
                )
                L.append(
                    f"**Causal validation:** {causal.replace('_', ' ')} "
                    "(observation strength does not establish cause)."
                )
                for lk in chain["links"]:
                    L.append(f"  {_ICON.get(lk['kind'], '•')} _{lk['kind']}_ ({lk['source']}): {lk['statement']}")
                cf = chain.get("counterfactual") or {}
                if cf.get("what_would_confirm"):
                    L.append(f"  ⊕ _counterfactual ({cf.get('status', 'not_tested')})_: {cf['what_would_confirm']}")
            for grp, fh in (f["code_evidence"] or {}).items():
                if not fh:
                    continue
                L.append(f"**{grp.replace('_', ' ')}:**")
                for fname, hunks in list(fh.items())[:3]:
                    L += [f"`{fname}`", "```diff", hunks[0], "```"]
            if f.get("recommendation"):
                L += [f"**{fix_label(f['is_primary'])} (proposed experiment):** " + "; ".join(f["recommendation"]), ""]
            for recipe in f.get("countermeasures", []):
                L.append(
                    f"_Remedy `{recipe['id']}`: {recipe.get('status', 'unspecified')}; "
                    f"implementation: {recipe.get('artifact_maturity', 'unspecified')}; "
                    f"exact-recipe efficacy: {(recipe.get('evidence_assessment') or {}).get('exact_recipe_efficacy', 'unspecified')}._"
                )
            for decision in f.get("countermeasure_decisions", []):
                if decision["decision"] != "selected":
                    L.append(f"_Remedy `{decision['id']}`: {decision['decision']} ({decision['reason']})._")
            L.append("")
        L.append("**Agent thinking (trajectory excerpt):**")
        for r in case["thinking_process"]:
            if r["kind"] == "thinks":
                L.append(f"- _[{r['step']}] thinks:_ {r['text']}")
            elif r["kind"] == "says":
                L.append(f"- _[{r['step']}] says:_ {r['text']}")
            elif r["kind"] in _ACTIONS:
                detail = r.get("files") or r.get("args") or ""
                L.append(f"- `[{r['step']}] {r['kind']}` {r.get('tool', '')} {detail}")
            else:
                L.append(f"- _[{r['step']}] {r['kind']}_")
        L += ["", "---", ""]
    return "\n".join(L) + "\n"


CSV_COLUMNS = (
    "case_id",
    "agent",
    "instance_id",
    "repo",
    "capability",
    "error_type",
    "confidence",
    "blame_weight",
    "evidence_strength",
    "observation_strength",
    "causal_validation",
    "fault_side",
    "fault_side_certainty",
    "reproduced_fraction",
    "f2p_pass_fraction",
    "is_primary",
    "n_gold_files",
    "task_summary",
    "arbiter_support",
)


def to_csv_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for f in case["faults"]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "agent": case["agent"],
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    "capability": f["capability"],
                    "error_type": f["error_type"],
                    "confidence": f["confidence"],
                    "blame_weight": f["blame_weight"],
                    "evidence_strength": (f.get("evidence_chain") or {}).get("strength", ""),
                    "observation_strength": (f.get("evidence_chain") or {}).get(
                        "observation_strength", (f.get("evidence_chain") or {}).get("strength", "")
                    ),
                    "causal_validation": ((f.get("evidence_chain") or {}).get("causal_support") or {}).get(
                        "status",
                        ((f.get("evidence_chain") or {}).get("counterfactual") or {}).get("status", "not_recorded"),
                    ),
                    # v0.33 case-level repair locus (same value on every row of a case).
                    "fault_side": (case.get("fault_side") or {}).get("side", ""),
                    "fault_side_certainty": (case.get("fault_side") or {}).get("certainty", ""),
                    # graded severity for incorrect_patch: share of the reference change reproduced
                    # (deductive, 0..1; blank when not an editing-on-right-files fault).
                    "reproduced_fraction": (f.get("detector_evidence") or {}).get("reproduced_fraction", ""),
                    # eval+ tier (v0.39): FAIL_TO_PASS pass fraction on partial_fix_only /
                    # regression_introduced rows (blank when no per-test record).
                    "f2p_pass_fraction": (f.get("detector_evidence") or {}).get("f2p_pass_fraction", ""),
                    "is_primary": f["is_primary"],
                    "n_gold_files": len(case["task"]["gold_files"]),
                    # context label only (the issue text) — NOT the evidence. The evidence is
                    # `evidence_strength` here + the full evidence_chain in faults.json.
                    "task_summary": case["task"]["problem_statement"][:160].replace("\n", " "),
                    # The arbiter adjudicates ONE fault (its target). Stamping the
                    # case verdict on every row made 32 deductive rows read as
                    # adjudicated in the CSV — violating deductive_never_adjudicated
                    # for CSV-only consumers (v0.31). Stamp the target row only.
                    "arbiter_support": _arbiter_support_for(case.get("arbiter"), f),
                }
            )
    return rows
