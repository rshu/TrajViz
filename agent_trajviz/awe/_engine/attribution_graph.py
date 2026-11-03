"""Numbered provenance DAGs over persisted diagnoses, not causal discovery.

Edges describe documented inputs, assessments and decisions. Observation bundles
preserve the existing chain without inventing dependencies between its statements.
Protocol/recovery episodes form a separate graph, never an inferred causal chain.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

from . import config
from .remedies import fault_support
from .remedy_plan import cases_digest
from .evidence import ASSOCIATIONAL, DEDUCTIVE, MODEL_INFERRED, strength_of

TEMPLATE = Path(__file__).parent / "templates" / "attribution.html"
RELATIONS = {
    "documents",
    "assessed_using",
    "emits_candidate",
    "reviews_candidate",
    "contributes_to_policy",
    "updates_attribution",
    "selects",
    "proposes_check",
    "tool_result",
    "same_action_later_success",
}
# Existing normalized evidence-chain aliases, independent of native log format.
REFERENCE_SOURCES = {
    "gold_patch",
    "gold_tests",
    "reference_patch",
    "ground_truth",
    "agent_patch ∩ gold",
    "agent_patch − gold",
    "agent_patch ∩ gold_tests",
    "diff(agent) vs diff(gold)",
    "trajectory:read_set ∩ gold − edit_set",
}


def validate_dag(nodes, edges):
    """Reject duplicate IDs, dangling links, cycles, and untyped relations."""
    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate DAG node IDs")
    children, incoming = {i: [] for i in ids}, dict.fromkeys(ids, 0)
    pairs = set()
    for edge in edges:
        a, b = edge["from"], edge["to"]
        if a not in children or b not in children:
            raise ValueError("DAG edge references a missing node")
        if edge["relation"] not in RELATIONS:
            raise ValueError("Unknown DAG edge relation")
        key = a, b, edge["relation"]
        if key in pairs:
            raise ValueError("Duplicate DAG edge")
        pairs.add(key)
        children[a].append(b)
        incoming[b] += 1
    pending, visited = deque(i for i in ids if incoming[i] == 0), 0
    while pending:
        visited += 1
        for child in children[pending.popleft()]:
            incoming[child] -= 1
            if incoming[child] == 0:
                pending.append(child)
    if visited != len(ids):
        raise ValueError("DAG contains a cycle")


def _source(pointer):
    return {"artifact": "faults.json", "json_pointer": pointer}


def _summary_evidence(fault):
    evidence = fault.get("detector_evidence") or {}
    # Keep official outcomes prominent, then accept arbitrary adapter metrics.
    # Missing values remain missing; no absent F2P/P2P result is imputed as zero.
    keys = [k for k in ("f2p_pass_fraction", "n_p2p_broken", "p2p_broken") if k in evidence]
    keys += [k for k in evidence if k not in keys]
    lines = [f"{k}: {json.dumps(evidence[k], ensure_ascii=False)}" for k in keys[:3]]
    if len(keys) > 3:
        remaining = keys[3:]
        names = ", ".join(remaining[:6]) + (f", +{len(remaining) - 6} more" if len(remaining) > 6 else "")
        lines.append("Additional recorded fields (open details): " + names)
    return lines or ["No structured detector evidence recorded."]


def _assessment_origin(fault):
    """Display evidence origin, never infer an LLM call from an unknown taxonomy.

    Adapters may set evidence_chain.assessment_origin to objective/llm/unknown.
    Conflicting explicit origin and recorded judge metadata stay unknown.
    Legacy built-in candidates retain their existing classification.
    """
    evidence = fault.get("detector_evidence") or {}
    chain = fault.get("evidence_chain") or {}
    judged = bool(evidence.get("judge_model") or evidence.get("judge_rationale"))
    explicit = chain.get("assessment_origin")
    if explicit is not None:
        if explicit not in ("objective", "llm", "unknown") or (explicit == "objective" and judged):
            return "unknown"
        return explicit
    if judged:
        return "llm"
    strength = chain.get("observation_strength", chain.get("strength"))
    if strength in (DEDUCTIVE, ASSOCIATIONAL):
        return "objective"
    # Reuse the shared taxonomy rather than maintaining a renderer-specific list.
    legacy_strength = strength_of(fault.get("capability"), fault.get("error_type"), default=None)
    if legacy_strength is None:
        return "unknown"
    return "llm" if legacy_strength == MODEL_INFERRED else "objective"


def _action_graph(case, base):
    ledger = case.get("event_ledger") or {}
    nodes, edges, known, pairs, episodes = [], [], {}, set(), []
    for index, episode in enumerate(ledger.get("events") or []):
        prefix = f"{base}/event_ledger/events/{index}"
        episodes.append(
            {k: v for k, v in episode.items() if k not in ("links", "failures", "latest_failure", "recovery")}
        )
        for j, link in enumerate(episode.get("links") or []):
            refs = []
            for end in ("from", "to"):
                ref = link[end]
                # Position disambiguates reused call IDs and cross-thread IDs.
                position = ref.get("position")
                if type(position) is not int or position < 0:
                    raise ValueError("Action link is missing its original position")
                identity = f"event:{position}:{json.dumps(ref.get('thread'), ensure_ascii=False)}"
                refs.append(identity)
                if identity in known and known[identity] != ref:
                    raise ValueError("Conflicting original event references")
                if identity not in known:
                    known[identity] = ref
                    result = end == "to" and link["relation"] == "tool_result"
                    observation = next((f for f in episode.get("failures", []) if f.get("result") == ref), None)
                    if observation is None and (episode.get("recovery") or {}).get("result") == ref:
                        observation = episode["recovery"]
                    observation = (
                        {k: v for k, v in observation.items() if k not in ("call", "result")} if observation else None
                    )
                    nodes.append(
                        {
                            "id": identity,
                            "position": position,
                            "kind": "tool_result" if result else "tool_call",
                            "title": f"Original step {ref.get('step_id')} · "
                            + ("result" if result else episode.get("action", "action")),
                            "reference": ref,
                            "episode_index": index,
                            "observation": observation,
                            "source": _source(prefix + f"/links/{j}/{end}"),
                        }
                    )
            if link["from"]["position"] >= link["to"]["position"]:
                raise ValueError("Action edge does not follow recorded event order")
            key = tuple(refs) + (link["relation"],)
            if key not in pairs:
                pairs.add(key)
                edges.append(
                    {
                        "from": refs[0],
                        "to": refs[1],
                        "relation": link["relation"],
                        "basis": link.get("basis", "recorded_ledger_link"),
                        "causal": False,
                    }
                )
    validate_dag(nodes, edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "episodes": episodes,
        "available": bool(ledger.get("available")),
        "missing_result_calls": ledger.get("missing_result_calls", 0),
        "scope": "Observed failure/recovery episodes; not the complete trajectory or semantic dependencies.",
    }


def case_graph(case, index):
    base, nodes, edges = f"/{index}", [], []

    def node(identity, phase, number, title, provenance, lines, details, pointer, fault=None, status="recorded"):
        nodes.append(
            {
                "id": identity,
                "phase": phase,
                "number": number,
                "title": title,
                "provenance": provenance,
                "status": status,
                "lines": lines,
                "details": details,
                "source": _source(pointer),
                "fault_id": fault,
            }
        )
        return identity

    def edge(a, b, relation):
        edges.append({"from": a, "to": b, "relation": relation, "causal": False})

    task = case.get("task") or {}
    node(
        "task",
        1,
        "1.1",
        "Task and stated requirements",
        "input",
        [(task.get("problem_statement") or "Task text unavailable.")[:240]],
        {"problem_statement": task.get("problem_statement")},
        base + "/task/problem_statement",
    )
    has_objective = any(_assessment_origin(f) == "objective" for f in case.get("faults", []))
    if has_objective:
        node(
            "reference",
            1,
            "1.2",
            "Post-hoc reference context",
            "input",
            [
                "Reference files: " + (", ".join(task.get("gold_files") or []) or "unavailable"),
                "Not supplied to the outcome-blinded process judge.",
            ],
            {"gold_files": task.get("gold_files")},
            base + "/task/gold_files",
        )
    trace = case.get("attribution_trace") or {}
    if trace:
        actual_primary = [
            {"capability": f["capability"], "error_type": f["error_type"]}
            for f in case.get("faults", [])
            if f.get("is_primary")
        ]
        selected = trace.get("selection") or {}
        if (
            trace.get("schema_version") != 1
            or selected.get("status") != case.get("blame_status")
            or actual_primary != ([selected["primary"]] if selected.get("primary") else [])
        ):
            raise ValueError("Recorded decision trace contradicts the dossier")
    rows = {(r["capability"], r["error_type"]): r for r in trace.get("candidates", [])}
    if trace:
        if set(rows) != {(f["capability"], f["error_type"]) for f in case.get("faults", [])}:
            raise ValueError("Decision trace candidate set differs from the dossier")
        for fault in case.get("faults", []):
            weight = rows[(fault["capability"], fault["error_type"])]["final_weight"]
            if (
                type(weight) not in (int, float)
                or not math.isfinite(weight)
                or weight < 0
                or not math.isclose(round(weight, 2), fault["blame_weight"], abs_tol=1e-12)
            ):
                raise ValueError("Decision trace weights contradict the dossier")
    fault_ids, support_states, primary_ids = [], {}, []
    for i, fault in enumerate(case.get("faults") or []):
        fid, prefix = f"fault:{i}", base + f"/faults/{i}"
        fault_ids.append(fid)
        support = fault_support(case, fault)
        support_states[fid] = support
        if fault.get("is_primary"):
            primary_ids.append(fid)
        chain = fault.get("evidence_chain") or {}
        links = chain.get("links") or []
        origin = _assessment_origin(fault)
        judge = origin == "llm"
        observations = [link for link in links if link.get("kind") == "observation"]
        assessments = [link for link in links if link.get("kind") != "observation"]
        observation_details = {
            "detector_evidence": fault.get("detector_evidence") or {},
            "assessment_origin": origin,
            "observations": observations,
            "code_evidence": fault.get("code_evidence") or {},
            "audit": fault.get("audit"),
            "source_refs": [
                _source(prefix + "/detector_evidence"),
                _source(prefix + "/evidence_chain/links"),
                _source(prefix + "/code_evidence"),
                _source(prefix + "/audit"),
            ],
            "source_note": "Aggregated artifact facts cite artifacts. Step-only citations do not imply a known thread.",
        }
        inputs = node(
            fid + ":evidence",
            2,
            f"2.{i + 1}",
            "Cited process evidence" if judge else "Recorded evidence",
            "llm" if judge else "observation" if origin == "objective" else "unknown",
            _summary_evidence(fault) + [o.get("statement", "") for o in observations[:2]],
            observation_details,
            prefix + "/detector_evidence",
            fid,
        )
        edge("task", inputs, "documents")
        # Reference context is not a premise of every objective runtime signal.
        cites_reference = any(
            link.get("source_role") == "reference"
            or (isinstance(link.get("source"), str) and link["source"] in REFERENCE_SOURCES)
            for link in links
        )
        compares_reference = bool({"missed_gold", "region_overlap"} & set(fault.get("detector_evidence") or {}))
        if origin == "objective" and (cites_reference or compares_reference):
            edge("reference", inputs, "documents")
        assessment = node(
            fid + ":assessment",
            3,
            f"3.{i + 1}",
            "LLM assessment"
            if judge
            else "Detector assessment"
            if origin == "objective"
            else "Assessment origin unavailable",
            "llm" if judge else "inference" if origin == "objective" else "unknown",
            [link.get("statement", "") for link in assessments][:2] or ["No assessment statement recorded."],
            {
                "statements": assessments,
                "confidence": fault.get("confidence"),
                "observation_strength": chain.get("observation_strength", chain.get("strength")),
                "assessment_origin": origin,
                "recorded_assessment_origin": chain.get("assessment_origin"),
                "strength_basis": chain.get("strength_basis"),
                "causal_support": chain.get("causal_support"),
            },
            prefix + "/evidence_chain",
            fid,
        )
        edge(inputs, assessment, "assessed_using")
        row = rows.get((fault.get("capability"), fault.get("error_type")))
        candidate = node(
            fid,
            4,
            f"4.{i + 1}",
            fault.get("error_type", "Unknown fault"),
            "inference",
            [
                fault.get("capability", "Unknown capability"),
                "Recorded final weight: " + str(row["final_weight"])
                if row
                else f"Displayed weight: {fault.get('blame_weight')} (rounded)",
                "Final support: " + support,
            ],
            {
                "capability": fault.get("capability"),
                "error_type": fault.get("error_type"),
                "confidence": fault.get("confidence"),
                "displayed_weight": fault.get("blame_weight"),
                "is_primary": fault.get("is_primary"),
                "audit": fault.get("audit"),
                "decision_trace": row,
            },
            prefix,
            fid,
            support,
        )
        edge(assessment, candidate, "emits_candidate")

    arbiter = trace.get("adjudication") or case.get("arbiter")
    if arbiter:
        target = arbiter.get("target") or {}
        arb = node(
            "arbiter",
            5,
            "5",
            "LLM arbiter review",
            "llm",
            [
                f"Verdict: {arbiter.get('support')} · {arbiter.get('confidence')}",
                f"Applied: {arbiter.get('applied')}",
                arbiter.get("rationale", ""),
            ],
            arbiter,
            base + ("/attribution_trace/adjudication" if trace.get("adjudication") else "/arbiter"),
        )
        for i, fault in enumerate(case.get("faults") or []):
            if (fault.get("capability"), fault.get("error_type")) == (
                target.get("capability"),
                target.get("error_type"),
            ):
                edge(f"fault:{i}", arb, "reviews_candidate")
    selection = trace.get("selection") or {}
    dominance = trace.get("dominance") or {}
    if trace:
        policy_lines = [
            f"Eligible leading share: {dominance.get('leading_share', 0)}; required: {dominance.get('required_share')}",
            f"Eligible margin: {dominance.get('actual_margin', 0)}; required: {dominance.get('required_margin')}",
            "Selection: " + str(selection.get("reason", "not recorded")),
        ]
    else:
        policy_lines = [
            "Historical dossier: exact decision trace unavailable.",
            "Rounded weights cannot establish exact ties or threshold decisions.",
        ]
    policy = node(
        "policy",
        6,
        "6",
        "Attribution policy",
        "policy",
        policy_lines,
        trace or {"status": "unavailable", "basis": "Persisted diagnosis only; no reverse reconstruction."},
        base + ("/attribution_trace" if trace else "/blame_status"),
    )
    for fid in fault_ids:
        edge(fid, policy, "contributes_to_policy")
    if arbiter:
        edge("arbiter", policy, "updates_attribution")
    if not fault_ids:
        edge("task", policy, "documents")
    selected_labels = [f["error_type"] for f in case.get("faults", []) if f.get("is_primary")]
    outcome = node(
        "decision",
        7,
        "7",
        "Final reported attribution",
        "decision",
        [
            "Status: " + case.get("blame_status", "unknown"),
            "Headline: " + (", ".join(selected_labels) or "No primary fault assigned"),
            "Causal validation is separate; inspect each candidate.",
        ],
        {
            "selection": selection,
            "fault_side": case.get("fault_side"),
            "primary_fault_ids": primary_ids,
            "support_states": support_states,
        },
        base + "/blame_status",
        status=case.get("blame_status", "unknown"),
    )
    edge(policy, outcome, "selects")
    for i, fault in enumerate(case.get("faults") or []):
        chain = fault.get("evidence_chain") or {}
        counterfactual = chain.get("counterfactual") or {}
        if counterfactual.get("what_would_confirm"):
            supported = support_states[f"fault:{i}"] == "supported"
            check = node(
                f"fault:{i}:check",
                8,
                f"8.{i + 1}",
                "Proposed causal check" if supported else "Inactive historical check",
                "hypothesis",
                (
                    [counterfactual["what_would_confirm"]]
                    if supported
                    else ["Inactive historical check: no supported attribution.", counterfactual["what_would_confirm"]]
                )
                + ["Status: " + counterfactual.get("status", "not_recorded")],
                {
                    "counterfactual": counterfactual,
                    "causal_support": chain.get("causal_support"),
                    "countermeasures": [
                        {
                            k: r.get(k)
                            for k in (
                                "id",
                                "summary",
                                "expected_effect",
                                "deployment_status",
                                "artifact_maturity",
                                "evidence_assessment",
                                "runtime_contract",
                            )
                        }
                        for r in fault.get("countermeasures", [])
                    ],
                },
                base + f"/faults/{i}/evidence_chain/counterfactual",
                f"fault:{i}",
                counterfactual.get("status", "not_recorded")
                if supported
                else "inactive_" + support_states[f"fault:{i}"],
            )
            edge(outcome if supported else f"fault:{i}", check, "proposes_check" if supported else "documents")
    validate_dag(nodes, edges)
    for item in nodes:
        pointer = item["source"]["json_pointer"]
        value, available = case, True
        try:
            for part in pointer.split("/")[2:]:
                value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, ValueError, TypeError):
            available = False
        item["source"]["available"] = available
    return {
        "case_id": case["case_id"],
        "agent": case["agent"],
        "instance_id": case["instance_id"],
        "status": case.get("blame_status", "unknown"),
        "primary_fault_ids": primary_ids,
        "nodes": nodes,
        "edges": edges,
        "action_graph": _action_graph(case, base),
        "trajectory_excerpt": case.get("thinking_process", []),
        "source": _source(base),
        "trace_available": bool(trace),
        "causal_discovery_performed": False,
    }


def build_graphs(cases, agents=None, instances=None, benchmark=None):
    agents = sorted(set(agents if agents is not None else (c["agent"] for c in cases)))
    instances = sorted(set(instances if instances is not None else (c["instance_id"] for c in cases)))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate attribution case IDs")
    if any(c["agent"] not in agents or c["instance_id"] not in instances for c in cases):
        raise ValueError("Attribution cases fall outside the run scope")
    return {
        "schema_version": 1,
        "source_cases_sha256": cases_digest(cases),
        "scope": {
            "agents": agents,
            "instance_ids": instances,
            "benchmark": benchmark or config.BENCHMARK_NAME,
            "diagnosed_failed_cases": len(cases),
        },
        "semantics": "Numbered evidence-and-attribution DAG; arrows are typed provenance links, not causal effects.",
        "graphs": [case_graph(case, index) for index, case in enumerate(cases)],
    }


def render_html(bundle):
    # Prevent user issue/tool text from closing the data script. The viewer puts
    # source text into textContent, never interpolated HTML or executable code.
    data = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("__AWE_GRAPH_DATA__", data)


def write(bundle, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "attribution_graphs.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "ATTRIBUTION.html").write_text(render_html(bundle), encoding="utf-8")
