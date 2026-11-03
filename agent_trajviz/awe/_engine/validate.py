"""Validation harness with controlled ground truth.

Three checks, none of which need an LLM or a human:
  1. false_positive_control  — run the detector on REAL resolved cases; it must
     not emit failure-attributing errors (the outcome-conditioning guarantee).
  2. patch_mutation          — inject patches with a KNOWN defect (wrong file,
     dropped gold file, bloat, empty); the detector must recover the planted
     error type and primary capability. Exercises diff parsing, gold matching,
     module/symbol logic, and the rule mapping end-to-end.
  3. synthetic_trajectory    — hand-built trajectories with known verification
     behavior; the verification rules must classify them correctly.

Reports controlled recall (planted error recovered) and outcome-gate compliance
on solved runs. Neither estimates causal precision on real failed trajectories."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

from . import config
from .gold import build_gold_reference
from .patches import AgentPatch
from .outcomes import resolved
from .diffutil import is_test_file
from .detect import detect
from .rootcause import build_blame
from .trajectory import NormalizedStep, NormalizedTrajectory, EDIT, RUN_TEST

VERIF_ERRORS = {
    "relevant_test_not_run",
    "unsupported_success_claim",
    "test_result_misinterpreted",
    "no_test_after_final_edit",
    "test_expectation_rewritten",
    "authored_test_never_run",
    "regression_test_missing",
}
# over_editing was pruned in v0.9 (behavioral flag only, never emitted as an
# error) — the quality-flag control set is empty until a successor metric lands.
QUALITY_ERRORS: set = set()


def _patch(instance_id, source_files, patch_loc, symbols=None, exists=True):
    src = [f for f in source_files if not is_test_file(f)]
    return AgentPatch(
        agent="synthetic",
        instance_id=instance_id,
        exists=exists,
        edited_files=list(source_files),
        edited_source_files=src,
        edited_test_files=[f for f in source_files if is_test_file(f)],
        patch_loc=patch_loc,
        symbols=list(symbols or []),
    )


def _top_cap(detection) -> str | None:
    caps = build_blame(detection).get("capability_blame", {})
    return max(caps, key=caps.get) if caps else None


# ---------------------------------------------------------- 1. false positives


def false_positive_control() -> dict:
    n = fail_err = quality = 0
    for agent in config.AGENTS:
        for inst in config.list_instances():
            if resolved(agent, inst) is not True:
                continue
            n += 1
            emitted = {c["error_type"] for c in detect(inst, agent, use_trajectory=False)["candidates"]}
            if emitted - QUALITY_ERRORS:
                fail_err += 1
            if emitted & QUALITY_ERRORS:
                quality += 1
    return {
        "n_resolved": n,
        "failure_errors_emitted": fail_err,
        "false_positive_rate": fail_err / n if n else 0,
        "over_editing_on_success": quality,
    }


# --------------------------------------------------------------- 2. mutations


def _mutations(instance_id, gold):
    """Yield (mutation, patch, resolved, expected_errors, expected_cap)."""
    g = gold.gold_files
    yield (
        "wrong_file",
        _patch(instance_id, ["docs/_mutation_target.py"], 12, gold.gold_symbols),
        False,
        {"missed_relevant_file"},
        "code_localization",
    )
    yield (
        "bloat",
        _patch(instance_id, g, max(10 * gold.gold_patch_loc, 40), gold.gold_symbols),
        False,
        {"incorrect_patch"},
        "code_editing",
    )
    yield ("empty_patch", _patch(instance_id, [], 0, [], exists=False), False, {"no_patch_produced"}, "code_editing")
    if len(g) >= 2:
        yield (
            "drop_gold_file",
            _patch(instance_id, g[:-1], gold.gold_patch_loc, gold.gold_symbols),
            False,
            {"cross_module_context_missing", "missed_relevant_file"},
            "code_localization",
        )


def patch_mutation(limit: int = 150) -> dict:
    stats = defaultdict(lambda: {"n": 0, "recall": 0, "cap_ok": 0})
    cap_confusion = defaultdict(Counter)
    insts = config.list_instances()[:limit]
    for inst in insts:
        try:
            gold = build_gold_reference(inst)
        except Exception:
            continue
        if not gold.gold_files:
            continue
        for mut, patch, res, expected, exp_cap in _mutations(inst, gold):
            # A planted search miss requires observed reachability. The current
            # detector correctly abstains when the read set is unavailable. Give
            # this controlled fixture an explicit trace containing only its
            # planted edits; no reference-file reads are observed in this trace.
            trace = NormalizedTrajectory(
                "synthetic",
                inst,
                [
                    NormalizedStep(
                        step_id=i,
                        turn_id=i,
                        actor="agent",
                        type="tool_call",
                        canonical_action=EDIT,
                        native_tool="Edit",
                        files=[path],
                    )
                    for i, path in enumerate(patch.edited_source_files)
                ],
            )
            d = detect(
                inst,
                "synthetic",
                use_trajectory=False,
                trajectory_override=trace if mut in ("wrong_file", "drop_gold_file") else None,
                patch_override=patch,
                resolved_override=res,
                use_judge=False,
            )
            emitted = {c["error_type"] for c in d["candidates"]}
            top = _top_cap(d)
            s = stats[mut]
            s["n"] += 1
            s["recall"] += bool(emitted & expected)
            s["cap_ok"] += top == exp_cap
            cap_confusion[mut][top] += 1
    return {"per_mutation": dict(stats), "cap_confusion": {k: dict(v) for k, v in cap_confusion.items()}}


# ------------------------------------------------------ 3. synthetic trajectory


def _ftp_token(gold) -> str | None:
    import re

    for tid in gold.fail_to_pass:
        m = re.search(r"test_\w+", tid)
        if m:
            return m.group(0)
    return None


def _S(i, **kw):
    return NormalizedStep(step_id=i, turn_id=i, **kw)


def _traj_cases(agent, inst, token):
    base_edit = _S(0, actor="agent", type="tool_call", canonical_action=EDIT, native_tool="Edit", files=["x.py"])
    return {
        "no_test": (
            NormalizedTrajectory(agent, inst, [base_edit, _S(1, actor="agent", type="message", text="Done editing.")]),
            {"relevant_test_not_run"},
        ),
        "claim_no_test": (
            NormalizedTrajectory(
                agent,
                inst,
                [base_edit, _S(1, actor="agent", type="message", text="The fix is complete and all tests pass.")],
            ),
            {"unsupported_success_claim"},
        ),
        "test_then_false_claim": (
            NormalizedTrajectory(
                agent,
                inst,
                [
                    _S(
                        0,
                        actor="agent",
                        type="tool_call",
                        canonical_action=RUN_TEST,
                        native_tool="Bash",
                        test_run=True,
                        args={"command": f"python -m pytest -k {token}"},
                    ),
                    _S(1, actor="tool", type="tool_result", tool_error=True, output_excerpt="1 failed"),
                    _S(2, actor="agent", type="message", text="Everything works now, the issue is resolved."),
                ],
            ),
            {"test_result_misinterpreted"},
        ),
        "ran_test_no_claim": (
            NormalizedTrajectory(
                agent,
                inst,
                [
                    _S(
                        0,
                        actor="agent",
                        type="tool_call",
                        canonical_action=RUN_TEST,
                        native_tool="Bash",
                        test_run=True,
                        args={"command": f"python -m pytest -k {token}"},
                    ),
                    _S(1, actor="tool", type="tool_result", tool_error=True, output_excerpt="1 failed"),
                    _S(2, actor="agent", type="message", text="Still failing; need to investigate further."),
                ],
            ),
            set(),
        ),  # control: no verification error expected
    }


def synthetic_trajectory(limit: int = 80) -> dict:
    stats = defaultdict(lambda: {"n": 0, "ok": 0})
    for inst in config.list_instances()[:limit]:
        try:
            gold = build_gold_reference(inst)
        except Exception:
            continue
        token = _ftp_token(gold)
        if not token:
            continue
        # This is a controlled wiring check: the generated command explicitly
        # names the gold gating test.  Corpus visibility policy must not disable
        # the two success-claim rules that this synthetic case is meant to test.
        gold = replace(gold, gating_tests_visible=True)
        patch = _patch(inst, gold.gold_files, gold.gold_patch_loc, gold.gold_symbols)
        for name, (traj, expected) in _traj_cases("synthetic", inst, token).items():
            d = detect(
                inst,
                "synthetic",
                use_trajectory=True,
                patch_override=patch,
                resolved_override=False,
                trajectory_override=traj,
                gold_override=gold,
            )
            emitted_verif = {c["error_type"] for c in d["candidates"]} & VERIF_ERRORS
            ok = (emitted_verif == expected) if not expected else bool(emitted_verif & expected)
            s = stats[name]
            s["n"] += 1
            s["ok"] += ok
    return dict(stats)


# --------------------------------------------------- 4. LLM-judge wiring (no net)


def judge_parse() -> dict:
    """The judge's parse/gate layer must be robust to messy model output and must
    enforce the bias rules (closed labels; low-confidence and evidence-less
    verdicts never blame). Deterministic — no model call."""
    from .judge import parse_verdict, verdict_to_specs

    blamed = lambda raw: {s["error_type"] for s in verdict_to_specs({"verdict": parse_verdict(raw), "model": "t"})}
    checks = []

    # valid, high-confidence, evidence-cited -> blamed
    checks.append(
        (
            "clean_misread",
            blamed(
                '{"requirement_understanding":{"error_type":"misread_requirement",'
                '"confidence":"high","evidence_steps":[3],"rationale":"x"},'
                '"task_planning":{"error_type":"none","confidence":"high",'
                '"evidence_steps":[],"rationale":""}}'
            )
            == {"misread_requirement"},
        )
    )
    # wrapped in prose + code fence -> still parsed
    checks.append(
        (
            "prose_wrapped",
            blamed(
                'Sure!\n```json\n{"requirement_understanding":{"error_type":'
                '"missed_constraint","confidence":"medium","evidence_steps":[1]},'
                '"task_planning":{"error_type":"none","confidence":"high"}}\n```'
            )
            == {"missed_constraint"},
        )
    )
    # invalid label -> coerced to none -> not blamed
    checks.append(
        (
            "invalid_label",
            blamed(
                '{"requirement_understanding":{"error_type":"made_up_label",'
                '"confidence":"high","evidence_steps":[2]},'
                '"task_planning":{"error_type":"none","confidence":"high"}}'
            )
            == set(),
        )
    )
    # error WITHOUT cited evidence -> downgraded to low -> not blamed
    checks.append(
        (
            "no_evidence_gate",
            blamed(
                '{"requirement_understanding":{"error_type":"scope_misjudged",'
                '"confidence":"high","evidence_steps":[]},'
                '"task_planning":{"error_type":"none","confidence":"high"}}'
            )
            == set(),
        )
    )
    # explicit low confidence -> not blamed
    checks.append(
        (
            "low_conf_gate",
            blamed(
                '{"requirement_understanding":{"error_type":"misread_requirement",'
                '"confidence":"low","evidence_steps":[4]},'
                '"task_planning":{"error_type":"flawed_plan","confidence":"low",'
                '"evidence_steps":[5]}}'
            )
            == set(),
        )
    )
    # garbage -> safe empty verdict, no crash
    checks.append(("garbage_safe", blamed("not json at all") == set()))
    return {"checks": dict(checks), "n": len(checks), "passed": sum(v for _, v in checks)}


def judge_blame_upstream() -> dict:
    """v0.32 coverage-only invariant: a judged upstream cause (requirement/
    planning) — at ANY confidence — must never take PRIMARY blame from a
    deductive fact; it contributes blame weight (coverage) only. (Pre-v0.32
    these checks asserted the opposite — judge-takes-primary — and validation
    §4 stayed red from the v0.32 merge until the 2026-07-12 review fixed it.)"""

    def det(cands, flag="right_place_wrong_change"):
        return {"resolved": False, "loc_flag": flag, "candidates": cands}

    cases = {}
    # requirement (high) + editing -> editing stays primary, requirement covers
    b = build_blame(
        det(
            [
                {
                    "capability": "code_editing",
                    "dimension": "patch_correctness",
                    "error_type": "incorrect_patch",
                    "confidence": "high",
                    "evidence": {},
                },
                {
                    "capability": "requirement_understanding",
                    "dimension": "requirement_comprehension",
                    "error_type": "misread_requirement",
                    "confidence": "high",
                    "evidence": {},
                },
            ]
        )
    )
    cases["req_high_covers_not_primary"] = (
        b["status"] == "primary"
        and b["primary"]["capability"] == "code_editing"
        and b["capability_blame"].get("requirement_understanding", 0) > 0
    )
    # planning (high) + localization -> localization stays primary, planning covers
    b = build_blame(
        det(
            [
                {
                    "capability": "code_localization",
                    "dimension": "file_level_localization",
                    "error_type": "missed_relevant_file",
                    "confidence": "high",
                    "evidence": {},
                },
                {
                    "capability": "task_planning",
                    "dimension": "solution_planning",
                    "error_type": "flawed_plan",
                    "confidence": "high",
                    "evidence": {},
                },
            ],
            flag="loc_miss",
        )
    )
    cases["plan_high_covers_not_primary"] = (
        b["status"] == "primary"
        and b["primary"]["capability"] == "code_localization"
        and b["capability_blame"].get("task_planning", 0) > 0
    )
    # no judge cause -> editing stays primary (regression guard for objective path)
    b = build_blame(
        det(
            [
                {
                    "capability": "code_editing",
                    "dimension": "patch_correctness",
                    "error_type": "incorrect_patch",
                    "confidence": "high",
                    "evidence": {},
                }
            ]
        )
    )
    cases["objective_unchanged"] = b["status"] == "primary" and b["primary"]["capability"] == "code_editing"
    # a MEDIUM judge verdict must NOT seize primary from a deductive editing fact
    b = build_blame(
        det(
            [
                {
                    "capability": "code_editing",
                    "dimension": "patch_correctness",
                    "error_type": "incorrect_patch",
                    "confidence": "high",
                    "evidence": {},
                },
                {
                    "capability": "requirement_understanding",
                    "dimension": "requirement_comprehension",
                    "error_type": "misread_requirement",
                    "confidence": "medium",
                    "evidence": {},
                },
            ]
        )
    )
    cases["medium_judge_not_primary_over_deductive"] = (
        b["status"] == "primary" and b["primary"]["capability"] == "code_editing"
    )
    return {"checks": cases, "n": len(cases), "passed": sum(cases.values())}


def arbiter_wiring() -> dict:
    """The LLM-arbiter layer must (a) parse/gate robustly and (b) only ever touch
    NON-deductive primaries — a confident refute demotes them, a confirm
    corroborates, and deductive loc/edit facts are never sent. Deterministic — no
    model call (a callable stands in for the cached verdict)."""
    from .arbiter import parse_verify

    checks = {}
    # --- parse/gate ---
    checks["parse_refute"] = (
        parse_verify('{"support":"refute","confidence":"high","evidence_steps":[3]}')["support"] == "refute"
    )
    checks["invalid_support_to_unsure"] = (
        parse_verify('{"support":"maybe","confidence":"high","evidence_steps":[1]}')["support"] == "unsure"
    )
    checks["evidence_gate_downgrades"] = (
        parse_verify('{"support":"refute","confidence":"high","evidence_steps":[]}')["confidence"] == "low"
    )
    checks["garbage_safe"] = parse_verify("nonsense")["support"] == "unsure"

    # --- blame integration ---
    det = {
        "resolved": False,
        "loc_flag": "loc_miss",
        "candidates": [
            {
                "capability": "code_editing",
                "dimension": "patch_correctness",
                "error_type": "relevant_change_omitted",
                "confidence": "high",
                "evidence": {},
            }
        ],
    }
    refute = lambda d, c: {"support": "refute", "confidence": "high", "evidence_steps": [3]}
    confirm = lambda d, c: {"support": "confirm", "confidence": "high", "evidence_steps": [3]}
    refuted = build_blame(det, adjudicate=refute)
    checks["refuted_singleton_is_unattributed"] = (
        refuted["status"] == "unattributed"
        and refuted["capability_blame"] == {"unattributed": 1.0}
        and all(f["weight"] == 0 for f in refuted["blame"])
    )
    with_contribution = {
        **det,
        "candidates": det["candidates"]
        + [
            {
                "capability": "code_verification",
                "dimension": "test_execution",
                "error_type": "relevant_test_not_run",
                "confidence": "medium",
                "evidence": {},
            }
        ],
    }
    checks["refute_with_surviving_contribution_is_conjunctive"] = (
        build_blame(with_contribution, adjudicate=refute)["status"] == "conjunctive"
    )
    checks["confirm_corroborates"] = bool(
        build_blame(det, adjudicate=confirm)["primary"].get("arbiter", {}).get("corroborated")
    )

    det_ded = {
        "resolved": False,
        "loc_flag": "right_place_wrong_change",
        "candidates": [
            {
                "capability": "code_editing",
                "dimension": "patch_correctness",
                "error_type": "incorrect_patch",
                "confidence": "high",
                "evidence": {},
            }
        ],
    }
    seen = []
    spy = lambda d, c: (seen.append(c), {"support": "refute", "confidence": "high", "evidence_steps": [1]})[1]
    b = build_blame(det_ded, adjudicate=spy)
    checks["deductive_never_adjudicated"] = (not seen) and b["primary"]["capability"] == "code_editing"
    return {"checks": checks, "n": len(checks), "passed": sum(checks.values())}


def countermeasure_catalog(entries=None) -> dict:
    """Validate the grounded countermeasure catalog. Fifteen checks enforce that
    every remedy is grounded and 'verified' cannot be claimed without Tier-B evidence:
      schema               — required fields present + valid enums; unique ids.
      referential_integrity— error_type in produced_error_types() (live diagnosis layer),
                             capability matches the taxonomy, AND evidence_signal is the
                             EXACT signal the diagnosis layer binds to that error type
                             (detect.expected_evidence_signal) — not merely a real signal.
      coverage             — every PRODUCED error type has >=1 entry (uncovered listed).
      honesty_invariant    — status=='verified' requires hashed, scoped paired-run
                             evidence with re-derived benefit and harm checks.
      artifact             — every entry has an artifact specification: non-empty content,
                             runtime_trigger, diagnostic_signature, and feasibility;
                             artifact_type valid and == intervention_type; example_case
                             is null or a dict with a non-empty instance_id.
      references           — each reference well-formed (title/id/supports/source_type)
                             and provenance.kind == provenance_kind_for(references).
      scaffold_layer       — every entry names a valid scaffold layer (system_prompt /
                             tool_definition / control_flow / retry_policy /
                             response_synthesis): WHERE the developer plugs the remedy in.
    v2 checks (2026-07 refinement — see catalog methodology[]):
      hook_event           — every entry names the scaffold event it fires at.
      expected_effect      — non-empty subset of {fix_rate, failure_surfacing,
                             budget_efficiency}: the endpoint a pilot must measure.
      oracle_free_trigger  — runtime_trigger never mentions gold data (word 'gold'):
                             deployable conditions must not consume benchmark oracles
                             (METH-ORACLE-1); gold vocabulary belongs in
                             diagnostic_signature only.
      guardrail_contract   — every guardrail carries {max_triggers>=1, escape_hatch}:
                             a blocking gate without an exit can deadlock a run into
                             budget exhaustion (METH-COMPOSE-1).
      bundles              — bundle members exist and every entry belongs to exactly
                             one bundle (an integration planning group).
      methodology          — every methodology note has an id, prose, and well-formed
                             references.
      annex                — operational-annex entries ground on REAL trajectory
                             signals (awe/trajsignals fields) with well-formed
                             references and valid status.
    v0.29 check (Jiang et al. crosswalk — see METH-HARNESS-1):
      benchmark_annex      — benchmark-artifact annex entries (harness-rigidity
                             suspects) ground on REAL trajectory signals plus an
                             explicit patch condition, carry a computed prevalence
                             block, and are review queues (mechanism must say
                             suspects are manually reviewed, never auto-blamed).
    Entry-level checks run on `entries` (injectable for tests); the document-level
    v2 checks (bundles/methodology/annexes) always read the shipped catalog document.
    """
    from . import countermeasures as cm
    from .detect import produced_error_types, expected_evidence_signal
    from .remedies import valid_selection_rules

    if entries is None:
        entries = cm.load_catalog()

    produced = produced_error_types()
    # capability lookup still from the taxonomy (valid capabilities)
    from .annotation import ERRORS_BY_CAPABILITY

    cap_of = {et: cap for cap, ets in ERRORS_BY_CAPABILITY.items() for et in ets}

    ids = [e.id for e in entries]
    schema = (
        bool(entries)
        and all(
            e.id
            and e.mechanism
            and e.summary
            and e.intervention_type in cm.VALID_TYPES
            and e.status in cm.VALID_STATUS
            and valid_selection_rules(e.selection_rules)
            for e in entries
        )
        and len(ids) == len(set(ids))
    )

    referential = all(
        e.error_type in produced  # the system actually diagnoses it
        and e.capability == cap_of.get(e.error_type)  # capability matches the taxonomy
        and e.evidence_signal == expected_evidence_signal(e.error_type)  # the EXACT per-type binding
        for e in entries
    )

    covered = {e.error_type for e in entries}
    uncovered = sorted(produced - covered)  # every PRODUCED type needs a remedy
    coverage = not uncovered

    from .remedy_verification import validate_verification

    honesty = all(validate_verification(e)["valid"] for e in entries if e.status == "verified")

    artifact = bool(entries) and all(
        e.artifact_content.strip()
        and e.runtime_trigger.strip()
        and e.diagnostic_signature.strip()
        and e.feasibility.strip()
        and e.artifact_type in cm.VALID_TYPES
        and e.artifact_type == e.intervention_type
        and (e.example_case is None or (isinstance(e.example_case, dict) and e.example_case.get("instance_id")))
        and isinstance(e.empirical_support, int)
        and e.empirical_support >= 0
        for e in entries
    )

    def _refs_ok(references) -> bool:
        return isinstance(references, list) and all(
            (
                isinstance(r, dict)
                and r.get("title")
                and r.get("id")
                and r.get("supports")
                and r.get("source_type") in cm.VALID_SOURCE_TYPES
            )
            for r in references
        )

    references_ok = bool(entries) and all(
        _refs_ok(e.references) and e.provenance_kind == cm.provenance_kind_for(e.references) for e in entries
    )

    # scaffold_layer — every entry names WHERE in the agent the remedy plugs in
    # (the developer-facing layer), from the closed enum.
    scaffold = bool(entries) and all(e.scaffold_layer in cm.VALID_SCAFFOLD_LAYERS for e in entries)

    # --- v2: deployment semantics on each entry ---
    hook_event = bool(entries) and all(e.hook_event in cm.VALID_HOOK_EVENTS for e in entries)

    expected_effect = bool(entries) and all(
        e.expected_effect and set(e.expected_effect) <= cm.VALID_EFFECTS for e in entries
    )

    # the deployable trigger must never consume benchmark-oracle data; 'gold'
    # vocabulary is diagnostic-side only (METH-ORACLE-1)
    oracle_free = bool(entries) and not any(re.search(r"\bgold\b", e.runtime_trigger, re.IGNORECASE) for e in entries)

    guardrail_contract = bool(entries) and all(
        (e.intervention_type != "guardrail")
        or (
            isinstance(e.contract, dict)
            and type(e.contract.get("max_triggers")) is int
            and e.contract["max_triggers"] >= 1
            and str(e.contract.get("escape_hatch", "")).strip()
        )
        for e in entries
    )

    # These check explicit honesty/implementation contracts, not source truth or
    # causal efficacy. Legacy catalogs load with unspecified maturity but cannot
    # pass the current publication-quality checks until upgraded.
    evidence_scope = bool(entries) and all(
        isinstance(e.evidence_assessment, dict)
        and e.evidence_assessment.get("exact_recipe_efficacy") in ("unmeasured", "scoped_verified")
        and isinstance(e.evidence_assessment.get("scope_notes"), str)
        and bool(e.evidence_assessment["scope_notes"].strip())
        and (
            e.evidence_assessment.get("exact_recipe_efficacy") != "scoped_verified"
            or (e.status == "verified" and validate_verification(e)["valid"])
        )
        and all(
            isinstance(r, dict)
            and r.get("evidence_role") in ("failure_pattern", "related_method", "implementation")
            and type(r.get("scope_reviewed")) is bool
            and isinstance(r.get("study_scope"), str)
            and bool(r["study_scope"].strip())
            for r in e.references
        )
        for e in entries
    )
    artifact_maturity = bool(entries) and all(
        e.artifact_maturity in ("specification", "portable_prototype", "native_adapter") for e in entries
    )
    from .remedy_runtime import valid_contract

    bounded_runtime = bool(entries) and all(e.intervention_type != "guardrail" or valid_contract(e) for e in entries)

    # --- v2: document-level sections (always the shipped catalog document) ---
    shipped_ids = {e.id for e in cm.load_catalog()}
    bundle_list = cm.bundles()
    members = [m for b in bundle_list for m in b.get("members", [])]
    bundles_ok = (
        bool(bundle_list)
        and len(members) == len(set(members))  # no double-assignment
        and set(members) == shipped_ids  # exact partition
        and all(
            b.get("id") and b.get("summary") and b.get("hook_events") and set(b["hook_events"]) <= cm.VALID_HOOK_EVENTS
            for b in bundle_list
        )
    )

    methodology_ok = all(
        m.get("id") and str(m.get("note", "")).strip() and _refs_ok(m.get("references", [])) for m in cm.methodology()
    ) and bool(cm.methodology())

    import dataclasses
    from .trajsignals import TrajectorySignals

    traj_fields = {f.name for f in dataclasses.fields(TrajectorySignals)}
    annex_ok = all(
        a.get("id")
        and str(a.get("mechanism", "")).strip()
        and a.get("grounds", {}).get("trajectory_signal") in traj_fields
        and a.get("status") in cm.VALID_STATUS
        and _refs_ok(a.get("references", []))
        and a.get("provenance", {}).get("kind") == cm.provenance_kind_for(a.get("references", []))
        for a in cm.operational_annex()
    )

    # v0.29: harness-rigidity suspects must ground on real signals + an explicit
    # patch condition, quantify their prevalence, and remain a manual-review
    # queue (mechanism must state review; suspects are never auto-blamed).
    benchmark_annex_ok = all(
        b.get("id")
        and str(b.get("mechanism", "")).strip()
        and "review" in str(b.get("mechanism", "")).lower()
        and set(b.get("grounds", {}).get("trajectory_signals", ())) <= traj_fields
        and b.get("grounds", {}).get("trajectory_signals")
        and str(b.get("grounds", {}).get("patch_condition", "")).strip()
        and isinstance(b.get("prevalence"), dict)
        and b.get("status") in cm.VALID_STATUS
        and _refs_ok(b.get("references", []))
        and b.get("provenance", {}).get("kind") == cm.provenance_kind_for(b.get("references", []))
        for b in cm.benchmark_artifact_annex()
    )

    checks = {
        "schema": schema,
        "referential_integrity": referential,
        "coverage": coverage,
        "honesty_invariant": honesty,
        "artifact": artifact,
        "references": references_ok,
        "scaffold_layer": scaffold,
        "hook_event": hook_event,
        "expected_effect": expected_effect,
        "oracle_free_trigger": oracle_free,
        "guardrail_contract": guardrail_contract,
        "bundles": bundles_ok,
        "methodology": methodology_ok,
        "annex": annex_ok,
        "benchmark_annex": benchmark_annex_ok,
        "evidence_scope": evidence_scope,
        "artifact_maturity": artifact_maturity,
        "bounded_runtime": bounded_runtime,
    }
    return {"checks": checks, "n": len(checks), "passed": sum(checks.values()), "uncovered": uncovered}


def evidence_chain_audit(limit: int | None = None) -> dict:
    """§7 — internal-consistency / tamper-evidence check on the persisted evidence chains.
    Reads data/dossier/faults.json; for each fault, agent_trajviz.awe._engine.audit re-derives its factual
    atoms with logic independent of the detector and compares to the persisted chain.
    Passes iff zero mismatches (unverifiable never fails). This certifies **internal
    consistency** — that atoms weren't corrupted or transcribed incorrectly — not
    correctness or ground-truth match (construct validity requires human annotation,
    not yet run)."""
    import json
    from . import audit, config

    path = config.PROJECT_DATA / "dossier" / "faults.json"
    counts = {audit.RE_DERIVED: 0, audit.STRUCTURAL_OK: 0, audit.UNVERIFIABLE: 0, audit.MISMATCH: 0}
    if not path.exists():
        return {"checks": {"no_mismatches": True}, "counts": counts, "n": 0, "passed": 1}
    cases = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    for rec in cases:
        out = audit.audit_case(rec["agent"], rec["instance_id"], persisted=rec)
        for k, v in out["verdict_counts"].items():
            counts[k] = counts.get(k, 0) + v
    no_mism = counts[audit.MISMATCH] == 0
    return {"checks": {"no_mismatches": no_mism}, "counts": counts, "n": sum(counts.values()), "passed": int(no_mism)}
