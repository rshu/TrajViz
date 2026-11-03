"""LLM arbiter — corroborate or refute NON-deductive primary blame.

The rule layer is trustworthy for DEDUCTIVE facts (a file is in/out of the gold ∩
patch sets). It is weaker for INTERPRETIVE attributions: the read-but-not-edited
omission (a fact, but whose *capability* — comprehension vs planning vs editing —
is a judgment), and the judge / contributing labels. This module asks
deepseek-v4-flash (any OpenRouter model) to second-opinion exactly those
non-deductive primaries, so a headline root cause is either a deduction or an
LLM-corroborated judgment — never an unverified heuristic.

Bias controls mirror agent_trajviz.awe._engine.judge: OUTCOME-BLINDED (no gold patch, no pass/fail),
closed verdict set, evidence-cited steps required, confidence-gated, model-
partitioned disk cache. Reading verdicts back needs no API key, so reports stay
offline/reproducible after one arbiter run. Used by rootcause.build_blame via the
``adjudicate`` hook (see agent_trajviz.awe._engine.rootcause); a confident, evidence-cited *refute*
demotes a non-deductive primary to ``conjunctive``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .provenance import fingerprint, verdict_fingerprint, validate_verdict
from . import config
from .gold import build_gold_reference
from .adapters import load_normalized
from .judge import build_evidence, _extract_json  # reuse the blinded evidence + JSON recovery

_SUPPORT = ("confirm", "refute", "unsure")
_CONF = ("high", "medium", "low")

ARBITER_CACHE_DIR = config.PROJECT_DATA / "cache" / "arbiter"

# Human descriptions of the claims the arbiter is asked to check. Only the
# NON-deductive ones are ever sent (see rootcause._needs_adjudication), but the
# map is permissive so a claim never renders as a bare slug.
_CLAIM_DESC = {
    "relevant_change_omitted": "the agent had a relevant file open (it read the file) but never "
    "edited it, so a needed change was omitted — claimed as an "
    "editing/selection failure rather than a search failure",
    "solved_unrelated_problem": "the agent's patch shares no code identifiers with the issue "
    "text and touches none of the code the fix needed — it "
    "plausibly solved a different problem than the one reported",
    "misread_requirement": "the agent misunderstood what the issue was asking for",
    "missed_constraint": "the agent overlooked an explicit constraint or edge case in the issue",
    "incomplete_requirement_coverage": "the agent addressed only part of a multi-part requirement",
    "scope_misjudged": "the agent tackled a broader or narrower problem than the issue asked",
    "no_decomposition": "the agent began editing with no discernible plan on a task that needed one",
    "flawed_plan": "the agent adopted an approach that cannot satisfy the stated requirement",
    "premature_implementation": "the agent began editing before locating/understanding the code",
    "abandoned_plan": "the agent stated a plan then diverged from it without justification",
    "test_result_misinterpreted": "the agent ran a gating test, saw failure, yet proceeded as if it passed",
    "gave_up_early": "the agent disengaged early — it neither investigated (no test run, almost "
    "no files read) nor produced any source fix",
    "no_test_after_final_edit": "the agent ran tests earlier but none after its final source edit, "
    "so the submitted patch state was never verified",
    "gating_test_modified": "the agent's patch modified the gating test file(s), so its local "
    "verification was against self-modified tests",  # legacy (pre-v0.39 records)
    "test_expectation_rewritten": "the agent replaced existing test assertions with new ones — it "
    "changed what the tests expect rather than the code",
    "regression_test_missing": "previously-passing tests broke under the patch and the agent never "
    "ran them (no suite-wide run, no command naming their files)",
    "regression_introduced": "the patch broke previously-passing tests (official per-test record)",
    "partial_fix_only": "the patch makes some but not all gating tests pass (official per-test record)",
    "relevant_test_not_run": "the agent never ran the gating test for the issue",
    "unsupported_success_claim": "the agent declared success without running the gating test",
    "repair_not_reverified": "the agent edited after a failure but never re-ran the test",
    "repeated_ineffective_attempt": "the agent re-edited the same file after failures with no progress",
    "retry_without_change": "after a failing run the agent re-ran the same test command without "
    "changing anything in between",
    "reverted_own_change": "the agent applied a change and later swapped the same content back — "
    "it undid its own edit (repair thrash)",
    "authored_test_never_run": "the agent wrote a test file during the run but never executed it",
    "tool_failure_not_retried": "a tool call errored and the agent took no recovering action",
    "repeated_patch_apply_failure": "the agent's edit/apply attempts repeatedly failed with "
    "patch-does-not-apply errors and it kept re-emitting the diff",
    "hallucinated_path_unrecovered": "the agent targeted the same nonexistent path repeatedly "
    "after being told it does not exist",
}

SYSTEM_PROMPT = """You audit a PROPOSED root-cause label for an AI software-engineering agent's \
attempt at a bug-fix/feature task. You judge ONLY the agent's process.

CRITICAL: You are NOT told whether the task passed or failed, and you are NOT shown the reference \
solution. Do not guess the outcome. Decide whether the agent's PROCESS supports the claimed cause:
- "confirm": the evidence clearly shows the claimed problem occurred and is the main thing wrong \
with the process.
- "refute": the evidence shows the claimed cause is NOT the main problem — e.g. the agent actually \
handled that aspect well, or a different, clearly-visible cause dominates.
- "unsure": the evidence is insufficient to tell. This is the DEFAULT — choose it unless you can \
point to specific step numbers.

Rules: (1) Cite specific step numbers for confirm/refute; with no citable evidence, answer "unsure". \
(2) A different-but-valid approach is not an error. (3) If the agent's internal reasoning is hidden \
(noted below), judge only from visible actions/messages and lower your confidence.

Respond with a SINGLE JSON object, nothing else:
{"support":"confirm|refute|unsure","confidence":"high|medium|low","evidence_steps":[<int>,...],"rationale":"<=240 chars"}"""


def _render_user(evidence: dict, claim_cap: str, claim_et: str) -> str:
    desc = _CLAIM_DESC.get(claim_et, claim_et)
    lines = [
        "TASK (the issue the agent must resolve):",
        evidence["problem_statement"].strip(),
        "",
        "AGENT PROCESS EVIDENCE (numbered steps; cite these step numbers):",
    ]
    from .evidenceview import render_rows

    lines.extend(render_rows(evidence))
    if evidence["hidden_reasoning"]:
        lines += [
            "",
            "NOTE: this agent's internal reasoning is HIDDEN/encrypted. Judge only from "
            "visible messages/actions and lower confidence.",
        ]
    lines += [
        "",
        f"CLAIMED PRIMARY CAUSE: {claim_cap} — {claim_et}: {desc}.",
        "Based only on the process evidence above, is this claim supported as the PRIMARY "
        "problem with the agent's process (if any)? Return the JSON verdict now.",
    ]
    return "\n".join(lines)


def build_messages(gold, traj, claim_cap: str, claim_et: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_user(build_evidence(gold, traj, focus_cap=claim_cap), claim_cap, claim_et)},
    ]


def parse_verify(content: str, valid_steps=None) -> dict:
    obj = _extract_json(content) or {}
    obj = obj if isinstance(obj, dict) else {}
    support = obj.get("support")
    if support not in _SUPPORT:
        support = "unsure"
    conf = obj.get("confidence")
    if conf not in _CONF:
        conf = "low"
    raw = obj.get("evidence_steps")
    steps = []
    for s in (raw if isinstance(raw, list) else [])[:8]:
        try:
            steps.append(int(s))
        except (TypeError, ValueError):
            continue
    # Citations must point at steps actually rendered into the prompt — a
    # fabricated id must not satisfy the evidence gate (nor the demote gate in
    # rootcause.build_blame, which requires cited steps to apply a refute).
    if valid_steps is not None:
        steps = [s for s in steps if s in valid_steps]
    # Evidence gate (mirrors the judge): a confirm/refute with no cited steps cannot
    # stand as a confident verdict, so it is downgraded to low confidence.
    if support != "unsure" and not steps and conf != "low":
        conf = "low"
    return {
        "support": support,
        "confidence": conf,
        "evidence_steps": steps,
        "rationale": str(obj.get("rationale", ""))[:240],
    }


def _valid_response(response: dict) -> bool:
    obj = _extract_json(response.get("content", ""))
    return (
        response.get("finish_reason") != "length"
        and isinstance(obj, dict)
        and obj.get("support") in _SUPPORT
        and obj.get("confidence") in _CONF
        and isinstance(obj.get("evidence_steps"), list)
        and isinstance(obj.get("rationale"), str)
    )


def _claim_key(cap: str, et: str) -> str:
    return f"{cap}::{et}"


def aggregate_arbiter(verdicts: list) -> dict:
    """Majority-vote N independent arbiter samples (panel mode). Confirm/refute wins
    only with a STRICT majority, else `unsure` — so a non-deductive primary is demoted
    only on a robust refute, not a single skeptical sample. Records `agreement`. Pure."""
    from collections import Counter

    n = len(verdicts) or 1
    cnt = Counter(v.get("support", "unsure") for v in verdicts)
    top, c = cnt.most_common(1)[0]
    support = top if c * 2 > n else "unsure"
    votes = [v for v in verdicts if v.get("support") == support]
    # v0.32 (mirrors the judge aggregator's fix): the representative supplies
    # the CITATIONS — prefer a cited vote, or a majority refute whose first
    # vote happens to be uncited fails rootcause's `cited` gate and the refute
    # silently evaporates. Confidence is the panel's lower median, not one
    # vote's self-rating.
    order = {"low": 0, "medium": 1, "high": 2}
    cited = [v for v in votes if v.get("evidence_steps")]
    rep = (cited or votes or [{}])[0]
    confs = sorted((v.get("confidence", "low") for v in votes), key=lambda x: order.get(x, 0))
    conf = confs[(len(confs) - 1) // 2] if confs else "low"
    return {
        "support": support,
        "confidence": conf,
        "evidence_steps": rep.get("evidence_steps", []) or [],
        "rationale": rep.get("rationale", "") or "",
        "agreement": round(cnt[support] / n, 3),
        "n_votes": n,
    }


def verdict_path(
    agent: str, instance_id: str, model: str | None = None, claim_cap: str | None = None, claim_et: str | None = None
) -> Path:
    """Cache path. The claim is part of the KEY: one (model, agent, instance) can
    carry different claimed primaries across pipeline runs (e.g. after judge
    verdicts land), and a single unkeyed slot would thrash — re-spending API calls
    and silently orphaning the previous claim's verdict. Legacy unkeyed paths are
    still read (see verdict_cached) so existing paid runs stay valid."""
    base = ARBITER_CACHE_DIR / config.model_slug(model) / agent
    if claim_cap and claim_et:
        return base / f"{instance_id}__{claim_cap}__{claim_et}.json"
    return base / f"{instance_id}.json"


def _verdict_key(agent, instance_id, model, claim_cap, claim_et):
    return verdict_fingerprint(
        agent,
        instance_id,
        model,
        SYSTEM_PROMPT,
        (build_evidence, _render_user),
        {"claim": _claim_key(claim_cap, claim_et), "description": _CLAIM_DESC.get(claim_et, claim_et)},
    )


def verdict_cached(
    agent: str, instance_id: str, model: str | None = None, claim_cap: str | None = None, claim_et: str | None = None
) -> dict | None:
    paths = [verdict_path(agent, instance_id, model, claim_cap, claim_et)]
    if claim_cap and claim_et:  # legacy pre-claim-key layout
        paths.append(verdict_path(agent, instance_id, model))
    for p in paths:
        if p.exists():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if rec.get("parse_failed") or rec.get("finish_reason") == "length":
                continue
            if (not claim_cap) or rec.get("claim") == _claim_key(claim_cap, claim_et):
                claim = rec.get("claim", "").split("::", 1)
                cap, et = (claim_cap, claim_et) if claim_cap else (claim if len(claim) == 2 else ("", ""))
                checked = validate_verdict(
                    rec, lambda cap=cap, et=et: _verdict_key(agent, instance_id, model, cap, et), "arbiter"
                )
                if checked is not None:
                    return checked
    return None


def verify_one(
    agent: str,
    instance_id: str,
    claim_cap: str,
    claim_et: str,
    client=None,
    model: str | None = None,
    force: bool = False,
    samples: int = 1,
    temperature: float | None = None,
) -> dict | None:
    """Adjudicate one claimed primary. Cached per (model, agent, instance, claim).
    With samples>1 (panel mode) the N independent verdicts are majority-voted via
    aggregate_arbiter, with an agreement score. Returns the record, or None if there
    is no trajectory."""
    if not force:
        cached = verdict_cached(agent, instance_id, model, claim_cap, claim_et)
        if cached is not None and (samples <= 1 or cached.get("n_votes", 1) >= samples):
            return cached
    try:
        gold = build_gold_reference(instance_id)
    except Exception:
        return None
    traj = load_normalized(agent, instance_id)
    if traj is None or not traj.steps:
        return None
    if client is None:
        from .llm import chat as client
    evidence = build_evidence(gold, traj, focus_cap=claim_cap)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_user(evidence, claim_cap, claim_et)},
    ]
    valid_steps = {r["step"] for r in evidence["rows"] if isinstance(r["step"], int)}

    def _call():
        r = client(
            messages,
            model=model,
            max_tokens=config.JUDGE_MAX_TOKENS,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        if r.get("finish_reason") == "length":
            r = client(
                messages,
                model=model,
                max_tokens=config.JUDGE_MAX_TOKENS * 2,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        return r

    vs, usages, model_name, fr = [], [], model or config.JUDGE_MODEL, None
    parse_failed = False
    for _ in range(max(1, samples)):
        r = _call()
        content = r.get("content", "")
        if not _valid_response(r):
            r = _call()  # parse failure ≠ 'unsure': retry once
            content = r.get("content", "")
            if not _valid_response(r):
                parse_failed = True
        vs.append(parse_verify(content, valid_steps))
        usages.append(r.get("usage") or {})
        model_name, fr = r.get("model", model_name), r.get("finish_reason")
    verdict = vs[0] if len(vs) == 1 else aggregate_arbiter(vs)
    usage = (
        usages[0]
        if len(usages) == 1
        else {
            k: sum(u.get(k, 0) or 0 for u in usages)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
        }
    )
    record = {
        "instance_id": instance_id,
        "agent": agent,
        "claim": _claim_key(claim_cap, claim_et),
        "model": model_name,
        "verdict": verdict,
        "usage": usage,
        "finish_reason": fr,
        "response_validation_version": 1,
        "input_fingerprint": _verdict_key(agent, instance_id, model, claim_cap, claim_et),
        "prompt_fingerprint": fingerprint(messages),
        "evidence_view": evidence.get("omissions", {}),
        "cache_provenance": {"status": "verified"},
    }
    if parse_failed:
        record["parse_failed"] = True
    if samples > 1:
        record["n_votes"] = samples
    p = verdict_path(agent, instance_id, model, claim_cap, claim_et)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception:
        pass
    return record


def active(agent: str, model: str | None = None) -> bool:
    d = ARBITER_CACHE_DIR / config.model_slug(model) / agent
    return d.is_dir() and any(d.glob("*.json"))


def n_arbiter(agent: str, model: str | None = None) -> int:
    d = ARBITER_CACHE_DIR / config.model_slug(model) / agent
    return sum(1 for _ in d.glob("*.json")) if d.is_dir() else 0


def auto_adjudicate(agent: str, model: str | None = None):
    """The adjudicate hook for build_blame when arbiter verdicts exist for this
    agent, else None (so the objective pipeline is byte-for-byte unchanged with no
    arbiter run). No API key needed — reads cached verdicts."""
    return adjudication_fn(model) if active(agent, model) else None


def adjudication_fn(model: str | None = None):
    """Return an ``adjudicate(detection, candidate)`` callable for
    rootcause.build_blame that reads cached arbiter verdicts (no API key needed).
    Returns None for a candidate when no matching verdict is cached, so the rule
    decision stands unchanged."""

    def adjudicate(detection: dict, cand: dict):
        rec = verdict_cached(
            detection.get("agent"), detection.get("instance_id"), model, cand.get("capability"), cand.get("error_type")
        )
        return {**rec.get("verdict", {}), "cache_provenance": rec.get("cache_provenance")} if rec else None

    return adjudicate


def mock_client(messages: list[dict], model: str | None = None, **_) -> dict:
    """Deterministic, network-free stand-in for llm.chat — lets --mock and the test
    suite exercise the full arbiter path with no API key. Most verdicts are
    'unsure' (the conservative default); a stable minority confirm/refute."""
    user = messages[-1]["content"]
    h = int(hashlib.md5(user.encode("utf-8")).hexdigest()[:8], 16)
    cite = [(h % 17) + 1]
    roll = h % 100
    if roll < 12:
        support, conf = "refute", ("high" if roll < 6 else "medium")
        rationale = "[mock] process indicates the claimed cause is not primary."
    elif roll < 45:
        support, conf = "confirm", ("high" if roll < 28 else "medium")
        rationale = "[mock] process supports the claimed cause."
    else:
        support, conf, cite, rationale = "unsure", "low", [], "[mock] insufficient evidence."
    verdict = {"support": support, "confidence": conf, "evidence_steps": cite, "rationale": rationale}
    return {
        "content": json.dumps(verdict),
        "model": "mock/deterministic",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
    }
