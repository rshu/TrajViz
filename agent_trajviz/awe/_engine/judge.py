"""LLM-as-judge for the two judgment-heavy capabilities.

Requirement Understanding and Task Planning have **no objective signature** — a
gold patch and a pass/fail bit cannot verify whether an agent *understood the
issue* or *planned a viable approach*. Those failures propagate downstream and,
in the objective-only slice, masquerade as Localization/Editing faults. This
module adds a constrained LLM judge that names them directly.

Design (bias controls baked in, see LLM_JUDGE.md):
  * **Outcome-blinded** — the judge never sees pass/fail or the gold patch, only
    the problem statement and the agent's own process. It cannot rationalize
    backward from the result.
  * **Closed taxonomy** — it must pick one error from a fixed set per capability,
    or ``none``. It cannot invent labels.
  * **Evidence-grounded** — every flagged error must cite trajectory step ids;
    low-confidence verdicts are recorded but never assign blame.
  * **Observability-aware** — when reasoning is hidden (e.g. Codex), the judge is
    told not to infer "didn't think about X" from absence.

The judge writes a per-(model, agent, instance) verdict to ``data/cache/judge``.
Reading those verdicts back needs **no API key**, so the whole report pipeline
stays offline and reproducible after one judging run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .provenance import fingerprint, verdict_fingerprint, validate_verdict
from . import config
from .gold import GoldReference, build_gold_reference
from .adapters import load_normalized
from .trajectory import SEARCH, READ, EDIT, WRITE, RUN_TEST, RUN_COMMAND

# ---------------------------------------------------------------- taxonomy ----
# Closed error sets. Keep in sync with annotation.ERRORS_BY_CAPABILITY and
# configs/taxonomy.slice.yaml.
REQUIREMENT_ERRORS = {
    "misread_requirement": "Misunderstood what the issue is actually asking for.",
    "missed_constraint": "Overlooked an explicit constraint, edge case, or requirement stated in the issue.",
    "incomplete_requirement_coverage": "Understood and addressed only part of a multi-part requirement.",
    "scope_misjudged": "Tackled a meaningfully broader or narrower problem than the issue asks.",
}
PLANNING_ERRORS = {
    "no_decomposition": "Started editing with no discernible plan on a task that needed one.",
    "flawed_plan": "Adopted an approach that cannot satisfy the stated requirement.",
    "premature_implementation": "Began editing before locating or understanding the relevant code.",
    "abandoned_plan": "Stated a reasonable plan, then diverged from it without justification.",
}
JUDGE_CAPS = ("requirement_understanding", "task_planning")
_CAP_ERRORS = {"requirement_understanding": REQUIREMENT_ERRORS, "task_planning": PLANNING_ERRORS}
_CAP_DIM = {"requirement_understanding": "requirement_comprehension", "task_planning": "solution_planning"}
_CONF = ("high", "medium", "low")

# ------------------------------------------------------------------ prompt ----
# A single, fixed system prompt = the shared, cacheable prefix across every call
# (providers that support prompt caching reuse it; correctness does not depend on
# caching). Everything case-specific goes in the user message.


def _error_menu() -> str:
    lines = []
    for cap in JUDGE_CAPS:
        lines.append(f"  {cap}:")
        for et, desc in _CAP_ERRORS[cap].items():
            lines.append(f"    - {et}: {desc}")
        lines.append(
            "    - none: the agent's "
            + ("understanding" if cap == "requirement_understanding" else "planning")
            + " looks adequate from the evidence."
        )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a rigorous evaluator of an AI software-engineering agent's PROCESS on a \
bug-fix / feature task. You assess exactly two capabilities:

- requirement_understanding: did the agent correctly grasp WHAT the issue asks for?
- task_planning: did the agent choose a viable APPROACH / sequence to get there?

For each capability, choose exactly ONE label from this closed set (or `none`):

{_error_menu()}

CRITICAL RULES:
1. You are NOT told whether the task ultimately passed or failed, and you are NOT shown the \
reference solution. Judge ONLY the agent's process from the evidence below. Do not try to guess \
the outcome.
2. Only flag an error you can support with SPECIFIC step numbers from the evidence. If you cannot \
point to evidence, answer `none`.
3. Code *correctness* is judged by a separate system. Do NOT label a misunderstanding just because \
you suspect the final code is wrong. A different-but-valid approach is NOT an error.
4. Some agents' internal reasoning is hidden/encrypted (noted below). When reasoning is hidden, do \
NOT infer that the agent "failed to consider" something from its absence — judge from visible \
actions and messages, and lower your confidence.
5. Be conservative. `none` is the right answer whenever the agent's comprehension/plan looks \
adequate. Reserve `low` confidence for genuinely ambiguous cases.

Respond with a SINGLE JSON object and nothing else, in exactly this shape:
{{"requirement_understanding": {{"error_type": "<label or none>", "confidence": "high|medium|low", \
"evidence_steps": [<int>, ...], "rationale": "<=240 chars"}}, \
"task_planning": {{"error_type": "<label or none>", "confidence": "high|medium|low", \
"evidence_steps": [<int>, ...], "rationale": "<=240 chars"}}}}"""


# --------------------------------------------------------------- evidence -----

_EV_ACTIONS = (SEARCH, READ, EDIT, WRITE, RUN_TEST, RUN_COMMAND)


def build_evidence(
    gold: GoldReference, traj, max_rows: int = 28, max_text: int = 220, focus_steps=None, focus_cap=None
) -> dict:
    """Blinded bounded view backed by complete canonical process evidence."""
    from .evidenceview import build_view

    return build_view(gold.problem_statement, traj, max_rows, max_text, focus_steps, focus_cap)


def _render_user(evidence: dict) -> str:
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
            "the visible messages and actions above, and lower confidence accordingly.",
        ]
    lines += ["", "Return the JSON verdict now."]
    return "\n".join(lines)


def build_messages(gold: GoldReference, traj) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_user(build_evidence(gold, traj))},
    ]


# ----------------------------------------------------------------- parse ------


def _extract_json(text: str):
    """First balanced {...} object in the text (models sometimes wrap it)."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    # A truncated outer object can contain a complete inner object. Never
    # promote that fragment to a standalone verdict after parsing fails.
    return None


def _coerce_cap(cap: str, obj, valid_steps=None) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    et = obj.get("error_type")
    if et not in _CAP_ERRORS[cap]:
        et = "none"
    conf = obj.get("confidence")
    if conf not in _CONF:
        conf = "low"
    raw_steps = obj.get("evidence_steps")
    raw_steps = raw_steps if isinstance(raw_steps, list) else []  # a model may emit {} or a scalar
    steps = []
    for s in raw_steps[:8]:
        try:
            steps.append(int(s))
        except (TypeError, ValueError):
            continue
    # Citations must point at steps that were actually RENDERED into the prompt:
    # a fabricated id (step 9999, a step in the omitted middle) must not satisfy
    # the evidence gate and become blame-eligible.
    if valid_steps is not None:
        steps = [s for s in steps if s in valid_steps]
    rationale = str(obj.get("rationale", ""))[:240]
    # An error with no cited evidence is downgraded to low confidence (rule 2).
    if et != "none" and not steps and conf != "low":
        conf = "low"
    return {"error_type": et, "confidence": conf, "evidence_steps": steps, "rationale": rationale}


def parse_verdict(content: str, valid_steps=None) -> dict:
    obj = _extract_json(content) or {}
    return {cap: _coerce_cap(cap, obj.get(cap), valid_steps) for cap in JUDGE_CAPS}


def _valid_response(response: dict) -> bool:
    obj = _extract_json(response.get("content", ""))
    if response.get("finish_reason") == "length" or not isinstance(obj, dict):
        return False
    for cap in JUDGE_CAPS:
        verdict = obj.get(cap)
        if (
            not isinstance(verdict, dict)
            or verdict.get("error_type") not in (*_CAP_ERRORS[cap], "none")
            or verdict.get("confidence") not in _CONF
            or not isinstance(verdict.get("evidence_steps"), list)
            or not isinstance(verdict.get("rationale"), str)
        ):
            return False
    return True


def rendered_step_ids(evidence: dict) -> set:
    """The step ids the model actually saw (the omission marker row excluded)."""
    return {r["step"] for r in evidence.get("rows", []) if isinstance(r.get("step"), int)}


# ----------------------------------------------------------------- cache ------


def verdict_path(agent: str, instance_id: str, model: str | None = None) -> Path:
    return config.JUDGE_CACHE_DIR / config.model_slug(model) / agent / f"{instance_id}.json"


def _verdict_key(agent, instance_id, model):
    return verdict_fingerprint(agent, instance_id, model, SYSTEM_PROMPT, (build_evidence, _render_user))


def verdict_cached(agent: str, instance_id: str, model: str | None = None) -> dict | None:
    p = verdict_path(agent, instance_id, model)
    if p.exists():
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
            if record.get("parse_failed") or record.get("finish_reason") == "length":
                return None
            return validate_verdict(record, lambda: _verdict_key(agent, instance_id, model), "judge")
        except Exception:
            return None
    return None


def _write_verdict(agent: str, instance_id: str, record: dict, model: str | None = None) -> None:
    p = verdict_path(agent, instance_id, model)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception:
        pass


# ----------------------------------------------------------------- judge ------


def judge_one(
    agent: str,
    instance_id: str,
    client=None,
    model: str | None = None,
    force: bool = False,
    samples: int = 1,
    temperature: float | None = None,
) -> dict | None:
    """Judge a single case. ``client`` defaults to the OpenRouter ``chat``; pass a
    callable with the same signature (e.g. ``mock_client``) to run offline.
    Returns the cached/record dict, or ``None`` if there is no trajectory."""
    if not force:
        cached = verdict_cached(agent, instance_id, model)
        # A panel request must not silently reuse a smaller-panel/single verdict.
        if cached is not None and (samples <= 1 or cached.get("n_votes", 1) >= samples):
            return cached
    try:
        gold = build_gold_reference(instance_id)
    except Exception:
        return None
    traj = load_normalized(agent, instance_id)
    if traj is None or not traj.steps:
        return None

    evidence = build_evidence(gold, traj)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _render_user(evidence)}]
    valid_steps = rendered_step_ids(evidence)
    if client is None:
        from .llm import chat as client  # lazy: avoid needing a key to import

    def _call():
        # Reasoning models spend the token budget on the (separate) reasoning trace and
        # can truncate the JSON -> a degenerate all-`none` parse. Retry once with double
        # the budget before accepting the verdict.
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

    samps, usages, model_name, fr = [], [], model or config.JUDGE_MODEL, None
    parse_failed = False
    for _ in range(max(1, samples)):
        r = _call()
        content = r.get("content", "")
        if not _valid_response(r):
            # Unparseable output is NOT a `none` verdict — retry once, and if it
            # still fails, flag the record so it is distinguishable from a genuine
            # clean bill (and re-judgeable with --force).
            r = _call()
            content = r.get("content", "")
            if not _valid_response(r):
                parse_failed = True
        samps.append(parse_verdict(content, valid_steps))
        usages.append(r.get("usage") or {})
        model_name, fr = r.get("model", model_name), r.get("finish_reason")
    # Panel mode (samples>1): majority-vote with an agreement score; single-sample is
    # byte-for-byte the prior behavior.
    verdict = samps[0] if len(samps) == 1 else aggregate_verdicts(samps)
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
        "model": model_name,
        "verdict": verdict,
        "usage": usage,
        "finish_reason": fr,
        "response_validation_version": 1,
        "input_fingerprint": _verdict_key(agent, instance_id, model),
        "prompt_fingerprint": fingerprint(messages),
        "evidence_view": evidence.get("omissions", {}),
        "cache_provenance": {"status": "verified"},
    }
    if parse_failed:
        record["parse_failed"] = True
    if samples > 1:
        record["n_votes"] = samples
    _write_verdict(agent, instance_id, record, model)
    return record


# Per-(error_type) confidence is gated here: low-confidence verdicts never blame.
def verdict_to_specs(record: dict) -> list[dict]:
    """Pure map: a verdict record -> blame-eligible candidate specs. Only
    medium/high-confidence, non-`none` errors qualify (rule: low confidence and
    evidence-less verdicts never assign blame)."""
    verdict = (record or {}).get("verdict") or {}
    specs = []
    for cap in JUDGE_CAPS:
        v = verdict.get(cap) or {}
        et, conf = v.get("error_type"), v.get("confidence")
        # Abstention bar applied at READ time too (v0.37 review fix): panel
        # records persist agreement + n_votes, so raising AWE_JUDGE_MIN_AGREE
        # after a judging run takes effect on cached aggregates without
        # re-paying the panel — previously the knob silently did nothing to
        # existing caches.
        if (
            config.JUDGE_MIN_AGREE > 0
            and et
            and et != "none"
            and v.get("n_votes", 1) > 1
            and (v.get("agreement") or 0) < config.JUDGE_MIN_AGREE
        ):
            continue
        if et and et != "none" and conf in ("high", "medium"):
            specs.append(
                {
                    "capability": cap,
                    "dimension": _CAP_DIM[cap],
                    "error_type": et,
                    "confidence": conf,
                    "evidence": {
                        "judge_rationale": v.get("rationale", ""),
                        "evidence_steps": v.get("evidence_steps", []),
                        "judge_model": record.get("model", ""),
                        "judge_cache_provenance": (record.get("cache_provenance") or {}).get(
                            "status", "legacy_unverified"
                        ),
                    },
                }
            )
    return specs


def aggregate_verdicts(records: list, min_agree: float | None = None) -> dict:
    """Majority-vote N independent judge samples (panel mode) per capability.

    HIERARCHICAL: first vote error-vs-`none` (so semantically adjacent labels —
    misread vs missed_constraint — cannot split the vote and hand `none` the win
    when the panel unanimously flags *some* error), then plurality among the
    error labels. Aggregate confidence is PANEL-DERIVED: the lower-median vote
    confidence for the winning label, or agreement-scaled for `none` — never the
    hard-coded 'high' the old fallback minted. Pure — no model call.

    ``min_agree`` (default ``config.JUDGE_MIN_AGREE``, env AWE_JUDGE_MIN_AGREE)
    is the selective-voting abstention bar (Raj et al. §6): a winning error
    label whose panel agreement falls below it keeps the label + agreement in
    the record but is demoted to low confidence (blame-ineligible via
    verdict_to_specs) and marked ``abstained`` — precision bought with
    coverage, never a silently discarded vote."""
    from collections import Counter

    if min_agree is None:
        min_agree = config.JUDGE_MIN_AGREE
    verdicts = [(r.get("verdict") or r) for r in records]  # accept records or bare verdicts
    n = len(verdicts) or 1
    order = {"low": 0, "medium": 1, "high": 2}
    out = {}
    for cap in JUDGE_CAPS:
        caps_v = [(v.get(cap) or {}) for v in verdicts]
        ets = [(c.get("error_type") or "none") for c in caps_v]
        err_votes = [e for e in ets if e != "none"]
        if len(err_votes) * 2 > n:  # strict majority: SOME error
            cnt = Counter(err_votes)

            def _rank(lbl, caps_v=caps_v, ets=ets, cnt=cnt):  # count, then confidence, then name
                lbl_votes = [c for c, e in zip(caps_v, ets, strict=False) if e == lbl]
                conf_mass = sum(order.get(v.get("confidence", "low"), 0) for v in lbl_votes)
                return (-cnt[lbl], -conf_mass, lbl)

            et = min(cnt, key=_rank)
            votes = [c for c, e in zip(caps_v, ets, strict=False) if e == et]
            confs = sorted((v.get("confidence", "low") for v in votes), key=lambda c: order.get(c, 0))
            conf = confs[(len(confs) - 1) // 2]  # lower median (conservative)
            # The representative vote supplies the CITATIONS: prefer a cited
            # vote — votes[0] could be the citation-less low-confidence one,
            # producing a blame-eligible verdict with empty evidence_steps
            # (which the audit's citation check would then flag as tamper).
            cited = [v for v in votes if v.get("evidence_steps")]
            rep = (cited or votes)[0]
            if not cited and conf != "low":
                conf = "low"  # panel-level evidence gate
            # v0.32: agreement is load-bearing. "Some error" can win the
            # hierarchical vote while the specific LABEL holds only a
            # plurality (e.g. 1 of 5) — a 20%-agreement label must not carry
            # blame-eligible confidence on the strength of its own votes'
            # self-ratings alone. Without a strict majority for the label
            # itself, confidence caps at low.
            if len(votes) * 2 <= n and conf != "low":
                conf = "low"
            # Selective-voting abstention (v0.36): below the configured
            # agreement bar the label is recorded but never blames.
            abstained = min_agree > 0 and (len(votes) / n) < min_agree
            if abstained and conf != "low":
                conf = "low"
        else:
            et = "none"
            abstained = False
            agreement_none = ets.count("none") / n
            conf = "high" if agreement_none > 0.8 else "medium" if agreement_none > 0.5 else "low"
            rep = next((c for c, e in zip(caps_v, ets, strict=False) if e == "none" and c), {}) or {}
        out[cap] = {
            "error_type": et,
            "confidence": conf,
            "evidence_steps": rep.get("evidence_steps", []) or [],
            "rationale": rep.get("rationale", "") or "",
            "agreement": round(ets.count(et) / n, 3),
            "n_votes": n,
        }
        if abstained:
            out[cap]["abstained"] = True
    return out


def judge_candidates(agent: str, instance_id: str, model: str | None = None):
    """Map a cached verdict to candidate-error specs for the detector.

    Returns ``(specs, available)``. ``available`` is True when a verdict exists
    for this case (so the detector can open the two capabilities' opportunities);
    ``specs`` only includes medium/high-confidence errors (blame-eligible)."""
    rec = verdict_cached(agent, instance_id, model)
    if rec is None:
        return [], False
    if rec.get("parse_failed"):
        # An unparseable-twice verdict is NOT a judged case: treating its
        # all-`none` shell as available would count it as a 0-blame
        # requirement/planning opportunity and silently deflate those rates —
        # the exact "unparseable is not a clean bill" route the flag exists
        # to close. Re-judge with --force to clear it.
        return [], False
    return verdict_to_specs(rec), True


# --------------------------------------------------------------- coverage -----

_warned_model_mismatch = False


def _warn_model_mismatch(model: str | None) -> None:
    """One-time loud warning when the CONFIGURED judge model has no cache but
    other models do — the silent judge-less-run footgun (v0.36), hoisted from
    run_awe.py into the read layer (v0.38) so gen_report / gen_dossier / any
    direct consumer warns too."""
    global _warned_model_mismatch
    if _warned_model_mismatch:
        return
    _warned_model_mismatch = True
    have = sorted(p.name for p in config.JUDGE_CACHE_DIR.glob("*") if p.is_dir() and any(p.rglob("*.json")))
    if have:
        import sys

        print(
            f"WARNING: no judge verdicts cached for AWE_JUDGE_MODEL="
            f"{(model or config.JUDGE_MODEL)!r}; caches exist for: "
            f"{', '.join(have)}. The judge capabilities will be silently "
            f"absent from this run.",
            file=sys.stderr,
        )


def active(agent: str, model: str | None = None) -> bool:
    """True when judge verdicts exist for this agent (so reports include the two
    capabilities). Needs no API key — purely a cache check. Warns (once) when
    the configured model has no cache while other models do."""
    d = config.JUDGE_CACHE_DIR / config.model_slug(model) / agent
    ok = d.is_dir() and any(d.glob("*.json"))
    if not ok:
        _warn_model_mismatch(model)
    return ok


def n_verdicts(agent: str, model: str | None = None) -> int:
    d = config.JUDGE_CACHE_DIR / config.model_slug(model) / agent
    return sum(1 for _ in d.glob("*.json")) if d.is_dir() else 0


# ------------------------------------------------------------------ mock ------


def mock_client(messages: list[dict], model: str | None = None, **_) -> dict:
    """Deterministic, network-free stand-in for ``llm.chat`` — lets the whole
    pipeline (and --mock) run with no API key. Derives a stable pseudo-verdict
    from the evidence so a realistic minority of cases get flagged."""
    user = messages[-1]["content"]
    h = hashlib.md5(user.encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    # Harvest step ids ONLY from clean "[<int>] ..." rows in the evidence region —
    # not from problem-statement lines like "[1292, 1293]" or "[29/Sep/2022]".
    region = user.split("AGENT PROCESS EVIDENCE", 1)[-1]
    steps = [int(m.group(1)) for m in (re.match(r"\[(\d+)\]", ln) for ln in region.splitlines()) if m][:60]
    cite = [steps[n % len(steps)]] if steps else []

    def pick(errors, salt):
        roll = (n >> salt) % 100
        if roll < 12:  # ~12% flagged, mostly medium confidence
            et = list(errors)[(n >> (salt + 3)) % len(errors)]
            return {
                "error_type": et,
                "confidence": "medium" if roll < 9 else "high",
                "evidence_steps": cite,
                "rationale": f"[mock] {et} inferred from cited steps.",
            }
        return {
            "error_type": "none",
            "confidence": "high",
            "evidence_steps": [],
            "rationale": "[mock] process looks adequate.",
        }

    verdict = {"requirement_understanding": pick(REQUIREMENT_ERRORS, 4), "task_planning": pick(PLANNING_ERRORS, 10)}
    return {
        "content": json.dumps(verdict),
        "model": "mock/deterministic",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
    }
