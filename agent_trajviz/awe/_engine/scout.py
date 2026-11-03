"""Scout — open-ended failure-mode discovery (IG idea 2).

Borrowed from the Insights Generator paper (Manglik et al., 2026): the fixed
7-capability taxonomy "cannot surface failure modes the taxonomy did not
anticipate." This scout reads a FAILED trace, proposes the failure mode(s) it
exhibits, and flags any that fall OUTSIDE the current taxonomy. Recurring novel
modes are aggregated into a human-review candidate list — the data-driven version
of the paper-extraction "Bucket 2." It NEVER edits the taxonomy or the catalog.

Honesty: discovered modes are LLM HYPOTHESES, not confirmed faults. The scout is
outcome-aware (it is told the attempt failed — discovery, not blame attribution) but
gold-BLINDED (no reference patch, so it reads the agent's process, not the answer).
Novelty is the model's self-report against the taxonomy; the real arbiter is human
review. Mirrors agent_trajviz.awe._engine.judge/arbiter: closed-ish output, evidence-cited steps, per
(model, agent, instance) cache so reads need no API key. True open-ended clustering
(LLM consolidation of near-duplicate labels) is future work; v1 groups by normalized
label.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import config
from .gold import build_gold_reference
from .adapters import load_normalized
from .annotation import ERRORS_BY_CAPABILITY
from .judge import build_evidence, _extract_json

SCOUT_CACHE_DIR = config.PROJECT_DATA / "cache" / "scout"


def _taxonomy_brief() -> str:
    return "\n".join(f"- {cap}: {', '.join(ets)}" for cap, ets in ERRORS_BY_CAPABILITY.items())


SYSTEM_PROMPT = """You analyze ONE failed attempt by an AI software-engineering agent and \
identify the failure mode(s) its PROCESS exhibits. The attempt did NOT resolve the task.

We already classify failures into a fixed taxonomy (given below). Your job is DISCOVERY: \
name the failure mode(s) visible in this trace, and especially flag any mode that the \
existing taxonomy does NOT capture well — a pattern worth adding.

Rules: (1) Only propose a mode you can point to specific step numbers for. (2) Prefer marking \
a mode as covered by an existing capability unless it is genuinely outside the taxonomy. \
(3) Be specific and mechanistic, not generic ("wrong-edit" is too vague). (4) You are NOT \
shown the reference solution — judge only the agent's visible process.

Respond with a SINGLE JSON object, nothing else:
{"modes":[{"label":"<short-kebab-label>","description":"<=160 chars, the mechanism","covered_by":"<capability name or null>","novel":<true|false>,"evidence_steps":[<int>,...]}]}"""


def _render_user(evidence: dict) -> str:
    lines = [
        "EXISTING FAILURE TAXONOMY (capability: error types):",
        _taxonomy_brief(),
        "",
        "TASK (the issue the agent failed to resolve):",
        evidence["problem_statement"].strip(),
        "",
        "AGENT PROCESS EVIDENCE (numbered steps; cite these step numbers):",
    ]
    for r in evidence["rows"]:
        tool = (" " + r["tool"]) if r.get("tool") else ""
        lines.append(f"[{r['step']}] {r['kind']}{tool}: {r.get('text', '')}".rstrip())
    if evidence["hidden_reasoning"]:
        lines += ["", "NOTE: internal reasoning is HIDDEN/encrypted — judge only visible actions/messages."]
    lines += [
        "",
        "This attempt FAILED. Identify the failure mode(s) and flag any not covered by the "
        "taxonomy above. Return the JSON now.",
    ]
    return "\n".join(lines)


def build_messages(gold, traj) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_user(build_evidence(gold, traj))},
    ]


_CAPS = set(ERRORS_BY_CAPABILITY)


def _norm_label(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")
    return s[:60] or "unlabeled"


def parse_modes(content: str, valid_steps=None) -> list[dict]:
    """Parse + sanitize the scout's mode list. Each mode: normalized label, capped
    description, covered_by (a valid capability or None), novel bool consistent with
    covered_by, int evidence_steps (filtered to steps actually rendered into the
    prompt, so fabricated citations don't survive into the candidate list).
    Drops malformed entries."""
    obj = _extract_json(content) or {}
    raw = obj.get("modes") if isinstance(obj, dict) else None
    out = []
    for m in (raw if isinstance(raw, list) else [])[:6]:
        if not isinstance(m, dict) or not m.get("label"):
            continue
        cov = m.get("covered_by")
        cov = cov if cov in _CAPS else None
        steps = []
        for s in (m.get("evidence_steps") if isinstance(m.get("evidence_steps"), list) else [])[:8]:
            try:
                steps.append(int(s))
            except (TypeError, ValueError):
                continue
        if valid_steps is not None:
            steps = [s for s in steps if s in valid_steps]
        # novel iff the model says so AND it did not map to a real capability.
        novel = bool(m.get("novel")) and cov is None
        out.append(
            {
                "label": _norm_label(m.get("label")),
                "description": str(m.get("description", ""))[:160],
                "covered_by": cov,
                "novel": novel,
                "evidence_steps": steps,
            }
        )
    return out


def verdict_path(agent: str, instance_id: str, model: str | None = None) -> Path:
    return SCOUT_CACHE_DIR / config.model_slug(model) / agent / f"{instance_id}.json"


def cached(agent: str, instance_id: str, model: str | None = None) -> dict | None:
    p = verdict_path(agent, instance_id, model)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def scout_one(agent: str, instance_id: str, client=None, model: str | None = None, force: bool = False) -> dict | None:
    """Discover failure modes for one FAILED case. Cached per (model, agent,
    instance). Returns the record, or None if there is no trajectory."""
    if not force:
        c = cached(agent, instance_id, model)
        if c is not None:
            return c
    try:
        gold = build_gold_reference(instance_id)
    except Exception:
        return None
    traj = load_normalized(agent, instance_id)
    if traj is None or not traj.steps:
        return None
    if client is None:
        from .llm import chat as client
    evidence = build_evidence(gold, traj)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _render_user(evidence)}]
    valid_steps = {r_["step"] for r_ in evidence["rows"] if isinstance(r_["step"], int)}
    r = client(messages, model=model, max_tokens=config.JUDGE_MAX_TOKENS, response_format={"type": "json_object"})
    if r.get("finish_reason") == "length":
        r = client(
            messages, model=model, max_tokens=config.JUDGE_MAX_TOKENS * 2, response_format={"type": "json_object"}
        )
    record = {
        "instance_id": instance_id,
        "agent": agent,
        "model": r.get("model", model or config.JUDGE_MODEL),
        "modes": parse_modes(r.get("content", ""), valid_steps),
        "usage": r.get("usage") or {},
        "finish_reason": r.get("finish_reason"),
    }
    if r.get("content") and _extract_json(r.get("content", "")) is None:
        record["parse_failed"] = True
    p = verdict_path(agent, instance_id, model)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    except Exception:
        pass
    return record


def aggregate_novel(records: list) -> list[dict]:
    """Group the NOVEL (uncovered) modes across cases by normalized label, with
    prevalence + example cases + a representative description. Sorted by count. This
    is the human-review candidate list — recurring modes the taxonomy may be missing."""
    groups: dict[str, dict] = {}
    for rec in records:
        if not rec:
            continue
        for m in rec.get("modes", []):
            if not m.get("novel"):
                continue
            g = groups.setdefault(m["label"], {"label": m["label"], "count": 0, "examples": [], "descriptions": []})
            g["count"] += 1
            if len(g["examples"]) < 5:
                g["examples"].append(f"{rec.get('agent')}/{rec.get('instance_id')}")
            if m.get("description"):
                g["descriptions"].append(m["description"])
    out = list(groups.values())
    for g in out:
        g["description"] = g["descriptions"][0] if g["descriptions"] else ""
        # keep a few alternates for the human reviewer (semantic-duplicate hints)
        g["alt_descriptions"] = g["descriptions"][1:3]
        del g["descriptions"]
    out.sort(key=lambda g: -g["count"])
    return out


# ------------------------------------------------------- label consolidation --

CONSOLIDATE_PROMPT = """You are consolidating a list of candidate failure-mode labels produced by \
independent runs of a trace-mining scout. Different runs name the SAME underlying failure mode with \
different labels (e.g. analysis-loop / analysis-without-action-loop / repeated-root-cause-analysis).

Group the labels: each group = one underlying failure mode. Do NOT merge genuinely distinct modes; \
when unsure, keep labels separate. Every input label must appear in exactly one group.

Respond with a SINGLE JSON object, nothing else:
{"groups": [{"canonical": "<kebab-case-label>", "members": ["<input label>", ...]}, ...]}"""


def consolidate_novel(novel: list, client=None, model: str | None = None) -> list:
    """LLM pass merging semantically-duplicate labels in an aggregate_novel list
    (the v1 exact-label dedup split e.g. the analysis-loop cluster ~6 ways —
    DESIGN §0r). Falls back to the UNCONSOLIDATED list on any parse/validation
    failure, and never invents labels: groups are validated against the input
    label set, uncovered labels become singleton groups."""
    if len(novel) < 2:
        return novel
    if client is None:
        from .llm import chat as client
    by_label = {g["label"]: g for g in novel}
    lines = [f"- {g['label']} — {g.get('description', '')} (seen {g['count']}x)" for g in novel]
    messages = [
        {"role": "system", "content": CONSOLIDATE_PROMPT},
        {"role": "user", "content": "\n".join(lines) + "\n\nReturn the JSON grouping now."},
    ]
    r = client(messages, model=model, max_tokens=config.JUDGE_MAX_TOKENS, response_format={"type": "json_object"})
    obj = _extract_json(r.get("content", "")) or {}
    raw_groups = obj.get("groups") if isinstance(obj, dict) else None
    if not isinstance(raw_groups, list):
        return novel
    merged, seen = [], set()
    for rg in raw_groups:
        if not isinstance(rg, dict):
            continue
        members = [m for m in (rg.get("members") or []) if m in by_label and m not in seen]
        if not members:
            continue
        seen.update(members)
        parts = sorted((by_label[m] for m in members), key=lambda g: -g["count"])
        top = parts[0]
        canonical = _norm_label(str(rg.get("canonical") or top["label"]))
        examples: list = []
        for p_ in parts:
            for e in p_["examples"]:
                if e not in examples and len(examples) < 5:
                    examples.append(e)
        merged.append(
            {
                "label": canonical,
                "count": sum(p_["count"] for p_ in parts),
                "examples": examples,
                "description": top.get("description", ""),
                "alt_descriptions": top.get("alt_descriptions", []),
                "members": [p_["label"] for p_ in parts],
            }
        )
    for lbl, g in by_label.items():  # anything the model dropped: singleton
        if lbl not in seen:
            merged.append({**g, "members": [lbl]})
    merged.sort(key=lambda g: -g["count"])
    return merged


def active(agent: str, model: str | None = None) -> bool:
    d = SCOUT_CACHE_DIR / config.model_slug(model) / agent
    return d.is_dir() and any(d.glob("*.json"))


def n_scout(agent: str, model: str | None = None) -> int:
    d = SCOUT_CACHE_DIR / config.model_slug(model) / agent
    return sum(1 for _ in d.glob("*.json")) if d.is_dir() else 0


def candidate_report_md(novel: list, n_records: int) -> str:
    """Render the aggregated novel modes as a human-review markdown file."""
    L = [
        "# Candidate failure modes (scout discovery)",
        "_LLM-proposed failure modes the current taxonomy may not capture, mined from "
        f"{n_records} failed traces. **Unverified hypotheses for human review** — novelty is "
        "the model's self-report, gold-blinded. Promote a mode into the taxonomy "
        "(`awe/annotation.py`) only after human confirmation; the scout never edits it._",
        "",
    ]
    if not novel:
        L += ["_No novel (uncovered) modes were proposed._", ""]
        return "\n".join(L)
    L += ["| Candidate mode | seen | example cases | mechanism |", "|---|---|---|---|"]
    for g in novel:
        ex = ", ".join(f"`{e}`" for e in g["examples"][:3])
        desc = g["description"]
        members = g.get("members") or []
        if len(members) > 1:
            desc += " _(consolidates: " + ", ".join(f"`{m}`" for m in members) + ")_"
        L.append(f"| `{g['label']}` | {g['count']} | {ex} | {desc} |")
    L.append("")
    return "\n".join(L)


def mock_client(messages: list[dict], model: str | None = None, **_) -> dict:
    """Deterministic, network-free stand-in for llm.chat — exercises the full scout
    path with no API key. Most cases map to an existing capability; a stable minority
    propose a novel mode (so aggregation/dedup is testable)."""
    user = messages[-1]["content"]
    h = int(hashlib.md5(user.encode("utf-8")).hexdigest()[:8], 16)
    cite = [(h % 13) + 1]
    if h % 100 < 30:
        modes = [
            {
                "label": "premature implementation without repro",
                "description": "[mock] edited before reproducing the failing case",
                "covered_by": None,
                "novel": True,
                "evidence_steps": cite,
            }
        ]
    else:
        modes = [
            {
                "label": "wrong-edit",
                "description": "[mock] right file, wrong change",
                "covered_by": "code_editing",
                "novel": False,
                "evidence_steps": cite,
            }
        ]
    return {
        "content": json.dumps({"modes": modes}),
        "model": "mock/deterministic",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        "finish_reason": "stop",
    }


def mock_consolidate_client(messages: list[dict], model: str | None = None, **_) -> dict:
    """Deterministic stand-in for the consolidation pass: groups input labels by
    their FIRST kebab token (analysis-loop + analysis-without-action -> one group).
    Network-free; lets --mock and the tests exercise consolidate_novel end-to-end."""
    labels = [ln[2:].split(" — ", 1)[0].strip() for ln in messages[-1]["content"].splitlines() if ln.startswith("- ")]
    groups: dict = {}
    for lbl in labels:
        groups.setdefault(lbl.split("-", 1)[0], []).append(lbl)
    out = [{"canonical": v[0] if len(v) == 1 else f"{k}-consolidated", "members": v} for k, v in groups.items()]
    return {
        "content": json.dumps({"groups": out}),
        "model": "mock/deterministic",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        "finish_reason": "stop",
    }
