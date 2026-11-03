#!/usr/bin/env python3
"""Cross-model judge agreement — the panel LLM_JUDGE.md §2 recommends, as tooling.

Compares two judge models' cached verdicts case-by-case (per capability):
error-vs-none agreement + Cohen's κ, exact-label agreement, and the
consensus/disagreement case lists. "Keep only labels both models agree on" is
then a mechanical filter (the consensus list), not a manual suggestion.

Reads caches only — no API key needed.

Run:  python3 scripts/judge_agreement.py [--model-a SLUG] [--model-b SLUG] [--write]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.agreement import cohen_kappa
from agent_trajviz.awe._engine.judge import JUDGE_CAPS

OUT = Path(__file__).resolve().parent.parent / "data" / "discovery"


def load_verdicts(model: str) -> dict:
    """{(agent, instance): verdict-dict} for every cached case of one model."""
    base = config.JUDGE_CACHE_DIR / config.model_slug(model)
    out = {}
    for p in base.rglob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (rec.get("agent") or p.parent.name, rec.get("instance_id") or p.stem)
        out[key] = rec.get("verdict") or {}
    return out


def compare(model_a: str, model_b: str) -> dict:
    va, vb = load_verdicts(model_a), load_verdicts(model_b)
    shared = sorted(set(va) & set(vb))
    caps = {}
    consensus, disagree = [], []
    for cap in JUDGE_CAPS:
        la, lb = [], []
        for key in shared:
            ea = (va[key].get(cap) or {}).get("error_type") or "none"
            eb = (vb[key].get(cap) or {}).get("error_type") or "none"
            la.append(ea)
            lb.append(eb)
            if ea != "none" and ea == eb:
                consensus.append((key, cap, ea))
            elif ea != eb and (ea != "none" or eb != "none"):
                disagree.append((key, cap, ea, eb))
        bin_a = ["err" if x != "none" else "none" for x in la]
        bin_b = ["err" if x != "none" else "none" for x in lb]
        n = len(shared)
        caps[cap] = {
            "n": n,
            "raw_agreement": (sum(1 for x, y in zip(la, lb, strict=False) if x == y) / n) if n else None,
            "error_vs_none_kappa": cohen_kappa(bin_a, bin_b),
            "flagged_a": sum(1 for x in la if x != "none"),
            "flagged_b": sum(1 for x in lb if x != "none"),
        }
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_shared": len(shared),
        "capabilities": caps,
        "consensus": consensus,
        "disagreements": disagree,
    }


def to_md(r: dict) -> str:
    L = [
        f"# Cross-model judge agreement — `{r['model_a']}` vs `{r['model_b']}`",
        "",
        f"_{r['n_shared']} cases judged by both models (outcome-blinded, closed "
        "taxonomy). κ is Cohen's kappa on the binary error-vs-none decision; with "
        "yields this low, κ is unstable — read the raw counts. **Consensus labels** "
        "(both models, same label) are the panel-filtered set LLM_JUDGE.md "
        "recommends trusting most._",
        "",
        "| Capability | n | raw agreement | κ (err-vs-none) | flagged by A | flagged by B |",
        "|---|---|---|---|---|---|",
    ]
    for cap, c in r["capabilities"].items():
        k = "—" if c["error_vs_none_kappa"] is None else f"{c['error_vs_none_kappa']:.2f}"
        ra = "—" if c["raw_agreement"] is None else f"{100 * c['raw_agreement']:.1f}%"
        L.append(f"| {cap} | {c['n']} | {ra} | {k} | {c['flagged_a']} | {c['flagged_b']} |")
    L += ["", f"## Consensus labels ({len(r['consensus'])})", ""]
    for (agent, inst), cap, et in r["consensus"]:
        L.append(f"- `{agent}` / `{inst}` — {cap}: **{et}**")
    if not r["consensus"]:
        L.append("_none — the two models never agreed on a specific error label_")
    L += ["", f"## Disagreements ({len(r['disagreements'])})", ""]
    for (agent, inst), cap, ea, eb in r["disagreements"][:40]:
        L.append(f"- `{agent}` / `{inst}` — {cap}: A={ea} vs B={eb}")
    if len(r["disagreements"]) > 40:
        L.append(f"- … and {len(r['disagreements']) - 40} more")
    L.append("")
    return "\n".join(L)


def main() -> None:
    args = sys.argv[1:]

    def _opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    model_a = _opt("--model-a", "z-ai/glm-5.2")
    model_b = _opt("--model-b", "deepseek/deepseek-v4-flash")
    r = compare(model_a, model_b)
    md = to_md(r)
    print(md)
    if "--write" in args:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "judge_cross_model.md").write_text(md, encoding="utf-8")
        print(f"wrote {OUT / 'judge_cross_model.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
