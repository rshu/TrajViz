#!/usr/bin/env python3
"""Cross-model judge agreement under a FIXED, stated protocol (DESIGN §0ar).

Rao & Callison-Burch (arXiv:2606.00093) show protocol choices alone — scale,
case retention, abstention handling, pooling — can move reported agreement
0.55→0.90 and carry κ across zero WITHOUT changing a verdict. AWE's headline
"κ≈0 cross-model" therefore gets a protocol card:

  PROTOCOL: per-capability; cases = the full overlap of both models' verdict
  caches for one agent; parse-failed records EXCLUDED (both models); labels =
  the closed error set + 'none'; no confidence filtering; no pooling across
  capabilities. Metrics: raw agreement (po), Cohen's κ, PABAK
  (prevalence-and-bias-adjusted κ = 2·po−1), positive agreement on the ERROR
  class (PA = 2a/(2a+b+c)), and per-model error prevalence.

The point of the battery: with ~3% error prevalence, po and PABAK are
dominated by none–none agreement while κ and PA isolate the question that
matters — do the models agree on WHICH cases are errors? (Spoiler recorded in
§0ar: po=0.96, PABAK≈0.92, κ≈−0.01, PA=0.0 — the models never co-flag.)

Reproduce:  python3 scripts/judge_kappa_protocol.py [agent]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.judge import JUDGE_CAPS

MODELS = ("deepseek__deepseek-v4-pro", "deepseek__deepseek-v4-flash")


def verdicts(model: str, agent: str) -> dict:
    base = config.JUDGE_CACHE_DIR / model / agent
    out = {}
    for p in base.glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("parse_failed"):
            continue  # protocol: excluded for BOTH models
        out[p.stem] = rec.get("verdict", {})
    return out


def main() -> None:
    agent = sys.argv[1] if len(sys.argv) > 1 else "claude_code"
    va, vb = (verdicts(m, agent) for m in MODELS)
    both = sorted(set(va) & set(vb))
    print(f"agent={agent}  models={MODELS}  overlap={len(both)} (parse-failed excluded)")
    for cap in JUDGE_CAPS:
        pairs = []
        for i in both:
            ea = (va[i].get(cap) or {}).get("error_type") or "none"
            eb = (vb[i].get(cap) or {}).get("error_type") or "none"
            pairs.append((ea, eb))
        n = len(pairs)
        po = sum(a == b for a, b in pairs) / n
        ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        pe = sum(ca[l] * cb.get(l, 0) for l in ca) / (n * n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        pabak = 2 * po - 1
        a_err = sum(1 for x, y in pairs if x != "none" and y != "none")
        b_only = sum(1 for x, y in pairs if x == "none" and y != "none")
        a_only = sum(1 for x, y in pairs if x != "none" and y == "none")
        pa = 2 * a_err / (2 * a_err + a_only + b_only) if (a_err + a_only + b_only) else float("nan")
        prev_a = sum(1 for x, _ in pairs if x != "none") / n
        prev_b = sum(1 for _, y in pairs if y != "none") / n
        print(f"\n{cap} (n={n}):")
        print(f"  po={po:.3f}  kappa={kappa:.3f}  PABAK={pabak:.3f}  PA(error)={pa:.3f}")
        print(
            f"  error prevalence: {MODELS[0].split('__')[-1]}={prev_a:.3f}, "
            f"{MODELS[1].split('__')[-1]}={prev_b:.3f}; "
            f"co-flagged={a_err}, one-sided={a_only + b_only}"
        )


if __name__ == "__main__":
    main()
