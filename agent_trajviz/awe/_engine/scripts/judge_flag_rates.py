"""Confidence-stratified judge flag rates on failed vs solved cases.

The measurement behind the v0.32 rule that judge capabilities are never
primary-eligible: the old design let a HIGH-self-reported-confidence judge
verdict seize primary blame and demote deductive facts, but nobody had computed
whether the confidence gates select a reliable subset. They do not — the
selected subset is UNSTABLE across judge models:

    deepseek-v4-pro   blame-eligible: solved 1.43% vs failed 1.17%
                      high-confidence: solved 0.72% vs failed 0.29% (2.5x MORE
                      per solved control — anti-discriminative)
    z-ai/glm-5.2      blame-eligible: solved 0.39% vs failed 0.58%
    (2026-07-12)      high-confidence: solved 0.17% vs failed 0.29% (direction
                      flipped, at 3-vs-2 case yields)

Either way the tier that was empowered to override deductions is not a
reliability signal. Self-reported confidence is not a reliability
signal here (consistent with the cross-model kappa ~ 0 measurement in
data/discovery/judge_cross_model.md).

Usage:  python3 scripts/judge_flag_rates.py [--model z-ai/glm-5.2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import resolved


def flag_rates(model_slug_dir: Path) -> dict:
    stats = {True: {"n": 0, "eligible": 0, "high": 0}, False: {"n": 0, "eligible": 0, "high": 0}}
    for f in sorted(model_slug_dir.glob("*/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        res = resolved(d["agent"], d["instance_id"])
        if res is None:
            continue
        s = stats[res]
        s["n"] += 1
        eligible = high = False
        for v in (d.get("verdict") or {}).values():
            et, conf = v.get("error_type"), v.get("confidence")
            cited = bool(v.get("evidence_steps"))
            if et and et != "none" and conf in ("high", "medium") and cited:
                eligible = True
                if conf == "high":
                    high = True
        s["eligible"] += eligible
        s["high"] += high
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="z-ai/glm-5.2")
    args = ap.parse_args()
    slug = args.model.replace("/", "__")
    base = config.PROJECT_DATA / "cache" / "judge" / slug
    if not base.exists():
        raise SystemExit(f"no judge cache at {base}")
    stats = flag_rates(base)
    print(f"model: {args.model}")
    print(f"{'population':10s} {'n':>5s} {'blame-eligible':>16s} {'high-confidence':>17s}")
    for res, label in ((False, "failed"), (True, "solved")):
        s = stats[res]
        if not s["n"]:
            continue
        print(
            f"{label:10s} {s['n']:5d} "
            f"{s['eligible']:5d} ({s['eligible'] / s['n']:6.2%}) "
            f"{s['high']:5d} ({s['high'] / s['n']:6.2%})"
        )
    f, so = stats[False], stats[True]
    if f["n"] and so["n"] and f["high"]:
        ratio = (so["high"] / so["n"]) / (f["high"] / f["n"])
        print(f"\nhigh-confidence solved/failed rate ratio: {ratio:.2f} (> 1 = anti-discriminative)")


if __name__ == "__main__":
    main()
