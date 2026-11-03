#!/usr/bin/env python3
"""Price the AWE_JUDGE_MIN_AGREE abstention bar from a 5-vote panel run
(DESIGN §0at; the measurement extraction-candidate #5 was waiting on).

Reads a PANEL verdict cache (n_votes>1 records, each capability verdict
carrying `agreement`) and reports, per abstention bar: how many blame-eligible
error labels survive (coverage), the agreement distribution, and how the
panel-majority labels compare with the shipped single-vote baseline.

Usage:
  python3 scripts/analyze_judge_panel.py <panel_cache_model_dir> [baseline_model_dir]
e.g.
  python3 scripts/analyze_judge_panel.py \
      <scratch>/panel_data/cache/judge/deepseek__deepseek-v4-pro \
      data/cache/judge/deepseek__deepseek-v4-pro
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_trajviz.awe._engine.judge import JUDGE_CAPS

BARS = (0.2, 0.4, 0.6, 0.8, 1.0)


def load(base: Path) -> dict:
    out = {}
    for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in agent_dir.glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("parse_failed"):
                continue
            out[(agent_dir.name, f.stem)] = rec.get("verdict", {})
    return out


def main() -> None:
    panel = load(Path(sys.argv[1]))
    baseline = load(Path(sys.argv[2])) if len(sys.argv) > 2 else {}
    n_panel = sum(1 for v in panel.values() if any((v.get(c) or {}).get("n_votes", 1) > 1 for c in JUDGE_CAPS))
    print(f"panel records: {len(panel)} ({n_panel} with n_votes>1)")

    for cap in JUDGE_CAPS:
        rows = [(k, v.get(cap) or {}) for k, v in panel.items()]
        errs = [(k, c) for k, c in rows if (c.get("error_type") or "none") != "none"]
        eligible = [(k, c) for k, c in errs if c.get("confidence") in ("high", "medium")]
        agr = Counter(round(c.get("agreement") or 0, 1) for _, c in errs)
        print(f"\n== {cap} (n={len(rows)}) ==")
        print(f"  panel error labels: {len(errs)} (blame-eligible under current rules: {len(eligible)})")
        print(f"  agreement distribution over error labels: {dict(sorted(agr.items()))}")
        print(f"  {'bar':>5} {'labels kept':>12} {'blame-eligible kept':>20}")
        for bar in BARS:
            keep = sum(1 for _, c in errs if (c.get("agreement") or 0) >= bar)
            keep_el = sum(1 for _, c in eligible if (c.get("agreement") or 0) >= bar)
            print(f"  {bar:>5.1f} {keep:>12} {keep_el:>20}")

        if baseline:
            both = [k for k, _ in rows if k in baseline]
            agree = flips = base_err = panel_err = co = 0
            for k in both:
                b = (baseline[k].get(cap) or {}).get("error_type") or "none"
                p = (panel[k].get(cap) or {}).get("error_type") or "none"
                agree += b == p
                base_err += b != "none"
                panel_err += p != "none"
                co += b != "none" and p != "none"
                flips += (b != "none") != (p != "none")
            print(
                f"  vs single-vote baseline (n={len(both)}): label agreement "
                f"{agree}/{len(both)}; error labels {base_err} (single) vs "
                f"{panel_err} (panel majority), co-flagged {co}, "
                f"one-sided flips {flips}"
            )


if __name__ == "__main__":
    main()
