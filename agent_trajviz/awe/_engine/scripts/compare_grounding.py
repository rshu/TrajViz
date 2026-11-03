#!/usr/bin/env python3
"""Measure the impact of the grounding fixes on the real ARGUS run.

Prints, per agent, the NEW capability scores and the primary-cause distribution
(objective slice; use_judge=False so this isolates the rule-layer changes). Also
populates the trajectory-signal cache as a side effect. Run with AWE_ARGUS_ROOT set.
"""

import json
from collections import Counter

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.detect import detect
from agent_trajviz.awe._engine.rootcause import build_blame
from agent_trajviz.awe._engine.scoring import agent_capability_scores

CAPS = ["code_localization", "code_editing", "code_verification", "self_repair_loop", "tool_use"]


def collect() -> dict:
    """Compute the objective comparison for the configured corpus."""
    out = {}
    for agent in config.AGENTS:
        sc = agent_capability_scores(agent, use_trajectory=True, use_judge=False)
        prim = Counter()
        omit = 0
        for inst in config.list_instances():
            if resolved(agent, inst) is not False:
                continue
            d = detect(inst, agent, use_trajectory=True, use_judge=False)
            b = build_blame(d)
            if b.get("primary"):
                prim[b["primary"]["capability"]] += 1
            else:
                prim["(conjunctive)"] += 1
            if any(c["error_type"] == "relevant_change_omitted" for c in d["candidates"]):
                omit += 1
        out[agent] = {
            "fail_pct": round(100 * sc["failure_rate"], 1),
            "scores": {k: round(sc["capabilities"].get(k, {}).get("score", float("nan")), 1) for k in CAPS},
            "primary": dict(prim),
            "cases_with_omission": omit,
        }
    return out


def main() -> None:
    out = collect()
    for agent, result in out.items():
        print(f"\n=== {agent} (fail {result['fail_pct']}%) ===")
        print("  scores:", result["scores"])
        print("  primary:", result["primary"])
        print("  cases flagging relevant_change_omitted:", result["cases_with_omission"])

    # aggregate primary distribution across agents
    agg = Counter()
    for a in out.values():
        for k, v in a["primary"].items():
            agg[k] += v
    print("\n=== AGGREGATE primary-cause distribution (all agents, objective slice) ===")
    for k, v in agg.most_common():
        print(f"  {k}: {v}")

    destination = config.OUTPUT_DATA / "grounding_compare.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
