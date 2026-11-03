#!/usr/bin/env python3
"""Tool-error capture fidelity: per-agent mechanical tool-call error rate.

The point of this report is CROSS-AGENT CAPTURE PARITY. Before the error-flag fix,
Codex registered 0 tool errors purely because its result payloads were never parsed
for failure — a measurement gap masquerading as a flawless agent. Every agent should
now surface tool errors (recovered or not), so the signal is comparable.

Behavioral, not blame: a tool error becomes a failure cause (`tool_failure_not_retried`)
only when the agent then gave up.

Run:  python3 scripts/tool_error_fidelity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.trajsignals import tool_behavior


def main():
    instances = config.list_instances()
    print(f"Tool-error capture fidelity — {len(instances)} tasks/agent\n")
    hdr = f"{'agent':16s}{'calls':>9s}{'errors':>8s}{'rate':>8s}{'tasks_w_err':>14s}{'dominant':>18s}"
    print(hdr)
    print("-" * len(hdr))
    rows = {a: tool_behavior(a, instances) for a in config.AGENTS}
    for a in config.AGENTS:
        tb = rows[a]
        subs = tb["subtypes"]
        dom = max(subs, key=subs.get) if subs else "—"
        print(
            f"{a:16s}{tb['n_tool_calls']:>9,d}{tb['n_tool_call_errors']:>8,d}"
            f"{100 * tb['tool_error_rate']:>7.1f}%"
            f"{tb['cases_with_error']:>8d} ({100 * tb['case_rate']:>3.0f}%){dom:>18s}"
        )
    print("\nSubtype breakdown:")
    for a in config.AGENTS:
        subs = rows[a]["subtypes"]
        if subs:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(subs.items(), key=lambda kv: -kv[1]))
            print(f"  {a:16s} {parts}")
    zero = [a for a in config.AGENTS if rows[a]["n_tool_call_errors"] == 0 and rows[a]["n_tool_calls"] > 0]
    if zero:
        print(f"\n⚠ structurally-zero agents (possible capture gap): {', '.join(zero)}")
    else:
        print("\n✓ all agents with tool calls surface tool errors — no capture gap")


if __name__ == "__main__":
    main()
