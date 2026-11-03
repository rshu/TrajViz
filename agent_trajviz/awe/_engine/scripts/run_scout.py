#!/usr/bin/env python3
"""Run the SCOUT to discover failure modes the fixed taxonomy may miss (IG idea 2).

Reads each FAILED trace, proposes the failure mode(s) it exhibits, and flags any
outside the current taxonomy. Recurring novel modes are aggregated into a
human-review candidate list (data/discovery/candidate_failure_modes.{md,json}).
It NEVER edits the taxonomy — promotion is a human decision.

Setup (same OpenRouter model as the judge/arbiter):
    export OPENROUTER_API_KEY=sk-or-...
    export AWE_JUDGE_MODEL=z-ai/glm-5.2

Examples:
    python3 scripts/run_scout.py --dry-run            # failed cases + token estimate
    python3 scripts/run_scout.py --mock               # offline plumbing test (no key)
    python3 scripts/run_scout.py --agent codex        # one agent, real calls
    python3 scripts/run_scout.py                      # all agents, then write the report
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.scout import (
    scout_one,
    cached,
    build_messages,
    mock_client,
    aggregate_novel,
    candidate_report_md,
    n_scout,
)
from agent_trajviz.awe._engine.llm import LLMError

OUT = Path(__file__).resolve().parent.parent / "data" / "discovery"


def targets(agent: str, limit):
    out = [i for i in config.list_instances() if resolved(agent, i) is False]
    return out[:limit] if limit else out


def dry_run(agents, limit):
    print(f"{'agent':<16} {'failed':>8} {'est_in_tok':>12}")
    grand = gi = 0
    for a in agents:
        ts = targets(a, limit)
        tok = 0
        for inst in ts:
            try:
                gold = build_gold_reference(inst)
                traj = load_normalized(a, inst)
            except Exception:
                continue
            if traj and traj.steps:
                tok += sum(len(m["content"]) for m in build_messages(gold, traj)) // 4
        grand += len(ts)
        gi += tok
        print(f"{a:<16} {len(ts):>8} {tok:>12,}")
    print("-" * 38)
    print(f"{'TOTAL':<16} {grand:>8} {gi:>12,}  (+~{grand * 150:,} output)")


def write_report(agents, model, consolidate=False, mock=False):
    records = []
    for a in agents:
        for inst in targets(a, None):
            r = cached(a, inst, model)
            if r:
                records.append(r)
    novel = aggregate_novel(records)
    n_raw = len(novel)
    if consolidate and novel:
        from agent_trajviz.awe._engine.scout import consolidate_novel, mock_consolidate_client

        client = mock_consolidate_client if mock else None
        novel = consolidate_novel(novel, client=client, model=model)
        merged = sum(1 for g in novel if len(g.get("members") or []) > 1)
        print(f"consolidated {n_raw} raw labels -> {len(novel)} modes ({merged} merged groups)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_failure_modes.md").write_text(candidate_report_md(novel, len(records)), encoding="utf-8")
    (OUT / "candidate_failure_modes.json").write_text(json.dumps(novel, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'candidate_failure_modes.md'}  ({len(novel)} candidate modes from {len(records)} traces)")


def run(agents, limit, mock, model, force, workers, consolidate=False):
    client = mock_client if mock else None
    grand = {"done": 0, "skipped": 0, "errors": 0}
    for a in agents:
        ts = targets(a, limit)
        todo = [i for i in ts if force or cached(a, i, model) is None]
        grand["skipped"] += len(ts) - len(todo)
        print(f"\n[{a}] {len(ts)} failed · {len(todo)} to scout" + (" · MOCK" if mock else ""))
        done = errs = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(scout_one, a, i, client, model, force) for i in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    if fut.result() is not None:
                        done += 1
                except LLMError as e:
                    errs += 1
                    if errs <= 3:
                        print(f"  ! {e}")
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)} scouted...")
        grand["done"] += done
        grand["errors"] += errs
        print(f"[{a}] scouted {done}, errors {errs}, cached now {n_scout(a, model)}")
    print("\n" + "=" * 48)
    print(f"scouted {grand['done']} · skipped(cached) {grand['skipped']} · errors {grand['errors']}")
    write_report(agents, model, consolidate=consolidate, mock=mock)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent")
    p.add_argument("--limit", type=int)
    p.add_argument("--mock", action="store_true", help="offline deterministic scout (no API key)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model", help="override AWE_JUDGE_MODEL for this run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--consolidate", action="store_true", help="LLM pass merging semantically-duplicate mode labels")
    args = p.parse_args()

    agents = [args.agent] if args.agent else config.AGENTS
    if args.dry_run:
        dry_run(agents, args.limit)
        return
    if not args.mock and not config.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY not set. Export it, or use --mock for an offline test.")
    model = args.model or ("mock" if args.mock else config.JUDGE_MODEL)
    print(f"scout model: {model}" + ("  (offline mock)" if args.mock else ""))
    run(agents, args.limit, args.mock, model, args.force, args.workers, consolidate=args.consolidate)


if __name__ == "__main__":
    main()
