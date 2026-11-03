#!/usr/bin/env python3
"""Run the LLM judge over agent trajectories to score the two judgment-heavy
capabilities (Requirement Understanding, Task Planning).

The judge is OUTCOME-BLINDED and taxonomy-constrained (see awe/judge.py). It runs
over each agent's FAILED tasks by default (the diagnosis set); verdicts are cached
per (model, agent, instance), so the run is resumable and the report pipeline then
reads them back with no API key.

Setup (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...
    export AWE_JUDGE_MODEL=anthropic/claude-sonnet-4.5   # or openai/gpt-5, google/gemini-2.5-pro, ...

Examples:
    python3 scripts/run_judge.py --dry-run                 # token estimate, no calls
    python3 scripts/run_judge.py --mock                    # offline plumbing test (no key)
    python3 scripts/run_judge.py --agent claude_code       # one agent, real calls
    python3 scripts/run_judge.py                           # all agents, all failed tasks
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.judge import (
    judge_one,
    verdict_cached,
    build_messages,
    mock_client,
    n_verdicts,
    JUDGE_CAPS,
)
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.llm import LLMError


def target_instances(agent: str, include_resolved: bool, limit: int | None) -> list[str]:
    out = []
    for inst in config.list_instances():
        r = resolved(agent, inst)
        if r is None:
            continue
        if r is False or include_resolved:
            out.append(inst)
    return out[:limit] if limit else out


def estimate_tokens(agent: str, inst: str) -> int | None:
    """~chars/4 input-token estimate for the dry run (no network)."""
    try:
        gold = build_gold_reference(inst)
    except Exception:
        return None
    traj = load_normalized(agent, inst)
    if traj is None or not traj.steps:
        return None
    chars = sum(len(m["content"]) for m in build_messages(gold, traj))
    return chars // 4


def dry_run(agents, include_resolved, limit, price_in, price_out) -> None:
    total_in = total_cases = 0
    print(f"{'agent':<16} {'cases':>6} {'est_in_tok':>12}")
    for a in agents:
        insts = target_instances(a, include_resolved, limit)
        tok = sum(t for t in (estimate_tokens(a, i) for i in insts) if t)
        total_in += tok
        total_cases += len(insts)
        print(f"{a:<16} {len(insts):>6} {tok:>12,}")
    out_tok = total_cases * 350  # ~verdict size
    print("-" * 36)
    print(f"{'TOTAL':<16} {total_cases:>6} {total_in:>12,}  (+~{out_tok:,} output)")
    if price_in is not None and price_out is not None:
        cost = total_in / 1e6 * price_in + out_tok / 1e6 * price_out
        print(f"est. cost @ ${price_in}/${price_out} per Mtok: ${cost:,.2f}")
    else:
        print("pass --price-in/--price-out (USD per Mtok for your model) to estimate cost.")
    print("note: prompt caching on the shared system prompt reduces real input cost.")


def run(agents, include_resolved, limit, mock, model, force, workers, panel=1, temperature=None) -> None:
    client = mock_client if mock else None
    grand = {"done": 0, "skipped": 0, "errors": 0, "in": 0, "out": 0, "cost": 0.0}
    if panel > 1:
        print(f"PANEL mode: {panel} votes/case @ temperature {temperature} (majority + agreement)")

    for a in agents:
        insts = target_instances(a, include_resolved, limit)
        todo = [i for i in insts if force or verdict_cached(a, i, model) is None]
        skipped = len(insts) - len(todo)
        grand["skipped"] += skipped
        print(f"\n[{a}] {len(insts)} target · {skipped} cached · {len(todo)} to judge" + (" · MOCK" if mock else ""))
        done = errs = 0

        def work(inst, a=a):
            return inst, judge_one(
                a, inst, client=client, model=model, force=force, samples=panel, temperature=temperature
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, i) for i in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    inst, rec = fut.result()
                    if rec is None:
                        continue
                    done += 1
                    u = rec.get("usage") or {}
                    grand["in"] += u.get("prompt_tokens", 0) or 0
                    grand["out"] += u.get("completion_tokens", 0) or 0
                    grand["cost"] += float(u.get("cost", 0) or 0)
                except LLMError as e:
                    errs += 1
                    if errs <= 3:
                        print(f"  ! {e}")
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)} judged...")
        grand["done"] += done
        grand["errors"] += errs
        print(f"[{a}] judged {done}, errors {errs}, total cached now {n_verdicts(a, model)}")

    print("\n" + "=" * 48)
    print(f"judged {grand['done']} · skipped(cached) {grand['skipped']} · errors {grand['errors']}")
    if not mock:
        print(f"tokens in {grand['in']:,} · out {grand['out']:,} · reported cost ${grand['cost']:.4f}")
    print(f"capabilities now available: {', '.join(JUDGE_CAPS)}")
    print("next: regenerate reports (gen_report / gen_dossier / gen_digest) — they auto-include them.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", help="single agent (default: all in config.AGENTS)")
    p.add_argument("--limit", type=int, help="cap cases per agent (smoke test)")
    p.add_argument("--mock", action="store_true", help="offline deterministic judge (no API key)")
    p.add_argument("--dry-run", action="store_true", help="estimate tokens/cost, make no calls")
    p.add_argument("--model", help="override AWE_JUDGE_MODEL for this run")
    p.add_argument("--force", action="store_true", help="re-judge even if cached")
    p.add_argument(
        "--include-resolved", action="store_true", help="also judge SOLVED tasks (control sample for validation)"
    )
    p.add_argument("--workers", type=int, default=6, help="concurrent requests (default 6)")
    p.add_argument("--panel", type=int, default=1, help="N votes/case, majority-aggregated (default 1)")
    p.add_argument("--temperature", type=float, help="sampling temp (panel defaults to 0.5 for diversity)")
    p.add_argument("--price-in", type=float, help="USD per Mtok input (dry-run cost)")
    p.add_argument("--price-out", type=float, help="USD per Mtok output (dry-run cost)")
    args = p.parse_args()

    agents = [args.agent] if args.agent else config.AGENTS
    if args.dry_run:
        dry_run(agents, args.include_resolved, args.limit, args.price_in, args.price_out)
        return
    if not args.mock and not config.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY not set. Export it, or use --mock for an offline test.")
    # Mock verdicts live in their OWN model namespace ("mock") so they never collide
    # with a real model's cache. To see mock-driven reports, run them with
    # AWE_JUDGE_MODEL=mock so the pipeline reads the same namespace.
    model = args.model or ("mock" if args.mock else config.JUDGE_MODEL)
    print(f"judge model: {model}" + ("  (offline mock)" if args.mock else ""))
    # Reports read config.JUDGE_MODEL (= $AWE_JUDGE_MODEL). If we judge under a
    # different namespace, the verdicts won't be picked up unless the operator
    # matches the env — warn loudly rather than fail silently.
    if model != config.JUDGE_MODEL:
        print(
            f"NOTE: verdicts cache under namespace '{model}', but the report scripts read "
            f"AWE_JUDGE_MODEL (currently '{config.JUDGE_MODEL}').\n"
            f"      Run reports with:  AWE_JUDGE_MODEL={model} python3 scripts/gen_digest.py"
        )
    temperature = args.temperature if args.temperature is not None else (0.5 if args.panel > 1 else None)
    run(
        agents,
        args.include_resolved,
        args.limit,
        args.mock,
        model,
        args.force,
        args.workers,
        panel=args.panel,
        temperature=temperature,
    )


if __name__ == "__main__":
    main()
