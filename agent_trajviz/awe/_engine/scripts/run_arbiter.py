#!/usr/bin/env python3
"""Run the LLM ARBITER to corroborate/refute non-deductive PRIMARY blame.

The arbiter complements the rule layer: it second-opinions only the primaries that
are NOT deductive set-arithmetic — the read-but-not-edited omission and the
judge/contributing labels (deductive loc/edit facts are never sent). It is
OUTCOME-BLINDED (see awe/arbiter.py); verdicts cache per (model, agent, instance),
so the run is resumable and the report pipeline reads them back with no API key. A
confident, evidence-cited refute demotes that primary to 'conjunctive'.

Setup (same OpenRouter model as the judge):
    export OPENROUTER_API_KEY=sk-or-...
    export AWE_JUDGE_MODEL=z-ai/glm-5.2

Examples:
    python3 scripts/run_arbiter.py --dry-run            # cases that need adjudication, token est.
    python3 scripts/run_arbiter.py --mock               # offline plumbing test (no key)
    python3 scripts/run_arbiter.py --agent claude_code  # one agent, real calls
    python3 scripts/run_arbiter.py                      # all agents
Then regenerate reports with the SAME model env so they read the verdicts back.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.outcomes import resolved
from agent_trajviz.awe._engine.detect import detect
from agent_trajviz.awe._engine.rootcause import build_blame, _needs_adjudication
from agent_trajviz.awe._engine.arbiter import verify_one, verdict_cached, build_messages, mock_client, n_arbiter
from agent_trajviz.awe._engine.gold import build_gold_reference
from agent_trajviz.awe._engine.adapters import load_normalized
from agent_trajviz.awe._engine.llm import LLMError


def primary_claim(agent: str, inst: str):
    """The (capability, error_type) of the rule's primary cause IF it is
    non-deductive (so the arbiter should weigh in); else None."""
    if resolved(agent, inst) is not False:
        return None
    d = detect(inst, agent, use_trajectory=True)  # judge auto-included if cached
    b = build_blame(d)  # no arbiter here — we want the rule primary
    p = b.get("primary")
    if p and _needs_adjudication(p):
        return p["capability"], p["error_type"]
    return None


def targets(agent: str, limit):
    out = []
    for inst in config.list_instances():
        c = primary_claim(agent, inst)
        if c:
            out.append((inst, c))
    return out[:limit] if limit else out


def dry_run(agents, limit):
    print(f"{'agent':<16} {'needs_adj':>10} {'est_in_tok':>12}")
    grand = gi = 0
    for a in agents:
        ts = targets(a, limit)
        tok = 0
        for inst, (cap, et) in ts:
            try:
                gold = build_gold_reference(inst)
                traj = load_normalized(a, inst)
            except Exception:
                continue
            if traj and traj.steps:
                tok += sum(len(m["content"]) for m in build_messages(gold, traj, cap, et)) // 4
        grand += len(ts)
        gi += tok
        print(f"{a:<16} {len(ts):>10} {tok:>12,}")
    print("-" * 40)
    print(f"{'TOTAL':<16} {grand:>10} {gi:>12,}  (+~{grand * 120:,} output)")
    print("only NON-deductive primaries are sent; deductive loc/edit facts are never adjudicated.")


def run(agents, limit, mock, model, force, workers, panel=1, temperature=None):
    client = mock_client if mock else None
    grand = {"done": 0, "skipped": 0, "errors": 0}
    if panel > 1:
        print(f"PANEL mode: {panel} votes/case @ temperature {temperature} (majority + agreement)")
    for a in agents:
        ts = targets(a, limit)
        # resume check must use the full claim key — without cap/et the lookup
        # misses every claim-keyed cache file and re-adjudicates the whole set
        # (v0.38 fix; cost money on 2026-08-28)
        todo = [(i, c) for (i, c) in ts if force or verdict_cached(a, i, model, c[0], c[1]) is None]
        grand["skipped"] += len(ts) - len(todo)
        print(f"\n[{a}] {len(ts)} non-deductive primaries · {len(todo)} to adjudicate" + (" · MOCK" if mock else ""))
        done = errs = 0

        def work(item, a=a):
            inst, (cap, et) = item
            return verify_one(
                a, inst, cap, et, client=client, model=model, force=force, samples=panel, temperature=temperature
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    if fut.result() is not None:
                        done += 1
                except LLMError as e:
                    errs += 1
                    if errs <= 3:
                        print(f"  ! {e}")
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)} adjudicated...")
        grand["done"] += done
        grand["errors"] += errs
        print(f"[{a}] adjudicated {done}, errors {errs}, cached now {n_arbiter(a, model)}")

    print("\n" + "=" * 48)
    print(f"adjudicated {grand['done']} · skipped(cached) {grand['skipped']} · errors {grand['errors']}")
    print(
        "next: regenerate reports (gen_report / gen_dossier / gen_digest) — they auto-apply the "
        "arbiter when verdicts exist for the model in AWE_JUDGE_MODEL."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent")
    p.add_argument("--limit", type=int)
    p.add_argument("--mock", action="store_true", help="offline deterministic arbiter (no API key)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model", help="override AWE_JUDGE_MODEL for this run")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--panel", type=int, default=1, help="N votes/case, majority-aggregated (default 1)")
    p.add_argument("--temperature", type=float, help="sampling temp (panel defaults to 0.5 for diversity)")
    args = p.parse_args()

    agents = [args.agent] if args.agent else config.AGENTS
    if args.dry_run:
        dry_run(agents, args.limit)
        return
    if not args.mock and not config.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY not set. Export it, or use --mock for an offline test.")
    model = args.model or ("mock" if args.mock else config.JUDGE_MODEL)
    print(f"arbiter model: {model}" + ("  (offline mock)" if args.mock else ""))
    if model != config.JUDGE_MODEL:
        print(
            f"NOTE: verdicts cache under '{model}'; reports read AWE_JUDGE_MODEL "
            f"('{config.JUDGE_MODEL}'). Run reports with AWE_JUDGE_MODEL={model}."
        )
    temperature = args.temperature if args.temperature is not None else (0.5 if args.panel > 1 else None)
    run(agents, args.limit, args.mock, model, args.force, args.workers, panel=args.panel, temperature=temperature)


if __name__ == "__main__":
    main()
