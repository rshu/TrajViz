#!/usr/bin/env python3
"""Run the full AWE pipeline end-to-end and snapshot the run.

  AWE_ARGUS_ROOT=/path/to/ARGUS python3 scripts/run_awe.py [--agent NAME] [--limit N]

Steps, in order:
  1. scripts/gen_report.py   — per-agent capability scorecards  (data/reports/)
  2. scripts/gen_dossier.py  — dossier + attribution DAGs + remedy plans (data/dossier/)
  3. scripts/gen_digest.py   — cross-agent INSIGHTS digest      (data/dossier/INSIGHTS.md)
  4. agent_trajviz.awe._engine.runs.snapshot()     — immutable copy of everything     (data/runs/<run-id>/)

The canonical paths keep the LATEST run (all existing consumers unchanged);
the run folder is the per-run archive. The anchor output is the run's
ATTRIBUTION.html, with INSIGHTS.md for aggregate analysis. Judge/arbiter verdicts are read from cache automatically when
present (run scripts/run_judge.py + run_arbiter.py first to add the two
judgment-heavy capabilities; no API key is needed to read cached verdicts).
"""

from __future__ import annotations

import subprocess
import sys
import os
import uuid
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_trajviz.awe._engine.cli import pipeline_args, selected_inputs
from agent_trajviz.awe._engine import config  # noqa: E402  (path bootstrap first)
from agent_trajviz.awe._engine.preflight import require_valid_inputs
from agent_trajviz.awe._engine.runs import snapshot  # noqa: E402


def main() -> None:
    passthrough = sys.argv[1:]
    args = pipeline_args(passthrough)
    agents, instances = selected_inputs(args)
    preflight = require_valid_inputs(agents, instances)
    print(f"Input preflight: {preflight['counts']}; {len(preflight['warnings'])} coverage warnings")
    partial = args.agent is not None or args.limit is not None
    output = config.OUTPUT_DATA
    if partial:
        from agent_trajviz.awe._engine.runs import new_run_id

        output = output / "partial" / (new_run_id() + "_" + uuid.uuid4().hex[:8])
    # Full runs also generate into a fresh directory. Failed generators cannot
    # partially overwrite the latest run or archive older files as current.
    generated = output if partial else output / ".staging" / uuid.uuid4().hex
    env = {**os.environ, "AWE_AGENTS": ",".join(agents), "AWE_OUTPUT_DATA": str(generated)}

    # Judge-model mismatch guard (v0.36): cached verdicts exist for SOME model
    # but not for the configured one -> the run would silently drop the two
    # judge capabilities (~60 faults on the shipped corpus) while still looking
    # like a full run. Loud warning, not an abort — a deliberately judge-less
    # run stays possible.
    jm_dir = config.JUDGE_CACHE_DIR / config.model_slug()
    have = sorted(p.name for p in config.JUDGE_CACHE_DIR.glob("*") if p.is_dir() and any(p.rglob("*.json")))
    if have and (not jm_dir.exists() or not any(jm_dir.rglob("*.json"))):
        print(
            f"WARNING: no judge verdicts cached for AWE_JUDGE_MODEL="
            f"{config.JUDGE_MODEL!r}, but caches exist for: {', '.join(have)}.\n"
            f"         The judge capabilities will be silently absent from "
            f"this run.\n         Set AWE_JUDGE_MODEL to one of the cached "
            f"models (or run scripts/run_judge.py first)."
        )

    for script in ("gen_report.py", "gen_dossier.py", "gen_digest.py"):
        cmd = [sys.executable, str(ROOT / "scripts" / script), *passthrough]
        print(f"── {script} " + "─" * (60 - len(script)))
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            sys.exit(f"{script} failed (exit {r.returncode}) — run aborted, no snapshot")

    dest = snapshot(
        dossier_dir=generated / "dossier",
        reports_dir=generated / "reports",
        runs_dir=output / "runs",
        extra={
            "args": passthrough,
            "agents": agents,
            "n_instances": len(instances),
            "instance_ids": instances,
            "partial": partial,
            "input_preflight": preflight,
        },
    )
    if not partial:
        for kind in ("dossier", "reports"):
            (output / kind).mkdir(parents=True, exist_ok=True)
            for src in sorted((generated / kind).iterdir()):
                if src.is_file():
                    os.replace(src, output / kind / src.name)
        shutil.copy2(dest / "manifest.json", generated / "last_run.json")
        os.replace(generated / "last_run.json", output / "last_run.json")
        shutil.rmtree(generated)
    print("─" * 63)
    print(f"run folder : {dest}")
    print(f"VISUAL     : {dest / 'ATTRIBUTION.html'}   ← start here")
    print(f"INSIGHTS   : {dest / 'INSIGHTS.md'}")
    print(f"RSI     : {dest / 'rsi_proposals.json'} + rsi_feedback.json")
    print(f"REMEDIES   : {dest / 'RECOMMENDATIONS.md'}")
    print(f"manifest   : {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
