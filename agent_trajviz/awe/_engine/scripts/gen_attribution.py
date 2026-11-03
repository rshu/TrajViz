#!/usr/bin/env python3
"""Build the default attribution viewer from an existing dossier, with no API calls."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_trajviz.awe._engine import config
from agent_trajviz.awe._engine.attribution_graph import build_graphs, write
from agent_trajviz.awe._engine.remedy_plan import cases_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=config.OUTPUT_DATA / "dossier")
    parser.add_argument(
        "--benchmark", help="Benchmark display name for a standalone dossier without a recommendation plan"
    )
    args = parser.parse_args()
    cases = json.loads((args.dossier / "faults.json").read_text(encoding="utf-8"))
    scope = {}
    plan_path = args.dossier / "recommendations.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["source_cases_sha256"] != cases_digest(cases):
            raise ValueError("Refresh the recommendation plan before exporting a changed dossier")
        scope = plan["scope"]
        if args.benchmark is not None and args.benchmark != scope["benchmark"]:
            raise ValueError("Benchmark label contradicts the recorded recommendation scope")
    benchmark = scope.get("benchmark") or args.benchmark
    if benchmark is None:
        # Offline dossier location does not establish which corpus produced it.
        benchmark = (
            config.BENCHMARK_NAME
            if os.environ.get("AWE_BENCHMARK_NAME") or os.environ.get("AWE_BENCHMARK_PROFILE")
            else "unspecified"
        )
    if not benchmark.strip():
        raise ValueError("Benchmark display name must be nonempty")
    write(
        build_graphs(cases, agents=scope.get("agents"), instances=scope.get("instance_ids"), benchmark=benchmark),
        args.dossier,
    )
    print(f"Wrote {args.dossier / 'ATTRIBUTION.html'} ({len(cases)} case graphs; no judging or agent reruns)")


if __name__ == "__main__":
    main()
