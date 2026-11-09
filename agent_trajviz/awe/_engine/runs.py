"""Per-run output folders (v0.34).

Every full pipeline run snapshots its artifacts into ``data/runs/<run-id>/`` so
runs are comparable over time and nothing is silently overwritten. The
canonical paths (``data/dossier/``, ``data/reports/``) stay the *latest* run —
every existing consumer (validation, showcase, tests, docs) keeps working; the
run folder is an immutable copy with a manifest.

Layout of one run folder:

    data/runs/<run-id>/
      manifest.json        run metadata: agents, instance count, judge model,
                           git commit, fault counts, fault_side distribution
      INSIGHTS.md          aggregate cross-agent digest for THIS run
      insights.json
      ATTRIBUTION.html     default interactive numbered attribution DAGs
      attribution_graphs.json  typed provenance graphs over the source dossier
      RECOMMENDATIONS.md    scoped candidate experiments and evaluation steps
      recommendations.json selected recipes, case IDs, decisions, prerequisites
      failure_composition_by_mode.csv
      faults.json          complete case records
      faults.csv
      <agent>.md           per-agent dossiers
      reports/<agent>.md   per-agent scorecards
      reports/<agent>.json exact score/CI/coverage records used by Markdown
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import config

RUNS_DIR = config.OUTPUT_DATA / "runs"


def new_run_id(now: datetime | None = None) -> str:
    """Sortable, filesystem-safe: 2026-08-13_2145. Collisions within the same
    minute get a numeric suffix at snapshot time."""
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M")


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _manifest(dossier_dir: Path) -> dict:
    """Run metadata derived from the just-written canonical artifacts — the
    manifest states what the run actually produced, not what was intended."""
    from .benchmark import resolve_profile

    m: dict = {
        "benchmark_profile": resolve_profile(),
        "agents": list(config.AGENTS),
        "n_instances": len(config.list_instances()),
        "argus_root": str(config.ARGUS_ROOT),
        "judge_model": config.JUDGE_MODEL,
        "git_commit": _git_commit(config.PROJECT_ROOT),
    }
    faults_path = dossier_dir / "faults.json"
    if faults_path.exists():
        cases = json.loads(faults_path.read_text(encoding="utf-8"))
        m["n_failed_cases"] = len(cases)
        m["n_faults"] = sum(len(c.get("faults", [])) for c in cases)
        m["blame_status"] = dict(Counter(c.get("blame_status", "?") for c in cases))
        m["fault_side"] = dict(Counter((c.get("fault_side") or {}).get("side", "?") for c in cases))
    plan_path = dossier_dir / "recommendations.json"
    if plan_path.exists():
        from .remedy_plan import build_plan, to_markdown

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        scope = plan["scope"]
        if not faults_path.exists() or plan != build_plan(
            cases, agents=scope["agents"], instances=scope["instance_ids"], benchmark=scope["benchmark"]
        ):
            raise ValueError("Cannot snapshot a stale recommendation plan")
        md_path = dossier_dir / "RECOMMENDATIONS.md"
        if not md_path.exists() or md_path.read_text(encoding="utf-8") != to_markdown(plan):
            raise ValueError("Recommendation Markdown does not match the plan")
        m["recommendation_plan"] = {
            "schema_version": plan["schema_version"],
            "source_cases_sha256": plan["source_cases_sha256"],
            "catalog_sha256": plan["catalog_sha256"],
            "n_candidates": len(plan["candidates"]),
            "shortlist": plan["shortlist"],
            "scope": scope,
            "efficacy_claimed_by_generation": False,
        }
    rsi_paths = [dossier_dir / "rsi_proposals.json", dossier_dir / "rsi_feedback.json"]
    if any(p.exists() for p in rsi_paths):
        from .rsi import build_exchange

        if not faults_path.exists() or not plan_path.exists() or not all(p.exists() for p in rsi_paths):
            raise ValueError("Incomplete RSI exchange artifacts")
        expected_exchange = build_exchange(cases, plan)
        if tuple(json.loads(p.read_text(encoding="utf-8")) for p in rsi_paths) != expected_exchange:
            raise ValueError("Cannot snapshot a stale RSI exchange")
        m["rsi_exchange"] = {
            "schema_version": 1,
            "source_cases_sha256": expected_exchange[0]["source_cases_sha256"],
            "scope": scope,
            "proposals": len(expected_exchange[0]["proposals"]),
            "runtime_feedback_allowed": False,
            "agent_runs_performed": 0,
        }
    graphs_path = dossier_dir / "attribution_graphs.json"
    if graphs_path.exists():
        from .attribution_graph import build_graphs, render_html

        graphs = json.loads(graphs_path.read_text(encoding="utf-8"))
        scope = graphs["scope"]
        if not faults_path.exists() or graphs != build_graphs(
            cases, agents=scope["agents"], instances=scope["instance_ids"], benchmark=scope["benchmark"]
        ):
            raise ValueError("Cannot snapshot stale attribution graphs")
        html = dossier_dir / "ATTRIBUTION.html"
        if not html.exists() or html.read_text(encoding="utf-8") != render_html(graphs):
            raise ValueError("Attribution HTML does not match graph data")
        m["attribution_graphs"] = {
            "schema_version": graphs["schema_version"],
            "source_cases_sha256": graphs["source_cases_sha256"],
            "scope": scope,
            "n_graphs": len(graphs["graphs"]),
            "causal_discovery_performed": False,
        }
    return m


def snapshot(
    run_id: str | None = None,
    dossier_dir: Path | None = None,
    reports_dir: Path | None = None,
    runs_dir: Path | None = None,
    extra: dict | None = None,
) -> Path:
    """Copy the canonical outputs into a fresh run folder and write its
    manifest. Pure file operation over already-generated artifacts — call it
    LAST, after gen_report / gen_dossier / gen_digest. Returns the run dir."""
    dossier_dir = dossier_dir if dossier_dir is not None else config.OUTPUT_DATA / "dossier"
    reports_dir = reports_dir if reports_dir is not None else config.OUTPUT_DATA / "reports"
    runs_dir = runs_dir if runs_dir is not None else RUNS_DIR

    metadata = _manifest(dossier_dir)
    if "recommendation_plan" in metadata:
        scope = metadata["recommendation_plan"]["scope"]
        for key, value in (("agents", scope["agents"]), ("instance_ids", scope["instance_ids"])):
            if key in (extra or {}) and sorted(extra[key]) != value:
                raise ValueError("Recommendation scope differs from requested snapshot scope")
    if "attribution_graphs" in metadata:
        scope = metadata["attribution_graphs"]["scope"]
        for key in ("agents", "instance_ids"):
            if key in (extra or {}) and sorted(extra[key]) != scope[key]:
                raise ValueError("Attribution graph scope differs from requested snapshot scope")
        if "recommendation_plan" in metadata and scope != metadata["recommendation_plan"]["scope"]:
            raise ValueError("Attribution and recommendation scopes differ")

    rid = run_id or new_run_id()
    dest = runs_dir / rid
    n = 1
    while dest.exists():  # same-minute rerun: suffix, never clobber
        n += 1
        dest = runs_dir / f"{rid}.{n}"
    dest.mkdir(parents=True)

    copied = 0
    for src in sorted(dossier_dir.glob("*")):
        if src.is_file() and src.suffix in (".json", ".csv", ".md", ".html"):
            shutil.copy2(src, dest / src.name)
            copied += 1
    if reports_dir.is_dir():
        rdest = dest / "reports"
        rdest.mkdir()
        for src in sorted(reports_dir.iterdir()):
            if src.is_file() and src.suffix in (".md", ".json"):
                shutil.copy2(src, rdest / src.name)
                copied += 1
    if not copied:
        dest.rmdir()
        raise FileNotFoundError(
            f"nothing to snapshot: no artifacts under {dossier_dir} or {reports_dir} "
            "— run the generation scripts first (scripts/run_awe.py does all of it)"
        )

    manifest = {
        "run_id": dest.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
        # e.g. {"args": ["--limit", "20"]} — a filtered smoke run must
        # be visibly partial in its own manifest, not pass as a full run
        **(extra or {}),
    }
    manifest["artifacts"] = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dest
