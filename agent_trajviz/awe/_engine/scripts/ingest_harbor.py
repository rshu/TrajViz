#!/usr/bin/env python3
"""Convert one Harbor run group into AWE's canonical data contract.

Each finished Harbor trial becomes a distinct AWE instance, so retries remain
independently diagnosable.  Trials without an objective ``resolved`` verdict are
reported in the manifest but skipped: AWE's diagnoses are outcome-conditioned.

Example:
    python scripts/ingest_harbor.py \
      --harbor-root /path/to/harbor_runs \
      --run-group verify_opencode_claude_opus_5_openrouter_effortxhigh \
      --output-root /tmp/awe-harbor \
      --trial cpython_14__SZSbJN3 --trial prometheus_1__6pnAykz
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(errors="replace", encoding="utf-8"))


def _lines(*paths: Path) -> list[str]:
    out = []
    for path in paths:
        if path.exists():
            out.extend(line.strip() for line in path.read_text(errors="replace", encoding="utf-8").splitlines() if line.strip())
    # Preserve order while avoiding duplicate orig/aug test identifiers.
    return list(dict.fromkeys(out))


def _patches(*paths: Path) -> str:
    return "\n".join(path.read_text(errors="replace", encoding="utf-8").rstrip() for path in paths if path.exists()).strip()


def _task_name(result: dict, result_path: Path) -> str:
    name = result.get("task_name") or ""
    if isinstance(name, str) and name:
        return name.rsplit("/", 1)[-1]
    # .../agent/<task>/<job>/<trial>/result.json
    try:
        return result_path.parents[2].name
    except IndexError:
        return "unknown_task"


def _task_bundle(run_dir: Path, task: str) -> Path | None:
    candidates = sorted((run_dir / "preflight" / "task_bundles").glob(f"*/{task}"))
    return candidates[0] if candidates else None


def _task_bundle_index(run_dirs: list[Path]) -> dict[str, list[Path]]:
    """Index reusable preflight bundles across run groups.

    Some archived Harbor groups retain their trial results but not their local
    preflight directory. LoLBench task ids identify the same immutable benchmark
    bundle across agent configurations, so another group's copy is valid input.
    """
    index: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        root = run_dir / "preflight" / "task_bundles"
        for candidate in root.glob("*/*"):
            if candidate.is_dir():
                index.setdefault(candidate.name, []).append(candidate)
    return index


def _submitted_patch(trial_dir: Path) -> Path | None:
    preferred = (
        trial_dir / "artifacts" / "logs" / "artifacts" / "solution.patch",
        trial_dir / "artifacts" / "solution.patch",
    )
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted((trial_dir / "artifacts").glob("**/solution.patch"))
    return candidates[0] if candidates else None


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _safe_instance_id(trial_dir: Path, namespace: str = "") -> str:
    # Harbor trial directory names are already stable and usually look like
    # task__token. Prefixing the run group makes retries from different agent
    # configurations collision-proof in an aggregate dataset.
    trial = _safe_id(trial_dir.name)
    return f"{_safe_id(namespace)}__{trial}" if namespace else trial


def _exception_class(exception: dict) -> str:
    exception_type = str(exception.get("exception_type", "")).lower()
    message = str(exception.get("exception_message", "")).lower()
    text = f"{exception_type} {message}"
    by_type = {
        "agenttimeouterror": "agent_timeout",
        "verifiertimeouterror": "verifier_timeout",
        "apiratelimiterror": "provider_rate_limit",
        "apiresponsestallederror": "provider_response_stalled",
        "apiconnectionclosederror": "provider_connection_closed",
        "networkconnectionerror": "network_connection_error",
        "unknownapierror": "provider_api_error",
        "cancellederror": "trial_cancelled",
    }
    if exception_type in by_type:
        return by_type[exception_type]
    if "exit 137" in text or "killed" in text:
        return "agent_process_killed_exit_137"
    if "exit 127" in text or "no such file or directory" in text:
        return "environment_missing_executable"
    if (
        "modulenotfounderror" in text
        or "importerror:" in text
        or "version `openssl_" in text
        or "version `openssl" in text
    ):
        return "environment_dependency_error"
    if "exit 124" in text or "timed out" in text:
        return "agent_timeout"
    if "no space left" in text:
        return "environment_storage_exhausted"
    if "permission denied" in text:
        return "environment_permission_error"
    if exception_type == "nonzeroagentexitcodeerror":
        return "agent_nonzero_exit"
    return "agent_runtime_exception"


def _trial_results(run_dir: Path) -> list[Path]:
    # Primary runs live at agent/<task>/<job>/<trial>. Status-tracked reruns may
    # live one campaign directory deeper at retry_*/agent/<task>/<job>/<trial>.
    out = set()
    for pattern in ("agent/*/*/*/result.json", "*/agent/*/*/*/result.json"):
        for result_path in run_dir.glob(pattern):
            if "__" in result_path.parent.name:
                out.add(result_path)
    return sorted(out)


def _status_match(rel_job: str, allow: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Match status rows that name either the trial or its parent Harbor job."""
    keys = [key for key in (rel_job, str(Path(rel_job).parent)) if key in allow]
    sources = sorted({source for key in keys for source in allow[key]})
    return keys, sources


def _write_operational_reports(data: Path, cases: list[dict]) -> None:
    """Write the non-workflow cohort without forcing it through AWE blame rules."""
    out = data / "operational"
    out.mkdir(parents=True, exist_ok=True)
    (out / "failures.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    fields = (
        "run_group",
        "agent",
        "trial",
        "task",
        "result",
        "reason",
        "failure_classification",
        "resolved",
        "exception_type",
        "exception_message",
    )
    with open(out / "failures.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            exc = case.get("exception_info") or {}
            writer.writerow(
                {
                    **{key: case.get(key, "") for key in fields},
                    "exception_type": exc.get("exception_type", ""),
                    "exception_message": exc.get("exception_message", ""),
                }
            )

    by_class = Counter(c.get("failure_classification", "unclassified") for c in cases)
    by_agent = Counter(c.get("agent", "unknown") for c in cases)
    lines = [
        "# Harbor operational failures",
        "",
        f"{len(cases)} finished trials were excluded from workflow attribution because "
        "the agent/runtime failed or no objective verifier verdict was available.",
        "",
        "## Failure classes",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in by_class.most_common())
    lines += ["", "## By agent configuration", "", "| Agent | Count |", "|---|---:|"]
    lines.extend(f"| `{name}` | {count} |" for name, count in by_agent.most_common())
    lines += ["", "## Instances", ""]
    for case in cases:
        exc = case.get("exception_info") or {}
        message = str(exc.get("exception_message", "")).replace("\n", " ")[:300]
        lines.append(
            f"- `{case.get('agent')}` / `{case.get('trial')}` — "
            f"`{case.get('failure_classification')}`" + (f": {message}" if message else "")
        )
    (out / "failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_allowlist(status_dir: Path, harbor_root: Path) -> tuple[dict[str, list[str]], dict]:
    """Return status-recorded Harbor job directories, normalized across hosts."""
    allow: dict[str, set[str]] = {}
    files = sorted(status_dir.glob("*status*.csv"))
    rows = with_path = outside_root = 0
    marker = "/harbor_runs/"
    root_prefix = str(harbor_root).rstrip("/") + "/"
    for status_file in files:
        with open(status_file, newline="", errors="replace", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                rows += 1
                raw = (row.get("agent_job_dir") or "").strip().rstrip("/")
                if not raw:
                    continue
                with_path += 1
                if raw.startswith(root_prefix):
                    rel = raw[len(root_prefix) :]
                elif marker in raw:
                    rel = raw.split(marker, 1)[1]
                else:
                    # Deliberately exclude sibling campaigns such as
                    # harbor_runs_with_apis and harbor_runs_with_files.
                    outside_root += 1
                    continue
                allow.setdefault(rel, set()).add(status_file.name)
    normalized = {key: sorted(value) for key, value in allow.items()}
    stats = {
        "status_dir": str(status_dir.resolve()),
        "n_status_files": len(files),
        "n_rows": rows,
        "n_rows_with_agent_job_dir": with_path,
        "n_rows_outside_harbor_root": outside_root,
        "n_unique_recorded_jobs": len(normalized),
    }
    return normalized, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harbor-root", type=Path, required=True, help="directory containing Harbor run-group directories")
    groups = p.add_mutually_exclusive_group(required=True)
    groups.add_argument(
        "--run-group", action="append", help="Harbor run-group directory name; repeat to aggregate groups"
    )
    groups.add_argument(
        "--all-run-groups", action="store_true", help="ingest every child containing an agent/ directory"
    )
    p.add_argument(
        "--output-root", type=Path, required=True, help="new canonical dataset root; data/ is created below it"
    )
    p.add_argument(
        "--status-dir", type=Path, help="only ingest agent_job_dir entries recorded by *status*.csv files here"
    )
    p.add_argument("--agent-name", help="AWE agent name for a single run group (default: run-group name)")
    p.add_argument(
        "--trial", action="append", default=[], help="include only this Harbor trial directory name; repeatable"
    )
    p.add_argument("--limit", type=int, help="maximum scored trials after filtering, across all groups")
    p.add_argument(
        "--include-exception-failures",
        action="store_true",
        help="feed failed trials with agent/runtime exceptions to AWE; "
        "by default they are classified separately to avoid blaming "
        "a missing patch on editing behavior",
    )
    p.add_argument(
        "--namespace-instances",
        action="store_true",
        help="prefix instance ids with the run-group name even for a "
        "single-group ingest (keeps ids/cache keys compatible with "
        "an aggregate ingest)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    harbor_root = args.harbor_root.resolve()
    if args.all_run_groups:
        run_dirs = sorted(path for path in harbor_root.iterdir() if path.is_dir() and (path / "agent").is_dir())
    else:
        run_dirs = [harbor_root / name for name in args.run_group]
    missing_groups = [str(path) for path in run_dirs if not path.is_dir()]
    if missing_groups:
        raise SystemExit("run group does not exist: " + ", ".join(missing_groups))
    if not run_dirs:
        raise SystemExit(f"no Harbor run groups found under {harbor_root}")
    if args.agent_name and len(run_dirs) != 1:
        raise SystemExit("--agent-name is valid only with one --run-group")
    # A single-group incremental ingest may target an archived run that retained
    # results but not preflight bundles. Search every local run group for the
    # benchmark's immutable task bundles, just as an --all-run-groups ingest does.
    bundle_sources = sorted(path for path in harbor_root.iterdir() if path.is_dir())
    bundle_index = _task_bundle_index(bundle_sources)
    status_allow = status_stats = None
    if args.status_dir:
        status_allow, status_stats = _status_allowlist(args.status_dir.resolve(), harbor_root)
        if not status_allow:
            raise SystemExit(f"no harbor-root jobs recorded in status files under {args.status_dir}")

    data = args.output_root.resolve() / "data"
    if data.exists() and any(data.iterdir()):
        raise SystemExit(f"refusing to mix with an existing dataset: {data}")

    req_dir = data / "requirements"
    req_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.trial)
    found_trials: set[str] = set()
    ingested, skipped, operational = [], [], []
    agents, group_stats = [], []
    stop = False
    for run_dir in run_dirs:
        run_group = run_dir.name
        agent = args.agent_name or run_group
        paths = _trial_results(run_dir)
        if status_allow is not None:
            paths = [
                p for p in paths if _status_match(str(p.parent.resolve().relative_to(harbor_root)), status_allow)[0]
            ]
        if selected:
            paths = [p for p in paths if p.parent.name in selected]
            found_trials.update(p.parent.name for p in paths)
        if not paths:
            group_stats.append(
                {"run_group": run_group, "agent": agent, "n_ingested": 0, "n_skipped": 0, "n_operational_failures": 0}
            )
            continue

        agents.append(agent)
        patch_dir = data / "patch" / agent
        traj_dir = data / "trajectory" / agent
        patch_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)
        outcomes: dict[str, bool] = {}
        group_ingested = group_skipped = group_operational = 0

        for result_path in paths:
            result = _json(result_path)
            base = {
                "run_group": run_group,
                "agent": agent,
                "trial": result_path.parent.name,
                "task": _task_name(result, result_path),
                "result": str(result_path.resolve()),
            }
            if status_allow is not None:
                rel_job = str(result_path.parent.resolve().relative_to(harbor_root))
                status_keys, status_sources = _status_match(rel_job, status_allow)
                base["status_keys"] = status_keys
                base["status_sources"] = status_sources
            if not result.get("finished_at"):
                skipped.append({**base, "reason": "unfinished"})
                group_skipped += 1
                continue
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            verdict = rewards.get("resolved")
            exception = result.get("exception_info")
            if verdict not in (0, 0.0, 1, 1.0, False, True):
                case = {
                    **base,
                    "reason": "finished_without_resolved_verdict",
                    "failure_classification": (
                        _exception_class(exception)
                        if isinstance(exception, dict) and exception
                        else "verifier_verdict_missing"
                    ),
                    "resolved": None,
                    "rewards": rewards,
                    "exception_info": exception,
                }
                skipped.append(case)
                operational.append(case)
                group_skipped += 1
                group_operational += 1
                continue
            if not bool(verdict) and isinstance(exception, dict) and exception and not args.include_exception_failures:
                case = {
                    **base,
                    "reason": "finished_with_agent_exception",
                    "failure_classification": _exception_class(exception),
                    "resolved": False,
                    "rewards": rewards,
                    "exception_info": exception,
                }
                skipped.append(case)
                operational.append(case)
                group_skipped += 1
                group_operational += 1
                continue

            task = base["task"]
            bundle = _task_bundle(run_dir, task)
            bundle_reused_from = None
            if bundle is None and bundle_index.get(task):
                bundle = bundle_index[task][0]
                bundle_reused_from = str(bundle)
            if bundle is None:
                skipped.append({**base, "reason": "task_bundle_not_found"})
                group_skipped += 1
                continue
            solution = bundle / "solution" / "solution.patch"
            instruction = bundle / "instruction.md"
            if not solution.exists() or not instruction.exists():
                skipped.append({**base, "reason": "task_bundle_missing_gold_or_instruction"})
                group_skipped += 1
                continue

            namespace = run_group if len(run_dirs) > 1 or args.namespace_instances else ""
            instance_id = _safe_instance_id(result_path.parent, namespace)
            private = bundle / "tests" / "private"
            meta_path = bundle / "environment" / "metadata.json"
            metadata = _json(meta_path) if meta_path.exists() else {}
            requirement = {
                "instance_id": instance_id,
                "problem_statement": instruction.read_text(errors="replace", encoding="utf-8"),
                "gold_patch": solution.read_text(errors="replace", encoding="utf-8"),
                "test_patch": _patches(private / "eval_tests.patch", private / "eval_tests_aug.patch"),
                "FAIL_TO_PASS": _lines(private / "f2p.txt", private / "f2p_aug.txt"),
                "PASS_TO_PASS": _lines(private / "p2p.txt", private / "p2p_aug.txt"),
                "gating_tests_visible": False,
                "repo": metadata.get("repo", metadata.get("project", "")),
                "base_commit": metadata.get("base_commit", ""),
            }
            (req_dir / f"{instance_id}.json").write_text(json.dumps(requirement, indent=2), encoding="utf-8")

            submitted = _submitted_patch(result_path.parent)
            if submitted is not None:
                shutil.copyfile(submitted, patch_dir / f"{instance_id}.diff")
            trajectory = result_path.parent / "agent" / "trajectory.json"
            pi_trajectory = result_path.parent / "agent" / "pi.txt"
            trajectory_format = None
            if trajectory.exists():
                shutil.copyfile(trajectory, traj_dir / f"{instance_id}.json")
                trajectory_format = "json"
            elif pi_trajectory.exists():
                # Pi's pi.txt is a v3 JSONL event stream despite the .txt suffix.
                shutil.copyfile(pi_trajectory, traj_dir / f"{instance_id}.jsonl")
                trajectory_format = "pi-v3-jsonl"

            outcomes[instance_id] = bool(verdict)
            ingested.append(
                {
                    **base,
                    "instance_id": instance_id,
                    "resolved": bool(verdict),
                    "rewards": rewards,
                    "has_patch": submitted is not None,
                    "has_trajectory": trajectory_format is not None,
                    "trajectory_format": trajectory_format,
                    "bundle_reused_from": bundle_reused_from,
                    "exception_info": exception,
                }
            )
            group_ingested += 1
            if args.limit and len(ingested) >= args.limit:
                stop = True
                break

        (data / f"eval_{agent}.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        group_stats.append(
            {
                "run_group": run_group,
                "agent": agent,
                "n_ingested": group_ingested,
                "n_skipped": group_skipped,
                "n_operational_failures": group_operational,
            }
        )
        if stop:
            break

    if selected:
        missing = sorted(selected - found_trials)
        if missing:
            raise SystemExit("requested Harbor trials not found: " + ", ".join(missing))

    _write_operational_reports(data, operational)
    status_selection = None
    if status_allow is not None:
        matched = {key for item in ingested + skipped for key in item.get("status_keys", [])}
        status_selection = {
            **status_stats,
            "n_matched_jobs": len(matched),
            "n_recorded_jobs_not_found": len(set(status_allow) - matched),
            "recorded_jobs_not_found": sorted(set(status_allow) - matched),
        }
    manifest = {
        "format": "awe-harbor-ingest-v1",
        "harbor_root": str(harbor_root),
        "run_groups": [path.name for path in run_dirs],
        "agents": agents,
        "n_ingested": len(ingested),
        "n_skipped": len(skipped),
        "n_operational_failures": len(operational),
        "status_selection": status_selection,
        "groups": group_stats,
        "instances": ingested,
        "skipped": skipped,
    }
    (data / "harbor_ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"agents={len(agents)} ingested={len(ingested)} skipped={len(skipped)} operational_failures={len(operational)}"
    )
    print(f"dataset={args.output_root.resolve()}")
    print("run with:")
    print(f"  AWE_ARGUS_ROOT={args.output_root.resolve()} \\")
    print(f"  AWE_PROJECT_DATA={data} \\")
    print("  AWE_BENCHMARK_NAME='LoLBench / Harbor' \\")
    print(f"  AWE_AGENTS={','.join(agents)} python scripts/gen_dossier.py")
    return 0 if ingested else 2


if __name__ == "__main__":
    sys.exit(main())
