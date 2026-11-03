"""All-outcome symptom review exports, independent of fault scoring and judges."""

import copy
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from . import config
from .adapters import load_normalized
from .outcomes import resolved
from .symptoms import analyze
from .provenance import file_fingerprint, trajectory_fingerprint


def compact(analysis, limit=1000):
    """Bound long recorded commands only after identity/lifecycle analysis.

    Fingerprints always cover the complete original command. This representation
    is for review export; it must never be fed back into detection or matching.
    """
    result = copy.deepcopy(analysis)
    for row in result["verification"]["runs"] + result["symptoms"]:
        if "command" in row:
            command = row["command"]
            row["command_sha256"] = hashlib.sha256(command.encode()).hexdigest()
            row["command_truncated"] = len(command) > limit
            row["command"] = command[:limit]
    return result


def case_record(agent, inst):
    trajectory = load_normalized(agent, inst)
    path = config.patch_path(agent, inst)
    patch = path.read_text(errors="replace", encoding="utf-8") if path.exists() else ""
    analysis = analyze(trajectory, patch)
    # Patch availability is separate from trajectory availability. An absent
    # patch cannot establish that no oracle was changed.
    analysis["patch_available"] = path.exists()
    return {
        "case_id": f"{agent}__{inst}",
        "agent": agent,
        "instance_id": inst,
        "resolved": resolved(agent, inst),
        "inputs": {
            "patch_sha256": file_fingerprint(path)["sha256"],
            "trajectory_fingerprint": trajectory_fingerprint(agent, inst),
        },
        "analysis": compact(analysis),
    }


def collect(agents, instances, workers=1):
    instances = config.require_instances(instances)
    payloads = [(a, i) for a in agents for i in instances]
    if workers <= 1:
        return [case_record(a, i) for a, i in payloads]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Fail loudly on a corrupt input; never silently drop cases from scope.
        return list(pool.map(lambda p: case_record(*p), payloads))


def summarize(cases):
    groups = {}
    for label, value in [("solved", True), ("failed", False), ("excluded_or_unknown", None)]:
        selected = [c for c in cases if c["resolved"] is value]
        kinds = Counter(s["kind"] for c in selected for s in c["analysis"]["symptoms"])
        affected = Counter(k for c in selected for k in {s["kind"] for s in c["analysis"]["symptoms"]})
        groups[label] = {
            "cases": len(selected),
            "trajectories_available": sum(c["analysis"]["available"] for c in selected),
            "patches_available": sum(c["analysis"]["patch_available"] for c in selected),
            "verification_runs": sum(len(c["analysis"]["verification"]["runs"]) for c in selected),
            "runs_with_known_execution_count": sum(
                type((r["summary"].get("counts") or {}).get("executed")) is int
                for c in selected
                for r in c["analysis"]["verification"]["runs"]
            ),
            "symptom_observations": dict(kinds),
            "cases_with_symptom": dict(affected),
        }
    return {
        "cases": len(cases),
        "by_outcome": groups,
        "scope": "observational review candidates; prevalence is not precision or causal effect",
        "scoring": "not_scored",
    }


CSV_COLUMNS = ["case_id", "agent", "instance_id", "resolved", "kind", "symptom_id", "scoring", "evidence_json"]


def to_csv_rows(cases):
    return [
        {
            **{k: c[k] for k in ("case_id", "agent", "instance_id", "resolved")},
            **{k: s[k] for k in ("kind", "symptom_id", "scoring")},
            "evidence_json": json.dumps(s, sort_keys=True),
        }
        for c in cases
        for s in c["analysis"]["symptoms"]
    ]


def write(cases, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "symptoms.json").write_text(
        json.dumps({"schema_version": 1, "summary": summarize(cases), "cases": cases}, indent=2), encoding="utf-8"
    )
    with (directory / "symptoms.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(to_csv_rows(cases))
    return summarize(cases)
