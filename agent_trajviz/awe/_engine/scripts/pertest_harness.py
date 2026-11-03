#!/usr/bin/env python3
"""Per-test re-evaluation of an agent's FAILED cases in the official SWE-bench
harness (the eval+ data source, DESIGN §0at / §0av).

For each failed case: pull the prebuilt x86_64 instance image with
``--platform linux/amd64`` (swebench 4.1.0 omits the platform on arm64 hosts
and errors otherwise), run ``swebench.harness.run_evaluation`` for that one
instance with ``cache_level none`` (the harness deletes the image afterwards),
then harvest:

* ``report.json``                 -> ``data/pertest/<agent>/<instance_id>.json``   (committed)
  (harness ``cache_level instance``: the driver removes the image itself, so
  concurrent driver runs never delete each other's freshly pulled images)
* ``test_output.txt``, ``run_instance.log`` -> ``data/cache/harness/<agent>/<instance_id>/`` (local; the
  raw grader log the environment fault_side arm needs — §0av item 4)

Progress is checkpointed per instance in ``data/cache/harness/<agent>/state.json``
(``ok`` / ``no_report`` / ``pull_failed`` / ``missing_patch`` / ``error``), so the
driver is safe to interrupt and re-run; finished instances are skipped.

Run (needs Docker running and a venv with ``swebench==4.1.0`` on Python >= 3.10):

    <venv>/bin/python scripts/pertest_harness.py --agent codex                # all failed cases
    <venv>/bin/python scripts/pertest_harness.py --agent codex --limit 1     # smoke test
    <venv>/bin/python scripts/pertest_harness.py --agent codex --instance psf__requests-2317 --gold

``--gold`` evaluates the dataset's gold patch instead of the agent's (the
gold-sanity check for false-negative candidates); its outputs go under
``data/cache/harness/gold/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_trajviz.awe._engine import config  # noqa: E402
from agent_trajviz.awe._engine.outcomes import resolved  # noqa: E402

# Benchmark knobs (ADDING_A_BENCHMARK.md §5.2): any swebench-compatible dataset +
# image namespace works — set them for SWE-bench Pro / Lite / Multimodal forks.
DATASET = os.environ.get("AWE_SWEBENCH_DATASET", "princeton-nlp/SWE-bench_Verified")
NAMESPACE = os.environ.get("AWE_IMAGE_NAMESPACE", "swebench")
PLATFORM = os.environ.get("AWE_IMAGE_PLATFORM", "linux/amd64")


def image_key(instance_id: str) -> str:
    """Mirror of swebench's TestSpec.instance_image_key for the remote namespace."""
    return f"{NAMESPACE}/sweb.eval.x86_64.{instance_id.lower()}:latest".replace("__", "_1776_")


def failed_instances(agent: str) -> list[str]:
    insts = sorted(p.stem for p in config.REQUIREMENTS_DIR.glob("*.json"))
    return [i for i in insts if resolved(agent, i) is False]


def strip_test_files(diff: str) -> str:
    """Drop every per-file section of a unified diff whose path is a test file
    (agent_trajviz.awe._engine.trajectory.is_test_file), keeping the rest byte-for-byte."""
    from agent_trajviz.awe._engine.diffutil import is_test_file

    out, keep = [], True
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1].strip() if " b/" in line else ""
            keep = not is_test_file(path)
        if keep:
            out.append(line)
    return "".join(out)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def pull(image: str, attempts: int = 3) -> tuple[bool, str]:
    err = ""
    for k in range(attempts):
        r = sh(["docker", "pull", "--platform", PLATFORM, image])
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
        err = err[0]
        if "manifest unknown" in err or "not found" in err:
            break  # no prebuilt image — retrying will not help
        time.sleep(10 * (k + 1))
    return False, err


def rmi(image: str) -> None:
    sh(["docker", "rmi", "-f", image])


class Driver:
    def __init__(self, agent: str, work: Path, gold: bool, timeout: int, keep_logs: bool, source_only: bool = False):
        self.agent = agent
        self.source_only = source_only
        self.model = "gold" if gold else (f"{agent}__srconly" if source_only else agent)
        self.gold = gold
        self.work = work
        self.timeout = timeout
        self.keep_logs = keep_logs
        self.run_id = f"pertest-{self.model}"
        self.out_reports = config.PROJECT_DATA / "pertest" / self.model
        self.out_logs = config.PROJECT_DATA / "cache" / "harness" / self.model
        self.out_reports.mkdir(parents=True, exist_ok=True)
        self.out_logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_logs / "state.json"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.preds = work / f"preds-{self.model}.jsonl"

    # -- predictions --------------------------------------------------------
    def write_predictions(self, ids: list[str]) -> list[str]:
        """One JSONL with every instance's patch; returns ids WITHOUT a patch."""
        missing = []
        with self.preds.open("w", encoding="utf-8") as fh:
            for iid in ids:
                if self.gold:
                    fh.write(
                        json.dumps({"instance_id": iid, "model_name_or_path": "gold", "model_patch": "gold"}) + "\n"
                    )
                    continue
                p = config.patch_path(self.agent, iid)
                if not p.exists() or not p.read_text(encoding="utf-8").strip():
                    missing.append(iid)
                    continue
                text = p.read_text(encoding="utf-8")
                if self.source_only:
                    text = strip_test_files(text)
                    if not text.strip():
                        missing.append(iid)
                        continue
                fh.write(json.dumps({"instance_id": iid, "model_name_or_path": self.model, "model_patch": text}) + "\n")
        return missing

    # -- one instance ---------------------------------------------------------
    def run_one(self, iid: str) -> str:
        t0 = time.time()
        image = image_key(iid)
        ok, err = pull(image)
        if not ok:
            return self.record(iid, "pull_failed", t0, err)
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            DATASET,
            "--split",
            "test",
            "--predictions_path",
            "gold" if self.gold else str(self.preds),
            "--instance_ids",
            iid,
            "--run_id",
            self.run_id,
            "--cache_level",
            "instance",
            "--max_workers",
            "1",
            "--timeout",
            str(self.timeout),
            "--report_dir",
            str(self.work / "reports"),
        ]
        (self.work / "reports").mkdir(exist_ok=True)
        r = sh(cmd, cwd=self.work)
        # A CONCURRENT harness run (cache_level none) snapshots the images that
        # existed when it started and deletes every other image when its own
        # instance finishes — including one this driver pulled seconds ago.
        # cache_level instance above stops THIS run from doing that to others;
        # a missing-image failure here is the race from an older run: re-pull
        # and try once more.
        if (
            r.returncode != 0
            or "BuildImageError" in (r.stderr or "")
            or "ImageNotFound" in (r.stderr or "")
            or "404 Client Error" in (r.stderr or "")
        ):
            if not (self.work / "logs" / "run_evaluation" / self.run_id / self.model / iid / "report.json").exists():
                ok2, _err = pull(image)
                if ok2:
                    r = sh(cmd, cwd=self.work)
        rmi(image)  # this driver's own cleanup
        log_dir = self.work / "logs" / "run_evaluation" / self.run_id / self.model / iid
        report = log_dir / "report.json"
        # raw grader logs first — they exist even when no report was produced
        dst = self.out_logs / iid
        dst.mkdir(exist_ok=True)
        for name in ("test_output.txt", "run_instance.log", "patch.diff"):
            if (log_dir / name).exists():
                shutil.copy2(log_dir / name, dst / name)
        if r.returncode != 0 and not report.exists():
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            (dst / "harness_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
            return self.record(iid, "error", t0, " | ".join(tail))
        if not report.exists():
            (dst / "harness_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
            return self.record(iid, "no_report", t0, "harness finished without report.json")
        shutil.copy2(report, self.out_reports / f"{iid}.json")
        if not self.keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)
        return self.record(iid, "ok", t0)

    def record(self, iid: str, status: str, t0: float, note: str = "") -> str:
        self.state[iid] = {
            "status": status,
            "seconds": round(time.time() - t0),
            "note": note,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.state_path.write_text(json.dumps(self.state, indent=1, sort_keys=True), encoding="utf-8")
        return status

    # -- manifest -------------------------------------------------------------
    def write_manifest(self, n_target: int) -> None:
        from collections import Counter

        c = Counter(v["status"] for v in self.state.values())
        m = {
            "agent": self.model,
            "run_id": self.run_id,
            "date": str(date.today()),
            "swebench": "4.1.0",
            "harness": "official swebench docker (Rosetta x86_64), "
            "cache_level none, one instance per invocation; driver scripts/pertest_harness.py",
            "n_target": n_target,
            "n_reports": c.get("ok", 0),
            "pull_failures": c.get("pull_failed", 0),
            "no_report": c.get("no_report", 0),
            "errors": c.get("error", 0),
            "missing_patch": c.get("missing_patch", 0),
            "raw_logs": f"data/cache/harness/{self.model}/<instance_id>/ (local only)",
            "note": "per-test FAIL_TO_PASS / PASS_TO_PASS status per failed case; "
            "source for regression_introduced / partial_fix_only / regression_test_missing",
        }
        (self.out_reports / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, choices=config.AGENTS)
    ap.add_argument("--instance", action="append", help="restrict to these instance ids (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new evaluations (smoke test)")
    ap.add_argument("--gold", action="store_true", help="evaluate the gold patch (gold-sanity)")
    ap.add_argument(
        "--source-only",
        action="store_true",
        help="drop the agent patch's hunks on test files before evaluating (adjudicates a "
        "test-patch apply collision; outputs go under <agent>__srconly)",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="per-instance harness timeout (s)")
    ap.add_argument(
        "--work",
        default=os.environ.get("AWE_HARNESS_WORK", str(ROOT / "data" / "cache" / "harness_work")),
        help="scratch dir for predictions + harness logs",
    )
    ap.add_argument("--keep-logs", action="store_true", help="keep the harness log tree after harvesting")
    ap.add_argument("--redo", action="store_true", help="re-run instances already in state.json")
    a = ap.parse_args()

    if sh(["docker", "info"]).returncode != 0:
        print("docker is not reachable — start the daemon first (colima start)", file=sys.stderr)
        return 2
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    d = Driver(a.agent, work, a.gold, a.timeout, a.keep_logs, source_only=a.source_only)
    ids = a.instance or failed_instances(a.agent)
    missing = d.write_predictions(ids)
    for iid in missing:
        if iid not in d.state:
            d.record(iid, "missing_patch", time.time(), "no patch file in ARGUS tree")
    todo = [i for i in ids if i not in missing and (a.redo or i not in d.state)]
    print(
        f"{a.agent}{' (gold)' if a.gold else ''}: {len(ids)} target, "
        f"{len(ids) - len(todo) - len(missing)} done, {len(missing)} missing patch, {len(todo)} to run",
        flush=True,
    )
    n = 0
    for k, iid in enumerate(todo, 1):
        status = d.run_one(iid)
        st = d.state[iid]
        print(f"[{k}/{len(todo)}] {iid:40s} {status:13s} {st['seconds']:5d}s {st['note'][:80]}", flush=True)
        n += 1
        if a.limit and n >= a.limit:
            break
    d.write_manifest(len(ids))
    from collections import Counter

    print("summary:", dict(Counter(v["status"] for v in d.state.values())), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
