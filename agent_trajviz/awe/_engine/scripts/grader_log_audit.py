#!/usr/bin/env python3
"""Grader-log audit — the eval-side data pass DESIGN §0av item 4 asked for.

For every harvested per-test case (``data/pertest/<agent>/<id>.json`` with raw
grader logs under ``data/cache/harness/<agent>/<id>/``) re-read the official
harness's ``test_output.txt`` with swebench's OWN log parser and classify what
the boolean/per-test report collapses:

  eval_status
    graded                  — tests ran to a runner summary; failures are real test failures
    test_patch_apply_failed — ``git apply`` of the TEST patch errored (the agent's patch
                              already created / changed a file the test patch adds), so the
                              gating tests never existed in the run
    crash_before_summary    — the run died before the runner summary (collection /
                              import error, framework crash): the gating tests never ran
    timeout                 — the harness timed the tests out
    patch_apply_failed      — the agent's patch did not apply (no test information)
    no_log                  — no harvested log

  attribution of a crash / collection error (first error line):
    agent_added       — the failing symbol / module is introduced by the agent's patch
    gold_required     — a SYMBOL the gold / test patch defines and the agent did not
                        provide (an incomplete fix, model-side)
    gold_dependency_unverified — a third-party MODULE the gold / test patch imports could
                        not be imported in the agent's run: EITHER the image lacks it
                        (instance unresolvable by construction) OR the agent's patch broke
                        the import path — only a gold-sanity run decides (pylint-4661:
                        gold passes, so it was the agent's run)
    env_missing       — a module is missing that neither the agent nor gold introduces
                        (environment / image defect — the affirmative environment signal)
    unknown           — none of the above could be established from the texts

  gating tests: per-test status from the official parser, or ``missing`` with a reason
    (``failed_block`` — a FAIL/ERROR block names it, the progress line was merged;
    ``interleaved_ok`` — its ``ok`` is the last segment of a merged line, a parser miss;
    ``interleaved_unknown`` — named on a merged line with no verdict of its own;
    ``name_absent`` — the test never ran).

Writes ``data/pertest/<agent>/<id>.log.json`` (committed, small) beside the per-test
record and prints a per-agent shape table. Must run in the harness venv (needs
``swebench``):

    <venv>/bin/python scripts/grader_log_audit.py --agent codex
    <venv>/bin/python scripts/grader_log_audit.py            # every agent with logs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from agent_trajviz.awe._engine import config

# Keep this registry stable so callers can override individual parsers. The
# optional harness is needed for an actual audit, not to import TrajViz.
MAP_REPO_TO_PARSER: dict = {}
_HARNESS_CONSTANTS = None


def _load_harness():
    global _HARNESS_CONSTANTS
    if _HARNESS_CONSTANTS is None:
        try:
            from swebench.harness import constants, log_parsers
        except ImportError as exc:
            raise RuntimeError("Grader-log auditing requires the optional SWE-bench harness.") from exc
        for repo, parser in log_parsers.MAP_REPO_TO_PARSER.items():
            MAP_REPO_TO_PARSER.setdefault(repo, parser)
        _HARNESS_CONSTANTS = constants
    return _HARNESS_CONSTANTS


_ERR_LINE = re.compile(
    r"^(?:E\s+)?((?:ModuleNotFoundError|ImportError|SyntaxError|IndentationError|"
    r"AttributeError|TypeError|NameError|RuntimeError|OSError|MemoryError|"
    r"django\.core\.exceptions\.\w+|django\.db\.utils\.\w+|\w+Error): .*)$",
    re.M,
)
_COLLECT = re.compile(r"^_+ ERROR collecting (\S+) _+$", re.M)
_MODULE = re.compile(r"No module named '?([\w.]+)'?")
_IMPORT_NAME = re.compile(r"cannot import name '(\w+)'")
_ATTR = re.compile(r"object has no attribute '(\w+)'")
_DJANGO_SUMMARY = re.compile(r"^(OK(?: \([^)]*\))?|FAILED \([^)]*\))\s*$", re.M)
_PYTEST_SUMMARY = re.compile(r"^=+ [^=\n]*?(?:passed|failed|error|skipped)[^=\n]* in [\d.]+s[^=\n]*=+\s*$", re.M)
_RAN = re.compile(r"^Ran \d+ tests? in ", re.M)


def _read(p: Path) -> str | None:
    try:
        return p.read_text(errors="replace", encoding="utf-8")
    except FileNotFoundError:
        return None


def _requirements(iid: str) -> dict:
    try:
        return json.loads(config.requirements_path(iid).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _agent_patch(agent: str, iid: str) -> str:
    p = config.patch_path(agent, iid)
    return p.read_text(errors="replace", encoding="utf-8") if p.exists() else ""


def _added_lines(diff: str) -> str:
    return "\n".join(line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))


def _test_patch_apply_error(log: str) -> str | None:
    """The harness applies the TEST patch with ``git apply -v -`` right after
    resetting the test files; an ``error:`` line in that window means the gating
    tests were never installed (git apply is atomic)."""
    i = log.find("+ git apply -v -")
    if i < 0:
        return None
    window = log[i : i + 4000]
    j = window.find("\n+ ")  # next shell command echo
    window = window[:j] if j > 0 else window
    m = re.search(r"^error: .*$", window, re.M)
    return m.group(0).strip() if m else None


def _test_body(log: str) -> str:
    i = log.find(_load_harness().START_TEST_OUTPUT)
    return log[i:] if i >= 0 else log


def _runner_summary(body: str) -> str | None:
    dj = _DJANGO_SUMMARY.findall(body)
    if dj:
        return dj[-1]
    py = _PYTEST_SUMMARY.findall(body)
    if py:
        return py[-1].strip("= ").strip()
    return None


def _first_error(body: str) -> dict:
    m_c = _COLLECT.search(body)
    m_e = _ERR_LINE.search(body)
    out = {}
    if m_c:
        out["collecting"] = m_c.group(1)
    if m_e:
        out["line"] = m_e.group(1).strip()[:200]
    return out


def _attribute(err: dict, agent_diff: str, gold_patch: str, test_patch: str) -> str:
    line = err.get("line", "")
    if not line:
        return "unknown"
    added = _added_lines(agent_diff)
    m = _MODULE.search(line)
    if m:
        mod = m.group(1).split(".")[0]
        if re.search(r"\b(?:import|from)\s+" + re.escape(mod) + r"\b", added):
            return "agent_added"
        if re.search(r"\b(?:import|from)\s+" + re.escape(mod) + r"\b", gold_patch + test_patch):
            return "gold_dependency_unverified"
        return "env_missing"
    for rx in (_IMPORT_NAME, _ATTR):
        m = rx.search(line)
        if m:
            sym = m.group(1)
            if sym in added:
                return "agent_added"
            if sym in gold_patch or sym in test_patch:
                return "gold_required"
            return "unknown"
    return "unknown"


def _gating_status(gating: list, statuses: dict, body: str) -> dict:
    out = {}
    for t in gating:
        if t in statuses:
            out[t] = statuses[t]
            continue
        fn = t.split(" (")[0] if " (" in t else t.split("::")[-1]
        # a FAIL/ERROR block names the test whatever the progress line looked like
        if re.search(r"^(?:FAIL|ERROR): " + re.escape(fn) + r" \(", body, re.M):
            out[t] = "missing:failed_block"
            continue
        lines = [line for line in body.splitlines() if fn in line and " ... " in line]
        if not lines:
            out[t] = "missing:name_absent"
            continue
        # Django prints "test_a (...) ... test_b (...) ... ok" when a test's own
        # verdict never terminated its line: the trailing ok belongs to the LAST
        # segment only; an earlier segment's verdict is unknown from this line
        last_seg_ok = any(
            line.rstrip().endswith(("... ok", "... OK"))
            and line.rsplit(" ... ", 2)[-2].strip().startswith(fn)
            for line in lines
        )
        if last_seg_ok:
            out[t] = "missing:interleaved_ok"
        elif any(line.count(" ... ") > 1 for line in lines):
            out[t] = "missing:interleaved_unknown"
        else:
            out[t] = "missing:unparsed"
    return out


def audit_case(agent: str, iid: str) -> dict | None:
    rec_p = config.PROJECT_DATA / "pertest" / agent / f"{iid}.json"
    if not rec_p.exists():
        return None
    rec = json.loads(rec_p.read_text(encoding="utf-8")).get(iid) or {}
    logdir = config.PROJECT_DATA / "cache" / "harness" / agent / iid
    log = _read(logdir / "test_output.txt")
    out = {
        "instance_id": iid,
        "agent": agent,
        "eval_status": None,
        "attribution": None,
        "runner_summary": None,
        "parsed_counts": {},
        "gating": {},
        "error": {},
        "test_patch_apply_error": None,
    }
    if log is None:
        out["eval_status"] = "no_log"
        return out
    constants = _load_harness()
    if not rec.get("patch_successfully_applied", True) or constants.APPLY_PATCH_FAIL in log:
        out["eval_status"] = "patch_apply_failed"
        return out
    body = _test_body(log)
    repo = iid.rsplit("-", 1)[0].replace("__", "/")
    parser = MAP_REPO_TO_PARSER.get(repo)
    statuses = parser(body, None) if parser else {}
    out["parsed_counts"] = dict(Counter(statuses.values()))
    ts = rec.get("tests_status") or {}
    f2p = ts.get("FAIL_TO_PASS") or {}
    gating = list(f2p.get("success") or []) + list(f2p.get("failure") or [])
    out["gating"] = _gating_status(gating, statuses, body)
    out["runner_summary"] = _runner_summary(body)
    tpe = _test_patch_apply_error(log)
    out["test_patch_apply_error"] = tpe
    err = _first_error(body)
    out["error"] = err
    is_django = repo == "django/django"
    all_gating_lost = bool(gating) and all(
        v.startswith("missing:") or v == constants.TestStatus.ERROR.value for v in out["gating"].values()
    )
    if constants.TESTS_TIMEOUT in log:
        out["eval_status"] = "timeout"
    elif tpe:
        out["eval_status"] = "test_patch_apply_failed"
    elif not statuses or (is_django and not _RAN.search(body)):
        # nothing parsed, or Django's runner died before its "Ran N tests" line
        out["eval_status"] = "crash_before_summary"
        out["crash_kind"] = "collection_error" if "collecting" in err else "runner_crash"
    elif "collecting" in err and all_gating_lost:
        # pytest could not import the gating module; every gating test is lost
        out["eval_status"] = "crash_before_summary"
        out["crash_kind"] = "collection_error"
    else:
        out["eval_status"] = "graded"
    if err and out["eval_status"] in ("crash_before_summary", "test_patch_apply_failed"):
        req = _requirements(iid)
        out["attribution"] = _attribute(
            err, _agent_patch(agent, iid), req.get("gold_patch", ""), req.get("test_patch", "")
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", action="append", choices=config.AGENTS)
    ap.add_argument("--no-write", action="store_true", help="report only; do not write .log.json")
    a = ap.parse_args()
    agents = a.agent or [ag for ag in config.AGENTS if (config.PROJECT_DATA / "cache" / "harness" / ag).is_dir()]
    for agent in agents:
        d = config.PROJECT_DATA / "pertest" / agent
        ids = sorted(p.stem for p in d.glob("*.json") if p.name != "MANIFEST.json" and not p.name.endswith(".log.json"))
        rows = []
        for iid in ids:
            r = audit_case(agent, iid)
            if r is None:
                continue
            rows.append(r)
            if not a.no_write:
                (d / f"{iid}.log.json").write_text(json.dumps(r, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        st = Counter(r["eval_status"] for r in rows)
        att = Counter((r["eval_status"], r["attribution"]) for r in rows if r["attribution"])
        gm = Counter(v.split(":")[1] for r in rows for v in r["gating"].values() if v.startswith("missing:"))
        ge = sum(1 for r in rows if any(v == _load_harness().TestStatus.ERROR.value for v in r["gating"].values()))
        print(f"\n== {agent}: {len(rows)} cases with a per-test record")
        print("  eval_status      :", dict(st))
        print("  crash attribution:", {f"{k[0]}/{k[1]}": v for k, v in att.items()})
        print(
            "  gating tests missing from the parsed log (by reason):",
            dict(gm),
            "| cases with a gating test in ERROR status:",
            ge,
        )
        for r in rows:
            if r["eval_status"] in ("test_patch_apply_failed", "crash_before_summary", "timeout"):
                print(
                    f"   {r['instance_id']:30s} {r['eval_status']:24s} {r['attribution'] or '':14s} "
                    f"{(r['test_patch_apply_error'] or r['error'].get('line') or '')[:90]}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
