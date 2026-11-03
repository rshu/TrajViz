"""Objective task outcome from the ARGUS SWE-bench eval results.

v0.38: outcomes pass through a DOCUMENTED override layer
(``data/overrides/resolved_overrides.json``) — the §0ac/§0ao contract
("correct via a documented re-label, never a silent edit") made mechanism.
Each override carries the corrected outcome (true / false / null = exclude
from every denominator), a reason, and an evidence pointer; the raw ARGUS
eval files are never edited. ``load_outcomes`` stays the RAW view."""

from __future__ import annotations

import json
from functools import lru_cache

from . import config
from .provenance import file_fingerprint

_UNSET = object()


@lru_cache(maxsize=8192)
def _read_json(path, source):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_json(path):
    return _read_json(str(path), file_fingerprint(path)["sha256"])


@lru_cache(maxsize=8192)
def _read_outcomes(path, source):
    doc = _read_json(path, source)
    if not isinstance(doc, dict) or any(
        not isinstance(k, str) or (v is not None and type(v) is not bool) for k, v in doc.items()
    ):
        raise ValueError(f"{path}: outcomes must map instance IDs to true/false/null")
    return doc


def load_outcomes(agent: str) -> dict[str, bool]:
    """The RAW per-agent eval map, no overrides applied."""
    path = config.eval_path(agent)
    if not path.exists():
        return {}
    return _read_outcomes(str(path), file_fingerprint(path)["sha256"])


def load_overrides() -> dict:
    """{(agent, instance_id): entry} from the override file; {} when absent.
    A malformed file fails LOUDLY — a silently ignored re-label would
    resurrect the very mislabels the mechanism exists to correct."""
    path = config.PROJECT_DATA / "overrides" / "resolved_overrides.json"
    if not path.exists():
        return {}
    doc = _load_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("overrides"), list):
        raise ValueError("resolved_overrides.json must contain an overrides list")
    out = {}
    for e in doc["overrides"]:
        if not isinstance(e, dict):
            raise ValueError("override entry must be an object")
        for k in ("agent", "instance_id", "reason", "evidence"):
            if not isinstance(e.get(k), str) or not e[k].strip():
                raise ValueError(f"resolved_overrides.json entry missing '{k}': {e}")
        if "resolved" not in e or (e["resolved"] is not None and type(e["resolved"]) is not bool):
            raise ValueError(f"resolved_overrides.json entry needs resolved true/false/null: {e}")
        config.safe_id(e["agent"])
        config.safe_id(e["instance_id"])
        if (e["agent"], e["instance_id"]) in out:
            raise ValueError("duplicate outcome override")
        out[(e["agent"], e["instance_id"])] = e
    return out


def override_for(agent: str, instance_id: str) -> dict | None:
    """The override entry applied to this case, if any (for reports/annexes)."""
    return load_overrides().get((agent, instance_id))


def resolved(agent: str, instance_id: str) -> bool | None:
    """True/False if known, None if this instance is missing from the eval set
    OR excluded by a documented override (``resolved: null``)."""
    ov = load_overrides().get((agent, instance_id), _UNSET)
    if ov is not _UNSET:
        return ov["resolved"]
    return load_outcomes(agent).get(instance_id)


# --- eval+ tier: per-test outcomes (v0.39) -----------------------------------
# A swebench ``report.json`` harvested per (agent, instance) to
# ``data/pertest/<agent>/<instance_id>.json`` — the official harness's per-test
# FAIL_TO_PASS / PASS_TO_PASS status (data/discovery/pertest_pilot.md,
# "Reproduce"). OPTIONAL: when absent the boolean tier runs unchanged. Read-only
# here, like the eval files; it is never edited by the pipeline.


def pertest_path(agent: str, instance_id: str):
    return config.PROJECT_DATA / "pertest" / config.safe_id(agent) / f"{config.safe_id(instance_id)}.json"


def load_pertest(agent: str, instance_id: str) -> dict | None:
    """Normalized per-test record for this case, or None when no report exists."""
    path = pertest_path(agent, instance_id)
    if not path.exists():
        return None
    doc = _load_json(path)
    expected = {}
    requirement = config.requirements_path(instance_id)
    if requirement.exists():
        from .gold import _as_list

        task = _load_json(requirement)
        expected = {"expected_f2p": _as_list(task["FAIL_TO_PASS"]), "expected_p2p": _as_list(task["PASS_TO_PASS"])}
    pt = parse_pertest(doc, instance_id, **expected)
    if pt is not None:
        effective = resolved(agent, instance_id)
        if effective is None:
            return None
        if pt["resolved"] != effective:
            if override_for(agent, instance_id) is not None:
                # A documented correction can supersede a stale harness record.
                # Retain the source file, but do not mix incompatible evidence.
                return None
            raise ValueError("per-test verdict disagrees with effective outcome")
    return pt


def pertest_log_path(agent: str, instance_id: str):
    return config.PROJECT_DATA / "pertest" / agent / f"{instance_id}.log.json"


def load_pertest_log(agent: str, instance_id: str) -> dict | None:
    """The grader-log audit record for this case (v0.42, DESIGN §0ba;
    written by scripts/grader_log_audit.py from the harness's raw
    test_output.txt): ``eval_status`` finer than the report's
    (graded / crash_before_summary / test_patch_apply_failed / timeout /
    patch_apply_failed / no_log), a crash ``attribution`` and ``crash_kind``,
    the ``test_patch_apply_error`` line, and per-gating-test statuses.
    OPTIONAL and read-only, like the per-test record."""
    path = pertest_log_path(agent, instance_id)
    if not path.exists():
        return None
    doc = _load_json(path)
    if not isinstance(doc, dict) or doc.get("instance_id") not in (None, instance_id):
        return None
    keep = ("eval_status", "attribution", "crash_kind", "test_patch_apply_error", "runner_summary", "gating", "error")
    return {k: doc.get(k) for k in keep}


# Preserve the public reset hook used by callers/tests; freshness no longer
# depends on callers remembering to invoke it.
for _loader in (load_outcomes, load_overrides, load_pertest, load_pertest_log):
    _loader.cache_clear = _read_json.cache_clear


def parse_pertest(doc: dict, instance_id: str | None = None, *, expected_f2p=None, expected_p2p=None) -> dict | None:
    """Normalize a swebench ``report.json`` record. Accepts the keyed form the
    harness writes (``{instance_id: {...}}``) or the bare record.

    Returns ``{patch_applied, resolved, f2p_passed, f2p_failed, f2p_total,
    p2p_passed_n, p2p_broken, f2p_pass_fraction, shape, eval_status}``.
    ``shape`` names the failure geometry the boolean eval erases:
    resolved / partial_fix / partial_fix_regressive / regression_only /
    unfixed / unfixed_regressive / patch_apply_failed / no_gating_tests.
    A patch the harness could not apply carries NO test information —
    ``f2p_pass_fraction`` is None and ``eval_status`` says so."""
    if not isinstance(doc, dict):
        return None
    bare = "tests_status" in doc or "patch_successfully_applied" in doc
    if instance_id and isinstance(doc.get(instance_id), dict):
        rec = doc[instance_id]
    elif bare:
        rec = doc
    elif len(doc) == 1 and isinstance(next(iter(doc.values())), dict):
        if instance_id:
            # keyed by ANOTHER instance: a mis-named / mis-copied file must not
            # grade this case with someone else's record (v0.40)
            return None
        rec = next(iter(doc.values()))
    else:
        return None
    if instance_id and rec.get("instance_id", instance_id) != instance_id:
        return None
    for key in ("patch_successfully_applied", "resolved"):
        if type(rec.get(key)) is not bool:
            raise ValueError(key + " must be a boolean")
    applied = rec["patch_successfully_applied"]
    # An unapplied patch has no trustworthy test results, even when a harness
    # copies stale partitions into the record.
    if not applied:
        if rec["resolved"]:
            raise ValueError("an unapplied patch cannot be resolved")
        f2p_ok = f2p_bad = p2p_ok = p2p_bad = []
    else:
        ts = rec.get("tests_status")
        if not isinstance(ts, dict):
            raise ValueError("applied patch requires explicit test partitions")
        parts = []
        for key, expected in (("FAIL_TO_PASS", expected_f2p), ("PASS_TO_PASS", expected_p2p)):
            part = ts.get(key)
            if not isinstance(part, dict) or set(part) != {"success", "failure"}:
                raise ValueError(key + " partition is missing")
            lists = []
            for state in ("success", "failure"):
                ids = part.get(state)
                if (
                    not isinstance(ids, list)
                    or any(not isinstance(x, str) or not x.strip() for x in ids)
                    or len(ids) != len(set(ids))
                ):
                    raise ValueError(key + " " + state + " must list unique test IDs")
                lists.append(sorted(ids))
            ok, bad = lists
            if set(ok) & set(bad):
                raise ValueError(key + " success/failure overlap")
            if expected is not None and set(ok + bad) != set(expected):
                raise ValueError(key + " coverage differs from requirement test set")
            parts.extend(lists)
        f2p_ok, f2p_bad, p2p_ok, p2p_bad = parts
        if set(f2p_ok + f2p_bad) & set(p2p_ok + p2p_bad):
            raise ValueError("F2P and P2P partitions overlap")
        if rec["resolved"] and (f2p_bad or p2p_bad):
            raise ValueError("resolved record contains failed gating tests")
    total = len(f2p_ok) + len(f2p_bad)
    frac = round(len(f2p_ok) / total, 4) if (applied and total) else None
    if not applied:
        shape = "patch_apply_failed"
    elif not total:
        shape = "regression_only" if p2p_bad else "no_gating_tests"
    elif frac == 1.0:
        shape = "regression_only" if p2p_bad else ("resolved" if rec["resolved"] else "unexplained_failure")
    elif frac > 0:
        shape = "partial_fix_regressive" if p2p_bad else "partial_fix"
    else:
        shape = "unfixed_regressive" if p2p_bad else "unfixed"
    return {
        "patch_applied": applied,
        "resolved": rec["resolved"],
        "f2p_passed": f2p_ok,
        "f2p_failed": f2p_bad,
        "f2p_total": total,
        "p2p_passed_n": len(p2p_ok),
        "p2p_broken": p2p_bad,
        "f2p_pass_fraction": frac,
        "shape": shape,
        "eval_status": "graded" if applied else "patch_apply_failed",
        "outcome_conflict": bool(applied and not rec["resolved"] and not f2p_bad and not p2p_bad),
    }
