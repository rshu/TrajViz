"""Bounded advisory controller and oracle-free observation helpers.

This does not install hooks, execute tests, or implement all catalog predicates.
Native adapters supply trusted observations and retain one controller for the
entire run, including retries/Stop re-entry. Runtime feedback contains no grader
data. All shipped guardrails are advisory while their efficacy is untested.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from threading import RLock

from .countermeasures import load_catalog
from .remedy_verification import recipe_digest

RUNTIME_KEYS = {
    "schema_version",
    "controller",
    "mode",
    "budget_scope",
    "max_triggers",
    "on_unknown",
    "on_exception",
    "on_exhaustion",
}


def valid_contract(entry):
    c = entry.runtime_contract
    return (
        isinstance(c, dict)
        and set(c) == RUNTIME_KEYS
        and type(c["schema_version"]) is int
        and c["schema_version"] == 1
        and c["controller"] == "agent_trajviz.awe._engine.remedy_runtime.GuardrailController"
        and c["mode"] == "advisory"
        and c["budget_scope"] == "run"
        and type(c["max_triggers"]) is int
        and c["max_triggers"] >= 1
        and isinstance(entry.contract, dict)
        and type(entry.contract.get("max_triggers")) is int
        and c["max_triggers"] == entry.contract.get("max_triggers")
        and c["on_unknown"] == "bounded_warning"
        and c["on_exception"] == "allow_with_receipt"
        and c["on_exhaustion"] == "continue_without_reprompt"
    )


def observation(status, reason, exception=None):
    return {"status": status, "reason": reason, "exception": exception}


class GuardrailController:
    """Serializes per-recipe and global warning budgets; never blocks an action.

    event_id is a native event identifier stable across duplicate hook delivery.
    Each new retry is a new event but uses the same run budget. Snapshots permit
    restart with strict consistency checks; storage authenticity is the native
    adapter's responsibility. Do not let the agent reset or edit this state.
    """

    def __init__(self, run_id, global_budget=6, entries=None, snapshot=None):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("Missing run identity")
        if type(global_budget) is not int or global_budget < 0:
            raise ValueError("Invalid global budget")
        recipes = [e for e in (load_catalog() if entries is None else entries) if e.intervention_type == "guardrail"]
        if (
            not recipes
            or len({e.id for e in recipes}) != len(recipes)
            or not all(valid_contract(e) and e.status in ("untested", "verified") for e in recipes)
        ):
            raise ValueError("Unsupported runtime contract")
        self._limits = {e.id: e.runtime_contract["max_triggers"] for e in recipes}
        self._identity = {
            "schema_version": 1,
            "run_id": run_id,
            "global_budget": global_budget,
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "recipes": {e.id: recipe_digest(e) for e in recipes},
        }
        self._events = []
        self._counts = Counter()
        self._lock = RLock()
        if snapshot is not None:
            self._restore(snapshot)

    def _restore(self, snapshot):
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != set(self._identity) | {"events"}
            or type(snapshot.get("schema_version")) is not int
            or type(snapshot.get("global_budget")) is not int
            or any(snapshot[k] != v for k, v in self._identity.items())
            or not isinstance(snapshot["events"], list)
            or len(snapshot["events"]) > self._identity["global_budget"]
        ):
            raise ValueError("Invalid or stale runtime snapshot")
        seen = set()
        for event in snapshot["events"]:
            if (
                not isinstance(event, dict)
                or set(event) != {"recipe_id", "event_id", "observation_sha256"}
                or not isinstance(event["recipe_id"], str)
                or event["recipe_id"] not in self._limits
                or not isinstance(event["event_id"], str)
                or not event["event_id"].strip()
                or not isinstance(event["observation_sha256"], str)
                or len(event["observation_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in event["observation_sha256"])
            ):
                raise ValueError("Invalid runtime event")
            key = (event["recipe_id"], event["event_id"])
            if key in seen:
                raise ValueError("Duplicate runtime event")
            seen.add(key)
            self._counts[event["recipe_id"]] += 1
            if self._counts[event["recipe_id"]] > self._limits[event["recipe_id"]]:
                raise ValueError("Runtime budget exceeded")
            self._events.append(dict(event))

    def snapshot(self):
        with self._lock:
            return {
                **self._identity,
                "recipes": dict(self._identity["recipes"]),
                "events": [dict(e) for e in self._events],
            }

    def decide(self, recipe_id, event_id, observed):
        if not isinstance(recipe_id, str) or recipe_id not in self._limits:
            raise ValueError("Unknown guardrail")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Missing native event identity")
        if (
            not isinstance(observed, dict)
            or set(observed) != {"status", "reason", "exception"}
            or observed["status"] not in ("match", "clear", "unknown")
            or not isinstance(observed["reason"], str)
            or not observed["reason"].strip()
            or (
                observed["exception"] is not None
                and (not isinstance(observed["exception"], str) or not observed["exception"].strip())
            )
        ):
            raise ValueError("Invalid observation")
        fingerprint = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()
        with self._lock:
            for event in self._events:
                if event["recipe_id"] == recipe_id and event["event_id"] == event_id:
                    if event["observation_sha256"] != fingerprint:
                        raise ValueError("Conflicting duplicate observation")
                    return {"action": "allow", "reason": "already_handled", "replayed": True}
            if observed["exception"] is not None:
                return {
                    "action": "allow",
                    "reason": "documented_exception",
                    "receipt": observed["exception"],
                    "replayed": False,
                }
            if observed["status"] == "clear":
                return {"action": "allow", "reason": observed["reason"], "replayed": False}
            if (
                self._counts[recipe_id] >= self._limits[recipe_id]
                or len(self._events) >= self._identity["global_budget"]
            ):
                return {"action": "allow", "reason": "budget_exhausted", "replayed": False}
            self._counts[recipe_id] += 1
            self._events.append({"recipe_id": recipe_id, "event_id": event_id, "observation_sha256": fingerprint})
            return {
                "action": "warn",
                "reason": observed["reason"],
                "observation_status": observed["status"],
                "replayed": False,
            }


def _ids(value):
    return (
        isinstance(value, (list, tuple, set))
        and all(isinstance(x, str) and bool(x.strip()) for x in value)
        and len(value) == len(set(value))
    )


def symbol_overlap(issue_symbols, file_symbols, causal_reason=None):
    """A lexical proxy; absent issue anchors cannot establish a wrong location."""
    if not _ids(issue_symbols) or not issue_symbols or not _ids(file_symbols):
        return observation("unknown", "symbol_evidence_unavailable", causal_reason)
    return observation(
        "clear" if set(issue_symbols) & set(file_symbols) else "match",
        "symbol_overlap" if set(issue_symbols) & set(file_symbols) else "no_symbol_overlap_proxy",
        causal_reason,
    )


def test_status(run):
    """Completed normalized public-test result, distinct from runner failure.

    Native language adapters must provide executed_ids, failed_ids, errors,
    completed and exit_code. An empty/skipped/failed collection is unknown.
    """
    if (
        not isinstance(run, dict)
        or run.get("completed") is not True
        or type(run.get("exit_code")) is not int
        or run.get("errors") != []
        or not _ids(run.get("executed_ids"))
        or not run["executed_ids"]
        or not _ids(run.get("failed_ids"))
        or not set(run["failed_ids"]) <= set(run["executed_ids"])
    ):
        return "unknown"
    if run["failed_ids"]:
        return "failed" if run["exit_code"] != 0 else "unknown"
    return "passed" if run["exit_code"] == 0 else "unknown"


def verification_gap(source_edited, revision, required_ids, run, exception=None):
    """Require completed tests of the submitted content and known public scope."""
    if source_edited is False:
        return observation("clear", "no_source_edit", exception)
    if (
        source_edited is not True
        or not isinstance(revision, str)
        or not revision
        or not _ids(required_ids)
        or not required_ids
    ):
        return observation("unknown", "verification_prerequisites_unknown", exception)
    if run is None:
        return observation("match", "no_completed_verification", exception)
    if test_status(run) == "unknown" or not isinstance(run.get("revision"), str) or not run["revision"]:
        return observation("unknown", "test_execution_or_revision_unknown", exception)
    if run["revision"] != revision or not set(required_ids) <= set(run["executed_ids"]):
        return observation("match", "verification_stale_or_scope_incomplete", exception)
    # A fresh failing test is verification too; repair advice is a separate predicate.
    return observation("clear", "fresh_completed_verification", exception)


def repair_stagnation(edit_count, runs):
    """Detect three no-progress transitions only on comparable test populations.

    New failures replacing old ones are changed evidence, not certified inertia.
    Exit codes alone never supply a failing-test count.
    """
    if type(edit_count) is not int or edit_count < 0 or not isinstance(runs, list):
        return observation("unknown", "repair_history_unknown")
    if edit_count < 3:
        return observation("clear", "below_loop_threshold")
    recent = runs[-4:]
    if (
        len(recent) < 4
        or any(test_status(r) == "unknown" for r in recent)
        or any(not isinstance(r.get("revision"), str) or not r["revision"] for r in recent)
        or len({r.get("revision") for r in recent}) != 4
        or any(
            not _ids(r.get("scope_ids")) or not r["scope_ids"] or set(r["executed_ids"]) != set(r["scope_ids"])
            for r in recent
        )
        or any(set(r["scope_ids"]) != set(recent[0]["scope_ids"]) for r in recent)
    ):
        return observation("unknown", "comparable_test_history_unavailable")
    failures = [set(r["failed_ids"]) for r in recent]
    if any(a != b for a, b in zip(failures, failures[1:], strict=False)) or not failures[-1]:
        return observation("clear", "test_evidence_changed_or_passed")
    return observation("match", "three_comparable_no_progress_transitions")


def self_revert(is_revert, reason=None):
    """Reverts always proceed; request learning only when it is absent."""
    if is_revert is None:
        return observation("unknown", "revert_detection_unknown", reason)
    if type(is_revert) is not bool:
        raise ValueError("Invalid revert observation")
    return observation(
        "match" if is_revert and not reason else "clear",
        "record_revert_learning" if is_revert else "not_a_revert",
        reason,
    )
