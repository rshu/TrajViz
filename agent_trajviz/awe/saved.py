"""Read saved AWE artifacts without importing or executing an AWE backend.

The saved dossier is the source of truth. Loading a run never recalculates its
scores, outcomes, attributions or proposed experiments. Original artifact bytes
and their fingerprints survive JSON and ZIP exports, including unknown fields.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import stat
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

RUN_TYPE = "agent-trajviz-awe-run"
SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ARTIFACTS = 4096
MAX_DEPTH = 128
_KNOWN = {"faults.json", "attribution_graphs.json", "symptoms.json", "recommendations.json",
          "manifest.json", "last_run.json", "insights.json", "rsi_proposals.json", "rsi_feedback.json",
          "operation-manifest.json", "operation-result.json"}


def _name(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value) or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))
            or len(PurePosixPath(value).parts) > 32):
        raise ValueError("Artifact names must be safe relative paths")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _finite(value: Any) -> None:
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise ValueError("Artifact JSON is nested too deeply")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Artifact JSON contains a non-finite number")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("Artifact JSON object keys must be strings")
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Artifact data must be JSON-compatible")


def _canonical(value: Any) -> bytes:
    try:
        # Serializing first also rejects recursive Python containers supplied to
        # load_run, before the iterative shape checks can revisit them forever.
        result = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
        _finite(value)
        return result
    except (TypeError, RecursionError, OverflowError) as exc:
        raise ValueError("Artifact data is not valid finite JSON") from exc


def _json(body: bytes) -> Any:
    try:
        result = json.loads(body.decode("utf-8-sig"), object_pairs_hook=_strict_pairs)
        _finite(result)
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Artifact contains invalid JSON") from exc


def _fingerprint(body: bytes, kind: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "kind": kind}


def _parse(name: str, body: bytes) -> tuple[Any, str]:
    if name.endswith(".json"):
        return _json(body), "json"
    try:
        return body.decode("utf-8-sig"), "text"
    except UnicodeError:
        return {"encoding": "binary", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}, "binary"


def _check_sizes(raw: dict[str, bytes]) -> None:
    if len(raw) > MAX_ARTIFACTS:
        raise ValueError("The run contains too many artifacts")
    if any(len(body) > MAX_ARTIFACT_BYTES for body in raw.values()):
        raise ValueError("An artifact exceeds the size limit")
    if sum(map(len, raw.values())) > MAX_TOTAL_BYTES:
        raise ValueError("The run exceeds the total size limit")


def _read_file(path: Path, *, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Run artifacts must be regular files, not symbolic links")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Run artifacts must be regular files")
        body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError("An artifact exceeds the size limit")
    return body


def _directory(path: Path) -> dict[str, bytes]:
    raw: dict[str, bytes] = {}
    # The mutable output root also holds caches and historical runs. Only its
    # current output branches belong to one import; never combine those runs.
    branches = [path / name for name in ("dossier", "reports", "discovery")]
    selected = branches if (path / "dossier").is_dir() else [path]
    if selected == branches:
        for name in ("manifest.json", "last_run.json", "operation-manifest.json", "operation-result.json", "operation.log"):
            if (path / name).exists():
                raw[name] = _read_file(path / name)
    for branch in selected:
        if not branch.exists():
            continue
        if branch.is_symlink():
            raise ValueError("Run directories cannot be symbolic links")
        def walk_error(error: OSError) -> None:
            raise ValueError("A run directory could not be read") from error

        for current, directories, files in os.walk(branch, followlinks=False, onerror=walk_error):
            current_path = Path(current)
            for directory in directories:
                if (current_path / directory).is_symlink():
                    raise ValueError("Run directories cannot contain symbolic links")
            for filename in sorted(files):
                entry = current_path / filename
                name = _name(entry.relative_to(path).as_posix())
                raw[name] = _read_file(entry)
                _check_sizes(raw)
    return raw


def _zip(path: Path) -> dict[str, bytes]:
    raw = {}
    declared = 0
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARTIFACTS * 2:
                raise ValueError("The run archive contains too many entries")
            names = set()
            for info in entries:
                name = _name(info.filename.rstrip("/"))
                if name in names:
                    raise ValueError("The run archive contains duplicate artifact paths")
                names.add(name)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("Run archives cannot contain links or special files")
                if info.flag_bits & 1:
                    raise ValueError("Encrypted run archives are not supported")
                if info.is_dir():
                    continue
                declared += info.file_size
                if info.file_size > MAX_ARTIFACT_BYTES or declared > MAX_TOTAL_BYTES:
                    raise ValueError("The run archive exceeds the size limit")
                with archive.open(info) as stream:
                    body = stream.read(MAX_ARTIFACT_BYTES + 1)
                if len(body) != info.file_size or len(body) > MAX_ARTIFACT_BYTES:
                    raise ValueError("The run archive contains an invalid artifact size")
                raw[name] = body
            _check_sizes(raw)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
        raise ValueError("The saved run is not a readable ZIP archive") from exc
    # Normal ZIP exports often wrap everything in one directory. Strip exactly
    # that wrapper, keeping meaningful dossier/reports/discovery paths intact.
    roots = {name.split("/", 1)[0] for name in raw}
    if len(roots) == 1 and all("/" in name for name in raw):
        root = next(iter(roots))
        if root not in ("dossier", "reports", "discovery"):
            raw = {name.split("/", 1)[1]: body for name, body in raw.items()}
    if "dossier/faults.json" in raw and "faults.json" not in raw:
        raw = {name: body for name, body in raw.items()
               if name.split("/", 1)[0] in ("dossier", "reports", "discovery")
               or name in ("manifest.json", "last_run.json", "operation-manifest.json", "operation-result.json", "operation.log")}
    return raw


def _find(artifacts: dict[str, Any], name: str) -> tuple[str | None, Any]:
    matches = [(key, value) for key, value in artifacts.items() if PurePosixPath(key).name == name]
    if len(matches) > 1:
        raise ValueError(f"Multiple {name} artifacts belong to different runs; import one run at a time")
    return matches[0] if matches else (None, None)


def _case_key(case: dict[str, Any]) -> str:
    identity = case.get("case_id")
    if identity is not None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("Case IDs must be nonempty strings")
        return identity
    agent, instance = case.get("agent"), case.get("instance_id")
    if not isinstance(agent, str) or not agent or not isinstance(instance, str) or not instance:
        raise ValueError("A case needs case_id or agent and instance_id")
    return f"{agent}__{instance}"


def _case_rows(value: Any, *, faults: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(case, dict) for case in value):
        raise ValueError("The case artifact must contain a list of objects")
    for case in value:
        for key in ("agent", "instance_id"):
            if key in case and (not isinstance(case[key], str) or not case[key]):
                raise ValueError("Case agent and instance identities must be nonempty strings")
        if case.get("resolved") is not None and type(case["resolved"]) is not bool:
            raise ValueError("Recorded case outcomes must be booleans or null")
        if faults and case.get("resolved") is True:
            raise ValueError("A fault dossier cannot describe a resolved case as failed")
    keys = [_case_key(case) for case in value]
    if len(keys) != len(set(keys)):
        raise ValueError("The run contains duplicate case IDs")
    pairs = [(case.get("agent"), case.get("instance_id")) for case in value
             if case.get("agent") is not None and case.get("instance_id") is not None]
    if len(pairs) != len(set(pairs)):
        raise ValueError("The run contains conflicting case identities")
    if faults:
        for case in value:
            if not isinstance(case.get("faults"), list) or any(not isinstance(fault, dict) for fault in case["faults"]):
                raise ValueError("Each dossier case must contain a faults list")
    return value


def _dag(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list) or not isinstance(value.get("edges"), list):
        raise ValueError("A saved attribution graph needs nodes and edges lists")
    nodes, edges = value["nodes"], value["edges"]
    if any(not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"] for node in nodes):
        raise ValueError("Graph nodes need nonempty string IDs")
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("The attribution graph contains duplicate node IDs")
    children: dict[str, list[str]] = {key: [] for key in ids}
    incoming = dict.fromkeys(ids, 0)
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Graph edges must be objects")
        start, end = edge.get("from"), edge.get("to")
        relation = edge.get("relation")
        if not isinstance(start, str) or not isinstance(end, str) or start not in children or end not in children:
            raise ValueError("The attribution graph contains a dangling edge")
        if not isinstance(relation, str) or not relation:
            raise ValueError("Graph edges need a recorded relation")
        key = start, end, relation
        if key in seen:
            raise ValueError("The attribution graph contains duplicate edges")
        seen.add(key)
        children[start].append(end)
        incoming[end] += 1
    pending = deque(key for key in ids if incoming[key] == 0)
    visited = 0
    while pending:
        visited += 1
        for child in children[pending.popleft()]:
            incoming[child] -= 1
            if incoming[child] == 0:
                pending.append(child)
    if visited != len(ids):
        raise ValueError("The attribution graph contains a cycle")


def _scope(document: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    scope = document.get("scope")
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise ValueError("Saved run scope must be an object")
    for field_name, row_name in (("agents", "agent"), ("instance_ids", "instance_id")):
        allowed = scope.get(field_name)
        if allowed is not None:
            if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
                raise ValueError("Saved run scope identities must be string lists")
            if len(allowed) != len(set(allowed)):
                raise ValueError("Saved run scope contains duplicate identities")
            if any(row.get(row_name) is not None and row[row_name] not in allowed for row in rows):
                raise ValueError("Saved cases fall outside the recorded run scope")
    expected = scope.get("diagnosed_failed_cases")
    if expected is not None and (type(expected) is not int or expected != len(rows)):
        raise ValueError("The saved run scope does not match its diagnosed cases")


def _shape(raw: dict[str, bytes], source: str | None, *, extra: dict[str, Any] | None = None) -> SavedRun:
    _check_sizes(raw)
    artifacts, fingerprints = {}, {}
    for name, body in sorted(raw.items()):
        _name(name)
        value, kind = _parse(name, body)
        artifacts[name] = value
        fingerprints[name] = _fingerprint(body, kind)
    if not any(PurePosixPath(name).name in _KNOWN for name in artifacts):
        raise ValueError("No recognized AWE run artifacts were found")
    cases_path, cases_value = _find(artifacts, "faults.json")
    cases = _case_rows(cases_value, faults=True) if cases_path is not None else []
    _, graphs = _find(artifacts, "attribution_graphs.json")
    _, symptoms = _find(artifacts, "symptoms.json")
    _, recommendations = _find(artifacts, "recommendations.json")
    _, manifest = _find(artifacts, "manifest.json")
    if manifest is None:
        _, manifest = _find(artifacts, "last_run.json")
    _, operation_manifest = _find(artifacts, "operation-manifest.json")
    if manifest is None:
        manifest = operation_manifest
    warnings = []
    case_hash = hashlib.sha256(_canonical(cases)).hexdigest() if cases_value is not None else None
    case_ids = {_case_key(case) for case in cases}
    known_identities: dict[str, dict[str, Any]] = {}
    known_pairs: dict[tuple[str, str], str] = {}

    def register_cases(rows: list[dict[str, Any]], *, failed: bool = False) -> None:
        for row in rows:
            identity = _case_key(row)
            known = known_identities.setdefault(identity, {})
            for field_name in ("agent", "instance_id"):
                value = row.get(field_name)
                if value is not None:
                    if field_name in known and known[field_name] != value:
                        raise ValueError("Saved artifacts contain conflicting case identities")
                    known[field_name] = value
            resolved = False if failed else row.get("resolved")
            if resolved is not None:
                if "resolved" in known and known["resolved"] is not resolved:
                    raise ValueError("Saved artifacts contain conflicting case outcomes")
                known["resolved"] = resolved
            if "agent" in known and "instance_id" in known:
                pair = known["agent"], known["instance_id"]
                if pair in known_pairs and known_pairs[pair] != identity:
                    raise ValueError("Saved artifacts contain conflicting case identities")
                known_pairs[pair] = identity

    register_cases(cases, failed=True)
    for label in ("attribution_graphs.json", "symptoms.json", "recommendations.json", "manifest.json", "last_run.json",
                  "operation-manifest.json"):
        artifact_path, document = _find(artifacts, label)
        if artifact_path is not None and not isinstance(document, dict):
            raise ValueError(f"The {label} artifact must be an object")
    for name in ("attribution_graphs.json", "recommendations.json", "rsi_proposals.json", "rsi_feedback.json"):
        _, document = _find(artifacts, name)
        if document is None:
            continue
        if not isinstance(document, dict):
            raise ValueError(f"The {name} artifact must be an object")
        if document.get("schema_version") not in (None, 1):
            warnings.append(f"{name} uses an unfamiliar schema; recorded fields are preserved.")
        claimed = document.get("source_cases_sha256")
        if claimed is not None:
            if not isinstance(claimed, str) or len(claimed) != 64 or any(c not in "0123456789abcdef" for c in claimed):
                raise ValueError("The saved source-case fingerprint is invalid")
            if case_hash is not None and claimed != case_hash:
                raise ValueError(f"The {name} artifact is stale or belongs to a different dossier")
            if case_hash is None:
                warnings.append(f"{name} source-case fingerprint cannot be verified without faults.json.")
        if cases_value is not None:
            _scope(document, cases)
    if graphs is not None:
        graph_rows = _case_rows(graphs.get("graphs"), faults=False)
        if cases_value is not None and {_case_key(row) for row in graph_rows} != case_ids:
            raise ValueError("Saved attribution graphs do not match the dossier case IDs")
        register_cases(graph_rows)
        for graph in graph_rows:
            _dag(graph)
            if graph.get("action_graph") is not None:
                _dag(graph["action_graph"])
        _scope(graphs, graph_rows)
    if recommendations is not None:
        candidates = recommendations.get("candidates")
        if not isinstance(candidates, list) or any(not isinstance(row, dict) for row in candidates):
            raise ValueError("Recommendations must contain a candidates list")
        recipe_ids = [row.get("id") for row in candidates]
        if any(not isinstance(rid, str) or not rid for rid in recipe_ids) or len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError("Recommendation IDs must be unique nonempty strings")
        for candidate in candidates:
            references = candidate.get("case_ids", [])
            if not isinstance(references, list) or any(not isinstance(ref, str) for ref in references):
                raise ValueError("Recommendation case references must be string lists")
            if cases_value is not None and not set(references) <= case_ids:
                raise ValueError("A recommendation references a case outside the dossier")
        shortlist = recommendations.get("shortlist") or {}
        if not isinstance(shortlist, dict):
            raise ValueError("The recommendation shortlist must be an object")
        lists = [shortlist.get("overall", [])]
        by_agent = shortlist.get("by_agent", {})
        if not isinstance(by_agent, dict):
            raise ValueError("Per-agent recommendation shortlists must be an object")
        lists.extend(by_agent.values())
        if any(not isinstance(ids, list) or any(not isinstance(rid, str) or rid not in recipe_ids for rid in ids)
               for ids in lists):
            raise ValueError("The shortlist references an unknown recommendation")
    if symptoms is not None:
        symptom_rows = _case_rows(symptoms.get("cases"), faults=False)
        register_cases(symptom_rows)
        scope_document = graphs or recommendations or manifest
        if scope_document:
            # Symptom exports also contain solved/unknown outcomes, so they are
            # checked against agent/task scope, not the failed-dossier row count.
            scope = scope_document.get("scope", scope_document)
            if isinstance(scope, dict):
                _scope({"scope": {k: v for k, v in scope.items() if k in ("agents", "instance_ids")}}, symptom_rows)
    if manifest is not None:
        listed = manifest.get("artifacts")
        if listed is not None:
            if not isinstance(listed, list) or any(not isinstance(name, str) for name in listed):
                raise ValueError("The run manifest artifact list must contain paths")
            if len(set(listed)) != len(listed):
                raise ValueError("The run manifest contains duplicate artifact paths")
            for name in listed:
                _name(name)
            # Snapshot manifests normally name files relative to the snapshot;
            # current output roots retain the dossier prefix instead.
            if any(name not in artifacts and f"dossier/{name}" not in artifacts for name in listed):
                raise ValueError("The run manifest references a missing artifact")
        if cases_value is not None:
            expected = manifest.get("n_failed_cases")
            if expected is not None and (type(expected) is not int or expected != len(cases)):
                raise ValueError("The run manifest does not match the dossier case count")
            expected_faults = manifest.get("n_faults")
            if expected_faults is not None and (type(expected_faults) is not int
                                               or expected_faults != sum(len(case["faults"]) for case in cases)):
                raise ValueError("The run manifest does not match the dossier fault count")
            _scope({"scope": {k: v for k, v in manifest.items() if k in ("agents", "instance_ids")}}, cases)
            for field_name, document in (("attribution_graphs", graphs), ("recommendation_plan", recommendations)):
                metadata = manifest.get(field_name)
                if isinstance(metadata, dict) and metadata.get("source_cases_sha256") is not None:
                    if metadata["source_cases_sha256"] != case_hash:
                        raise ValueError("The run manifest contains a stale dossier fingerprint")
                    if document is not None and metadata.get("scope") is not None and metadata["scope"] != document.get("scope"):
                        raise ValueError("The run manifest and artifact scopes disagree")
    if operation_manifest is not None:
        hashes = operation_manifest.get("artifact_sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError("Operation artifact fingerprints must be an object")
        for name, expected in hashes.items():
            _name(name)
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError("An operation artifact fingerprint is invalid")
            if name not in raw:
                raise ValueError("The operation manifest references a missing artifact")
            if fingerprints[name]["sha256"] != expected:
                raise ValueError("An operation artifact does not match its saved fingerprint")
    if cases_value is None:
        warnings.append("No fault dossier was supplied; source case evidence is unavailable.")
    if manifest is None:
        warnings.append("No run manifest was supplied; original run provenance is unavailable.")
    return SavedRun(source, artifacts, cases, graphs, symptoms, recommendations, manifest,
                    tuple(dict.fromkeys(warnings)), fingerprints, dict(raw), extra or {})


@dataclass(frozen=True)
class SavedRun:
    """A saved analysis and its original artifacts; no backend is required."""

    source: str | None
    artifacts: dict[str, Any]
    cases: list[dict[str, Any]]
    graphs: dict[str, Any] | None
    symptoms: dict[str, Any] | None
    recommendations: dict[str, Any] | None
    manifest: dict[str, Any] | None
    warnings: tuple[str, ...] = ()
    fingerprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_artifacts: dict[str, bytes] = field(default_factory=dict, repr=False)
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def _check_integrity(self) -> None:
        if set(self.artifacts) != set(self.raw_artifacts) or set(self.artifacts) != set(self.fingerprints):
            raise ValueError("Saved artifact data no longer matches its original bytes")
        for name, body in self.raw_artifacts.items():
            parsed, kind = _parse(name, body)
            if (_canonical(parsed) != _canonical(self.artifacts[name])
                    or _fingerprint(body, kind) != self.fingerprints[name]):
                raise ValueError("Saved artifact data or fingerprints have changed")

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON envelope preserving every original artifact."""
        self._check_integrity()
        return {**copy.deepcopy(self._extra), "type": RUN_TYPE, "schema_version": SCHEMA_VERSION,
                "source": self.source, "imported": True, "artifacts": copy.deepcopy(self.artifacts),
                "fingerprints": copy.deepcopy(self.fingerprints),
                "raw_artifacts": {name: base64.b64encode(body).decode("ascii") for name, body in self.raw_artifacts.items()},
                "warnings": list(self.warnings)}

    def write(self, dest: str | os.PathLike[str]) -> Path:
        """Write the portable JSON envelope to *dest* and return its path."""
        path = Path(dest)
        body = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def export(self, dest: str | os.PathLike[str]) -> Path:
        """Write a ZIP containing the original, byte-identical saved artifacts."""
        self._check_integrity()
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, body in self.raw_artifacts.items():
                archive.writestr(_name(name), body)
        return path


def _envelope(value: dict[str, Any], source: str | None) -> SavedRun:
    if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported saved AWE run envelope schema")
    artifacts, raw_values, fingerprints = value.get("artifacts"), value.get("raw_artifacts"), value.get("fingerprints")
    if not isinstance(artifacts, dict) or not isinstance(raw_values, dict) or not isinstance(fingerprints, dict):
        raise ValueError("The saved run envelope lacks artifact data or fingerprints")
    if set(artifacts) != set(raw_values) or set(artifacts) != set(fingerprints):
        raise ValueError("The saved run envelope contains inconsistent artifact paths")
    if len(raw_values) > MAX_ARTIFACTS:
        raise ValueError("The run contains too many artifacts")
    raw = {}
    for name, encoded in raw_values.items():
        _name(name)
        if not isinstance(encoded, str) or len(encoded) > (MAX_ARTIFACT_BYTES + 2) // 3 * 4:
            raise ValueError("An encoded artifact is invalid or exceeds the size limit")
        try:
            raw[name] = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("An encoded artifact is not valid base64") from exc
        _check_sizes(raw)
        parsed, kind = _parse(name, raw[name])
        if _canonical(parsed) != _canonical(artifacts[name]) or _fingerprint(raw[name], kind) != fingerprints[name]:
            raise ValueError("The saved run envelope artifact fingerprint or data does not match")
    known = {"type", "schema_version", "source", "imported", "artifacts", "raw_artifacts", "fingerprints", "warnings"}
    extra = {key: copy.deepcopy(item) for key, item in value.items() if key not in known}
    original_source = value.get("source")
    if original_source is not None and not isinstance(original_source, str):
        raise ValueError("The saved run source must be a string or null")
    result = _shape(raw, source if source is not None else original_source, extra=extra)
    return result


def _standalone(value: Any, name: str | None = None) -> str:
    if isinstance(value, list):
        return "faults.json"
    if not isinstance(value, dict):
        raise ValueError("The saved AWE artifact must be an object or dossier list")
    if name in _KNOWN:
        return str(name)
    if "graphs" in value:
        return "attribution_graphs.json"
    if "candidates" in value and ("shortlist" in value or "source_cases_sha256" in value):
        return "recommendations.json"
    if "cases" in value and ("summary" in value or any("analysis" in row for row in value.get("cases", []) if isinstance(row, dict))):
        return "symptoms.json"
    if "faults" in value and ("case_id" in value or ("agent" in value and "instance_id" in value)):
        return "case.json"
    if "proposals" in value and "source_cases_sha256" in value:
        return "rsi_proposals.json"
    if "feedback" in value and "source_cases_sha256" in value:
        return "rsi_feedback.json"
    if "cases" in value and "source_cases_sha256" in value and value.get("runtime_input_allowed") is False:
        return "rsi_feedback.json"
    if "operation" in value and "artifact_sha256" in value:
        return "operation-manifest.json"
    if "artifacts" in value or "run_id" in value:
        return "manifest.json"
    raise ValueError("Unrecognized saved AWE artifact shape")


def load_run(source: str | os.PathLike[str] | dict[str, Any] | list[Any]) -> SavedRun:
    """Import a directory, ZIP, saved JSON artifact, or portable run envelope.

    Selecting a local faults/manifest file also loads its sibling artifacts.
    Archives are read in memory, never extracted or executed. A source-case hash
    mismatch, ambiguous run, malformed DAG or unsafe archive fails explicitly.
    """
    if isinstance(source, (dict, list)):
        body = _canonical(source)
        if len(body) > MAX_TOTAL_BYTES * 3:
            raise ValueError("The run exceeds the total size limit")
        if isinstance(source, dict) and source.get("type") == RUN_TYPE:
            return _envelope(source, None)
        name = _standalone(source)
        if name == "case.json":
            return _shape({"faults.json": _canonical([source])}, None)
        return _shape({name: body}, None)
    path = Path(source).expanduser()
    if path.is_symlink():
        raise ValueError("A saved run cannot be a symbolic link")
    origin = str(path.absolute())
    if path.is_dir():
        return _shape(_directory(path), origin)
    if not path.is_file():
        raise ValueError("The saved AWE run does not exist")
    if path.suffix.lower() == ".zip":
        return _shape(_zip(path), origin)
    body = _read_file(path, limit=MAX_TOTAL_BYTES * 3)
    value = _json(body)
    if isinstance(value, dict) and value.get("type") == RUN_TYPE:
        return _envelope(value, origin)
    name = _standalone(value, path.name)
    if name == "case.json":
        return _shape({"faults.json": _canonical([value])}, origin)
    # A manifest describes sibling artifacts. A dossier selected from an output
    # directory is also an unambiguous invitation to read that saved run.
    if path.name in ("manifest.json", "last_run.json", "faults.json", "operation-manifest.json", "operation-result.json"):
        return _shape(_directory(path.parent), origin)
    return _shape({name: body}, origin)
