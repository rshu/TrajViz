"""Content fingerprints for evidence dependencies; no raw content is persisted."""

import hashlib
import inspect
import os
import warnings
import json
from functools import lru_cache
from pathlib import Path

from . import config


def fingerprint(value) -> str:
    def stable(obj):
        if isinstance(obj, dict):
            items = [[stable(k), stable(v)] for k, v in obj.items()]
            return {"mapping": sorted(items, key=lambda kv: json.dumps(kv[0], sort_keys=True))}
        if isinstance(obj, (tuple, list)):
            return [stable(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            return {"set": sorted((stable(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))}
        return obj

    payload = json.dumps(stable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=8192)
def _file_hash(path, identity):
    with open(path, "rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path) -> dict:
    path = Path(path).resolve()
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "sha256": None}
    identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return {"path": str(path), "sha256": _file_hash(str(path), identity)}


def _trajectory_sources(agent, instance_id):
    config.safe_id(agent)
    config.safe_id(instance_id)
    base = config.TRAJECTORY_DIR / agent
    from agent_trajviz.awe.native import _files

    sources = [base / (instance_id + suffix) for suffix in (".json", ".jsonl")]
    children = [child for parent in sources if parent.is_file() for child in _files(parent)]
    return list(dict.fromkeys([*sources, *children]))


def trajectory_fingerprint(agent, instance_id) -> str:
    """Path-sensitive identity for normalization caches and mutation checks."""
    from agent_trajviz.awe.native import adapter_code_hash

    return fingerprint({"sources": [file_fingerprint(path) for path in _trajectory_sources(agent, instance_id)],
                        "adapter_sha256": adapter_code_hash()})


def trajectory_content_fingerprint(agent, instance_id) -> str:
    """Portable parent/child content identity for blinded model verdicts.

    Logical paths retain each child's place in the agent session tree, while
    excluding its dataset location and the machine-specific resolved path.
    """
    from agent_trajviz.awe.native import adapter_code_hash

    base = config.TRAJECTORY_DIR / agent
    sources = [{"path": Path(os.path.relpath(path, base)).as_posix(),
                "sha256": file_fingerprint(path)["sha256"]}
               for path in _trajectory_sources(agent, instance_id)]
    return fingerprint({"agent": agent, "sources": sorted(sources, key=lambda row: row["path"]),
                        "adapter_sha256": adapter_code_hash()})


@lru_cache(maxsize=16)
def _renderer_fingerprint(renderers):
    return fingerprint([inspect.getsource(fn) for fn in renderers])


@lru_cache(maxsize=1024)
def _problem_statement(path, source):
    if source is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8")).get("problem_statement", "")


def verdict_fingerprint(agent, instance_id, model, prompt, renderers, extra=None):
    """Bind blinded verdicts to process inputs, prompt, renderer and schema.

    Content identities omit machine-specific paths. Gold patch and objective
    outcomes are excluded; only the issue statement enters this blinded key.
    """
    from .trajectory import SCHEMA_VERSION
    from . import evidenceview

    raw = config.TRAJECTORY_DIR / agent
    requirement = config.requirements_path(instance_id)
    problem = _problem_statement(str(requirement), file_fingerprint(requirement)["sha256"])
    return fingerprint(
        {
            "raw": [file_fingerprint(raw / (instance_id + ext))["sha256"] for ext in (".json", ".jsonl")],
            "problem_statement": problem,
            "model": model or config.JUDGE_MODEL,
            "prompt": prompt,
            "renderer": _renderer_fingerprint(tuple(renderers)),
            "schema_version": SCHEMA_VERSION,
            "trajectory_fingerprint": trajectory_content_fingerprint(agent, instance_id),
            "evidence_view_code": inspect.getsource(evidenceview),
            "extra": extra,
        }
    )


_warned_verdicts = set()


def validate_verdict(record, expected_key, kind):
    """Return an explicitly labelled cache record, or abstain on stale inputs.

    Older paid verdicts remain readable with an honest legacy marker; strict
    provenance mode abstains from them. Never retroactively certify a legacy
    verdict by hashing its current inputs.
    """
    key = record.get("input_fingerprint")
    if key is None:
        strict = os.environ.get("AWE_STRICT_VERDICT_PROVENANCE", "0") == "1"
        status = "legacy_unverified"
        notice = f"{kind}: legacy verdict provenance is unverified"
        if strict:
            notice += "; strict mode excludes these verdicts"
    elif key == expected_key():
        return {**record, "cache_provenance": {"status": "verified", "fingerprint": key}}
    else:
        strict, status = True, "stale"
        notice = f"{kind}: verdict inputs/prompt changed; stale verdicts are excluded"
    if notice not in _warned_verdicts:
        warnings.warn(notice, RuntimeWarning, stacklevel=2)
        _warned_verdicts.add(notice)
    if strict:
        return None
    return {**record, "cache_provenance": {"status": status}}
