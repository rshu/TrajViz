"""evaluation engine failure-attribution seam for TrajViz.

The dashboard uses the bundled evaluation engine through :func:`diagnose`, which returns a plain
:class:`AttributionResult` (no Gradio types) so it is unit-testable headless.

Integrity model (hardened over three review rounds):

* **Gold-grounded.** evaluation engine's blame needs the run's reference patch + test
  outcome; without them we degrade honestly (never fabricate).
* **Canonical identity.** The displayed run must match the current canonical
  run. Native parent/child trees use TrajViz's complete composite identity;
  single-file archives retain byte identity. Cached normalized source stamps
  must independently match those current inputs.
* **Cache provenance.** Normalized and signal caches bind parent/child inputs
  and the installed parser implementation. Judge/arbiter verdicts require stamped
  ``trajectory_sha256`` matches the canonical trajectory; unverifiable verdicts
  disable the LLM layers for that diagnosis (noted in the result) rather than
  shape it.
* **Isolation.** ``diagnose`` always configures an explicit reference root (the
  immutable import-time default unless the caller passes one) under a module
  lock, so no request inherits another session's root and concurrent callers
  cannot interleave configuration and diagnosis.
* **Input validation.** ``agent`` must be an agent evaluation engine knows and
  ``instance_id`` a plain filename stem — path traversal cannot reach the
  filesystem.
"""
from __future__ import annotations

import json
import inspect
import copy
import importlib
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Causal capability order (mirrors evaluation engine's UPSTREAM_ORDER).
CAPABILITIES: tuple[str, ...] = (
    "requirement_understanding", "task_planning", "code_localization",
    "code_editing", "code_verification", "self_repair_loop", "tool_use",
)
_JUDGE_CAPS = frozenset({"requirement_understanding", "task_planning"})
# Always assessed on a failed case (the deductive core); the rest are conditional
# on an opportunity in the trajectory / a judged verdict.
_ALWAYS_ASSESSED = frozenset({"code_localization", "code_editing"})

# A safe instance id: a single filename stem (no separators, no leading dot).
_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# RLock: diagnose() holds it end-to-end, and the public configure() acquires the
# SAME lock so no caller can rewrite evaluation engine's process-global root/caches while a
# diagnosis is in flight (reentrancy lets diagnose call configure internally).
_LOCK = threading.RLock()


# The evaluation engine is part of TrajViz. Reference data is input, never an
# import path: neither environment paths nor neighbouring repositories select code.
def _reference_root() -> Path:
    configured = os.environ.get("TRAJVIZ_REFERENCE_ROOT")
    return Path(configured).expanduser() if configured else Path.cwd()


# Captured ONCE at import and immutable thereafter. Every diagnose() call
# explicitly configures either this default or the caller's root — never
# "whatever the previous caller left behind".
_DEFAULT_ROOT = _reference_root()
# Read before the bundled engine is imported: an explicit AWE_JUDGE_MODEL always wins.
_JUDGE_MODEL_EXPLICIT = bool(os.environ.get("AWE_JUDGE_MODEL"))


def _select_judge_model(judge_cache_dir: Path, schema: int) -> str | None:
    """The model whose cached verdicts evaluation engine's current schema can verify.

    evaluation engine partitions judge verdicts by model slug and stamps each with the
    evidence schema it was produced under; :func:`_verdict_verifies` refuses any
    other schema. A model named by hand therefore goes stale on the next
    move to another model or schema bump — every verdict in its namespace is refused and
    both LLM layers silently switch off. Choose by content instead: among the
    namespaces holding verdicts stamped with *schema*, the one with the most.
    The model id comes from the records, because the directory slug is lossy
    (``/`` becomes ``__``). ``None`` when no namespace qualifies.
    """
    counts: dict[str, int] = {}
    try:
        namespaces = sorted(d for d in Path(judge_cache_dir).iterdir() if d.is_dir())
    except OSError:
        return None
    for ns in namespaces:
        for record in ns.glob("*/*.json"):
            try:
                doc = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            model = doc.get("model") if isinstance(doc, dict) else None
            if isinstance(model, str) and model and doc.get("evidence_schema") == schema:
                counts[model] = counts.get(model, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda m: (-counts[m], m))


ATTRIBUTION_AVAILABLE = False
_IMPORT_ERROR: str | None = None
try:
    from agent_trajviz.awe._engine import config as _cfg  # noqa: E402
    from agent_trajviz.awe.native import install_native_adapters
    install_native_adapters()
    ATTRIBUTION_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    # Bundled engine import failures can include private environment values
    # or paths. Keep the public reason useful without forwarding their text.
    _IMPORT_ERROR = f"Bundled attribution engine import failed ({type(exc).__name__})"

if ATTRIBUTION_AVAILABLE and not _JUDGE_MODEL_EXPLICIT:
    # evaluation engine reads config.JUDGE_MODEL at call time (judge, arbiter and
    # model_slug), so setting it after import takes effect everywhere. Without a
    # match evaluation engine keeps its own default and the layers degrade with a note.
    from agent_trajviz.awe._engine.trajectory import SCHEMA_VERSION as _SCHEMA_VERSION  # noqa: E402

    _detected_model = _select_judge_model(_cfg.JUDGE_CACHE_DIR, _SCHEMA_VERSION)
    if _detected_model:
        _cfg.JUDGE_MODEL = _detected_model


def configure(reference_root: str | os.PathLike) -> None:
    """Point evaluation engine at a reference root and clear its root-dependent memory caches.

    Acquires the SAME module lock as diagnose(), so an external caller cannot
    rewrite evaluation engine's process-global root/caches while a diagnosis is in flight.
    diagnose() itself re-configures per request (explicit root or the immutable
    default) — configuration is never inherited across callers.
    Older releases cache build_gold_reference/load_outcomes themselves; newer
    releases use uncached wrappers around content-keyed private readers.
    """
    if not ATTRIBUTION_AVAILABLE:
        return
    with _LOCK:
        _configure_unlocked(reference_root)


def _configure_unlocked(reference_root: str | os.PathLike) -> None:
    # evaluation engine resolves every input path from these attributes at call time, so
    # setting them is the whole configuration; the process environment is left
    # alone.
    root = Path(reference_root).resolve()
    _cfg.DATA = root / "data"
    _cfg.REQUIREMENTS_DIR = _cfg.DATA / "requirements"
    _cfg.PATCH_DIR = _cfg.DATA / "patch"
    _cfg.TRAJECTORY_DIR = _cfg.DATA / "trajectory"
    _cfg.LABELS_DIR = _cfg.DATA / "labels"
    from agent_trajviz.awe._engine import gold, outcomes
    for function in (gold.build_gold_reference, outcomes.load_outcomes,
                     getattr(outcomes, "_read_json", None), getattr(outcomes, "_read_outcomes", None)):
        clear = getattr(function, "cache_clear", None)
        if callable(clear):
            clear()


if ATTRIBUTION_AVAILABLE:
    # evaluation engine's own default is its checkout; point it at the reference root from
    # import on, so a caller that reaches evaluation engine before any diagnose() reads the
    # same data diagnose() will. Every diagnose() re-configures explicitly.
    try:
        _configure_unlocked(_DEFAULT_ROOT)
    except Exception as exc:
        # Importing the public attribution facade must remain safe on API drift.
        # Exception text can contain private paths or source data.
        ATTRIBUTION_AVAILABLE = False
        _IMPORT_ERROR = f"Bundled attribution engine configuration failed ({type(exc).__name__})"


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityScore:
    capability: str
    assessed: bool                # was this capability in scope for this case?
    blamed: bool                  # NON-ZERO blame — refuted faults are not blamed
    weight: float
    tier: str | None              # deductive | associational | model_inferred
    top_error: str | None


@dataclass
class AttributionResult:
    available: bool
    reason: str | None = None
    mode: str = "gold_free"       # reference | gold_free
    agent: str | None = None
    instance_id: str | None = None
    blame_status: str | None = None
    primary: dict | None = None
    scorecard: list[CapabilityScore] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)
    arbiter: dict | None = None
    used_judge: bool = False      # judge verdict AVAILABLE for this case
    task: dict | None = None
    notes: list[str] = field(default_factory=list)
    dossier: dict | None = None   # complete backend record, including additive fields


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(p: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _canonical_trajectory_path(agent: str, instance_id: str) -> Path | None:
    from agent_trajviz.awe._engine import adapters

    resolver = getattr(adapters, "canonical_trajectory_path", None)
    if callable(resolver):
        return resolver(agent, instance_id)
    # Current evaluation engine resolves these two sources in _parse_normalized. The
    # normalized source stamp is checked independently before any diagnosis, so
    # an adapter that changes its resolution cannot silently certify this path.
    base = _cfg.TRAJECTORY_DIR / agent
    json_file, jsonl_file = base / f"{instance_id}.json", base / f"{instance_id}.jsonl"
    selected = jsonl_file if agent == "codex" or (jsonl_file.exists() and not json_file.exists()) else json_file
    return selected if selected.is_file() else None


def _supports_layer_controls(case_record) -> bool:
    parameters = inspect.signature(case_record).parameters
    return {"use_judge", "adjudicate"} <= parameters.keys()


def _engine_helper(module: str, name: str):
    """Use additive dossier helpers when this engine API generation provides them.

    Missing optional modules are different from a helper that exists but fails:
    the latter must fail the diagnosis rather than silently drop its evidence.
    """
    qualified = f"agent_trajviz.awe._engine.{module}"
    try:
        imported = importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name != qualified:
            raise
        return None
    helper = getattr(imported, name, None)
    return helper if callable(helper) else None


def _objective_record(agent: str, instance_id: str, canon: Path, canon_sha: str):
    """Current evaluation engine without per-call dossier controls: keep LLM layers off.

    Never monkeypatch process-global evaluation engine functions or temporarily move its
    cache directories. Such changes could affect independent evaluation engine consumers.
    Build the objective dossier from its explicit detector and blame APIs.
    """
    from agent_trajviz.awe._engine.adapters import load_normalized
    from agent_trajviz.awe._engine.detect import detect
    from agent_trajviz.awe._engine import dossier as dossier_module
    from agent_trajviz.awe._engine.dossier import file_hunks
    from agent_trajviz.awe._engine.evidence import build_chain, changed_lines
    from agent_trajviz.awe._engine.gold import build_gold_reference
    from agent_trajviz.awe._engine.rootcause import build_blame

    trajectory = load_normalized(agent, instance_id)
    stamp = getattr(trajectory, "source", None)
    if (not isinstance(stamp, dict) or stamp.get("sha256") != canon_sha
            or not isinstance(stamp.get("path"), str)
            or Path(stamp["path"]).resolve() != canon.resolve()):
        raise ValueError("normalized trajectory identity could not be verified")
    detection = detect(instance_id, agent, use_judge=False)
    if detection.get("resolved") is not False:
        return None, detection
    blame = build_blame(detection, adjudicate=None)
    gold = build_gold_reference(instance_id)
    requirement = json.loads(_cfg.requirements_path(instance_id).read_text(encoding="utf-8"))
    gold_text = requirement.get("gold_patch") or ""
    patch = _cfg.patch_path(agent, instance_id)
    agent_text = patch.read_text(encoding="utf-8") if patch.is_file() else ""
    gold_changed, agent_changed = changed_lines(gold_text), changed_lines(agent_text)
    gold_hunks, agent_hunks = file_hunks(gold_text), file_hunks(agent_text)
    primary = blame.get("primary")
    weights = {(row["capability"], row["error_type"]): row["weight"] for row in blame["blame"]}
    step_text = {step.step_id: (step.text or step.output_excerpt or "")[:200]
                 for step in trajectory.steps if step.text or step.output_excerpt}
    load_pertest = _engine_helper("outcomes", "load_pertest")
    pertest = load_pertest(agent, instance_id) if load_pertest else None
    audit_fault = _engine_helper("dossier", "audit_fault")
    select_remedies = _engine_helper("dossier", "select_remedies")
    faults = []
    for candidate in detection["candidates"]:
        capability, error = candidate["capability"], candidate["error_type"]
        weight = weights.get((capability, error), 0.0)
        is_primary = bool(primary and (primary["capability"], primary["error_type"]) == (capability, error))
        code = {}
        if capability == "code_localization":
            missed = candidate.get("evidence", {}).get("missed_gold") or gold.gold_files
            code["code_that_needed_changing_but_was_missed"] = {name: gold_hunks[name] for name in missed if name in gold_hunks}
        elif capability == "code_editing":
            code = {"what_the_agent_changed": agent_hunks, "reference_fix": gold_hunks}
        faults.append({"capability": capability, "error_type": error, "confidence": candidate["confidence"],
                       "blame_weight": weight, "is_primary": is_primary,
                       "detector_evidence": candidate.get("evidence", {}), "code_evidence": code,
                       "evidence_chain": build_chain(candidate, detection, gold, gold_changed, agent_changed,
                                                     weight, is_primary, step_text=step_text)})
        labels = getattr(dossier_module, "CAP_LABEL", {})
        if isinstance(labels, dict):
            faults[-1]["label"] = f"{labels.get(capability, capability)} → {error}"
        if audit_fault:
            audit = audit_fault(
                {"capability": capability, "error_type": error, "confidence": candidate["confidence"],
                 "detector_evidence": candidate.get("evidence", {})},
                agent=agent, inst=instance_id, gold_text=gold_text, agent_text=agent_text,
                resolved=detection["resolved"], traj=trajectory,
                test_files=getattr(gold, "test_patch_files", []),
                problem_statement=gold.problem_statement, fail_to_pass=getattr(gold, "fail_to_pass", []),
                pertest=pertest, profile=getattr(gold, "benchmark_profile", {}))
            faults[-1]["audit"] = {"verdict": audit["verdict"], "mismatches": audit["mismatches"]}
        if select_remedies:
            faults[-1].update(select_remedies({"agent": agent, "arbiter": None}, faults[-1], weight=weight))
    faults.sort(key=lambda fault: -fault["blame_weight"])
    record = {"case_id": f"{agent}__{instance_id}", "agent": agent, "instance_id": instance_id,
              "blame_status": blame["status"], "faults": faults, "arbiter": None,
              "task": {"problem_statement": gold.problem_statement, "gold_files": gold.gold_files,
                       "multi_file": gold.is_multi_file, "cross_module": gold.is_cross_module},
              "reasoning_observability": ("partly hidden (encrypted)" if any(
                  getattr(step, "observability", None) == "hidden" for step in trajectory.steps) else "observed")}
    if hasattr(gold, "repo"):
        record["repo"] = gold.repo
    if "decision_trace" in blame:
        record["attribution_trace"] = blame["decision_trace"]
    context = _engine_helper("dossier", "for_case")
    if context:
        record.update(context(agent, instance_id, getattr(gold, "benchmark_profile", {})))
    fault_side = _engine_helper("dossier", "fault_side_assess")
    if fault_side:
        record["fault_side"] = fault_side(detection.get("signals"))
    thinking_excerpt = _engine_helper("dossier", "thinking_excerpt")
    if thinking_excerpt:
        record["thinking_process"] = thinking_excerpt(trajectory)
    event_ledger = _engine_helper("dossier", "event_ledger")
    if event_ledger:
        record["event_ledger"] = event_ledger(trajectory)
    analyze_symptoms = _engine_helper("dossier", "analyze_symptoms")
    compact_symptoms = _engine_helper("dossier", "compact_symptoms")
    if analyze_symptoms and compact_symptoms:
        symptoms = compact_symptoms(analyze_symptoms(trajectory, agent_text))
        symptoms["patch_available"] = patch.is_file()
        record["symptom_analysis"] = symptoms
    return record, detection


def _verdict_verifies(rec: dict, *, canon_sha: str, req_sha: str | None,
                      prompt_version: str, schema: int) -> bool:
    """A cached LLM verdict is trusted only when its FULL prompt provenance is
    verifiable: the trajectory it judged, the gold reference whose task
    statement its prompt embedded, the prompt version, and the evidence schema
    all match the current inputs. Legacy/unstamped records fail (cannot verify)."""
    return (rec.get("trajectory_sha256") == canon_sha
            and rec.get("requirements_sha256") == req_sha
            and rec.get("prompt_version") == prompt_version
            and rec.get("evidence_schema") == schema)


def _llm_layer_policy(agent: str, instance_id: str, canon_sha: str):
    """Decide whether cached judge/arbiter verdicts may shape this diagnosis.

    Validates the complete prompt fingerprint (trajectory + requirements +
    prompt_version + evidence_schema; the arbiter's claim is already part of
    its cache key). Any unverifiable record disables the corresponding layer —
    a stale verdict must never judge a different trajectory OR a different task.

    Returns (use_judge, adjudicate, notes): use_judge False disables the judge;
    adjudicate None disables the arbiter; _AUTO keeps evaluation engine's default.
    """
    import hashlib
    import json as _json
    import agent_trajviz.awe._engine.arbiter as _arb
    import agent_trajviz.awe._engine.judge as _judge
    from agent_trajviz.awe._engine.adapters import requirements_source_sha256
    from agent_trajviz.awe._engine.dossier import _AUTO
    from agent_trajviz.awe._engine.trajectory import SCHEMA_VERSION

    notes: list[str] = []
    req_sha = requirements_source_sha256(instance_id)
    j_prompt = hashlib.sha256(_judge.SYSTEM_PROMPT.encode()).hexdigest()[:12]
    a_prompt = hashlib.sha256(_arb.SYSTEM_PROMPT.encode()).hexdigest()[:12]

    use_judge: bool | None = None      # evaluation engine auto
    jrec = _judge.verdict_cached(agent, instance_id)
    if jrec is not None and not _verdict_verifies(
            jrec, canon_sha=canon_sha, req_sha=req_sha,
            prompt_version=j_prompt, schema=SCHEMA_VERSION):
        use_judge = False
        notes.append("judge verdict ignored: its recorded provenance "
                     "(trajectory/task/prompt/schema) does not match the current "
                     "inputs (unverifiable)")

    adjudicate = _AUTO
    adir = _arb.ARBITER_CACHE_DIR / _cfg.model_slug() / agent
    if adir.is_dir():
        # Match ONLY this instance's verdict files: exact legacy "{id}.json" or
        # claim-keyed "{id}__{cap}__{et}.json". A bare "{id}*" prefix glob would
        # also match a DIFFERENT instance whose id extends this one (e.g.
        # proj-1147 matching proj-11477's files) and wrongly disable the arbiter.
        candidates = (list(adir.glob(f"{instance_id}.json"))
                      + list(adir.glob(f"{instance_id}__*.json")))
        for f in candidates:
            try:
                rec = _json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # Unreadable or malformed candidate file — skip it; provenance
                # verification below decides whether the arbiter stays enabled.
                continue
            if not _verdict_verifies(rec, canon_sha=canon_sha, req_sha=req_sha,
                                     prompt_version=a_prompt,
                                     schema=SCHEMA_VERSION):
                adjudicate = None
                notes.append("arbiter verdict ignored: its recorded provenance "
                             "(trajectory/task/prompt/schema) does not match "
                             "the current inputs (unverifiable)")
                break
    return use_judge, adjudicate, notes


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def diagnose(*, agent: str | None, instance_id: str | None,
             source_path: str | os.PathLike | None = None, fmt: str | None = None,
             expected_sha: str | None = None, merged_sources: int = 0,
             reference_root: str | os.PathLike | None = None) -> AttributionResult:
    """Diagnose the failure of the *displayed* trajectory. Never raises; never
    fabricates; degrades with an explicit reason. See the module docstring for
    the integrity model.

    ``expected_sha`` — the sha256 captured when the UI LOADED the trajectory
    (the immutable identity of the displayed content). Diagnosis requires the
    canonical file's CURRENT bytes to equal it, so a real export mutated between
    load and diagnosis is refused rather than diagnosed while the UI still
    shows the old state.

    ``merged_sources`` — how many additional child files were displayed. Native
    trees are certified by their composite identity and the complete source
    fingerprint, rather than by the parent file alone.
    """
    if fmt == "otlp":
        return AttributionResult(False, reason="failure attribution does not support saved OTLP captures; "
                                 "use the native trajectory and its verified reference data.")
    if not ATTRIBUTION_AVAILABLE:
        return AttributionResult(False, reason=f"Attribution unavailable ({_IMPORT_ERROR})")
    with _LOCK:
        try:
            return _diagnose_locked(agent=agent, instance_id=instance_id,
                                    source_path=source_path, fmt=fmt,
                                    expected_sha=expected_sha,
                                    merged_sources=merged_sources,
                                    reference_root=reference_root)
        except Exception as exc:
            return AttributionResult(False, mode="reference", agent=agent, instance_id=instance_id,
                                     reason=f"Attribution diagnosis unavailable ({type(exc).__name__})")


def _diagnose_locked(*, agent, instance_id, source_path, fmt, expected_sha,
                     reference_root, merged_sources=0):
    # per-call, explicit configuration — never inherited from a previous caller
    _configure_unlocked(reference_root if reference_root else _DEFAULT_ROOT)

    if not (agent and instance_id):
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="capability attribution needs (agent, instance_id); an arbitrary "
                   "trajectory upload has no reference patch to ground blame")
    # Validate BEFORE any filesystem access: agent from evaluation engine's known agents,
    # instance_id a single safe filename stem (no traversal, no separators).
    from agent_trajviz.awe._engine import config
    if not isinstance(agent, str) or not _VALID_ID.fullmatch(agent) or ".." in agent:
        return AttributionResult(False, mode="gold_free", agent=agent, instance_id=instance_id,
                                 reason="invalid agent id (must be a plain agent name)")
    if not _VALID_ID.fullmatch(instance_id) or ".." in instance_id:
        return AttributionResult(False, mode="gold_free", agent=agent, instance_id=instance_id,
                                 reason="invalid instance id (must be a plain instance name)")
    native_agents = {"claude_code", "cursor", "opencode", "codex", "pi", "dsh", "deepseek_harness",
                     "deepseek-harness", "codearts", "icode"}
    declared_agent = ((config.TRAJECTORY_DIR / agent).is_dir()
                      and (config.DATA / f"eval_{agent}.json").is_file())
    if agent not in config.AGENTS and agent not in native_agents and not declared_agent:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"could not determine the agent for '{instance_id}' "
                   f"(got '{agent}'). Set the agent override to one of: "
                   f"{', '.join(config.AGENTS)}")
    if not config.requirements_path(instance_id).is_file():
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"no gold reference (data/requirements/{instance_id}.json) under "
                   f"{config.DATA.parent} — attribution needs the reference patch + "
                   f"test outcome. Provide matching reference data, or inspect "
                   f"trajectory-only diagnostics in Overview.")

    # Canonical identity: resolve via evaluation engine's OWN source resolution so we verify
    # the very file diagnosis will parse (codex prefers .jsonl).
    canon = _canonical_trajectory_path(agent, instance_id)
    if canon is None:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"no real trajectory for {agent}/{instance_id} under "
                   f"{config.DATA.parent} — cannot verify the displayed trajectory "
                   f"belongs to this run")
    # The UI's immutable identity includes every merged native child. A raw
    # parent SHA identifies only a single-file archive; obtain the independent
    # native identity before trusting any cached normalized source stamp.
    from agent_trajviz.awe._engine.provenance import trajectory_fingerprint
    from agent_trajviz.awe._engine.adapters import load_normalized
    from agent_trajviz.awe.native import _native_candidate

    parent_sha = _sha256(canon)
    source_fingerprint = trajectory_fingerprint(agent, instance_id)
    if source_path and Path(source_path).suffix.lower() == ".zip":
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="a ZIP archive identifies its archive bytes, while the reference identifies a native session tree; "
                   "load the matching native session files to certify attribution.")
    normalized = load_normalized(agent, instance_id)
    if normalized is None or not normalized.steps:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason="the evaluation engine could not extract workflow steps from this trajectory; "
                   "no gold-grounded verdict is issued. Use a supported native session format.")
    native = _native_candidate(canon)
    canon_sha = native.sha256 if native is not None else parent_sha
    stamp = getattr(normalized, "source", None)
    if (canon_sha is None or not isinstance(stamp, dict) or stamp.get("sha256") != canon_sha
            or not isinstance(stamp.get("path"), str) or Path(stamp["path"]).resolve() != canon.resolve()):
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason="the evaluation engine could not verify the canonical trajectory's content identity; "
                   "reload the native session and retry.")
    displayed_sha = expected_sha
    if displayed_sha is None and source_path is not None:
        if native is not None:
            try:
                from agent_trajviz import load
                displayed_sha = load(source_path).sha256
            except (ValueError, OSError):
                displayed_sha = None
        else:
            displayed_sha = _sha256(Path(source_path))
    if source_path is not None and displayed_sha is None:
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="cannot establish the displayed trajectory's content identity "
                   "(source unreadable) — refusing rather than diagnosing unverified bytes")
    if displayed_sha is not None and displayed_sha != canon_sha:
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"the displayed trajectory does not match the canonical {agent}/{instance_id} run's current content — "
                   "either it belongs to a different execution, or the real export changed after it was loaded "
                   "(reload to re-sync). A gold-grounded verdict would otherwise describe different bytes than the ones shown")

    from agent_trajviz.awe._engine.dossier import case_record
    from agent_trajviz.awe._engine.detect import detect
    requirement_sha = _sha256(config.requirements_path(instance_id))
    patch_path = config.patch_path(agent, instance_id)
    eval_resolver = getattr(config, "eval_path", None)
    outcome_path = eval_resolver(agent) if callable(eval_resolver) else config.DATA / f"eval_{agent}.json"
    patch_sha, outcome_sha = _sha256(patch_path), _sha256(outcome_path)
    project_data = getattr(config, "PROJECT_DATA", None)
    optional_evidence = ([Path(project_data) / "overrides" / "resolved_overrides.json",
                          Path(project_data) / "pertest" / agent / f"{instance_id}.json",
                          Path(project_data) / "pertest" / agent / f"{instance_id}.log.json"]
                         if project_data is not None else [])
    # Full dossiers can read declared execution context. Its identity belongs to
    # the same before/after guard as patches, per-test records and overrides.
    context_path = os.environ.get("AWE_EXECUTION_CONTEXT")
    if context_path:
        optional_evidence.append(Path(context_path))
    optional_fingerprints = [_sha256(path) for path in optional_evidence]
    used_judge = False
    try:
        if _supports_layer_controls(case_record):
            # Legacy evaluation engine: use its full dossier only with verified cached
            # verdicts and its explicit per-call controls.
            use_judge, adjudicate, notes = _llm_layer_policy(agent, instance_id, canon_sha)
            rec = case_record(agent, instance_id, use_judge=use_judge, adjudicate=adjudicate)
            d = detect(instance_id, agent, use_judge=use_judge)
            # None selects evaluation engine's automatic cached-verdict mode. Its detector
            # reports whether this case actually had a usable verdict; task
            # opportunities or the policy flag alone cannot establish that.
            used_judge = (use_judge is not False
                          and (d.get("signals") or {}).get("judge_available") is True)
        else:
            rec, d = _objective_record(agent, instance_id, canon, canon_sha)
            notes = ["Automatic attribution uses objective evidence only, without cached LLM verdicts; "
                     "use AWE operations for verified cached judgments."]
    except Exception as exc:  # never surface a raw traceback to the UI
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"diagnosis failed for {agent}/{instance_id} ({type(exc).__name__})")
    if rec is None:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason="this run resolved (or its outcome is unknown); attribution applies to "
                   "failures only")
    # Close the mid-diagnosis mutation window: case_record/detect re-read the
    # reference data from disk AFTER the identity gate above, and an external
    # writer (e.g. a git pull in that data) could swap the file in between — the
    # result would describe bytes the gate never approved and the UI never
    # displayed. Recompute the canonical hash after ALL reads and refuse on any
    # change (the in-process lock cannot serialize other processes).
    if _sha256(canon) != parent_sha or trajectory_fingerprint(agent, instance_id) != source_fingerprint:
        return AttributionResult(
            False, mode="reference", agent=agent, instance_id=instance_id,
            reason=f"the real trajectory for {agent}/{instance_id} changed "
                   f"while the diagnosis was running — reload and retry")
    if _sha256(config.requirements_path(instance_id)) != requirement_sha:
        return AttributionResult(False, mode="reference", agent=agent, instance_id=instance_id,
                                 reason="the reference task changed while the diagnosis was running — reload and retry")
    if _sha256(patch_path) != patch_sha:
        return AttributionResult(False, mode="reference", agent=agent, instance_id=instance_id,
                                 reason="the final agent patch changed while the diagnosis was running — reload and retry")
    if _sha256(outcome_path) != outcome_sha:
        return AttributionResult(False, mode="reference", agent=agent, instance_id=instance_id,
                                 reason="the benchmark outcome changed while the diagnosis was running — reload and retry")
    if [_sha256(path) for path in optional_evidence] != optional_fingerprints:
        return AttributionResult(False, mode="reference", agent=agent, instance_id=instance_id,
                                 reason="the optional evaluation evidence changed while the diagnosis was running — reload and retry")
    rec = {**rec, "source_certification": {
        "status": "displayed_content_verified", "trajectory_sha256": canon_sha,
        "trajectory_fingerprint": source_fingerprint, "trajectory": copy.deepcopy(stamp),
        "requirements_sha256": requirement_sha, "patch_sha256": patch_sha, "outcome_sha256": outcome_sha,
    }}
    return _shape(rec, d.get("opportunities", {}), notes, used_judge=used_judge)


def _shape(rec: dict, opportunities: dict, notes: list[str], *, used_judge: bool | None = None) -> AttributionResult:
    """case_record dict + detection opportunities -> AttributionResult.

    Blame is NON-ZERO only (arbiter-refuted faults carry weight 0). Assessment
    is per capability: the deductive core always; every other capability —
    including EACH judge capability separately — only when its own opportunity
    opened (evaluation engine can judge requirement_understanding while task_planning is out
    of scope for the same case)."""
    faults = rec.get("faults", [])
    primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                    for f in faults if f.get("is_primary")), None)

    agg: dict[str, dict] = {}
    for f in faults:
        cap = f["capability"]
        w = float(f.get("blame_weight") or 0.0)
        a = agg.setdefault(cap, {"weight": 0.0, "errors": {}, "tier_of": {}})
        a["weight"] += w
        if w > 0:
            et = f["error_type"]
            a["errors"][et] = a["errors"].get(et, 0.0) + w
            a["tier_of"][et] = (f.get("evidence_chain") or {}).get("strength")

    scorecard = []
    for cap in CAPABILITIES:
        a = agg.get(cap)
        weight = round(a["weight"], 3) if a else 0.0
        blamed = weight > 0
        assessed = blamed or cap in _ALWAYS_ASSESSED or bool(opportunities.get(cap))
        top = (max(a["errors"], key=a["errors"].get) if (a and a["errors"]) else None)
        tier = a["tier_of"].get(top) if (a and top) else None
        scorecard.append(CapabilityScore(cap, assessed, blamed, weight, tier, top))

    if used_judge is None:
        used_judge = any(bool(opportunities.get(c)) for c in _JUDGE_CAPS) or \
            any(f["capability"] in _JUDGE_CAPS for f in faults)

    return AttributionResult(
        available=True, mode="reference",
        agent=rec.get("agent"), instance_id=rec.get("instance_id"),
        blame_status=rec.get("blame_status"), primary=primary,
        scorecard=scorecard, faults=faults, arbiter=rec.get("arbiter"),
        used_judge=used_judge, task=rec.get("task"), notes=notes, dossier=copy.deepcopy(rec),
    )
