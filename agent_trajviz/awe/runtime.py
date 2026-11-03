"""Bundled evaluation operations, isolated from the server and source data."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MAX_LOG = 64_000
_MAX_INPUT = 16_000_000
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REFERENCE_OPERATIONS = {"diagnose", "pipeline", "judge", "arbiter", "scout", "symptoms", "cohort",
                         "behaviors", "fragile", "preflight", "validation", "pertest_report", "pertest_harness",
                         "grader_log_audit", "harness_suspects", "calibration", "calibration_set",
                         "threshold_stability", "difficulty_strata"}
_INSTANCE_OPERATIONS = {"diagnose", "preprocessing", "pertest_harness", "judge", "arbiter", "scout"}
_LIMIT_OPERATIONS = {"pipeline", "judge", "arbiter", "scout", "symptoms", "cohort", "behaviors", "fragile",
                    "preflight", "pertest_harness", "ingest_harbor", "calibration_set"}
_OPERATIONS = {
    "diagnose": ("Diagnose one case", "Analysis", None, False),
    "pipeline": ("Analyze and archive a batch", "Analysis", "run_awe.py", False),
    "judge": ("Generate judge verdicts", "Models", "run_judge.py", True),
    "arbiter": ("Review primary candidates", "Models", "run_arbiter.py", True),
    "symptoms": ("Scan symptoms across outcomes", "Analysis", "scan_symptoms.py", False),
    "graphs": ("Generate attribution graphs", "Saved results", "gen_attribution.py", False),
    "recommendations": ("Generate recommendation plans", "Saved results", "gen_recommendations.py", False),
    "remedies": ("Refresh case remedies", "Saved results", "refresh_remedies.py", False),
    "rsi": ("Export RSI proposals and feedback", "Experiments", "gen_rsi.py", False),
    "cohort": ("Contrast solved and failed cohorts", "Discovery", "cohort_report.py", False),
    "scout": ("Discover candidate failure modes", "Discovery", "run_scout.py", True),
    "behaviors": ("Report discovered behaviors", "Discovery", "behaviors_report.py", False),
    "fragile": ("Review fragile solves", "Discovery", "fragile_report.py", False),
    "preflight": ("Validate benchmark inputs", "Validation", "validate_inputs.py", False),
    "preprocessing": ("Validate preprocessing retention", "Validation", "validate_preprocessing.py", False),
    "validation": ("Run detector and evidence controls", "Validation", "run_validation.py", False),
    "fault_injection": ("Run controlled fault injection", "Validation", "run_fault_injection.py", False),
    "publication": ("Validate a saved publication", "Validation", None, False),
    "rsi_prepare": ("Prepare a policy variant", "Experiments", None, False),
    "rsi_decide": ("Evaluate a prepared policy", "Experiments", None, False),
    "ingest_harbor": ("Import Harbor / LoLBench runs", "Inputs", "ingest_harbor.py", False),
    "pertest_report": ("Review per-test coverage", "Inputs", "pertest_report.py", False),
    "pertest_harness": ("Collect official per-test evaluations", "Inputs", "pertest_harness.py", True),
    "grader_log_audit": ("Audit official grader logs", "Inputs", "grader_log_audit.py", False),
    "harness_suspects": ("Review evaluator suspects", "Inputs", "harness_suspects.py", False),
    "catalog": ("Inspect remedy catalog and source provenance", "Experiments", None, False),
    "remedy_verify": ("Validate scoped remedy experiment evidence", "Experiments", None, False),
    "guardrail_decide": ("Evaluate an advisory guardrail", "Experiments", None, False),
    "guardrail_observe": ("Evaluate public runtime observations", "Experiments", None, False),
    "papers_status": ("Inspect paper processing provenance", "Experiments", "papers_status.py", False),
    "scaffold_view": ("View remedies by scaffold layer", "Experiments", "scaffold_view.py", False),
    "calibration": ("Review thresholds and blame sensitivity", "Validation", "calibrate.py", False),
    "calibration_set": ("Prepare blinded human annotation tasks", "Validation", "build_calibration_set.py", False),
    "agreement": ("Measure human annotation agreement", "Validation", "score_agreement.py", False),
    "judge_agreement": ("Compare cached judge models", "Validation", "judge_agreement.py", False),
    "judge_panel": ("Analyze cached judge panel votes", "Validation", "analyze_judge_panel.py", False),
    "threshold_stability": ("Review threshold stability", "Validation", "threshold_stability.py", False),
    "difficulty_strata": ("Review difficulty strata calibration", "Validation", "calibrate_difficulty_strata.py", False),
    "bibliography": ("Export catalog reference bibliography", "Experiments", "export_bib.py", False),
    "showcase": ("Export a saved results showcase", "Saved results", "build_showcase.py", False),
}
_BOOLEAN_OPTIONS = {"mock", "dry_run", "force", "include_resolved", "consolidate"}
_PAYLOAD_FIELDS = {
    "diagnose": {"use_judge", "use_arbiter"},
    "judge": {"mock", "dry_run", "force", "workers", "panel", "include_resolved"},
    "arbiter": {"mock", "dry_run", "force", "workers", "panel"},
    "scout": {"mock", "dry_run", "force", "workers", "consolidate"},
    "pipeline": {"workers"}, "symptoms": {"workers"},
    "graphs": {"dossier"}, "recommendations": {"dossier"}, "remedies": {"dossier"},
    "rsi": {"dossier"}, "validation": {"dossier"}, "publication": {"dossier"},
    "rsi_prepare": {"proposal", "observed_context"},
    "rsi_decide": {"prepared", "state", "trigger_count"},
    "fault_injection": {"language", "case_timeout"},
    "ingest_harbor": {"harbor_root", "run_groups", "all_run_groups", "status_dir", "agent_name", "trials",
                      "include_exception_failures", "namespace_instances"},
    "pertest_harness": {"confirm_execution", "gold", "source_only", "case_timeout", "keep_logs", "redo",
                        "python_executable", "dataset", "image_namespace", "image_platform"},
    "grader_log_audit": {"python_executable"}, "agreement": {"mock"},
    "judge_agreement": {"model_a", "model_b"},
    "judge_panel": {"panel_cache", "baseline_cache"},
    "papers_status": {"papers_dir"}, "catalog": {"catalog_path"},
    "remedy_verify": {"recipe_id", "evidence_root", "catalog_path"},
    "guardrail_decide": {"run_id", "global_budget", "snapshot", "recipe_id", "event_id", "observed"},
    "guardrail_observe": {"kind", "values"},
    "showcase": {"showcase_data"},
}


@dataclass(frozen=True)
class EvaluationSettings:
    """Session configuration. Credentials are never serialized or passed as flags."""

    reference_root: str | os.PathLike[str] | None = None
    project_data: str | os.PathLike[str] | None = None
    output_root: str | os.PathLike[str] | None = None
    agents: tuple[str, ...] = ()
    benchmark_profile: dict[str, Any] | str | os.PathLike[str] | None = None
    benchmark_name: str = ""
    execution_context: dict[str, Any] | str | os.PathLike[str] | None = None
    judge_model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    temperature: float = 0.0
    max_tokens: int = 2400
    timeout: int = 300
    allow_network: bool = False
    judge_min_agree: float = 0.0

    def nonsecret(self) -> dict[str, Any]:
        """Connection preferences suitable for optional browser storage."""
        return {item.name: (str(value) if isinstance(value, os.PathLike) else value)
                for item in dataclasses.fields(self) if item.name != "api_key"
                for value in [getattr(self, item.name)]}


@dataclass(frozen=True)
class OperationResult:
    success: bool
    available: bool
    reason: str | None
    operation: str
    output_dir: str | None = None
    project_data: str | None = None
    data: Any = None
    artifacts: tuple[str, ...] = ()
    logs: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _engine_root() -> Path:
    return Path(__file__).with_name("_engine").resolve()


def available_operations(settings: EvaluationSettings | None = None) -> list[dict[str, Any]]:
    """List the bundled engine's operation catalog and required input contracts."""
    root = _engine_root()
    found = (root / "__init__.py").is_file()
    result = []
    for identity, (label, group, script, network) in _OPERATIONS.items():
        present = found and (script is None or (root / "scripts" / script).is_file())
        reason = None if present else ("The bundled evaluation engine is unavailable; reinstall TrajViz." if not found
                                       else "This installed package does not provide the operation.")
        requirements = (["Docker", "SWE-bench evaluator Python", "Pulls and removes per-instance Docker images",
                         "Explicit external-execution confirmation"]
                        if identity == "pertest_harness" else ["SWE-bench evaluator Python"]
                        if identity == "grader_log_audit" else [])
        result.append({"id": identity, "label": label, "group": group, "requires_network": network,
                       "requires_reference": identity in _REFERENCE_OPERATIONS,
                       "selectors": {"instance_id": identity in _INSTANCE_OPERATIONS, "limit": identity in _LIMIT_OPERATIONS},
                       "requires_execution": identity == "pertest_harness", "requirements": requirements,
                       "options": sorted(_PAYLOAD_FIELDS.get(identity, set())),
                       "available": present, "reason": reason})
    return result


def _identifier(value: str | None, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not _SAFE_ID.fullmatch(value) or ".." in value):
        raise ValueError(f"{name} must be a plain artifact identifier.")
    return value


def _json_object(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        path = Path(value).expanduser()
        if path.stat().st_size > _MAX_INPUT:
            raise ValueError(f"{name} is too large.")
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    json.dumps(value, allow_nan=False)
    return value


def _validated(operation: str, settings: EvaluationSettings, agent: str | None, instance_id: str | None,
               limit: int | None, payload: dict[str, Any] | None) -> tuple[Path, Path | None, dict[str, Any]]:
    if operation not in _OPERATIONS:
        raise ValueError("Unknown AWE operation.")
    needs_reference = operation in _REFERENCE_OPERATIONS or (operation == "preprocessing" and bool(agent or instance_id))
    engine = _engine_root()
    if not (engine / "__init__.py").is_file():
        raise ValueError("The bundled evaluation engine is unavailable; reinstall TrajViz.")
    script = _OPERATIONS[operation][2]
    if script and not (engine / "scripts" / script).is_file():
        raise ValueError("This installed package does not provide the operation.")
    reference = None
    if settings.reference_root is not None and str(settings.reference_root).strip():
        reference = Path(settings.reference_root).expanduser().resolve()
        if not reference.is_dir():
            raise ValueError("Reference data root must be an existing directory.")
    elif needs_reference:
        raise ValueError("Select an explicit reference data directory for this evaluation operation.")
    _identifier(agent, "Agent")
    _identifier(instance_id, "Instance")
    if instance_id is not None and operation not in _INSTANCE_OPERATIONS:
        raise ValueError("This operation does not support an instance selector; choose a supported case operation or clear the selector.")
    if limit is not None and operation not in _LIMIT_OPERATIONS:
        raise ValueError("This operation does not support a case limit; clear the limit before running it.")
    if operation == "diagnose" and not (agent and instance_id):
        raise ValueError("Diagnosis needs an agent and instance identifier.")
    if operation == "preprocessing" and bool(agent) != bool(instance_id):
        raise ValueError("Native preprocessing replay needs both agent and instance.")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1_000_000):
        raise ValueError("Limit must be a positive integer.")
    if not isinstance(settings.agents, (tuple, list)) or len(set(settings.agents)) != len(settings.agents):
        raise ValueError("Agents must be unique identifiers.")
    for selected in settings.agents:
        _identifier(selected, "Agent")
    if type(settings.allow_network) is not bool:
        raise ValueError("Allow network must be a checkbox value.")
    if type(settings.timeout) is not int or not 1 <= settings.timeout <= 86_400:
        raise ValueError("Timeout must be an integer between 1 and 86400 seconds.")
    if type(settings.max_tokens) is not int or not 1 <= settings.max_tokens <= 1_000_000:
        raise ValueError("Max tokens must be a positive integer.")
    if isinstance(settings.judge_min_agree, bool) or not isinstance(settings.judge_min_agree, (int, float)) or not 0 <= settings.judge_min_agree <= 1:
        raise ValueError("Minimum panel agreement must be between zero and one.")
    if isinstance(settings.temperature, bool) or not isinstance(settings.temperature, (int, float)) or not math.isfinite(settings.temperature):
        raise ValueError("Temperature must be finite.")
    if settings.base_url:
        parsed = urlsplit(settings.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Base URL must be an HTTP(S) endpoint without credentials, query or fragment.")
    if not isinstance(settings.api_key, str) or not isinstance(settings.judge_model, str):
        raise ValueError("Model and API key must be text.")
    value = {} if payload is None else dict(payload)
    if set(value) - _PAYLOAD_FIELDS.get(operation, set()):
        raise ValueError("Unknown options for this AWE operation.")
    for key in _BOOLEAN_OPTIONS | {"use_judge", "use_arbiter", "confirm_execution", "gold", "source_only",
                                   "keep_logs", "redo", "all_run_groups", "include_exception_failures", "namespace_instances"}:
        if key in value and type(value[key]) is not bool:
            raise ValueError(f"{key} must be a checkbox value.")
    for key, maximum in (("workers", 16), ("panel", 9)):
        if key in value and (type(value[key]) is not int or not 1 <= value[key] <= maximum):
            raise ValueError(f"{key} is outside the supported range.")
    if "trigger_count" in value and (type(value["trigger_count"]) is not int or value["trigger_count"] < 0):
        raise ValueError("Trigger count must be a nonnegative integer.")
    case_maximum = 86_400 if operation == "pertest_harness" else 120
    if "case_timeout" in value and (type(value["case_timeout"]) not in (float, int) or not 0 < value["case_timeout"] <= case_maximum):
        raise ValueError("Case timeout is outside the supported range.")
    if value.get("language", "all") not in {"all", "python", "javascript"}:
        raise ValueError("Unknown fault-injection language.")
    json.dumps(value, allow_nan=False)
    if operation == "pertest_harness":
        if not settings.allow_network or value.get("confirm_execution") is not True:
            raise ValueError("Official evaluation executes Docker trials and pulls images; enable network and confirm execution explicitly.")
        if not agent:
            raise ValueError("Official evaluation needs an agent identifier.")
        if shutil.which("docker") is None:
            raise ValueError("Official evaluation needs Docker installed and running.")
    if operation == "ingest_harbor":
        if not isinstance(value.get("harbor_root"), str) or not Path(value["harbor_root"]).expanduser().is_dir():
            raise ValueError("Harbor import needs an existing harbor_root directory.")
        groups = value.get("run_groups", [])
        if not isinstance(groups, list) or any(_identifier(item, "Run group") is None for item in groups):
            raise ValueError("Run groups must be plain identifiers.")
        if bool(groups) == bool(value.get("all_run_groups")):
            raise ValueError("Select run_groups or all_run_groups for Harbor import.")
        trials = value.get("trials", [])
        if not isinstance(trials, list) or any(_identifier(item, "Trial") is None for item in trials):
            raise ValueError("Trials must be plain identifiers.")
        _identifier(value.get("agent_name"), "Agent name")
    if operation == "papers_status":
        papers_dir = value.get("papers_dir")
        if not isinstance(papers_dir, str) or not papers_dir.strip() or not Path(papers_dir).expanduser().is_dir():
            raise ValueError("Paper processing review needs an existing papers_dir containing your locally supplied papers.")
    if operation == "judge_panel":
        if not isinstance(value.get("panel_cache"), str) or not Path(value["panel_cache"]).expanduser().is_dir():
            raise ValueError("Judge panel analysis needs an existing panel_cache directory.")
    if operation == "judge_agreement" and not all(isinstance(value.get(name), str) and value[name].strip()
                                                  for name in ("model_a", "model_b")):
        raise ValueError("Judge agreement needs explicit model_a and model_b cache namespaces.")
    for field_name in ("proposal", "observed_context") if operation == "rsi_prepare" else (
            ("prepared", "state") if operation == "rsi_decide" else ()):
        if not isinstance(value.get(field_name), dict):
            raise ValueError(f"{field_name} must be a JSON object.")
    if operation in {"judge", "arbiter", "scout"} and not (value.get("mock") or value.get("dry_run")):
        if not settings.allow_network:
            raise ValueError("Fresh model requests require enabling network explicitly; use mock or dry-run for offline work.")
        if not (settings.api_key and settings.base_url and settings.judge_model):
            raise ValueError("Fresh model requests need an API key, base URL and judge model.")
    return engine, reference, value


def _dataset_agents(reference: Path) -> tuple[str, ...]:
    """Official outcome filenames declare the dataset cohort; logs do not."""
    agents = []
    for path in sorted((reference / "data").glob("eval_*.json")):
        if path.is_file():
            value = path.stem[len("eval_"):]
            _identifier(value, "Dataset agent")
            agents.append(value)
    return tuple(agents)


def _safe_copy(source: Path, target: Path) -> None:
    """Copy selected data, rejecting symlinks so configured roots stay bounded."""
    if source.is_symlink():
        raise ValueError("Symbolic links are not supported in AWE working inputs.")
    if source.is_dir():
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symbolic links are not supported in AWE working inputs.")
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _redact(value: str, api_key: str) -> str:
    if isinstance(api_key, str) and api_key:
        value = value.replace(api_key, "[redacted]")
    value = re.sub("de" + "caf", "AWE", value, flags=re.IGNORECASE)
    return value[-_MAX_LOG:]


def _stop_process_tree(process: subprocess.Popen, *, force: bool = False) -> bool:
    """Terminate the isolated job and its descendants without invoking a shell."""
    if os.name == "nt":
        # A worker can spawn evaluator children. Terminating its PID alone leaves
        # them running on Windows; taskkill's /T handles the whole process tree.
        system_root = os.environ.get("SYSTEMROOT")
        executable = str(Path(system_root) / "System32" / "taskkill.exe") if system_root else "taskkill"
        try:
            result = subprocess.run([executable, "/PID", str(process.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=10, check=False)
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        # Still reap the direct child if the OS utility is unavailable, while
        # reporting that cleanup of the entire tree could not be confirmed.
        try:
            process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass
        return False
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass  # The worker or its last descendant exited during timeout handling.
    return True


def run_operation(operation: str, settings: EvaluationSettings, *, agent: str | None = None,
                  instance_id: str | None = None, limit: int | None = None,
                  payload: dict[str, Any] | None = None) -> OperationResult:
    """Run an allowlisted engine operation in a private process and output tree.

    All operations are offline unless network is explicitly enabled. Fresh judge,
    arbiter and scout calls additionally require complete model configuration.
    Work is never run while importing saved data or applying settings. Engine
    code comes only from the installed TrajViz package.
    """
    job = None
    try:
        engine, reference, options = _validated(operation, settings, agent, instance_id, limit, payload)
        profile = _json_object(settings.benchmark_profile, "Benchmark profile")
        context = _json_object(settings.execution_context, "Execution context")
        base = Path(settings.output_root).expanduser().resolve() if settings.output_root else None
        if base is not None:
            if base in (engine, reference) or base.is_relative_to(engine) or (reference is not None and base.is_relative_to(reference)):
                raise ValueError("Choose an output root outside the packaged engine and reference inputs.")
            base.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix="trajviz-awe-", dir=base))
        if reference is None:
            reference = job / "reference-inputs"
            (reference / "data").mkdir(parents=True)
        agents = (agent,) if agent is not None else tuple(settings.agents)
        if not agents and operation in _REFERENCE_OPERATIONS:
            agents = _dataset_agents(reference)
            if not agents:
                raise ValueError("No agents are declared by data/eval_<agent>.json files; select agents explicitly or provide official outcomes.")
        shadow, output = job / "engine", job / "engine" / "data"
        _safe_copy(engine, shadow)
        output.mkdir(parents=True, exist_ok=True)
        code_files = {str(path.relative_to(shadow)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(shadow.rglob("*")) if path.is_file()}
        engine_digest = hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest()
        inputs = Path(settings.project_data).expanduser().resolve() if settings.project_data else engine / "data"
        if operation == "publication" and not options.get("dossier"):
            if settings.project_data is None:
                raise ValueError("Publication checking needs a saved run directory; provide dossier or project_data.")
            options["publication_root"] = str(inputs)
        for name in ("cache/judge", "cache/arbiter", "cache/scout", "cache/harness", "calibration", "pertest", "overrides",
                     "discovery/swebench_verified_difficulty.json", "remedy_verification.json"):
            _safe_copy(inputs / name, output / name)
        if operation == "papers_status":
            papers = Path(options.get("papers_dir") or engine / "papers").expanduser().resolve()
            _safe_copy(papers, shadow / "papers")
        if "catalog_path" in options:
            catalog = Path(options["catalog_path"]).expanduser().resolve()
            if catalog.stat().st_size > _MAX_INPUT:
                raise ValueError("Catalog JSON is too large.")
            _safe_copy(catalog, job / "catalog.json")
            options["catalog_path"] = str(job / "catalog.json")
        if operation == "showcase":
            if not options.get("showcase_data"):
                raise ValueError("Showcase export needs an explicitly saved showcase_data JSON object or file.")
            showcase = _json_object(options["showcase_data"], "Saved showcase")
            if not {"dataset", "insights", "cases", "countermeasures", "agents_meta"} <= showcase.keys():
                raise ValueError("Saved showcase is missing required dataset, insights, cases, countermeasures or agent metadata.")
            (output / "showcase_data.json").write_text(json.dumps(showcase, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        if operation == "remedy_verify":
            evidence = Path(options.get("evidence_root") or engine).expanduser().resolve()
            if evidence == engine:
                # Only explicit verification references are read by this operation.
                options["evidence_root"] = str(engine)
            elif not evidence.is_dir():
                raise ValueError("Experiment evidence root must be an existing directory.")
        if "python_executable" in options:
            interpreter = Path(options["python_executable"]).expanduser().resolve()
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                raise ValueError("Select an executable Python from the evaluator environment.")
            options["python_executable"] = str(interpreter)
        if operation in {"grader_log_audit", "pertest_harness"}:
            interpreter = options.get("python_executable") or sys.executable
            probe_environment = {name: value for name, value in os.environ.items()
                                 if name in {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}}
            check = subprocess.run([interpreter, "-I", "-c", "import importlib.util;raise SystemExit(0 if importlib.util.find_spec('swebench') else 1)"],
                                   env=probe_environment, capture_output=True, timeout=10, check=False)
            if check.returncode != 0:
                raise ValueError("This operation needs SWE-bench installed in its selected evaluator Python.")
        if options.get("dossier"):
            dossier = Path(options["dossier"]).expanduser().resolve()
            if not dossier.is_dir():
                raise ValueError("Dossier must be an existing saved-result directory.")
            if operation == "publication":
                options["publication_root"] = str(dossier)
            else:
                _safe_copy(dossier, output / "dossier")
            options["dossier"] = str(output / "dossier")
        for name, doc in (("benchmark-profile.json", profile), ("execution-context.json", context)):
            if doc is not None:
                (job / name).write_text(json.dumps(doc, allow_nan=False), encoding="utf-8")
        environment = {name: value for name, value in os.environ.items()
                       if name in {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}}
        environment.update({"HOME": str(job), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                            "AWE_ARGUS_ROOT": str(reference), "AWE_PROJECT_DATA": str(output),
                            "AWE_OUTPUT_DATA": str(output), "AWE_JUDGE_TEMPERATURE": str(settings.temperature),
                            "AWE_JUDGE_MAX_TOKENS": str(settings.max_tokens),
                            "AWE_JUDGE_MIN_AGREE": str(settings.judge_min_agree),
                            "TRAJVIZ_AWE_OFFLINE": "0" if settings.allow_network else "1",
                            "AWE_STRICT_VERDICT_PROVENANCE": "1"})
        if agents:
            environment["AWE_AGENTS"] = ",".join(agents)
        for name, value in (("AWE_JUDGE_MODEL", settings.judge_model), ("AWE_OPENROUTER_BASE_URL", settings.base_url),
                            ("AWE_BENCHMARK_NAME", settings.benchmark_name)):
            if value:
                environment[name] = value
        if profile is not None:
            environment["AWE_BENCHMARK_PROFILE"] = str(job / "benchmark-profile.json")
        if context is not None:
            environment["AWE_EXECUTION_CONTEXT"] = str(job / "execution-context.json")
        if (settings.allow_network and settings.api_key and operation in {"judge", "arbiter", "scout"}
                and not options.get("mock") and not options.get("dry_run")):
            environment["OPENROUTER_API_KEY"] = settings.api_key
        if operation == "pertest_harness":
            for key, env_name in (("dataset", "AWE_SWEBENCH_DATASET"), ("image_namespace", "AWE_IMAGE_NAMESPACE"),
                                  ("image_platform", "AWE_IMAGE_PLATFORM")):
                if key in options:
                    if not isinstance(options[key], str) or not options[key].strip():
                        raise ValueError(f"{key} must be text.")
                    environment[env_name] = options[key]
        package_root = str(Path(__file__).resolve().parents[2])
        request = {"operation": operation, "engine": str(shadow), "reference": str(reference),
                   "output": str(output), "job": str(job), "agent": agent, "instance_id": instance_id,
                   "limit": limit, "payload": options, "package_root": package_root}
        encoded = json.dumps(request, allow_nan=False)
        if len(encoded.encode()) > _MAX_INPUT:
            raise ValueError("Operation payload is too large.")
        command = [sys.executable, "-I", "-c",
                   "import sys;sys.path.insert(0,sys.argv[1]);from agent_trajviz.awe._worker import main;main()",
                   package_root]
        log_path = job / "operation.log"
        with log_path.open("w+b") as stream:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stream, stderr=stream,
                                       cwd=job, env=environment, start_new_session=os.name != "nt")
            try:
                process.communicate(encoded.encode(), timeout=settings.timeout)
            except subprocess.TimeoutExpired:
                tree_stopped = _stop_process_tree(process)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    tree_stopped = _stop_process_tree(process, force=True)
                    process.wait()
                if os.name != "nt":
                    # The parent may exit on SIGTERM while a child ignores it.
                    # Kill any surviving group members even after reaping it.
                    tree_stopped = _stop_process_tree(process, force=True)
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - _MAX_LOG))
                logs = _redact(stream.read().decode(errors="replace"), settings.api_key)
                log_path.write_text(logs, encoding="utf-8")
                reason = ("Evaluation operation timed out; its process tree was stopped." if tree_stopped
                          else "Evaluation operation timed out; descendant termination could not be confirmed.")
                return OperationResult(False, True, reason,
                                       operation, str(output), str(output), logs=logs)
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - _MAX_LOG))
            logs = _redact(stream.read().decode(errors="replace"), settings.api_key)
        log_path.write_text(logs, encoding="utf-8")
        (output / "operation.log").write_text(logs, encoding="utf-8")
        result_path = output / "operation-result.json"
        data = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        selected_output = output
        if isinstance(data, dict) and isinstance(data.get("saved_run_dir"), str):
            candidate = Path(data["saved_run_dir"]).resolve()
            if candidate.is_relative_to(output) and candidate.is_dir():
                selected_output = candidate
                shutil.copyfile(output / "operation.log", selected_output / "operation.log")
                shutil.copyfile(result_path, selected_output / "operation-result.json")
        artifacts = tuple(str(path.relative_to(selected_output)) for path in sorted(selected_output.rglob("*"))
                          if path.is_file() and "cache" not in path.relative_to(selected_output).parts)
        fingerprints = {name: hashlib.sha256((selected_output / name).read_bytes()).hexdigest() for name in artifacts}
        manifest = {"schema_version": 1, "operation": operation, "success": process.returncode == 0,
                    "engine_source_sha256": engine_digest,
                    "native_adapter_sha256": hashlib.sha256(Path(__file__).with_name("native.py").read_bytes()).hexdigest()
                    if Path(__file__).with_name("native.py").is_file() else None,
                    "artifact_sha256": fingerprints, "network_enabled": settings.allow_network,
                    "agent": agent, "instance_id": instance_id, "limit": limit,
                    "model": settings.judge_model or None, "api_key_stored": False}
        (selected_output / "operation-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return OperationResult(process.returncode == 0, True,
                               None if process.returncode == 0 else (data.get("reason")
                               if isinstance(data, dict) and data.get("available") is False and isinstance(data.get("reason"), str)
                               else "AWE operation failed; inspect its validation report or sanitized log."),
                               operation, str(selected_output), str(output), data,
                               (*artifacts, "operation-manifest.json"), logs)
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        # Validation messages are deliberately written here, not forwarded from a engine.
        reason = str(exc) if isinstance(exc, ValueError) else f"AWE operation could not start ({type(exc).__name__})."
        if job is not None:
            shutil.rmtree(job, ignore_errors=True)
        return OperationResult(False, False, _redact(reason, settings.api_key), operation)
