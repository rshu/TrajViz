"""Private subprocess entry for optional AWE execution; not a public API."""
from __future__ import annotations

import hashlib
import dataclasses
import importlib.util
import json
import os
import runpy
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .runtime import _OPERATIONS


def _no_network(*_args, **_kwargs):
    raise PermissionError("Network is disabled for this AWE operation.")


def _block_network() -> None:
    if os.environ.get("TRAJVIZ_AWE_OFFLINE") == "1":
        socket.socket.connect = _no_network
        socket.socket.connect_ex = _no_network
        socket.create_connection = _no_network


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load_engine(path: str) -> None:
    """Resolve canonical internal imports to this job's private bundled copy."""
    import agent_trajviz.awe as namespace

    name = "agent_trajviz.awe._engine"
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(name, Path(path) / "__init__.py",
                                                  submodule_search_locations=[path])
    if spec is None or spec.loader is None:
        raise ImportError("Bundled evaluation engine is unavailable.")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[name] = engine
    namespace._engine = engine
    spec.loader.exec_module(engine)


def _hooks(request: dict[str, Any]) -> None:
    """Propagate the same native adapters/network policy to script descendants."""
    hook = Path(request["job"]) / "python-hooks"
    hook.mkdir()
    source = (
        "import os,socket,sys\n"
        f"sys.path.insert(0,{request['package_root']!r})\n"
        "if os.environ.get('TRAJVIZ_AWE_OFFLINE')=='1':\n"
        " def blocked(*args,**kwargs):\n"
        "  raise PermissionError('Network is disabled for this AWE operation.')\n"
        " socket.socket.connect=blocked\n"
        " socket.socket.connect_ex=blocked\n"
        " socket.create_connection=blocked\n"
        "from agent_trajviz.awe._worker import _load_engine\n"
        f"_load_engine({request['engine']!r})\n"
        "from agent_trajviz.awe.native import install_native_adapters\n"
        "install_native_adapters()\n"
    )
    (hook / "sitecustomize.py").write_text(source, encoding="utf-8")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(hook), request["engine"], request["package_root"]])
    _block_network()
    _load_engine(request["engine"])
    from agent_trajviz.awe.native import install_native_adapters

    install_native_adapters()


def _script(request: dict[str, Any], script: str, args: list[str]) -> int:
    interpreter = request["payload"].get("python_executable") or sys.executable
    command = [interpreter, str(Path(request["engine"]) / "scripts" / script), *args]
    completed = subprocess.run(command, cwd=request["engine"], check=False)
    return completed.returncode


def _selection(request: dict[str, Any], *, workers: bool = False) -> list[str]:
    args = []
    if request["agent"]:
        args += ["--agent", request["agent"]]
    if request["limit"] is not None:
        args += ["--limit", str(request["limit"])]
    if workers:
        args += ["--workers", str(request["payload"].get("workers", 1))]
    return args


def _diagnose(request: dict[str, Any]) -> dict[str, Any]:
    from agent_trajviz.awe._engine import adapters, dossier
    from agent_trajviz.awe._engine import config
    from agent_trajviz.awe._engine.symptom_exports import case_record as symptom_case
    from agent_trajviz.awe._engine.symptom_exports import write as write_symptoms

    agent, instance = request["agent"], request["instance_id"]
    base = config.TRAJECTORY_DIR / agent
    json_path, jsonl_path = base / (instance + ".json"), base / (instance + ".jsonl")
    canonical = jsonl_path if agent == "codex" or (jsonl_path.exists() and not json_path.exists()) else json_path
    if not canonical.is_file():
        return {"available": False, "reason": "Canonical trajectory is missing.", "agent": agent, "instance_id": instance}
    inputs = [canonical, config.requirements_path(instance), config.patch_path(agent, instance), config.eval_path(agent)]
    def stamps():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in inputs}
    from agent_trajviz.awe._engine.provenance import trajectory_fingerprint
    before = stamps()
    trajectory_stamp = trajectory_fingerprint(agent, instance)
    trajectory = adapters.load_normalized(agent, instance)
    if trajectory is None or not trajectory.steps:
        return {"available": False, "reason": "The backend could not extract workflow steps from this trajectory.",
                "agent": agent, "instance_id": instance}
    original_detect = dossier.detect
    original_arbiter = getattr(dossier, "auto_adjudicate", None)
    detection = {}
    def controlled_detect(*args, **kwargs):
        kwargs["use_judge"] = request["payload"].get("use_judge", True)
        result = original_detect(*args, **kwargs)
        detection.update(result)
        return result
    dossier.detect = controlled_detect
    if original_arbiter is not None and not request["payload"].get("use_arbiter", True):
        dossier.auto_adjudicate = lambda _agent: None
    try:
        case = dossier.case_record(agent, instance)
    finally:
        dossier.detect = original_detect
        if original_arbiter is not None:
            dossier.auto_adjudicate = original_arbiter
    if stamps() != before or trajectory_fingerprint(agent, instance) != trajectory_stamp:
        raise ValueError("Reference inputs changed during diagnosis; reload and retry.")
    output = Path(request["output"]) / "dossier"
    output.mkdir(parents=True, exist_ok=True)
    symptom = symptom_case(agent, instance)
    write_symptoms([symptom], output)
    _write(output / "faults.json", [] if case is None else [case])
    if case is None:
        return {"available": False, "reason": "This case has no failed-run attribution; its observations remain available.",
                "agent": agent, "instance_id": instance, "symptoms": symptom}
    from agent_trajviz.awe._engine.attribution_graph import build_graphs, write as write_graphs
    from agent_trajviz.awe._engine.remedy_plan import build_plan, write as write_plan
    from agent_trajviz.awe._engine.rsi import write_exchange
    plan = build_plan([case], agents=[agent], instances=[instance], benchmark=config.BENCHMARK_NAME)
    write_plan(plan, output)
    write_exchange([case], plan, output)
    write_graphs(build_graphs([case], agents=[agent], instances=[instance], benchmark=config.BENCHMARK_NAME), output)
    return {"available": True, "case": case, "symptoms": symptom,
            "used_judge": (detection.get("signals") or {}).get("judge_available") is True,
            "input_sha256": before, "trajectory_fingerprint": trajectory_stamp,
            "trajectory_source": trajectory.source,
            "judge_policy": "verified_cache_only", "fresh_model_requests": 0}


def _catalog(request: dict[str, Any]) -> dict[str, Any]:
    from agent_trajviz.awe._engine import countermeasures
    from agent_trajviz.awe._engine.remedy_verification import validate_verification
    from agent_trajviz.awe._engine.validate import countermeasure_catalog
    options = request["payload"]
    path = Path(options["catalog_path"]) if options.get("catalog_path") else None
    entries = countermeasures.load_catalog(path)
    if request["operation"] == "remedy_verify":
        entry = next((row for row in entries if row.id == options.get("recipe_id")), None)
        if entry is None:
            raise ValueError("Unknown remedy recipe.")
        result = {"recipe_id": entry.id, "declared_status": entry.status,
                  "validation": validate_verification(entry, root=Path(options["evidence_root"]))}
    else:
        result = {"entries": [dataclasses.asdict(row) for row in entries],
                  "bundles": countermeasures.bundles(path), "methodology": countermeasures.methodology(path),
                  "operational_annex": countermeasures.operational_annex(path),
                  "benchmark_artifact_annex": countermeasures.benchmark_artifact_annex(path),
                  "quality_checks": countermeasure_catalog() if path is None else None,
                  "scope": "Recipe specifications and scoped evidence; catalog presence does not establish efficacy."}
    _write(Path(request["output"]) / (request["operation"] + ".json"), result)
    return result


def _guardrail(request: dict[str, Any]) -> dict[str, Any]:
    from agent_trajviz.awe._engine import remedy_runtime as runtime
    options = request["payload"]
    if request["operation"] == "guardrail_decide":
        controller = runtime.GuardrailController(options["run_id"], global_budget=options.get("global_budget", 6),
                                                 snapshot=options.get("snapshot"))
        decision = controller.decide(options["recipe_id"], options["event_id"], options["observed"])
        result = {"decision": decision, "snapshot": controller.snapshot(), "advisory_only": True,
                  "installed_hooks": False, "state_custody": "Persist and protect the returned snapshot in the native controller."}
    else:
        functions = {"symbol_overlap": runtime.symbol_overlap, "test_status": runtime.test_status,
                     "verification_gap": runtime.verification_gap, "repair_stagnation": runtime.repair_stagnation,
                     "self_revert": runtime.self_revert}
        kind = options.get("kind")
        if kind not in functions or not isinstance(options.get("values"), dict):
            raise ValueError("Unknown public observation helper or invalid values.")
        result = {"kind": kind, "observation": functions[kind](**options["values"])}
    _write(Path(request["output"]) / (request["operation"] + ".json"), result)
    return result


def _discovery(request: dict[str, Any]) -> dict[str, Any]:
    from agent_trajviz.awe._engine import config
    selected = [request["agent"]] if request["agent"] else list(config.AGENTS)
    instances = config.require_instances(config.list_instances())
    if request["limit"] is not None:
        instances = instances[:request["limit"]]
    output = Path(request["output"]) / "discovery"
    output.mkdir(parents=True, exist_ok=True)
    if request["operation"] == "cohort":
        from agent_trajviz.awe._engine.cohort import cohort_report_md, discover_agent, export_discovery
        rows = [discover_agent(agent, instances) for agent in selected]
        markdown = "\n\n".join(cohort_report_md(row) for row in rows)
        rows = [export_discovery(row) for row in rows]
        name = "cohort_contrast"
    elif request["operation"] == "behaviors":
        from agent_trajviz.awe._engine.behaviors import behaviors_report_md, behaviors_summary
        rows = [behaviors_summary(agent, instances) for agent in selected]
        markdown, name = behaviors_report_md(rows), "discovered_behaviors"
    else:
        from agent_trajviz.awe._engine.fragile import report_md, scan
        from agent_trajviz.awe._engine.outcomes import resolved
        skipped = []
        rows = [row for agent in selected for row in scan(agent, instances, skipped=skipped)]
        count = sum(resolved(agent, instance) is True for agent in selected for instance in instances)
        markdown, name = report_md(rows, count, len(skipped)), "fragile_solves"
    (output / (name + ".md")).write_text(markdown, encoding="utf-8")
    _write(output / (name + ".json"), rows)
    return {"report": name, "agents": selected, "instances": instances,
            "scope": "observational review; associations do not establish causal effects"}


def _publication(request: dict[str, Any]) -> dict[str, Any]:
    from agent_trajviz.awe.saved import load_run

    source = Path(request["payload"]["publication_root"]).resolve()
    run_source = source
    if (source / "data" / "dossier" / "faults.json").is_file():
        run_source = source / "data" / "dossier"
    run = load_run(run_source)
    checked_hashes = sum(len(value.get("artifact_sha256", {})) for name, value in run.artifacts.items()
                         if name.endswith("operation-manifest.json") and isinstance(value, dict))
    publication_manifest = source / "data" / "publication.json"
    if not publication_manifest.is_file():
        publication_manifest = source / "publication.json"
    if publication_manifest.is_file():
        doc = json.loads(publication_manifest.read_text(encoding="utf-8"))
        base = publication_manifest.parent.parent if publication_manifest.parent.name == "data" else source
        for name, expected in doc.get("artifact_sha256", {}).items():
            candidate = (base / name).resolve()
            if not candidate.is_relative_to(base) or ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError("Publication artifact path escapes the selected source tree.")
            if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                raise ValueError("A recorded publication artifact is missing or has changed.")
            checked_hashes += 1
    report = {"passed": True, "scope": "Provided saved artifacts only; no reruns or efficacy claims.",
              "verification": "recorded_hashes_and_structure" if checked_hashes else "structure_only",
              "recorded_hashes_checked": checked_hashes, "artifacts": len(run.artifacts),
              "cases": len(run.cases), "faults": sum(len(case.get("faults", [])) for case in run.cases),
              "graphs": len((run.graphs or {}).get("graphs", [])), "warnings": list(run.warnings)}
    _write(Path(request["output"]) / "publication-validation.json", report)
    return report


def _models(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Record actual verdict coverage; upstream CLI exit zero can hide failures."""
    operation, options = request["operation"], request["payload"]
    path = Path(request["engine"]) / "scripts" / _OPERATIONS[operation][2]
    namespace = runpy.run_path(str(path))
    target_name = "target_instances" if operation == "judge" else "targets"
    if target_name not in namespace:
        return 1, {"available": False, "reason": "The installed model runner lacks structured target support; reinstall TrajViz."}
    from agent_trajviz.awe._engine import config
    agents = [request["agent"]] if request["agent"] else list(config.AGENTS)
    mock, model = options.get("mock", False), ("mock" if options.get("mock") else config.JUDGE_MODEL)
    calls = []
    for agent in agents:
        selection_limit = None if request["instance_id"] else request["limit"]
        targets = (namespace[target_name](agent, options.get("include_resolved", False), selection_limit)
                   if operation == "judge" else namespace[target_name](agent, selection_limit))
        if request["instance_id"]:
            targets = [target for target in targets
                       if (target if isinstance(target, str) else target[0]) == request["instance_id"]]
            if request["limit"] is not None:
                targets = targets[:request["limit"]]
        for target in targets:
            calls.append((agent, target))
    if request["instance_id"] and not calls:
        return 1, {"available": False, "reason": "The selected case has no eligible targets for this operation.", "targets": 0}
    if options.get("dry_run"):
        estimated, unknown = 0, 0
        for agent, target in calls:
            if operation == "judge":
                tokens = namespace["estimate_tokens"](agent, target)
            else:
                instance = target if isinstance(target, str) else target[0]
                gold = namespace["build_gold_reference"](instance)
                trajectory = namespace["load_normalized"](agent, instance)
                if trajectory is None or not trajectory.steps:
                    tokens = None
                else:
                    args = (gold, trajectory, *target[1]) if operation == "arbiter" else (gold, trajectory)
                    tokens = sum(len(message["content"]) for message in namespace["build_messages"](*args)) // 4
            if tokens is None:
                unknown += 1
            else:
                estimated += tokens
        panel = options.get("panel", 1)
        result = {"mode": "dry_run", "model_requests": 0, "effective_model": model, "targets": len(calls),
                  "estimated_input_tokens": estimated * panel, "targets_without_estimate": unknown,
                  "output_token_budget_per_pass": config.JUDGE_MAX_TOKENS * len(calls) * panel,
                  "retry_budget_included": False, "estimate_scope": "Approximate character-based input estimate, not measured usage or cost."}
        _write(Path(request["output"]) / (operation + "-coverage.json"), result)
        return 0, result
    from agent_trajviz.awe._engine import arbiter, judge, scout
    def work(item):
        agent, target = item
        common = {"model": model, "force": options.get("force", False)}
        if operation == "judge":
            return judge.judge_one(agent, target, client=judge.mock_client if mock else None,
                                   samples=options.get("panel", 1), temperature=config.JUDGE_TEMPERATURE, **common)
        if operation == "arbiter":
            instance, (capability, error) = target
            return arbiter.verify_one(agent, instance, capability, error,
                                       client=arbiter.mock_client if mock else None,
                                       samples=options.get("panel", 1), temperature=config.JUDGE_TEMPERATURE, **common)
        return scout.scout_one(agent, target, client=scout.mock_client if mock else None, **common)
    summary = {"mode": "mock" if mock else "configured_provider", "effective_model": model,
               "targets": len(calls), "verdicts": 0, "unavailable": 0, "invalid_responses": 0, "errors": []}
    with ThreadPoolExecutor(max_workers=options.get("workers", 1)) as pool:
        pending = {pool.submit(work, item): item for item in calls}
        for future in as_completed(pending):
            agent, target = pending[future]
            try:
                record = future.result()
                if record is None:
                    summary["unavailable"] += 1
                elif record.get("parse_failed") or record.get("finish_reason") == "length":
                    summary["invalid_responses"] += 1
                else:
                    summary["verdicts"] += 1
            except Exception as exc:
                error = {"agent": agent, "instance": target if isinstance(target, str) else target[0],
                         "error_type": type(exc).__name__}
                if type(getattr(exc, "status_code", None)) is int:
                    error["status_code"] = exc.status_code
                summary["errors"].append(error)
    if operation == "scout":
        namespace["write_report"](agents, model, consolidate=options.get("consolidate", False), mock=mock)
    summary["coverage_note"] = "Unavailable or invalid responses do not establish an assessed capability."
    _write(Path(request["output"]) / (operation + "-coverage.json"), summary)
    failed = bool(summary["errors"] or summary["invalid_responses"])
    return int(failed), summary


def execute(request: dict[str, Any]) -> tuple[int, Any]:
    operation, options = request["operation"], request["payload"]
    output = Path(request["output"])
    if operation == "diagnose":
        return 0, _diagnose(request)
    if operation in {"cohort", "behaviors", "fragile"}:
        return 0, _discovery(request)
    if operation in {"rsi_prepare", "rsi_decide"}:
        from agent_trajviz.awe._engine.rsi import decide, prepare
        if operation == "rsi_prepare":
            result = prepare(options["proposal"], options["observed_context"])
        else:
            result = decide(options["prepared"], options["state"], trigger_count=options.get("trigger_count", 0))
        _write(output / (operation + ".json"), result)
        return 0, result
    if operation in {"catalog", "remedy_verify"}:
        return 0, _catalog(request)
    if operation in {"guardrail_decide", "guardrail_observe"}:
        return 0, _guardrail(request)
    if operation in {"judge", "arbiter", "scout"}:
        return _models(request)
    if operation == "publication":
        return 0, _publication(request)
    if operation in {"calibration", "threshold_stability"}:
        from agent_trajviz.awe._engine import config
        from agent_trajviz.awe._engine.outcomes import resolved
        outcomes = [resolved(agent, instance) for agent in config.AGENTS for instance in config.list_instances()]
        if not any(value is True for value in outcomes) or not any(value is False for value in outcomes):
            return 1, {"available": False, "reason": "Calibration needs both solved and failed cases in the configured benchmark.",
                       "solved": sum(value is True for value in outcomes), "failed": sum(value is False for value in outcomes)}
    if operation == "agreement" and not (output / "calibration" / "detector_predictions.json").is_file():
        return 1, {"available": False, "reason": "Generate a calibration set first, then add independent human annotations or select mock scoring."}
    if operation == "agreement" and not options.get("mock") and not list((output / "calibration" / "annotations").glob("*.json")):
        return 1, {"available": False, "reason": "Independent human annotations are required for agreement scoring; mock mode only checks the scorer."}
    if operation == "difficulty_strata" and not (output / "discovery" / "swebench_verified_difficulty.json").is_file():
        return 1, {"available": False, "reason": "Difficulty-stratified calibration needs recorded SWE-bench difficulty labels in project data."}
    if operation == "preflight":
        from agent_trajviz.awe._engine import config
        from agent_trajviz.awe._engine.preflight import validate_inputs
        agents = [request["agent"]] if request["agent"] else list(config.AGENTS)
        instances = config.list_instances()
        if request["limit"] is not None:
            instances = instances[:request["limit"]]
        report = validate_inputs(agents, instances)
        _write(output / "preflight.json", report)
        return int(bool(report["errors"])), report
    if operation in {"pipeline", "symptoms"}:
        args = _selection(request, workers=True)
    elif operation in {"judge", "arbiter", "scout"}:
        args = _selection(request, workers=True)
        for key in ("mock", "dry_run", "force", "include_resolved", "consolidate"):
            if options.get(key):
                args += ["--" + key.replace("_", "-")]
        if "panel" in options:
            args += ["--panel", str(options["panel"])]
    elif operation in {"graphs", "recommendations", "remedies", "rsi"}:
        args = ["--dossier", str(output / "dossier")]
        if operation == "graphs" and os.environ.get("AWE_BENCHMARK_NAME"):
            args += ["--benchmark", os.environ["AWE_BENCHMARK_NAME"]]
    elif operation == "preprocessing":
        args = ["--output", str(output / "preprocessing.json")]
        if request["agent"]:
            args += ["--agent", request["agent"], "--instance", request["instance_id"]]
    elif operation == "fault_injection":
        args = ["--language", options.get("language", "all"), "--timeout", str(options.get("case_timeout", 10)),
                "--output", str(output / "fault-injection.json")]
    elif operation == "ingest_harbor":
        args = ["--harbor-root", str(Path(options["harbor_root"]).expanduser().resolve()),
                "--output-root", str(output / "canonical-inputs")]
        if options.get("all_run_groups"):
            args += ["--all-run-groups"]
        for key, flag in (("run_groups", "run-group"), ("trials", "trial")):
            for value in options.get(key, []):
                args += ["--" + flag, value]
        for key in ("status_dir", "agent_name"):
            if options.get(key):
                args += ["--" + key.replace("_", "-"), options[key]]
        for key in ("include_exception_failures", "namespace_instances"):
            if options.get(key):
                args += ["--" + key.replace("_", "-")]
        if request["limit"] is not None:
            args += ["--limit", str(request["limit"])]
    elif operation == "pertest_harness":
        args = ["--agent", request["agent"], "--work", str(output / "cache" / "harness_work"),
                "--timeout", str(options.get("case_timeout", 1800))]
        if request["instance_id"]:
            args += ["--instance", request["instance_id"]]
        if request["limit"] is not None:
            args += ["--limit", str(request["limit"])]
        for key in ("gold", "source_only", "keep_logs", "redo"):
            if options.get(key):
                args += ["--" + key.replace("_", "-")]
    elif operation in {"pertest_report", "grader_log_audit"}:
        args = ["--agent", request["agent"]] if request["agent"] else []
    elif operation == "agreement":
        args = ["--mock"] if options.get("mock") else []
    elif operation == "judge_agreement":
        args = []
        for key in ("model_a", "model_b"):
            if options.get(key):
                args += ["--" + key.replace("_", "-"), options[key]]
        args += ["--write"]
    elif operation == "judge_panel":
        args = [options["panel_cache"]]
        if options.get("baseline_cache"):
            args += [options["baseline_cache"]]
    elif operation == "calibration_set":
        args = [str(request["limit"] or 60)]
    else:
        args = []
    code = _script(request, _OPERATIONS[operation][2], args)
    result = {"returncode": code, "offline": os.environ.get("TRAJVIZ_AWE_OFFLINE") == "1"}
    if operation == "pipeline" and code == 0:
        archives = [path.parent for path in output.rglob("manifest.json") if path.parent.parent.name == "runs"]
        if len(archives) == 1:
            result["saved_run_dir"] = str(archives[0])
    if operation == "ingest_harbor" and code == 0:
        canonical_root = output / "canonical-inputs"
        ingested = json.loads((canonical_root / "data" / "harbor_ingest_manifest.json").read_text(encoding="utf-8"))
        result.update(canonical_root=str(canonical_root), agents=ingested["agents"],
                      ingestion=ingested, benchmark_name=os.environ.get("AWE_BENCHMARK_NAME") or "unspecified")
    return code, result


def main() -> None:
    request = json.loads(sys.stdin.read())
    output = Path(request["output"])
    try:
        _hooks(request)
        code, result = execute(request)
        _write(output / "operation-result.json", result)
    except Exception as exc:
        # Do not forward raw exception values: provider errors may contain secrets.
        print(f"AWE operation failed ({type(exc).__name__}).", file=sys.stderr)
        _write(output / "operation-result.json", {"error_type": type(exc).__name__})
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
