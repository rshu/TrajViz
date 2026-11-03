"""Saved AWE runs and explicit, session-configured backend operations."""
from __future__ import annotations

import atexit
import html
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import gradio as gr

from ...awe import render

_PREF_FIELDS = ("reference_root", "project_data", "output_root", "agents", "benchmark_name",
                "benchmark_profile", "execution_context", "judge_model", "base_url", "temperature", "max_tokens", "timeout", "judge_min_agree")
_DEFAULTS = {"reference_root": "", "project_data": "", "output_root": "", "agents": "",
             "benchmark_name": "", "benchmark_profile": "", "execution_context": "", "judge_model": "",
             "base_url": "", "temperature": 0.0, "max_tokens": 2400, "timeout": 300, "judge_min_agree": 0.0}


@dataclass
class AWERefs:
    run: gr.State
    settings: gr.State
    export_state: gr.State
    file: gr.File
    directory: gr.Textbox
    import_button: gr.Button
    status: gr.HTML
    agent: gr.Dropdown
    outcome: gr.Dropdown
    query: gr.Textbox
    case: gr.Dropdown
    case_html: gr.HTML
    graph_kind: gr.Dropdown
    node: gr.Dropdown
    graph_html: gr.HTML
    node_html: gr.HTML
    symptoms_html: gr.HTML
    recommendations_html: gr.HTML
    report: gr.Dropdown
    report_html: gr.HTML
    provenance_html: gr.HTML
    rsi_html: gr.HTML
    export_button: gr.Button
    export_file: gr.File
    runtime: dict[str, Any]


def _operation_choices() -> list[tuple[str, str]]:
    from ...awe.runtime import available_operations

    return [(f"{row.get('group', 'AWE')} · {row['label']}", row["id"]) for row in available_operations()]


def layout(*, allow_server_paths: bool = True) -> AWERefs:
    with gr.Tab("AWE"):
        gr.Markdown("Import a saved AWE run to inspect diagnoses, provenance graphs, symptoms and experimental recommendations. "
                    "Viewing a saved run sends no model requests.")
        run, settings, export_state = gr.State(None), gr.State(None), gr.State(None)
        with gr.Row():
            file = gr.File(label="Saved AWE run (.zip / .json)", file_types=[".zip", ".json"], type="filepath", scale=3, height=110)
            directory = gr.Textbox(label="Saved run folder", visible=allow_server_paths,
                                   interactive=allow_server_paths, scale=3)
        import_button = gr.Button("Import saved run", variant="primary", scale=0)
        status = gr.HTML(render.notice("Choose a saved run, faults.json, manifest.json or an artifact bundle."))
        with gr.Row():
            agent = gr.Dropdown(label="AWE agent", choices=["All agents"], value="All agents", scale=1)
            outcome = gr.Dropdown(label="Outcome", choices=["All outcomes", "solved", "failed", "unknown"],
                                  value="All outcomes", scale=1)
            query = gr.Textbox(label="Find a case", placeholder="Agent or task ID", scale=2)
        case = gr.Dropdown(label="AWE case", choices=[], value=None)
        with gr.Tabs():
            with gr.Tab("Case evidence"):
                case_html = gr.HTML(render.notice("Import a saved run to begin."))
            with gr.Tab("Provenance graph"):
                gr.Markdown("Select a numbered node or follow its card to the original recorded details. "
                            "Action/result episodes preserve original thread and call references.")
                with gr.Row():
                    graph_kind = gr.Dropdown(label="Graph", choices=["Attribution", "Action/result episodes"], value="Attribution")
                    node = gr.Dropdown(label="Graph node", choices=[], value=None, scale=3)
                graph_html = gr.HTML("")
                node_html = gr.HTML("")
            with gr.Tab("Symptoms"):
                symptoms_html = gr.HTML("")
            with gr.Tab("Recommendations"):
                recommendations_html = gr.HTML("")
            with gr.Tab("Reports & discovery"):
                report = gr.Dropdown(label="Saved report or artifact", choices=[], value=None)
                report_html = gr.HTML("")
            with gr.Tab("Provenance"):
                provenance_html = gr.HTML("")
            with gr.Tab("RSI exchange"):
                rsi_html = gr.HTML("")
        with gr.Row():
            export_button = gr.Button("Prepare run download", scale=0)
            export_file = gr.File(label="Preserved AWE run", interactive=False, type="filepath", height=70)
        runtime: dict[str, Any] = {}
        with gr.Accordion("Evaluation settings and operations", open=False, visible=allow_server_paths):
            gr.Markdown("Configure this session's benchmark data, outputs and model connection. Operations run only when you click Run operation. "
                        "API keys stay in this session. Importing, applying settings and inspecting evidence send no model request.")
            for name, label in (("reference_root", "Benchmark data root"),
                                ("project_data", "Project data / caches"), ("output_root", "Evaluation output folder")):
                runtime[name] = gr.Textbox(label=label, value="")
            with gr.Row():
                runtime["agents"] = gr.Textbox(label="AWE agents", placeholder="Comma-separated agent IDs")
                runtime["benchmark_name"] = gr.Textbox(label="Benchmark name")
            with gr.Row():
                runtime["benchmark_profile"] = gr.Textbox(label="Benchmark profile", placeholder="JSON object or local JSON file", lines=3)
                runtime["execution_context"] = gr.Textbox(label="Execution context", placeholder="JSON object or local JSON file", lines=3)
            with gr.Row():
                runtime["base_url"] = gr.Textbox(label="AWE API base URL", placeholder="OpenAI-compatible chat endpoint")
                runtime["judge_model"] = gr.Textbox(label="AWE judge model", placeholder="Model ID or cached-verdict namespace")
            runtime["api_key"] = gr.Textbox(label="AWE API key", type="password", value="",
                                            html_attributes={"autocomplete": "off"}, info="Blank retains the key for the same endpoint in this session.")
            runtime["clear_key"] = gr.Checkbox(label="Clear AWE API key", value=False)
            with gr.Row():
                runtime["temperature"] = gr.Number(label="AWE temperature", value=0.0)
                runtime["max_tokens"] = gr.Number(label="AWE max output tokens", value=2400, precision=0)
                runtime["timeout"] = gr.Number(label="AWE timeout (seconds)", value=300, precision=0)
                runtime["judge_min_agree"] = gr.Number(label="AWE minimum panel agreement", value=0.0, minimum=0, maximum=1,
                                                       info="Fraction from 0 to 1. Higher thresholds allow the judge panel to abstain on contested verdicts.")
            runtime["remember"] = gr.Checkbox(label="Remember AWE connection details in this browser", value=True)
            runtime["browser"] = gr.BrowserState({}, storage_key="trajviz-awe-settings-v1", secret="trajviz-awe-nonsecret-preferences-v1")
            with gr.Row():
                runtime["apply"] = gr.Button("Apply AWE settings", variant="secondary", scale=0)
                runtime["reset"] = gr.Button("Clear AWE settings", scale=0)
            runtime["settings_status"] = gr.Markdown("No evaluation session settings applied.")
            with gr.Accordion("Evaluation capabilities", open=False):
                runtime["capabilities"] = gr.HTML("")
            with gr.Row():
                choices = _operation_choices()
                runtime["operation"] = gr.Dropdown(label="AWE operation", choices=choices, value="preflight")
                runtime["agent"] = gr.Textbox(label="Operation agent", placeholder="Optional single agent")
                runtime["instance_id"] = gr.Textbox(label="Operation task ID", placeholder="Optional single task")
                runtime["limit"] = gr.Textbox(label="Operation task limit", value="", placeholder="Optional positive integer",
                                              info="Leave blank to use the operation’s default scope.")
            runtime["operation_help"] = gr.HTML(_operation_help(None, "preflight"))
            runtime["payload"] = gr.Code(label="Operation options (JSON)", value="{}", language="json", lines=6)
            gr.Markdown("Use operation options for supported worker, mock/dry-run, dossier and RSI request fields. "
                        "RSI prepare takes `proposal` and `observed_context`; RSI decide takes `prepared`, `state` and `trigger_count`.")
            runtime["allow_network"] = gr.Checkbox(label="Enable network and requested external execution for this operation", value=False,
                                                    info="Fresh model calls require this setting. Benchmark harness execution additionally requires confirm_execution in operation options.")
            runtime["run"] = gr.Button("Run AWE operation", variant="primary", scale=0)
            runtime["result"] = gr.JSON(label="Operation result", value=None)
            runtime["logs"] = gr.Textbox(label="Operation log", lines=7, interactive=False)
            runtime["import_result"] = gr.Button("Load operation output into viewer", scale=0)
            runtime["result_path"] = gr.State(None)
    return AWERefs(run, settings, export_state, file, directory, import_button, status, agent, outcome, query, case,
                   case_html, graph_kind, node, graph_html, node_html, symptoms_html, recommendations_html, report,
                   report_html, provenance_html, rsi_html, export_button, export_file, runtime)


def _status(text: str) -> str:
    return "<p class='awe-status'>" + html.escape(text) + "</p>"


def _case_outputs(run: Any, key: str | None, kind: str = "Attribution") -> tuple[Any, ...]:
    if run is None:
        return ("", gr.Dropdown(choices=[], value=None), "", "", "", "")
    case = render.selected_case(run, key)
    graph = render.case_graph(run, case, kind)
    options = render.node_options(graph)
    selected = options[0][1] if options else None
    return (render.case_html(case), gr.Dropdown(choices=options, value=selected), render.graph_html(graph, selected, run=run),
            render.node_html(graph, selected, run=run), render.symptoms_html(run, case), render.recommendations_html(run, case))


def _load_outputs(run: Any, message: str) -> tuple[Any, ...]:
    if run is None:
        return (None, _status(message), gr.Dropdown(choices=["All agents"], value="All agents"),
                "All outcomes", "", gr.Dropdown(choices=[], value=None),
                *_case_outputs(None, None), gr.Dropdown(choices=[], value=None), "", "", "", None)
    agents = sorted({str(c.get("agent")) for c in render.all_cases(run) if c.get("agent")})
    options = render.case_options(run)
    selected = options[0][1] if options else None
    reports = render.report_options(run)
    report = next((name for name in reports if name.endswith("INSIGHTS.md")),
                  next((name for name in reports if name.lower().endswith((".md", ".txt"))), None))
    return (run, _status(message), gr.Dropdown(choices=["All agents", *agents], value="All agents"),
            "All outcomes", "", gr.Dropdown(choices=options, value=selected), *_case_outputs(run, selected),
            gr.Dropdown(choices=reports, value=report), render.report_html(run, report), render.provenance_html(run),
            render.rsi_html(run), None)


def _json_or_path(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith(("{", "[")):
        try:
            result = json.loads(text)
        except ValueError:
            raise ValueError("Profile and context JSON must be valid objects.") from None
        if not isinstance(result, dict):
            raise ValueError("Profile and context JSON must be objects.")
        return result
    return text


def _settings(values: dict[str, Any], key: str, previous: Any = None, *, clear_key: bool = False, allow_network: bool = False):
    from ...awe.runtime import EvaluationSettings

    if type(values["max_tokens"]) is not int or not 1 <= values["max_tokens"] <= 1_000_000:
        raise ValueError("AWE max output tokens must be a positive integer.")
    if type(values["timeout"]) is not int or not 1 <= values["timeout"] <= 86_400:
        raise ValueError("AWE timeout must be a positive integer.")
    temperature = values["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("AWE temperature must be finite.")
    agreement = values["judge_min_agree"]
    if isinstance(agreement, bool) or not isinstance(agreement, (int, float)) or not math.isfinite(agreement) or not 0 <= agreement <= 1:
        raise ValueError("AWE minimum panel agreement must be between 0 and 1.")
    base = str(values["base_url"] or "").strip()
    if base:
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("AWE base URL must be an HTTP(S) endpoint without embedded credentials, query or fragment.")
    # Changing endpoints must never silently forward a key retained for another
    # service. Model changes on the same endpoint retain the session key.
    retained = (getattr(previous, "api_key", "") if previous is not None
                and getattr(previous, "base_url", "") == str(values["base_url"] or "").strip() else "")
    api_key = "" if clear_key else (str(key or "").strip() or retained)
    args = {name: str(values[name] or "").strip() for name in
            ("reference_root", "judge_model", "base_url", "benchmark_name")}
    args.update({name: str(values[name] or "").strip() or None for name in ("project_data", "output_root")})
    args.update(agents=tuple(a.strip() for a in str(values["agents"] or "").split(",") if a.strip()),
                benchmark_profile=_json_or_path(values["benchmark_profile"]),
                execution_context=_json_or_path(values["execution_context"]),
                temperature=values["temperature"], max_tokens=values["max_tokens"], timeout=values["timeout"],
                judge_min_agree=agreement,
                api_key=api_key, allow_network=allow_network)
    return EvaluationSettings(**args)


def _capabilities(settings: Any) -> str:
    from ...awe.runtime import available_operations

    rows = available_operations(settings)
    body = "<table><thead><tr><th>Operation</th><th>Bundled support</th></tr></thead><tbody>"
    for row in rows:
        state = "Included; input prerequisites checked when run" if row.get("available") else str(row.get("reason") or "Unavailable")
        body += "<tr><td>" + html.escape(str(row["label"])) + "</td><td>" + html.escape(state) + "</td></tr>"
    return body + "</tbody></table>"


def _operation_help(settings: Any, operation: str) -> str:
    from ...awe.runtime import available_operations

    descriptor = next((item for item in available_operations(settings) if item["id"] == operation), None)
    if descriptor is None:
        return render.notice("Choose an available AWE operation.")
    parts = ["<p><b>" + html.escape(str(descriptor["label"])) + "</b></p>"]
    if descriptor.get("reason"):
        parts.append("<p>" + html.escape(str(descriptor["reason"])) + "</p>")
    if descriptor.get("options"):
        options = ", ".join(str(value) for value in descriptor["options"])
        parts.append("<p>Supported JSON options: <code>" + html.escape(options) + "</code>.</p>")
    for requirement in descriptor.get("requirements", []):
        parts.append("<p>Required: " + html.escape(str(requirement)) + ".</p>")
    if descriptor.get("requires_execution"):
        parts.append('<p>Executes external benchmark trials. Enable external execution and set <code>{"confirm_execution": true}</code> explicitly.</p>')
    elif descriptor.get("requires_network"):
        parts.append('<p>Fresh model calls can incur provider charges. Use <code>{"mock": true}</code> or <code>{"dry_run": true}</code> for offline checks.</p>')
    return "".join(parts)


def bind(refs: AWERefs, app: gr.Blocks, *, allow_server_paths: bool = True) -> None:
    from ...awe.saved import load_run

    load_outputs = [refs.run, refs.status, refs.agent, refs.outcome, refs.query, refs.case, refs.case_html,
                    refs.node, refs.graph_html, refs.node_html, refs.symptoms_html, refs.recommendations_html,
                    refs.report, refs.report_html, refs.provenance_html, refs.rsi_html, refs.export_file]
    case_outputs = [refs.case_html, refs.node, refs.graph_html, refs.node_html, refs.symptoms_html, refs.recommendations_html]

    def on_awe_import(file, directory):
        try:
            if directory and not allow_server_paths:
                raise ValueError("Server paths are disabled for this launch.")
            source = (directory or "").strip() or file
            if not source:
                raise ValueError("Choose a saved JSON/ZIP file or a local run folder.")
            run = load_run(source)
            return _load_outputs(run, f"Imported {len(render.all_cases(run))} cases and {len(run.artifacts)} artifacts. No model request was sent.")
        except (ValueError, OSError, TypeError) as exc:
            # Avoid returning untrusted exception text (which may contain server
            # paths or source data); structural loader messages are in its API.
            message = "Import failed. Check the run format, artifact consistency and size limits."
            if isinstance(exc, ValueError) and str(exc) == "Server paths are disabled for this launch.":
                message = str(exc)
            return _load_outputs(None, message)

    refs.import_button.click(fn=on_awe_import, inputs=[refs.file, refs.directory], outputs=load_outputs, api_name="awe_import")

    def on_awe_filter(run, agent, outcome, query, kind):
        options = render.case_options(run, agent, outcome, query) if run is not None else []
        selected = options[0][1] if options else None
        return gr.Dropdown(choices=options, value=selected), *_case_outputs(run, selected, kind)

    for component in (refs.agent, refs.outcome, refs.query):
        component.change(fn=on_awe_filter, inputs=[refs.run, refs.agent, refs.outcome, refs.query, refs.graph_kind],
                         outputs=[refs.case, *case_outputs], api_name=False)

    def on_awe_case(run, key, kind):
        return _case_outputs(run, key, kind)

    refs.case.change(fn=on_awe_case, inputs=[refs.run, refs.case, refs.graph_kind], outputs=case_outputs, api_name="awe_case")
    refs.graph_kind.change(fn=on_awe_case, inputs=[refs.run, refs.case, refs.graph_kind], outputs=case_outputs, api_name=False)

    def on_awe_node(run, key, kind, node):
        graph = render.case_graph(run, render.selected_case(run, key), kind) if run is not None else None
        return render.graph_html(graph, node, run=run), render.node_html(graph, node, run=run)

    refs.node.change(fn=on_awe_node, inputs=[refs.run, refs.case, refs.graph_kind, refs.node],
                     outputs=[refs.graph_html, refs.node_html], api_name="awe_node")

    def on_awe_report(run, name):
        return render.report_html(run, name) if run is not None else ""

    refs.report.change(fn=on_awe_report, inputs=[refs.run, refs.report], outputs=refs.report_html, api_name="awe_report")

    def on_awe_export(run, previous):
        if run is None:
            raise gr.Error("Import a saved AWE run first.")
        directory = tempfile.mkdtemp(prefix="trajviz-awe-download-")
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        destination = Path(directory) / "awe-run.zip"
        try:
            run.export(destination)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise gr.Error("The AWE run could not be exported.") from None
        if previous and Path(previous).name.startswith("trajviz-awe-download-"):
            shutil.rmtree(previous, ignore_errors=True)
        return str(destination), directory

    refs.export_button.click(fn=on_awe_export, inputs=[refs.run, refs.export_state],
                             outputs=[refs.export_file, refs.export_state], api_name="awe_export")
    _bind_runtime(refs, app, load_outputs, allow_server_paths=allow_server_paths)


def _bind_runtime(refs: AWERefs, app: gr.Blocks, load_outputs: list, *, allow_server_paths: bool) -> None:
    from ...awe.saved import load_run

    controls = refs.runtime
    fields = [controls[name] for name in _PREF_FIELDS]

    def on_awe_apply(*args):
        if not allow_server_paths:
            raise gr.Error("Evaluation operations require a local launch.")
        values = dict(zip(_PREF_FIELDS, args[:len(_PREF_FIELDS)], strict=True))
        key, clear, remember, previous = args[len(_PREF_FIELDS):]
        try:
            selected = _settings(values, key, previous, clear_key=bool(clear))
            capabilities = _capabilities(selected)
        except (ValueError, TypeError, OSError):
            raise gr.Error("AWE settings are invalid. Check the data paths and request options.") from None
        saved = {"version": 1, "values": values} if remember else {}
        return selected, "", False, "AWE settings applied. API keys remain session-only. No operation was started.", saved, capabilities

    applied = controls["apply"].click(fn=on_awe_apply, inputs=[*fields, controls["api_key"], controls["clear_key"], controls["remember"], refs.settings],
                                    outputs=[refs.settings, controls["api_key"], controls["clear_key"], controls["settings_status"], controls["browser"], controls["capabilities"]],
                                    api_name="awe_apply_settings")

    def on_awe_operation_help(settings, operation):
        return _operation_help(settings, operation)

    controls["operation"].change(fn=on_awe_operation_help, inputs=[refs.settings, controls["operation"]],
                                 outputs=controls["operation_help"], api_name=False)
    applied.then(fn=on_awe_operation_help, inputs=[refs.settings, controls["operation"]],
                 outputs=controls["operation_help"], api_name=False)

    def on_awe_restore(saved):
        values = dict(_DEFAULTS)
        cleaned = {}
        if allow_server_paths and isinstance(saved, dict) and saved.get("version") == 1 and isinstance(saved.get("values"), dict):
            candidate = saved["values"]
            for name in _PREF_FIELDS:
                value = candidate.get(name, values[name])
                if name in ("temperature", "max_tokens", "timeout", "judge_min_agree"):
                    if type(value) not in (int, float):
                        continue
                elif not isinstance(value, str):
                    continue
                values[name] = value
            try:
                _settings(values, "", clear_key=True)
            except (ValueError, TypeError, OSError):
                values = dict(_DEFAULTS)
            else:
                cleaned = {"version": 1, "values": values}
        return (None, *(values[name] for name in _PREF_FIELDS), "", False,
                "AWE connection preferences restored. Enter a key and apply settings before running an operation.", cleaned)

    app.load(fn=on_awe_restore, inputs=[controls["browser"]],
             outputs=[refs.settings, *fields, controls["api_key"], controls["clear_key"], controls["settings_status"], controls["browser"]], api_name=False)

    def on_awe_reset():
        return (None, *(_DEFAULTS[name] for name in _PREF_FIELDS), "", False, False, {}, "AWE settings cleared.", "")

    controls["reset"].click(fn=on_awe_reset,
                            outputs=[refs.settings, *fields, controls["api_key"], controls["clear_key"], controls["remember"],
                                     controls["browser"], controls["settings_status"], controls["capabilities"]], api_name="awe_reset_settings")

    def on_awe_operation(settings, operation, agent, instance, limit, payload, allow_network):
        if not allow_server_paths:
            raise gr.Error("Evaluation operations require a local launch.")
        if settings is None:
            raise gr.Error("Apply AWE settings before running an operation.")
        from dataclasses import asdict, replace

        from ...awe.runtime import run_operation

        try:
            limit_text = "" if limit is None else str(limit).strip()
            case_limit = int(limit_text) if limit_text else None
            if case_limit is not None and case_limit <= 0:
                raise ValueError("non-positive limit")
        except (TypeError, ValueError):
            raise gr.Error("Operation task limit must be blank or a positive integer.") from None

        try:
            options = json.loads(payload or "{}")
            if not isinstance(options, dict):
                raise ValueError("Operation options must be an object.")
            result = run_operation(operation, replace(settings, allow_network=bool(allow_network)),
                                   agent=(agent or "").strip() or None, instance_id=(instance or "").strip() or None,
                                   limit=case_limit, payload=options)
        except (TypeError, ValueError, OSError):
            raise gr.Error("AWE operation options are invalid. Check the selected operation and its JSON fields.") from None
        document = asdict(result)
        # Settings and keys never become outputs. OperationResult is the public,
        # redacted result boundary provided by the runtime integration.
        active = settings
        if result.success and result.project_data:
            active = replace(settings, project_data=result.project_data, allow_network=False)
        if result.success and operation == "ingest_harbor" and isinstance(result.data, dict) and result.data.get("canonical_root"):
            active = replace(active, reference_root=str(result.data["canonical_root"]),
                             agents=tuple(result.data.get("agents") or active.agents),
                             benchmark_name=str(result.data.get("benchmark_name") or active.benchmark_name))
        # Subsequent batch analysis must see verdicts/per-test records produced
        # by earlier operations in this session. External execution consent is
        # one-shot and is reset after each completed request.
        return (document, result.logs, result.output_dir if result.success else None, active, False,
                str(active.project_data or ""), str(active.reference_root), ",".join(active.agents), active.benchmark_name)

    controls["run"].click(fn=on_awe_operation,
                          inputs=[refs.settings, controls["operation"], controls["agent"], controls["instance_id"], controls["limit"], controls["payload"], controls["allow_network"]],
                          outputs=[controls["result"], controls["logs"], controls["result_path"], refs.settings, controls["allow_network"],
                                   controls["project_data"], controls["reference_root"], controls["agents"], controls["benchmark_name"]], api_name="awe_run_operation")

    def on_awe_import_result(path):
        if not allow_server_paths:
            raise gr.Error("Evaluation operations require a local launch.")
        try:
            if not path:
                raise ValueError("No completed operation output.")
            run = load_run(path)
        except (ValueError, OSError, TypeError):
            return _load_outputs(None, "No importable run was produced. Inspect the operation result and log.")
        return _load_outputs(run, f"Loaded operation output: {len(render.all_cases(run))} cases, {len(run.artifacts)} artifacts.")

    controls["import_result"].click(fn=on_awe_import_result, inputs=controls["result_path"], outputs=load_outputs, api_name="awe_import_operation_output")
