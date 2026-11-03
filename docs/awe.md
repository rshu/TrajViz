# AWE analysis and saved runs

TrajViz includes a self-contained AWE workflow evaluation engine, its catalog
resources, and a viewer for complete saved analyses. The evaluation engine,
Python API, CLI and offline HTML export use only the standard library. The
dashboard requires the `app` extra. Model judging is optional and uses your
explicitly configured connection. Use versioned documentation that matches your
installed package.

## Inspect a saved analysis

In the dashboard's **AWE** tab, upload a run ZIP or JSON artifact, or select a
local run folder. AWE's immutable `data/runs/<run-id>/` directory is the best
source for a complete analysis. Current `data/dossier/` outputs and their
sibling reports also work. A local `faults.json` or `manifest.json` selects its
sibling artifacts; a portable TrajViz run JSON contains the whole imported run.

The viewer provides:

- Case filters by agent, task and recorded outcome, including symptom-only
  solved and unknown cases.
- Full task, detector, patch and confidence evidence; independent audits,
  mismatches, repair-locus context, and exact attribution policy traces.
- Numbered attribution graphs, selectable node details, and recorded
  action/result and failure/recovery episodes.
- Unscored symptoms, remedy selection decisions, recommendation plans, RSI
  exchange, aggregate reports, and discovery artifacts when supplied.
- Original manifests, per-artifact fingerprints, and a download of the
  byte-identical source artifacts.

Missing artifacts remain unavailable. Import does not recompute diagnoses,
invent missing outcomes, or call a model. Stored HTML is treated as evidence,
never executed. The graph arrows describe recorded provenance and policy
relations; they do not establish causal effects.

The public API is [`agent_trajviz.awe`](api/agent_trajviz/awe.html):

```python
from agent_trajviz import awe

run = awe.load_run("/path/to/saved-run.zip")
print(run.manifest)
print(run.cases[0]["faults"] if run.cases else "No fault dossier supplied")
run.write("portable-awe-run.json")
run.export("original-awe-artifacts.zip")
awe.write_report(run, "awe-report.html")
```

Use separate destinations for exported files. Exports protect imported sources
and artifacts, and preserve previous output if writing fails.

`SavedRun.artifacts` holds JSON objects or UTF-8 text keyed by original relative
path. Opaque attachments have a descriptor and preserved bytes in
`raw_artifacts`. `fingerprints` records each artifact's SHA-256 and byte length.
`cases`, `graphs`, `symptoms`, `recommendations` and `manifest` expose the
corresponding recorded data. Unknown additive fields survive round trips.
Treat these objects as read-only: an export refuses edited evidence that no
longer matches its original bytes.

This small offline example needs no saved corpus or model connection:

<!-- trajviz:run -->
```python
from agent_trajviz import awe

example = awe.load_run({
    "case_id": "demo__task", "agent": "demo", "instance_id": "task",
    "blame_status": "unattributed", "faults": [],
    "example_note": "A minimal saved record; no cause was assigned.",
})
example.write("example-awe-run.json")
example.export("example-awe-run.zip")
restored = awe.load_run("example-awe-run.zip")
assert restored.cases == example.cases
awe.write_report(restored, "example-awe-report.html")
```

ZIP paths, links, oversized inputs, malformed graphs, duplicate identities,
stale dossier hashes and conflicting artifact scopes are rejected. Original
run provenance is distinguished from TrajViz's import fingerprints; importing
a historical verdict does not retroactively verify its model inputs.

## CLI import and export

```bash
agent-trajviz-awe inspect /path/to/saved-run.zip
agent-trajviz-awe inspect /path/to/saved-run --output portable-run.json
agent-trajviz-awe export portable-run.json --output original-artifacts.zip
agent-trajviz-awe export portable-run.json --output report.html
```

`python -m agent_trajviz.awe` provides the same commands. An `.html` output
produces a self-contained TrajViz report; `.zip` preserves the original
artifacts, and other export suffixes write the versioned JSON envelope.

## Configure evaluation

Evaluation logic and resources ship with TrajViz. List the built-in operations
without configuring a model or external code location:

```bash
agent-trajviz-awe operations
```

Use the dashboard's **AWE Settings and operations** section for the reference
root, optional working data/cache root, output folder, agents, benchmark profile,
execution context, judge model, base URL and request parameters. These controls
are separate from the sidebar AI analysis settings. Server-path controls are
available only for a local dashboard launch.

Nonsecret connection details can be remembered in this browser. API keys remain
in the current session, are hidden from settings representations and operation
logs, and are omitted by `EvaluationSettings.nonsecret()`. Applying or restoring
settings does not execute an operation. Fresh model requests additionally
require the operation's network checkbox; that permission is not remembered.
AWE model operations use an OpenAI-compatible chat endpoint.

The reference root uses AWE's benchmark contract:

```text
benchmark/
  data/
    requirements/<task>.json
    patch/<agent>/<task>.diff
    trajectory/<agent>/<task>.jsonl
    eval_<agent>.json
```

Requirements supply the problem statement, reference patch and benchmark test
metadata. Official per-test records and grader logs belong under the configured
AWE project data root's `pertest/<agent>/`; documented outcome overrides belong
under `overrides/`. A benchmark profile supplies file roles and evaluator policy.
Declared execution context supplies model, scaffold revision, languages and
runtime capabilities. Missing declarations remain unknown.

Native Claude Code, Codex, Cursor, OpenCode, Pi and DeepSeek Harness sessions use
TrajViz's loaders inside isolated workers. Their projection preserves recorded
tool inputs/results, exit codes, reasoning visibility and child thread identity.
Evidence references identify TrajViz workflow rows and parts; a native record
offset is included only when known. Parent and child fingerprints, adapter
version and code identity bind caches to the projected evidence. Saved telemetry
captures cannot supply the transcript/reference evidence required for AWE.

## Execute explicit operations

```python
from agent_trajviz import awe

settings = awe.EvaluationSettings(
    reference_root="/path/to/benchmark",
    project_data="/path/to/awe-working-data",
    agents=("claude_code",),
    judge_model="your-cached-model-id",
)
result = awe.run_operation(
    "diagnose", settings, agent="claude_code", instance_id="task-id",
    payload={"use_judge": True, "use_arbiter": True},
)
if result.success and result.output_dir:
    saved = awe.load_run(result.output_dir)
    awe.write_report(saved, "diagnosis.html")
else:
    print(result.reason)
```

`available_operations(settings)` returns the bundled operation catalog. `run_operation`
returns an `OperationResult` with `success`, `available`, `reason`, `data`,
`output_dir`, `project_data`, `artifacts`, sanitized `logs` and `warnings`.
Its result shape is operation-specific. Every job runs in a private process
using the bundled engine and isolated working inputs. Outputs carry an operation
manifest, engine and native-adapter identities, and artifact hashes. Pipeline results select their immutable
run archive. Failed jobs retain available reports and logs for inspection.

Operations are offline by default. Diagnosis and batch analysis read only
usable cached judge/arbiter verdicts. Strict verdict provenance excludes stale
or unverifiable caches. Fresh `judge`, `arbiter` and `scout` calls require
`allow_network=True` and a complete base URL, API key and model. Their
`{"mock": true}` or `{"dry_run": true}` options support offline checks.

```bash
agent-trajviz-awe run preflight \
  --reference-root /path/to/benchmark --agents claude_code
agent-trajviz-awe run pipeline \
  --reference-root /path/to/benchmark --agents claude_code --limit 20
```

For a requested fresh model run, load the key into `AWE_API_KEY` securely and
pass `--allow-network`, `--base-url` and `--judge-model`. `--api-key-env` selects
another environment variable; keys are never command-line argument values.
`--payload options.json` supplies operation-specific JSON options.

| Family | Operation IDs |
|---|---|
| Analysis and archives | `diagnose`, `pipeline`, `symptoms` |
| Saved-result generation | `graphs`, `recommendations`, `remedies`, `rsi`, `showcase` |
| Model evidence | `judge`, `arbiter`, `scout` |
| Discovery | `cohort`, `behaviors`, `fragile` |
| Benchmark inputs and evaluator evidence | `ingest_harbor`, `pertest_report`, `pertest_harness`, `grader_log_audit`, `harness_suspects` |
| Validation and controls | `preflight`, `preprocessing`, `validation`, `fault_injection`, `publication` |
| Calibration and agreement | `calibration`, `calibration_set`, `agreement`, `judge_agreement`, `judge_panel`, `threshold_stability`, `difficulty_strata` |
| Remedy provenance and experiments | `catalog`, `remedy_verify`, `bibliography`, `papers_status`, `scaffold_view`, `rsi_prepare`, `rsi_decide`, `guardrail_observe`, `guardrail_decide` |

Supported options are validated per operation. Examples include `workers` for
batch scans, `dossier` for refreshing saved outputs, and `model_a`/`model_b` for
cached-model agreement. Official `pertest_harness` execution additionally
requires `confirm_execution: true`, Docker, the benchmark evaluator environment
and its data. Docker resources have a lifecycle managed by the evaluator and
the Docker daemon; stopping the Python job does not prove that every container
or image has been removed. Review evaluator receipts and daemon state after a
run. Local validation covers request checks and mocked harness boundaries;
no live Docker benchmark trial is required for ordinary TrajViz use.

## RSI and remedy boundaries

Recommendation frequency describes diagnostic coverage, not expected benefit.
Catalog recipes are experiment specifications. Generation does not install a
policy, modify an agent, execute a trial or prove an improvement. The bundled engine currently
provides one bounded local-verification policy prototype; the full recipes
still require native adapters and controlled evaluation.

`rsi` exports typed proposals and optimizer-development feedback. The latter's
`runtime_input_allowed: false` boundary remains visible: keep privileged
reference evidence and private test identifiers outside a task agent's inputs.
`rsi_prepare` accepts `proposal` and independently observed `observed_context`.
`rsi_decide` accepts `prepared`, public runtime `state` and `trigger_count`, and
returns an advisory action. Neither operation executes the requested project
tests or installs a hook.

`guardrail_observe` evaluates public observations; `guardrail_decide` accepts a
controller snapshot, event identity and bounded budget. A native controller
must persist the returned state across hooks and retries, collect real action
receipts, run matched controls and retain unsuccessful trials. TrajViz exposes
these contracts and evidence checks; native agent execution remains the
controller's responsibility.
