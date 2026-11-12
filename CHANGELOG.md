# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/): before 1.0, a minor version may
change the public API (the names exported from `agent_trajviz`).

## [Unreleased]

## [0.5.1] - 2026-11-12

### Fixed

- Show label coverage and mismatch errors, and hide the plan timeline when
  the session recorded no plan.
- Recover shell reads and searches from the recorded working directory and
  retain writes when the same command also reads the file.
- Show nonzero tool exit codes as failures in Workflow and AWE.
- Recognize passing pytest summaries in piped output and clarify the
  **Last Recognized Edit** milestone in comparison views.

## [0.5.0] - 2026-11-09

### Added

- Switch Insight and Converge between English and Simplified Chinese and
  choose the language for AI replies.
- Load CodeAgent session exports, including nested subagent sessions.
- Export grouped Overview issues as JSON and request AI analysis for an
  individual issue or the entire run.
- Inspect execution timelines, context growth, compaction, and per-step
  token use.

### Changed

- Group related Overview issues, collapse duplicate failure cascades, and keep
  their evidence links and AI actions aligned across Settings changes.
- Support custom gateway headers for model requests and connection tests.

### Fixed

- Load OpenCode desktop and Windows exports, including UTF-16 session files
  with a BOM. Preserve attachments and provider errors in native sessions.
- Read UTF-16 DeepSeek Harness sessions and preserve subagent discovery in
  exports containing multibyte text.
- Count cached tokens and reasoning correctly and handle incomplete telemetry
  without crashing the Overview or Hotspots views.
- Keep trajectory loading offline. Scope gateway credentials to the configured
  connection and mask the active API key after applying settings.

## [0.4.2] - 2026-11-04

### Added

- Inspect loaded trajectories directly in AWE, with recorded findings, file and
  test activity, recovery observations, and links to supporting Workflow steps.
- Add `awe.inspect_trajectory(run)` for inspection from an already-loaded session.
- Test model connections from AI Trajectory Analysis Settings with a small,
  explicit request that sends no trajectory data.

### Changed

- Make trajectory inspection the default AWE view, with optional reference-based
  attribution and a separate saved-run viewer.
- Move model settings into AI Trajectory Analysis and require an explicit action
  to start model analysis. Keep API keys in the current session.

### Fixed

- Preserve native tool outcomes, file actions and child-thread identities; keep
  quoted output and incomplete recordings from creating unsupported findings.
- Correct comparison file identities, context-window explanations and label
  coverage, and remove excess padding from labeled step timelines.
- Protect source files during label exports and handle malformed optional saved
  evidence without crashing or inventing assessments.
- Improve responsive settings, evidence links, contrast and AWE navigation.

## [0.4.1] - 2026-11-03

### Fixed

- Protect imported source and taxonomy files during exports and labeling, and
  keep previous exports intact when a write fails.
- Preserve explicitly selected trajectory formats through loading and exports.
- Export unbounded cohort risk ratios as valid JSON with a clear interpretation.
- Preserve computed scorecards in saved runs and distinguish historical catalog
  notes from evidence collected for the current run.
- Improve dashboard layouts, chart legends, wide tables, pattern navigation and
  label provenance, including Converge milestone and waterfall labels.
- Keep missing success, context and timing measurements unavailable.
- Keep the independent AWE workspace accessible after a native import error.
- Keep portable HTML report references independent of live dashboard navigation.

## [0.4.0] - 2026-11-03

### Added

- Bundle the self-contained workflow evaluation engine and catalog resources.
- Import complete saved AWE runs as JSON, ZIP or directories, preserving original
  evidence, manifests and artifact fingerprints, with JSON, ZIP and offline HTML exports.
- Add numbered provenance graphs with source navigation, full case evidence, symptoms,
  recommendation plans, aggregate reports, discovery results and RSI exchange views.
- Provide AWE Settings with session-only keys and explicit isolated operations for
  analysis, batch archives, judging, benchmark ingestion, evaluation, discovery,
  validation, calibration, catalog inspection and experimental policy controls.
- Add the `agent-trajviz-awe` CLI and public `agent_trajviz.awe` API.

### Fixed

- Refuse failure attribution when the evaluation engine cannot extract workflow
  steps from a native session, instead of treating missing parser evidence as
  workflow failures.
- Preserve native tool results, thread identities, working directories and hidden
  reasoning when projecting TrajViz sessions into built-in workflow evaluation.
- Keep complete attribution dossiers and audits, with bundled engine and native-session checks.

## [0.3.2] - 2026-11-02

### Fixed

- Mark a reply the model cut off at the output-token limit as cut, with the limit and how to
  raise it, and explain when the whole budget went to hidden reasoning, instead of showing
  "(The model returned an empty reply.)" or "The analysis model request failed: ValueError."
- Name the Settings fields in analysis error messages when the connection came from Settings,
  and report a read timeout as a timeout rather than as an unreachable API.
- Accept Anthropic-style base URLs with or without `/v1` in the dashboard and the labeler; the
  documented `https://api.anthropic.com` returned HTTP 404. A base URL without `/v1` now posts
  to `<base>/v1/messages` (previously `<base>/messages`), so a gateway that served only the old
  path must also serve `<base>/v1/messages`.
- Keep the analysis brief's bottleneck, failure, error and step sections for sessions with many
  sub-agents by listing at most 24 agents and 50 insights, and name the sections a truncated
  brief dropped.
- Make `agent-trajviz-label` exit with status 2 and write no sidecar when the API rejects its
  credentials or every classification fails, name HTTP statuses in its log, retry replies that
  were all hidden reasoning, and count only written records in `--assistant-only` sidecars.
- End each Codex turn at its `task_complete` event so idle time between turns is not reported
  as inference, and never report a negative first-response latency.
- Surface DeepSeek Harness model-request failures (`assistant/attempt` and `turn/end` errors)
  as error steps with load warnings.
- Reject label sidecars with conflicting native-session identity and allow selecting a
  captured reference session for pairwise comparison.
- Keep shell counts, formatter arguments and executable invocations out of file-
  interaction charts while retaining genuine file operands and write targets.
- Follow executed validation commands through their observed completion in comparison
  milestones, and explain when a capture lacks transcript evidence.
- Order automatic failure attribution reliably, distinguish AWE evidence numbering, and
  show evidence qualifications and untested counterfactual checks.
- Improve mobile attribution layouts and remove empty warning and reasoning panels.

### Changed

- The default Max output tokens is 8192 for the dashboard (`ANALYZE_MAX_TOKENS`) and 4096 for
  `agent-trajviz-label` (`LABEL_MAX_TOKENS`), up from 2048 and 1024, so reasoning models can
  finish a reply.
- Steps whose model request failed (`finish` "error") count in the Errors health check and
  appear under ERROR_STEPS in the analysis brief for every format, including OpenCode and Pi,
  and are also flagged `ERR` in the brief's STEP_INDEX.

### Added

- Publish versioned API documentation and tested usage guides, with stable and
  development sites built automatically from the supported public interfaces.
- Provide a Homebrew tap for the dashboard and CLI, with tested updates tied to
  the corresponding PyPI release.

## [0.3.1] - 2026-10-29

### Added

- Dashboard Settings for model connections, request options and reply language,
  with session-only API keys and optional remembered connection details.

### Fixed

- Reject malformed normalized trajectories before analysis, preserve large JSONL
  round trips and handle invalid numeric observations without crashing.
- Retain failed batch inputs and unpaired tasks; keep comparison effects unknown
  when tool outcomes are unavailable.
- Preserve unrelated files during JSON exports, reject unsafe or colliding view
  filenames, and report output-path errors without a traceback.
- Handle empty model responses and align label timeline schemas with labeler output.
- Support unavailable measurements and explicit session selection in example apps,
  and clear stale labels, comparisons and downloads when the input changes.
- Correct mobile header overlap, repeated warning padding, upload action sizing
  and clipped comparison tables.

## [0.3.0] - 2026-10-27

### Added

- Read OpenCode 2 typed session exports and collect selected parent/child families
  from its native session store without modifying the database.
- Import Cursor Agent CLI stream JSONL, retaining thinking, tool results and
  execution clocks while deduplicating response flushes and replayed events.
- Collect Codex native parent/reviewer sessions with an explicit read-only
  collector, and load portable verified child rollout trees.
- Read current DeepSeek Harness numbered native generations and compressed v4
  session trees. Python 3.11–3.13 can install the optional `zstd` extra.
- Add focused regression fixtures for current coding-agent session formats.
- Import saved OTLP JSON/JSONL traces from Claude Code, with correlated event
  logs, explicit directory/ZIP manifests, subagent hierarchy and provenance.
  Native Claude Code JSONL and its subagent bundle loading remain supported.
- Discover and select captured sessions with `list_sessions()`,
  `load(..., session_id=...)`, CLI `--list-sessions` / `--session-id`, and the
  dashboard session selector. No telemetry SDK dependency or network receiver
  is needed for importing files.
- Explain observed, partial and unavailable evidence in the `data_quality`
  view, JSON view availability, dashboard and reports. Missing observations
  do not become measured zero values or healthy verdicts.
- Document a local Collector configuration and capture manifest in the README.

### Changed

- View JSON schemas advance to version 2 for nullable unavailable metrics,
  unavailable health checks and evidence coverage. Existing normalized
  trajectory schema-v1 exports remain readable.
- Identify an imported OTLP run by its source capture, selected session and
  mapping profile. Raw source hashes and original span/log context are retained.
  Transcript comparison and failure attribution refuse OTLP captures until
  their assumptions can be supported by the available evidence.

### Fixed

- Preserve current Codex structured tool failures and DeepSeek v4 result fields.
  Show readable Codex error diagnostics while retaining original output payloads.
- Include active and interrupted OpenCode reviewer sessions during collection,
  respect native parent relationships and keep unfinished response timing unknown.
  Distinguish child prompt and model-mode lanes in workflow filter labels.
- Measure post-compaction context from the next live request rather than the
  compaction request’s billed input; allow unavailable compaction measurements
  in exported view schemas.
- Keep missing native clocks and unresolved search outcomes unavailable instead
  of reporting measured zero durations or fruitless-search verdicts.
- Use OpenCode 2 model response clocks separately from later tool settlement,
  preserving running tools, lifecycle failures and compaction status.
- Harden saved OTLP imports against corrupt compressed members and ambiguous
  request joins; preserve model errors, token evidence and sub-millisecond tool
  timing. Keep session selection, missing-data displays and report exports
  consistent across the API, CLI and dashboard.
- Skip unnecessary secret-name matching on delimiter-free text, keeping
  pathological transcript redaction fast on slower macOS runners.
- Preserve Unicode separators inside Cursor JSON strings, keep reviewer peers
  at the correct depth, and surface missing outcomes, outputs and clocks.
  Unknown tool outcomes no longer establish success, recovery or edit precision.
- Exclude delayed Claude Code interruption markers from conversation duration
  while retaining an explicit incomplete-transcript warning.
- Trim inherited Codex child history and retain current model metadata,
  usage records, source identities and parent/reviewer relationships.
- Correct Pi and DeepSeek Harness token totals when reasoning is nested in
  output; use recorded request intervals separately from tool execution time.
- Select Pi's active history branch and retain compaction evidence.
- Keep context composition percentages unavailable when an assumed window
  contradicts the capture, and preserve measured model intervals in diagnostics.
- Warn when an OpenCode capture ends before its last response finishes.
- Preserve OpenCode native part timing during collection and include recorded
  cache/reasoning consumption in raw aggregate token totals.

## [0.2.0] - 2026-10-15

### Added

- Read Claude Code's native session JSONL directly, including associated
  subagent logs. Local parent files discover children; session directories and
  ZIP bundles preserve the parent/subagent layout for loading or uploading.

### Changed

- Claude Code's format key is `claude_code`. Replace `format="ccsession"`
  with `format="claude_code"`, or use auto-detection with native session logs.
  Existing normalized TrajViz exports migrate automatically, preserving agent
  labels and subagent spawn links.

### Removed

- The non-native `ccsession-trajectory` input wrapper. Claude Code no longer
  needs an intermediate export. TrajViz's public `trajectory.json` round-trip
  format remains supported.

### Fixed

- Keep endpoint URL credentials out of the analysis status and provider error
  details out of labeler retry logs. Limit labeler `.env` imports to
  `ANALYZE_*` and `LABEL_*`, matching the dashboard's configuration boundary.
- Remove dashboard report scratch files when their session unloads or the
  process exits normally, without removing other viewers' active downloads or
  standalone CLI/library reports. Failed report rendering/writing no longer
  leaves a newly allocated scratch directory behind.
- Create dashboard reports inside Gradio's cache so the download serves the
  session-owned file directly instead of retaining a second cached copy.
- Support current attribution backend APIs while preserving offline attribution, verified
  cache provenance, and canonical source identity.
- Report whether attribution backend actually used an available judge verdict when its
  automatic cache mode is enabled.

## [0.1.0] - 2026-10-12

First release as the `agent-trajviz` package.

### Added

- A public library API: `load()` returns a `Trajectory` (format, normalized
  steps, provenance, load warnings) with cached `metrics`, `health` and
  `agents`, plus `context_pressure()` and a versioned, JSON-serializable
  `to_dict()`; `compare()` aligns two runs; `write_report()` writes the
  standalone HTML report. Steps are described by the `Step`, `ToolCall` and
  `TokenUsage` typed records; errors derive from `TrajVizError` (`LoadError`,
  `ReportError`, `MissingExtraError`). The package ships `py.typed`.
- Optional extras: `charts` (figures, HTML report), `llm` (AI sidebar, Issues
  judge, labeler), `app` (both dashboards) and `all`. The core needs only the
  standard library, and a command whose extra is missing names it.
- Console commands: `agent-trajviz` (dashboard, or `--report` for HTML),
  `agent-trajviz-converge`, `agent-trajviz-collect cursor|opencode|codearts`
  and `agent-trajviz-label`; each takes `--version`. The Converge dashboard
  is `python -m agent_trajviz.converge`.
- Supported formats: Claude Code, Cursor, OpenCode, CodeArts, ICode, Codex CLI,
  Pi and DeepSeek Harness.
- Dashboard server options shared by both apps: `--host`, `--port`, `--share`
  and `--max-file-size` (default 500 MB), and a login from
  `TRAJVIZ_AUTH=user:password`. The Converge dashboard takes `--data-root`.
- `SECURITY.md`: how to report a vulnerability, and the security model.
- Python 3.14 support.
- Every chart and table as data: `Trajectory.views()` / `.view(id, **options)`
  return `View`s (id, title, kind, description, JSON data) for 35 charts
  and tables, listed in `VIEW_CATALOG`; the dashboard draws its charts from
  exactly this data. `export_json()` writes them (`<id>.json`, `index.json`,
  `trajectory.json`), as does `agent-trajviz --json RUN [-o DIR] [--view ID]`.
- A JSON Schema (draft 2020-12) for every view's data, shipped in the package:
  `view_schema()`, `view_schemas()` and `agent-trajviz --schemas DIR`;
  `register_view(..., schema=)` describes a view of your own. The test suite
  validates every view strictly against its schema.
- The dashboard is built on the public API: it opens files with `load()`
  (so formats registered by an application or a plugin open, and so do
  exported `trajectory.json` files), draws its Overview and diagnostics
  charts from the public views with `figures.figure()`, and keeps the
  loaded `Trajectory` for the Comparison tab, the Issues judge and the HTML
  export. Views added with `register_view()` get a Views tab (absent
  otherwise). `write_report()` takes any `Trajectory`, including one built
  with `from_steps()`.
- Extension points: `Trajectory.from_steps()` analyzes steps parsed by the
  caller (missing keys are filled as for a format that does not record
  them); `register_format()` with `Format` teaches `load()` a format of your
  own; `register_view()` adds a view to `VIEW_CATALOG` and everything that
  reads it; `figures.register_figure()` draws it; `load_plugins()` runs the
  hooks installed packages declare in the `agent_trajviz.plugins`
  entry-point group, as `agent-trajviz` does on start (`PluginError` names a
  failing one). `FORMATS`, `VIEW_CATALOG` and `figures.FIGURE_VIEWS` are live
  read-only views of the registries.
- Round trip: `Trajectory.to_dict(include_raw=True)` keeps the converted
  export, `Trajectory.from_dict()` rebuilds the trajectory, and `load()`
  reads such a document (the `trajectory.json` that `export_json()` writes)
  back to identical views.
- The dashboard's inputs as API: `load(..., format=)` (the format selector),
  view options for the context-utilization agent, window and snapshot, a
  `workflow` view (the step browser's classification and filters), a
  `session` view, `compare(..., anchor_patch=)`, `compare_runs()` with
  `run_group_views()` (N-run scorecard), and `label_views()` /
  `label_comparison_views()` (step-label sidecars).
- Optional public layers: `agent_trajviz.figures` (charts extra) draws any
  chart view with the dashboard's code; `agent_trajviz.dashboard` (app
  extra) launches or embeds the dashboards and turns views into Gradio
  components; `agent_trajviz.llm` (llm extra) is the AI analysis and
  suggested fixes, on the configured endpoint or any chat function you pass
  (`chat=`), in the reply language you choose (`language=`);
  `agent_trajviz.attribution` is failure attribution.
- `examples/`: the dashboard rebuilt on the public API alone, and a small
  app of your own, both driven end to end in the test suite.

### Changed

- The import package is `agent_trajviz` (was `trajviz`) and the dashboard
  command is `agent-trajviz` (was `trajviz`). Every command carries the
  `agent-trajviz` prefix; the Converge command line is
  `agent-trajviz-converge` (was `trajectory-converge`).
- The collection and labeling tools moved from `scripts/` into the package.
- The step labeler is one command, `agent-trajviz-label`: `--assistant-only`
  replaces the separate compatibility command, the default output is
  `<input>_labeled.json`, and progress goes through `logging`. Module and
  file names no longer carry version labels (`step_labeler_v2` is
  `agent_trajviz.labeling.labeler`); the CodeArts tool writes
  `<session>_trajectory.json`.
- Every command's entry point takes `argv` and returns its exit status: 0
  success, 1 a failure caused by the input, 2 a usage error, and for
  `agent-trajviz-collect codearts` 3 a partial export (was 2). Input failures
  print one sentence; a genuine bug keeps its traceback.
- Library code logs through `logging` (the package root has a
  `NullHandler`) instead of printing tracebacks to stderr; the commands
  attach a stderr handler.
- Agent rows (`Trajectory.agents`, the Agents cards) report
  `output_tokens_per_sec` -- output tokens over generation time, the same
  measurement as the Overview's "gen tok/s" -- with its timed-step coverage,
  instead of all tokens over wall-clock time. Cards count "assistant steps".
- Converge: the batch aggregate's `success_count`/`failure_count` are
  `compared_count`/`error_count` (they counted comparisons, not successful
  runs) and per-task status is `compared`/`error`; the pairwise summary says
  whether each run "stopped normally" rather than "Success"/"Failure".
- Converge before/after significance is a pure-Python Wilcoxon signed-rank
  test with Holm correction across metrics (`paired_significance`, was
  `test_significance`), so a result no longer depends on whether SciPy is
  installed.
- The Converge dashboard listens on port 7861 by default, beside Insight's
  7860.
- The AI sidebar and the Issues judge reply in English by default (was
  Simplified Chinese); `ANALYZE_LANGUAGE` still picks any language, and
  Simplified Chinese keeps its own prompt with Chinese headings.
- `load()` raises `LoadError` with code `unknown_format` for a file no format
  recognizes, where it used to return a trajectory of format `"unknown"`
  when the file happened to parse (the dashboard already refused such a
  file). On a format mismatch, `LoadError.detected` names what the file is.
- Views on long runs take linear time: each shell command is lexed once
  instead of once per chart that names tools.
- Dependency lower bounds are the oldest releases the suite passes with:
  plotly 5.15, pygments 2.13, requests 2.32.4, pytest 8.0, hypothesis 6.100.

### Removed

- `requirements.txt`; `pyproject.toml` is the single source of dependencies.
- The `tokens_per_second` field of agent rows (see `output_tokens_per_sec`).

### Fixed

- A run with no tool calls reported a tool success rate of 0% ("every call
  failed"); the rate is now `None` and shown as n/a.
- The per-step token chart stacked the remaining fields of a step whose
  export reported an impossible count (a negative input), drawing it taller
  than its own total. Such steps are drawn as their reported total in a
  separate, labelled series, and the title counts them.
- The Converge dashboard resolved a manifest's relative paths against the
  upload's temporary directory, so they could never be found.
- A cleared Token Rate box in the Converge dashboard raised a `TypeError`.
- `compare_runs()` refused a trajectory that was not loaded from a file, and
  read a file only a registered format understands as an error.
- The anchor patch, the Converge batch manifest and cached attribution
  verdicts were read in the platform's default encoding, which fails on
  Windows for UTF-8 files; they are read as UTF-8, and a patch byte that is
  not UTF-8 no longer drops the anchor.
- `agent-trajviz-collect opencode` printed usage examples without its
  subcommand.
- Help and setup text that described a source checkout to installed users,
  or claimed cache-read anomalies were left out of the token totals.

### Security

- An exposed dashboard (`--share`, or a non-loopback `--host`) warns unless
  `TRAJVIZ_AUTH` sets a login, and hides inputs that name paths on the
  serving machine (the Attribution tab's reference-data root), ignoring such
  a value even when a client sends one.
- Converge manifest paths must stay inside `--data-root`, `..` and symlinks
  included.
- Errors shown in the Converge dashboard carry no traceback and no server
  paths.
- The dashboards no longer send Gradio's usage telemetry.
- `llm.LLMConfig` no longer shows its API key when printed, logged or
  shown in a traceback.
- Every LLM request (AI sidebar, Issues judge, step labeler) is redacted
  before it leaves: provider API keys, tokens, JWTs, private keys, URL
  passwords, authorization headers and generated-looking values of
  secret-named assignments become `[REDACTED]`.

[Unreleased]: https://github.com/rshu/TrajViz/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/rshu/TrajViz/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/rshu/TrajViz/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/rshu/TrajViz/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/rshu/TrajViz/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/rshu/TrajViz/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/rshu/TrajViz/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/rshu/TrajViz/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/rshu/TrajViz/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rshu/TrajViz/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rshu/TrajViz/releases/tag/v0.1.0
