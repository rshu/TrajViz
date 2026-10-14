# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/): before 1.0, a minor version may
change the public API (the names exported from `agent_trajviz`).

## [Unreleased]

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
  `agent_trajviz.attribution` is DECAF attribution.
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

[0.1.0]: https://github.com/rshu/TrajViz/releases/tag/v0.1.0
