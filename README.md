# TrajViz

[![CI](https://github.com/rshu/TrajViz/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rshu/TrajViz/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-trajviz)](https://pypi.org/project/agent-trajviz/)
[![Homebrew](https://raw.githubusercontent.com/rshu/TrajViz/main/.github/homebrew-badge.svg)](https://github.com/rshu/homebrew-tap#readme)
[![Python](https://img.shields.io/pypi/pyversions/agent-trajviz)](https://pypi.org/project/agent-trajviz/)
[![Downloads](https://static.pepy.tech/badge/agent-trajviz)](https://pepy.tech/projects/agent-trajviz)
[![License](https://img.shields.io/pypi/l/agent-trajviz)](https://github.com/rshu/TrajViz/blob/main/LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/rshu/TrajViz/blob/main/CONTRIBUTING.md)

**Analyze coding-agent trajectories as a Python library, JSON exports or an interactive dashboard.**

Inspect tokens, timing, tool calls, subagents and failure patterns across native
agent sessions. TrajViz also imports saved Claude Code OpenTelemetry (OTLP)
captures. The core uses only the Python standard library.

[Documentation](https://rshu.github.io/TrajViz/)

## Install

Requires Python 3.11+.

```bash
pip install "agent-trajviz[app]"   # dashboards, figures, reports and LLM features
```

For the library and collection tools alone: `pip install agent-trajviz`.

[Homebrew](https://github.com/rshu/homebrew-tap): `brew install rshu/tap/trajviz`
includes the dashboards and Python runtime.

| Extra | Features |
|---|---|
| `charts` | Plotly figures and HTML reports |
| `llm` | AI analysis and step labeling |
| `app` | Dashboards, `charts` and `llm` |
| `zstd` | Compressed native logs on Python 3.11–3.13; built in on Python 3.14+ |
| `all` | All optional features |

## Quickstart

### Python

```python
import agent_trajviz as tv

run = tv.load("trajectory.json")
print(run.format_label, run.metrics["tokens"]["total"])
print(run.view("tool_usage").data)
tv.export_json(run, "out/")
```

Formats are detected from file contents. `run.views()` exposes the public chart
and table data as JSON. `tv.view_schema("tool_usage")` returns its JSON Schema.
Reload `out/trajectory.json` with `tv.load()` to continue an analysis.

View exports use **schema version 2**: unavailable measurements can be `null`,
health checks can be `unavailable`, and `availability` describes their evidence.
The `data_quality` view explains missing or partial observations. Normalized
trajectory exports remain schema version 1 and existing exports stay readable.

Use `tv.compare()` or `tv.compare_runs()` for run comparisons. Optional Plotly
figures are available through `agent_trajviz.figures`; custom formats and views
can be registered through the public API. See the
[examples](https://github.com/rshu/TrajViz/tree/main/examples) for applications
built from these interfaces.

### Dashboard and CLI

```bash
agent-trajviz
agent-trajviz --json trajectory.json -o out/
agent-trajviz --report trajectory.json -o report.html --inline-plotly
```

Open `http://127.0.0.1:7860`, upload a `.json`, `.jsonl`, `.zstd` or `.zip` file,
and select **Load Trajectory**. Compressed DeepSeek Harness logs need the
`zstd` extra on Python 3.11–3.13. **Export HTML** saves an Overview, charts and
Patterns snapshot that loads Plotly from a CDN. Use `--inline-plotly` with
`--report` to embed the chart runtime for offline viewing.

Use **EN | 中** in the header to switch between English and Simplified Chinese. The
browser remembers the choice and sets the default AI reply language. In
**AI Trajectory Analysis → Settings → Request options**, choose **English** or
**简体中文** under **Reply language** for the next analysis, answer or suggested
fix.

Use `--port 8080` to change ports. Set `TRAJVIZ_AUTH=user:password` before
exposing the dashboard with `--host 0.0.0.0` or `--share`; remote visitors can
use the configured LLM features. For localhost proxy errors, include
`127.0.0.1,localhost` in `no_proxy`.

The Converge comparison dashboard runs on port 7861:

```bash
python -m agent_trajviz.converge --data-root /path/to/trajectories
```

With Homebrew, use `agent-trajviz-converge-app --data-root /path/to/trajectories`.

Batch manifest paths must remain inside `--data-root`. For headless comparisons,
use `agent-trajviz-converge --help`. Every command provides `--help`.

## Supported inputs

| Agent | Input |
|---|---|
| Claude Code | Native session JSONL under `~/.claude/projects/`, including associated subagent logs |
| Cursor | CLI `stream-json`, or a consolidated Agent transcript family with Composer metadata |
| OpenCode | Native v1/v2, desktop and Windows session JSON, including parent/child families from the collector |
| Codex CLI | Persisted `rollout-*.jsonl` under `~/.codex/sessions/`, or a collected session family |
| Pi | Native JSONL under `~/.pi/agent/sessions/`, using the active history branch |
| DeepSeek Harness | Native session directories, `session.jsonl`, `session.vN.jsonl[.zstd]`, or legacy session exports |
| CodeArts | AgentKernel session exports from its `opencode.db` store |
| CodeAgent | Session JSON exports with `session` + `child_sessions` (data-wrapped messages/parts) |
| ICode | Chrys expanded-session JSON with nested subagent sessions |

Session files with a UTF-16 byte order mark are supported. For OpenTelemetry
files, see [Saved OTLP captures](https://github.com/rshu/TrajViz#saved-otlp-captures).

### Collect sessions and subagents

Read-only collectors turn selected local session families into portable JSON:

```bash
agent-trajviz-collect cursor CHAT_ID cursor.json
agent-trajviz-collect opencode SESSION_ID opencode.json
agent-trajviz-collect codex SESSION_ID -o codex.json
agent-trajviz-collect codearts /path/to/opencode.db --session-id SESSION_ID --output codearts.json
```

Use `agent-trajviz-collect TOOL --help` for each collector's options. Set
`OPENCODE_DATABASE` if the OpenCode database is outside its default location.
OpenCode also provides root-session exports:

```bash
opencode session export SESSION_ID > opencode.json  # OpenCode 2
opencode export SESSION_ID > opencode.json          # OpenCode 1
```

Claude Code needs no exporter. Loading its parent JSONL locally discovers
associated children. For upload, ZIP `SESSION_ID.jsonl` together with
`SESSION_ID/subagents/`, preserving their relative paths. An uploaded parent
file alone cannot transfer child logs from your computer.

For Cursor CLI captures, save the stream while running a task:

```bash
agent -p --output-format stream-json 'Implement the requested feature and run its tests' > cursor.jsonl
```

Cursor token counts are estimates, not billed usage. Native transcript families
may lack tool outcomes, outputs and clocks without Composer metadata; the CLI
stream can supply parent tool results and execution clocks.

For Codex, use the persisted rollout rather than `codex exec --json` stdout;
avoid `--ephemeral` when collecting. For DeepSeek Harness, loading a session
directory selects the highest numbered generation and includes available child
logs. To make a locally loaded family portable, export it and upload
`out/trajectory.json`:

```bash
agent-trajviz --json /path/to/session-directory -o out/
```

## Saved OTLP captures

TrajViz imports **Claude Code** OTLP JSON traces and Collector JSONL files,
with optional companion event logs. Native Claude Code JSONL remains supported
and provides richer transcript content. Importing saved files requires no
telemetry SDK or live receiver.

```python
run = tv.load("traces.jsonl")
sessions = tv.list_sessions("capture/")
run = tv.load("capture/", session_id="SESSION_ID")
```

Captures containing multiple sessions require an explicit selection; the
dashboard provides a session selector. The CLI supports the same choice:

```bash
agent-trajviz --list-sessions capture/
agent-trajviz --json capture/ --session-id SESSION_ID -o out/
```

For rotated traces or companion logs, place an explicit `manifest.json` in the
capture directory. The directory can also be zipped for upload:

```json
{
  "type": "agent-trajviz-otlp-bundle",
  "schema_version": 1,
  "profile": "claude_code",
  "traces": ["traces.jsonl"],
  "logs": ["logs.jsonl"]
}
```

Paths must stay within the bundle. A standalone trace file is sufficient; logs
only enrich it. Logs or metrics alone, binary protobuf and compressed Collector
framing are unsupported. Other OTLP producers need a mapping profile.

Totals describe captured operations, with missing or redacted evidence marked
unavailable. Native and OTLP inputs are not merged; transcript comparisons and
failure attribution are unavailable for OTLP captures.

<details>
<summary>Record a new Claude Code OTLP capture</summary>

To record a new capture, install OpenTelemetry Collector Contrib with its file
exporter. In a separate capture directory, save the manifest above as
`capture/manifest.json` and this loopback-only configuration as `otelcol.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
processors:
  batch: {}
exporters:
  file/traces:
    path: ./capture/traces.jsonl
    format: json
  file/logs:
    path: ./capture/logs.jsonl
    format: json
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [file/traces]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [file/logs]
```

```bash
mkdir -p capture
otelcol-contrib --config otelcol.yaml
```

Then run Claude Code in a separate terminal with its beta tracing enabled:

```bash
CLAUDE_CODE_ENABLE_TELEMETRY=1 \
CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1 \
OTEL_TRACES_EXPORTER=otlp \
OTEL_LOGS_EXPORTER=otlp \
OTEL_METRICS_EXPORTER=none \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
claude
```

Exit Claude Code, then stop the Collector gracefully so buffered records are
flushed before loading `capture/`. Use a new output directory for another
recording: these file exporters truncate existing files at startup.
Tracing requires a Claude Code version that supports the beta span schema;
see [Claude Code monitoring](https://code.claude.com/docs/en/monitoring-usage#traces-beta)
and the [Collector file exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter/fileexporter).

This configuration retains Claude's default content redaction. If you want prompt,
response, tool arguments or tool output in the capture, enable the appropriate
`OTEL_LOG_USER_PROMPTS`, `OTEL_LOG_ASSISTANT_RESPONSES`, `OTEL_LOG_TOOL_DETAILS`
and `OTEL_LOG_TOOL_CONTENT` settings deliberately; these can record sensitive
code and command output. Missing, redacted and truncated content stays marked
as such in TrajViz. No content logging is required for timing and observed
usage analyses.

</details>

## Optional LLM features

Trajectory loading and opening the **AI Trajectory Analysis** sidebar run
locally. Select **Analyze trajectory**, submit a question, select **Analyze
issue with AI** on an Overview issue, or select **Suggest fixes with LLM** to
send relevant trajectory excerpts to the configured provider. **Export issues
into JSON** saves the ranked findings and their supporting evidence locally.

Open **AI Trajectory Analysis → Settings** to configure the provider, API key,
base URL, model and request options. **Test API key** sends a small model
request without trajectory data. Select **Apply settings** to use the entered
connection. The form masks the active API key, which stays in the current
server-side session. Only non-secret connection details can be remembered in
the browser. You can also set `ANALYZE_BASE_URL`,
`ANALYZE_API_KEY` and `ANALYZE_MODEL` in `.env`. See
[.env.example](https://github.com/rshu/TrajViz/blob/main/.env.example) for provider,
language, timeout, and optional gateway headers. Headers apply only to the
configured base URL and provider. Reasoning models spend output tokens on hidden
thinking; if a reply comes back empty or cut off, raise Max output tokens
(default 8192). Anthropic-style endpoints accept the base URL with or without
`/v1`.

For step labeling, set `LABEL_BASE_URL`, `LABEL_API_KEY` and `LABEL_MODEL`, then:

```bash
agent-trajviz-label trajectory.json --output labels.json
```

Upload the label sidecar alongside its trajectory and select **Load Labels** to
enable phase-aware views.

## AWE inspection and evaluation

The **AWE** tab inspects the loaded trajectory locally, with evidence linked to
Workflow steps. Optional failure attribution uses the bundled evaluation engine
and canonical reference data. **Saved runs & evaluation** provides provenance
graphs, symptoms, recommendations and reports. Evaluation settings configure
explicit operations and optional model connections. See the
[AWE guide](https://github.com/rshu/TrajViz/blob/main/docs/awe.md) for the Python
API and CLI.

## Development

```bash
git clone https://github.com/rshu/TrajViz.git
cd TrajViz
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked ruff check agent_trajviz tests examples
```

See [CONTRIBUTING.md](https://github.com/rshu/TrajViz/blob/main/CONTRIBUTING.md)
for API conventions, schema checks and releases, and the
[architecture diagram](https://raw.githubusercontent.com/rshu/TrajViz/main/architecture.svg)
for the parsing and analysis layers. Changes are recorded in
[CHANGELOG.md](https://github.com/rshu/TrajViz/blob/main/CHANGELOG.md).

MIT licensed — see [LICENSE](https://github.com/rshu/TrajViz/blob/main/LICENSE).
Report vulnerabilities using
[SECURITY.md](https://github.com/rshu/TrajViz/blob/main/SECURITY.md).
