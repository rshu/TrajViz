# Task guides

The examples use `capture.jsonl` and `capture-2.jsonl` as two native runs of the
same task. Replace them with your session files. See the
[quickstart](quickstart.md) for installation and first exports.

## Choose a captured session

[`list_sessions`](api/agent_trajviz.html#list_sessions) discovers sessions in
saved Claude Code OpenTelemetry (OTLP) captures. Load native session files and
supported subagent bundles directly with `load`.

For a saved capture bundle containing multiple sessions, inspect the IDs and
choose one explicitly:

<!-- trajviz:run -->
```python
import agent_trajviz as tv

sessions = tv.list_sessions("sessions.zip")
for session in sessions:
    print(session["id"])

# Replace docs-first with the ID of the run you want to analyze.
selected = tv.load("sessions.zip", session_id="docs-first")
```

Loading an ambiguous capture without a selection raises
`LoadError(code="ambiguous_session")`. In the dashboard, choose the ID under
**Captured session**. From the CLI, list the IDs and pass the one you want to
`--session-id`.

```bash
agent-trajviz --list-sessions sessions.zip
agent-trajviz --json sessions.zip --session-id docs-first -o analysis/
```

## Compare runs of the same task

[`compare`](api/agent_trajviz.html#compare) aligns two trajectories and returns a
comparison report. Convert it to the same view objects used elsewhere in the
API:

<!-- trajviz:run -->
```python
reference = tv.load("capture.jsonl")
candidate = tv.load("capture-2.jsonl")
report = tv.compare(reference, candidate)
comparison = tv.comparison_views(report)
tv.export_json(comparison, "comparison")
```

Pass `anchor_patch="expected.patch"` when you have a unified reference diff.
`fuzzy_commands=True` enables fuzzy command matching. Inspect the report's
confidence and evidence alongside its scores: unknown tool outcomes do not
establish successful execution or surviving edits. Transcript comparison is
not supported for saved OTLP captures.

For several runs, the first is the baseline:

<!-- trajviz:run -->
```python
group = tv.compare_runs(
    [reference, candidate], labels=["baseline", "candidate"]
)
tv.export_json(tv.run_group_views(group), "run-group")
```

Group results include scorecard rows, behavioral comparisons and timelines.
Inspect error rows when a source could not be loaded.

## Analyze label sidecars

Use a sidecar produced by `agent-trajviz-label`:

<!-- trajviz:run -->
```python
labels = tv.label_views("labels.json")
print(labels["labels_summary"].data)
tv.export_json(labels, "labels-analysis")
```

`tv.label_comparison_views("baseline_labeled.json", "candidate_labeled.json")`
compares phase counts and durations. Labels classified as `unknown` remain in
the summary but are excluded from phase/action distributions and the timeline.
These functions analyze saved labels; they do not call a model.

## Analyze steps you already parsed

Use [`Trajectory.from_steps`](api/agent_trajviz.html#Trajectory.from_steps) for
partial custom records. `from_dict` instead expects TrajViz's own exported
trajectory layout.

<!-- trajviz:run -->
```python
custom = tv.Trajectory.from_steps([
    {"role": "user", "text_preview": "Explain this function"},
    {"role": "assistant", "text_preview": "The function validates an input."},
])
print(custom.view("summary").data)
```

The input records are copied and normalized. Supply the token, timing and tool
fields you actually have; see [`Step`](api/agent_trajviz.html#Step),
[`TokenUsage`](api/agent_trajviz.html#TokenUsage) and
[`ToolCall`](api/agent_trajviz.html#ToolCall) for field definitions. Treat the
resulting `steps` as read-only.

## Register a view

A view builder receives a trajectory and returns JSON-compatible data. Give it
a schema for its `data` member:

<!-- trajviz:run -->
```python
def count_assistant_steps(run):
    return {"count": sum(step["role"] == "assistant" for step in run.steps)}

tv.register_view(
    "assistant_step_count",
    count_assistant_steps,
    title="Assistant steps",
    schema={
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 0}},
        "required": ["count"],
    },
)
assert custom.view("assistant_step_count").data == {"count": 1}
```

Registered views join `VIEW_CATALOG`, trajectory exports and dashboard view
lists. Declare accepted view options with `options=`; callers pass their values
to `run.view`. Without a supplied schema, the data portion is unconstrained.
A custom chart also needs a renderer registered through
[`register_figure`](api/agent_trajviz/figures.html#register_figure).

## Register a format or package a plugin

A [`Format`](api/agent_trajviz.html#Format) parses a JSON document or a list of
JSONL records into steps:

<!-- trajviz:run -->
```python
import json
from pathlib import Path

def parse_example(document):
    return [
        {"role": turn["role"], "text_preview": turn["text"]}
        for turn in document["turns"]
    ]

tv.register_format(tv.Format(
    key="example_agent",
    label="Example Agent",
    parse=parse_example,
    detect=lambda doc: isinstance(doc, dict) and doc.get("kind") == "example-agent",
))
Path("example-agent.json").write_text(json.dumps({
    "kind": "example-agent",
    "turns": [{"role": "assistant", "text": "Finished the example."}],
}), encoding="utf-8")
assert tv.load("example-agent.json").format == "example_agent"
```

Custom detection runs after built-in formats. Omit `detect` to require an
explicit `format="example_agent"` when loading.

To distribute registrations, expose a no-argument hook in a package:

```toml
[project.entry-points."agent_trajviz.plugins"]
example_agent = "example_agent_plugin:register"
```

The hook calls the public registration functions. Library users explicitly
call [`load_plugins`](api/agent_trajviz.html#load_plugins); the `agent-trajviz`
command runs installed hooks at startup. Hooks execute Python code from those
installed packages.

## Use model assistance from Python

The [`llm` module](api/agent_trajviz/llm.html) requires the `llm` extra. Configure
`ANALYZE_*` environment variables, or read a local `.env` explicitly:

```python
from agent_trajviz import llm

config = llm.config_from_env(load_dotenv=True)
history = llm.ask(reference, "Which steps are the main bottlenecks?", config=config)
```

Calling `ask` sends the question and statistics brief to the configured
provider after redaction. Use `chat=` for your own client. If your provider
requires extra gateway headers, set `ANALYZE_HEADERS` to a JSON object in the
environment. TrajViz uses those headers only when the base URL and provider
match the configured `ANALYZE_BASE_URL` and `ANALYZE_PROVIDER`.

Dashboard Settings apply to the current browser session. Set the Python API's
connection separately with `config=` or the environment.
