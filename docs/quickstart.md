# Quickstart

## Install

Python 3.11 or newer is required. Install the core for parsing and JSON analysis:

```bash
pip install agent-trajviz
```

Add `charts` for figures and HTML reports, or `app` for the dashboard and model
features:

```bash
pip install "agent-trajviz[charts]"
# Or: pip install "agent-trajviz[app]"
```

For the dashboards and CLI, Homebrew includes Python and the app dependencies:

```bash
brew install rshu/tap/trajviz
```

Use pip in your project environment for the Python examples below; Homebrew
keeps the application in an isolated environment.

## Load a session

Use a native session export of your own in place of `capture.jsonl`. TrajViz
identifies formats from their contents; `tv.FORMATS` lists the supported format
IDs. A supported ZIP bundle can retain a parent session and its subagent logs.

<!-- trajviz:run -->
```python
import agent_trajviz as tv

run = tv.load("capture.jsonl")
print(run.format_label, len(run.steps))
for warning in run.warnings:
    print(warning)
```

If you require a particular format, pass its ID, such as
`tv.load("capture.jsonl", format="claude_code")`. A detected format mismatch is
an error. See [`load`](api/agent_trajviz.html#load) for supported source types and
[`LoadError`](api/agent_trajviz.html#LoadError) for machine-readable error codes.

## Inspect JSON views

A view contains the data behind a chart or table. You do not need Plotly or
Gradio to inspect it.

<!-- trajviz:run -->
```python
summary = run.view("summary")
print(summary.to_json(indent=2))

tokens = run.view("token_usage")
print(tokens.data)
print(tokens.availability)
```

Choose view IDs and options from the [view catalog](data/views.html). Check
[availability and missing values](data-reference.md) before treating an absent
measurement as zero.

## Export and reload

[`export_json`](api/agent_trajviz.html#export_json) writes each selected view,
`index.json`, and a reloadable `trajectory.json`. Existing files with those
names are replaced; unrelated files are preserved. Exports refuse to overwrite
consumed source files, including child logs, even through filesystem aliases.

<!-- trajviz:run -->
```python
written = tv.export_json(run, "analysis", only=["summary", "token_usage"])
restored = tv.load("analysis/trajectory.json")
assert restored.view("token_usage").data == tokens.data
```

The trajectory export retains source content and provenance. Review those
contents when choosing what to share. For an in-memory round trip, use
`tv.Trajectory.from_dict(run.to_dict(include_raw=True))`.

## Charts and offline reports

These functions require the `charts` extra. Embedding Plotly makes the resulting
HTML usable without fetching its JavaScript from a CDN.

<!-- trajviz:run extras=charts -->
```python
from agent_trajviz.figures import figure

fig = figure(tokens)
fig.write_html("tokens.html", include_plotlyjs=True)
report_path = tv.write_report(run, "report.html", inline_plotly=True)
```

## Open the dashboard

With the `app` extra installed through pip, or with Homebrew, run
`agent-trajviz` and open the local URL it prints. Load the same session there.
Open **AI Trajectory Analysis → Settings** to configure the model
connection and request options. The **Reply language** dropdown selects English
or Simplified Chinese for the next response; **EN | 中** updates its default.
Use **Analyze trajectory** on
the Analysis tab when you are ready. **Test API key** checks the entered connection
with a small model request without applying settings or sending trajectory data. Connection details can be
remembered in the browser; entered API keys remain in that session. Applying
settings sends no model request.

Continue with the [task guides](guides.md), or browse the
[Python reference](api/agent_trajviz.html).
