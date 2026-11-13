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

On Python 3.11–3.13, install the `zstd` extra to read compressed DeepSeek
Harness sessions. Python 3.14 and newer can read them with the core package.

Homebrew includes Python and the app dependencies for the dashboards and CLI.
The first two commands are one-time setup. The trust step applies to Homebrew 6
and later.

```bash
brew tap rshu/tap
brew trust --formula rshu/tap/trajviz
brew install trajviz
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
`agent-trajviz` and open the local URL it prints. Upload the session and select
**Load Trajectory**. Use **EN | 中** in the header to switch between English and
Simplified Chinese. The browser remembers the interface language.

In **Overview**, follow issue links to their evidence in **Workflow**, or select
**Export issues into JSON** to save the findings. Under **Summary**, click a bar
in **Token Usage** to inspect its recorded token breakdown. The **Execution
Timeline** skips idle gaps by default. Choose **True wall-clock** to include
the elapsed gaps.

Loading a trajectory computes the dashboard locally. To request model assistance,
open **AI Trajectory Analysis → Settings**, enter the provider details, and select
**Apply settings**. **Test API key** is an optional connection check. It sends a
small model request with no trajectory data and leaves the current settings in
place. The browser can remember connection details, and entered keys remain in
the session. A masked key indicates an active key for the selected endpoint.

On the **Analysis** tab, select **Analyze trajectory** or submit a question.
In **Overview**, **Analyze issue with AI** focuses on one issue, while **Suggest
fixes with LLM** summarizes the run and suggests fixes. These actions send
relevant trajectory excerpts to the configured provider. Under **Settings →
Request options → Reply language**, choose **English** or **简体中文** for the
next response. Switching **EN | 中** updates the reply language choice.

Continue with the [task guides](guides.md), or browse the
[Python reference](api/agent_trajviz.html).
