# TrajViz API documentation

TrajViz turns coding-agent sessions into normalized steps, analysis results and
JSON views. The Python API works without a running dashboard; the core package
uses only the standard library.

Start with the [quickstart](quickstart.md) to load a session, inspect a view and
save an analysis. The [task guides](guides.md) cover session selection,
comparisons, labels and extensions. Read the [data reference](data-reference.md)
before interpreting missing measurements or consuming exported JSON.

## Reference

- [Core Python API](api/agent_trajviz.html): loading, trajectories, views,
  comparisons, exports and plugins.
- [View catalog](data/views.html): view IDs, options and JSON Schemas.
- [Figures](api/agent_trajviz/figures.html): Plotly charts from the same view data.
- [Dashboards](api/agent_trajviz/dashboard.html): launch TrajViz or embed its views
  in a Gradio application.
- [Model assistance](api/agent_trajviz/llm.html): questions and suggested fixes,
  with an explicit model connection or your own client.
- [Failure attribution](api/agent_trajviz/attribution.html): check availability
  and diagnose a run against reference data.
- [AWE guide](awe.md) and [API](api/agent_trajviz/awe.html): inspect a loaded
  trajectory, import saved analyses, export reports, and run built-in evaluation.

Use the released documentation for an installed PyPI version. Development
reference pages can describe changes that have not been released.

## The API's main objects

A [`Trajectory`](api/agent_trajviz.html#Trajectory) holds the normalized steps,
source information and loader warnings for one run. Treat its steps as read-only:
analysis is cached for reuse.

A [`View`](api/agent_trajviz.html#View) holds one chart's or table's data. Its JSON
is useful on its own; optional figure and dashboard helpers draw from that data.
The [data reference](data-reference.md) explains the view envelope, schema
versions and evidence availability.

The supported core names are exported by `agent_trajviz.__all__`. Use these names
and the optional modules linked above; implementation modules are not a stable
extension interface.
