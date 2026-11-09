# Examples

Both apps use the public `agent_trajviz` API. Run the commands below from the
repository root after installing `agent-trajviz[app]`. With `uv`, run
`uv sync --locked --extra app` and replace `python` with `uv run --locked --extra app python`.

| Example | Purpose | Run |
|---|---|---|
| [`trajviz_app/app.py`](trajviz_app/app.py) | A dashboard with workflow, comparisons, HTML reports and optional model assistance | `python examples/trajviz_app/app.py` → http://127.0.0.1:7870 |
| [`views_app.py`](views_app.py) | Browse views, export JSON and compare trajectories | `python examples/views_app.py` → http://127.0.0.1:7871 |

The dashboard example reads model settings from `ANALYZE_*` environment
variables, with `LABEL_*` as a fallback. Opening its **AI Trajectory Analysis**
sidebar for a loaded run with a configured model starts the first request and
sends relevant trajectory excerpts to the provider.

[`test_replica_app.py`](../tests/test_replica_app.py) and
[`test_example_app.py`](../tests/test_example_app.py) exercise the apps over HTTP.
