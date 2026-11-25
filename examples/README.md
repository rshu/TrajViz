# Examples

Both apps use the public `agent_trajviz` API. Run the commands below from the
repository root after installing `agent-trajviz[app]`. With `uv`, run
`uv sync --locked --extra app` and replace `python` with `uv run --locked --extra app python`.

| Example | Purpose | Run |
|---|---|---|
| [`trajviz_app/app.py`](trajviz_app/app.py) | A dashboard with workflow, comparisons, HTML reports and optional model assistance | `python examples/trajviz_app/app.py` → http://127.0.0.1:7870 |
| [`views_app.py`](views_app.py) | Browse views, export JSON and compare trajectories | `python examples/views_app.py` → http://127.0.0.1:7871 |

Both examples bind to `127.0.0.1` by default and launch Gradio directly. The
packaged dashboards' `TRAJVIZ_AUTH` login, `--max-file-size` upload cap and
exposed-launch server-path restrictions do not apply to these examples. The
dashboard example accepts `--port` and `--host`; changing `--host` only changes
the bind address, and its reference-data path input remains available. Keep
the examples on loopback, or add deployment controls when adapting them.

The dashboard example reads model settings from `ANALYZE_*` environment
variables, with `LABEL_*` as a fallback; it does not load `.env` automatically.
Opening its **AI Trajectory Analysis** sidebar with a loaded run, an empty chat
and a configured model automatically sends the first request and relevant
trajectory excerpts to the provider. Submitting a question or selecting
**Suggest fixes with LLM** also sends model requests. In the packaged Insight
dashboard, opening the sidebar alone stays local; **Analyze trajectory** starts
the first request.

[`test_replica_app.py`](../tests/test_replica_app.py) and
[`test_example_app.py`](../tests/test_example_app.py) exercise the apps over HTTP.
