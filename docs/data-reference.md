# Data reference

## Trajectories and views

A trajectory is the normalized source for analysis. Its `steps` are plain
mappings described by [`Step`](api/agent_trajviz.html#Step), with token and tool
records described by [`TokenUsage`](api/agent_trajviz.html#TokenUsage) and
[`ToolCall`](api/agent_trajviz.html#ToolCall). Preserve the supplied agent and
parent relationships; an agent is not necessarily identified by a unique
`session_id`.

A view is a JSON envelope containing `schema_version`, `id`, `title`, `kind`,
`description` and `data`, plus `availability` when supplied. `kind` is `summary`,
`table` or `chart`. Optional plotting helpers render this data rather than
recomputing metrics.

The [generated catalog](data/views.html) lists views, their accepted options and
schemas. `VIEW_CATALOG` covers single-run views; `view_schemas()` also includes
comparison, label and run-group views.

<!-- trajviz:run -->
```python
import agent_trajviz as tv

for view_id, info in tv.VIEW_CATALOG.items():
    print(view_id, info["kind"], info["options"])

schemas = tv.view_schemas()
assert "token_usage" in schemas
```

## Availability and missing values

Inspect both the values and their evidence. Missing token counts, timing or
outcomes must not be converted into measured zeros or successful outcomes.

View availability can be `observed`, `partial`, `unavailable` or `unknown`.
`observed` describes the fields available among captured operations; it does
not prove that the entire run was captured. A partial number can cover only a
subset of the run. An unavailable value may be `None` in Python and `null` in
JSON; some analyses instead return no rows or an unavailable verdict.

Native formats use source-record semantics and can report `unknown`
availability without detailed field-coverage metadata. That status alone does
not mean every displayed native measurement is absent. Read loader warnings,
`tokens_estimated`, individual values and the data-quality view together.

<!-- trajviz:run -->
```python
run = tv.load("capture.jsonl")
print(run.warnings)
print(run.tokens_estimated)
print(run.view("data_quality").data)
print(run.view("token_usage").availability)
```

Some views have no `availability` member. Its absence is not a completeness
claim. Keep null values and evidence metadata when exporting data or building
another UI.

## Units and token accounting

Normalized step `duration` is in seconds; tool `duration_ms` and timestamp fields
ending in `_ms` use milliseconds. A step's `timing_source` distinguishes an
enclosing interval from model generation time where that information is known.
Elapsed time can include tool waits and idle gaps, so it is not interchangeable
with model generation time.

Use reported normalized token totals instead of adding provider fields blindly.
For example, some sources count cache reads inside input or reasoning inside
output. TrajViz accounts for those overlaps; adding the raw components again
can double-count usage. Estimated tokens remain estimates.

The [`token_usage`](data/views.html#token_usage) view separates fresh input,
cache reads, cache writes, output, and reasoning when the source provides a
breakdown. A source can also report only a total. The
[`hotspots`](data/views.html#hotspots) view ranks context growth by comparing
consecutive observed prompts in the same agent session. Growth measures the
change in prompt size and can include the previous reply, tool results, and
new messages.

## Schema versions

Trajectory exports and view documents have separate schema versions:

| Document | Version constant | Purpose |
| --- | --- | --- |
| `Trajectory.to_dict()` / `trajectory.json` | `SCHEMA_VERSION` | Reloadable normalized trajectory |
| `View.to_dict()` / each view JSON file | `VIEWS_SCHEMA_VERSION` | Chart/table data envelope and its schema |

These are data-format versions, independent of the package version. They are
currently 1 and 2 respectively. View schema version 2 permits
unavailable measurements and verdicts where each view's schema specifies them.

[`view_schema`](api/agent_trajviz.html#view_schema) returns a JSON Schema draft
2020-12 document for a complete view envelope. Validate `view.to_dict()`, not
just `view.data`. For example, with the separate `jsonschema` package installed:

```python
import jsonschema

view = run.view("token_usage")
jsonschema.validate(view.to_dict(), tv.view_schema(view.id))
```

Schemas allow additional record fields for forward-compatible consumers. Use
the schema shipped with the package version that produced the data and tolerate
additional keys. Custom views use the schema supplied at registration; without
one, their `data` is unrestricted.

## Export and reload boundaries

Use `run.to_dict(include_raw=True)` or `export_json(run, directory)` when the
reloaded run must retain raw-dependent analyses. A lightweight `to_dict()` keeps
normalized steps but can omit information used by session, timing or subagent
analyses. Do not assume the view envelope and trajectory export are
interchangeable.

`export_json` replaces each output file atomically, not the directory as one
transaction. A view mapping produces view files and an index; only exporting a
trajectory also produces `trajectory.json`. See the
[quickstart](quickstart.md) for a checked round trip.
