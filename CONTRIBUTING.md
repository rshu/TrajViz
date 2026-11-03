# Contributing

Thanks for helping. Bug reports with the smallest trajectory that shows the
problem are as welcome as code; security issues go through
[SECURITY.md](https://github.com/rshu/TrajViz/blob/main/SECURITY.md), not
public issues.

## Set up and check

```bash
uv sync --extra dev                                  # locked environment, every extra
uv run pytest                                        # the suite
uv run ruff check agent_trajviz tests examples       # lint
uv run --with mypy mypy --strict agent_trajviz/__init__.py agent_trajviz/api.py \
    agent_trajviz/errors.py agent_trajviz/model.py agent_trajviz/views.py agent_trajviz/plugins.py
```

CI runs the same commands on Python 3.11 to 3.14, with the oldest allowed
dependencies, on macOS, and from the built wheel on Windows. The bundled evaluation
engine and its synthetic regression tests run without an external checkout.

## Ground rules

- **Fixtures are synthetic or scrubbed.** A trajectory records whatever the
  agent saw: never commit one with real paths, user names, hostnames,
  credentials or private source. The sdist ships no tests.
- **The public API is pinned.** `tests/test_public_api.py` lists every name
  in `agent_trajviz.__all__` and their signatures, so changing the API is a
  deliberate edit of that test and a changelog entry. Before 1.0 a minor
  version may change the API; say so under "Changed".
- **Charts draw from views.** Every chart and table is a view
  (`agent_trajviz/views.py`) and the dashboard draws it with
  `agent_trajviz.figures`, so the JSON and the picture cannot disagree. A new
  chart is a new view first.
- **A view's data has a schema.** `agent_trajviz/schemas/<id>.schema.json`
  describes each view, and the suite validates every view strictly: a new
  field, a changed type or a removed key fails with the path to fix. Update
  the schema in the same change. Bump `VIEWS_SCHEMA_VERSION` only for an
  incompatible change (a field removed or renamed, or its type changed).
- **Text I/O names its encoding** (`encoding="utf-8"`); a test enforces it,
  because the platform default breaks UTF-8 files on Windows.
- **Nothing leaves the machine unasked.** The core makes no network calls;
  LLM features run only on an explicit action and every request is redacted.

## Releasing

1. Set `__version__` in `agent_trajviz/__init__.py` and date the version's
   section in `CHANGELOG.md`.
2. Merge to `main` with CI green, then tag the merge commit `vX.Y.Z` and push
   the tag.
3. `.github/workflows/release.yml` checks the tag against `__version__`, runs
   the suite, builds and checks the distributions, and publishes to PyPI
   through Trusted Publishing once the `pypi` environment is approved.
4. Verify that version is available on PyPI, then publish the matching GitHub
   release `vX.Y.Z` with focused release notes. Check that the Documentation
   workflow retains its versioned archive and updates the stable site.
5. Run the [Homebrew Update formula workflow](https://github.com/rshu/homebrew-tap/actions/workflows/update.yml)
   with the same `X.Y.Z`. Review its prepared branch and passing macOS/Linux
   checks, then merge it into the tap's `main`. Confirm the formula reports the
   same upstream version.

## Documentation

Keep public API descriptions in docstrings and task guides in the maintained
Markdown files under `docs/`. Local notes in that directory remain ignored.
Generated HTML is built in `site/` and is never committed.

```bash
uv run --locked --extra dev --with pdoc==16.0.0 python .github/scripts/build_docs.py \
    --package-root . --output site --version-label development --check-examples
```

CI builds the site and runs marked examples on pull requests. A successful
`main` build publishes development documentation; the latest published package
provides the stable API reference. Release documentation is retained as a ZIP
asset on its GitHub release and served at its versioned URL, starting with
0.3.1. The documentation
workflow can rebuild the stable snapshot explicitly after a documentation fix.
