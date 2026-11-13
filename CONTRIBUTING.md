# Contributing

Thanks for helping. For bug reports, include the smallest trajectory that
shows the problem. Report vulnerabilities privately as described in
[SECURITY.md](https://github.com/rshu/TrajViz/blob/main/SECURITY.md).

## Set up and check

```bash
uv sync --locked --extra dev
uv run --locked python -m pytest tests -q
uv run --locked ruff check agent_trajviz tests examples .github/scripts
uv run --locked --with 'mypy>=1.13' mypy --strict agent_trajviz/__init__.py agent_trajviz/api.py \
    agent_trajviz/errors.py agent_trajviz/model.py agent_trajviz/views.py agent_trajviz/plugins.py
```

CI runs the test suite on Python 3.11 through 3.14 and on macOS. Separate
jobs test the minimum direct dependencies and smoke-test the installed wheel
on Windows. The bundled evaluation engine and its synthetic regression tests
run without an external checkout.

## Ground rules

- **Fixtures are synthetic or scrubbed.** A trajectory records whatever the
  agent saw: never commit one with real paths, user names, hostnames,
  credentials or private source. The sdist ships no tests.
- **The public API is pinned.** `tests/test_public_api.py` lists every name
  in `agent_trajviz.__all__` and their signatures, so changing the API is a
  deliberate edit of that test and a changelog entry. Before 1.0 a minor
  version may change the API; say so under "Changed".
- **Share data between charts and exports.** Define reusable chart and table
  data in `agent_trajviz/views.py`. Use `agent_trajviz.figures` for the public
  figure adapters and keep dashboard rendering based on the same data.
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
2. Land the version update on `main` and wait for CI on that commit to pass.
   Tag that commit `vX.Y.Z` and push the tag.
3. `.github/workflows/release.yml` checks the tag against `__version__`, runs
   the suite, builds and checks the distributions, and publishes to PyPI
   through Trusted Publishing once the `pypi` environment is approved.
4. Verify the PyPI wheel and source archive, then publish the matching GitHub
   release `vX.Y.Z` with focused notes and the same tested distributions. Check
   that the Documentation workflow publishes the stable site and archives its
   versioned snapshot on the release.
5. Run the [Homebrew Update formula workflow](https://github.com/rshu/homebrew-trajviz/actions/workflows/update.yml)
   with the same `X.Y.Z`. Follow the [tap's release steps](https://github.com/rshu/homebrew-trajviz#maintaining-the-formula)
   to test and publish bottles, then confirm the formula reports the same version.

## Documentation

Keep public API descriptions in docstrings and task guides in the maintained
Markdown files under `docs/`. Local notes in that directory remain ignored.
Generated HTML is built in `site/` and is never committed.

```bash
uv run --locked --extra dev --with pdoc==16.0.0 python .github/scripts/build_docs.py \
    --package-root . --output site --version-label development --check-examples
```

CI builds the site and runs marked examples on pull requests. After CI passes
on `main`, the Documentation workflow publishes the development docs. Stable
API pages come from the latest published package. Versioned docs starting at
0.3.1 are also archived as ZIP assets on their GitHub releases.

To publish a correction to the current stable guides, land it on `main` and run
the Documentation workflow with `rebuild_stable=true`. The workflow validates
the current guides against the released package and refreshes its docs archive.

```bash
gh workflow run docs.yml --ref main -f rebuild_stable=true
```
