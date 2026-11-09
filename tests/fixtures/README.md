# Test fixtures

## `reference/` + `attribution_cache/`

A minimal reference set for integration tests of the bundled attribution engine.
[`tests/conftest.py`](../conftest.py) configures two Claude Code runs on public
open-source tasks and puts writable engine state in a temporary directory.

- `astropy__astropy-13033` has a deductive diagnosis (primary
  `code_editing/incorrect_patch`)
- `django__django-11477` has historical judge and arbiter verdicts under
  [`attribution_cache/`](attribution_cache/README.md) in the `deepseek-v4-pro`
  namespace. Legacy contract tests expect the arbiter to refute the attribution.
  Current verification tests generate matching fingerprints at runtime.

`reference/data/` follows the engine's layout: `requirements/<id>.json` (task
and reference patch), `patch/<agent>/<id>.diff` (the agent's edit set),
`trajectory/<agent>/<id>.json` (the run) and `eval_<agent>.json` (whether each
run resolved its task).

The reference trajectories retain their historical `ccsession-trajectory`
shape to preserve the evidence and cached-verdict hashes.
`native_claude_code/` contains reconstructed Claude Code saved-session JSONL copies
of the two redacted reference runs for loader, API, and dashboard coverage.
Native parsing tests also construct small session records and subagent trees.

**Sanitization:** session/message UUIDs and request ids are deterministic
placeholders, signature/encrypted-content blobs are emptied, and absolute paths
are rewritten under `/home/user`. The golden tests verify the sanitized runs
produce the pinned diagnosis. Fixtures are additionally **checked for
credential-shaped strings** by [`tests/test_fixture_hygiene.py`](../test_fixture_hygiene.py) (redacted
reporting). Do not add fixture content from non-public runs. A new case needs the
same transformations, after which any vendored verdict for it must be
regenerated on the sanitized bytes. Verdict stamps cover the exact bytes the
model saw. See the [verdict fixture guide](attribution_cache/README.md).

Fixture contents must stay byte-stable: the golden tests pin the expected
diagnosis as literals, and the identity check in `attribution.diagnose()`
compares displayed files byte-for-byte against `reference/data/trajectory/...`.
