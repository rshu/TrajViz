# Recorded attribution verdicts

This directory contains historical judge and arbiter verdicts for
`claude_code / django__django-11477`. The schema-20 records preserve the inputs
for legacy contract checks. Current provenance tests generate matching
fingerprints from the fixture inputs.

[`tests/conftest.py`](../../conftest.py) points the bundled engine's judge and
arbiter cache directories at these fixtures and sets the `deepseek-v4-pro`
namespace. It directs `config.PROJECT_DATA` and writable caches to a fresh
temporary directory for each test process. The tests run offline without
developer-local data or credentials.

## Provenance

TrajViz trusts a cached verdict only when its provenance matches the current
inputs. It checks `trajectory_sha256`, `requirements_sha256`, `prompt_version`,
and `evidence_schema`. A mismatch disables the corresponding layer. Preserve
the original stamps in these recorded verdicts.

The historical engine used `deepseek-v4-pro` with `judge_one` and `verify_one`
against the fixture reference data and an empty `PROJECT_DATA` directory. The
arbiter refuted the `code_editing` / `relevant_change_omitted` claim. The
recorded stamps describe the bytes supplied to the model at that time.

To add fresh recorded verdicts, run the current judge and arbiter against the
sanitized fixture and retain their generated provenance. Tests that exercise
rejection can keep using recorded verdicts with a deliberately mismatched
fingerprint.
