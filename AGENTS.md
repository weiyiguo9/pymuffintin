# Repository guidance

## Local only reference
@AGENTS.local.md


## Test scope

- Default to focused tests for the affected module or execution path.
- Do not run the full `pytest tests/ -q` suite after ordinary feature work,
  localized fixes, documentation changes, or routine API edits.
- Run the full suite only for a major cross-module refactor or another change
  whose blast radius genuinely spans most of the package.
- `tests/test_muffintin_exchange_pipeline.py` and
  `tests/test_mto_hydrogen_checkpoint.py` require the native `libmuffintin`
  extension; both modules skip when it is not built. The remaining tests,
  including `tests/test_hf.py`, require no native import.
- Treat the reported exchange-energy and self-energy differences as pipeline
  consistency checks on the tracked fixture, not material-accuracy claims.

## Commit messages

Follow Conventional Commits 1.0.0.

Format:

    <type>[optional scope]: <description>

    <body: 1-3 sentences explaining what changed and why>

    [optional footer(s)]

Rules:

- type: feat | fix | refactor | perf | docs | test | build | ci | chore
- description: imperative mood, lowercase, no trailing period, ≤72 chars total
- Body is required for feat, fix, refactor, perf; optional for docs/chore.
  Explain motivation and effect, not a restatement of the diff.
- Breaking changes: append `!` after type/scope and add a
  `BREAKING CHANGE:` footer describing migration.
- scope: use module/crate/package name when the change is localized,
  e.g. `feat(solver): ...`
- One logical change per commit; do not mix refactor with behavior changes.
- Do not add "Co-Authored-By" or tool attribution lines.
