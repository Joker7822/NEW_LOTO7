# NEW_LOTO7 Repository Layout

## Current stable layout

```text
.github/workflows/   GitHub Actions orchestration
config/              Repository and runtime policy
scripts/             CLI, validation, reporting and maintenance tools
tests/               Focused regression and leakage tests
docs/architecture/   Architecture, audits and migration decisions
outputs/              Versioned production evidence, state and diagnostics
root *.py             Compatibility layer for established training imports
```

## Production path

`LOTO7 Generation 4 Production` is the only workflow allowed to build and
commit the production prediction CSV, cumulative history and latest report.
Evolution workflows own model/state generation only. Nested validation owns
candidate promotion evidence only.

## Phase-2 package migration

Root Python implementations will move gradually to:

```text
src/loto7/data/
src/loto7/models/
src/loto7/validation/
src/loto7/portfolio/
src/loto7/reporting/
```

Every move must keep a root compatibility wrapper until all workflow imports,
unit tests and historical resume files are verified.
