# NEW_LOTO7 Repository Layout

Effective: 2026-07-20

## Canonical layout

```text
.github/workflows/   GitHub Actions orchestration
src/loto7/           Reusable implementation package
scripts/             Thin CLI and compatibility entry points
config/              Repository, gate and output-layout policy
tests/               Unit, integration and compatibility regression tests
docs/architecture/   Architecture and migration decisions
outputs/production/  Latest public production files
outputs/evidence/    Sealed and validation evidence
outputs/state/       Compact resumable state
outputs/diagnostics/ Compact diagnostics
root *.py             Compatibility layer for established imports
```

## Package ownership

Reusable evaluation code lives under `src/loto7`.

```text
src/loto7/evaluation/core.py        canonical financial evaluator
src/loto7/evaluation/hit_metrics.py payout-independent high-match metrics
src/loto7/evaluation/robust.py      robust payout and hit-quality diagnostics
src/loto7/validation/hit_rate_gate.py accuracy-first nested promotion gate
src/loto7/paths.py                  canonical/legacy output bindings
```

Migrated files under `scripts/` must delegate to package modules. They remain
available so workflow commands and historical imports do not change abruptly.

## Production ownership

`LOTO7 Generation 4 Production` is the only workflow allowed to build the
legacy production prediction, cumulative history and latest report. The
`LOTO7 Canonical Output Sync` workflow may only mirror existing files into the
canonical four-directory layout; it must not generate a new prediction.

## Migration safety

Output migration is non-destructive:

1. Existing legacy paths stay available.
2. Resume state is copied to `outputs/state/`, not moved.
3. Sealed evidence is copied without modification.
4. Large reproducible details remain workflow artifacts.
5. Compatibility tests verify package imports, workflow paths and resume aliases.
