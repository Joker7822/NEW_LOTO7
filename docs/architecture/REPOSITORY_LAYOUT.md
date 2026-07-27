# NEW_LOTO7 Repository Layout

Effective: 2026-07-27

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
src/loto7/evaluation/core.py          canonical financial evaluator
src/loto7/evaluation/hit_metrics.py   payout-independent hit and portfolio metrics
src/loto7/evaluation/robust.py        robust payout and hit-quality diagnostics
src/loto7/evolution/hit_first.py      high-match learning and adoption safety
src/loto7/validation/hit_rate_gate.py accuracy-first nested promotion gate
src/loto7/paths.py                    canonical/legacy output bindings
```

Migrated files under `scripts/` must delegate to package modules. They remain
available so workflow commands and historical imports do not change abruptly.

## Production ownership

`LOTO7 Production Prediction Publisher` is the sole builder for the legacy
production prediction, cumulative history, result history, latest report and
sealed production evidence.

`LOTO7 Generation 4 Evaluation` produces only files named as candidate or
diagnostic evidence under `outputs/generation4/`. It may never write the four
legacy production outputs.

`LOTO7 Canonical Output Sync` mirrors existing files into the canonical
four-directory layout and never generates a prediction.

## Evaluation completeness

Holdout detail contains each ticket in the `ticket` column. Summary repair must
reconstruct the five-ticket portfolio for every draw before publishing
portfolio diversity metrics. Missing or incomplete portfolios are represented
as unavailable (`null` plus an explicit flag), never as genuine zero diversity.

## Migration safety

1. Existing legacy paths stay available.
2. Resume state is copied to `outputs/state/`, not moved.
3. Sealed evidence is copied without modification.
4. Large reproducible details remain workflow artifacts.
5. Compatibility tests verify imports, workflow ownership and resume aliases.
