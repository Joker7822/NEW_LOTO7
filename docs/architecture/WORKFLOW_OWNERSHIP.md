# Workflow Ownership

Effective: 2026-07-27

| Responsibility | Owner workflow | Retention |
|---|---|---|
| Dataset refresh, long evolution, holdout, role backtest | `LOTO7 Evolution Trainer` | model/state and compact diagnostics; large detail as artifacts |
| Full-model standalone evolution | `LOTO7 Model Self Evolution` | resumable state and diagnostics |
| Recent and Super candidate generation | `LOTO7 Recent Era Self Evolution` | guarded candidates and state |
| Sealed nested validation and model promotion | `LOTO7 Nested Walk Forward Validation` | validation evidence and adopted models |
| Null League, strict gates, Generation 4 candidate diagnostics | `LOTO7 Generation 4 Evaluation` | diagnostic candidate, gate and champion/challenger evidence |
| Latest approved-model prediction, cumulative history and seal | `LOTO7 Production Prediction Publisher` | production files and immutable prediction evidence |
| Canonical four-directory mirror | `LOTO7 Canonical Output Sync` | mirror only |
| Regression, package, workflow and resume checks | `LOTO7 Validation Tests` | Actions result only |
| Architecture verification | `Repository Structure Audit` | architecture reports |

## Production ownership

`LOTO7 Production Prediction Publisher` is the only workflow allowed to create
or replace the legacy production prediction, cumulative prediction history,
result history and latest prediction report. It always publishes
`latest actual draw + 1` from the currently approved models.

`LOTO7 Generation 4 Evaluation` may read production history, but writes only
candidate and diagnostic files under `outputs/generation4/`. A strict-gate pass
or rejection never writes a production prediction directly.

`LOTO7 Canonical Output Sync` may mirror already-published files into
`outputs/production`, `outputs/evidence`, `outputs/state` and
`outputs/diagnostics`; it may not run a predictor or change model adoption.

## Promotion ownership

`LOTO7 Nested Walk Forward Validation` owns model promotion. Promotion requires:

1. payout-independent high-match improvement;
2. sealed walk-forward consistency;
3. financial and payout-concentration safety;
4. no-op model ID/SHA rejection.

`LOTO7 Generation 4 Evaluation` independently evaluates Null League, PBO,
conformal coverage and portfolio constraints. Its output is diagnostic evidence
consumed by the production publisher; it does not replace approved models.

## Latest-state concurrency

Production publishing, Generation 4 evaluation and canonical mirroring use
stable latest-state-wins concurrency groups. Long evolution and sealed nested
validation do not cancel an active run.
