# Workflow Ownership

Effective: 2026-07-20

| Responsibility | Owner workflow | Retention |
|---|---|---|
| Dataset refresh, long evolution, holdout, role backtest | `LOTO7 Evolution Trainer` | legacy model/state and compact diagnostics; large detail as artifacts |
| Full-model standalone evolution | `LOTO7 Model Self Evolution` | resumable state and diagnostics |
| Recent and Super candidate generation | `LOTO7 Recent Era Self Evolution` | guarded candidates and state |
| Sealed nested validation, high-match gate and financial promotion | `LOTO7 Nested Walk Forward Validation` | validation evidence and adopted models |
| Production prediction, live history, Null gate, e-process and seal | `LOTO7 Generation 4 Production` | legacy production and immutable evidence |
| Canonical four-directory mirror | `LOTO7 Canonical Output Sync` | production/evidence/state/diagnostics mirror only |
| Regression, package, workflow and resume checks | `LOTO7 Validation Tests` | Actions result only |
| Architecture verification | `Repository Structure Audit` | architecture reports |

## Production ownership

Only `LOTO7 Generation 4 Production` may create or replace the four legacy
production prediction files. Retired Quick Finish and TXT aggregation workflows
remain forbidden by `config/repository_layout.json`.

`LOTO7 Canonical Output Sync` is a mirror workflow. It may copy already-created
files into `outputs/production`, `outputs/evidence`, `outputs/state` and
`outputs/diagnostics`, but it may not run a predictor or change model adoption.

## Promotion ownership

`LOTO7 Nested Walk Forward Validation` owns both stages of model promotion:

1. payout-independent high-match gate;
2. existing robust financial and payout-concentration gate.

A high-match rejection stops before model replacement and records the rejection
as validation evidence.

## Latest-state concurrency

Production prediction and canonical output sync each use stable concurrency
groups with `cancel-in-progress: true`. Newer state supersedes older queued or
running mirror/prediction work. Long evolution and sealed nested validation do
not cancel an active run.
