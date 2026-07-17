# Workflow Ownership

Effective: 2026-07-17

| Responsibility | Owner workflow | Retention |
|---|---|---|
| Dataset refresh, long evolution, holdout, role backtest | `LOTO7 Evolution Trainer` | model/state/compact diagnostics; ML and Complete AI detail as artifacts |
| Full-model standalone evolution | `LOTO7 Model Self Evolution` | model/state/diagnostics |
| Recent and Super candidate generation | `LOTO7 Recent Era Self Evolution` | guarded candidates/diagnostics |
| Sealed nested validation and promotion | `LOTO7 Nested Walk Forward Validation` | validation evidence/models |
| Production prediction, live history, Null gate, e-process and seal | `LOTO7 Generation 4 Production` | production/evidence |
| Regression and leakage checks | `LOTO7 Validation Tests` | Actions result only |
| Architecture verification | `Repository Structure Audit` | architecture reports |

## Production ownership

Only `LOTO7 Generation 4 Production` may write the four production prediction
files. Retired Quick Finish and TXT aggregation workflows are forbidden by
`config/repository_layout.json`.

## Latest-state concurrency

Production prediction uses one stable concurrency group with
`cancel-in-progress: true`. A newer upstream model state supersedes an older
queued or running prediction. GitHub Actions does not use a custom `queue:` key.
