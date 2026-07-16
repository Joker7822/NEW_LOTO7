# Workflow Ownership

| Responsibility | Owner workflow | Committed output class |
|---|---|---|
| Dataset refresh, long evolution, holdout, role backtest | `LOTO7 Evolution Trainer` | model/state/diagnostics |
| Full-model standalone evolution | `LOTO7 Model Self Evolution` | model/state/diagnostics |
| Recent and Super candidate generation | `LOTO7 Recent Era Self Evolution` | guarded candidates/diagnostics |
| Sealed nested validation and promotion | `LOTO7 Nested Walk Forward Validation` | validation evidence/models |
| Production prediction, live history, e-process and seal | `LOTO7 Generation 4 Production` | production/evidence |
| Report aggregation | `LOTO7 TXT Reports` | derived reports |
| Architecture verification | `Repository Structure Audit` | architecture reports |

## Latest-state concurrency

Production prediction uses one stable concurrency group with
`cancel-in-progress: true`. A newer model state supersedes an older queued or
running prediction. GitHub Actions does not use a custom `queue:` key.
