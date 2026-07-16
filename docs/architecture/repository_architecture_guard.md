# Repository Architecture Guard

Generated: `2026-07-16T22:43:09.506278+00:00`

Status: **pass**

## Summary

- Workflows: **11**
- Root Python files: **29**
- Tracked output files: **172**
- Errors: **0**
- Warnings: **2**

## Production output writers

- `outputs/evolution_best_prediction.csv`: `.github/workflows/loto7_generation4_run.yml`
- `outputs/evolution_prediction_history.csv`: `.github/workflows/loto7_generation4_run.yml`
- `outputs/evolution_prediction_history_result.txt`: `.github/workflows/loto7_generation4_run.yml`
- `outputs/holdout/latest_prediction_report.txt`: `.github/workflows/loto7_generation4_run.yml`

## Errors

- None

## Warnings

- Root still contains 29 Python modules; retain as compatibility layer until Phase 2 migration
- outputs/ contains 172 tracked files; reproducible diagnostics should move to Actions artifacts

## Policy

- Generation 4 Production is the only workflow that may build committed production predictions.
- Evolution workflows produce models, candidates, state, and diagnostics only.
- Sealed manifests are immutable evidence and are not treated as disposable diagnostics.
- Root Python implementations remain a compatibility layer until package migration tests exist.
