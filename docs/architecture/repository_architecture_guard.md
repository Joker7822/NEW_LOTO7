# Repository Architecture Guard

Generated: `2026-07-28T00:43:48.418959+00:00`

Status: **pass**

## Summary

- Workflows: **15**
- Root Python files: **12**
- Tracked output files: **385**
- Errors: **0**
- Warnings: **2**

## Production output writers

- `outputs/evolution_best_prediction.csv`: `.github/workflows/loto7_refresh_latest_prediction.yml`
- `outputs/evolution_prediction_history.csv`: `.github/workflows/loto7_refresh_latest_prediction.yml`
- `outputs/evolution_prediction_history_result.txt`: `.github/workflows/loto7_refresh_latest_prediction.yml`
- `outputs/holdout/latest_prediction_report.txt`: `.github/workflows/loto7_refresh_latest_prediction.yml`

## Errors

- None

## Warnings

- Root still contains 12 Python modules; retain as compatibility layer until Phase 2 migration
- outputs/ contains 385 tracked files; reproducible diagnostics should move to Actions artifacts

## Policy

- `LOTO7 Production Prediction Publisher` is the only workflow that may build committed production predictions.
- Generation 4 evaluation writes candidate and diagnostic outputs only.
- Evolution workflows produce models, candidates, state and diagnostics only.
- Sealed production manifests are immutable evidence.
- Root Python implementations remain a compatibility layer until package migration tests exist.
