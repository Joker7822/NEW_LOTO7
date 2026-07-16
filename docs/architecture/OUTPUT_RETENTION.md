# Output Retention Policy

## Production — retain in Git

- `outputs/evolution_best_prediction.csv`
- `outputs/evolution_prediction_history.csv`
- `outputs/evolution_prediction_history_result.txt`
- `outputs/holdout/latest_prediction_report.txt`

## Immutable evidence — retain in Git

- `outputs/generation4/latest_sealed_manifest.json`
- `outputs/generation4/sealed_index.json`
- `outputs/generation4/sealed/*`

## State — retain while resumable

Model evolution, Recent/Super and validation state remain versioned until a
checkpoint compaction workflow is introduced.

## Reproducible diagnostics — migrate to Actions artifacts

Large training frames, memory banks, backtest detail CSVs and binary model
experiments should be uploaded as Actions artifacts. Compact summaries and the
latest accepted model may remain in Git. No existing diagnostic is deleted in
Phase 1 because several workflows still consume these paths.
