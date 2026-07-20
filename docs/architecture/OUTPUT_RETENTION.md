# Output Retention Policy

Effective: 2026-07-20

## Migration rule

The canonical layout is a non-destructive mirror. Existing legacy paths remain
available until every workflow and resume consumer has migrated. No resumable
state or sealed evidence is removed during this phase.

## Production — retain in Git

Canonical:

- `outputs/production/latest_prediction.csv`
- `outputs/production/prediction_history.csv`
- `outputs/production/prediction_history_result.txt`
- `outputs/production/latest_prediction_report.txt`

Legacy compatibility copies remain retained:

- `outputs/evolution_best_prediction.csv`
- `outputs/evolution_prediction_history.csv`
- `outputs/evolution_prediction_history_result.txt`
- `outputs/holdout/latest_prediction_report.txt`

## Immutable evidence — retain in Git

Canonical:

- `outputs/evidence/generation4/latest_sealed_manifest.json`
- `outputs/evidence/generation4/sealed_index.json`
- `outputs/evidence/generation4/sealed/*`
- compact Nested and promotion decisions under `outputs/evidence/validation/`

Legacy sealed paths remain retained until migration completion.

## State — retain while resumable

Canonical compact state:

- `outputs/state/full/`
- `outputs/state/recent/`
- `outputs/state/super_recent/`

Legacy state directories remain the active resume source during the
compatibility phase:

- `outputs/model_self_evolution/`
- `outputs/recent_era/`
- `outputs/super_recent/`
- `outputs/validation/`

The mirror copies JSON and TXT state/evidence. It does not copy large history
CSV files into the canonical state directories.

## Compact diagnostics — retain in Git

- JSON summaries
- human-readable TXT reports
- Generation 4 shadow history required for Champion/Challenger evidence
- `outputs/layout_manifest.json`

Canonical location: `outputs/diagnostics/`.

## Artifact-only diagnostics

The following are reproducible and should be uploaded by GitHub Actions instead
of duplicated in the canonical Git layout:

- Holdout ticket-level result CSV
- Role Ensemble ticket-level backtest CSV
- Fold-internal data and model search histories
- ML Stack and Complete AI details
- large bootstrap and Null League trial-level details

## Never retain

- Python cache and bytecode
- fold-internal temporary data
- current-run snapshots and dispatch markers
- full-run status snapshots and input fingerprints
- obsolete model shards
- one-time patch files
- outputs from retired predictors and report aggregators
