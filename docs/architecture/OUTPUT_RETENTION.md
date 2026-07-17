# Output Retention Policy

Effective: 2026-07-17

## Production — retain in Git

- `outputs/evolution_best_prediction.csv`
- `outputs/evolution_prediction_history.csv`
- `outputs/evolution_prediction_history_result.txt`
- `outputs/holdout/latest_prediction_report.txt`

## Immutable evidence — retain in Git

- `outputs/generation4/latest_sealed_manifest.json`
- `outputs/generation4/sealed_index.json`
- `outputs/generation4/sealed/*`

## Compact Generation 4 diagnostics — retain in Git

- strict adoption decision
- Null League summary and report
- latest Generation 4 and shadow summaries
- shadow history and Champion / Challenger evidence

## State — retain while resumable

- `outputs/model_self_evolution/`
- `outputs/recent_era/`
- `outputs/super_recent/`
- `outputs/validation/`

The state directories preserve interruption recovery, guarded candidates and
sealed promotion evidence. Compaction must not remove the latest resumable state
or adopted model.

## Artifact-only diagnostics

The following are reproducible and are uploaded by GitHub Actions instead of
being committed:

- `outputs/ml_stack/`
- `outputs/complete_ai/`

Legacy Deep AI experiments and derived TXT aggregation were removed because no
current production workflow consumes them.

## Never retain

- Python cache and bytecode
- fold-internal temporary data
- current-run snapshots and dispatch markers
- full-run status snapshots and input fingerprints
- obsolete model shards
- one-time patch files
- outputs from retired predictors and report aggregators
