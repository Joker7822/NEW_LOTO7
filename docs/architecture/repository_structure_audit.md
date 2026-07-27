# NEW_LOTO7 Repository Structure Audit

Generated: `2026-07-27T09:49:52.050731+00:00`

## Summary

- Tracked files: **456**
- Workflows: **12**
- Python implementation/CLI files: **54**
- Test files: **15**
- Tracked files under `outputs/`: **357**
- Root Python files: **12**

## Directory distribution

| Location | Files |
|---|---:|
| `.github` | 12 |
| `<root>` | 21 |
| `config` | 2 |
| `docs` | 7 |
| `outputs` | 357 |
| `scripts` | 32 |
| `src` | 10 |
| `tests` | 15 |

## Workflows

| Workflow | Triggers | Lines | Notes |
|---|---|---:|---|
| `Backfill LOTO7 prize amounts`<br>`.github/workflows/backfill-loto7-prizes.yml` | workflow_dispatch, push | 66 | - |
| `Evaluator Full Data PR Check`<br>`.github/workflows/evaluator_full_data_pr_check.yml` | pull_request | 256 | - |
| `LOTO7 Smoke Test`<br>`.github/workflows/loto7-smoke.yml` | workflow_dispatch, push | 114 | - |
| `LOTO7 Evolution Trainer`<br>`.github/workflows/loto7_evolution.yml` | workflow_dispatch, schedule | 460 | - |
| `LOTO7 Generation 4 Evaluation`<br>`.github/workflows/loto7_generation4_run.yml` | workflow_dispatch, workflow_run, push | 342 | - |
| `LOTO7 Model Self Evolution`<br>`.github/workflows/loto7_model_self_evolution.yml` | workflow_dispatch, push, schedule | 196 | - |
| `LOTO7 Nested Walk Forward Validation`<br>`.github/workflows/loto7_nested_walk_forward.yml` | workflow_dispatch, workflow_run | 286 | - |
| `LOTO7 Canonical Output Sync`<br>`.github/workflows/loto7_output_layout_sync.yml` | workflow_dispatch, workflow_run | 112 | - |
| `LOTO7 Recent Era Self Evolution`<br>`.github/workflows/loto7_recent_era_self_evolution.yml` | workflow_dispatch, schedule | 306 | - |
| `LOTO7 Production Prediction Publisher`<br>`.github/workflows/loto7_refresh_latest_prediction.yml` | workflow_dispatch, workflow_run, push, pull_request | 297 | - |
| `LOTO7 Validation Tests`<br>`.github/workflows/loto7_validation_tests.yml` | workflow_dispatch, push, pull_request | 140 | - |
| `Repository Structure Audit`<br>`.github/workflows/repository_structure_audit.yml` | workflow_dispatch, push | 88 | - |

## Highest-priority findings

### P0 — tracked_generated_outputs

Repository tracks 357 files under outputs/.

**Recommended action:** Separate immutable prediction evidence from reproducible intermediate outputs; retain only latest, sealed, and compact history files.

### P1 — possibly_orphaned_python

6 Python files have no detected workflow/import reference.

**Recommended action:** Review before archiving; static detection can miss dynamic calls.

### P1 — package_boundaries

Training, evaluation, prediction, workflow helpers and reporting are mixed.

**Recommended action:** Adopt src/loto7/{data,models,validation,portfolio,reporting} and keep scripts as thin CLI entry points.

### P2 — output_retention

State, reports, model candidates and sealed evidence share outputs/.

**Recommended action:** Split outputs into production/, validation/, state/, diagnostics/, sealed/ and define retention rules.

## Possibly unreferenced Python files

- `src/loto7/evaluation/core.py`
- `src/loto7/evaluation/hit_metrics.py`
- `src/loto7/evaluation/robust.py`
- `src/loto7/evolution/hit_first.py`
- `src/loto7/paths.py`
- `src/loto7/validation/hit_rate_gate.py`

## Largest tracked files

| File | Bytes |
|---|---:|
| `outputs/role_ensemble/role_ensemble_backtest.csv` | 769265 |
| `outputs/holdout/holdout_result.csv` | 325490 |
| `outputs/recent_era/recent_era_model_history.pre_hit_first_20260722035646.csv` | 223169 |
| `outputs/super_recent/super_recent_model_history.csv` | 156642 |
| `outputs/model_self_evolution/history.pre_hit_first_20260721231909.csv` | 149791 |
| `loto7.csv` | 135724 |
| `outputs/super_recent/super_recent_model_history.pre_hit_first_20260722083852.csv` | 127953 |
| `outputs/state/super_recent/super_recent_model_state.json` | 63481 |
| `outputs/super_recent/super_recent_model_state.json` | 63481 |
| `outputs/recent_era/recent_era_model_state.json` | 60016 |
| `outputs/state/recent/recent_era_model_state.json` | 60016 |
| `outputs/recent_era/recent_era_model_history.csv` | 59917 |
| `outputs/diagnostics/generation4/null_strategy_league_summary.json` | 46340 |
| `outputs/generation4/null_strategy_league_summary.json` | 46340 |
| `merge_evolution_shards.py` | 42620 |
| `outputs/model_self_evolution/history.csv` | 40957 |
| `_loto7_evolution_trainer_impl.py` | 39244 |
| `outputs/model_self_evolution/state.json` | 39190 |
| `outputs/state/full/state.json` | 39190 |
| `outputs/model_self_evolution/standalone_state.json` | 38337 |

> Static-reference detection is conservative. A file listed as possibly unreferenced must be reviewed before deletion.
