# NEW_LOTO7 Repository Structure Audit

Generated: `2026-07-16T12:01:51.856953+00:00`

## Summary

- Tracked files: **241**
- Workflows: **15**
- Python implementation/CLI files: **54**
- Test files: **4**
- Tracked files under `outputs/`: **150**
- Root Python files: **29**

## Directory distribution

| Location | Files |
|---|---:|
| `.github` | 15 |
| `<root>` | 45 |
| `outputs` | 150 |
| `patches` | 1 |
| `scripts` | 26 |
| `tests` | 4 |

## Workflows

| Workflow | Triggers | Lines | Notes |
|---|---|---:|---|
| `Backfill LOTO7 prize amounts`<br>`.github/workflows/backfill-loto7-prizes.yml` | workflow_dispatch, push | 66 | - |
| `LOTO7 Smoke Test`<br>`.github/workflows/loto7-smoke.yml` | workflow_dispatch, push | 114 | - |
| `LOTO7 Generation 4 Prediction`<br>`.github/workflows/loto7_dual_prediction.yml` | workflow_dispatch, workflow_run | 285 | non-standard `queue` key |
| `LOTO7 Evolution Trainer`<br>`.github/workflows/loto7_evolution.yml` | workflow_dispatch, schedule | 616 | - |
| `LOTO7 Generation 4 Full Run`<br>`.github/workflows/loto7_generation4_run.yml` | workflow_dispatch, push | 255 | - |
| `LOTO7 Model Self Evolution`<br>`.github/workflows/loto7_model_self_evolution.yml` | workflow_dispatch, push, schedule | 219 | - |
| `LOTO7 Nested Walk Forward Validation`<br>`.github/workflows/loto7_nested_walk_forward.yml` | workflow_dispatch, workflow_run | 202 | non-standard `queue` key |
| `LOTO7 Quick Finish Check`<br>`.github/workflows/loto7_quick_finish.yml` | workflow_dispatch | 248 | - |
| `LOTO7 Recent Era Self Evolution`<br>`.github/workflows/loto7_recent_era_self_evolution.yml` | workflow_dispatch, schedule | 307 | non-standard `queue` key |
| `LOTO7 TXT Reports`<br>`.github/workflows/loto7_txt_reports.yml` | workflow_dispatch, push, schedule | 67 | - |
| `LOTO7 Validation Tests`<br>`.github/workflows/loto7_validation_tests.yml` | workflow_dispatch, push | 69 | - |
| `Monitor Generation 4 Full Run Once`<br>`.github/workflows/monitor_generation4_once.yml` | workflow_dispatch, push | 84 | - |
| `Repository Structure Audit`<br>`.github/workflows/repository_structure_audit.yml` | workflow_dispatch, push | 74 | - |
| `Snapshot Generation 4 Run Status`<br>`.github/workflows/snapshot_generation4_status.yml` | workflow_dispatch, push | 46 | - |
| `Trigger Generation 4 Full Run Once`<br>`.github/workflows/trigger_generation4_once.yml` | workflow_dispatch, push | 54 | - |

## Highest-priority findings

### P0 — root_python_clutter

Root contains 29 Python files.

**Recommended action:** Keep compatibility wrappers at root and move implementations into src/loto7 or scripts by responsibility.

### P0 — tracked_generated_outputs

Repository tracks 150 files under outputs/.

**Recommended action:** Separate immutable prediction evidence from reproducible intermediate outputs; retain only latest, sealed, and compact history files.

### P0 — invalid_workflow_concurrency_key

.github/workflows/loto7_dual_prediction.yml, .github/workflows/loto7_nested_walk_forward.yml, .github/workflows/loto7_recent_era_self_evolution.yml

**Recommended action:** Remove non-standard concurrency.queue keys and use documented group/cancel-in-progress only.

### P1 — possibly_orphaned_python

14 Python files have no detected workflow/import reference.

**Recommended action:** Review before archiving; static detection can miss dynamic calls.

### P1 — generation_ownership

Multiple generations and workflows can write prediction outputs.

**Recommended action:** Make Generation 4 the sole writer of production prediction outputs; legacy workflows write only candidate or diagnostic artifacts.

### P1 — package_boundaries

Training, evaluation, prediction, workflow helpers and reporting are mixed.

**Recommended action:** Adopt src/loto7/{data,models,validation,portfolio,reporting} and keep scripts as thin CLI entry points.

### P2 — output_retention

State, reports, model candidates and sealed evidence share outputs/.

**Recommended action:** Split outputs into production/, validation/, state/, diagnostics/, sealed/ and define retention rules.

## Possibly unreferenced Python files

- `export_latest_prediction_txt.py`
- `loto7_chunked_backtest.py`
- `loto7_deep_ai.py`
- `loto7_evaluate_backtest.py`
- `loto7_pipeline_enhanced.py`
- `loto7_precision_evolution_trainer.py`
- `loto7_resumable_backtest.py`
- `loto7_self_evolution_engine.py`
- `loto7_self_evolver.py`
- `loto7_train_from_backtest.py`
- `resumable_loto7_backtest.py`
- `sitecustomize.py`
- `summarize_backtest_results.py`
- `validate_evolution_resume.py`

## Largest tracked files

| File | Bytes |
|---|---:|
| `outputs/ml_stack/ml_training_frame.csv` | 4503009 |
| `outputs/deep_ai/transformer_model.pt` | 796920 |
| `outputs/role_ensemble/role_ensemble_backtest.csv` | 767005 |
| `outputs/ml_stack/loto7_memorybank_mb4.csv` | 477155 |
| `outputs/ml_stack/loto7_memorybank_mb5.csv` | 379234 |
| `outputs/holdout/holdout_result.csv` | 324540 |
| `outputs/complete_ai/complete_ai_candidates.csv` | 236135 |
| `outputs/recent_era/recent_era_model_history.csv` | 188427 |
| `outputs/ml_stack/loto7_memorybank_mb6.csv` | 142957 |
| `loto7.csv` | 135323 |
| `outputs/model_self_evolution/history.csv` | 132316 |
| `outputs/deep_ai/ppo_policy.pt` | 108727 |
| `outputs/model_self_evolution/standalone_history.csv` | 77989 |
| `outputs/txt_reports/99_combined_report.txt` | 54329 |
| `loto7_advanced_optimizer.py` | 52640 |
| `outputs/super_recent/super_recent_model_history.csv` | 52076 |
| `outputs/super_recent/super_recent_model_state.json` | 46447 |
| `outputs/generation4/null_strategy_league_summary.json` | 46319 |
| `outputs/recent_era/recent_era_model_state.json` | 42959 |
| `merge_evolution_shards.py` | 42620 |

> Static-reference detection is conservative. A file listed as possibly unreferenced must be reviewed before deletion.
