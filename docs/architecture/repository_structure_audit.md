# NEW_LOTO7 Repository Structure Audit

Generated: `2026-07-17T11:36:23.516215+00:00`

## Summary

- Tracked files: **187**
- Workflows: **9**
- Python implementation/CLI files: **36**
- Test files: **5**
- Tracked files under `outputs/`: **121**
- Root Python files: **10**

## Directory distribution

| Location | Files |
|---|---:|
| `.github` | 9 |
| `<root>` | 18 |
| `config` | 1 |
| `docs` | 7 |
| `outputs` | 121 |
| `scripts` | 26 |
| `tests` | 5 |

## Workflows

| Workflow | Triggers | Lines | Notes |
|---|---|---:|---|
| `Backfill LOTO7 prize amounts`<br>`.github/workflows/backfill-loto7-prizes.yml` | workflow_dispatch, push | 66 | - |
| `LOTO7 Smoke Test`<br>`.github/workflows/loto7-smoke.yml` | workflow_dispatch, push | 114 | - |
| `LOTO7 Evolution Trainer`<br>`.github/workflows/loto7_evolution.yml` | workflow_dispatch, schedule | 460 | - |
| `LOTO7 Generation 4 Production`<br>`.github/workflows/loto7_generation4_run.yml` | workflow_dispatch, workflow_run, push | 360 | - |
| `LOTO7 Model Self Evolution`<br>`.github/workflows/loto7_model_self_evolution.yml` | workflow_dispatch, push, schedule | 196 | - |
| `LOTO7 Nested Walk Forward Validation`<br>`.github/workflows/loto7_nested_walk_forward.yml` | workflow_dispatch, workflow_run | 216 | - |
| `LOTO7 Recent Era Self Evolution`<br>`.github/workflows/loto7_recent_era_self_evolution.yml` | workflow_dispatch, schedule | 306 | - |
| `LOTO7 Validation Tests`<br>`.github/workflows/loto7_validation_tests.yml` | workflow_dispatch, push | 82 | - |
| `Repository Structure Audit`<br>`.github/workflows/repository_structure_audit.yml` | workflow_dispatch, push | 95 | - |

## Highest-priority findings

### P0 — tracked_generated_outputs

Repository tracks 121 files under outputs/.

**Recommended action:** Separate immutable prediction evidence from reproducible intermediate outputs; retain only latest, sealed, and compact history files.

### P1 — package_boundaries

Training, evaluation, prediction, workflow helpers and reporting are mixed.

**Recommended action:** Adopt src/loto7/{data,models,validation,portfolio,reporting} and keep scripts as thin CLI entry points.

### P2 — output_retention

State, reports, model candidates and sealed evidence share outputs/.

**Recommended action:** Split outputs into production/, validation/, state/, diagnostics/, sealed/ and define retention rules.

## Possibly unreferenced Python files

- None detected

## Largest tracked files

| File | Bytes |
|---|---:|
| `outputs/role_ensemble/role_ensemble_backtest.csv` | 767005 |
| `outputs/holdout/holdout_result.csv` | 324540 |
| `outputs/recent_era/recent_era_model_history.csv` | 202889 |
| `outputs/model_self_evolution/history.csv` | 136264 |
| `loto7.csv` | 135323 |
| `outputs/model_self_evolution/standalone_history.csv` | 77989 |
| `outputs/super_recent/super_recent_model_history.csv` | 68342 |
| `outputs/super_recent/super_recent_model_state.json` | 46524 |
| `outputs/generation4/null_strategy_league_summary.json` | 46293 |
| `outputs/recent_era/recent_era_model_state.json` | 42860 |
| `merge_evolution_shards.py` | 42620 |
| `outputs/model_self_evolution/standalone_state.json` | 39771 |
| `loto7_evolution_trainer.py` | 39244 |
| `scripts/backtest_role_ensemble.py` | 31625 |
| `docs/architecture/repository_structure_audit.json` | 31325 |
| `outputs/holdout/holdout_report.txt` | 30801 |
| `loto7_model_self_evolver.py` | 30690 |
| `scripts/adaptive_model_safety_guard.py` | 28220 |
| `holdout_evaluator.py` | 27801 |
| `scripts/build_dual_model_prediction.py` | 23954 |

> Static-reference detection is conservative. A file listed as possibly unreferenced must be reviewed before deletion.
