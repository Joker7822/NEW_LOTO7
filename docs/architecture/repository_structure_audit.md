# NEW_LOTO7 Repository Structure Audit

Generated: `2026-07-28T04:12:57.164737+00:00`

## Summary

- Policy schema: **4**
- Tracked files: **511**
- Workflows: **15**
- Registered Workflows: **15**
- Package Python files: **15**
- Script/compatibility Python files: **36**
- Root compatibility Python files: **12**
- Tracked outputs: **389**

## Output classes

| Class | Files |
|---|---:|
| `diagnostics` | 50 |
| `evidence` | 244 |
| `production` | 9 |
| `state` | 84 |
| `unclassified` | 2 |

## Workflow inventory

| Workflow | Triggers | Lines |
|---|---|---:|
| `Backfill LOTO7 prize amounts`<br>`.github/workflows/backfill-loto7-prizes.yml` | workflow_dispatch, push | 66 |
| `Evaluator Full Data PR Check`<br>`.github/workflows/evaluator_full_data_pr_check.yml` | pull_request | 256 |
| `LOTO7 Smoke Test`<br>`.github/workflows/loto7-smoke.yml` | workflow_dispatch, push | 114 |
| `LOTO7 Evolution Trainer`<br>`.github/workflows/loto7_evolution.yml` | workflow_dispatch, schedule | 460 |
| `LOTO7 Generation 4 Evaluation`<br>`.github/workflows/loto7_generation4_run.yml` | workflow_dispatch, workflow_run, push | 342 |
| `LOTO7 Generation 5 Precision Evolution`<br>`.github/workflows/loto7_generation5.yml` | workflow_dispatch, push, schedule, pull_request | 245 |
| `LOTO7 Holdout Summary Integrity`<br>`.github/workflows/loto7_holdout_summary_integrity.yml` | workflow_dispatch, push, pull_request | 170 |
| `LOTO7 Model Self Evolution`<br>`.github/workflows/loto7_model_self_evolution.yml` | workflow_dispatch, push, schedule | 197 |
| `LOTO7 Nested Walk Forward Validation`<br>`.github/workflows/loto7_nested_walk_forward.yml` | workflow_dispatch, workflow_run | 286 |
| `LOTO7 Canonical Output Sync`<br>`.github/workflows/loto7_output_layout_sync.yml` | workflow_dispatch, workflow_run | 114 |
| `LOTO7 Recent Era Self Evolution`<br>`.github/workflows/loto7_recent_era_self_evolution.yml` | workflow_dispatch, schedule | 306 |
| `LOTO7 Production Prediction Publisher`<br>`.github/workflows/loto7_refresh_latest_prediction.yml` | workflow_dispatch, workflow_run, push, pull_request | 297 |
| `LOTO7 Validation Tests`<br>`.github/workflows/loto7_validation_tests.yml` | workflow_dispatch, push, pull_request | 140 |
| `LOTO7 Model Promotion Authorization`<br>`.github/workflows/model_promotion_authorization.yml` | push, pull_request | 57 |
| `Repository Structure Audit`<br>`.github/workflows/repository_structure_audit.yml` | workflow_dispatch, push, pull_request | 117 |

## Migration recommendations

### P1 — legacy_output_aliases

191 files remain under legacy output roots.

**Action:** Convert active workflows to canonical paths before removing aliases.

### P1 — remaining_root_implementations

11 allowlisted root implementations remain.

**Action:** Migrate one responsibility at a time with import and Resume tests.
