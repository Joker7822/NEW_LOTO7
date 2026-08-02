# Repository Architecture Guard

Generated: `2026-08-02T21:54:03.010401+00:00`

Status: **pass**

## Summary

- Policy schema: **4**
- Workflows: **15**
- Root Python compatibility files: **12**
- Tracked outputs: **588**
- Unclassified outputs: **2**
- Errors: **0**
- Warnings: **2**

## Workflow ownership

- **production_prediction_publication**: `LOTO7 Production Prediction Publisher` (`.github/workflows/loto7_refresh_latest_prediction.yml`)
- **automatic_model_promotion**: `LOTO7 Generation 5 Precision Evolution` (`.github/workflows/loto7_generation5.yml`)
- **generation4_diagnostics**: `LOTO7 Generation 4 Evaluation` (`.github/workflows/loto7_generation4_run.yml`)
- **canonical_output_sync**: `LOTO7 Canonical Output Sync` (`.github/workflows/loto7_output_layout_sync.yml`)
- **repository_architecture_guard**: `Repository Structure Audit` (`.github/workflows/repository_structure_audit.yml`)

## Errors

- None

## Warnings

- 2 output files are not classified by repository policy
- 290 files remain in legacy output roots during compatibility migration
