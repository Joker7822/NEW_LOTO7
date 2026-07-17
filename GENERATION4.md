# LOTO7 Generation 4

Generation 4 is the sole production prediction layer of `NEW_LOTO7`.
It combines adopted Full, Recent, Super Recent and Regime candidates with sealed
validation, fail-closed statistical adoption gates, five-ticket portfolio
optimization and immutable evidence.

> Lottery drawings are highly random. Historical validation does not guarantee
> winning, profit, or a future predictive advantage.

## Processing order

1. Update Full / Recent / Super Recent candidate models.
2. Run dedicated and robust safety guards.
3. Run sealed Nested Walk-Forward validation.
4. Aggregate all Nested fold costs and payouts.
5. Reject promotion unless aggregate candidate ROI is at least 8.0% and is not below the baseline ROI.
6. Run the Null Strategy League and PBO diagnostic.
7. Reject production prediction unless the Null League explicitly passes.
8. Recalibrate the Conformal pool using prior-only rolling draw coverage.
9. Detect distribution change and update bounded source weights.
10. Generate Full / Recent / independent Super / Regime candidate pools.
11. Select five original candidates with DPP + Hypergraph beam search.
12. Verify usage and overlap constraints without replacing selected numbers.
13. Update live and shadow histories and the Champion / Challenger e-process.
14. Seal prediction, dataset, models and execution metadata with SHA-256.

## Production workflow

```text
.github/workflows/loto7_generation4_run.yml
LOTO7 Generation 4 Production
```

The workflow runs manually, after successful upstream model workflows, or when
its production implementation changes. A fixed concurrency group uses
`cancel-in-progress: true`, so the newest model state supersedes an older run.

## Strict adoption entry points

| Script | Purpose |
|---|---|
| `scripts/build_generation4_prediction_strict.py` | Rejects production generation when Null League is missing or failed; applies recalibrated Conformal |
| `scripts/promote_nested_candidate_strict.py` | Rejects Recent / Super promotion when aggregate Nested ROI is below either standard |
| `scripts/strict_adoption_gates.py` | Shared fail-closed Null, Nested ROI and Conformal logic |
| `scripts/build_generation4_prediction.py` | Five-ticket Generation 4 selector |
| `scripts/generation4_core.py` | Change-Point, Bayesian weights, DPP, Hypergraph and e-process utilities |
| `scripts/null_strategy_league.py` | Null strategy league and CSCV-style PBO diagnostic |
| `scripts/seal_generation4_prediction.py` | Immutable SHA-256 manifest and sealed index |

## Recalibrated Conformal pool

For every calibration draw, scores and candidate pools are constructed only from
earlier draws. Pool sizes from 14 through 24 are evaluated at draw level.

Default target:

```text
alpha: 0.20
target draw coverage: 80%
required covered main numbers: 4 of 7
calibration draws: 104
pool-size range: 14..24
```

The smallest pool meeting the target is selected. If no pool reaches the target,
the best empirical pool is selected and `coverage_target_met=false` is recorded.
No future row is used in calibration.

## Null Strategy League — fail closed

The adopted model is compared at the same five-ticket cost with seeded variants
of random, balanced, frequency, dormancy, recent and hybrid strategies.

Production adoption requires:

```text
decision.passed == true
```

A failed, missing or malformed decision stops before prediction history and seal
files are replaced. The result is written to:

```text
outputs/generation4/strict_adoption_gate.json
```

## Nested aggregate ROI — complete rejection

All fold costs and payouts are summed before promotion.

```text
candidate aggregate ROI >= 8.0%
candidate ROI - baseline ROI >= 0.0 percentage points
```

Model-ID mismatch, future leakage, missing fold totals or either threshold failure
causes complete rejection. The current production model remains unchanged.

## DPP + Hypergraph selection

Hard constraints:

```text
purchase count: 5
maximum use of one number: 4 tickets
maximum overlap between two tickets: 4 numbers
post-selection number replacement: prohibited
```

DPP penalizes similar tickets and Hypergraph scoring rewards distinct historical
pair and triple coverage. Super Recent receives a source quota only when its
model ID is independent from Recent Era.

## Retained outputs

Production:

```text
outputs/evolution_best_prediction.csv
outputs/evolution_prediction_history.csv
outputs/evolution_prediction_history_result.txt
outputs/holdout/latest_prediction_report.txt
```

Compact Generation 4 evidence:

```text
outputs/generation4/latest_generation4_summary.json
outputs/generation4/strict_adoption_gate.json
outputs/generation4/null_strategy_league_summary.json
outputs/generation4/null_strategy_league_report.txt
outputs/generation4/latest_shadow_predictions.json
outputs/generation4/shadow_history.csv
outputs/generation4/champion_challenger_summary.json
outputs/generation4/champion_challenger_report.txt
outputs/generation4/latest_sealed_manifest.json
outputs/generation4/sealed_index.json
outputs/generation4/sealed/*
```

Transient run snapshots, dispatch markers, full-run status files and input
fingerprints are not committed.

## Tests

```bash
python -m unittest \
  tests.test_prediction_output_consistency \
  tests.test_robust_validation_and_portfolio \
  tests.test_generation4_pipeline \
  tests.test_strict_adoption_gates -v
```
