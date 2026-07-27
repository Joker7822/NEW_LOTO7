# LOTO7 Generation 4

Generation 4 is the statistical evaluation and five-ticket candidate layer of
`NEW_LOTO7`. It combines approved Full, Recent, Super Recent and Regime models
with sealed validation, fail-closed adoption gates, Null Strategy League, PBO,
Conformal calibration and portfolio optimization.

Production publication is intentionally separate. `LOTO7 Production Prediction
Publisher` is the sole owner of the latest prediction, cumulative history,
result history and sealed production evidence.

> Lottery drawings are highly random. Historical validation does not guarantee
> winning, profit, or a future predictive advantage.

## Processing order

1. Update Full / Recent / Super Recent candidate models.
2. Run dedicated and robust safety guards.
3. Run sealed Nested Walk-Forward validation and model-promotion gates.
4. Run the Null Strategy League and PBO diagnostic.
5. Recalibrate the Conformal pool using prior-only rolling coverage.
6. Detect distribution change and update bounded source weights.
7. Generate Full / Recent / independent Super / Regime candidate pools.
8. Select five original candidate tickets with DPP + Hypergraph beam search.
9. Verify usage and overlap constraints without replacing selected numbers.
10. Record strict-gate and Champion / Challenger diagnostic evidence.
11. Trigger the production publisher after successful evaluation completion.
12. Publish `latest actual draw + 1` from the currently approved models.
13. Update cumulative history and seal the production prediction with SHA-256.

## Workflow ownership

```text
.github/workflows/loto7_generation4_run.yml
LOTO7 Generation 4 Evaluation
  └─ candidate and diagnostic outputs only

.github/workflows/loto7_refresh_latest_prediction.yml
LOTO7 Production Prediction Publisher
  └─ sole production writer
```

Both workflows use stable latest-state concurrency. The evaluator never writes
the four legacy production files.

## Strict adoption entry points

| Script | Purpose |
|---|---|
| `scripts/build_generation4_prediction_strict.py` | Applies Null League and Conformal fail-closed checks to a Generation 4 candidate |
| `scripts/promote_nested_candidate_strict.py` | Rejects Recent / Super promotion when sealed Nested requirements fail |
| `scripts/strict_adoption_gates.py` | Shared Null, Nested ROI and Conformal logic |
| `scripts/build_generation4_prediction.py` | Five-ticket selector used by candidate evaluation and approved-model publishing |
| `scripts/generation4_core.py` | Change-Point, Bayesian weights, DPP, Hypergraph and e-process utilities |
| `scripts/null_strategy_league.py` | Null strategy league and CSCV-style PBO diagnostic |
| `scripts/seal_generation4_prediction.py` | Immutable SHA-256 production manifest and sealed index |

## Recalibrated Conformal pool

For every calibration draw, scores and candidate pools are constructed only from
earlier draws. Pool sizes from 14 through 24 are evaluated at draw level.

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

The approved model is compared at the same five-ticket cost with seeded random,
balanced, frequency, dormancy, recent and hybrid strategies.

Candidate adoption requires:

```text
decision.passed == true
```

A failed, missing or malformed decision records
`outputs/generation4/strict_adoption_gate.json` with adoption blocked. The
production publisher still creates the next prediction from currently approved
models; it does not promote the rejected candidate.

## DPP + Hypergraph selection

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

Generation 4 diagnostic candidates:

```text
outputs/generation4/candidate_prediction.csv
outputs/generation4/candidate_prediction_report.txt
outputs/generation4/candidate_generation4_summary.json
outputs/generation4/candidate_shadow_predictions.json
outputs/generation4/strict_adoption_gate.json
outputs/generation4/null_strategy_league_summary.json
outputs/generation4/null_strategy_league_report.txt
outputs/generation4/production_history_result_snapshot.txt
outputs/generation4/shadow_history.csv
outputs/generation4/champion_challenger_summary.json
outputs/generation4/champion_challenger_report.txt
```

Production Publisher outputs:

```text
outputs/evolution_best_prediction.csv
outputs/evolution_prediction_history.csv
outputs/evolution_prediction_history_result.txt
outputs/holdout/latest_prediction_report.txt
outputs/generation4/latest_generation4_summary.json
outputs/generation4/latest_shadow_predictions.json
outputs/generation4/latest_sealed_manifest.json
outputs/generation4/sealed_index.json
outputs/generation4/sealed/*
```

## Tests

```bash
python -m unittest \
  tests.test_prediction_output_consistency \
  tests.test_robust_validation_and_portfolio \
  tests.test_generation4_pipeline \
  tests.test_strict_adoption_gates \
  tests.test_production_ownership -v
```
