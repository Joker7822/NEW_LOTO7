# LOTO7 Generation 4

Generation 4 is the production prediction and validation layer of `NEW_LOTO7`.
It keeps the existing Full-period, Recent Era, Super Recent, Regime, Nested
Walk-Forward and Robust ROI components, and adds statistical controls around
them.

> Lottery drawings are highly random. This system does not guarantee winning,
> profit, or predictive advantage.

## Processing order

1. Update Full / Recent / Super Recent candidate models.
2. Apply dedicated and robust safety guards.
3. Run sealed Nested Walk-Forward validation before model promotion.
4. Run the Null Strategy League and PBO diagnostic.
5. Build a rolling Conformal number pool using prior-only scores.
6. Detect recent distribution change and adjust source weights conservatively.
7. Update Bayesian-style source weights from shadow-operation history.
8. Generate candidate pools from Full, Recent, independent Super Recent and Regime.
9. Select five original candidates with DPP + Hypergraph beam search.
10. Verify hard constraints without changing selected numbers.
11. Update live and shadow histories.
12. Update the anytime-valid Champion / Challenger e-process.
13. Seal prediction, dataset, model and script hashes with SHA-256.

## Production workflow

`.github/workflows/loto7_dual_prediction.yml`

Displayed workflow name:

```text
LOTO7 Generation 4 Prediction
```

It runs after the major evolution and nested-validation workflows, or manually.
Runs are serialized with `queue: max`. An input fingerprint skips duplicate
non-manual runs.

## Main scripts

| Script | Purpose |
|---|---|
| `scripts/generation4_core.py` | Conformal, Change-Point, Bayesian weights, DPP, Hypergraph and e-process utilities |
| `scripts/null_strategy_league.py` | Random/simple strategy league and CSCV-style PBO diagnostic |
| `scripts/build_generation4_prediction.py` | Complete five-ticket Generation 4 selector |
| `scripts/update_generation4_shadow_history.py` | Shadow evaluation and Champion / Challenger evidence |
| `scripts/finalize_generation4_report.py` | Final synchronized Generation 4 TXT report |
| `scripts/seal_generation4_prediction.py` | Immutable SHA-256 prediction manifest |

## Candidate sources

- `full`: adopted full-period model
- `recent`: 2020+ dedicated model
- `super`: independent 2023+ model only
- `regime`: role-adjusted Full model candidates

Super Recent receives a quota only when its model ID differs from Recent Era.

## Dynamic source allocation

Initial source weights are conservative priors. Evaluated shadow results update
bounded utilities for each source. The current Change-Point score may move a
small portion of Full weight toward Recent or independent Super Recent.

Every five-ticket portfolio retains minimum representation from:

- Full
- Recent
- Regime

Independent Super Recent can receive a minimum slot when its posterior weight is
large enough.

## Rolling Conformal number pool

For every calibration draw, number scores are calculated from draws strictly
before that draw. Current prediction uses the resulting historical
nonconformity threshold.

Defaults:

```text
alpha: 0.20
calibration draws: 104
minimum pool size: 14
maximum pool size: 24
minimum preferred hits per ticket: 4
```

This is a rolling calibration diagnostic. Exchangeability or future coverage is
not guaranteed under distribution shift.

## DPP + Hypergraph selection

DPP discourages selecting highly similar tickets. Candidate similarity includes:

- number membership
- number-band composition
- source model
- sum, odd/even and span features

Hypergraph coverage rewards distinct historically observed:

- pairs
- triples

Hard constraints are checked during selection:

```text
purchase count: 5
maximum use of one number: 4 tickets
maximum overlap between two tickets: 4 numbers
exact operational-history duplicate: prohibited by candidate generation
post-selection number replacement: prohibited
```

## Null Strategy League

The adopted model is compared under the same five-ticket cost with many seeded
variants of:

- random
- balanced random
- frequency weighted
- dormancy weighted
- recent weighted
- hybrid weighted

Diagnostics include:

- normal ROI
- ROI excluding the largest draw payout
- median yearly ROI
- model null-exceedance rate
- CSCV-style Probability of Backtest Overfitting

A failed Null League does not crash prediction generation. It reduces trust in
the affected production model and remains visible in the report.

## Champion / Challenger e-process

Shadow strategies include:

- `generation4`
- `beam_baseline`
- `full`
- `recent`
- `super` when independent
- `regime`
- `random_control`

The default comparison is:

```text
challenger: generation4
champion: beam_baseline
minimum evaluated draws: 30
promotion evidence threshold: e-value 20
```

The bounded utility uses maximum main-number match, total main-number matches and
winning-ticket count. e-process results can be inspected repeatedly without
using an ordinary fixed-horizon p-value as though it remained valid.

## Sealed prediction

Each run hashes:

- final prediction CSV
- Generation 4 summary
- shadow predictions
- source dataset
- adopted models
- Generation 4 builder script
- Git commit and Actions run metadata

Outputs are versioned by target draw and digest. Existing sealed files are not
overwritten.

## Main outputs

```text
outputs/evolution_best_prediction.csv
outputs/holdout/latest_prediction_report.txt
outputs/generation4/latest_generation4_summary.json
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

## Tests

Generation 4 tests cover:

- bounded prior-only Conformal pool
- bounded Change-Point score
- five-ticket quota allocation
- DPP diversity preference
- hard usage and overlap constraints
- no post-selection replacement
- minimum evidence for e-process decisions
- canonical SHA-256 manifest hashing

Run locally:

```bash
python -m unittest \
  tests.test_prediction_output_consistency \
  tests.test_robust_validation_and_portfolio \
  tests.test_generation4_pipeline -v
```
