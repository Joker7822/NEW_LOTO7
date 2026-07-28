# LOTO7 Generation 5 Precision Evolution

Generation 5 is the guarded model-improvement layer for `NEW_LOTO7`.
It is designed to improve main-number agreement on unseen chronological periods,
not to maximize historical payout or profit.

> LOTO7 drawings are highly random. No historical method can guarantee a future
> winning ticket, profit, or predictive advantage.

## Ownership

| Responsibility | Owner |
|---|---|
| Candidate search | `LOTO7 Model Self Evolution` and Recent/Super Recent workflows |
| Precision evolution and automatic model promotion | `LOTO7 Generation 5 Precision Evolution` |
| Generation 4 portfolio diagnostics | `LOTO7 Generation 4 Evaluation` |
| Production prediction publication | `LOTO7 Production Prediction Publisher` |

Scheduled legacy self-evolution runs no longer apply a model directly. Automatic
replacement of `loto7_best_model.json` is performed only by Generation 5 after
all internal and final fixed-Null gates pass. A legacy manual direct-apply option
remains available only through an explicit workflow-dispatch choice.

## Search architecture

Generation 5 uses four independent islands:

1. `average_max` — average maximum main-number agreement.
2. `draw4` — draw-level four-or-more main-number reach.
3. `high_match` — five- and six-number agreement.
4. `robust_diversity` — worst chronological fold and five-ticket diversity.

Each island uses mutation, crossover, random restart, Pareto archives and periodic
migration. Stagnation increases mutation intensity; prolonged stagnation also
increases the random reset rate.

## Successive halving

Candidates are evaluated in stages:

```text
Stage 1: latest 104 targets
Stage 2: latest 260 targets
Stage 3: complete selected holdout + five chronological folds
```

The default island population is reduced `4 -> 2 -> 1`, allowing broad cheap
screening before expensive complete walk-forward evaluation.

## Five-fold internal walk-forward objective

The target sequence is partitioned chronologically into five disjoint folds. Each
ticket is generated only from draws preceding its target draw.

The Generation 5 scalar is used only after Pareto filtering:

```text
30% fold objective median
25% worst fold objective
20% draw-level 4+ rate
15% average maximum main match
10% 5+/6+ high-match component
```

ROI, profit and prize payouts never add learning points.

## Internal adoption gate

A candidate must satisfy all defaults:

```text
Improved folds: at least 3 of 5
Per-fold material improvement: at least +0.05
Average maximum main match delta: at least +0.03
Draw-level 4+ rate delta: at least +0.50 percentage points
Draw-level 5+ count: no regression
Draw-level 6+ count: no regression
Worst-fold drop: no more than 2%
Average portfolio unique numbers: at least 13
Mean ticket-pair overlap: at most 4.2
Maximum ticket-pair overlap: at most 4
Payout ROI: safety floor only
Largest-payout share: at most 50%
```

Failure is fail-closed: the candidate remains diagnostic evidence and the current
best model is unchanged.

## Fixed Null Seed Bank

`outputs/generation5/null_seed_bank.json` is derived from:

- dataset SHA-256
- evaluator version
- seed-bank version
- phase name and position

The bank contains disjoint phases:

```text
learning: 700 seeds
selection: 150 seeds
final: 150 seeds
```

The final phase is not exposed to candidate search. Model promotion requires the
fixed final Null Strategy League to pass:

```text
Null exceedance <= 10%
PBO <= 40%
```

The same dataset and evaluator version therefore produce the same final Null
comparison instead of depending on `GITHUB_RUN_ID`.

## Promotion

`scripts/promote_generation5_candidate.py` verifies:

- Generation 5 internal gate passed
- fixed final Null League passed
- candidate genome ID differs from the baseline
- candidate file SHA differs from the baseline
- objective version is present

Only then may it replace `loto7_best_model.json`. Production tickets are still
created exclusively by `LOTO7 Production Prediction Publisher`.

## Outputs

```text
outputs/generation5/generation5_candidate_model.json
outputs/generation5/generation5_summary.json
outputs/generation5/generation5_report.txt
outputs/generation5/generation5_history.csv
outputs/generation5/null_seed_bank.json
outputs/generation5/null_strategy_league_summary.json
outputs/generation5/null_strategy_league_report.txt
outputs/generation5/promotion_decision.json
outputs/generation5/promotion_report.txt
```

Canonical mirror:

```text
outputs/diagnostics/generation5/
```

## Validation

```bash
python -m pip install -e .
python -m unittest tests.test_generation5_precision -v
```

The workflow also compiles every Generation 5 module and independently rebuilds
the seed bank twice to verify deterministic, disjoint output.
