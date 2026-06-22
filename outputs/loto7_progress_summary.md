# LOTO7 Progress Summary

- evaluated_genomes_rows: 9089
- max_generation_seen: 59
- best_score: 67991.782386
- history_files: 8
- state_files: 8
- best_model_files: 9

## Rank counts
- rank_1等: 0
- rank_2等: 89
- rank_3等: 212
- rank_4等: 18411
- rank_5等: 137614
- rank_6等: 236562
- rank_外れ: 8178312

## Best row
```json
{
  "generation": "2",
  "genome_id": "g002_0013_8486",
  "score": "67991.782386",
  "targets": "160",
  "tickets": "800",
  "max_main_match": "6",
  "rank_1等": "0",
  "rank_2等": "0",
  "rank_3等": "1",
  "rank_4等": "1",
  "rank_5等": "9",
  "rank_6等": "21",
  "rank_外れ": "768",
  "full_weight": "0.3032937958195276",
  "recent240_weight": "0.3444181065874426",
  "recent120_weight": "0.2328013140826817",
  "recent60_weight": "0.11948678351034817",
  "pair_weight": "0.08323802332455774",
  "pair_recency_weight": "0.09639309396022575",
  "pair_stability_weight": "0.1595570365521473",
  "triple_weight": "0.012776004750939308",
  "dormancy_weight": "0.04916003393539038",
  "odd_bonus": "0.3571829368097169",
  "sum_bonus": "0.5785715541028872",
  "low_high_bonus": "0.2661977650130201",
  "consecutive_penalty": "0.2307354823692875",
  "overlap_limit": "4",
  "pool_size": "17",
  "target_sum_min": "103",
  "target_sum_max": "175",
  "max_consecutive_pairs": "1",
  "shard_id": "5",
  "num_shards": "8",
  "completed_at": "2026-06-19T13:38:41.345971+00:00",
  "source_file": "outputs/evolution_history_shard05_of_08.csv"
}
```

## ML reports
- {'model': 'meta_logistic', 'auc': '0.8860812370883461', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'meta_rf', 'auc': '0.8323637509804571', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'xgboost', 'auc': '0.8493035647212188', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'lightgbm', 'auc': '0.8691869938907854', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'catboost', 'auc': '0.852960428197395', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
