# LOTO7 Progress Summary

- evaluated_genomes_rows: 6691
- max_generation_seen: 59
- best_score: 20226.079173
- history_files: 8
- state_files: 0
- best_model_files: 13

## Rank counts
- rank_1等: 0
- rank_2等: 89
- rank_3等: 210
- rank_4等: 16126
- rank_5等: 116436
- rank_6等: 189982
- rank_外れ: 6329957

## Best row
```json
{
  "generation": "25",
  "genome_id": "g025_0043_5739",
  "score": "20226.079173",
  "targets": "240",
  "tickets": "1200",
  "max_main_match": "6",
  "rank_1等": "0",
  "rank_2等": "1",
  "rank_3等": "1",
  "rank_4等": "4",
  "rank_5等": "22",
  "rank_6等": "38",
  "rank_外れ": "1134",
  "full_weight": "0.5274346814018912",
  "recent240_weight": "0.026579364077156283",
  "recent120_weight": "0.40396494719927667",
  "recent60_weight": "0.042021007321675734",
  "pair_weight": "0.20578871533139215",
  "pair_recency_weight": "0.026796908234771058",
  "pair_stability_weight": "0.07382408347235558",
  "triple_weight": "0.05281866436331362",
  "dormancy_weight": "0.02733036062759643",
  "odd_bonus": "0.2657106965165",
  "sum_bonus": "0.0873039818895971",
  "low_high_bonus": "0.33626023572845526",
  "consecutive_penalty": "0.1103409568984363",
  "overlap_limit": "4",
  "pool_size": "17",
  "target_sum_min": "95",
  "target_sum_max": "185",
  "max_consecutive_pairs": "1",
  "shard_id": "3",
  "num_shards": "8",
  "completed_at": "2026-06-15T16:38:37.815793+00:00",
  "source_file": "outputs/evolution_history_shard03_of_08.csv"
}
```

## ML reports
- {'model': 'meta_logistic', 'auc': '0.8849713181762944', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'meta_rf', 'auc': '0.8320281860714325', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'xgboost', 'auc': '0.8415529474474973', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'lightgbm', 'auc': '0.8619858123708835', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'catboost', 'auc': '0.8529338453805279', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
