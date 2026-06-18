# LOTO7 Progress Summary

- evaluated_genomes_rows: 6096
- max_generation_seen: 49
- best_score: 26726.975523
- history_files: 12
- state_files: 12
- best_model_files: 13

## Rank counts
- rank_1等: 0
- rank_2等: 97
- rank_3等: 320
- rank_4等: 17684
- rank_5等: 126332
- rank_6等: 201015
- rank_外れ: 6694102

## Best row
```json
{
  "generation": "1",
  "genome_id": "g001_0123_5401",
  "score": "26726.975523",
  "targets": "620",
  "tickets": "3100",
  "max_main_match": "6",
  "rank_1等": "0",
  "rank_2等": "1",
  "rank_3等": "0",
  "rank_4等": "7",
  "rank_5等": "44",
  "rank_6等": "74",
  "rank_外れ": "2974",
  "full_weight": "0.2278336324152209",
  "recent240_weight": "0.17379981755090654",
  "recent120_weight": "0.5859969873845637",
  "recent60_weight": "0.012369562649308745",
  "pair_weight": "0.15408561388886538",
  "pair_recency_weight": "0.0948697308414499",
  "pair_stability_weight": "0.060217068839834964",
  "triple_weight": "0.08156663360620788",
  "dormancy_weight": "0.03979595958183958",
  "odd_bonus": "0.1791756665039416",
  "sum_bonus": "0.25912556055266467",
  "low_high_bonus": "0.3155572359788559",
  "consecutive_penalty": "0.06632018113690571",
  "overlap_limit": "4",
  "pool_size": "22",
  "target_sum_min": "89",
  "target_sum_max": "170",
  "max_consecutive_pairs": "2",
  "shard_id": "3",
  "num_shards": "4",
  "completed_at": "2026-06-12T02:06:01.156431+00:00",
  "source_file": "outputs/evolution_history_shard03_of_04.csv"
}
```

## ML reports
- {'model': 'meta_logistic', 'auc': '0.8849713181762944', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'meta_rf', 'auc': '0.8320281860714325', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'xgboost', 'auc': '0.8415529474474973', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'lightgbm', 'auc': '0.8619858123708835', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
- {'model': 'catboost', 'auc': '0.8529338453805279', 'test_rows': '5266', 'positive_rate': '0.11849601215343715'}
