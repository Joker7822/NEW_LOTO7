# LOTO7 Self Evolution Report

generated_at: 2026-06-15T20:25:43.206385+00:00

## Distributed Integration
- history rows: 3625
- top score: 26726.975523
- max generation: 34

## Champion Breeding
- parent count: 12
- child count: 64
- top parent score: 26726.97552347337

## Reinforcement Policy
- last action: diversity
- probabilities: `{"explore": 0.24403250715541022, "exploit": 0.24403250715541022, "diversity": 0.26790247853376925, "roi": 0.24403250715541022}`
- reward: `{"total_reward": 0.3912141372656682, "score_reward": 0.7891139888170506, "roi_reward": -0.48733333333333334, "match_reward": 0.7142857142857143, "top_score": 26726.975523, "roi": -0.48733333333333334, "max_match": 5.0}`

## Next Config
```json
{
  "evolution_mode": "recommended",
  "generations": 100,
  "population": 240,
  "max_targets": 240,
  "target_stride": 2,
  "mutation_mode": "diversity",
  "seed_genome_file": "outputs/self_evolution/breeder_seed_genomes.json",
  "bred_child_count": 64,
  "policy_action": "diversity",
  "reward": {
    "total_reward": 0.3912141372656682,
    "score_reward": 0.7891139888170506,
    "roi_reward": -0.48733333333333334,
    "match_reward": 0.7142857142857143,
    "top_score": 26726.975523,
    "roi": -0.48733333333333334,
    "max_match": 5.0
  }
}
```
