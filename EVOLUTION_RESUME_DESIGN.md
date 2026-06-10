# Evolution Resume / Chunk / Parallel Design

Implemented operational design:

## Resume

Store:
- outputs/evolution_state.json
- current_generation
- best_score
- best_model_sha
- completed_genomes

Restart logic:
- Resume from last completed generation.
- Skip completed genomes.

## Mid-generation Save

Every genome evaluation:
- append outputs/evolution_history.csv
- update outputs/evolution_state.json
- git commit/push

## Chunk Execution

Split population:
- chunk_01
- chunk_02
- chunk_03
- chunk_04

Each workflow evaluates a subset.
Merge results into evolution_history.csv.

## Parallel Evaluation

GitHub Actions matrix:

matrix:
  shard: [0,1,2,3]

Each shard evaluates population where:
index % 4 == shard

Expected speedup:
~4x

## Full Power Mode

100 generations
100 population
4 shards
resume enabled
per genome checkpoint enabled
