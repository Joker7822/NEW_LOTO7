#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation 5 walk-forward Pareto/island evolution for LOTO7."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.evolution.generation5 import (  # noqa: E402
    ISLANDS,
    OBJECTIVE_NAME,
    OBJECTIVE_VERSION,
    aggregate_walk_forward_metrics,
    build_walk_forward_folds,
    file_sha256,
    generation5_adoption_gate,
    select_pareto_records,
    stable_seed,
    stage_key,
    successive_halving_counts,
)
from loto7.evolution.hit_first import evaluate_model_on_holdout  # noqa: E402
from loto7_evolution_trainer import (  # noqa: E402
    Genome,
    crossover,
    genome_from_dict,
    load_draws,
    mutate,
    normalize_weights,
    random_genome,
)
from merge_evolution_shards import load_prize_rows, select_target_indices  # noqa: E402

HISTORY_FIELDS = [
    "created_at",
    "generation",
    "island",
    "genome_id",
    "generation5_score",
    "fold_objective_median",
    "fold_objective_min",
    "average_max_main_match",
    "draw_main4_plus_rate_percent",
    "draw_main5_plus_count",
    "draw_main6_plus_count",
    "average_portfolio_unique_numbers",
    "mean_ticket_pair_overlap",
    "max_ticket_pair_overlap",
    "payout_roi_percent",
    "top1_payout_share",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_genome(path: str) -> Optional[Genome]:
    try:
        payload = read_json(path)
        raw = payload.get("genome", payload)
        return genome_from_dict(raw) if isinstance(raw, dict) else None
    except Exception as exc:
        print(f"[WARN] cannot load seed model {path}: {exc}")
        return None


def load_seed_genomes(patterns: Sequence[str]) -> List[Genome]:
    result: List[Genome] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            genome = load_genome(path)
            if genome is None or genome.id in seen:
                continue
            seen.add(genome.id)
            result.append(genome)
    return result


def clone_genome(genome: Genome, *, genome_id: str, generation: int) -> Genome:
    payload = asdict(genome)
    payload.update(
        {
            "id": genome_id,
            "generation": generation,
            "score": 0.0,
            "max_main_match": 0,
            "best_rank_count": 0,
        }
    )
    return Genome(**payload)


def specialize_genome(
    genome: Genome,
    *,
    island: str,
    generation: int,
    index: int,
    rng: random.Random,
) -> Genome:
    candidate = clone_genome(
        genome,
        genome_id=f"g5_{island}_{generation:03d}_{index:04d}_{rng.randint(1000, 9999)}",
        generation=generation,
    )
    data = asdict(candidate)
    data["overlap_limit"] = 4
    data["pool_size"] = max(18, min(26, int(data["pool_size"])))

    if island == "average_max":
        data["pair_stability_weight"] = min(
            0.35, float(data["pair_stability_weight"]) * 1.08 + 0.005
        )
        data["recent120_weight"] = float(data["recent120_weight"]) * 1.05
    elif island == "draw4":
        data["pair_weight"] = min(0.35, float(data["pair_weight"]) * 1.10 + 0.005)
        data["pair_recency_weight"] = min(
            0.40, float(data["pair_recency_weight"]) * 1.10 + 0.005
        )
    elif island == "high_match":
        data["pair_weight"] = min(0.35, float(data["pair_weight"]) * 1.12 + 0.008)
        data["triple_weight"] = min(0.18, float(data["triple_weight"]) * 1.18 + 0.005)
        data["pool_size"] = max(18, min(22, int(data["pool_size"])))
    elif island == "robust_diversity":
        data["pool_size"] = max(22, int(data["pool_size"]))
        data["pair_weight"] = max(0.0, float(data["pair_weight"]) * 0.92)
        data["triple_weight"] = max(0.0, float(data["triple_weight"]) * 0.90)
        weights = normalize_weights(
            [
                max(0.15, float(data["full_weight"])),
                max(0.15, float(data["recent240_weight"])),
                max(0.15, float(data["recent120_weight"])),
                max(0.10, float(data["recent60_weight"])),
            ]
        )
        (
            data["full_weight"],
            data["recent240_weight"],
            data["recent120_weight"],
            data["recent60_weight"],
        ) = weights
    else:
        raise ValueError(f"unknown island: {island}")

    data.update(
        {
            "id": candidate.id,
            "generation": generation,
            "score": 0.0,
            "max_main_match": 0,
            "best_rank_count": 0,
        }
    )
    return Genome(**data)


def make_candidate(
    parents: Sequence[Genome],
    *,
    island: str,
    generation: int,
    index: int,
    rng: random.Random,
    mutation_intensity: int,
    random_reset_rate: float,
) -> Genome:
    if not parents or rng.random() < random_reset_rate:
        candidate = random_genome(generation, index, rng)
    elif len(parents) >= 2 and rng.random() < 0.45:
        left, right = rng.sample(list(parents), 2)
        candidate = crossover(left, right, generation, index, rng)
    else:
        candidate = rng.choice(list(parents))
        for step in range(max(1, mutation_intensity)):
            candidate = mutate(candidate, generation, index * 10 + step, rng)
    return specialize_genome(
        candidate,
        island=island,
        generation=generation,
        index=index,
        rng=rng,
    )


def evaluate_metrics(
    genome: Genome,
    *,
    label: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    return evaluate_model_on_holdout(
        genome=genome,
        model_path=label,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=purchase_count,
        unit_cost=unit_cost,
    )


def evaluate_full_record(
    genome: Genome,
    *,
    island: str,
    generation: int,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    folds: Sequence[object],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    full_metrics = evaluate_metrics(
        genome,
        label=f"generation5:{island}:{generation}:{genome.id}:full",
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=purchase_count,
        unit_cost=unit_cost,
    )
    fold_metrics: List[Dict[str, object]] = []
    for fold in folds:
        metrics = evaluate_metrics(
            genome,
            label=f"generation5:{island}:{generation}:{genome.id}:{fold.label}",
            draws=draws,
            prize_rows=prize_rows,
            target_indices=fold.target_indices,
            purchase_count=purchase_count,
            unit_cost=unit_cost,
        )
        metrics["fold"] = fold.label
        metrics["first_target_index"] = int(fold.target_indices[0])
        metrics["last_target_index"] = int(fold.target_indices[-1])
        fold_metrics.append(metrics)
    walk_forward = aggregate_walk_forward_metrics(fold_metrics)
    return {
        "genome_id": genome.id,
        "island": island,
        "generation": generation,
        "_genome": genome,
        "full_metrics": full_metrics,
        "fold_metrics": fold_metrics,
        "walk_forward": walk_forward,
    }


def public_record(record: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in record.items() if key != "_genome"}


def append_history(path: str, record: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists() and target.stat().st_size > 0
    walk = record.get("walk_forward", {})
    full = record.get("full_metrics", {})
    if not isinstance(walk, Mapping) or not isinstance(full, Mapping):
        return
    row = {
        "created_at": now_iso(),
        "generation": record.get("generation"),
        "island": record.get("island"),
        "genome_id": record.get("genome_id"),
        "generation5_score": walk.get("generation5_score"),
        "fold_objective_median": walk.get("fold_objective_median"),
        "fold_objective_min": walk.get("fold_objective_min"),
        "average_max_main_match": walk.get("average_max_main_match"),
        "draw_main4_plus_rate_percent": walk.get("draw_main4_plus_rate_percent"),
        "draw_main5_plus_count": walk.get("draw_main5_plus_count"),
        "draw_main6_plus_count": walk.get("draw_main6_plus_count"),
        "average_portfolio_unique_numbers": walk.get(
            "average_portfolio_unique_numbers"
        ),
        "mean_ticket_pair_overlap": walk.get("mean_ticket_pair_overlap"),
        "max_ticket_pair_overlap": walk.get("max_ticket_pair_overlap"),
        "payout_roi_percent": full.get("payout_roi_percent"),
        "top1_payout_share": full.get("top1_payout_share"),
    }
    with target.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generation 5 LOTO7 precision evolution"
    )
    result.add_argument("--csv", default="loto7.csv")
    result.add_argument("--best-model", default="loto7_best_model.json")
    result.add_argument(
        "--seed-patterns",
        nargs="*",
        default=[
            "loto7_best_model.json",
            "outputs/model_self_evolution/standalone_best_candidate_model.json",
            "outputs/recent_era/recent_era_best_model.json",
            "outputs/super_recent/super_recent_best_model.json",
        ],
    )
    result.add_argument(
        "--candidate-model",
        default="outputs/generation5/generation5_candidate_model.json",
    )
    result.add_argument(
        "--summary", default="outputs/generation5/generation5_summary.json"
    )
    result.add_argument(
        "--report", default="outputs/generation5/generation5_report.txt"
    )
    result.add_argument(
        "--history", default="outputs/generation5/generation5_history.csv"
    )
    result.add_argument("--generations", type=int, default=4)
    result.add_argument("--island-population", type=int, default=4)
    result.add_argument("--archive-size", type=int, default=8)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--stage1-targets", type=int, default=104)
    result.add_argument("--stage2-targets", type=int, default=260)
    result.add_argument("--retention-rate", type=float, default=0.5)
    result.add_argument("--migration-interval", type=int, default=2)
    result.add_argument("--stagnation-intensify", type=int, default=8)
    result.add_argument("--stagnation-reset", type=int, default=12)
    result.add_argument("--purchase-count", type=int, default=5)
    result.add_argument("--unit-cost", type=int, default=300)
    result.add_argument("--min-train-draws", type=int, default=52)
    result.add_argument("--holdout-start-draw", type=int, default=2)
    result.add_argument("--max-targets", type=int, default=0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-runtime-minutes", type=float, default=300.0)
    result.add_argument("--safe-exit-minutes", type=float, default=20.0)
    result.add_argument("--min-positive-folds", type=int, default=3)
    result.add_argument("--min-fold-objective-delta", type=float, default=0.05)
    result.add_argument("--min-average-max-delta", type=float, default=0.03)
    result.add_argument(
        "--min-draw4-rate-delta-percent", type=float, default=0.50
    )
    result.add_argument("--min-draw5-count-delta", type=int, default=0)
    result.add_argument("--min-draw6-count-delta", type=int, default=0)
    result.add_argument(
        "--max-worst-fold-drop-percent", type=float, default=2.0
    )
    result.add_argument(
        "--min-average-unique-numbers", type=float, default=13.0
    )
    result.add_argument("--max-mean-overlap", type=float, default=4.2)
    result.add_argument("--max-pair-overlap", type=int, default=4)
    result.add_argument("--min-payout-roi-percent", type=float, default=8.0)
    result.add_argument("--max-roi-drop-percent", type=float, default=5.0)
    result.add_argument("--max-top1-payout-share", type=float, default=0.50)
    return result


def write_report(path: str, payload: Mapping[str, object]) -> None:
    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    adoption = payload.get("adoption", {})
    lines = [
        "LOTO7 Generation 5 Precision Evolution",
        "========================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"status: {payload.get('status')}",
        f"objective: {OBJECTIVE_NAME}",
        f"objective_version: {OBJECTIVE_VERSION}",
        f"stable_seed: {payload.get('stable_seed')}",
        f"generations_completed: {payload.get('generations_completed')}",
        f"candidates_fully_evaluated: {payload.get('candidates_fully_evaluated')}",
        "",
        "[Baseline]",
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Candidate]",
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Adoption Gate]",
        json.dumps(adoption, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "Generation 5 never treats ROI or profit as learning rewards.",
        "Lottery drawings remain highly random and future winnings are not guaranteed.",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.generations <= 0 or args.island_population <= 0:
        raise SystemExit("generations and island-population must be positive")
    if args.purchase_count != 5:
        raise SystemExit("Generation 5 currently requires exactly five tickets")

    started = time.monotonic()
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=None,
    )
    if args.max_targets > 0:
        target_indices = target_indices[-args.max_targets :]
    if len(target_indices) < args.folds:
        raise SystemExit("not enough target draws for Generation 5 folds")

    baseline = load_genome(args.best_model)
    if baseline is None:
        raise SystemExit(f"baseline model is missing: {args.best_model}")
    seed_models = load_seed_genomes(args.seed_patterns)
    if all(item.id != baseline.id for item in seed_models):
        seed_models.insert(0, baseline)

    dataset_sha = file_sha256(args.csv)
    model_sha = file_sha256(args.best_model)
    stable = args.seed or stable_seed(dataset_sha, model_sha, OBJECTIVE_VERSION)
    rng = random.Random(stable)
    folds = build_walk_forward_folds(target_indices, fold_count=args.folds)
    stage1_indices = target_indices[
        -min(len(target_indices), args.stage1_targets) :
    ]
    stage2_indices = target_indices[
        -min(len(target_indices), args.stage2_targets) :
    ]
    halving_counts = successive_halving_counts(
        args.island_population,
        retention_rate=args.retention_rate,
        stages=3,
    )

    baseline_record = evaluate_full_record(
        baseline,
        island="baseline",
        generation=0,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        folds=folds,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )

    archives: Dict[str, List[Mapping[str, object]]] = {
        island: [] for island in ISLANDS
    }
    parent_pools: Dict[str, List[Genome]] = {
        island: list(seed_models) for island in ISLANDS
    }
    stagnation: Dict[str, int] = {island: 0 for island in ISLANDS}
    best_scores: Dict[str, float] = {island: -math.inf for island in ISLANDS}
    evaluated_count = 0
    generations_completed = 0
    status = "completed"

    for generation in range(1, args.generations + 1):
        elapsed = (time.monotonic() - started) / 60.0
        if elapsed >= max(
            0.0, args.max_runtime_minutes - args.safe_exit_minutes
        ):
            status = f"safe_exit_at_{elapsed:.2f}_minutes"
            break

        migrants = [
            record.get("_genome")
            for island in ISLANDS
            for record in archives[island][:1]
            if isinstance(record.get("_genome"), Genome)
        ]
        for island_index, island in enumerate(ISLANDS):
            parents = list(parent_pools[island])
            if generation % max(1, args.migration_interval) == 0:
                parents.extend(
                    item for item in migrants if isinstance(item, Genome)
                )

            intensity = 1
            random_reset_rate = 0.10
            if stagnation[island] >= args.stagnation_intensify:
                intensity = 3
                random_reset_rate = 0.20
            if stagnation[island] >= args.stagnation_reset:
                intensity = 5
                random_reset_rate = 0.30

            batch = [
                make_candidate(
                    parents,
                    island=island,
                    generation=generation,
                    index=island_index * 10_000 + index,
                    rng=rng,
                    mutation_intensity=intensity,
                    random_reset_rate=random_reset_rate,
                )
                for index in range(args.island_population)
            ]

            stage1: List[Tuple[Genome, Dict[str, object]]] = []
            for candidate in batch:
                metrics = evaluate_metrics(
                    candidate,
                    label=f"generation5:{island}:{generation}:{candidate.id}:stage1",
                    draws=draws,
                    prize_rows=prize_rows,
                    target_indices=stage1_indices,
                    purchase_count=args.purchase_count,
                    unit_cost=args.unit_cost,
                )
                stage1.append((candidate, metrics))
            stage1.sort(
                key=lambda item: stage_key(item[1], island), reverse=True
            )
            survivors1 = stage1[: halving_counts[1]]

            stage2: List[Tuple[Genome, Dict[str, object]]] = []
            for candidate, _metrics in survivors1:
                metrics = evaluate_metrics(
                    candidate,
                    label=f"generation5:{island}:{generation}:{candidate.id}:stage2",
                    draws=draws,
                    prize_rows=prize_rows,
                    target_indices=stage2_indices,
                    purchase_count=args.purchase_count,
                    unit_cost=args.unit_cost,
                )
                stage2.append((candidate, metrics))
            stage2.sort(
                key=lambda item: stage_key(item[1], island), reverse=True
            )
            survivors2 = stage2[: halving_counts[2]]

            generation_records: List[Mapping[str, object]] = []
            for candidate, _metrics in survivors2:
                record = evaluate_full_record(
                    candidate,
                    island=island,
                    generation=generation,
                    draws=draws,
                    prize_rows=prize_rows,
                    target_indices=target_indices,
                    folds=folds,
                    purchase_count=args.purchase_count,
                    unit_cost=args.unit_cost,
                )
                generation_records.append(record)
                evaluated_count += 1
                append_history(args.history, record)

            archives[island] = select_pareto_records(
                [*archives[island], *generation_records],
                limit=args.archive_size,
                island=island,
            )
            parent_pools[island] = [
                record["_genome"]
                for record in archives[island]
                if isinstance(record.get("_genome"), Genome)
            ] or parents
            current_score = max(
                (
                    float(
                        record.get("walk_forward", {}).get(
                            "generation5_score", -math.inf
                        )
                    )
                    for record in archives[island]
                ),
                default=-math.inf,
            )
            if current_score > best_scores[island] + 1e-12:
                best_scores[island] = current_score
                stagnation[island] = 0
            else:
                stagnation[island] += 1
        generations_completed = generation
        print(
            json.dumps(
                {
                    "generation": generation,
                    "evaluated": evaluated_count,
                    "archive_sizes": {
                        key: len(value) for key, value in archives.items()
                    },
                    "best_scores": best_scores,
                    "stagnation": stagnation,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    all_records = [
        record for island in ISLANDS for record in archives[island]
    ]
    if not all_records:
        raise SystemExit("Generation 5 produced no fully evaluated candidate")
    global_front = select_pareto_records(
        all_records,
        limit=max(1, len(all_records)),
        island="robust_diversity",
    )
    best_record = max(
        global_front,
        key=lambda record: (
            float(
                record.get("walk_forward", {}).get(
                    "generation5_score", 0.0
                )
            ),
            int(
                record.get("walk_forward", {}).get(
                    "draw_main6_plus_count", 0
                )
            ),
            int(
                record.get("walk_forward", {}).get(
                    "draw_main5_plus_count", 0
                )
            ),
            float(
                record.get("walk_forward", {}).get(
                    "average_max_main_match", 0.0
                )
            ),
            -float(
                record.get("walk_forward", {}).get(
                    "mean_ticket_pair_overlap", 7.0
                )
            ),
        ),
    )
    best_genome = best_record.get("_genome")
    if not isinstance(best_genome, Genome):
        raise SystemExit("best Generation 5 genome is missing")

    adoption = generation5_adoption_gate(
        best_record,
        baseline_record,
        min_positive_folds=args.min_positive_folds,
        min_fold_objective_delta=args.min_fold_objective_delta,
        min_average_max_delta=args.min_average_max_delta,
        min_draw4_rate_delta_percent=args.min_draw4_rate_delta_percent,
        min_draw5_count_delta=args.min_draw5_count_delta,
        min_draw6_count_delta=args.min_draw6_count_delta,
        max_worst_fold_drop_percent=args.max_worst_fold_drop_percent,
        min_average_unique_numbers=args.min_average_unique_numbers,
        max_mean_overlap=args.max_mean_overlap,
        max_pair_overlap=args.max_pair_overlap,
        min_payout_roi_percent=args.min_payout_roi_percent,
        max_roi_drop_percent=args.max_roi_drop_percent,
        max_top1_payout_share=args.max_top1_payout_share,
    )

    candidate_payload: Dict[str, object] = {
        "updated_at": now_iso(),
        "kind": "loto7_generation5_candidate_model",
        "source": "generation5_evolver",
        "objective": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "dataset_sha256": dataset_sha,
        "baseline_model_sha256": model_sha,
        "stable_seed": stable,
        "selection_mode": (
            "five_fold_walk_forward_pareto_four_island_successive_halving"
        ),
        "genome": asdict(best_genome),
        "selected_holdout": best_record.get("full_metrics"),
        "walk_forward": best_record.get("walk_forward"),
        "fold_metrics": best_record.get("fold_metrics"),
        "adoption_gate": adoption,
        "notes": [
            "ROI and profit are excluded from candidate ranking and used only as safety gates.",
            "The candidate is not promoted until the fixed final Null League also passes.",
            "Historical evaluation does not guarantee future lottery winnings.",
        ],
    }
    write_json(args.candidate_model, candidate_payload)

    summary: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_generation5_evolution",
        "status": status,
        "objective": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "csv": args.csv,
        "dataset_sha256": dataset_sha,
        "baseline_model_sha256": model_sha,
        "stable_seed": stable,
        "target_draws": len(target_indices),
        "fold_count": len(folds),
        "folds": [
            {
                "label": fold.label,
                "target_count": len(fold.target_indices),
                "first_target_index": fold.target_indices[0],
                "last_target_index": fold.target_indices[-1],
            }
            for fold in folds
        ],
        "successive_halving": {
            "stage_targets": [
                len(stage1_indices),
                len(stage2_indices),
                len(target_indices),
            ],
            "candidate_counts_per_island": halving_counts,
            "retention_rate": args.retention_rate,
        },
        "islands": list(ISLANDS),
        "generations_requested": args.generations,
        "generations_completed": generations_completed,
        "island_population": args.island_population,
        "candidates_fully_evaluated": evaluated_count,
        "archive_sizes": {
            key: len(value) for key, value in archives.items()
        },
        "baseline": public_record(baseline_record),
        "candidate": public_record(best_record),
        "adoption": adoption,
        "candidate_model": args.candidate_model,
        "archive": {
            island: [public_record(record) for record in archives[island]]
            for island in ISLANDS
        },
    }
    write_json(args.summary, summary)
    write_report(args.report, summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "archive"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
