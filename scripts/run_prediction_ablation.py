#!/usr/bin/env python3
"""Separate model ranking value from portfolio diversification value."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.model_audit import year_of
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evolution.hit_first import hit_first_score
from loto7_evolution_trainer import generate_tickets, load_draws
from merge_evolution_shards import select_target_indices
from scripts.robust_model_metrics import load_genome


def random_portfolio(rng: random.Random) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    attempts = 0
    while len(result) < 5 and attempts < 5000:
        attempts += 1
        ticket = tuple(sorted(rng.sample(range(1, 38), 7)))
        if all(len(set(ticket).intersection(other)) <= 4 for other in result):
            result.append(ticket)
    if len(result) != 5:
        raise RuntimeError("failed to construct diversified random portfolio")
    return result


def metrics(
    portfolios: Sequence[Sequence[Sequence[int]]],
    mains: Sequence[Sequence[int]],
) -> Dict[str, object]:
    maximums = []
    ticket_matches = []
    for portfolio, main in zip(portfolios, mains):
        target = set(main)
        values = [len(target.intersection(ticket)) for ticket in portfolio]
        maximums.append(max(values, default=0))
        ticket_matches.extend(values)
    result = summarize_hit_metrics(
        maximums,
        ticket_main_matches=ticket_matches,
        portfolios=portfolios,
    )
    result.update(summarize_portfolio_ranking(portfolios, mains))
    result["hit_first_objective_score"] = hit_first_score(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="outputs/generation5/generation5_candidate_model.json")
    parser.add_argument("--output", default="outputs/generation5/prediction_ablation.json")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    indices = select_target_indices(
        draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None
    )
    indices = [index for index in indices if year_of(draws[index]) >= args.start_year]
    genome = load_genome(args.model)
    rng = random.Random(args.seed)
    learned = [generate_tickets(draws[:index], genome, 5) for index in indices]
    random_only = [random_portfolio(rng) for _ in indices]
    learned_pool = [generate_tickets(draws[:index], genome, 20) for index in indices]
    learned_randomized = [rng.sample(list(portfolio), 5) for portfolio in learned_pool]
    mains = [draws[index].main for index in indices]
    a = metrics(learned, mains)
    b = metrics(random_only, mains)
    c = metrics(learned_randomized, mains)
    payload = {
        "kind": "loto7_model_portfolio_ablation",
        "target_draws": len(indices),
        "A_learned_model_and_optimizer": a,
        "B_random_scores_and_diversified_portfolio": b,
        "C_learned_model_and_randomized_portfolio": c,
        "effects": {
            "model_ranking_value_A_minus_B": round(
                float(a["hit_first_objective_score"])
                - float(b["hit_first_objective_score"]),
                6,
            ),
            "portfolio_optimizer_value_A_minus_C": round(
                float(a["hit_first_objective_score"])
                - float(c["hit_first_objective_score"]),
                6,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["effects"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
