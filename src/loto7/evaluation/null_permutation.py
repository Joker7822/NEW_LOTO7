"""Permutation Null League primitives for payout-independent model testing."""
from __future__ import annotations

import random
import statistics
from typing import Dict, Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import wilson_interval
from loto7.evolution.hit_first import hit_first_score


def evaluate_portfolios(
    portfolios: Sequence[Sequence[Sequence[int]]],
    mains: Sequence[Sequence[int]],
) -> Dict[str, object]:
    maximums = []
    ticket_matches = []
    for portfolio, main in zip(portfolios, mains):
        target = set(int(value) for value in main)
        matches = [len(target.intersection(int(value) for value in ticket)) for ticket in portfolio]
        ticket_matches.extend(matches)
        maximums.append(max(matches, default=0))
    metrics = summarize_hit_metrics(
        maximums,
        ticket_main_matches=ticket_matches,
        portfolios=portfolios,
    )
    metrics.update(summarize_portfolio_ranking(portfolios, mains))
    metrics["hit_first_objective_score"] = hit_first_score(metrics)
    return metrics


def adaptive_null_test(
    *,
    portfolios: Sequence[Sequence[Sequence[int]]],
    mains: Sequence[Sequence[int]],
    seeds: Sequence[int],
    checkpoints: Sequence[int],
    search_width: int = 6,
    max_exceedance: float = 0.10,
) -> Dict[str, object]:
    observed = evaluate_portfolios(portfolios, mains)
    observed_score = float(observed["hit_first_objective_score"])
    raw_scores: list[float] = []
    adjusted_scores: list[float] = []
    decisions = []
    stop = max(checkpoints)
    for position, seed in enumerate(seeds[: max(checkpoints)], start=1):
        shuffled = list(mains)
        random.Random(int(seed)).shuffle(shuffled)
        raw_scores.append(
            float(evaluate_portfolios(portfolios, shuffled)["hit_first_objective_score"])
        )
        if len(raw_scores) % max(1, search_width) == 0:
            adjusted_scores.append(max(raw_scores[-search_width:]))
        if position in checkpoints:
            comparison = adjusted_scores or raw_scores
            exceedances = sum(value >= observed_score for value in comparison)
            rate = exceedances / len(comparison)
            lower, upper = wilson_interval(exceedances, len(comparison))
            verdict = "continue"
            if upper <= max_exceedance:
                verdict = "pass"
            elif lower > max_exceedance:
                verdict = "fail"
            decisions.append(
                {
                    "simulations": position,
                    "search_adjusted_trials": len(comparison),
                    "exceedances": exceedances,
                    "exceedance": round(rate, 6),
                    "wilson_ci_lower": round(lower, 6),
                    "wilson_ci_upper": round(upper, 6),
                    "verdict": verdict,
                }
            )
            if verdict != "continue":
                stop = position
                break
    comparison = adjusted_scores or raw_scores
    final = decisions[-1]
    ordered = sorted(comparison)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "observed_metrics": observed,
        "observed_score": round(observed_score, 6),
        "adaptive_checkpoints": decisions,
        "stopped_at_simulations": stop,
        "search_width": search_width,
        "null_distribution": {
            "raw_count": len(raw_scores),
            "search_adjusted_count": len(adjusted_scores),
            "raw_median": round(statistics.median(raw_scores), 6),
            "adjusted_median": round(statistics.median(comparison), 6),
            "adjusted_p90": round(ordered[p90_index], 6),
        },
        "decision": {
            "passed": final["verdict"] == "pass",
            "max_null_exceedance": max_exceedance,
            "exceedance": final["exceedance"],
            "wilson_ci_upper": final["wilson_ci_upper"],
            "reason": "upper Wilson bound must be at or below the threshold",
        },
    }
