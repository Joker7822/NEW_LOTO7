"""Permutation Null League primitives for payout-independent model testing."""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import wilson_interval
from loto7.evolution.hit_first import hit_first_score


def evaluate_portfolios(
    portfolios: Sequence[Sequence[Sequence[int]]],
    mains: Sequence[Sequence[int]],
) -> dict[str, object]:
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


def _search_trial_seed(parent_seed: int, trial: int) -> int:
    """Derive deterministic independent sub-seeds without consuming another phase seed."""
    payload = f"loto7-fixed-final-search-v2:{int(parent_seed)}:{int(trial)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    return value or 1


def adaptive_null_test(
    *,
    portfolios: Sequence[Sequence[Sequence[int]]],
    mains: Sequence[Sequence[int]],
    seeds: Sequence[int],
    checkpoints: Sequence[int],
    search_width: int = 6,
    max_exceedance: float = 0.10,
) -> dict[str, object]:
    """Run search-adjusted permutation testing while preserving final-seed power.

    Each phase seed is one independent adjusted trial. ``search_width`` internal
    permutations are derived from that seed and their maximum score is used for
    the trial. This keeps multiplicity adjustment but does not collapse six
    final seeds into one observation, so a 150-seed fixed-final phase retains
    150 Wilson observations instead of only 25.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    if search_width <= 0:
        raise ValueError("search_width must be positive")

    observed = evaluate_portfolios(portfolios, mains)
    observed_score = float(observed["hit_first_objective_score"])
    raw_scores: list[float] = []
    adjusted_scores: list[float] = []
    decisions = []
    requested = sorted({int(value) for value in checkpoints if int(value) > 0})
    if not requested:
        raise ValueError("checkpoints must contain a positive value")
    available_checkpoints = [value for value in requested if value <= len(seeds)]
    if not available_checkpoints:
        raise ValueError("no checkpoint fits the supplied fixed seed phase")

    stop = max(available_checkpoints)
    for position, parent_seed in enumerate(seeds[:stop], start=1):
        within_seed_scores: list[float] = []
        for trial in range(search_width):
            shuffled = list(mains)
            random.Random(_search_trial_seed(int(parent_seed), trial)).shuffle(shuffled)
            score = float(
                evaluate_portfolios(portfolios, shuffled)["hit_first_objective_score"]
            )
            raw_scores.append(score)
            within_seed_scores.append(score)
        adjusted_scores.append(max(within_seed_scores))

        if position in available_checkpoints:
            exceedances = sum(value >= observed_score for value in adjusted_scores)
            rate = exceedances / len(adjusted_scores)
            lower, upper = wilson_interval(exceedances, len(adjusted_scores))
            verdict = "continue"
            if upper <= max_exceedance:
                verdict = "pass"
            elif lower > max_exceedance:
                verdict = "fail"
            decisions.append(
                {
                    "simulations": position,
                    "search_adjusted_trials": len(adjusted_scores),
                    "raw_permutation_trials": len(raw_scores),
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

    if not decisions:
        raise RuntimeError("adaptive null test produced no checkpoint decision")
    final = decisions[-1]
    ordered = sorted(adjusted_scores)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "observed_metrics": observed,
        "observed_score": round(observed_score, 6),
        "adaptive_checkpoints": decisions,
        "stopped_at_simulations": stop,
        "search_width": search_width,
        "search_adjustment_method": "within_seed_max",
        "null_distribution": {
            "raw_count": len(raw_scores),
            "search_adjusted_count": len(adjusted_scores),
            "raw_median": round(statistics.median(raw_scores), 6),
            "adjusted_median": round(statistics.median(adjusted_scores), 6),
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
