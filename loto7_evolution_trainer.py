#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for the high-match-first LOTO7 evolution trainer.

Core ticket generation and Genome operations remain in
``_loto7_evolution_trainer_impl.py``. Walk-forward candidate evaluation and
survivor selection use the payout-independent high-match objective.

Generation 5 candidates use a calibrated number-ranking layer that shrinks
short-window noise toward the lottery base rate and rewards agreement across
multiple time windows. Existing approved/legacy models keep the original
number-scoring path unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import _loto7_evolution_trainer_impl as _impl
from _loto7_evolution_trainer_impl import *  # noqa: F401,F403
from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evolution.hit_first import (
    OBJECTIVE_NAME,
    OBJECTIVE_VERSION,
    diversity_quality_score,
    hit_first_score,
    match_quality_score,
)

_LEGACY_BLEND_NUMBER_SCORES = _impl.blend_number_scores
_GENERATION5_PREFIX = "g5_"
_BASE_MAIN_PROBABILITY = 7.0 / 37.0
_POSTERIOR_PRIOR_STRENGTH = 24.0
_WINDOW_DISAGREEMENT_PENALTY = 0.18


def _segment_bounds(length: int, count: int = 4) -> List[Tuple[int, int]]:
    if length <= 0:
        return []
    count = min(max(1, count), length)
    result: List[Tuple[int, int]] = []
    for segment in range(count):
        start = math.floor(length * segment / count)
        end = math.floor(length * (segment + 1) / count)
        if end > start:
            result.append((start, end))
    return result


def _standardize_cross_section(values: Mapping[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    ordered = [float(values[n]) for n in _impl.NUMBERS]
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    std = math.sqrt(max(0.0, variance))
    if std <= 1e-12:
        return {n: 0.0 for n in _impl.NUMBERS}
    return {n: (float(values[n]) - mean) / std for n in _impl.NUMBERS}


def _posterior_window_signal(
    draws: Sequence[object],
    decay: float,
) -> Dict[int, float]:
    """Return shrinkage-calibrated per-number evidence for one time window.

    Each number is treated as a Bernoulli event with prior probability 7/37.
    The prior prevents a short noisy streak from dominating merely because the
    window is small. No explicit "overdue number" bonus is added here.
    """
    weighted_hits = {n: 0.0 for n in _impl.NUMBERS}
    effective_draws = 0.0
    total = len(draws)
    for index, draw in enumerate(draws):
        age = total - index - 1
        weight = decay ** age
        effective_draws += weight
        for number in getattr(draw, "main", ()):
            if number in weighted_hits:
                weighted_hits[number] += weight

    prior_hits = _POSTERIOR_PRIOR_STRENGTH * _BASE_MAIN_PROBABILITY
    denominator = effective_draws + _POSTERIOR_PRIOR_STRENGTH
    if denominator <= 0.0:
        return {n: 0.0 for n in _impl.NUMBERS}

    posterior = {
        n: (weighted_hits[n] + prior_hits) / denominator
        for n in _impl.NUMBERS
    }
    return _standardize_cross_section(posterior)


def _window_consensus_signal(
    window_signals: Sequence[Mapping[int, float]],
    weights: Sequence[float],
) -> Dict[int, float]:
    """Blend standardized windows and penalize unstable cross-window spikes."""
    if not window_signals:
        return {n: 0.0 for n in _impl.NUMBERS}
    clipped = [max(0.0, float(value)) for value in weights]
    total_weight = sum(clipped)
    if total_weight <= 0.0:
        normalized = [1.0 / len(window_signals)] * len(window_signals)
    else:
        normalized = [value / total_weight for value in clipped]

    combined: Dict[int, float] = {}
    for number in _impl.NUMBERS:
        values = [float(signal.get(number, 0.0)) for signal in window_signals]
        mean = sum(weight * value for weight, value in zip(normalized, values))
        variance = sum(
            weight * (value - mean) ** 2
            for weight, value in zip(normalized, values)
        )
        disagreement = math.sqrt(max(0.0, variance))
        combined[number] = mean - _WINDOW_DISAGREEMENT_PENALTY * disagreement
    return _standardize_cross_section(combined)


def _generation5_blend_number_scores(train: Sequence[object], genome) -> Dict[int, float]:
    """Calibrate Generation 5 number ranking while preserving legacy score scale."""
    legacy = _LEGACY_BLEND_NUMBER_SCORES(train, genome)
    windows = (
        (train, 0.986),
        (train[-240:] if len(train) >= 2 else train, 0.982),
        (train[-120:] if len(train) >= 2 else train, 0.976),
        (train[-60:] if len(train) >= 2 else train, 0.965),
    )
    signals = [_posterior_window_signal(draws, decay) for draws, decay in windows]
    consensus = _window_consensus_signal(
        signals,
        (
            float(genome.full_weight),
            float(genome.recent240_weight),
            float(genome.recent120_weight),
            float(genome.recent60_weight),
        ),
    )

    legacy_values = [float(legacy[n]) for n in _impl.NUMBERS]
    legacy_mean = sum(legacy_values) / len(legacy_values)
    legacy_variance = sum(
        (value - legacy_mean) ** 2 for value in legacy_values
    ) / len(legacy_values)
    legacy_std = math.sqrt(max(0.0, legacy_variance))
    if legacy_std <= 1e-12:
        legacy_std = 1.0

    return {
        n: legacy_mean + legacy_std * float(consensus[n])
        for n in _impl.NUMBERS
    }


def blend_number_scores(train: Sequence[object], genome) -> Dict[int, float]:
    """Use calibrated scoring only for Generation 5 genomes.

    This preserves the exact legacy scoring path for the currently approved
    model and all pre-Generation-5 genomes.
    """
    genome_id = str(getattr(genome, "id", ""))
    if genome_id.startswith(_GENERATION5_PREFIX):
        return _generation5_blend_number_scores(train, genome)
    return _LEGACY_BLEND_NUMBER_SCORES(train, genome)


# _impl.generate_tickets resolves blend_number_scores at runtime. Replacing only
# this helper keeps the rest of ticket generation identical and makes the new
# path automatically available to promoted g5_* models while preserving legacy
# behavior for every other model id.
_impl.blend_number_scores = blend_number_scores


def _high_match_evaluate_genome(
    genome,
    draws: Sequence[object],
    purchase_count: int,
    min_train_draws: int,
    max_targets: Optional[int],
    target_stride: int,
):
    target_indices = list(range(min_train_draws, len(draws), max(1, target_stride)))
    if max_targets is not None:
        target_indices = target_indices[-max_targets:]

    rank_counts = {rank: 0 for rank in _impl.RANK_ORDER}
    draw_max_matches: List[int] = []
    ticket_main_matches: List[int] = []
    portfolios: List[Sequence[Sequence[int]]] = []

    for index in target_indices:
        target = draws[index]
        tickets = _impl.generate_tickets(draws[:index], genome, purchase_count)
        portfolios.append(tickets)
        draw_max = 0
        for ticket in tickets:
            main_match, _bonus_match, rank = _impl.evaluate_ticket(ticket, target)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            ticket_main_matches.append(main_match)
            draw_max = max(draw_max, main_match)
        draw_max_matches.append(draw_max)

    hit_metrics = summarize_hit_metrics(
        draw_max_matches,
        ticket_main_matches=ticket_main_matches,
        portfolios=portfolios,
    )
    segment_scores: List[float] = []
    segment_metrics: List[Dict[str, object]] = []
    for segment_index, (start, end) in enumerate(_segment_bounds(len(draw_max_matches)), start=1):
        ticket_start = start * purchase_count
        ticket_end = end * purchase_count
        summary = summarize_hit_metrics(
            draw_max_matches[start:end],
            ticket_main_matches=ticket_main_matches[ticket_start:ticket_end],
            portfolios=portfolios[start:end],
        )
        summary["segment"] = segment_index
        summary["match_quality_score"] = match_quality_score(summary)
        segment_metrics.append(summary)
        segment_scores.append(float(summary["match_quality_score"]))

    match_score = match_quality_score(hit_metrics)
    metrics: Dict[str, object] = {
        "genome_id": genome.id,
        "generation": genome.generation,
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "targets": len(draw_max_matches),
        "target_draws": len(draw_max_matches),
        "tickets": len(ticket_main_matches),
        "total_tickets": len(ticket_main_matches),
        "max_main_match": max(draw_max_matches, default=0),
        "match_quality_score": match_score,
        "diversity_quality_score": diversity_quality_score(hit_metrics),
        "temporal_segment_metrics": segment_metrics,
        "temporal_segment_match_score_median": (
            __import__("statistics").median(segment_scores) if segment_scores else match_score
        ),
        "temporal_segment_match_score_min": min(segment_scores) if segment_scores else match_score,
        **hit_metrics,
        **{f"rank_{rank}": rank_counts.get(rank, 0) for rank in _impl.RANK_ORDER},
    }
    metrics["hit_first_objective_score"] = hit_first_score(metrics)
    metrics["score"] = metrics["hit_first_objective_score"]

    genome.score = float(metrics["hit_first_objective_score"])
    genome.max_main_match = int(metrics["max_main_match"])
    genome.best_rank_count = int(metrics.get("draw_main5_plus_count", 0) or 0)
    return genome, metrics


_impl.evaluate_genome = _high_match_evaluate_genome
evaluate_genome = _high_match_evaluate_genome


if __name__ == "__main__":
    raise SystemExit(_impl.main())
