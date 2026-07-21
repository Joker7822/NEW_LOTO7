#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for the high-match-first LOTO7 evolution trainer.

Core ticket generation and Genome operations remain in
``_loto7_evolution_trainer_impl.py``. Walk-forward candidate evaluation and
survivor selection use the payout-independent high-match objective.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
