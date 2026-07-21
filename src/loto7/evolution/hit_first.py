#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-match-first evaluation and adoption rules for LOTO7 evolution.

Prize amounts and profit are deliberately excluded from the learning score.
Financial values are calculated only for independent safety gates.
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence, Tuple

from loto7.evaluation.hit_metrics import summarize_hit_metrics

OBJECTIVE_NAME = "hit_first_temporal_robustness"
OBJECTIVE_VERSION = "loto7-hit-first-2026.07.22-v1"


def _float(metrics: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int(metrics: Dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def match_quality_score(metrics: Dict[str, object]) -> float:
    """Return a 0-100 score based only on main-number matches."""
    average_match = max(0.0, min(7.0, _float(metrics, "average_max_main_match")))
    draw4 = max(0.0, min(1.0, _float(metrics, "draw_main4_plus_rate")))
    draw5 = max(0.0, min(1.0, _float(metrics, "draw_main5_plus_rate")))
    draw6 = max(0.0, min(1.0, _float(metrics, "draw_main6_plus_rate")))
    draw7 = max(0.0, min(1.0, _float(metrics, "draw_main7_plus_rate")))
    score = (
        35.0 * (average_match / 7.0)
        + 30.0 * draw4
        + 20.0 * draw5
        + 10.0 * draw6
        + 5.0 * draw7
    )
    return round(score, 6)


def diversity_quality_score(metrics: Dict[str, object]) -> float:
    """Return a 0-100 portfolio diversity score without using payout data."""
    unique_numbers = max(0.0, min(35.0, _float(metrics, "average_portfolio_unique_numbers")))
    mean_overlap = max(0.0, min(7.0, _float(metrics, "mean_ticket_pair_overlap")))
    coverage = min(1.0, unique_numbers / 20.0)
    overlap_quality = max(0.0, min(1.0, (6.0 - mean_overlap) / 4.0))
    return round(100.0 * (0.70 * coverage + 0.30 * overlap_quality), 6)


def hit_first_score(metrics: Dict[str, object]) -> float:
    """Return the learning objective; ROI and profit never add points."""
    match_score = _float(metrics, "match_quality_score", match_quality_score(metrics))
    segment_median = _float(metrics, "temporal_segment_match_score_median", match_score)
    segment_minimum = _float(metrics, "temporal_segment_match_score_min", match_score)
    stability_score = 0.60 * segment_median + 0.40 * segment_minimum
    diversity_score = _float(metrics, "diversity_quality_score", diversity_quality_score(metrics))
    score = 0.70 * match_score + 0.20 * stability_score + 0.10 * diversity_score
    return round(score, 6)


def hit_first_key(metrics: Dict[str, object]) -> Tuple[float, int, float, float, float, int, float]:
    """Lexicographic key used for best selection and parent-pool ranking."""
    return (
        hit_first_score(metrics),
        _int(metrics, "draw_main5_plus_count"),
        _float(metrics, "draw_main4_plus_rate"),
        _float(metrics, "average_max_main_match"),
        _float(metrics, "temporal_segment_match_score_min"),
        _int(metrics, "draw_main6_plus_count"),
        _float(metrics, "payout_roi_percent"),  # final tie-break only
    )


def _segment_bounds(length: int, segment_count: int = 4) -> List[Tuple[int, int]]:
    if length <= 0:
        return []
    count = min(max(1, segment_count), length)
    bounds: List[Tuple[int, int]] = []
    for segment in range(count):
        start = math.floor(length * segment / count)
        end = math.floor(length * (segment + 1) / count)
        if end > start:
            bounds.append((start, end))
    return bounds


def evaluate_model_on_holdout(
    *,
    genome: object,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    """Evaluate one Genome with high-match, temporal and financial diagnostics."""
    from loto7_evolution_trainer import evaluate_ticket, generate_tickets
    from merge_evolution_shards import RANK_ORDER, has_any_prize_amount, prize_amount_for_rank

    rank_counts = {rank: 0 for rank in RANK_ORDER}
    missing_prize_draws: List[int] = []
    draw_records: List[Dict[str, object]] = []
    ticket_main_matches: List[int] = []
    portfolios: List[Sequence[Sequence[int]]] = []

    for index in target_indices:
        target = draws[index]
        tickets = generate_tickets(draws[:index], genome, purchase_count)
        portfolios.append(tickets)
        prize_row = prize_rows.get(int(target.draw_no), {})
        if not prize_row or not has_any_prize_amount(prize_row):
            missing_prize_draws.append(int(target.draw_no))

        draw_payout = 0
        draw_max_main = 0
        draw_max_bonus = 0
        draw_ticket_matches: List[int] = []
        for ticket in tickets:
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            payout = prize_amount_for_rank(prize_row, rank)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            draw_payout += payout
            draw_max_main = max(draw_max_main, main_match)
            draw_max_bonus = max(draw_max_bonus, bonus_match)
            draw_ticket_matches.append(main_match)
            ticket_main_matches.append(main_match)

        draw_records.append(
            {
                "draw_no": int(target.draw_no),
                "max_main_match": draw_max_main,
                "max_bonus_match": draw_max_bonus,
                "payout": draw_payout,
                "cost": len(tickets) * unit_cost,
                "ticket_main_matches": draw_ticket_matches,
                "portfolio": tickets,
            }
        )

    total_cost = sum(int(record["cost"]) for record in draw_records)
    total_payout = sum(int(record["payout"]) for record in draw_records)
    profit = total_payout - total_cost
    payout_roi_percent = (total_payout / total_cost * 100.0) if total_cost else 0.0
    payouts = sorted((int(record["payout"]) for record in draw_records), reverse=True)
    top1 = payouts[0] if payouts else 0
    top1_share = (top1 / total_payout) if total_payout else 0.0

    hit_metrics = summarize_hit_metrics(
        [int(record["max_main_match"]) for record in draw_records],
        ticket_main_matches=ticket_main_matches,
        portfolios=portfolios,
    )

    segment_metrics: List[Dict[str, object]] = []
    for segment_index, (start, end) in enumerate(_segment_bounds(len(draw_records)), start=1):
        records = draw_records[start:end]
        segment_ticket_matches = [
            int(value)
            for record in records
            for value in record.get("ticket_main_matches", [])
        ]
        segment_portfolios = [record.get("portfolio", []) for record in records]
        summary = summarize_hit_metrics(
            [int(record["max_main_match"]) for record in records],
            ticket_main_matches=segment_ticket_matches,
            portfolios=segment_portfolios,
        )
        summary["segment"] = segment_index
        summary["start_draw_no"] = int(records[0]["draw_no"])
        summary["end_draw_no"] = int(records[-1]["draw_no"])
        summary["match_quality_score"] = match_quality_score(summary)
        segment_metrics.append(summary)

    segment_scores = [float(item["match_quality_score"]) for item in segment_metrics]
    match_score = match_quality_score(hit_metrics)
    diversity_score = diversity_quality_score(hit_metrics)
    temporal_median = statistics.median(segment_scores) if segment_scores else match_score
    temporal_min = min(segment_scores) if segment_scores else match_score

    metrics: Dict[str, object] = {
        "path": model_path,
        "genome_id": str(getattr(genome, "id", "")),
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "target_draws": len(draw_records),
        "purchase_count": purchase_count,
        "unit_cost": unit_cost,
        "total_tickets": sum(len(record.get("ticket_main_matches", [])) for record in draw_records),
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": profit,
        "roi": round((total_payout / total_cost) if total_cost else 0.0, 6),
        "roi_percent": round(payout_roi_percent, 3),
        "payout_roi_percent": round(payout_roi_percent, 3),
        "largest_draw_payout": top1,
        "top1_payout_share": round(top1_share, 6),
        "max_main_match": max((int(record["max_main_match"]) for record in draw_records), default=0),
        "max_bonus_match": max((int(record["max_bonus_match"]) for record in draw_records), default=0),
        "rank_counts": rank_counts,
        "grade_hit_count": sum(rank_counts.get(rank, 0) for rank in RANK_ORDER if rank != "外れ"),
        "high_grade_hit_count": sum(rank_counts.get(rank, 0) for rank in ["1等", "2等", "3等", "4等"]),
        "missing_prize_draw_count": len(set(missing_prize_draws)),
        "missing_prize_draws": sorted(set(missing_prize_draws)),
        "match_quality_score": round(match_score, 6),
        "diversity_quality_score": round(diversity_score, 6),
        "temporal_segment_metrics": segment_metrics,
        "temporal_segment_match_score_median": round(float(temporal_median), 6),
        "temporal_segment_match_score_min": round(float(temporal_min), 6),
        **hit_metrics,
    }
    metrics["hit_first_objective_score"] = hit_first_score(metrics)
    return metrics


def adoption_decision(
    candidate: Dict[str, object],
    baseline: Dict[str, object],
    *,
    min_objective_delta: float = 0.05,
    min_draw4_rate_delta_percent: float = 0.0,
    min_draw5_count_delta: int = 0,
    min_average_max_delta: float = 0.0,
    min_temporal_min_delta: float = 0.0,
    min_payout_roi_percent: float = 8.0,
    max_roi_drop_percent: float = 5.0,
    max_top1_payout_share: float = 0.50,
) -> Tuple[bool, List[str]]:
    """Require high-match improvement, then enforce financial safety floors."""
    reasons: List[str] = []
    failures: List[str] = []

    objective_delta = hit_first_score(candidate) - hit_first_score(baseline)
    draw4_delta = (
        _float(candidate, "draw_main4_plus_rate_percent")
        - _float(baseline, "draw_main4_plus_rate_percent")
    )
    draw5_delta = _int(candidate, "draw_main5_plus_count") - _int(baseline, "draw_main5_plus_count")
    average_delta = _float(candidate, "average_max_main_match") - _float(baseline, "average_max_main_match")
    temporal_delta = (
        _float(candidate, "temporal_segment_match_score_min")
        - _float(baseline, "temporal_segment_match_score_min")
    )

    checks = [
        (objective_delta >= min_objective_delta, f"objective delta={objective_delta:.6f}", f"objective delta {objective_delta:.6f} < {min_objective_delta:.6f}"),
        (draw4_delta >= min_draw4_rate_delta_percent, f"draw4+ delta={draw4_delta:.3f}pt", f"draw4+ delta {draw4_delta:.3f}pt < {min_draw4_rate_delta_percent:.3f}pt"),
        (draw5_delta >= min_draw5_count_delta, f"draw5+ count delta={draw5_delta}", f"draw5+ count delta {draw5_delta} < {min_draw5_count_delta}"),
        (average_delta >= min_average_max_delta, f"average max delta={average_delta:.6f}", f"average max delta {average_delta:.6f} < {min_average_max_delta:.6f}"),
        (temporal_delta >= min_temporal_min_delta, f"worst-segment delta={temporal_delta:.6f}", f"worst-segment delta {temporal_delta:.6f} < {min_temporal_min_delta:.6f}"),
    ]
    for passed, success, failure in checks:
        (reasons if passed else failures).append(success if passed else failure)

    candidate_roi = _float(candidate, "payout_roi_percent", _float(candidate, "roi_percent"))
    baseline_roi = _float(baseline, "payout_roi_percent", _float(baseline, "roi_percent"))
    roi_floor = max(min_payout_roi_percent, baseline_roi - max_roi_drop_percent)
    if candidate_roi >= roi_floor:
        reasons.append(f"financial floor passed: {candidate_roi:.3f}% >= {roi_floor:.3f}%")
    else:
        failures.append(f"financial floor failed: {candidate_roi:.3f}% < {roi_floor:.3f}%")

    top1_share = _float(candidate, "top1_payout_share")
    if top1_share <= max_top1_payout_share:
        reasons.append(f"top1 payout share={top1_share:.6f} <= {max_top1_payout_share:.6f}")
    else:
        failures.append(f"top1 payout share {top1_share:.6f} > {max_top1_payout_share:.6f}")

    return not failures, reasons + [f"FAIL: {message}" for message in failures]
