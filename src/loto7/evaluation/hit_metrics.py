#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Payout-independent LOTO7 hit-quality and portfolio metrics.

The primary unit is one draw with a five-ticket portfolio. A draw is counted
once even when several tickets reach the same threshold, preventing duplicated
low-rank tickets from inflating model quality.
"""
from __future__ import annotations

import itertools
import statistics
from typing import Dict, Iterable, Sequence


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _normalized_ticket(ticket: Iterable[int]) -> frozenset[int]:
    return frozenset(int(value) for value in ticket)


def summarize_hit_metrics(
    draw_max_main_matches: Sequence[int],
    *,
    ticket_main_matches: Sequence[int] | None = None,
    portfolios: Sequence[Sequence[Sequence[int]]] | None = None,
) -> Dict[str, object]:
    """Summarize high-match accuracy without using prize amounts.

    ``draw_max_main_matches`` contains the best main-number match achieved by
    the portfolio in each draw. ``ticket_main_matches`` is optional ticket-level
    detail. ``portfolios`` is optional and is used only for diversity metrics.
    """
    draw_values = [max(0, min(7, int(value))) for value in draw_max_main_matches]
    ticket_values = [max(0, min(7, int(value))) for value in (ticket_main_matches or [])]
    draw_count = len(draw_values)
    ticket_count = len(ticket_values)

    result: Dict[str, object] = {
        "hit_metric_version": "loto7-hit-metrics-2026.07.20-v1",
        "draw_main4_plus_count": sum(value >= 4 for value in draw_values),
        "draw_main5_plus_count": sum(value >= 5 for value in draw_values),
        "draw_main6_plus_count": sum(value >= 6 for value in draw_values),
        "draw_main7_count": sum(value >= 7 for value in draw_values),
        "average_max_main_match": round(statistics.fmean(draw_values), 6) if draw_values else 0.0,
        "median_max_main_match": round(float(statistics.median(draw_values)), 6) if draw_values else 0.0,
        "draw_max_main_match_distribution": {
            str(value): sum(item == value for item in draw_values) for value in range(8)
        },
    }
    for threshold in (4, 5, 6, 7):
        count = int(result["draw_main7_count"] if threshold == 7 else result[f"draw_main{threshold}_plus_count"])
        rate = _rate(count, draw_count)
        result[f"draw_main{threshold}_plus_rate"] = round(rate, 6)
        result[f"draw_main{threshold}_plus_rate_percent"] = round(rate * 100.0, 3)

    for threshold in (4, 5, 6, 7):
        count = sum(value >= threshold for value in ticket_values)
        rate = _rate(count, ticket_count)
        result[f"ticket_main{threshold}_plus_count"] = count
        result[f"ticket_main{threshold}_plus_rate"] = round(rate, 6)
        result[f"ticket_main{threshold}_plus_rate_percent"] = round(rate * 100.0, 3)

    unique_counts: list[int] = []
    pair_overlaps: list[int] = []
    max_pair_overlaps: list[int] = []
    for portfolio in portfolios or []:
        tickets = [_normalized_ticket(ticket) for ticket in portfolio]
        if not tickets:
            continue
        unique_counts.append(len(set().union(*tickets)))
        overlaps = [len(left & right) for left, right in itertools.combinations(tickets, 2)]
        pair_overlaps.extend(overlaps)
        max_pair_overlaps.append(max(overlaps) if overlaps else 0)

    result.update(
        {
            "average_portfolio_unique_numbers": round(statistics.fmean(unique_counts), 6) if unique_counts else 0.0,
            "mean_ticket_pair_overlap": round(statistics.fmean(pair_overlaps), 6) if pair_overlaps else 0.0,
            "max_ticket_pair_overlap": max(max_pair_overlaps) if max_pair_overlaps else 0,
        }
    )
    result["hit_objective_score"] = hit_objective_score(result)
    return result


def hit_objective_score(metrics: Dict[str, object]) -> float:
    """Return a 0-100 accuracy-first score.

    The score emphasizes draw-level 4+, 5+ and 6+ reach. Payout and profit are
    intentionally absent; financial metrics remain separate safety gates.
    """
    average_match = max(0.0, min(7.0, float(metrics.get("average_max_main_match", 0.0) or 0.0)))
    draw4 = max(0.0, min(1.0, float(metrics.get("draw_main4_plus_rate", 0.0) or 0.0)))
    draw5 = max(0.0, min(1.0, float(metrics.get("draw_main5_plus_rate", 0.0) or 0.0)))
    draw6 = max(0.0, min(1.0, float(metrics.get("draw_main6_plus_rate", 0.0) or 0.0)))
    draw7 = max(0.0, min(1.0, float(metrics.get("draw_main7_plus_rate", 0.0) or 0.0)))
    unique_numbers = max(0.0, min(37.0, float(metrics.get("average_portfolio_unique_numbers", 0.0) or 0.0)))
    diversity = unique_numbers / 37.0
    score = (
        30.0 * (average_match / 7.0)
        + 25.0 * draw4
        + 20.0 * draw5
        + 15.0 * draw6
        + 5.0 * draw7
        + 5.0 * diversity
    )
    return round(score, 6)
