"""Payout-independent ranking diagnostics for five-ticket portfolios."""

from __future__ import annotations

from collections.abc import Sequence

NUMBERS = tuple(range(1, 38))


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def summarize_portfolio_ranking(
    portfolios: Sequence[Sequence[Sequence[int]]],
    actual_main_numbers: Sequence[Sequence[int]],
) -> dict[str, object]:
    if len(portfolios) != len(actual_main_numbers):
        raise ValueError("portfolio and result lengths differ")
    top_hits = {7: 0, 14: 0, 18: 0}
    rank_sum = 0.0
    winner_count = 0
    all_scores: list[float] = []
    all_labels: list[int] = []
    for portfolio, actual in zip(portfolios, actual_main_numbers):
        ticket_count = max(1, len(portfolio))
        counts = {number: 0 for number in NUMBERS}
        for ticket in portfolio:
            for number in set(int(value) for value in ticket):
                if number in counts:
                    counts[number] += 1
        scores = {number: counts[number] / ticket_count for number in NUMBERS}
        ordered = sorted(NUMBERS, key=lambda number: (scores[number], -number), reverse=True)
        actual_set = {int(value) for value in actual}
        for cutoff in top_hits:
            top_hits[cutoff] += len(actual_set.intersection(ordered[:cutoff]))
        positions = {number: index + 1 for index, number in enumerate(ordered)}
        rank_sum += sum(positions[number] for number in actual_set)
        winner_count += len(actual_set)
        all_scores.extend(scores[number] for number in NUMBERS)
        all_labels.extend(1 if number in actual_set else 0 for number in NUMBERS)
    brier = sum((score - label) ** 2 for score, label in zip(all_scores, all_labels)) / max(
        1, len(all_scores)
    )
    calibration_error = 0.0
    for bin_index in range(5):
        lower = bin_index / 5
        upper = (bin_index + 1) / 5
        members = []
        for score, label in zip(all_scores, all_labels):
            inside = lower <= score <= upper if bin_index == 4 else lower <= score < upper
            if inside:
                members.append((score, label))
        if members:
            mean_score = sum(item[0] for item in members) / len(members)
            mean_label = sum(item[1] for item in members) / len(members)
            calibration_error += len(members) / len(all_scores) * abs(mean_score - mean_label)
    draw_count = max(1, len(portfolios))
    return {
        "ranking_metric_version": "loto7-portfolio-ranking-2026.07.31-v1",
        "top7_main_recall": round(top_hits[7] / (draw_count * 7), 6),
        "top14_main_recall": round(top_hits[14] / (draw_count * 7), 6),
        "top18_main_recall": round(top_hits[18] / (draw_count * 7), 6),
        "winning_number_mean_rank": round(rank_sum / max(1, winner_count), 6),
        "portfolio_inclusion_auc": round(_auc(all_scores, all_labels), 6),
        "portfolio_inclusion_brier": round(brier, 6),
        "portfolio_calibration_error": round(calibration_error, 6),
    }
