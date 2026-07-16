#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core statistical utilities for the fourth-generation LOTO7 pipeline."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Ticket = Tuple[int, ...]
Pair = Tuple[int, int]
Triple = Tuple[int, int, int]


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = clamp(q, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def draw_main(draw: object) -> Ticket:
    return tuple(sorted(int(value) for value in getattr(draw, "main", ())))


def exp_weighted_number_scores(
    draws: Sequence[object], *, half_life: float = 104.0, dormancy_weight: float = 0.12
) -> Dict[int, float]:
    scores = {number: 1e-9 for number in range(1, 38)}
    if not draws:
        return scores
    decay = math.log(2.0) / max(1.0, float(half_life))
    last_seen: Dict[int, Optional[int]] = {number: None for number in range(1, 38)}
    for age, draw in enumerate(reversed(draws)):
        weight = math.exp(-decay * age)
        for number in draw_main(draw):
            scores[number] += weight
            if last_seen[number] is None:
                last_seen[number] = age
    max_gap = max((gap or 0) for gap in last_seen.values()) or 1
    for number, gap in last_seen.items():
        normalized_gap = (gap if gap is not None else max_gap + 1) / float(max_gap + 1)
        scores[number] += dormancy_weight * normalized_gap
    total = sum(scores.values()) or 1.0
    return {number: value / total for number, value in scores.items()}


def rank_nonconformity(scores: Mapping[int, float]) -> Dict[int, float]:
    ordered = sorted(scores, key=lambda n: (scores[n], -n), reverse=True)
    denominator = max(1, len(ordered) - 1)
    return {number: index / denominator for index, number in enumerate(ordered)}


def _pool_from_scores(
    scores: Mapping[int, float], threshold: float, min_pool_size: int, max_pool_size: int
) -> List[int]:
    nonconformity = rank_nonconformity(scores)
    ordered = sorted(scores, key=lambda n: (scores[n], -n), reverse=True)
    pool = [number for number in ordered if nonconformity[number] <= threshold]
    if len(pool) < min_pool_size:
        pool = ordered[:min_pool_size]
    if len(pool) > max_pool_size:
        pool = ordered[:max_pool_size]
    return pool


def conformal_number_pool(
    draws: Sequence[object], *, alpha: float = 0.20, calibration_draws: int = 104,
    min_train_draws: int = 52, half_life: float = 104.0,
    min_pool_size: int = 14, max_pool_size: int = 24,
) -> Dict[str, object]:
    """Build a rolling conformal number set using prior-only calibration."""
    if len(draws) <= min_train_draws:
        pool = list(range(1, min_pool_size + 1))
        return {
            "numbers": pool,
            "alpha": alpha,
            "threshold": 1.0,
            "calibration_score_count": 0,
            "empirical_main_number_coverage": 0.0,
            "pool_size": len(pool),
            "future_data_used": False,
        }
    start = max(min_train_draws, len(draws) - max(1, calibration_draws))
    calibration_scores: List[float] = []
    for index in range(start, len(draws)):
        scores = exp_weighted_number_scores(draws[:index], half_life=half_life)
        nc = rank_nonconformity(scores)
        calibration_scores.extend(nc[number] for number in draw_main(draws[index]))
    if calibration_scores:
        finite_q = math.ceil((len(calibration_scores) + 1) * (1.0 - alpha)) / len(calibration_scores)
        threshold = percentile(calibration_scores, min(1.0, finite_q))
    else:
        threshold = 1.0
    current_scores = exp_weighted_number_scores(draws, half_life=half_life)
    pool = _pool_from_scores(current_scores, threshold, min_pool_size, max_pool_size)
    covered = 0
    total_main = 0
    for index in range(start, len(draws)):
        prior_scores = exp_weighted_number_scores(draws[:index], half_life=half_life)
        prior_pool = set(_pool_from_scores(prior_scores, threshold, min_pool_size, max_pool_size))
        actual = draw_main(draws[index])
        covered += sum(1 for number in actual if number in prior_pool)
        total_main += len(actual)
    ranked = sorted(current_scores, key=lambda n: (current_scores[n], -n), reverse=True)
    return {
        "numbers": sorted(pool),
        "ranked_numbers": ranked,
        "number_scores": {str(n): round(current_scores[n], 9) for n in ranked},
        "alpha": round(alpha, 6),
        "threshold": round(threshold, 6),
        "calibration_score_count": len(calibration_scores),
        "calibration_draw_count": max(0, len(draws) - start),
        "empirical_main_number_coverage": round(covered / total_main, 6) if total_main else 0.0,
        "pool_size": len(pool),
        "future_data_used": False,
        "half_life": half_life,
    }


def _distribution(draws: Sequence[object]) -> List[float]:
    counts = [1e-9] * 37
    for draw in draws:
        for number in draw_main(draw):
            counts[number - 1] += 1.0
    total = sum(counts) or 1.0
    return [value / total for value in counts]


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(p * math.log(p / q) for p, q in zip(left, right) if p > 0.0 and q > 0.0)


def _aggregate_features(draws: Sequence[object]) -> Tuple[float, float, float, float]:
    if not draws:
        return (0.0, 0.0, 0.0, 0.0)
    sums: List[float] = []
    odds: List[float] = []
    consecutive: List[float] = []
    repeats: List[float] = []
    previous: Optional[set[int]] = None
    for draw in draws:
        numbers = draw_main(draw)
        sums.append(sum(numbers) / 259.0)
        odds.append(sum(1 for n in numbers if n % 2) / 7.0)
        consecutive.append(sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1) / 6.0)
        current = set(numbers)
        repeats.append(len(current & previous) / 7.0 if previous is not None else 0.0)
        previous = current
    return tuple(statistics.fmean(values) for values in (sums, odds, consecutive, repeats))


def detect_change_point(
    draws: Sequence[object], *, recent_window: int = 52, reference_window: int = 104
) -> Dict[str, object]:
    needed = recent_window + reference_window
    if len(draws) < needed:
        return {"score": 0.0, "level": "insufficient_data", "recent_window": recent_window,
                "reference_window": reference_window, "js_divergence": 0.0,
                "aggregate_distance": 0.0}
    recent = draws[-recent_window:]
    reference = draws[-needed:-recent_window]
    p = _distribution(recent)
    q = _distribution(reference)
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]
    js = 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)
    recent_features = _aggregate_features(recent)
    reference_features = _aggregate_features(reference)
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(recent_features, reference_features)))
    score = clamp(1.0 - math.exp(-(js * 18.0 + distance * 5.0)), 0.0, 1.0)
    level = "high" if score >= 0.80 else "moderate" if score >= 0.55 else "low"
    return {
        "score": round(score, 6), "level": level, "recent_window": recent_window,
        "reference_window": reference_window, "js_divergence": round(js, 9),
        "aggregate_distance": round(distance, 9),
        "recent_features": [round(v, 9) for v in recent_features],
        "reference_features": [round(v, 9) for v in reference_features],
    }


def strategy_posteriors(
    history_path: str, *, strategies: Sequence[str], prior_strength: float = 2.0
) -> Dict[str, object]:
    utility_sum = {strategy: 0.0 for strategy in strategies}
    evaluated = {strategy: 0 for strategy in strategies}
    path = Path(history_path)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                strategy = str(row.get("strategy") or "").strip()
                if strategy not in utility_sum or str(row.get("status") or "") != "evaluated":
                    continue
                try:
                    utility = clamp(float(row.get("utility") or 0.0), 0.0, 1.0)
                except (TypeError, ValueError):
                    continue
                utility_sum[strategy] += utility
                evaluated[strategy] += 1
    alphas = {strategy: prior_strength + utility_sum[strategy] for strategy in strategies}
    total = sum(alphas.values()) or 1.0
    weights = {strategy: alphas[strategy] / total for strategy in strategies}
    return {
        "weights": {s: round(weights[s], 9) for s in strategies},
        "alpha": {s: round(alphas[s], 9) for s in strategies},
        "utility_sum": {s: round(utility_sum[s], 9) for s in strategies},
        "evaluated_draws": evaluated, "prior_strength": prior_strength,
    }


def dynamic_source_weights(
    posterior: Mapping[str, float], change_point: Mapping[str, object], *, super_independent: bool
) -> Dict[str, float]:
    defaults = {"full": 0.34, "recent": 0.28, "super": 0.10, "regime": 0.28}
    weights = {source: max(0.001, float(posterior.get(source, defaults[source]))) for source in defaults}
    if not super_independent:
        weights["recent"] += weights["super"]
        weights["super"] = 0.0
    change_score = float(change_point.get("score") or 0.0)
    if change_score >= 0.80:
        shift = min(0.12, weights["full"] * 0.35)
        weights["full"] -= shift
        weights["recent"] += shift * (0.65 if super_independent else 1.0)
        if super_independent:
            weights["super"] += shift * 0.35
    elif change_score >= 0.55:
        shift = min(0.06, weights["full"] * 0.20)
        weights["full"] -= shift
        weights["recent"] += shift
    total = sum(weights.values()) or 1.0
    return {source: value / total for source, value in weights.items()}


def allocate_quotas(
    weights: Mapping[str, float], *, purchase_count: int, super_independent: bool
) -> Dict[str, int]:
    minimums = {"full": 1, "recent": 1, "super": 0, "regime": 1}
    if super_independent and purchase_count >= 5 and float(weights.get("super", 0.0)) >= 0.08:
        minimums["super"] = 1
    while sum(minimums.values()) > purchase_count:
        choices = [s for s, count in minimums.items() if count > 0 and s != "full"]
        minimums[min(choices, key=lambda s: float(weights.get(s, 0.0)))] -= 1
    quotas = dict(minimums)
    remaining = purchase_count - sum(quotas.values())
    raw = {source: float(weights.get(source, 0.0)) * remaining for source in quotas}
    floors = {source: int(math.floor(value)) for source, value in raw.items()}
    for source, value in floors.items():
        quotas[source] += value
    leftover = purchase_count - sum(quotas.values())
    order = sorted(quotas, key=lambda s: (raw[s] - floors[s], weights.get(s, 0.0)), reverse=True)
    for source in order[:leftover]:
        quotas[source] += 1
    if not super_independent:
        quotas["recent"] += quotas.get("super", 0)
        quotas["super"] = 0
    return quotas


def hypergraph_weights(draws: Sequence[object], *, lookback: int = 260) -> Dict[str, Dict[Tuple[int, ...], float]]:
    selected = draws[-lookback:] if lookback > 0 else draws
    pair_counts: Dict[Pair, int] = defaultdict(int)
    triple_counts: Dict[Triple, int] = defaultdict(int)
    for draw in selected:
        numbers = draw_main(draw)
        for pair in combinations(numbers, 2):
            pair_counts[pair] += 1
        for triple in combinations(numbers, 3):
            triple_counts[triple] += 1
    max_pair = max(pair_counts.values(), default=1)
    max_triple = max(triple_counts.values(), default=1)
    return {
        "pairs": {p: math.log1p(c + 1.0) / math.log1p(max_pair + 1.0) for p, c in pair_counts.items()},
        "triples": {t: math.log1p(c + 0.5) / math.log1p(max_triple + 0.5) for t, c in triple_counts.items()},
    }


def candidate_feature(ticket: Ticket, *, source: str) -> List[float]:
    source_order = ("full", "recent", "super", "regime")
    features = [0.0] * (37 + 4 + len(source_order) + 3)
    for number in ticket:
        features[number - 1] = 1.0 / math.sqrt(7.0)
        band = 0 if number <= 9 else 1 if number <= 19 else 2 if number <= 29 else 3
        features[37 + band] += 0.10
    if source in source_order:
        features[41 + source_order.index(source)] = 0.20
    offset = 41 + len(source_order)
    features[offset] = sum(ticket) / (259.0 * 5.0)
    features[offset + 1] = sum(1 for number in ticket if number % 2) / (7.0 * 5.0)
    features[offset + 2] = (ticket[-1] - ticket[0]) / (36.0 * 5.0)
    norm = math.sqrt(sum(value * value for value in features)) or 1.0
    return [value / norm for value in features]


def determinant(matrix: Sequence[Sequence[float]]) -> float:
    if not matrix:
        return 1.0
    work = [list(map(float, row)) for row in matrix]
    result = 1.0
    for column in range(len(work)):
        pivot = max(range(column, len(work)), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            return 0.0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result *= -1.0
        pivot_value = work[column][column]
        result *= pivot_value
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot_value
            for index in range(column + 1, len(work)):
                work[row][index] -= factor * work[column][index]
    return result


def dpp_logdet(tickets: Sequence[Ticket], sources: Sequence[str], qualities: Sequence[float]) -> float:
    if not tickets:
        return 0.0
    features = [candidate_feature(ticket, source=source) for ticket, source in zip(tickets, sources)]
    positives = [max(0.05, float(quality)) for quality in qualities]
    scale = max(positives) or 1.0
    quality = [0.40 + 1.60 * value / scale for value in positives]
    matrix: List[List[float]] = []
    for i in range(len(tickets)):
        row: List[float] = []
        for j in range(len(tickets)):
            similarity = clamp(sum(a * b for a, b in zip(features[i], features[j])), 0.0, 1.0)
            value = quality[i] * quality[j] * similarity
            if i == j:
                value += 1e-6
            row.append(value)
        matrix.append(row)
    return math.log(max(determinant(matrix), 1e-15))


def hypergraph_coverage_score(
    tickets: Sequence[Ticket], weights: Mapping[str, Mapping[Tuple[int, ...], float]]
) -> Dict[str, float]:
    pairs: set[Pair] = set()
    triples: set[Triple] = set()
    for ticket in tickets:
        pairs.update(combinations(ticket, 2))
        triples.update(combinations(ticket, 3))
    pair_map = weights.get("pairs", {})
    triple_map = weights.get("triples", {})
    pair_score = sum(float(pair_map.get(pair, 0.05)) for pair in pairs)
    triple_score = sum(float(triple_map.get(triple, 0.02)) for triple in triples)
    return {"pair_score": pair_score, "triple_score": triple_score,
            "pair_count": float(len(pairs)), "triple_count": float(len(triples)),
            "total": pair_score + triple_score * 0.35}


def bounded_strategy_utility(max_main_match: int, total_main_matches: int, winning_tickets: int = 0) -> float:
    value = 0.62 * clamp(max_main_match / 7.0, 0.0, 1.0)
    value += 0.28 * clamp(total_main_matches / 35.0, 0.0, 1.0)
    value += 0.10 * clamp(winning_tickets / 5.0, 0.0, 1.0)
    return clamp(value, 0.0, 1.0)


def eprocess_from_history(
    history_path: str, *, challenger: str = "generation4", champion: str = "beam_baseline",
    betting_fraction: float = 0.25, promotion_threshold: float = 20.0,
    min_evaluated_draws: int = 30,
) -> Dict[str, object]:
    by_draw: Dict[int, Dict[str, float]] = defaultdict(dict)
    path = Path(history_path)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("status") or "") != "evaluated":
                    continue
                strategy = str(row.get("strategy") or "")
                if strategy not in {challenger, champion}:
                    continue
                try:
                    draw_no = int(str(row.get("prediction_draw_no") or "").replace("第", "").replace("回", ""))
                    utility = clamp(float(row.get("utility") or 0.0), 0.0, 1.0)
                except (TypeError, ValueError):
                    continue
                by_draw[draw_no][strategy] = utility
    e_value = 1.0
    reverse = 1.0
    differences: List[float] = []
    for draw_no in sorted(by_draw):
        values = by_draw[draw_no]
        if challenger not in values or champion not in values:
            continue
        difference = clamp(values[challenger] - values[champion], -1.0, 1.0)
        differences.append(difference)
        e_value *= max(1e-12, 1.0 + betting_fraction * difference)
        reverse *= max(1e-12, 1.0 - betting_fraction * difference)
    count = len(differences)
    if count >= min_evaluated_draws and e_value >= promotion_threshold:
        decision = "promote_challenger"
    elif count >= min_evaluated_draws and reverse >= promotion_threshold:
        decision = "retain_champion"
    else:
        decision = "continue"
    return {
        "challenger": challenger, "champion": champion, "evaluated_draws": count,
        "e_value": round(e_value, 9), "reverse_e_value": round(reverse, 9),
        "mean_utility_difference": round(statistics.fmean(differences), 9) if differences else 0.0,
        "betting_fraction": betting_fraction, "promotion_threshold": promotion_threshold,
        "min_evaluated_draws": min_evaluated_draws, "decision": decision,
    }
