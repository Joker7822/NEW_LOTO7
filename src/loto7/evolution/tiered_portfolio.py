from __future__ import annotations

import itertools
from collections import Counter
from typing import List, Optional, Sequence, Tuple

ScoredTicket = Tuple[float, Tuple[int, ...]]


def _normalized_quality(raw_score: float, high_score: float, low_score: float) -> float:
    span = max(1e-12, high_score - low_score)
    return (float(raw_score) - low_score) / span


def _elite_support(
    shortlist: Sequence[ScoredTicket],
    *,
    elite_limit: int,
) -> Tuple[Counter[int], Counter[Tuple[int, int]], float, float]:
    elite = list(shortlist[: max(1, elite_limit)])
    high = float(elite[0][0])
    low = float(elite[-1][0])
    number_support: Counter[int] = Counter()
    pair_support: Counter[Tuple[int, int]] = Counter()
    total_weight = 0.0
    for raw_score, combo in elite:
        # Keep every elite ticket represented while emphasizing the top of the list.
        weight = 0.20 + 0.80 * _normalized_quality(raw_score, high, low)
        total_weight += weight
        for number in combo:
            number_support[number] += weight
        for pair in itertools.combinations(combo, 2):
            pair_support[tuple(sorted(pair))] += weight
    return number_support, pair_support, max(total_weight, 1e-12), max(total_weight, 1e-12)


def _support_score(
    combo: Sequence[int],
    number_support: Counter[int],
    pair_support: Counter[Tuple[int, int]],
    number_denominator: float,
    pair_denominator: float,
) -> Tuple[float, float]:
    number_score = sum(number_support[number] for number in combo) / (7.0 * number_denominator)
    pairs = [tuple(sorted(pair)) for pair in itertools.combinations(combo, 2)]
    pair_score = sum(pair_support[pair] for pair in pairs) / (21.0 * pair_denominator)
    return number_score, pair_score


def select_tiered_generation5_portfolio(
    scored: Sequence[ScoredTicket],
    purchase_count: int,
    overlap_limit: int,
    *,
    anchor_count: int = 2,
    target_coverage: int = 18,
    candidate_limit: int = 320,
    elite_limit: int = 96,
) -> List[Tuple[int, ...]]:
    """Build a two-layer portfolio: high-match anchors plus coverage tickets.

    The first two tickets emphasize score and support that repeats across the
    elite candidate set. Remaining tickets emphasize marginal number/pair
    coverage while retaining a quality floor. The overlap limit stays a hard
    constraint until the final compatibility fallback.
    """
    if purchase_count <= 0 or not scored:
        return []

    shortlist = list(scored[: max(purchase_count, candidate_limit)])
    high_score = float(shortlist[0][0])
    low_score = float(shortlist[-1][0])
    number_support, pair_support, number_denominator, pair_denominator = _elite_support(
        shortlist,
        elite_limit=min(elite_limit, len(shortlist)),
    )

    selected: List[Tuple[int, ...]] = []
    used_numbers: set[int] = set()
    used_pairs: set[Tuple[int, int]] = set()
    number_usage: Counter[int] = Counter()
    anchors = min(max(0, anchor_count), purchase_count)

    while len(selected) < purchase_count:
        anchor_phase = len(selected) < anchors
        best: Optional[Tuple[float, float, Tuple[int, ...]]] = None
        for raw_score, combo in shortlist:
            if combo in selected:
                continue
            combo_set = set(combo)
            overlaps = [len(combo_set & set(previous)) for previous in selected]
            max_overlap = max(overlaps, default=0)
            if max_overlap > overlap_limit:
                continue

            quality = _normalized_quality(raw_score, high_score, low_score)
            elite_number_support, elite_pair_support = _support_score(
                combo,
                number_support,
                pair_support,
                number_denominator,
                pair_denominator,
            )

            new_numbers = len(combo_set - used_numbers)
            remaining_coverage = max(0, target_coverage - len(used_numbers))
            coverage_gain = min(new_numbers, remaining_coverage) / 7.0
            combo_pairs = {
                tuple(sorted(pair)) for pair in itertools.combinations(combo, 2)
            }
            pair_novelty = len(combo_pairs - used_pairs) / 21.0
            overlap_ratio = max_overlap / 7.0
            projected_max_usage = max((number_usage[number] + 1 for number in combo), default=1)
            concentration_penalty = max(0.0, projected_max_usage - 3.0) / 2.0

            if anchor_phase:
                # Anchors intentionally exploit candidate consensus. The second
                # anchor receives a mild reward for sharing 3-4 numbers with the
                # first, concentrating two tickets around the strongest core
                # without allowing near duplicates.
                core_overlap = 0.0
                if selected:
                    overlap = len(combo_set & set(selected[0]))
                    core_overlap = max(0.0, 1.0 - abs(4.0 - overlap) / 4.0)
                marginal = (
                    0.72 * quality
                    + 0.14 * elite_number_support
                    + 0.08 * elite_pair_support
                    + 0.05 * core_overlap
                    + 0.02 * coverage_gain
                    - 0.01 * concentration_penalty
                )
            else:
                # Later tickets explore enough of the candidate pool to improve
                # draw-level coverage, while the elite-support terms prevent the
                # diversification layer from becoming random noise.
                marginal = (
                    0.56 * quality
                    + 0.08 * elite_number_support
                    + 0.04 * elite_pair_support
                    + 0.22 * coverage_gain
                    + 0.08 * pair_novelty
                    - 0.025 * overlap_ratio
                    - 0.01 * concentration_penalty
                )

            candidate = (marginal, float(raw_score), combo)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            break
        chosen = best[2]
        selected.append(chosen)
        used_numbers.update(chosen)
        used_pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(chosen, 2))
        number_usage.update(chosen)

    if len(selected) < purchase_count:
        for _score, combo in scored:
            if combo in selected:
                continue
            if all(len(set(combo) & set(previous)) <= overlap_limit for previous in selected):
                selected.append(combo)
            if len(selected) >= purchase_count:
                return selected[:purchase_count]

    if len(selected) < purchase_count:
        for _score, combo in scored:
            if combo not in selected:
                selected.append(combo)
            if len(selected) >= purchase_count:
                break
    return selected[:purchase_count]
