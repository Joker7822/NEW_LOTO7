from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

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


def _effective_support_size(values: Sequence[float]) -> float:
    positive = [max(0.0, float(value)) for value in values if float(value) > 0.0]
    if not positive:
        return 0.0
    total = sum(positive)
    square_sum = sum(value * value for value in positive)
    if square_sum <= 1e-12:
        return 0.0
    return (total * total) / square_sum


def infer_anchor_profile(
    scored: Sequence[ScoredTicket],
    *,
    candidate_limit: int = 320,
    elite_limit: int = 96,
) -> Dict[str, float | int | str]:
    """Diagnostic-only uncertainty profile inferred from training candidates."""
    if not scored:
        return {
            "name": "diffuse",
            "anchor_overlap_target": 2,
            "number_effective_size": 0.0,
            "pair_effective_size": 0.0,
            "confidence": 0.0,
        }

    shortlist = list(scored[: max(1, candidate_limit)])
    number_support, pair_support, _number_denom, _pair_denom = _elite_support(
        shortlist,
        elite_limit=min(elite_limit, len(shortlist)),
    )
    number_effective = _effective_support_size(list(number_support.values()))
    pair_effective = _effective_support_size(list(pair_support.values()))
    number_confidence = max(0.0, min(1.0, (19.0 - number_effective) / 7.0))
    pair_confidence = max(0.0, min(1.0, (90.0 - pair_effective) / 55.0))
    confidence = 0.72 * number_confidence + 0.28 * pair_confidence

    if confidence >= 0.66:
        name = "concentrated"
        overlap_target = 4
    elif confidence >= 0.36:
        name = "balanced"
        overlap_target = 3
    else:
        name = "diffuse"
        overlap_target = 2

    return {
        "name": name,
        "anchor_overlap_target": overlap_target,
        "number_effective_size": number_effective,
        "pair_effective_size": pair_effective,
        "confidence": confidence,
    }


def _shared_core_support(
    first: Sequence[int],
    candidate: Sequence[int],
    number_support: Counter[int],
    pair_support: Counter[Tuple[int, int]],
    number_denominator: float,
    pair_denominator: float,
) -> float:
    """Score whether the shared anchor core is supported by elite candidates."""
    shared = sorted(set(first) & set(candidate))
    if not shared:
        return 0.0
    number_component = sum(
        number_support[number] / number_denominator for number in shared
    ) / len(shared)
    shared_pairs = [tuple(sorted(pair)) for pair in itertools.combinations(shared, 2)]
    if shared_pairs:
        pair_component = sum(
            pair_support[pair] / pair_denominator for pair in shared_pairs
        ) / len(shared_pairs)
    else:
        pair_component = 0.0
    return 0.65 * number_component + 0.35 * pair_component


def select_tiered_generation5_portfolio(
    scored: Sequence[ScoredTicket],
    purchase_count: int,
    overlap_limit: int,
    *,
    anchor_count: int = 1,
    anchor_overlap_target: int = 4,
    anchor_core_support_weight: float = 0.0,
    target_coverage: int = 18,
    candidate_limit: int = 320,
    elite_limit: int = 96,
) -> List[Tuple[int, ...]]:
    """Build one high-match anchor plus evidence-backed coverage tickets.

    The one-anchor default was selected after a direct 1/2/3-anchor ablation:
    it improved recent average maximum matches and 4+ draw count without losing
    the observed 5+ hit, diversity, or overlap constraint. Explicit anchor_count
    values remain available for controlled comparison and future validation.

    ``anchor_core_support_weight`` is zero for the validated production path.
    Experimental support-core selection assigns part of the anchor objective to
    whether numbers/pairs shared by multiple anchors repeat through the elite
    candidate set.
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
    overlap_target = min(max(0, int(anchor_overlap_target)), overlap_limit)
    support_weight = max(0.0, min(0.12, float(anchor_core_support_weight)))

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
                core_overlap = 0.0
                core_support = 0.0
                if selected:
                    overlap = len(combo_set & set(selected[0]))
                    distance = abs(float(overlap_target) - float(overlap))
                    core_overlap = max(0.0, 1.0 - distance / max(1.0, float(overlap_target)))
                    core_support = _shared_core_support(
                        selected[0],
                        combo,
                        number_support,
                        pair_support,
                        number_denominator,
                        pair_denominator,
                    )
                marginal = (
                    (0.72 - support_weight) * quality
                    + 0.14 * elite_number_support
                    + 0.08 * elite_pair_support
                    + 0.05 * core_overlap
                    + support_weight * core_support
                    + 0.02 * coverage_gain
                    - 0.01 * concentration_penalty
                )
            else:
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


def select_support_core_generation5_portfolio(
    scored: Sequence[ScoredTicket],
    purchase_count: int,
    overlap_limit: int,
    *,
    target_coverage: int = 18,
    candidate_limit: int = 320,
    elite_limit: int = 96,
    core_support_weight: float = 0.06,
) -> List[Tuple[int, ...]]:
    """Experimental two-anchor selector that prefers evidence-backed shared cores."""
    return select_tiered_generation5_portfolio(
        scored,
        purchase_count,
        overlap_limit,
        anchor_count=2,
        anchor_overlap_target=min(4, overlap_limit),
        anchor_core_support_weight=core_support_weight,
        target_coverage=target_coverage,
        candidate_limit=candidate_limit,
        elite_limit=elite_limit,
    )


def select_adaptive_tiered_generation5_portfolio(
    scored: Sequence[ScoredTicket],
    purchase_count: int,
    overlap_limit: int,
    *,
    anchor_count: int = 2,
    target_coverage: int = 18,
    candidate_limit: int = 320,
    elite_limit: int = 96,
) -> Tuple[List[Tuple[int, ...]], Dict[str, float | int | str]]:
    """Diagnostic uncertainty adaptation retained for evidence comparison."""
    profile = infer_anchor_profile(
        scored,
        candidate_limit=candidate_limit,
        elite_limit=elite_limit,
    )
    tickets = select_tiered_generation5_portfolio(
        scored,
        purchase_count,
        overlap_limit,
        anchor_count=anchor_count,
        anchor_overlap_target=int(profile["anchor_overlap_target"]),
        target_coverage=target_coverage,
        candidate_limit=candidate_limit,
        elite_limit=elite_limit,
    )
    return tickets, profile
