#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation 5 precision-oriented evolution primitives for LOTO7.

Candidate selection uses walk-forward main-number agreement, temporal robustness,
and five-ticket portfolio diversity. Financial values remain safety gates only.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from loto7.evolution.hit_first import hit_first_score

OBJECTIVE_NAME = "generation5_walk_forward_pareto"
OBJECTIVE_VERSION = "loto7-generation5-2026.07.28-v1"
SEED_BANK_VERSION = "loto7-null-seed-bank-2026.07.28-v1"
ISLANDS = ("average_max", "draw4", "high_match", "robust_diversity")


@dataclass(frozen=True)
class WalkForwardFold:
    label: str
    target_indices: Tuple[int, ...]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, nonce: int = 0) -> int:
    payload = "\x1f".join(str(part) for part in (*parts, nonce)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    return value or 1


def build_seed_bank(
    dataset_sha256: str,
    *,
    evaluator_version: str,
    learning_count: int = 700,
    selection_count: int = 150,
    final_count: int = 150,
    namespace: str = "NEW_LOTO7",
) -> Dict[str, object]:
    counts = {
        "learning": max(0, int(learning_count)),
        "selection": max(0, int(selection_count)),
        "final": max(0, int(final_count)),
    }
    used: set[int] = set()
    phases: Dict[str, List[int]] = {}
    for phase, count in counts.items():
        seeds: List[int] = []
        for index in range(count):
            nonce = 0
            seed = stable_seed(
                namespace,
                SEED_BANK_VERSION,
                dataset_sha256,
                evaluator_version,
                phase,
                index,
                nonce=nonce,
            )
            while seed in used:
                nonce += 1
                seed = stable_seed(
                    namespace,
                    SEED_BANK_VERSION,
                    dataset_sha256,
                    evaluator_version,
                    phase,
                    index,
                    nonce=nonce,
                )
            used.add(seed)
            seeds.append(seed)
        phases[phase] = seeds
    return {
        "kind": "loto7_fixed_null_seed_bank",
        "version": SEED_BANK_VERSION,
        "namespace": namespace,
        "dataset_sha256": dataset_sha256,
        "evaluator_version": evaluator_version,
        "counts": counts,
        "phases": phases,
        "total_seeds": sum(counts.values()),
    }


def seeds_for_phase(payload: Mapping[str, object], phase: str, limit: int = 0) -> List[int]:
    phases = payload.get("phases")
    if not isinstance(phases, Mapping):
        raise ValueError("seed bank phases are missing")
    raw = phases.get(phase)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"seed bank phase is missing or empty: {phase}")
    seeds = [int(value) for value in raw]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seed bank phase contains duplicate seeds: {phase}")
    if limit > 0:
        seeds = seeds[:limit]
    return seeds


def build_walk_forward_folds(
    target_indices: Sequence[int],
    *,
    fold_count: int = 5,
) -> List[WalkForwardFold]:
    ordered = tuple(sorted(dict.fromkeys(int(value) for value in target_indices)))
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(ordered) < fold_count:
        raise ValueError("not enough targets for requested fold_count")
    quotient, remainder = divmod(len(ordered), fold_count)
    result: List[WalkForwardFold] = []
    cursor = 0
    for index in range(fold_count):
        size = quotient + (1 if index < remainder else 0)
        part = ordered[cursor : cursor + size]
        cursor += size
        if part:
            result.append(WalkForwardFold(label=f"fold_{index + 1}", target_indices=part))
    if tuple(value for fold in result for value in fold.target_indices) != ordered:
        raise AssertionError("walk-forward fold partition is incomplete")
    return result


def _float(metrics: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = metrics.get(key, default)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _int(metrics: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        value = metrics.get(key, default)
        return int(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _weighted_average(metrics: Sequence[Mapping[str, object]], key: str) -> float:
    total_weight = sum(max(0, _int(item, "target_draws")) for item in metrics)
    if total_weight <= 0:
        return 0.0
    return sum(
        _float(item, key) * max(0, _int(item, "target_draws")) for item in metrics
    ) / total_weight


def aggregate_walk_forward_metrics(
    fold_metrics: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not fold_metrics:
        raise ValueError("fold_metrics must not be empty")
    scores = [hit_first_score(dict(item)) for item in fold_metrics]
    target_draws = sum(_int(item, "target_draws") for item in fold_metrics)
    average_max = _weighted_average(fold_metrics, "average_max_main_match")
    draw4 = _weighted_average(fold_metrics, "draw_main4_plus_rate_percent")
    draw5_rate = _weighted_average(fold_metrics, "draw_main5_plus_rate_percent")
    draw6_rate = _weighted_average(fold_metrics, "draw_main6_plus_rate_percent")
    unique_numbers = _weighted_average(fold_metrics, "average_portfolio_unique_numbers")
    mean_overlap = _weighted_average(fold_metrics, "mean_ticket_pair_overlap")
    max_overlap = max(_int(item, "max_ticket_pair_overlap") for item in fold_metrics)
    fold_median = statistics.median(scores)
    fold_minimum = min(scores)
    high_match_component = draw5_rate * 4.0 + draw6_rate * 8.0
    scalar = (
        0.30 * fold_median
        + 0.25 * fold_minimum
        + 0.20 * draw4 * 2.0
        + 0.15 * average_max * 5.0
        + 0.10 * high_match_component
    )
    return {
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "target_draws": target_draws,
        "fold_count": len(fold_metrics),
        "fold_objective_scores": [round(value, 6) for value in scores],
        "fold_objective_median": round(float(fold_median), 6),
        "fold_objective_min": round(float(fold_minimum), 6),
        "average_max_main_match": round(average_max, 6),
        "draw_main4_plus_rate_percent": round(draw4, 6),
        "draw_main5_plus_rate_percent": round(draw5_rate, 6),
        "draw_main6_plus_rate_percent": round(draw6_rate, 6),
        "draw_main5_plus_count": sum(_int(item, "draw_main5_plus_count") for item in fold_metrics),
        "draw_main6_plus_count": sum(_int(item, "draw_main6_plus_count") for item in fold_metrics),
        "average_portfolio_unique_numbers": round(unique_numbers, 6),
        "mean_ticket_pair_overlap": round(mean_overlap, 6),
        "max_ticket_pair_overlap": max_overlap,
        "generation5_score": round(scalar, 6),
    }


def pareto_vector(record: Mapping[str, object]) -> Tuple[float, ...]:
    metrics = record.get("walk_forward")
    source = metrics if isinstance(metrics, Mapping) else record
    return (
        _float(source, "fold_objective_median"),
        _float(source, "fold_objective_min"),
        _float(source, "draw_main4_plus_rate_percent"),
        _float(source, "average_max_main_match"),
        float(_int(source, "draw_main5_plus_count")),
        float(_int(source, "draw_main6_plus_count")),
        _float(source, "average_portfolio_unique_numbers"),
        -_float(source, "mean_ticket_pair_overlap"),
        -float(_int(source, "max_ticket_pair_overlap")),
    )


def pareto_dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    a = pareto_vector(left)
    b = pareto_vector(right)
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(records: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [
        record
        for index, record in enumerate(records)
        if not any(
            other_index != index and pareto_dominates(other, record)
            for other_index, other in enumerate(records)
        )
    ]


def island_key(record: Mapping[str, object], island: str) -> Tuple[float, ...]:
    metrics = record.get("walk_forward")
    source = metrics if isinstance(metrics, Mapping) else record
    common = (
        _float(source, "generation5_score"),
        _float(source, "fold_objective_min"),
        -_float(source, "mean_ticket_pair_overlap"),
    )
    if island == "average_max":
        return (_float(source, "average_max_main_match"), _float(source, "fold_objective_median"), *common)
    if island == "draw4":
        return (_float(source, "draw_main4_plus_rate_percent"), _float(source, "average_max_main_match"), *common)
    if island == "high_match":
        return (
            float(_int(source, "draw_main6_plus_count")),
            float(_int(source, "draw_main5_plus_count")),
            _float(source, "draw_main5_plus_rate_percent"),
            *common,
        )
    if island == "robust_diversity":
        return (
            _float(source, "fold_objective_min"),
            _float(source, "average_portfolio_unique_numbers"),
            -_float(source, "mean_ticket_pair_overlap"),
            *common,
        )
    raise ValueError(f"unknown island: {island}")


def select_pareto_records(
    records: Sequence[Mapping[str, object]],
    *,
    limit: int,
    island: str,
) -> List[Mapping[str, object]]:
    remaining = list(records)
    selected: List[Mapping[str, object]] = []
    while remaining and len(selected) < max(1, limit):
        front = sorted(pareto_front(remaining), key=lambda item: island_key(item, island), reverse=True)
        take = min(len(front), max(1, limit) - len(selected))
        chosen = front[:take]
        selected.extend(chosen)
        chosen_ids = {id(item) for item in chosen}
        remaining = [item for item in remaining if id(item) not in chosen_ids]
        if take < len(front):
            break
    return selected[: max(1, limit)]


def successive_halving_counts(
    initial_count: int,
    *,
    retention_rate: float = 0.5,
    stages: int = 3,
) -> List[int]:
    if initial_count <= 0:
        raise ValueError("initial_count must be positive")
    if not 0.0 < retention_rate <= 1.0:
        raise ValueError("retention_rate must be in (0, 1]")
    counts = [int(initial_count)]
    for _ in range(max(0, stages - 1)):
        counts.append(max(1, math.ceil(counts[-1] * retention_rate)))
    return counts


def stage_key(metrics: Mapping[str, object], island: str) -> Tuple[float, ...]:
    proxy: Dict[str, object] = dict(metrics)
    proxy.setdefault("fold_objective_median", hit_first_score(dict(metrics)))
    proxy.setdefault("fold_objective_min", hit_first_score(dict(metrics)))
    proxy.setdefault("generation5_score", hit_first_score(dict(metrics)))
    return island_key(proxy, island)


def generation5_adoption_gate(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    min_positive_folds: int = 3,
    min_fold_objective_delta: float = 0.05,
    min_average_max_delta: float = 0.03,
    min_draw4_rate_delta_percent: float = 0.50,
    min_draw5_count_delta: int = 0,
    min_draw6_count_delta: int = 0,
    max_worst_fold_drop_percent: float = 2.0,
    min_average_unique_numbers: float = 13.0,
    max_mean_overlap: float = 4.2,
    max_pair_overlap: int = 4,
    min_payout_roi_percent: float = 8.0,
    max_roi_drop_percent: float = 5.0,
    max_top1_payout_share: float = 0.50,
) -> Dict[str, object]:
    candidate_wf = candidate.get("walk_forward")
    baseline_wf = baseline.get("walk_forward")
    candidate_full = candidate.get("full_metrics")
    baseline_full = baseline.get("full_metrics")
    if not all(isinstance(value, Mapping) for value in (candidate_wf, baseline_wf, candidate_full, baseline_full)):
        raise ValueError("candidate and baseline must include walk_forward and full_metrics")
    candidate_folds = candidate.get("fold_metrics")
    baseline_folds = baseline.get("fold_metrics")
    if not isinstance(candidate_folds, list) or not isinstance(baseline_folds, list):
        raise ValueError("candidate and baseline fold_metrics are required")
    if len(candidate_folds) != len(baseline_folds):
        raise ValueError("candidate and baseline fold count mismatch")

    fold_deltas: List[float] = []
    positive_folds = 0
    for candidate_fold, baseline_fold in zip(candidate_folds, baseline_folds):
        delta = hit_first_score(dict(candidate_fold)) - hit_first_score(dict(baseline_fold))
        fold_deltas.append(delta)
        if delta >= min_fold_objective_delta:
            positive_folds += 1

    deltas = {
        "average_max_main_match": _float(candidate_wf, "average_max_main_match") - _float(baseline_wf, "average_max_main_match"),
        "draw_main4_plus_rate_percent": _float(candidate_wf, "draw_main4_plus_rate_percent") - _float(baseline_wf, "draw_main4_plus_rate_percent"),
        "draw_main5_plus_count": _int(candidate_wf, "draw_main5_plus_count") - _int(baseline_wf, "draw_main5_plus_count"),
        "draw_main6_plus_count": _int(candidate_wf, "draw_main6_plus_count") - _int(baseline_wf, "draw_main6_plus_count"),
        "fold_objective_min": _float(candidate_wf, "fold_objective_min") - _float(baseline_wf, "fold_objective_min"),
        "generation5_score": _float(candidate_wf, "generation5_score") - _float(baseline_wf, "generation5_score"),
    }
    baseline_min = _float(baseline_wf, "fold_objective_min")
    worst_floor = baseline_min * (1.0 - max_worst_fold_drop_percent / 100.0)
    candidate_roi = _float(candidate_full, "payout_roi_percent")
    baseline_roi = _float(baseline_full, "payout_roi_percent")
    roi_floor = max(min_payout_roi_percent, baseline_roi - max_roi_drop_percent)

    checks = [
        (positive_folds >= min_positive_folds, f"positive folds={positive_folds}", f"positive folds {positive_folds} < {min_positive_folds}"),
        (deltas["average_max_main_match"] >= min_average_max_delta, f"average max delta={deltas['average_max_main_match']:.6f}", f"average max delta {deltas['average_max_main_match']:.6f} < {min_average_max_delta:.6f}"),
        (deltas["draw_main4_plus_rate_percent"] >= min_draw4_rate_delta_percent, f"draw4+ delta={deltas['draw_main4_plus_rate_percent']:.3f}pt", f"draw4+ delta {deltas['draw_main4_plus_rate_percent']:.3f}pt < {min_draw4_rate_delta_percent:.3f}pt"),
        (deltas["draw_main5_plus_count"] >= min_draw5_count_delta, f"draw5+ count delta={deltas['draw_main5_plus_count']}", f"draw5+ count delta {deltas['draw_main5_plus_count']} < {min_draw5_count_delta}"),
        (deltas["draw_main6_plus_count"] >= min_draw6_count_delta, f"draw6+ count delta={deltas['draw_main6_plus_count']}", f"draw6+ count delta {deltas['draw_main6_plus_count']} < {min_draw6_count_delta}"),
        (_float(candidate_wf, "fold_objective_min") >= worst_floor, f"worst fold={_float(candidate_wf, 'fold_objective_min'):.6f}", f"worst fold {_float(candidate_wf, 'fold_objective_min'):.6f} < floor {worst_floor:.6f}"),
        (_float(candidate_wf, "average_portfolio_unique_numbers") >= min_average_unique_numbers, f"average unique={_float(candidate_wf, 'average_portfolio_unique_numbers'):.6f}", f"average unique {_float(candidate_wf, 'average_portfolio_unique_numbers'):.6f} < {min_average_unique_numbers:.6f}"),
        (_float(candidate_wf, "mean_ticket_pair_overlap") <= max_mean_overlap, f"mean overlap={_float(candidate_wf, 'mean_ticket_pair_overlap'):.6f}", f"mean overlap {_float(candidate_wf, 'mean_ticket_pair_overlap'):.6f} > {max_mean_overlap:.6f}"),
        (_int(candidate_wf, "max_ticket_pair_overlap") <= max_pair_overlap, f"max overlap={_int(candidate_wf, 'max_ticket_pair_overlap')}", f"max overlap {_int(candidate_wf, 'max_ticket_pair_overlap')} > {max_pair_overlap}"),
        (candidate_roi >= roi_floor, f"financial floor={candidate_roi:.3f}% >= {roi_floor:.3f}%", f"financial floor {candidate_roi:.3f}% < {roi_floor:.3f}%"),
        (_float(candidate_full, "top1_payout_share") <= max_top1_payout_share, f"top1 payout share={_float(candidate_full, 'top1_payout_share'):.6f}", f"top1 payout share {_float(candidate_full, 'top1_payout_share'):.6f} > {max_top1_payout_share:.6f}"),
    ]
    reasons = [success for passed, success, _failure in checks if passed]
    failures = [failure for passed, _success, failure in checks if not passed]
    return {
        "kind": "loto7_generation5_adoption_gate",
        "passed": not failures,
        "positive_folds": positive_folds,
        "fold_deltas": [round(value, 6) for value in fold_deltas],
        "deltas": {key: round(float(value), 6) for key, value in deltas.items()},
        "reasons": reasons,
        "failures": failures,
        "thresholds": {
            "min_positive_folds": min_positive_folds,
            "min_fold_objective_delta": min_fold_objective_delta,
            "min_average_max_delta": min_average_max_delta,
            "min_draw4_rate_delta_percent": min_draw4_rate_delta_percent,
            "min_draw5_count_delta": min_draw5_count_delta,
            "min_draw6_count_delta": min_draw6_count_delta,
            "max_worst_fold_drop_percent": max_worst_fold_drop_percent,
            "min_average_unique_numbers": min_average_unique_numbers,
            "max_mean_overlap": max_mean_overlap,
            "max_pair_overlap": max_pair_overlap,
            "min_payout_roi_percent": min_payout_roi_percent,
            "max_roi_drop_percent": max_roi_drop_percent,
            "max_top1_payout_share": max_top1_payout_share,
        },
    }
