#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation 5 walk-forward evolution primitives.

The v2 objective prioritizes recent, repeatable main-number agreement and portfolio
coverage. Payout metrics are retained only as fail-closed safety gates.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from loto7.evolution.hit_first import hit_first_score

OBJECTIVE_NAME = "generation5_recent_stability_pareto"
OBJECTIVE_VERSION = "loto7-generation5-2026.08.01-v2"
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
        values: List[int] = []
        for index in range(count):
            nonce = 0
            seed = stable_seed(
                namespace, SEED_BANK_VERSION, dataset_sha256,
                evaluator_version, phase, index, nonce=nonce,
            )
            while seed in used:
                nonce += 1
                seed = stable_seed(
                    namespace, SEED_BANK_VERSION, dataset_sha256,
                    evaluator_version, phase, index, nonce=nonce,
                )
            used.add(seed)
            values.append(seed)
        phases[phase] = values
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
    return seeds[:limit] if limit > 0 else seeds


def build_walk_forward_folds(
    target_indices: Sequence[int], *, fold_count: int = 5
) -> List[WalkForwardFold]:
    ordered = tuple(sorted(dict.fromkeys(int(value) for value in target_indices)))
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(ordered) < fold_count:
        raise ValueError("not enough targets for requested fold_count")
    quotient, remainder = divmod(len(ordered), fold_count)
    folds: List[WalkForwardFold] = []
    cursor = 0
    for index in range(fold_count):
        size = quotient + (1 if index < remainder else 0)
        part = ordered[cursor : cursor + size]
        cursor += size
        if part:
            folds.append(WalkForwardFold(f"fold_{index + 1}", part))
    if tuple(value for fold in folds for value in fold.target_indices) != ordered:
        raise AssertionError("walk-forward fold partition is incomplete")
    return folds


def _float(source: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = source.get(key, default)
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _int(source: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        value = source.get(key, default)
        return int(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _weighted(folds: Sequence[Mapping[str, object]], key: str) -> float:
    weights = [max(1, _int(item, "target_draws", 1)) for item in folds]
    return sum(_float(item, key) * weight for item, weight in zip(folds, weights)) / sum(weights)


def _recent_weighted(
    folds: Sequence[Mapping[str, object]], key: str, *, decay: float = 0.78
) -> float:
    last = len(folds) - 1
    weights = [
        max(1, _int(item, "target_draws", 1)) * decay ** (last - index)
        for index, item in enumerate(folds)
    ]
    return sum(_float(item, key) * weight for item, weight in zip(folds, weights)) / sum(weights)


def aggregate_walk_forward_metrics(
    fold_metrics: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not fold_metrics:
        raise ValueError("fold_metrics must not be empty")
    scores = [hit_first_score(dict(item)) for item in fold_metrics]
    score_folds = [dict(item, _score=score) for item, score in zip(fold_metrics, scores)]
    median = statistics.median(scores)
    minimum = min(scores)
    stddev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    recent_score = _recent_weighted(score_folds, "_score")
    latest_score = scores[-1]
    average_max = _weighted(fold_metrics, "average_max_main_match")
    draw4 = _weighted(fold_metrics, "draw_main4_plus_rate_percent")
    draw5_rate = _weighted(fold_metrics, "draw_main5_plus_rate_percent")
    draw6_rate = _weighted(fold_metrics, "draw_main6_plus_rate_percent")
    unique = _weighted(fold_metrics, "average_portfolio_unique_numbers")
    overlap = _weighted(fold_metrics, "mean_ticket_pair_overlap")
    recent_average = _recent_weighted(fold_metrics, "average_max_main_match")
    recent_draw4 = _recent_weighted(fold_metrics, "draw_main4_plus_rate_percent")
    recent_draw5 = _recent_weighted(fold_metrics, "draw_main5_plus_rate_percent")
    recent_draw6 = _recent_weighted(fold_metrics, "draw_main6_plus_rate_percent")
    recent_unique = _recent_weighted(fold_metrics, "average_portfolio_unique_numbers")
    recent_overlap = _recent_weighted(fold_metrics, "mean_ticket_pair_overlap")
    scalar = (
        0.18 * median
        + 0.20 * minimum
        + 0.24 * recent_score
        + 0.14 * recent_draw4 * 2.0
        + 0.12 * recent_average * 5.0
        + 0.08 * (recent_draw5 * 4.0 + recent_draw6 * 8.0)
        + 0.04 * recent_unique
        - 0.10 * stddev
        - 0.05 * recent_overlap
    )
    return {
        "objective_name": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "target_draws": sum(_int(item, "target_draws") for item in fold_metrics),
        "fold_count": len(fold_metrics),
        "fold_objective_scores": [round(value, 6) for value in scores],
        "fold_objective_median": round(float(median), 6),
        "fold_objective_min": round(float(minimum), 6),
        "fold_objective_stddev": round(float(stddev), 6),
        "fold_objective_range": round(float(max(scores) - min(scores)), 6),
        "recent_weighted_objective": round(float(recent_score), 6),
        "recent_fold_objective_mean": round(float(statistics.mean(scores[-min(2, len(scores)):])), 6),
        "latest_fold_objective": round(float(latest_score), 6),
        "average_max_main_match": round(average_max, 6),
        "draw_main4_plus_rate_percent": round(draw4, 6),
        "draw_main5_plus_rate_percent": round(draw5_rate, 6),
        "draw_main6_plus_rate_percent": round(draw6_rate, 6),
        "recent_average_max_main_match": round(recent_average, 6),
        "recent_draw_main4_plus_rate_percent": round(recent_draw4, 6),
        "recent_draw_main5_plus_rate_percent": round(recent_draw5, 6),
        "recent_draw_main6_plus_rate_percent": round(recent_draw6, 6),
        "draw_main5_plus_count": sum(_int(item, "draw_main5_plus_count") for item in fold_metrics),
        "draw_main6_plus_count": sum(_int(item, "draw_main6_plus_count") for item in fold_metrics),
        "average_portfolio_unique_numbers": round(unique, 6),
        "mean_ticket_pair_overlap": round(overlap, 6),
        "max_ticket_pair_overlap": max(_int(item, "max_ticket_pair_overlap") for item in fold_metrics),
        "recent_average_portfolio_unique_numbers": round(recent_unique, 6),
        "recent_mean_ticket_pair_overlap": round(recent_overlap, 6),
        "generation5_score": round(float(scalar), 6),
    }


def _source(record: Mapping[str, object]) -> Mapping[str, object]:
    metrics = record.get("walk_forward")
    return metrics if isinstance(metrics, Mapping) else record


def pareto_vector(record: Mapping[str, object]) -> Tuple[float, ...]:
    source = _source(record)
    return (
        _float(source, "recent_weighted_objective"),
        _float(source, "latest_fold_objective"),
        _float(source, "fold_objective_median"),
        _float(source, "fold_objective_min"),
        -_float(source, "fold_objective_stddev"),
        _float(source, "recent_draw_main4_plus_rate_percent", _float(source, "draw_main4_plus_rate_percent")),
        _float(source, "recent_average_max_main_match", _float(source, "average_max_main_match")),
        float(_int(source, "draw_main5_plus_count")),
        float(_int(source, "draw_main6_plus_count")),
        _float(source, "average_portfolio_unique_numbers"),
        -_float(source, "mean_ticket_pair_overlap"),
        -float(_int(source, "max_ticket_pair_overlap")),
    )


def pareto_dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    a, b = pareto_vector(left), pareto_vector(right)
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(records: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [
        record for index, record in enumerate(records)
        if not any(
            other_index != index and pareto_dominates(other, record)
            for other_index, other in enumerate(records)
        )
    ]


def island_key(record: Mapping[str, object], island: str) -> Tuple[float, ...]:
    source = _source(record)
    common = (
        _float(source, "recent_weighted_objective"),
        _float(source, "latest_fold_objective"),
        _float(source, "generation5_score"),
        _float(source, "fold_objective_min"),
        -_float(source, "fold_objective_stddev"),
        -_float(source, "mean_ticket_pair_overlap"),
    )
    if island == "average_max":
        return (_float(source, "recent_average_max_main_match", _float(source, "average_max_main_match")), *common)
    if island == "draw4":
        return (_float(source, "recent_draw_main4_plus_rate_percent", _float(source, "draw_main4_plus_rate_percent")), *common)
    if island == "high_match":
        return (
            float(_int(source, "draw_main6_plus_count")),
            float(_int(source, "draw_main5_plus_count")),
            _float(source, "recent_draw_main5_plus_rate_percent", _float(source, "draw_main5_plus_rate_percent")),
            *common,
        )
    if island == "robust_diversity":
        return (
            _float(source, "fold_objective_min"),
            -_float(source, "fold_objective_stddev"),
            _float(source, "recent_average_portfolio_unique_numbers", _float(source, "average_portfolio_unique_numbers")),
            -_float(source, "recent_mean_ticket_pair_overlap", _float(source, "mean_ticket_pair_overlap")),
            *common,
        )
    raise ValueError(f"unknown island: {island}")


def select_pareto_records(
    records: Sequence[Mapping[str, object]], *, limit: int, island: str
) -> List[Mapping[str, object]]:
    remaining = list(records)
    selected: List[Mapping[str, object]] = []
    while remaining and len(selected) < max(1, limit):
        front = sorted(pareto_front(remaining), key=lambda item: island_key(item, island), reverse=True)
        chosen = front[: min(len(front), max(1, limit) - len(selected))]
        selected.extend(chosen)
        chosen_ids = {id(item) for item in chosen}
        remaining = [item for item in remaining if id(item) not in chosen_ids]
        if len(chosen) < len(front):
            break
    return selected[: max(1, limit)]


def successive_halving_counts(
    initial_count: int, *, retention_rate: float = 0.5, stages: int = 3
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
    base = hit_first_score(dict(metrics))
    for key in (
        "fold_objective_median", "fold_objective_min", "recent_weighted_objective",
        "latest_fold_objective", "generation5_score",
    ):
        proxy.setdefault(key, base)
    proxy.setdefault("fold_objective_stddev", 0.0)
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
    min_recent_weighted_delta: float = 0.05,
    max_latest_fold_drop: float = 0.10,
    max_fold_stddev_increase: float = 0.35,
    max_worst_fold_drop_percent: float = 2.0,
    min_average_unique_numbers: float = 13.0,
    max_mean_overlap: float = 4.2,
    max_pair_overlap: int = 4,
    min_payout_roi_percent: float = 8.0,
    max_roi_drop_percent: float = 5.0,
    max_top1_payout_share: float = 0.50,
) -> Dict[str, object]:
    candidate_wf, baseline_wf = candidate.get("walk_forward"), baseline.get("walk_forward")
    candidate_full, baseline_full = candidate.get("full_metrics"), baseline.get("full_metrics")
    if not all(isinstance(value, Mapping) for value in (candidate_wf, baseline_wf, candidate_full, baseline_full)):
        raise ValueError("candidate and baseline must include walk_forward and full_metrics")
    candidate_folds, baseline_folds = candidate.get("fold_metrics"), baseline.get("fold_metrics")
    if not isinstance(candidate_folds, list) or not isinstance(baseline_folds, list):
        raise ValueError("candidate and baseline fold_metrics are required")
    if len(candidate_folds) != len(baseline_folds):
        raise ValueError("candidate and baseline fold count mismatch")
    fold_deltas = [
        hit_first_score(dict(left)) - hit_first_score(dict(right))
        for left, right in zip(candidate_folds, baseline_folds)
    ]
    positive_folds = sum(delta >= min_fold_objective_delta for delta in fold_deltas)
    deltas = {
        "average_max_main_match": _float(candidate_wf, "average_max_main_match") - _float(baseline_wf, "average_max_main_match"),
        "draw_main4_plus_rate_percent": _float(candidate_wf, "draw_main4_plus_rate_percent") - _float(baseline_wf, "draw_main4_plus_rate_percent"),
        "draw_main5_plus_count": _int(candidate_wf, "draw_main5_plus_count") - _int(baseline_wf, "draw_main5_plus_count"),
        "draw_main6_plus_count": _int(candidate_wf, "draw_main6_plus_count") - _int(baseline_wf, "draw_main6_plus_count"),
        "fold_objective_min": _float(candidate_wf, "fold_objective_min") - _float(baseline_wf, "fold_objective_min"),
        "recent_weighted_objective": _float(candidate_wf, "recent_weighted_objective") - _float(baseline_wf, "recent_weighted_objective"),
        "latest_fold_objective": _float(candidate_wf, "latest_fold_objective") - _float(baseline_wf, "latest_fold_objective"),
        "fold_objective_stddev": _float(candidate_wf, "fold_objective_stddev") - _float(baseline_wf, "fold_objective_stddev"),
        "generation5_score": _float(candidate_wf, "generation5_score") - _float(baseline_wf, "generation5_score"),
    }
    baseline_min = _float(baseline_wf, "fold_objective_min")
    worst_floor = baseline_min * (1.0 - max_worst_fold_drop_percent / 100.0)
    candidate_roi = _float(candidate_full, "payout_roi_percent")
    roi_floor = max(
        min_payout_roi_percent,
        _float(baseline_full, "payout_roi_percent") - max_roi_drop_percent,
    )
    checks: List[Tuple[bool, str, str]] = []

    def add(passed: bool, success: str, failure: str) -> None:
        checks.append((bool(passed), success, failure))

    add(positive_folds >= min_positive_folds, f"positive folds={positive_folds}", f"positive folds {positive_folds} < {min_positive_folds}")
    add(deltas["average_max_main_match"] >= min_average_max_delta, f"average max delta={deltas['average_max_main_match']:.6f}", f"average max delta {deltas['average_max_main_match']:.6f} < {min_average_max_delta:.6f}")
    add(deltas["draw_main4_plus_rate_percent"] >= min_draw4_rate_delta_percent, f"draw4+ delta={deltas['draw_main4_plus_rate_percent']:.3f}pt", f"draw4+ delta {deltas['draw_main4_plus_rate_percent']:.3f}pt < {min_draw4_rate_delta_percent:.3f}pt")
    add(deltas["draw_main5_plus_count"] >= min_draw5_count_delta, f"draw5+ count delta={deltas['draw_main5_plus_count']}", f"draw5+ count delta {deltas['draw_main5_plus_count']} < {min_draw5_count_delta}")
    add(deltas["draw_main6_plus_count"] >= min_draw6_count_delta, f"draw6+ count delta={deltas['draw_main6_plus_count']}", f"draw6+ count delta {deltas['draw_main6_plus_count']} < {min_draw6_count_delta}")
    add(deltas["recent_weighted_objective"] >= min_recent_weighted_delta, f"recent weighted delta={deltas['recent_weighted_objective']:.6f}", f"recent weighted delta {deltas['recent_weighted_objective']:.6f} < {min_recent_weighted_delta:.6f}")
    add(deltas["latest_fold_objective"] >= -max_latest_fold_drop, f"latest fold delta={deltas['latest_fold_objective']:.6f}", f"latest fold delta {deltas['latest_fold_objective']:.6f} < {-max_latest_fold_drop:.6f}")
    add(deltas["fold_objective_stddev"] <= max_fold_stddev_increase, f"fold stddev delta={deltas['fold_objective_stddev']:.6f}", f"fold stddev delta {deltas['fold_objective_stddev']:.6f} > {max_fold_stddev_increase:.6f}")
    add(_float(candidate_wf, "fold_objective_min") >= worst_floor, f"worst fold={_float(candidate_wf, 'fold_objective_min'):.6f}", f"worst fold {_float(candidate_wf, 'fold_objective_min'):.6f} < floor {worst_floor:.6f}")
    add(_float(candidate_wf, "average_portfolio_unique_numbers") >= min_average_unique_numbers, f"average unique={_float(candidate_wf, 'average_portfolio_unique_numbers'):.6f}", f"average unique {_float(candidate_wf, 'average_portfolio_unique_numbers'):.6f} < {min_average_unique_numbers:.6f}")
    add(_float(candidate_wf, "mean_ticket_pair_overlap") <= max_mean_overlap, f"mean overlap={_float(candidate_wf, 'mean_ticket_pair_overlap'):.6f}", f"mean overlap {_float(candidate_wf, 'mean_ticket_pair_overlap'):.6f} > {max_mean_overlap:.6f}")
    add(_int(candidate_wf, "max_ticket_pair_overlap") <= max_pair_overlap, f"max overlap={_int(candidate_wf, 'max_ticket_pair_overlap')}", f"max overlap {_int(candidate_wf, 'max_ticket_pair_overlap')} > {max_pair_overlap}")
    add(candidate_roi >= roi_floor, f"financial floor={candidate_roi:.3f}% >= {roi_floor:.3f}%", f"financial floor {candidate_roi:.3f}% < {roi_floor:.3f}%")
    top1 = _float(candidate_full, "top1_payout_share")
    add(top1 <= max_top1_payout_share, f"top1 payout share={top1:.6f}", f"top1 payout share {top1:.6f} > {max_top1_payout_share:.6f}")
    return {
        "kind": "loto7_generation5_adoption_gate",
        "objective_version": OBJECTIVE_VERSION,
        "passed": all(passed for passed, _success, _failure in checks),
        "positive_folds": positive_folds,
        "fold_deltas": [round(value, 6) for value in fold_deltas],
        "deltas": {key: round(float(value), 6) for key, value in deltas.items()},
        "reasons": [success for passed, success, _failure in checks if passed],
        "failures": [failure for passed, _success, failure in checks if not passed],
        "thresholds": {
            "min_positive_folds": min_positive_folds,
            "min_fold_objective_delta": min_fold_objective_delta,
            "min_average_max_delta": min_average_max_delta,
            "min_draw4_rate_delta_percent": min_draw4_rate_delta_percent,
            "min_draw5_count_delta": min_draw5_count_delta,
            "min_draw6_count_delta": min_draw6_count_delta,
            "min_recent_weighted_delta": min_recent_weighted_delta,
            "max_latest_fold_drop": max_latest_fold_drop,
            "max_fold_stddev_increase": max_fold_stddev_increase,
            "max_worst_fold_drop_percent": max_worst_fold_drop_percent,
            "min_average_unique_numbers": min_average_unique_numbers,
            "max_mean_overlap": max_mean_overlap,
            "max_pair_overlap": max_pair_overlap,
            "min_payout_roi_percent": min_payout_roi_percent,
            "max_roi_drop_percent": max_roi_drop_percent,
            "max_top1_payout_share": max_top1_payout_share,
        },
    }
