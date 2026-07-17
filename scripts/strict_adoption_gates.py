#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict statistical adoption gates shared by LOTO7 production workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_json(path: str) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: str, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def recalibrated_conformal_number_pool(
    draws: Sequence[object],
    *,
    alpha: float = 0.20,
    calibration_draws: int = 104,
    min_train_draws: int = 52,
    half_life: float = 104.0,
    min_pool_size: int = 14,
    max_pool_size: int = 24,
    required_hits: int = 4,
) -> Dict[str, object]:
    """Recalibrate pool size against rolling prior-only draw coverage.

    The target is the probability that a historical draw contains at least
    ``required_hits`` main numbers from a pool built only from earlier draws.
    This directly matches the production minimum-conformal-hit constraint.
    """
    from scripts.generation4_core import draw_main, exp_weighted_number_scores

    lower = max(1, int(min_pool_size))
    upper = min(37, max(lower, int(max_pool_size)))
    required = min(7, max(1, int(required_hits)))
    target = min(1.0, max(0.0, 1.0 - float(alpha)))

    if len(draws) <= min_train_draws:
        pool = list(range(1, lower + 1))
        return {
            "numbers": pool,
            "ranked_numbers": list(range(1, 38)),
            "number_scores": {},
            "alpha": round(float(alpha), 6),
            "threshold": round((len(pool) - 1) / 36.0, 6),
            "calibration_score_count": 0,
            "calibration_draw_count": 0,
            "empirical_main_number_coverage": 0.0,
            "empirical_draw_coverage": 0.0,
            "target_draw_coverage": round(target, 6),
            "coverage_gap": round(-target, 6),
            "coverage_target_met": False,
            "required_main_hits": required,
            "pool_size": len(pool),
            "future_data_used": False,
            "half_life": half_life,
            "recalibration_method": "rolling_prior_top_k_minimum_hit_coverage_v1",
            "coverage_by_pool_size": {},
            "insufficient_data": True,
        }

    start = max(min_train_draws, len(draws) - max(1, int(calibration_draws)))
    statistics_by_size = {
        size: {"draw_successes": 0, "main_hits": 0, "total_main": 0}
        for size in range(lower, upper + 1)
    }

    for index in range(start, len(draws)):
        prior_scores = exp_weighted_number_scores(draws[:index], half_life=half_life)
        ranked = sorted(prior_scores, key=lambda number: (prior_scores[number], -number), reverse=True)
        actual = set(draw_main(draws[index]))
        for size, stats in statistics_by_size.items():
            hits = len(actual.intersection(ranked[:size]))
            stats["draw_successes"] += int(hits >= required)
            stats["main_hits"] += hits
            stats["total_main"] += len(actual)

    draw_count = max(0, len(draws) - start)
    coverage_by_size: Dict[str, Dict[str, object]] = {}
    selected_size: Optional[int] = None
    for size in range(lower, upper + 1):
        stats = statistics_by_size[size]
        draw_coverage = stats["draw_successes"] / draw_count if draw_count else 0.0
        main_coverage = stats["main_hits"] / stats["total_main"] if stats["total_main"] else 0.0
        coverage_by_size[str(size)] = {
            "draw_coverage": round(draw_coverage, 6),
            "main_number_coverage": round(main_coverage, 6),
            "draw_successes": int(stats["draw_successes"]),
            "draw_count": draw_count,
        }
        if selected_size is None and draw_coverage >= target:
            selected_size = size

    if selected_size is None:
        selected_size = max(
            range(lower, upper + 1),
            key=lambda size: (
                as_float(coverage_by_size[str(size)]["draw_coverage"]),
                as_float(coverage_by_size[str(size)]["main_number_coverage"]),
                -size,
            ),
        )

    current_scores = exp_weighted_number_scores(draws, half_life=half_life)
    ranked_numbers = sorted(
        current_scores,
        key=lambda number: (current_scores[number], -number),
        reverse=True,
    )
    pool = ranked_numbers[:selected_size]
    selected_stats = coverage_by_size[str(selected_size)]
    empirical_draw_coverage = as_float(selected_stats["draw_coverage"])
    empirical_main_coverage = as_float(selected_stats["main_number_coverage"])

    return {
        "numbers": sorted(pool),
        "ranked_numbers": ranked_numbers,
        "number_scores": {str(number): round(current_scores[number], 9) for number in ranked_numbers},
        "alpha": round(float(alpha), 6),
        "threshold": round((selected_size - 1) / 36.0, 6),
        "calibration_score_count": draw_count * 7,
        "calibration_draw_count": draw_count,
        "empirical_main_number_coverage": round(empirical_main_coverage, 6),
        "empirical_draw_coverage": round(empirical_draw_coverage, 6),
        "target_draw_coverage": round(target, 6),
        "coverage_gap": round(empirical_draw_coverage - target, 6),
        "coverage_target_met": bool(empirical_draw_coverage >= target),
        "required_main_hits": required,
        "pool_size": selected_size,
        "future_data_used": False,
        "half_life": half_life,
        "effective_draw_alpha": round(1.0 - empirical_draw_coverage, 6),
        "recalibration_method": "rolling_prior_top_k_minimum_hit_coverage_v1",
        "coverage_by_pool_size": coverage_by_size,
        "insufficient_data": False,
    }


def null_league_adoption_gate(
    payload: Optional[Mapping[str, object]], *, require_available: bool = False
) -> Dict[str, object]:
    """Return a fail-closed adoption decision for a Null Strategy League result."""
    if not payload:
        allowed = not require_available
        return {
            "available": False,
            "passed": allowed,
            "adoption_allowed": allowed,
            "reasons": [
                "null strategy league summary unavailable"
                + ("; adoption rejected" if require_available else "; strict check skipped")
            ],
        }

    decision = payload.get("decision", {})
    raw_passed = decision.get("passed") if isinstance(decision, Mapping) else None
    allowed = raw_passed is True
    if raw_passed is None and not require_available:
        allowed = True
    reasons = [
        f"null league passed={raw_passed}",
        f"null exceedance={payload.get('model_percentile')}",
        f"PBO={payload.get('pbo')}",
    ]
    if raw_passed is False:
        reasons.append("Null League failed; production adoption is completely rejected")
    elif raw_passed is None and require_available:
        reasons.append("Null League decision missing; production adoption is rejected")
    return {
        "available": True,
        "passed": raw_passed,
        "adoption_allowed": bool(allowed),
        "model_percentile": payload.get("model_percentile"),
        "pbo": payload.get("pbo"),
        "reasons": reasons,
    }


def nested_total_roi_gate(
    summary: Mapping[str, object],
    *,
    min_candidate_roi_percent: float = 8.0,
    min_roi_delta_percent: float = 0.0,
    expected_model_id: Optional[str] = None,
) -> Dict[str, object]:
    """Aggregate all sealed folds and reject candidates below both ROI standards."""
    reasons = []
    failures = []
    reference_model_id = str(summary.get("reference_model_id") or "")
    if expected_model_id and reference_model_id != expected_model_id:
        failures.append(
            f"nested model mismatch: nested={reference_model_id} candidate={expected_model_id}"
        )
    if bool(summary.get("future_leakage_detected")):
        failures.append("nested validation reports future leakage")

    folds = summary.get("folds")
    if not isinstance(folds, list) or not folds:
        failures.append("nested folds are unavailable")
        folds = []

    baseline_cost = baseline_payout = 0
    candidate_cost = candidate_payout = 0
    valid_folds = 0
    for fold in folds:
        if not isinstance(fold, Mapping):
            continue
        baseline = fold.get("baseline_metrics")
        candidate = fold.get("candidate_metrics")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            continue
        baseline_cost += as_int(baseline.get("total_cost"))
        baseline_payout += as_int(baseline.get("total_payout"))
        candidate_cost += as_int(candidate.get("total_cost"))
        candidate_payout += as_int(candidate.get("total_payout"))
        valid_folds += 1

    if valid_folds == 0 or baseline_cost <= 0 or candidate_cost <= 0:
        failures.append("nested aggregate cost/payout metrics are incomplete")

    baseline_roi = baseline_payout / baseline_cost * 100.0 if baseline_cost > 0 else 0.0
    candidate_roi = candidate_payout / candidate_cost * 100.0 if candidate_cost > 0 else 0.0
    roi_delta = candidate_roi - baseline_roi

    if candidate_roi < min_candidate_roi_percent:
        failures.append(
            f"nested total ROI failed: {candidate_roi:.3f}% < {min_candidate_roi_percent:.3f}%"
        )
    else:
        reasons.append(
            f"nested total ROI ok: {candidate_roi:.3f}% >= {min_candidate_roi_percent:.3f}%"
        )
    if roi_delta < min_roi_delta_percent:
        failures.append(
            f"nested total ROI delta failed: {roi_delta:.3f}pt < {min_roi_delta_percent:.3f}pt"
        )
    else:
        reasons.append(
            f"nested total ROI delta ok: {roi_delta:.3f}pt >= {min_roi_delta_percent:.3f}pt"
        )

    return {
        "passed": not failures,
        "valid_fold_count": valid_folds,
        "reference_model_id": reference_model_id,
        "expected_model_id": expected_model_id,
        "baseline": {
            "total_cost": baseline_cost,
            "total_payout": baseline_payout,
            "roi_percent": round(baseline_roi, 6),
        },
        "candidate": {
            "total_cost": candidate_cost,
            "total_payout": candidate_payout,
            "roi_percent": round(candidate_roi, 6),
        },
        "roi_delta_percent": round(roi_delta, 6),
        "thresholds": {
            "min_candidate_roi_percent": float(min_candidate_roi_percent),
            "min_roi_delta_percent": float(min_roi_delta_percent),
        },
        "reasons": reasons,
        "failures": failures,
    }
