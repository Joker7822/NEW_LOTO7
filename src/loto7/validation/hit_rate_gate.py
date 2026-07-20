#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accuracy-first promotion gate for sealed nested validation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _metrics(fold: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = fold.get(name)
    return value if isinstance(value, Mapping) else {}


def _aggregate(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    items = list(rows)
    target_draws = sum(as_int(item.get("target_draws")) for item in items)
    result: Dict[str, object] = {"target_draws": target_draws}
    for threshold in (4, 5, 6, 7):
        key = "draw_main7_count" if threshold == 7 else f"draw_main{threshold}_plus_count"
        count = sum(as_int(item.get(key)) for item in items)
        rate = count / target_draws if target_draws else 0.0
        result[key] = count
        result[f"draw_main{threshold}_plus_rate"] = round(rate, 6)
        result[f"draw_main{threshold}_plus_rate_percent"] = round(rate * 100.0, 3)
    weighted_average = (
        sum(as_float(item.get("average_max_main_match")) * as_int(item.get("target_draws")) for item in items)
        / target_draws
        if target_draws
        else 0.0
    )
    weighted_objective = (
        sum(as_float(item.get("hit_objective_score")) * as_int(item.get("target_draws")) for item in items)
        / target_draws
        if target_draws
        else 0.0
    )
    result["average_max_main_match"] = round(weighted_average, 6)
    result["hit_objective_score"] = round(weighted_objective, 6)
    return result


def evaluate_nested_hit_gate(
    nested: Mapping[str, object],
    *,
    min_positive_folds: int = 2,
    min_fold_objective_delta: float = 0.05,
    min_objective_delta: float = 0.05,
    min_draw4_rate_delta_percent: float = 0.0,
    min_draw5_count_delta: int = 0,
    min_average_max_delta: float = 0.0,
) -> Dict[str, object]:
    folds_value = nested.get("folds")
    folds = folds_value if isinstance(folds_value, list) else []
    baseline_rows: List[Mapping[str, object]] = []
    candidate_rows: List[Mapping[str, object]] = []
    fold_results: List[Dict[str, object]] = []
    positive_folds = 0

    for fold in folds:
        if not isinstance(fold, Mapping):
            continue
        baseline = _metrics(fold, "baseline_metrics")
        candidate = _metrics(fold, "candidate_metrics")
        baseline_rows.append(baseline)
        candidate_rows.append(candidate)
        delta = as_float(candidate.get("hit_objective_score")) - as_float(baseline.get("hit_objective_score"))
        improved = delta >= min_fold_objective_delta
        if improved:
            positive_folds += 1
        fold_results.append(
            {
                "label": fold.get("label"),
                "evaluation_year": fold.get("evaluation_year"),
                "baseline_hit_objective_score": as_float(baseline.get("hit_objective_score")),
                "candidate_hit_objective_score": as_float(candidate.get("hit_objective_score")),
                "hit_objective_delta": round(delta, 6),
                "materially_improved": improved,
            }
        )

    baseline = _aggregate(baseline_rows)
    candidate = _aggregate(candidate_rows)
    objective_delta = as_float(candidate.get("hit_objective_score")) - as_float(baseline.get("hit_objective_score"))
    draw4_delta = as_float(candidate.get("draw_main4_plus_rate_percent")) - as_float(baseline.get("draw_main4_plus_rate_percent"))
    draw5_count_delta = as_int(candidate.get("draw_main5_plus_count")) - as_int(baseline.get("draw_main5_plus_count"))
    average_max_delta = as_float(candidate.get("average_max_main_match")) - as_float(baseline.get("average_max_main_match"))

    checks = [
        (
            positive_folds >= min_positive_folds,
            f"materially improved folds={positive_folds} >= {min_positive_folds}",
            f"materially improved folds failed: {positive_folds} < {min_positive_folds}",
        ),
        (
            objective_delta >= min_objective_delta,
            f"hit objective delta={objective_delta:.6f} >= {min_objective_delta:.6f}",
            f"hit objective delta failed: {objective_delta:.6f} < {min_objective_delta:.6f}",
        ),
        (
            draw4_delta >= min_draw4_rate_delta_percent,
            f"draw main4+ rate delta={draw4_delta:.3f}pt",
            f"draw main4+ rate delta failed: {draw4_delta:.3f}pt < {min_draw4_rate_delta_percent:.3f}pt",
        ),
        (
            draw5_count_delta >= min_draw5_count_delta,
            f"draw main5+ count delta={draw5_count_delta}",
            f"draw main5+ count delta failed: {draw5_count_delta} < {min_draw5_count_delta}",
        ),
        (
            average_max_delta >= min_average_max_delta,
            f"average max-main delta={average_max_delta:.6f}",
            f"average max-main delta failed: {average_max_delta:.6f} < {min_average_max_delta:.6f}",
        ),
    ]
    reasons = [message for passed, message, _ in checks if passed]
    failures = [message for passed, _, message in checks if not passed]
    return {
        "created_at": now_iso(),
        "kind": "loto7_nested_high_match_gate",
        "passed": not failures,
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "hit_objective_score": round(objective_delta, 6),
            "draw_main4_plus_rate_percent": round(draw4_delta, 3),
            "draw_main5_plus_count": draw5_count_delta,
            "average_max_main_match": round(average_max_delta, 6),
        },
        "thresholds": {
            "min_positive_folds": min_positive_folds,
            "min_fold_objective_delta": min_fold_objective_delta,
            "min_objective_delta": min_objective_delta,
            "min_draw4_rate_delta_percent": min_draw4_rate_delta_percent,
            "min_draw5_count_delta": min_draw5_count_delta,
            "min_average_max_delta": min_average_max_delta,
        },
        "positive_folds": positive_folds,
        "folds": fold_results,
        "reasons": reasons,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate payout-independent high-match promotion conditions.")
    parser.add_argument("--nested-summary", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--min-fold-objective-delta", type=float, default=0.05)
    parser.add_argument("--min-objective-delta", type=float, default=0.05)
    parser.add_argument("--min-draw4-rate-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-draw5-count-delta", type=int, default=0)
    parser.add_argument("--min-average-max-delta", type=float, default=0.0)
    args = parser.parse_args()

    nested = json.loads(Path(args.nested_summary).read_text(encoding="utf-8"))
    payload = evaluate_nested_hit_gate(
        nested,
        min_positive_folds=args.min_positive_folds,
        min_fold_objective_delta=args.min_fold_objective_delta,
        min_objective_delta=args.min_objective_delta,
        min_draw4_rate_delta_percent=args.min_draw4_rate_delta_percent,
        min_draw5_count_delta=args.min_draw5_count_delta,
        min_average_max_delta=args.min_average_max_delta,
    )
    output = Path(args.decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
