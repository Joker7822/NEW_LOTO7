#!/usr/bin/env python3
"""Apply paired chronological evidence and canonical metrics to G5 summary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from loto7.evaluation.model_audit import evaluate_model, fold_indices, year_of
from loto7.evaluation.statistics import paired_moving_block_bootstrap
from loto7_evolution_trainer import load_draws
from merge_evolution_shards import load_prize_rows, select_target_indices


def read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument(
        "--candidate", default="outputs/generation5/generation5_candidate_model.json"
    )
    parser.add_argument("--baseline", default="loto7_best_model.json")
    parser.add_argument("--summary", default="outputs/generation5/generation5_summary.json")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-block-size", type=int, default=0)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    prizes = load_prize_rows(args.csv)
    indices = select_target_indices(
        draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None
    )
    indices = [index for index in indices if year_of(draws[index]) >= args.start_year]
    if len(indices) < 30:
        raise SystemExit("not enough chronological targets")
    candidate = evaluate_model(
        model_path=args.candidate,
        draws=draws,
        prize_rows=prizes,
        target_indices=indices,
    )
    baseline = evaluate_model(
        model_path=args.baseline,
        draws=draws,
        prize_rows=prizes,
        target_indices=indices,
    )
    candidate_rows = candidate["draw_outcomes"]
    baseline_rows = baseline["draw_outcomes"]
    paired_average = paired_moving_block_bootstrap(
        [float(row["max_main_match"]) for row in candidate_rows],
        [float(row["max_main_match"]) for row in baseline_rows],
        samples=args.bootstrap_samples,
        block_size=args.bootstrap_block_size,
    )
    paired_draw4 = paired_moving_block_bootstrap(
        [1.0 if row["draw4_plus"] else 0.0 for row in candidate_rows],
        [1.0 if row["draw4_plus"] else 0.0 for row in baseline_rows],
        samples=args.bootstrap_samples,
        block_size=args.bootstrap_block_size,
        seed=20260732,
    )
    folds = []
    positive_folds = 0
    for number, part in enumerate(fold_indices(indices), start=1):
        left = evaluate_model(
            model_path=args.candidate,
            draws=draws,
            prize_rows=prizes,
            target_indices=part,
        )
        right = evaluate_model(
            model_path=args.baseline,
            draws=draws,
            prize_rows=prizes,
            target_indices=part,
        )
        delta = float(left["hit_first_objective_score"]) - float(right["hit_first_objective_score"])
        positive_folds += int(delta >= 0.05)
        folds.append({"fold": number, "target_draws": len(part), "delta": round(delta, 6)})
    average_delta = float(candidate["average_max_main_match"]) - float(
        baseline["average_max_main_match"]
    )
    draw4_delta = float(candidate["draw_main4_plus_rate_percent"]) - float(
        baseline["draw_main4_plus_rate_percent"]
    )
    previous = read_json(args.summary) if Path(args.summary).exists() else {}
    old_gate = previous.get("adoption", {})
    old_passed = bool(old_gate.get("passed", True)) if isinstance(old_gate, Mapping) else True
    checks = {
        "generation5_search_gate": old_passed,
        "positive_folds_3_of_5": positive_folds >= 3,
        "average_max_delta_at_least_0_03": average_delta >= 0.03,
        "draw4_delta_at_least_0_50pt": draw4_delta >= 0.50,
        "draw5_non_regression": int(candidate["draw_main5_plus_count"])
        >= int(baseline["draw_main5_plus_count"]),
        "draw6_non_regression": int(candidate["draw_main6_plus_count"])
        >= int(baseline["draw_main6_plus_count"]),
        "paired_average_ci_positive": float(paired_average["ci_lower"]) > 0.0,
        "paired_draw4_ci_positive": float(paired_draw4["ci_lower"]) > 0.0,
        "average_unique_at_least_13": float(candidate["average_portfolio_unique_numbers"]) >= 13.0,
        "mean_overlap_at_most_4_2": float(candidate["mean_ticket_pair_overlap"]) <= 4.2,
        "max_overlap_at_most_4": int(candidate["max_ticket_pair_overlap"]) <= 4,
        "payout_floor": float(candidate["payout_roi_percent"])
        >= max(8.0, float(baseline["payout_roi_percent"]) - 5.0),
        "top1_share_at_most_0_50": float(candidate["top1_payout_share"]) <= 0.50,
    }
    failures = [name for name, passed in checks.items() if not passed]
    adoption = {
        "kind": "loto7_generation5_hardened_adoption_gate",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "positive_folds": positive_folds,
        "folds": folds,
        "deltas": {
            "average_max_main_match": round(average_delta, 6),
            "draw_main4_plus_rate_percent": round(draw4_delta, 6),
            "draw_main5_plus_count": int(candidate["draw_main5_plus_count"])
            - int(baseline["draw_main5_plus_count"]),
            "draw_main6_plus_count": int(candidate["draw_main6_plus_count"])
            - int(baseline["draw_main6_plus_count"]),
        },
        "paired_bootstrap": {
            "average_max_main_match": paired_average,
            "draw4_plus_rate": paired_draw4,
        },
        "legacy_search_gate": old_gate,
    }
    previous.update(
        {
            "kind": "loto7_generation5_evolution_hardened",
            "metric_schema_version": candidate["metric_schema_version"],
            "baseline": {"full_metrics": baseline},
            "candidate": {"full_metrics": candidate},
            "adoption": adoption,
        }
    )
    Path(args.summary).write_text(
        json.dumps(previous, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(adoption, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
