#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""True nested walk-forward validation for LOTO7 evolution.

For every fold, model evolution is run on a physically truncated CSV that ends
at the selection year.  The resulting fold model is then evaluated on the next
calendar year from the full CSV.  Evaluation-year rows therefore cannot enter
model selection.

Example fold ``2021:2022:2023`` means:
  training history: through 2021
  model selection/evolution targets: 2022
  sealed evaluation: 2023
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, load_draws  # noqa: E402
from merge_evolution_shards import load_prize_rows, select_target_indices  # noqa: E402
from scripts.robust_model_metrics import evaluate_model_robust, indices_for_years, load_genome  # noqa: E402

Fold = Tuple[int, int, int]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_year(value: object) -> int:
    match = re.search(r"(20\d{2}|19\d{2})", str(value or ""))
    return int(match.group(1)) if match else 0


def validate_fold_boundaries(train_end_year: int, selection_year: int, evaluation_year: int) -> None:
    if not (train_end_year < selection_year < evaluation_year):
        raise ValueError(
            f"invalid nested fold: train_end={train_end_year} selection={selection_year} evaluation={evaluation_year}"
        )


def parse_folds(raw: str) -> List[Fold]:
    folds: List[Fold] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        parts = [int(value.strip()) for value in item.split(":")]
        if len(parts) != 3:
            raise SystemExit(f"invalid fold: {item}; expected TRAIN_END:SELECTION:EVALUATION")
        fold = (parts[0], parts[1], parts[2])
        validate_fold_boundaries(*fold)
        folds.append(fold)
    if not folds:
        raise SystemExit("no nested folds configured")
    return folds


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if not fieldnames or not rows:
        raise SystemExit(f"empty CSV: {path}")
    return fieldnames, rows


def row_year(row: Dict[str, str]) -> int:
    for key in ("抽せん日", "抽選日", "date", "draw_date"):
        year = parse_year(row.get(key))
        if year:
            return year
    return 0


def write_truncated_csv(source: Path, destination: Path, selection_year: int) -> Dict[str, object]:
    fieldnames, rows = read_csv(source)
    selected = [row for row in rows if 0 < row_year(row) <= selection_year]
    if not selected:
        raise SystemExit(f"no rows through selection year {selection_year}")
    max_year = max(row_year(row) for row in selected)
    if max_year > selection_year:
        raise SystemExit("future leakage while constructing truncated CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    return {"row_count": len(selected), "max_year": max_year}


def first_draw_for_year(draws: Sequence[Draw], year: int) -> int:
    matches = [draw.draw_no for draw in draws if parse_year(getattr(draw, "date", "")) == year]
    if not matches:
        raise SystemExit(f"no draw found for selection year {year}")
    return min(matches)


def evaluate_year(
    *,
    model_path: str,
    full_draws: Sequence[Draw],
    prize_rows: Dict[int, Dict[str, str]],
    base_indices: Sequence[int],
    year: int,
    purchase_count: int,
    unit_cost: int,
    bootstrap_samples: int,
) -> Dict[str, object]:
    indices = indices_for_years(full_draws, base_indices, year, year)
    if not indices:
        raise SystemExit(f"no evaluation draws for year {year}")
    return evaluate_model_robust(
        genome=load_genome(model_path),
        model_path=model_path,
        draws=full_draws,
        prize_rows=prize_rows,
        target_indices=indices,
        purchase_count=purchase_count,
        unit_cost=unit_cost,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=year,
    )


def median(values: Sequence[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def write_report(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "LOTO7 Nested Walk-Forward Validation",
        "======================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"reference_model_id: {payload.get('reference_model_id')}",
        f"fold_count: {payload.get('fold_count')}",
        f"future_leakage_detected: {payload.get('future_leakage_detected')}",
        f"positive_roi_delta_folds: {payload.get('positive_roi_delta_folds')}",
        f"median_roi_delta_percent: {payload.get('median_roi_delta_percent')}",
        f"worst_roi_delta_percent: {payload.get('worst_roi_delta_percent')}",
        f"median_top1_removed_roi_delta_percent: {payload.get('median_top1_removed_roi_delta_percent')}",
        "",
        "[Folds]",
    ]
    for fold in payload.get("folds", []) if isinstance(payload.get("folds"), list) else []:
        if not isinstance(fold, dict):
            continue
        lines.extend(
            [
                f"- {fold.get('label')}",
                f"  leakage_check: {fold.get('leakage_check')}",
                f"  fold_model_id: {fold.get('fold_model_id')}",
                f"  baseline_roi: {fold.get('baseline_metrics', {}).get('roi_percent') if isinstance(fold.get('baseline_metrics'), dict) else ''}",
                f"  candidate_roi: {fold.get('candidate_metrics', {}).get('roi_percent') if isinstance(fold.get('candidate_metrics'), dict) else ''}",
                f"  roi_delta: {fold.get('roi_delta_percent')}",
                f"  top1_removed_delta: {fold.get('top1_removed_roi_delta_percent')}",
            ]
        )
    lines.extend(
        [
            "",
            "This validation evolves each fold model without access to its evaluation year.",
            "It is historical validation and does not guarantee future lottery results.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run true nested walk-forward evolution and sealed evaluation.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--seed-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--output-dir", default="outputs/validation/nested_walk_forward")
    parser.add_argument("--summary", default="outputs/validation/nested_walk_forward_summary.json")
    parser.add_argument("--report", default="outputs/validation/nested_walk_forward_report.txt")
    parser.add_argument("--folds", default="2021:2022:2023,2022:2023:2024,2023:2024:2025")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-runtime-minutes-per-fold", type=float, default=35.0)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    args = parser.parse_args()

    source_csv = Path(args.csv)
    seed_model = Path(args.seed_model)
    if not source_csv.exists() or not seed_model.exists():
        raise SystemExit("CSV or seed model is missing")

    folds = parse_folds(args.folds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_draws = load_draws(str(source_csv))
    prize_rows = load_prize_rows(str(source_csv))
    base_indices = select_target_indices(full_draws, min_train_draws=1, holdout_start_draw=2, holdout_end_draw=None)
    reference_genome = load_genome(str(seed_model))
    fold_results: List[Dict[str, object]] = []

    for fold_index, (train_end_year, selection_year, evaluation_year) in enumerate(folds, start=1):
        label = f"train_through_{train_end_year}_select_{selection_year}_evaluate_{evaluation_year}"
        fold_dir = output_dir / f"fold_{fold_index}_{evaluation_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        truncated_csv = fold_dir / "selection_data.csv"
        subset = write_truncated_csv(source_csv, truncated_csv, selection_year)
        leakage_check = bool(int(subset["max_year"]) < evaluation_year)
        if not leakage_check:
            raise SystemExit(f"future leakage detected in {label}")

        fold_best = fold_dir / "best_model.json"
        shutil.copyfile(seed_model, fold_best)
        truncated_draws = load_draws(str(truncated_csv))
        holdout_start = first_draw_for_year(truncated_draws, selection_year)
        command = [
            sys.executable,
            str(REPO_ROOT / "loto7_model_self_evolver.py"),
            "--csv", str(truncated_csv),
            "--best-model", str(fold_best),
            "--seed-patterns", str(fold_best), str(seed_model),
            "--iterations", str(args.iterations),
            "--purchase-count", str(args.purchase_count),
            "--unit-cost", str(args.unit_cost),
            "--min-train-draws", "1",
            "--holdout-start-draw", str(holdout_start),
            "--max-targets", "0",
            "--seed", str(args.seed + fold_index),
            "--state", str(fold_dir / "state.json"),
            "--history", str(fold_dir / "history.csv"),
            "--summary", str(fold_dir / "evolution_summary.json"),
            "--report", str(fold_dir / "evolution_report.txt"),
            "--candidate-model", str(fold_dir / "candidate_model.json"),
            "--max-runtime-minutes", str(args.max_runtime_minutes_per_fold),
            "--safe-exit-minutes", "2",
            "--min-roi-delta-percent", "0.0",
            "--min-profit-delta", "0",
            "--allow-high-grade-drop",
            "--apply",
            "--no-resume",
        ]
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

        baseline_metrics = evaluate_year(
            model_path=str(seed_model),
            full_draws=full_draws,
            prize_rows=prize_rows,
            base_indices=base_indices,
            year=evaluation_year,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
            bootstrap_samples=args.bootstrap_samples,
        )
        candidate_metrics = evaluate_year(
            model_path=str(fold_best),
            full_draws=full_draws,
            prize_rows=prize_rows,
            base_indices=base_indices,
            year=evaluation_year,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
            bootstrap_samples=args.bootstrap_samples,
        )
        roi_delta = float(candidate_metrics.get("roi_percent", 0.0)) - float(baseline_metrics.get("roi_percent", 0.0))
        top1_delta = float(candidate_metrics.get("roi_excluding_top1_percent", 0.0)) - float(baseline_metrics.get("roi_excluding_top1_percent", 0.0))
        fold_results.append(
            {
                "label": label,
                "train_end_year": train_end_year,
                "selection_year": selection_year,
                "evaluation_year": evaluation_year,
                "selection_csv_rows": subset["row_count"],
                "selection_csv_max_year": subset["max_year"],
                "leakage_check": leakage_check,
                "fold_model_path": str(fold_best),
                "fold_model_id": load_genome(str(fold_best)).id,
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "roi_delta_percent": round(roi_delta, 3),
                "top1_removed_roi_delta_percent": round(top1_delta, 3),
            }
        )

    roi_deltas = [float(fold["roi_delta_percent"]) for fold in fold_results]
    top1_deltas = [float(fold["top1_removed_roi_delta_percent"]) for fold in fold_results]
    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_true_nested_walk_forward",
        "reference_model_path": str(seed_model),
        "reference_model_id": reference_genome.id,
        "fold_count": len(fold_results),
        "folds": fold_results,
        "future_leakage_detected": not all(bool(fold["leakage_check"]) for fold in fold_results),
        "positive_roi_delta_folds": sum(1 for value in roi_deltas if value >= 0.0),
        "positive_top1_removed_delta_folds": sum(1 for value in top1_deltas if value >= 0.0),
        "median_roi_delta_percent": median(roi_deltas),
        "worst_roi_delta_percent": round(min(roi_deltas), 3) if roi_deltas else 0.0,
        "median_top1_removed_roi_delta_percent": median(top1_deltas),
        "worst_top1_removed_roi_delta_percent": round(min(top1_deltas), 3) if top1_deltas else 0.0,
        "notes": [
            "Every fold selection CSV ends before its sealed evaluation year.",
            "Fold evolution is performed independently and cannot read evaluation-year rows.",
            "Historical validation does not guarantee future winnings.",
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(Path(args.report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
