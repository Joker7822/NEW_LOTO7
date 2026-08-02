#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the LOTO7 Null Strategy League from a deterministic seed-bank phase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.evolution.generation5 import seeds_for_phase  # noqa: E402
from loto7_evolution_trainer import load_draws  # noqa: E402
from merge_evolution_shards import load_prize_rows, select_target_indices  # noqa: E402
from scripts.null_strategy_league import (  # noqa: E402
    evaluate_strategy,
    now_iso,
    paired_model_pbo,
    probability_of_backtest_overfitting,
    summarize_records,
    write_report,
)
from scripts.robust_model_metrics import (  # noqa: E402
    evaluate_model_robust,
    indices_for_years,
    load_genome,
    percentile,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fixed-seed LOTO7 Null Strategy League"
    )
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="loto7_best_model.json")
    parser.add_argument(
        "--seed-bank", default="outputs/generation5/null_seed_bank.json"
    )
    parser.add_argument(
        "--seed-phase",
        choices=["learning", "selection", "final"],
        default="final",
    )
    parser.add_argument(
        "--summary",
        default="outputs/generation4/null_strategy_league_summary.json",
    )
    parser.add_argument(
        "--report",
        default="outputs/generation4/null_strategy_league_report.txt",
    )
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="0 uses every seed in the phase",
    )
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--max-null-exceedance", type=float, default=0.10)
    parser.add_argument("--max-pbo", type=float, default=0.40)
    args = parser.parse_args()

    bank = json.loads(Path(args.seed_bank).read_text(encoding="utf-8"))
    seeds = seeds_for_phase(bank, args.seed_phase, args.simulations)
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base = select_target_indices(
        draws,
        min_train_draws=52,
        holdout_start_draw=2,
        holdout_end_draw=None,
    )
    target_indices = indices_for_years(
        draws, base, args.start_year, args.end_year
    )
    if not target_indices:
        raise SystemExit("no null-league target draws selected")

    model_metrics = evaluate_model_robust(
        genome=load_genome(args.model),
        model_path=args.model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
        bootstrap_samples=200,
        bootstrap_seed=seeds[0],
        include_draw_records=True,
    )
    model_records = model_metrics.get("draw_records", [])
    if not isinstance(model_records, list):
        raise SystemExit("model draw records missing")
    model_summary = {
        "roi_percent": float(model_metrics.get("roi_percent", 0.0)),
        "roi_excluding_top1_percent": float(
            model_metrics.get("roi_excluding_top1_percent", 0.0)
        ),
        "median_year_roi_percent": float(
            model_metrics.get("median_year_roi_percent", 0.0)
        ),
        "max_main_match": float(model_metrics.get("max_main_match", 0.0)),
    }
    model_summary["robust_score"] = (
        0.50 * model_summary["roi_excluding_top1_percent"]
        + 0.25 * model_summary["median_year_roi_percent"]
        + 0.15 * model_summary["roi_percent"]
        + 0.10 * model_summary["max_main_match"] * 10.0
    )

    strategy_names = (
        "random",
        "balanced",
        "frequency",
        "dormancy",
        "recent",
        "hybrid",
    )
    null_results: List[Dict[str, object]] = []
    all_records: List[Sequence[Mapping[str, object]]] = [model_records]
    null_record_sets: List[Sequence[Mapping[str, object]]] = []
    for simulation, seed in enumerate(seeds):
        strategy = strategy_names[simulation % len(strategy_names)]
        records = evaluate_strategy(
            draws,
            target_indices,
            prize_rows,
            strategy=strategy,
            seed=seed,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
        )
        metrics = summarize_records(records)
        null_results.append(
            {
                "simulation": simulation,
                "strategy": strategy,
                "seed": seed,
                **{
                    key: round(value, 6)
                    for key, value in metrics.items()
                },
            }
        )
        null_record_sets.append(records)
        all_records.append(records)

    model_score = float(model_summary["robust_score"])
    exceedance = sum(
        1
        for result in null_results
        if float(result["robust_score"]) >= model_score
    ) / len(null_results)
    league_pbo = probability_of_backtest_overfitting(all_records, block_count=6)
    paired_pbo = paired_model_pbo(
        model_records,
        null_record_sets,
        block_count=6,
    )
    null_scores = [float(result["robust_score"]) for result in null_results]
    null_top1 = [
        float(result["roi_excluding_top1_percent"])
        for result in null_results
    ]
    passed = bool(
        exceedance <= args.max_null_exceedance
        and float(paired_pbo["pbo"]) <= args.max_pbo
    )
    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_fixed_seed_null_strategy_league",
        "csv": args.csv,
        "model": args.model,
        "target_draws": len(target_indices),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "null_simulations": len(seeds),
        "strategy_types": list(strategy_names),
        "seed_bank": {
            "path": args.seed_bank,
            "phase": args.seed_phase,
            "version": bank.get("version"),
            "dataset_sha256": bank.get("dataset_sha256"),
            "evaluator_version": bank.get("evaluator_version"),
        },
        "model_metrics": {
            key: round(value, 6) for key, value in model_summary.items()
        },
        "null_distribution": {
            "robust_score_p50": round(percentile(null_scores, 0.50), 6),
            "robust_score_p90": round(percentile(null_scores, 0.90), 6),
            "robust_score_p95": round(percentile(null_scores, 0.95), 6),
            "top1_removed_roi_p90": round(percentile(null_top1, 0.90), 6),
            "top1_removed_roi_p95": round(percentile(null_top1, 0.95), 6),
        },
        "model_percentile": round(exceedance, 6),
        "pbo": paired_pbo.get("pbo"),
        "pbo_detail": paired_pbo,
        "league_pbo": league_pbo.get("pbo"),
        "league_pbo_detail": league_pbo,
        "decision": {
            "passed": passed,
            "max_null_exceedance": args.max_null_exceedance,
            "max_pbo": args.max_pbo,
            "pbo_method": "paired_model_vs_null_is_oos_reversal",
            "reasons": [
                f"null exceedance={exceedance:.6f}",
                f"paired PBO={float(paired_pbo['pbo']):.6f}",
                f"league-wide PBO diagnostic={float(league_pbo['pbo']):.6f}",
            ],
        },
        "null_results": null_results,
        "notes": [
            "The same dataset and evaluator version produce the same disjoint seed bank.",
            "The final phase is not used during Generation 5 candidate search.",
            "Paired PBO measures model-vs-null IS/OOS reversals and is the PBO gate.",
            "League-wide CSCV PBO remains diagnostic because null simulations are challengers, not selectable models.",
            "PBO is a diagnostic and not proof of predictability.",
        ],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(Path(args.report), payload)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "null_results"
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
