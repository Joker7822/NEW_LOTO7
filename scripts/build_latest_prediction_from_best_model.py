#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_latest_prediction_from_best_model.py

採用済み loto7_best_model.json から、最新予測CSV/TXTだけを生成する。
merge_evolution_shards.py と違い、loto7_best_model.json は上書きしない。

用途:
  - LOTO7 Model Self Evolution workflow で採用モデル更新後に、
    outputs/evolution_best_prediction.csv を即時更新する。

出力:
  outputs/evolution_best_prediction.csv
  outputs/holdout/latest_prediction_report.txt

注意:
  宝くじはランダム性が高く、当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from loto7_evolution_trainer import Genome, load_draws
from merge_evolution_shards import (
    load_model,
    load_role_strategy,
    make_prediction_rows,
    make_role_ensemble_prediction_rows,
    write_prediction,
    write_prediction_report,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build latest LOTO7 prediction directly from adopted best model.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--prediction-mode", choices=["best_model", "role_ensemble"], default="role_ensemble")
    parser.add_argument("--role-strategy", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--overlap-limit", type=int, default=4)
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.overlap_limit < 0 or args.overlap_limit > 7:
        raise SystemExit("--overlap-limit must be between 0 and 7")

    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit(f"no draws loaded from {args.csv}")

    model_item = load_model(Path(args.best_model))
    if model_item is None:
        raise SystemExit(f"cannot load best model: {args.best_model}")
    genome: Genome = model_item["genome"]  # type: ignore[assignment]
    source_model = str(model_item.get("path") or args.best_model)

    if args.prediction_mode == "role_ensemble":
        role_sequence = load_role_strategy(args.role_strategy, args.purchase_count)
        rows = make_role_ensemble_prediction_rows(
            genome,
            source_model,
            draws,
            args.purchase_count,
            args.overlap_limit,
            role_sequence=role_sequence,
        )
    else:
        rows = make_prediction_rows(genome, source_model, draws, args.purchase_count)

    write_prediction(args.prediction, rows)
    write_prediction_report(
        args.prediction_report,
        rows,
        genome,
        source_model,
        model_count=1,
        min_models=1,
        selection_reason="採用済み loto7_best_model.json から直接生成",
        prediction_mode=args.prediction_mode,
        role_strategy_path=args.role_strategy if args.prediction_mode == "role_ensemble" else "",
    )
    print(f"[OK] prediction={args.prediction}")
    print(f"[OK] report={args.prediction_report}")
    print(f"[OK] model_id={genome.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
