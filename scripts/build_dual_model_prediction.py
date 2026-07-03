#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dual_model_prediction.py

Dual Model Prediction

5口を以下の構成で生成する。
  1-2口: 全期間型 best model
  3-4口: Recent Era専用モデル
  5口  : Regime Strategy補正枠

Recent Era専用モデルがまだ存在しない場合は、全期間型モデルへfallbackする。

出力:
  outputs/evolution_best_prediction.csv
  outputs/holdout/latest_prediction_report.txt

注意:
  宝くじはランダム性が高く、当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Genome, generate_tickets, load_draws  # noqa: E402
from merge_evolution_shards import (  # noqa: E402
    fmt_ticket,
    load_model,
    load_role_strategy,
    make_role_ensemble_prediction_rows,
    ticket_key,
    write_prediction,
    write_prediction_report,
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_required_model(path: str) -> Dict[str, object]:
    item = load_model(Path(path))
    if item is None:
        raise SystemExit(f"cannot load model: {path}")
    return item


def load_optional_model(path: str, fallback: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return fallback, True
    item = load_model(p)
    if item is None:
        return fallback, True
    return item, False


def pick_unique_tickets(genome: Genome, draws: Sequence[object], count: int, used: Set[Tuple[int, ...]], overlap_limit: int) -> List[Tuple[int, ...]]:
    raw = generate_tickets(draws, genome, max(40, count * 20))
    selected: List[Tuple[int, ...]] = []
    for ticket in raw:
        key = ticket_key(ticket)
        if key in used:
            continue
        if all(len(set(key) & set(prev)) <= overlap_limit for prev in selected):
            selected.append(key)
            used.add(key)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for ticket in raw:
            key = ticket_key(ticket)
            if key in used:
                continue
            selected.append(key)
            used.add(key)
            if len(selected) >= count:
                break
    return selected[:count]


def make_row(
    *,
    rank: int,
    ticket: Tuple[int, ...],
    genome: Genome,
    source_model: str,
    method: str,
    support: str,
    base_latest_draw_no: int,
    base_latest_date: str,
    created_at: str,
    score: float,
) -> Dict[str, object]:
    return {
        "confidence_rank": rank,
        "base_latest_draw_no": base_latest_draw_no,
        "base_latest_date": base_latest_date,
        "prediction_draw_no": base_latest_draw_no + 1,
        "combo_index": rank,
        "numbers": fmt_ticket(ticket),
        "model_id": genome.id,
        "model_score": round(float(getattr(genome, "score", 0.0)), 6),
        "source_model": source_model,
        "prediction_method": method,
        "ensemble_score": round(score, 6),
        "support_models": support,
        "created_at": created_at,
    }


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Genome, str]:
    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit(f"no draws loaded: {args.csv}")
    latest = draws[-1]
    created_at = now_iso()

    full_item = load_required_model(args.full_model)
    full_genome: Genome = full_item["genome"]  # type: ignore[assignment]
    recent_item, recent_fallback = load_optional_model(args.recent_model, full_item)
    recent_genome: Genome = recent_item["genome"]  # type: ignore[assignment]

    rows: List[Dict[str, object]] = []
    used: Set[Tuple[int, ...]] = set()

    full_tickets = pick_unique_tickets(full_genome, draws, args.full_count, used, args.overlap_limit)
    for ticket in full_tickets:
        rows.append(
            make_row(
                rank=len(rows) + 1,
                ticket=ticket,
                genome=full_genome,
                source_model=str(full_item.get("path") or args.full_model),
                method="dual_full_period",
                support="全期間型: best_model",
                base_latest_draw_no=latest.draw_no,
                base_latest_date=latest.date,
                created_at=created_at,
                score=30.0 - len(rows),
            )
        )

    recent_tickets = pick_unique_tickets(recent_genome, draws, args.recent_count, used, args.overlap_limit)
    recent_support = "Recent Era型: 2020年以降専用モデル" if not recent_fallback else "Recent Era型: fallback to full-period model"
    for ticket in recent_tickets:
        rows.append(
            make_row(
                rank=len(rows) + 1,
                ticket=ticket,
                genome=recent_genome,
                source_model=str(recent_item.get("path") or args.recent_model),
                method="dual_recent_era",
                support=recent_support,
                base_latest_draw_no=latest.draw_no,
                base_latest_date=latest.date,
                created_at=created_at,
                score=20.0 - len(rows),
            )
        )

    regime_needed = max(0, args.purchase_count - len(rows))
    if regime_needed > 0:
        role_sequence = load_role_strategy(args.regime_strategy, regime_needed)
        regime_rows = make_role_ensemble_prediction_rows(
            full_genome,
            str(full_item.get("path") or args.full_model),
            draws,
            max(1, regime_needed),
            args.overlap_limit,
            role_sequence=role_sequence,
        )
        for r in regime_rows:
            nums = tuple(int(part) for part in str(r.get("numbers", "")).split())
            key = ticket_key(nums)
            if key in used:
                continue
            used.add(key)
            row = dict(r)
            row["confidence_rank"] = len(rows) + 1
            row["combo_index"] = len(rows) + 1
            row["prediction_method"] = "dual_regime"
            row["support_models"] = f"Regime型: {row.get('support_models', '')}"
            rows.append(row)
            if len(rows) >= args.purchase_count:
                break

    if len(rows) < args.purchase_count:
        fallback = pick_unique_tickets(full_genome, draws, args.purchase_count - len(rows), used, args.overlap_limit)
        for ticket in fallback:
            rows.append(
                make_row(
                    rank=len(rows) + 1,
                    ticket=ticket,
                    genome=full_genome,
                    source_model=str(full_item.get("path") or args.full_model),
                    method="dual_fallback",
                    support="補完: 全期間型best_model",
                    base_latest_draw_no=latest.draw_no,
                    base_latest_date=latest.date,
                    created_at=created_at,
                    score=10.0 - len(rows),
                )
            )

    rows = rows[: args.purchase_count]
    for i, row in enumerate(rows, start=1):
        row["confidence_rank"] = i
        row["combo_index"] = i
    return rows, full_genome, str(full_item.get("path") or args.full_model)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build LOTO7 dual model prediction.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--regime-strategy", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--full-count", type=int, default=2)
    parser.add_argument("--recent-count", type=int, default=2)
    parser.add_argument("--overlap-limit", type=int, default=4)
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.full_count < 0 or args.recent_count < 0:
        raise SystemExit("full/recent counts must be non-negative")
    if args.full_count + args.recent_count > args.purchase_count:
        raise SystemExit("full_count + recent_count must be <= purchase_count")

    rows, full_genome, source_model = build_rows(args)
    write_prediction(args.prediction, rows)
    write_prediction_report(
        args.prediction_report,
        rows,
        full_genome,
        source_model,
        model_count=2,
        min_models=1,
        selection_reason="Dual Model Prediction: 全期間型2口 + Recent Era型2口 + Regime型1口",
        prediction_mode="dual_model",
        role_strategy_path=args.regime_strategy,
    )
    print(f"[OK] prediction={args.prediction}")
    print(f"[OK] report={args.prediction_report}")
    print(f"[OK] full_model={args.full_model}")
    print(f"[OK] recent_model={args.recent_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
