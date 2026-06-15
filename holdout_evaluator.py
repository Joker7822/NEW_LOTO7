#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holdout_evaluator.py

進化型探索で作成した loto7_best_model.json を固定し、未使用区間の
holdout成績を実当せん金額ベースで評価する。

目的:
    - 進化済みGenomeを固定したまま、holdout区間だけを検証する
    - 各検証回の予測生成は train = draws[:idx] のみを使い、対象回以降は使わない
    - 実当せん金額、購入金額、収支、回収率、年別回収率を出力する
    - 当せん金額欠損を検出し、必要に応じて失敗扱いにできる

例:
    python holdout_evaluator.py \
      --csv loto7.csv \
      --best-model loto7_best_model.json \
      --holdout-start-draw 641 \
      --purchase-count 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loto7_evolution_trainer import (
    Draw,
    evaluate_ticket,
    generate_tickets,
    load_best_model,
    load_draws,
)

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]


def draw_no_int(text: object) -> Optional[int]:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def parse_money_yen(text: object) -> int:
    raw = str(text or "").strip()
    if not raw or raw == "該当なし":
        return 0
    m = re.search(r"([0-9,]+)", raw)
    if not m:
        return 0
    return int(m.group(1).replace(",", ""))


def load_prize_rows(csv_path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = draw_no_int(row.get("回別"))
            if no is not None:
                out[no] = {k: str(v or "").strip() for k, v in row.items()}
    return out


def prize_amount_for_rank(row: Dict[str, str], rank: str) -> int:
    if rank == "外れ":
        return 0
    return parse_money_yen(row.get(f"{rank}当選金額", ""))


def has_any_prize_amount(row: Dict[str, str]) -> bool:
    return any(str(row.get(f"{rank}当選金額", "")).strip() for rank in PRIZE_RANKS)


def fmt_ticket(ticket: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in ticket)


def draw_year(draw: Draw) -> str:
    text = str(draw.date or "")
    return text[:4] if re.match(r"^\d{4}", text) else "unknown"


def empty_year_stats() -> Dict[str, object]:
    return {
        "target_draws": 0,
        "total_tickets": 0,
        "total_cost": 0,
        "total_payout": 0,
        "profit": 0,
        "roi": 0.0,
        "roi_percent": 0.0,
        "max_main_match": 0,
        "rank_counts": {rank: 0 for rank in RANK_ORDER},
    }


def update_year_stats(stats: Dict[str, object], *, cost: int, payout: int, rank: str, main_match: int) -> None:
    stats["total_tickets"] = int(stats["total_tickets"]) + 1
    stats["total_cost"] = int(stats["total_cost"]) + cost
    stats["total_payout"] = int(stats["total_payout"]) + payout
    stats["profit"] = int(stats["total_payout"]) - int(stats["total_cost"])
    stats["max_main_match"] = max(int(stats["max_main_match"]), main_match)
    rank_counts = stats["rank_counts"]
    assert isinstance(rank_counts, dict)
    rank_counts[rank] = int(rank_counts.get(rank, 0)) + 1
    total_cost = int(stats["total_cost"])
    total_payout = int(stats["total_payout"])
    roi = (total_payout / total_cost) if total_cost else 0.0
    stats["roi"] = round(roi, 6)
    stats["roi_percent"] = round(roi * 100.0, 3)


def evaluate_holdout(args: argparse.Namespace) -> int:
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    genome = load_best_model(args.best_model)
    if genome is None:
        raise SystemExit(f"best model not found or invalid: {args.best_model}")

    target_indices = []
    for idx, draw in enumerate(draws):
        if idx < args.min_train_draws:
            continue
        if draw.draw_no < args.holdout_start_draw:
            continue
        if args.holdout_end_draw is not None and draw.draw_no > args.holdout_end_draw:
            continue
        target_indices.append(idx)

    if not target_indices:
        raise SystemExit("no holdout targets selected")

    detail_rows: List[Dict[str, object]] = []
    rank_counts = {rank: 0 for rank in RANK_ORDER}
    year_summary: Dict[str, Dict[str, object]] = {}
    max_main_match = 0
    total_cost = 0
    total_payout = 0
    total_tickets = 0
    missing_prize_draws = []

    for idx in target_indices:
        target: Draw = draws[idx]
        train = draws[:idx]
        tickets = generate_tickets(train, genome, args.purchase_count)
        prize_row = prize_rows.get(target.draw_no, {})
        y = draw_year(target)
        if y not in year_summary:
            year_summary[y] = empty_year_stats()
        year_summary[y]["target_draws"] = int(year_summary[y]["target_draws"]) + 1

        if not prize_row or not has_any_prize_amount(prize_row):
            missing_prize_draws.append(target.draw_no)

        for combo_index, ticket in enumerate(tickets, start=1):
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            payout = prize_amount_for_rank(prize_row, rank)
            cost = args.unit_cost
            total_cost += cost
            total_payout += payout
            total_tickets += 1
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            max_main_match = max(max_main_match, main_match)
            update_year_stats(year_summary[y], cost=cost, payout=payout, rank=rank, main_match=main_match)

            detail_rows.append(
                {
                    "draw_no": target.draw_no,
                    "date": target.date,
                    "year": y,
                    "combo_index": combo_index,
                    "ticket": fmt_ticket(ticket),
                    "actual_main": fmt_ticket(target.main),
                    "actual_bonus": fmt_ticket(target.bonus),
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "rank": rank,
                    "purchase_cost": cost,
                    "prize_amount": payout,
                    "profit": payout - cost,
                    "prize_data_missing": 1 if target.draw_no in missing_prize_draws else 0,
                }
            )

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "draw_no", "date", "year", "combo_index", "ticket", "actual_main", "actual_bonus",
        "main_match", "bonus_match", "rank", "purchase_cost", "prize_amount", "profit", "prize_data_missing",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    profit = total_payout - total_cost
    roi = (total_payout / total_cost) if total_cost else 0.0
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "csv": args.csv,
        "best_model": args.best_model,
        "model_id": genome.id,
        "model_score": genome.score,
        "holdout_start_draw": args.holdout_start_draw,
        "holdout_end_draw": args.holdout_end_draw,
        "target_draws": len(target_indices),
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "total_tickets": total_tickets,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": profit,
        "roi": round(roi, 6),
        "roi_percent": round(roi * 100.0, 3),
        "max_main_match": max_main_match,
        "rank_counts": rank_counts,
        "missing_prize_draw_count": len(set(missing_prize_draws)),
        "missing_prize_draws": sorted(set(missing_prize_draws)),
        "year_summary": dict(sorted(year_summary.items())),
        "detail_csv": str(output_csv),
    }

    summary_json = Path(args.summary)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.fail_on_missing_prize and summary["missing_prize_draw_count"]:
        raise SystemExit(f"missing prize amount rows: {summary['missing_prize_draws']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LOTO7 best model on holdout draws with real prize returns.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--holdout-start-draw", type=int, required=True)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--output", default="outputs/holdout_result.csv")
    parser.add_argument("--summary", default="outputs/holdout_summary.json")
    parser.add_argument("--fail-on-missing-prize", action="store_true", help="当せん金額が未取得のholdout回があれば失敗扱いにする")
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.unit_cost <= 0:
        raise SystemExit("--unit-cost must be positive")
    if args.holdout_end_draw is not None and args.holdout_end_draw < args.holdout_start_draw:
        raise SystemExit("--holdout-end-draw must be >= --holdout-start-draw")
    return evaluate_holdout(args)


if __name__ == "__main__":
    raise SystemExit(main())
