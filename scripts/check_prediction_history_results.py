#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check cumulative LOTO7 prediction history against actual results and write TXT."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws, parse_nums
from holdout_evaluator import load_prize_rows, prize_amount_for_rank

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ", "未抽せん"]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def draw_no_int(text: object) -> Optional[int]:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in nums)


def fmt_yen(value: int) -> str:
    return f"{value:,}円"


def prediction_draw_no(row: Dict[str, str]) -> Optional[int]:
    for field in ("prediction_draw_no", "prediction_key", "draw_no"):
        no = draw_no_int(row.get(field))
        if no is not None:
            return no
    return None


def combo_sort_key(row: Dict[str, str]) -> tuple:
    draw_no = prediction_draw_no(row) or 10**9
    rank = draw_no_int(row.get("confidence_rank")) or draw_no_int(row.get("combo_index")) or 9999
    return (draw_no, rank)


def summarize_rank_counts(counter: Counter) -> str:
    return " / ".join(f"{rank}:{counter.get(rank, 0)}" for rank in RANK_ORDER if counter.get(rank, 0)) or "なし"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check prediction history results and output a TXT report.")
    parser.add_argument("--history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--output", default="outputs/evolution_prediction_history_result.txt")
    parser.add_argument("--unit-cost", type=int, default=300)
    args = parser.parse_args()

    history_path = Path(args.history)
    output_path = Path(args.output)
    rows = read_rows(history_path)
    if not rows:
        raise SystemExit(f"prediction history CSV is empty or missing: {history_path}")

    draws = {draw.draw_no: draw for draw in load_draws(args.csv)}
    prize_rows = load_prize_rows(args.csv)

    detail_lines: List[str] = []
    by_draw: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    rank_counts: Counter = Counter()
    evaluated_rows = 0
    pending_rows = 0
    winning_rows = 0
    max_main_match = 0
    total_cost = 0
    total_payout = 0

    for row in sorted(rows, key=combo_sort_key):
        no = prediction_draw_no(row)
        nums = parse_nums(row.get("numbers"))
        if no is None or len(nums) != 7:
            continue

        confidence = row.get("confidence_rank") or row.get("combo_index") or ""
        created_at = row.get("created_at", "")
        model_id = row.get("model_id", "")
        base_latest_date = row.get("base_latest_date", "")
        draw = draws.get(no)

        if draw is None:
            pending_rows += 1
            rank_counts["未抽せん"] += 1
            by_draw[no].append({
                "confidence": confidence,
                "numbers": fmt_nums(nums),
                "rank": "未抽せん",
                "main_match": "-",
                "bonus_match": "-",
                "payout": 0,
                "created_at": created_at,
                "model_id": model_id,
                "base_latest_date": base_latest_date,
                "actual_main": "-",
                "actual_bonus": "-",
                "actual_date": "-",
            })
            continue

        main_match, bonus_match, rank = evaluate_ticket(nums, draw)
        prize_row = prize_rows.get(no, {})
        payout = prize_amount_for_rank(prize_row, rank) if prize_row else 0
        evaluated_rows += 1
        total_cost += args.unit_cost
        total_payout += payout
        max_main_match = max(max_main_match, main_match)
        rank_counts[rank] += 1
        if rank != "外れ":
            winning_rows += 1

        by_draw[no].append({
            "confidence": confidence,
            "numbers": fmt_nums(nums),
            "rank": rank,
            "main_match": main_match,
            "bonus_match": bonus_match,
            "payout": payout,
            "created_at": created_at,
            "model_id": model_id,
            "base_latest_date": base_latest_date,
            "actual_main": fmt_nums(draw.main),
            "actual_bonus": fmt_nums(draw.bonus),
            "actual_date": draw.date,
        })

    total_profit = total_payout - total_cost
    hit_rate = (winning_rows / evaluated_rows * 100.0) if evaluated_rows else 0.0
    payout_roi = (total_payout / total_cost * 100.0) if total_cost else 0.0
    profit_roi = (total_profit / total_cost * 100.0) if total_cost else 0.0

    lines = [
        "LOTO7 Prediction History Result Check",
        "====================================",
        "",
        f"created_at: {utc_now()}",
        f"history_csv: {history_path}",
        f"loto7_csv: {args.csv}",
        "",
        "[Summary]",
        f"history_rows: {len(rows)}",
        f"evaluated_rows: {evaluated_rows}",
        f"pending_rows: {pending_rows}",
        f"winning_rows: {winning_rows}",
        f"hit_rate: {hit_rate:.3f}%",
        f"max_main_match: {max_main_match}",
        f"total_cost: {fmt_yen(total_cost)}",
        f"total_payout: {fmt_yen(total_payout)}",
        f"profit: {fmt_yen(total_profit)}",
        f"payout_roi: {payout_roi:.3f}%",
        f"profit_roi: {profit_roi:.3f}%",
        f"rank_counts: {summarize_rank_counts(rank_counts)}",
        "",
        "[Details]",
    ]

    for no in sorted(by_draw.keys()):
        items = sorted(by_draw[no], key=lambda item: int(str(item.get("confidence") or "9999")) if str(item.get("confidence") or "").isdigit() else 9999)
        first = items[0]
        lines.extend([
            "",
            f"第{no}回 / actual_date: {first.get('actual_date')} / base_latest_date: {first.get('base_latest_date')}",
            f"actual_main: {first.get('actual_main')} / actual_bonus: {first.get('actual_bonus')}",
        ])
        for item in items:
            payout = int(item.get("payout", 0) or 0)
            lines.append(
                f"  rank#{item.get('confidence')}: {item.get('numbers')} | "
                f"main={item.get('main_match')} bonus={item.get('bonus_match')} | "
                f"result={item.get('rank')} | payout={fmt_yen(payout)} | "
                f"model={item.get('model_id')} | predicted_at={item.get('created_at')}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
