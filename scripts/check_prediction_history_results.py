#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check wide LOTO7 prediction history CSV against actual results and write TXT."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws, parse_nums
from holdout_evaluator import load_prize_rows, prize_amount_for_rank

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ", "未抽せん"]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_no_int(text: object) -> Optional[int]:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def fmt_nums(nums: Sequence[int]) -> str:
    return ", ".join(str(n) for n in nums)


def fmt_actual_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in nums)


def fmt_yen(value: int) -> str:
    return f"{value:,}円"


def normalize_date(text: object) -> str:
    raw = str(text or "").strip()
    return raw[:10]


def max_prediction_index(row: Dict[str, str]) -> int:
    max_idx = 0
    for key in row.keys():
        m = re.fullmatch(r"予測(\d+)", str(key))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def summarize_rank_counts(counter: Counter) -> str:
    return " / ".join(f"{rank}:{counter.get(rank, 0)}" for rank in RANK_ORDER if counter.get(rank, 0)) or "なし"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check wide prediction history results and output a TXT report.")
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

    draws = load_draws(args.csv)
    draws_by_date: Dict[str, Draw] = {normalize_date(draw.date): draw for draw in draws}
    draws_by_no: Dict[int, Draw] = {draw.draw_no: draw for draw in draws}
    prize_rows = load_prize_rows(args.csv)

    by_date: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    rank_counts: Counter = Counter()
    history_rows = len(rows)
    prediction_rows = 0
    evaluated_rows = 0
    pending_rows = 0
    winning_rows = 0
    max_main_match = 0
    total_cost = 0
    total_payout = 0

    for row in sorted(rows, key=lambda r: normalize_date(r.get("抽せん日"))):
        draw_date = normalize_date(row.get("抽せん日"))
        draw_no = draw_no_int(row.get("回別"))
        if not draw_date and draw_no is None:
            continue
        draw = draws_by_no.get(draw_no) if draw_no is not None else None
        if draw is None and draw_date:
            draw = draws_by_date.get(draw_date)
        display_date = draw_date or (normalize_date(draw.date) if draw else "")
        display_draw_label = str(row.get("回別") or (f"第{draw.draw_no}回" if draw else "-")).strip()
        limit = max_prediction_index(row)
        for idx in range(1, limit + 1):
            prediction = str(row.get(f"予測{idx}", "")).strip()
            if not prediction:
                continue
            nums = parse_nums(prediction)
            if len(nums) != 7:
                continue
            confidence = str(row.get(f"信頼度{idx}", "")).strip()
            prediction_rows += 1

            if draw is None:
                pending_rows += 1
                rank_counts["未抽せん"] += 1
                by_date[display_date].append({
                    "index": idx,
                    "numbers": fmt_nums(nums),
                    "confidence": confidence,
                    "rank": "未抽せん",
                    "main_match": "-",
                    "bonus_match": "-",
                    "payout": 0,
                    "actual_main": "-",
                    "actual_bonus": "-",
                    "draw_label": display_draw_label or "-",
                })
                continue

            main_match, bonus_match, rank = evaluate_ticket(nums, draw)
            prize_row = prize_rows.get(draw.draw_no, {})
            payout = prize_amount_for_rank(prize_row, rank) if prize_row else 0
            evaluated_rows += 1
            total_cost += args.unit_cost
            total_payout += payout
            max_main_match = max(max_main_match, main_match)
            rank_counts[rank] += 1
            if rank != "外れ":
                winning_rows += 1

            by_date[display_date].append({
                "index": idx,
                "numbers": fmt_nums(nums),
                "confidence": confidence,
                "rank": rank,
                "main_match": main_match,
                "bonus_match": bonus_match,
                "payout": payout,
                "actual_main": fmt_actual_nums(draw.main),
                "actual_bonus": fmt_actual_nums(draw.bonus),
                "draw_label": f"第{draw.draw_no}回",
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
        f"history_draw_rows: {history_rows}",
        f"prediction_rows: {prediction_rows}",
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

    for draw_date in sorted(by_date.keys()):
        items = sorted(by_date[draw_date], key=lambda item: int(item.get("index", 9999)))
        first = items[0]
        lines.extend([
            "",
            f"抽せん日: {draw_date} / {first.get('draw_label')}",
            f"actual_main: {first.get('actual_main')} / actual_bonus: {first.get('actual_bonus')}",
        ])
        for item in items:
            payout = int(item.get("payout", 0) or 0)
            lines.append(
                f"  予測{item.get('index')}: {item.get('numbers')} | "
                f"信頼度={item.get('confidence')} | "
                f"main={item.get('main_match')} bonus={item.get('bonus_match')} | "
                f"result={item.get('rank')} | payout={fmt_yen(payout)}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
