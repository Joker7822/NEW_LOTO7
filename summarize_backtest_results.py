#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_backtest_results.py

loto7_backtest_summary.csv を読み込み、loto7_backtest_result.txt / csv を出力する。
現在の NEW_LOTO7 では、loto7_backtest_summary.csv は集計済み1行形式で出力されるため、
その形式を正しく読み取って集計結果を生成する。
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_INPUT_CSV = "loto7_backtest_summary.csv"
DEFAULT_OUTPUT_TXT = "loto7_backtest_result.txt"
DEFAULT_OUTPUT_CSV = "loto7_backtest_result.csv"
GRADE_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "ハズレ"]


def to_int(value: object, default: int = 0) -> int:
    try:
        text = str(value or "").replace(",", "").replace("円", "").strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def to_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value or "").replace("%", "").strip()
        return float(text) if text else default
    except Exception:
        return default


def parse_counter(value: object) -> Counter:
    text = str(value or "").strip()
    if not text:
        return Counter()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return Counter({str(k): int(v) for k, v in parsed.items()})
    except Exception:
        pass
    return Counter()


def yen(value: int) -> str:
    return f"{value:,}円"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def read_latest_summary(input_csv: str) -> Dict[str, str]:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"入力CSVに有効な行がありません: {path}")
    return rows[-1]


def ordered(counter: Counter) -> List[Tuple[str, int]]:
    keys = list(GRADE_ORDER)
    for k in sorted(counter.keys()):
        if k not in keys:
            keys.append(k)
    return [(k, int(counter.get(k, 0))) for k in keys]


def build_summary(row: Dict[str, str]) -> Dict[str, object]:
    trials = to_int(row.get("検証回数"))
    tickets_per_draw = to_int(row.get("1回あたり口数"))
    total_tickets = trials * tickets_per_draw
    total_purchase = to_int(row.get("総購入金額"))
    total_prize = to_int(row.get("総当せん金額"))
    total_profit = to_int(row.get("総収支"), total_prize - total_purchase)
    return_rate = to_float(row.get("総回収率"))
    if return_rate == 0 and total_purchase:
        return_rate = total_prize / total_purchase

    grade_counts = parse_counter(row.get("全予測口等級分布"))
    best_grade_counts = parse_counter(row.get("各回ベスト等級分布"))
    top1_dist = parse_counter(row.get("1口目一致数分布"))
    best_dist = parse_counter(row.get("全口ベスト一致数分布"))

    winning_tickets = sum(v for k, v in grade_counts.items() if k != "ハズレ")
    losing_tickets = int(grade_counts.get("ハズレ", max(total_tickets - winning_tickets, 0)))
    winning_draws = sum(v for k, v in best_grade_counts.items() if k != "ハズレ")
    losing_draws = int(best_grade_counts.get("ハズレ", max(trials - winning_draws, 0)))

    return {
        "trials": trials,
        "min_train": to_int(row.get("初期学習回数")),
        "start_draw_index": to_int(row.get("検証開始回相当")),
        "requested_pool": to_int(row.get("要求バックテスト候補プール")),
        "effective_pool": to_int(row.get("実効バックテスト候補プール")),
        "tickets_per_draw": tickets_per_draw,
        "total_tickets": total_tickets,
        "total_purchase": total_purchase,
        "total_prize": total_prize,
        "total_profit": total_profit,
        "return_rate": return_rate,
        "winning_tickets": winning_tickets,
        "losing_tickets": losing_tickets,
        "winning_ticket_rate": winning_tickets / total_tickets if total_tickets else 0.0,
        "winning_draws": winning_draws,
        "losing_draws": losing_draws,
        "winning_draw_rate": winning_draws / trials if trials else 0.0,
        "top1_avg": to_float(row.get("1口目平均一致数")),
        "best_avg": to_float(row.get("全口ベスト平均一致数")),
        "top1_ge2": to_float(row.get("1口目_2個以上率")),
        "top1_ge3": to_float(row.get("1口目_3個以上率")),
        "top1_ge4": to_float(row.get("1口目_4個以上率")),
        "best_ge2": to_float(row.get("全口ベスト_2個以上率")),
        "best_ge3": to_float(row.get("全口ベスト_3個以上率")),
        "best_ge4": to_float(row.get("全口ベスト_4個以上率")),
        "grade_counts": grade_counts,
        "best_grade_counts": best_grade_counts,
        "top1_dist": top1_dist,
        "best_dist": best_dist,
    }


def build_text(s: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("Loto7 バックテスト結果")
    lines.append("=" * 28)
    lines.append(f"検証範囲: 第{s['start_draw_index']}回相当から最新まで")
    lines.append(f"初期学習回数: {s['min_train']}回")
    lines.append(f"検証回数: {s['trials']}回")
    lines.append(f"1回あたり口数: {s['tickets_per_draw']}口")
    lines.append(f"総購入口数: {s['total_tickets']}口")
    lines.append(f"バックテスト候補プール: {s['effective_pool']} / 要求 {s['requested_pool']}")
    lines.append("")
    lines.append("収支")
    lines.append("-" * 28)
    lines.append(f"購入金額: {yen(int(s['total_purchase']))}")
    lines.append(f"当せん金額: {yen(int(s['total_prize']))}")
    lines.append(f"収支: {yen(int(s['total_profit']))}")
    lines.append(f"回収率: {pct(float(s['return_rate']))}")
    lines.append("")
    lines.append("当選回数")
    lines.append("-" * 28)
    lines.append(f"当選した抽せん回数: {s['winning_draws']}回")
    lines.append(f"ハズレのみの抽せん回数: {s['losing_draws']}回")
    lines.append(f"抽せん回ベース当選率: {pct(float(s['winning_draw_rate']))}")
    lines.append(f"当選口数: {s['winning_tickets']}口")
    lines.append(f"ハズレ口数: {s['losing_tickets']}口")
    lines.append(f"口数ベース当選率: {pct(float(s['winning_ticket_rate']))}")
    lines.append("")
    lines.append("一致数")
    lines.append("-" * 28)
    lines.append(f"1口目平均一致数: {float(s['top1_avg']):.6f}")
    lines.append(f"全口ベスト平均一致数: {float(s['best_avg']):.6f}")
    lines.append(f"1口目 2個以上率: {pct(float(s['top1_ge2']))}")
    lines.append(f"1口目 3個以上率: {pct(float(s['top1_ge3']))}")
    lines.append(f"1口目 4個以上率: {pct(float(s['top1_ge4']))}")
    lines.append(f"全口ベスト 2個以上率: {pct(float(s['best_ge2']))}")
    lines.append(f"全口ベスト 3個以上率: {pct(float(s['best_ge3']))}")
    lines.append(f"全口ベスト 4個以上率: {pct(float(s['best_ge4']))}")
    lines.append("")
    lines.append("等級別当選口数")
    lines.append("-" * 28)
    for grade, count in ordered(s['grade_counts']):  # type: ignore[arg-type]
        if grade != "ハズレ":
            lines.append(f"{grade}: {count}口")
    lines.append("")
    lines.append("各回ベスト等級分布")
    lines.append("-" * 28)
    for grade, count in ordered(s['best_grade_counts']):  # type: ignore[arg-type]
        lines.append(f"{grade}: {count}回")
    lines.append("")
    lines.append("1口目一致数分布")
    lines.append("-" * 28)
    for k in sorted(s['top1_dist'].keys(), key=lambda x: int(x)):  # type: ignore[union-attr]
        lines.append(f"{k}個一致: {s['top1_dist'][k]}回")  # type: ignore[index]
    lines.append("")
    lines.append("全口ベスト一致数分布")
    lines.append("-" * 28)
    for k in sorted(s['best_dist'].keys(), key=lambda x: int(x)):  # type: ignore[union-attr]
        lines.append(f"{k}個一致: {s['best_dist'][k]}回")  # type: ignore[index]
    lines.append("")
    lines.append("注意")
    lines.append("-" * 28)
    lines.append("当せん金額は、コード内の設定金額に基づく概算です。")
    lines.append("正確な収支には、各回の実際の等級別当せん金額データが必要です。")
    lines.append("的中保証は確認できません。")
    return "\n".join(lines) + "\n"


def build_csv_rows(s: Dict[str, object]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def add(cat: str, item: str, value: object) -> None:
        rows.append({"カテゴリ": cat, "項目": item, "値": str(value)})

    add("基本", "検証開始回相当", s["start_draw_index"])
    add("基本", "初期学習回数", s["min_train"])
    add("基本", "検証回数", s["trials"])
    add("基本", "1回あたり口数", s["tickets_per_draw"])
    add("基本", "総購入口数", s["total_tickets"])
    add("基本", "実効バックテスト候補プール", s["effective_pool"])
    add("収支", "購入金額", s["total_purchase"])
    add("収支", "当せん金額", s["total_prize"])
    add("収支", "収支", s["total_profit"])
    add("収支", "回収率", f"{float(s['return_rate']) * 100:.6f}%")
    add("当選", "当選した抽せん回数", s["winning_draws"])
    add("当選", "ハズレのみの抽せん回数", s["losing_draws"])
    add("当選", "抽せん回ベース当選率", f"{float(s['winning_draw_rate']) * 100:.6f}%")
    add("当選", "当選口数", s["winning_tickets"])
    add("当選", "ハズレ口数", s["losing_tickets"])
    add("当選", "口数ベース当選率", f"{float(s['winning_ticket_rate']) * 100:.6f}%")
    add("一致数", "1口目平均一致数", f"{float(s['top1_avg']):.6f}")
    add("一致数", "全口ベスト平均一致数", f"{float(s['best_avg']):.6f}")
    add("一致率", "1口目_2個以上率", f"{float(s['top1_ge2']) * 100:.6f}%")
    add("一致率", "1口目_3個以上率", f"{float(s['top1_ge3']) * 100:.6f}%")
    add("一致率", "1口目_4個以上率", f"{float(s['top1_ge4']) * 100:.6f}%")
    add("一致率", "全口ベスト_2個以上率", f"{float(s['best_ge2']) * 100:.6f}%")
    add("一致率", "全口ベスト_3個以上率", f"{float(s['best_ge3']) * 100:.6f}%")
    add("一致率", "全口ベスト_4個以上率", f"{float(s['best_ge4']) * 100:.6f}%")

    for grade, count in ordered(s['grade_counts']):  # type: ignore[arg-type]
        add("等級別当選口数", grade, count)
    for grade, count in ordered(s['best_grade_counts']):  # type: ignore[arg-type]
        add("各回ベスト等級分布", grade, count)
    for k in sorted(s['top1_dist'].keys(), key=lambda x: int(x)):  # type: ignore[union-attr]
        add("1口目一致数分布", f"{k}個一致", s['top1_dist'][k])  # type: ignore[index]
    for k in sorted(s['best_dist'].keys(), key=lambda x: int(x)):  # type: ignore[union-attr]
        add("全口ベスト一致数分布", f"{k}個一致", s['best_dist'][k])  # type: ignore[index]
    return rows


def write_txt(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["カテゴリ", "項目", "値"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Loto7バックテスト結果をTXT/CSVへ変換します。")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    row = read_latest_summary(args.input_csv)
    summary = build_summary(row)
    text = build_text(summary)
    write_txt(args.output_txt, text)
    write_csv(args.output_csv, build_csv_rows(summary))
    print(text)
    print(f"保存完了: {args.output_txt}")
    print(f"保存完了: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
