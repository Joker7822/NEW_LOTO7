#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_backtest_results.py

loto7_backtest_summary.csv を読み込み、バックテスト結果の集計を
TXT と CSV に出力する。

出力:
    - loto7_backtest_result.txt
    - loto7_backtest_result.csv

集計内容:
    - 検証回数
    - 口数
    - 購入金額
    - 当せん金額
    - 収支
    - 回収率
    - 当選回数
    - 等級別当選回数
    - 最高等級分布
    - 最高一致数分布
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INPUT_CSV = "loto7_backtest_summary.csv"
DEFAULT_OUTPUT_TXT = "loto7_backtest_result.txt"
DEFAULT_OUTPUT_CSV = "loto7_backtest_result.csv"

GRADE_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "ハズレ"]


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).replace(",", "").replace("円", "").strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).replace("%", "").strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def yen(value: int) -> str:
    return f"{value:,}円"


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def read_detail_rows(input_csv: str) -> List[Dict[str, str]]:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"バックテスト明細CSVが見つかりません: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"バックテスト明細CSVに有効な行がありません: {path}")

    return rows


def ticket_indices(row: Dict[str, str]) -> List[int]:
    indices: List[int] = []
    for key in row.keys():
        if key.startswith("予測") and key.endswith("_等級"):
            middle = key.replace("予測", "").replace("_等級", "")
            if middle.isdigit():
                indices.append(int(middle))
    return sorted(set(indices))


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    first = rows[0]
    indices = ticket_indices(first)

    total_draws = len(rows)
    tickets_per_draw = len(indices)
    total_tickets = total_draws * tickets_per_draw

    total_purchase = sum(to_int(r.get("購入金額")) for r in rows)
    total_prize = sum(to_int(r.get("当せん金額")) for r in rows)
    total_profit = total_prize - total_purchase
    return_rate = total_prize / total_purchase if total_purchase else 0.0

    winning_draws = sum(1 for r in rows if to_int(r.get("当せん口数")) > 0)
    losing_draws = total_draws - winning_draws
    winning_tickets = sum(to_int(r.get("当せん口数")) for r in rows)
    losing_tickets = total_tickets - winning_tickets

    grade_counts: Counter = Counter()
    best_grade_counts: Counter = Counter()
    best_main_match_counts: Counter = Counter()
    best_bonus_match_counts: Counter = Counter()

    max_draw_prize = 0
    max_draw_profit = None
    max_draw_prize_row: Dict[str, str] | None = None
    max_draw_profit_row: Dict[str, str] | None = None

    for row in rows:
        best_grade = row.get("最高等級", "ハズレ") or "ハズレ"
        best_grade_counts[best_grade] += 1
        best_main_match_counts[str(to_int(row.get("最高本数字一致数")))] += 1
        best_bonus_match_counts[str(to_int(row.get("最高ボーナス一致数")))] += 1

        draw_prize = to_int(row.get("当せん金額"))
        draw_profit = to_int(row.get("収支"))
        if draw_prize > max_draw_prize:
            max_draw_prize = draw_prize
            max_draw_prize_row = row
        if max_draw_profit is None or draw_profit > max_draw_profit:
            max_draw_profit = draw_profit
            max_draw_profit_row = row

        for idx in indices:
            grade = row.get(f"予測{idx}_等級", "ハズレ") or "ハズレ"
            grade_counts[grade] += 1

    prize_rows = [r for r in rows if to_int(r.get("当せん金額")) > 0]
    positive_profit_draws = sum(1 for r in rows if to_int(r.get("収支")) > 0)
    break_even_draws = sum(1 for r in rows if to_int(r.get("収支")) == 0)
    negative_profit_draws = sum(1 for r in rows if to_int(r.get("収支")) < 0)

    start_date = rows[0].get("抽せん日", "")
    end_date = rows[-1].get("抽せん日", "")
    start_draw_no = rows[0].get("回別", "")
    end_draw_no = rows[-1].get("回別", "")

    summary: Dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
        "start_draw_no": start_draw_no,
        "end_draw_no": end_draw_no,
        "total_draws": total_draws,
        "tickets_per_draw": tickets_per_draw,
        "total_tickets": total_tickets,
        "total_purchase": total_purchase,
        "total_prize": total_prize,
        "total_profit": total_profit,
        "return_rate": return_rate,
        "winning_draws": winning_draws,
        "losing_draws": losing_draws,
        "winning_draw_rate": winning_draws / total_draws if total_draws else 0.0,
        "winning_tickets": winning_tickets,
        "losing_tickets": losing_tickets,
        "winning_ticket_rate": winning_tickets / total_tickets if total_tickets else 0.0,
        "positive_profit_draws": positive_profit_draws,
        "break_even_draws": break_even_draws,
        "negative_profit_draws": negative_profit_draws,
        "positive_profit_rate": positive_profit_draws / total_draws if total_draws else 0.0,
        "grade_counts": grade_counts,
        "best_grade_counts": best_grade_counts,
        "best_main_match_counts": best_main_match_counts,
        "best_bonus_match_counts": best_bonus_match_counts,
        "max_draw_prize": max_draw_prize,
        "max_draw_prize_date": max_draw_prize_row.get("抽せん日", "") if max_draw_prize_row else "",
        "max_draw_prize_grade": max_draw_prize_row.get("最高等級", "") if max_draw_prize_row else "",
        "max_draw_profit": max_draw_profit if max_draw_profit is not None else 0,
        "max_draw_profit_date": max_draw_profit_row.get("抽せん日", "") if max_draw_profit_row else "",
        "max_draw_profit_grade": max_draw_profit_row.get("最高等級", "") if max_draw_profit_row else "",
        "prize_draw_count": len(prize_rows),
    }
    return summary


def ordered_grade_lines(counter: Counter) -> List[Tuple[str, int]]:
    keys = list(GRADE_ORDER)
    for key in sorted(counter.keys()):
        if key not in keys:
            keys.append(key)
    return [(key, int(counter.get(key, 0))) for key in keys]


def build_text(summary: Dict[str, object]) -> str:
    grade_counts: Counter = summary["grade_counts"]  # type: ignore[assignment]
    best_grade_counts: Counter = summary["best_grade_counts"]  # type: ignore[assignment]
    best_main_match_counts: Counter = summary["best_main_match_counts"]  # type: ignore[assignment]

    lines: List[str] = []
    lines.append("Loto7 バックテスト結果")
    lines.append("=" * 24)
    lines.append(f"検証期間: {summary['start_date']} ～ {summary['end_date']}")
    lines.append(f"検証回別: {summary['start_draw_no']} ～ {summary['end_draw_no']}")
    lines.append(f"検証回数: {summary['total_draws']}回")
    lines.append(f"1回あたり口数: {summary['tickets_per_draw']}口")
    lines.append(f"総購入口数: {summary['total_tickets']}口")
    lines.append("")

    lines.append("収支")
    lines.append("-" * 24)
    lines.append(f"購入金額: {yen(int(summary['total_purchase']))}")
    lines.append(f"当せん金額: {yen(int(summary['total_prize']))}")
    lines.append(f"収支: {yen(int(summary['total_profit']))}")
    lines.append(f"回収率: {percent(float(summary['return_rate']))}")
    lines.append("")

    lines.append("当選回数")
    lines.append("-" * 24)
    lines.append(f"当選した抽せん回数: {summary['winning_draws']}回")
    lines.append(f"ハズレのみの抽せん回数: {summary['losing_draws']}回")
    lines.append(f"抽せん回ベース当選率: {percent(float(summary['winning_draw_rate']))}")
    lines.append(f"当選口数: {summary['winning_tickets']}口")
    lines.append(f"ハズレ口数: {summary['losing_tickets']}口")
    lines.append(f"口数ベース当選率: {percent(float(summary['winning_ticket_rate']))}")
    lines.append("")

    lines.append("黒字・赤字回数")
    lines.append("-" * 24)
    lines.append(f"黒字回数: {summary['positive_profit_draws']}回")
    lines.append(f"収支ゼロ回数: {summary['break_even_draws']}回")
    lines.append(f"赤字回数: {summary['negative_profit_draws']}回")
    lines.append(f"黒字率: {percent(float(summary['positive_profit_rate']))}")
    lines.append("")

    lines.append("等級別当選口数")
    lines.append("-" * 24)
    for grade, count in ordered_grade_lines(grade_counts):
        if grade == "ハズレ":
            continue
        lines.append(f"{grade}: {count}口")
    lines.append("")

    lines.append("各回ベスト等級分布")
    lines.append("-" * 24)
    for grade, count in ordered_grade_lines(best_grade_counts):
        lines.append(f"{grade}: {count}回")
    lines.append("")

    lines.append("各回最高本数字一致数")
    lines.append("-" * 24)
    for key in sorted(best_main_match_counts.keys(), key=lambda x: int(x)):
        lines.append(f"{key}個一致: {best_main_match_counts[key]}回")
    lines.append("")

    lines.append("最大結果")
    lines.append("-" * 24)
    lines.append(
        f"最大当せん金額回: {summary['max_draw_prize_date']} / {summary['max_draw_prize_grade']} / {yen(int(summary['max_draw_prize']))}"
    )
    lines.append(
        f"最大収支回: {summary['max_draw_profit_date']} / {summary['max_draw_profit_grade']} / {yen(int(summary['max_draw_profit']))}"
    )
    lines.append("")

    lines.append("注意")
    lines.append("-" * 24)
    lines.append("当せん金額は、コード内の設定金額に基づく概算です。")
    lines.append("正確な収支には、各回の実際の等級別当せん金額データが必要です。")
    lines.append("的中保証は確認できません。")
    return "\n".join(lines) + "\n"


def flat_summary_rows(summary: Dict[str, object]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def add(category: str, metric: str, value: object) -> None:
        rows.append({"カテゴリ": category, "項目": metric, "値": str(value)})

    add("期間", "開始日", summary["start_date"])
    add("期間", "終了日", summary["end_date"])
    add("期間", "開始回", summary["start_draw_no"])
    add("期間", "終了回", summary["end_draw_no"])
    add("基本", "検証回数", summary["total_draws"])
    add("基本", "1回あたり口数", summary["tickets_per_draw"])
    add("基本", "総購入口数", summary["total_tickets"])
    add("収支", "購入金額", summary["total_purchase"])
    add("収支", "当せん金額", summary["total_prize"])
    add("収支", "収支", summary["total_profit"])
    add("収支", "回収率", f"{float(summary['return_rate']) * 100:.6f}%")
    add("当選", "当選した抽せん回数", summary["winning_draws"])
    add("当選", "ハズレのみの抽せん回数", summary["losing_draws"])
    add("当選", "抽せん回ベース当選率", f"{float(summary['winning_draw_rate']) * 100:.6f}%")
    add("当選", "当選口数", summary["winning_tickets"])
    add("当選", "ハズレ口数", summary["losing_tickets"])
    add("当選", "口数ベース当選率", f"{float(summary['winning_ticket_rate']) * 100:.6f}%")
    add("収支回数", "黒字回数", summary["positive_profit_draws"])
    add("収支回数", "収支ゼロ回数", summary["break_even_draws"])
    add("収支回数", "赤字回数", summary["negative_profit_draws"])
    add("収支回数", "黒字率", f"{float(summary['positive_profit_rate']) * 100:.6f}%")

    grade_counts: Counter = summary["grade_counts"]  # type: ignore[assignment]
    for grade, count in ordered_grade_lines(grade_counts):
        add("等級別当選口数", grade, count)

    best_grade_counts: Counter = summary["best_grade_counts"]  # type: ignore[assignment]
    for grade, count in ordered_grade_lines(best_grade_counts):
        add("各回ベスト等級分布", grade, count)

    best_main_match_counts: Counter = summary["best_main_match_counts"]  # type: ignore[assignment]
    for key in sorted(best_main_match_counts.keys(), key=lambda x: int(x)):
        add("各回最高本数字一致数", f"{key}個一致", best_main_match_counts[key])

    add("最大結果", "最大当せん金額", summary["max_draw_prize"])
    add("最大結果", "最大当せん金額日", summary["max_draw_prize_date"])
    add("最大結果", "最大当せん金額等級", summary["max_draw_prize_grade"])
    add("最大結果", "最大収支", summary["max_draw_profit"])
    add("最大結果", "最大収支日", summary["max_draw_profit_date"])
    add("最大結果", "最大収支等級", summary["max_draw_profit_grade"])
    return rows


def write_summary_csv(summary: Dict[str, object], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = flat_summary_rows(summary)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["カテゴリ", "項目", "値"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(summary: Dict[str, object], output_txt: str) -> None:
    path = Path(output_txt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_text(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Loto7バックテスト明細CSVから当選回数・収支などを集計出力します。")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="入力バックテスト明細CSV")
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT, help="出力TXT")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="出力CSV")
    args = parser.parse_args()

    rows = read_detail_rows(args.input_csv)
    summary = summarize(rows)
    write_summary_txt(summary, args.output_txt)
    write_summary_csv(summary, args.output_csv)

    print(f"保存完了: {args.output_txt}")
    print(f"保存完了: {args.output_csv}")
    print(build_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
