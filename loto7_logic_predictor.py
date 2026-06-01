#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_logic_predictor.py

NEW_LOTO7 用 Loto7 予測・バックテスト統合エントリポイント。

既存の実行ファイル名とCLIを維持したまま、Advanced Optimizerを直接利用します。

接続済みAdvanced機能:
    - 等級特化学習: 6本一致・5本一致・4本一致を強く評価
    - Walk Forward完全検証: 検証対象回より前のデータだけで予測
    - Optuna自動最適化: 未導入環境ではRandom Searchへ自動フォールバック
    - MonteCarlo組合せ探索: 固定候補プール外も探索
    - 的中履歴MemoryBank: 4本一致以上の構造を蓄積・再利用

注意:
    ロト7は独立抽せんのため、的中保証は確認できません。
    スコア正規化値は候補順位スコアであり、当選確率ではありません。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_CSV_URL = "loto7.csv"

DEFAULT_OUTPUT_CSV = "loto7_predictions.csv"
DEFAULT_LATEST_TXT = "latest_loto7_prediction.txt"

DEFAULT_BACKTEST_SUMMARY_CSV = "loto7_backtest_summary.csv"
DEFAULT_BACKTEST_DETAIL_CSV = "loto7_backtest_detail.csv"
DEFAULT_BACKTEST_REPORT_TXT = "loto7_backtest_report.txt"

NUM_MIN = 1
NUM_MAX = 37
PICK_SIZE = 7

DEFAULT_TICKETS = 10
DEFAULT_SAVE_COUNT = 10
DEFAULT_UNIT_COST = 300
DEFAULT_BACKTEST_POOL_CAP = 16

DEFAULT_PRIZE_TABLE = {
    1: 700_000_000,
    2: 7_300_000,
    3: 730_000,
    4: 9_100,
    5: 1_400,
    6: 1_000,
}


@dataclass(frozen=True)
class Draw:
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]
    draw_no: Optional[int] = None


@dataclass
class TicketScore:
    ticket: Tuple[int, ...]
    score: float
    detail: Dict[str, float]
    strategy: str = "CORE"


@dataclass(frozen=True)
class PrizeResult:
    main_matches: int
    bonus_matches: int
    grade: Optional[int]
    prize: int


def parse_numbers(value: object) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(value or "")))


def parse_draw_no(value: object) -> Optional[int]:
    nums = re.findall(r"\d+", str(value or ""))
    return int(nums[0]) if nums else None


def validate_main_numbers(nums: Sequence[int]) -> Tuple[int, ...]:
    if len(nums) != PICK_SIZE:
        raise ValueError(f"本数字は7個必要です: {nums}")
    if len(set(nums)) != PICK_SIZE:
        raise ValueError(f"本数字に重複があります: {nums}")
    for n in nums:
        if not (NUM_MIN <= n <= NUM_MAX):
            raise ValueError(f"数字が範囲外です: {n}")
    return tuple(sorted(nums))


def read_text_from_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(
            source,
            headers={
                "User-Agent": "Mozilla/5.0 loto7-logic-predictor",
                "Accept": "text/csv,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.read().decode("utf-8-sig")

    with open(source, "r", encoding="utf-8-sig", newline="") as f:
        return f.read()


def load_draws(source: str = DEFAULT_CSV_URL) -> List[Draw]:
    text = read_text_from_source(source)
    reader = csv.reader(io.StringIO(text))
    draws: List[Draw] = []

    for row in reader:
        if not row or len(row) < 3:
            continue
        if "抽せん" in row[0] or "date" in row[0].lower():
            continue

        date = row[0].strip()
        main_raw = parse_numbers(row[1])
        bonus = tuple(sorted(n for n in parse_numbers(row[2]) if NUM_MIN <= n <= NUM_MAX))
        draw_no = parse_draw_no(row[3]) if len(row) >= 4 else None

        try:
            main = validate_main_numbers(main_raw)
        except ValueError:
            continue

        draws.append(Draw(date=date, main=main, bonus=bonus, draw_no=draw_no))

    draws.sort(key=lambda d: d.date)
    return draws


def format_ticket(ticket: Sequence[int], zero_pad: bool = True) -> str:
    if zero_pad:
        return ", ".join(f"{n:02d}" for n in sorted(ticket))
    return ", ".join(str(n) for n in sorted(ticket))


def score_normalized_values(ranked: Sequence[TicketScore]) -> List[float]:
    if not ranked:
        return []
    scores = [x.score for x in ranked]
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.000 for _ in ranked]
    return [round((s - min_score) / (max_score - min_score), 3) for s in scores]


def next_friday_after(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (dt + timedelta(days=7)).isoformat()


def classify_loto7_prize(
    ticket: Sequence[int],
    actual_main: Sequence[int],
    actual_bonus: Sequence[int],
    prize_table: Dict[int, int],
) -> PrizeResult:
    ticket_set = set(ticket)
    main_matches = len(ticket_set & set(actual_main))
    bonus_matches = len(ticket_set & set(actual_bonus))

    if main_matches == 7:
        grade = 1
    elif main_matches == 6 and bonus_matches >= 1:
        grade = 2
    elif main_matches == 6:
        grade = 3
    elif main_matches == 5:
        grade = 4
    elif main_matches == 4:
        grade = 5
    elif main_matches == 3 and bonus_matches >= 1:
        grade = 6
    else:
        grade = None

    prize = prize_table.get(grade, 0) if grade is not None else 0
    return PrizeResult(main_matches=main_matches, bonus_matches=bonus_matches, grade=grade, prize=prize)


def yen(value: int) -> str:
    return f"{value:,}円"


def count_numbers(draws: Sequence[Draw], window: int) -> Counter:
    c: Counter = Counter()
    for d in draws[-window:] if window > 0 else draws:
        c.update(d.main)
    return c


def count_combinations(draws: Sequence[Draw], window: int, k: int) -> Counter:
    import itertools

    c: Counter = Counter()
    for d in draws[-window:] if window > 0 else draws:
        for comb in itertools.combinations(sorted(d.main), k):
            c[comb] += 1
    return c


def print_recent_summary(draws: Sequence[Draw]) -> None:
    latest = draws[-1]
    print("=== 最新データ ===")
    print(f"抽せん日: {latest.date}")
    print(f"回別: {latest.draw_no if latest.draw_no is not None else '不明'}")
    print(f"本数字: {format_ticket(latest.main)}")
    print(f"ボーナス: {format_ticket(latest.bonus)}")
    print(f"次回想定日: {next_friday_after(latest.date)}")
    print()

    print("=== 直近10回の出現回数 ===")
    c10 = count_numbers(draws, 10)
    for n, cnt in sorted(c10.items(), key=lambda x: (-x[1], x[0])):
        print(f"{n:02d}: {cnt}回")
    print()

    print("=== 直近20回の強ペア TOP15 ===")
    p20 = count_combinations(draws, 20, 2)
    for pair, cnt in p20.most_common(15):
        print(f"{pair[0]:02d}-{pair[1]:02d}: {cnt}回")
    print()


def print_predictions(tickets: Sequence[TicketScore]) -> None:
    print("=== Advanced Optimizer 次回予測 ===")
    scores = score_normalized_values(tickets)
    for i, t in enumerate(tickets, start=1):
        d = t.detail
        score = scores[i - 1] if i - 1 < len(scores) else 0.0
        print(
            f"{i:02d}. {format_ticket(t.ticket)}"
            f" | スコア正規化値={score:.3f}"
            f" | strategy={t.strategy}"
            f" | raw_score={t.score:.3f}"
            f" | sum={int(d.get('sum', 0))}"
            f" | odd={int(d.get('odd', 0))}"
            f" | low/mid/high={int(d.get('low', 0))}/{int(d.get('mid', 0))}/{int(d.get('high', 0))}"
            f" | repeat_last={int(d.get('repeat_last', 0))}"
            f" | pair={d.get('pair', 0):.3f}"
            f" | triple={d.get('triple', 0):.3f}"
            f" | memory={d.get('memory', d.get('pattern', 0)):.3f}"
            f" | grade6={d.get('grade6', 0):.3f}"
            f" | cycle={d.get('cycle', 0):.3f}"
            f" | meta6={d.get('meta6', 0):.3f}"
            f" | shap={d.get('shap', 0):.3f}"
        )
    print()


def prediction_csv_header(save_count: int = DEFAULT_SAVE_COUNT) -> List[str]:
    header = ["抽せん日"]
    for i in range(1, save_count + 1):
        header.extend([f"予測{i}", f"スコア正規化値{i}", f"戦略{i}"])
    return header


def save_predictions_csv(output_path: str, target_date: str, ranked: Sequence[TicketScore], save_count: int) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = prediction_csv_header(save_count)

    scores = score_normalized_values(ranked[:save_count])
    new_row = {key: "" for key in header}
    new_row["抽せん日"] = target_date
    for i, ticket in enumerate(ranked[:save_count], start=1):
        new_row[f"予測{i}"] = format_ticket(ticket.ticket, zero_pad=False)
        new_row[f"スコア正規化値{i}"] = f"{scores[i - 1]:.3f}".rstrip("0").rstrip(".")
        new_row[f"戦略{i}"] = ticket.strategy

    rows: List[Dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for old in csv.DictReader(f):
                rows.append({key: old.get(key, "") for key in header})

    rows = [row for row in rows if row.get("抽せん日") != target_date]
    rows.append(new_row)
    rows.sort(key=lambda r: r.get("抽せん日", ""))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_compat_report(summary: Dict[str, object], report_path: str) -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Loto7 Advanced Optimizer バックテストレポート",
        "===========================================",
        "",
        "基本条件",
        "----------------------------",
        f"検証回数: {summary.get('検証回数', '')}",
        f"初期学習回数: {summary.get('初期学習回数', '')}",
        f"検証開始回相当: {summary.get('検証開始回相当', '')}",
        f"1回あたり口数: {summary.get('1回あたり口数', '')}",
        f"候補プール: {summary.get('実効バックテスト候補プール', summary.get('候補プール', ''))}",
        "",
        "一致数",
        "----------------------------",
        f"1口目 平均一致数: {summary.get('1口目平均一致数', '')}",
        f"全口ベスト 平均一致数: {summary.get('全口ベスト平均一致数', '')}",
        f"1口目 3個以上率: {summary.get('1口目_3個以上率', '')}",
        f"1口目 4個以上率: {summary.get('1口目_4個以上率', '')}",
        f"全口ベスト 3個以上率: {summary.get('全口ベスト_3個以上率', '')}",
        f"全口ベスト 4個以上率: {summary.get('全口ベスト_4個以上率', '')}",
        "",
        "等級・収支",
        "----------------------------",
        f"総購入金額: {summary.get('総購入金額', '')}",
        f"総当せん金額: {summary.get('総当せん金額', '')}",
        f"総収支: {summary.get('総収支', '')}",
        f"総回収率: {summary.get('総回収率', '')}",
        f"全予測口等級分布: {summary.get('全予測口等級分布', '')}",
        f"各回ベスト等級分布: {summary.get('各回ベスト等級分布', '')}",
        "",
        "Advanced情報",
        "----------------------------",
        f"最適化重み: {summary.get('最適化重み', '')}",
        f"MemoryBank件数: {summary.get('MemoryBank件数', '')}",
        "",
        "注意",
        "----------------------------",
        "当せん金額は設定値ベースの概算です。",
        "正確な回収率には各回の実当せん金額データが必要です。",
        "的中保証は確認できません。",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prize_table_from_args(args: argparse.Namespace) -> Dict[int, int]:
    return {
        1: int(args.prize1),
        2: int(args.prize2),
        3: int(args.prize3),
        4: int(args.prize4),
        5: int(args.prize5),
        6: int(args.prize6),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="loto7.csv だけでロト7予測とAdvancedバックテストを出力します。")
    parser.add_argument("--csv", default=DEFAULT_CSV_URL, help="loto7.csv のURLまたはローカルパス。既定: loto7.csv")
    parser.add_argument("--tickets", type=int, default=DEFAULT_TICKETS, help="画面表示・バックテストする口数。既定: 10")
    parser.add_argument("--pool-size", type=int, default=24, help="候補プールサイズ。大きいほど遅くなります。既定: 24")
    parser.add_argument("--backtest", action="store_true", help="バックテストも実行する")
    parser.add_argument("--min-train", type=int, default=100, help="バックテストの初期学習回数。第2回から検証するなら1")
    parser.add_argument("--backtest-pool-size", type=int, default=16, help="バックテスト時の候補プールサイズ。既定: 16")
    parser.add_argument("--backtest-pool-cap", type=int, default=DEFAULT_BACKTEST_POOL_CAP, help="Actionsタイムアウト防止用の上限。既定: 16")
    parser.add_argument("--max-backtest-draws", type=int, default=0, help="直近N回だけバックテストする。0なら全件")
    parser.add_argument("--monte-carlo", type=int, default=None, help="通常予測のMonteCarlo探索数。未指定なら環境変数または既定値")
    parser.add_argument("--backtest-monte-carlo", type=int, default=None, help="バックテスト時のMonteCarlo探索数。環境変数LOTO7_BACKTEST_MONTE_CARLOへ反映")
    parser.add_argument("--disable-optimize", action="store_true", help="Optuna/Random Search最適化を無効化し、保存済み重みまたは既定重みを使う")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="予測CSV保存先")
    parser.add_argument("--latest-txt", default=DEFAULT_LATEST_TXT, help="最新予測TXT保存先")
    parser.add_argument("--save-count", type=int, default=DEFAULT_SAVE_COUNT, help="保存する予測数。既定: 10")
    parser.add_argument("--no-save", action="store_true", help="予測CSV/TXTを保存しない")
    parser.add_argument("--unit-cost", type=int, default=DEFAULT_UNIT_COST, help="1口の購入金額。既定: 300")
    parser.add_argument("--backtest-summary-csv", default=DEFAULT_BACKTEST_SUMMARY_CSV, help="バックテストサマリーCSV")
    parser.add_argument("--backtest-detail-csv", default=DEFAULT_BACKTEST_DETAIL_CSV, help="バックテスト明細CSV")
    parser.add_argument("--backtest-report-txt", default=DEFAULT_BACKTEST_REPORT_TXT, help="バックテストレポートTXT")
    parser.add_argument("--prize1", type=int, default=DEFAULT_PRIZE_TABLE[1], help="1等の設定当せん金額")
    parser.add_argument("--prize2", type=int, default=DEFAULT_PRIZE_TABLE[2], help="2等の設定当せん金額")
    parser.add_argument("--prize3", type=int, default=DEFAULT_PRIZE_TABLE[3], help="3等の設定当せん金額")
    parser.add_argument("--prize4", type=int, default=DEFAULT_PRIZE_TABLE[4], help="4等の設定当せん金額")
    parser.add_argument("--prize5", type=int, default=DEFAULT_PRIZE_TABLE[5], help="5等の設定当せん金額")
    parser.add_argument("--prize6", type=int, default=DEFAULT_PRIZE_TABLE[6], help="6等の設定当せん金額")
    parser.add_argument("--backtest-output-csv", default=None, help="旧workflow互換。指定時は backtest-summary-csv として扱います。")
    args = parser.parse_args(argv)

    if args.backtest_output_csv:
        args.backtest_summary_csv = args.backtest_output_csv

    if args.disable_optimize:
        import os
        os.environ["LOTO7_DISABLE_OPTIMIZE"] = "1"
    if args.monte_carlo is not None:
        import os
        os.environ["LOTO7_MONTE_CARLO"] = str(args.monte_carlo)
    if args.backtest_monte_carlo is not None:
        import os
        os.environ["LOTO7_BACKTEST_MONTE_CARLO"] = str(args.backtest_monte_carlo)

    from loto7_advanced_optimizer import advanced_predict, save_latest_txt, advanced_backtest

    draws = load_draws(args.csv)
    if not draws:
        print("抽せんデータを読み込めませんでした。", file=sys.stderr)
        return 1

    latest = draws[-1]
    target_date = next_friday_after(latest.date)

    print_recent_summary(draws)

    display_tickets = advanced_predict(
        draws,
        num_tickets=args.tickets,
        pool_size=args.pool_size,
        hit_pattern_csv=args.backtest_detail_csv,
        monte_carlo_iterations=args.monte_carlo,
        optimize=not args.disable_optimize,
    )
    print_predictions(display_tickets)

    if not args.no_save:
        save_predictions_csv(
            output_path=args.output_csv,
            target_date=target_date,
            ranked=display_tickets,
            save_count=args.save_count,
        )
        save_latest_txt(
            path=args.latest_txt,
            target_date=target_date,
            tickets=display_tickets[: args.save_count],
        )
        print(f"保存完了: {args.output_csv}")
        print(f"保存完了: {args.latest_txt}")
        print(f"保存形式: 抽せん日, 予測1, スコア正規化値1, 戦略1 ...")
        print(f"対象抽せん日: {target_date}")
        print()

    if args.backtest:
        prize_table = build_prize_table_from_args(args)
        print("=== バックテスト用 設定金額 ===")
        print(f"1口購入金額: {yen(args.unit_cost)}")
        for grade in range(1, 7):
            print(f"{grade}等: {yen(prize_table.get(grade, 0))}")
        print()

        max_backtest_draws = args.max_backtest_draws if args.max_backtest_draws > 0 else 0
        summary = advanced_backtest(
            draws,
            min_train=args.min_train,
            num_tickets=args.tickets,
            pool_size=min(args.backtest_pool_size, args.backtest_pool_cap) if args.backtest_pool_cap > 0 else args.backtest_pool_size,
            hit_pattern_csv=args.backtest_detail_csv,
            max_backtest_draws=max_backtest_draws,
            summary_csv=args.backtest_summary_csv,
            detail_csv=args.backtest_detail_csv,
        )
        write_compat_report(summary, args.backtest_report_txt)

        print("=== Advanced バックテスト結果 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"保存完了: {args.backtest_summary_csv}")
        print(f"保存完了: {args.backtest_detail_csv}")
        print(f"保存完了: {args.backtest_report_txt}")
        print()

    print("注意: スコア正規化値は当選確率ではありません。")
    print("注意: 的中保証は確認できません。過去実績ベースの候補生成です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
