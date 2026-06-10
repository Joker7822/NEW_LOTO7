#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a human-readable evaluation report for NEW_LOTO7 backtest outputs."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_UNIT_COST = 300
DEFAULT_PRIZE_TABLE = {
    "1等": 700_000_000,
    "2等": 7_300_000,
    "3等": 730_000,
    "4等": 9_100,
    "5等": 1_400,
    "6等": 1_000,
    "外れ": 0,
}


def yen(v: int | float) -> str:
    return f"{int(round(v)):,}円"


def pct(n: int | float, d: int | float) -> str:
    return f"{(float(n) / float(d) * 100):.4f}%" if d else "0.0000%"


def read_rows(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {path}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_latest_prediction(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return [row.get("numbers", "") for row in csv.DictReader(f) if row.get("numbers")]


def evaluate(rows: Iterable[Dict[str, str]], unit_cost: int) -> Dict[str, object]:
    rows = list(rows)
    total_tickets = len(rows)
    targets = sorted(
        {r.get("target_draw_no", "") for r in rows if r.get("target_draw_no")},
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )
    total_targets = len(targets)

    rank_counts: Counter[str] = Counter(r.get("prize_rank", "外れ") or "外れ" for r in rows)
    match_counts: Counter[int] = Counter()
    per_draw_best: Dict[str, int] = defaultdict(int)
    per_draw_win: Counter[str] = Counter()
    per_draw_prize: Counter[str] = Counter()

    for row in rows:
        draw_no = row.get("target_draw_no", "")
        rank = row.get("prize_rank", "外れ") or "外れ"
        try:
            main_match = int(row.get("main_match", 0) or 0)
        except ValueError:
            main_match = 0
        match_counts[main_match] += 1
        if draw_no:
            per_draw_best[draw_no] = max(per_draw_best[draw_no], main_match)
            if rank != "外れ":
                per_draw_win[draw_no] += 1
            per_draw_prize[draw_no] += int(DEFAULT_PRIZE_TABLE.get(rank, 0))

    total_cost = total_tickets * unit_cost
    total_prize = sum(int(DEFAULT_PRIZE_TABLE.get(rank, 0)) * count for rank, count in rank_counts.items())
    profit = total_prize - total_cost
    roi = (total_prize / total_cost) if total_cost else 0.0
    best_match = max(match_counts.keys()) if match_counts else 0
    hit_tickets = sum(v for k, v in rank_counts.items() if k != "外れ")
    hit_draws = sum(1 for v in per_draw_win.values() if v > 0)
    best4plus_draws = sum(1 for v in per_draw_best.values() if v >= 4)
    best5plus_draws = sum(1 for v in per_draw_best.values() if v >= 5)
    best6plus_draws = sum(1 for v in per_draw_best.values() if v >= 6)

    return {
        "total_tickets": total_tickets,
        "total_targets": total_targets,
        "rank_counts": rank_counts,
        "match_counts": match_counts,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "profit": profit,
        "roi": roi,
        "best_match": best_match,
        "hit_tickets": hit_tickets,
        "hit_draws": hit_draws,
        "best4plus_draws": best4plus_draws,
        "best5plus_draws": best5plus_draws,
        "best6plus_draws": best6plus_draws,
        "per_draw_prize": per_draw_prize,
    }


def build_recommendations(stats: Dict[str, object]) -> List[str]:
    total_targets = int(stats["total_targets"])
    total_tickets = int(stats["total_tickets"])
    roi = float(stats["roi"])
    best_match = int(stats["best_match"])
    rank_counts: Counter[str] = stats["rank_counts"]  # type: ignore[assignment]
    match_counts: Counter[int] = stats["match_counts"]  # type: ignore[assignment]

    recs: List[str] = []
    if roi < 0.30:
        recs.append("回収率が30%未満の場合、現在の順位スコアだけでは収益性は弱い。候補プールを広げるより、5本一致以上を目的変数にした重み再探索を優先する。")
    elif roi < 0.70:
        recs.append("回収率は改善余地あり。4等・5等の頻度を維持しながら、6等狙いに寄りすぎた組合せを削る。")
    else:
        recs.append("回収率は相対的に良好。現在の分散制約を維持しつつ、直近240回モデルとのアンサンブル比率を調整する。")

    if best_match < 6:
        recs.append("最大一致が5本以下のため、3等以上狙いの6本一致MetaClassifierを強化する。特徴量は相性ペア、直近出現間隔、合計値帯、奇偶、低高、前回重複数を優先する。")
    if rank_counts.get("4等", 0) == 0:
        recs.append("4等が出ていない場合、5本一致候補の探索密度が不足。MCTS/MonteCarlo候補数を増やすか、上位5口の多様性制約を緩める。")
    if sum(match_counts.get(i, 0) for i in range(0, 3)) / max(total_tickets, 1) > 0.80:
        recs.append("0〜2本一致が80%超なら、候補プールが履歴頻度に寄りすぎ。休眠数字・周期ギャップ・低頻度ペアを一部混ぜる探索戦略を追加する。")
    if total_targets < 300:
        recs.append("検証対象が300回未満なら評価が不安定。第2回相当から最新までの全件検証を優先する。")

    recs.append("次の改善は、A/B比較用に outputs/loto7_backtest_result.csv を上書きせず、設定名つきファイルへ保存する構成が望ましい。")
    return recs


def write_report(
    output_path: str,
    stats: Dict[str, object],
    latest_predictions: List[str],
    unit_cost: int,
) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rank_counts: Counter[str] = stats["rank_counts"]  # type: ignore[assignment]
    match_counts: Counter[int] = stats["match_counts"]  # type: ignore[assignment]
    total_tickets = int(stats["total_tickets"])
    total_targets = int(stats["total_targets"])
    purchase_count = int(total_tickets / total_targets) if total_targets else 0

    lines: List[str] = []
    lines.append("LOTO7 バックテスト評価レポート")
    lines.append("=" * 40)
    lines.append(f"生成日時UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append("")
    lines.append("結論")
    lines.append("-" * 40)
    lines.append(f"検証回数: {total_targets}回")
    lines.append(f"総予測口数: {total_tickets}口")
    lines.append(f"1回あたり購入口数: {purchase_count}口")
    lines.append(f"最大本数字一致数: {stats['best_match']}本")
    lines.append(f"総購入額: {yen(int(stats['total_cost']))}")
    lines.append(f"推定当せん額: {yen(int(stats['total_prize']))}")
    lines.append(f"推定収支: {yen(int(stats['profit']))}")
    lines.append(f"推定回収率: {float(stats['roi']) * 100:.4f}%")
    lines.append("")

    lines.append("理由")
    lines.append("-" * 40)
    lines.append("この評価は outputs/loto7_backtest_result.csv の各予測口について、本数字一致数・ボーナス一致数・等級を集計したものです。")
    lines.append("当せん額は固定賞金表による推定値であり、実際の販売口数・キャリーオーバー・回別賞金変動は反映していません。")
    lines.append("ロト7は独立抽せんのため、過去バックテストの成績は次回的中を保証しません。")
    lines.append("")

    lines.append("数字")
    lines.append("-" * 40)
    lines.append("等級別件数:")
    for rank in ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]:
        count = int(rank_counts.get(rank, 0))
        lines.append(f"  {rank}: {count}件 / {pct(count, total_tickets)}")
    lines.append("")
    lines.append("本数字一致数分布:")
    for m in range(0, 8):
        count = int(match_counts.get(m, 0))
        lines.append(f"  {m}本一致: {count}件 / {pct(count, total_tickets)}")
    lines.append("")
    lines.append(f"当せん口数: {stats['hit_tickets']}口 / {pct(int(stats['hit_tickets']), total_tickets)}")
    lines.append(f"当せん発生回数: {stats['hit_draws']}回 / {pct(int(stats['hit_draws']), total_targets)}")
    lines.append(f"各回ベスト4本以上: {stats['best4plus_draws']}回 / {pct(int(stats['best4plus_draws']), total_targets)}")
    lines.append(f"各回ベスト5本以上: {stats['best5plus_draws']}回 / {pct(int(stats['best5plus_draws']), total_targets)}")
    lines.append(f"各回ベスト6本以上: {stats['best6plus_draws']}回 / {pct(int(stats['best6plus_draws']), total_targets)}")
    lines.append("")

    lines.append("改善提案")
    lines.append("-" * 40)
    for idx, rec in enumerate(build_recommendations(stats), start=1):
        lines.append(f"{idx}. {rec}")
    lines.append("")

    if latest_predictions:
        lines.append("最新予測5口")
        lines.append("-" * 40)
        for idx, nums in enumerate(latest_predictions, start=1):
            lines.append(f"{idx}. {nums}")
        lines.append("")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EVALUATE] wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write LOTO7 backtest evaluation txt report")
    parser.add_argument("--result-csv", default="outputs/loto7_backtest_result.csv")
    parser.add_argument("--latest-prediction-csv", default="outputs/loto7_latest_prediction.csv")
    parser.add_argument("--output", default="outputs/loto7_backtest_evaluation.txt")
    parser.add_argument("--unit-cost", type=int, default=DEFAULT_UNIT_COST)
    args = parser.parse_args()

    rows = read_rows(args.result_csv)
    stats = evaluate(rows, unit_cost=args.unit_cost)
    latest = read_latest_prediction(args.latest_prediction_csv)
    write_report(args.output, stats, latest, unit_cost=args.unit_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
