#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable NEW_LOTO7 walk-forward backtest.

This script processes only a fixed number of target draws per execution.
GitHub Actions can commit after each execution, so canceled or timed-out runs
can resume from the already committed detail CSV / resume JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from loto7_logic_predictor import (
    DEFAULT_PRIZE_TABLE,
    DEFAULT_UNIT_COST,
    Draw,
    classify_loto7_prize,
    format_ticket,
    load_draws,
)

try:
    from loto7_advanced_v2 import (
        advanced_v2_predict,
        cluster_hit_structures,
        train_meta_classifier,
        build_memory5,
    )
    HAS_V2 = True
except Exception:
    HAS_V2 = False
    from loto7_advanced_optimizer import advanced_predict


def read_existing_rows(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def write_rows(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def grade_label(grade: object) -> str:
    return "ハズレ" if grade is None else f"{grade}等"


def summarize(rows: Sequence[Dict[str, object]], tickets: int) -> Dict[str, object]:
    best = []
    purchase = 0
    prize = 0
    grade_counter: Counter = Counter()
    best_grade_counter: Counter = Counter()

    for row in rows:
        try:
            best.append(int(row.get("最高本数字一致数", 0) or 0))
        except Exception:
            best.append(0)
        try:
            purchase += int(float(row.get("購入金額", 0) or 0))
            prize += int(float(row.get("当せん金額", 0) or 0))
        except Exception:
            pass
        best_grade_counter[str(row.get("最高等級", "ハズレ") or "ハズレ")] += 1
        for k, v in row.items():
            if k.endswith("_等級"):
                grade_counter[str(v or "ハズレ")] += 1

    def rate(threshold: int) -> float:
        return sum(1 for v in best if v >= threshold) / len(best) if best else 0.0

    return {
        "検証回数": len(rows),
        "1回あたり口数": tickets,
        "全口ベスト平均一致数": round(sum(best) / len(best), 6) if best else 0,
        "全口ベスト_3個以上率": round(rate(3), 6),
        "全口ベスト_4個以上率": round(rate(4), 6),
        "全口ベスト_5個以上率": round(rate(5), 6),
        "全口ベスト_6個以上率": round(rate(6), 6),
        "総購入金額": purchase,
        "総当せん金額": prize,
        "総収支": prize - purchase,
        "総回収率": round(prize / purchase, 6) if purchase else 0,
        "全予測口等級分布": dict(sorted(grade_counter.items())),
        "各回ベスト等級分布": dict(sorted(best_grade_counter.items())),
    }


def write_report(path: str, summary: Dict[str, object], resume: Dict[str, object]) -> None:
    lines = [
        "Loto7 Resumable Backtest Report",
        "================================",
        "",
        f"completed: {resume.get('completed')}",
        f"completed_count: {resume.get('completed_count')}",
        f"last_date: {resume.get('last_date')}",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict_for_draw(train: Sequence[Draw], args: argparse.Namespace):
    if HAS_V2:
        return advanced_v2_predict(
            train,
            num_tickets=args.tickets,
            pool_size=args.pool_size,
            detail_csv=args.detail_csv,
            mcts_iterations=args.mcts_iterations,
        )
    return advanced_predict(
        train,
        num_tickets=args.tickets,
        pool_size=args.pool_size,
        hit_pattern_csv=args.detail_csv,
        monte_carlo_iterations=args.monte_carlo_iterations,
        optimize=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--detail-csv", default="loto7_backtest_detail.csv")
    parser.add_argument("--summary-csv", default="loto7_backtest_summary.csv")
    parser.add_argument("--report-txt", default="loto7_backtest_report.txt")
    parser.add_argument("--resume-json", default="loto7_backtest_resume.json")
    parser.add_argument("--mcts-iterations", type=int, default=int(os.getenv("LOTO7_MCTS_ITERATIONS", "4000")))
    parser.add_argument("--monte-carlo-iterations", type=int, default=int(os.getenv("LOTO7_BACKTEST_MONTE_CARLO", "1200")))
    args = parser.parse_args()

    draws = load_draws(args.csv)
    if len(draws) <= args.min_train:
        raise SystemExit("Not enough draw data")

    rows = read_existing_rows(args.detail_csv)
    done_dates = {str(row.get("抽せん日", "")) for row in rows if row.get("抽せん日")}
    processed = 0
    last_date = None

    for i in range(args.min_train, len(draws)):
        actual = draws[i]
        if actual.date in done_dates:
            continue
        if processed >= args.chunk_size:
            break

        preds = predict_for_draw(draws[:i], args)
        results = [classify_loto7_prize(p.ticket, actual.main, actual.bonus, DEFAULT_PRIZE_TABLE) for p in preds]
        hits = [r.main_matches for r in results]
        grades = [r.grade for r in results if r.grade is not None]
        purchase = len(preds) * DEFAULT_UNIT_COST
        prize = sum(r.prize for r in results)

        row: Dict[str, object] = {
            "抽せん日": actual.date,
            "回別": actual.draw_no or "",
            "本数字": format_ticket(actual.main),
            "ボーナス数字": format_ticket(actual.bonus),
            "口数": len(preds),
            "購入金額": purchase,
            "当せん金額": prize,
            "収支": prize - purchase,
            "最高等級": grade_label(min(grades) if grades else None),
            "最高本数字一致数": max(hits) if hits else 0,
            "当せん口数": sum(1 for r in results if r.grade is not None),
        }
        for idx, (pred, result) in enumerate(zip(preds, results), start=1):
            row[f"予測{idx}"] = format_ticket(pred.ticket)
            row[f"予測{idx}_戦略"] = pred.strategy
            row[f"予測{idx}_本数字一致"] = result.main_matches
            row[f"予測{idx}_ボーナス一致"] = result.bonus_matches
            row[f"予測{idx}_等級"] = grade_label(result.grade)
            row[f"予測{idx}_当せん金額"] = result.prize

        rows.append(row)
        done_dates.add(actual.date)
        processed += 1
        last_date = actual.date
        print(f"processed {actual.date} ({processed}/{args.chunk_size})")

    rows.sort(key=lambda r: str(r.get("抽せん日", "")))
    summary = summarize(rows, args.tickets)
    completed = len(done_dates) >= max(0, len(draws) - args.min_train)
    resume = {
        "completed": completed,
        "completed_count": len(done_dates),
        "target_count": max(0, len(draws) - args.min_train),
        "last_date": last_date or (rows[-1].get("抽せん日") if rows else ""),
        "has_v2": HAS_V2,
    }

    write_rows(args.detail_csv, rows)
    write_rows(args.summary_csv, [summary])
    Path(args.resume_json).write_text(json.dumps(resume, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report_txt, summary, resume)

    if HAS_V2:
        try:
            cluster_hit_structures(args.detail_csv)
            train_meta_classifier(args.detail_csv)
            mem5 = build_memory5(args.detail_csv, None)
            mem5.save()
        except Exception as exc:
            print(f"v2 postprocess skipped: {exc}")

    print(json.dumps(resume, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
