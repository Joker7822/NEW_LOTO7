#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunked resumable backtest runner for NEW_LOTO7.

Purpose:
- Run walk-forward validation from draw #2 to the latest draw.
- Avoid GitHub Actions timeout by processing only a fixed chunk per invocation.
- Resume from loto7_backtest_detail.csv / loto7_backtest_resume.json.

Important:
Loto7 is an independent lottery. This script validates historical ranking behavior; it
cannot guarantee future wins.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from loto7_logic_predictor import (
    DEFAULT_PRIZE_TABLE,
    DEFAULT_UNIT_COST,
    classify_loto7_prize,
    format_ticket,
    load_draws,
    write_compat_report,
)
from loto7_advanced_optimizer import (
    AdvancedWeights,
    CLUSTER_CSV,
    MEMORYBANK_5PLUS_CSV,
    MEMORYBANK_CSV,
    META_CLASSIFIER_JSON,
    NEXTGEN_META6_JSON,
    NEXTGEN_SHAP_JSON,
    RESUME_JSON,
    MemoryBank,
    MemoryBank5Plus,
    advanced_predict,
    build_hit_structure_clusters,
    summarize_rows,
    train_meta_classifier,
)


def _read_rows(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        return
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_resume(path: str, completed: int, last_date: str, completed_all: bool) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "completed": completed,
                "last_date": last_date,
                "completed_all": completed_all,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _grade_label(grade: int | None) -> str:
    return "ハズレ" if grade is None else f"{grade}等"


def _target_indices(draws_len: int, min_train: int, done_dates: set[str], draw_dates: Sequence[str]) -> List[int]:
    start = max(1, int(min_train))
    return [i for i in range(start, draws_len) if draw_dates[i] not in done_dates]


def _train_models(detail_csv: str) -> None:
    build_hit_structure_clusters(detail_csv, CLUSTER_CSV)
    train_meta_classifier(detail_csv, META_CLASSIFIER_JSON)
    try:
        from loto7_nextgen_models import shap_feature_selection, train_meta6_classifier

        train_meta6_classifier(detail_csv, NEXTGEN_META6_JSON)
        shap_feature_selection(detail_csv, NEXTGEN_SHAP_JSON)
    except Exception as exc:
        if os.getenv("LOTO7_DEBUG_NEXTGEN", "0") == "1":
            print(f"[WARN] nextgen training skipped: {exc}")


def run_chunked_backtest(
    csv_path: str,
    min_train: int,
    num_tickets: int,
    pool_size: int,
    chunk_size: int,
    max_chunks: int,
    detail_csv: str,
    summary_csv: str,
    report_txt: str,
) -> Dict[str, object]:
    draws = load_draws(csv_path)
    if len(draws) <= min_train:
        raise RuntimeError("抽せんデータ数が不足しています。")

    rows = _read_rows(detail_csv)
    done_dates = {str(r.get("抽せん日", "")) for r in rows if r.get("抽せん日")}
    draw_dates = [d.date for d in draws]
    pending = _target_indices(len(draws), min_train, done_dates, draw_dates)

    limit = max(1, int(chunk_size)) * max(1, int(max_chunks))
    targets = pending[:limit]

    weights = AdvancedWeights()
    bank = MemoryBank()
    bank5 = MemoryBank5Plus()

    monte_carlo = int(os.getenv("LOTO7_BACKTEST_MONTE_CARLO", "100"))
    mcts = int(os.getenv("LOTO7_BACKTEST_MCTS", os.getenv("LOTO7_MCTS_ITERATIONS", "50")))

    print("=== Chunked Backtest ===")
    print(f"全抽せん数: {len(draws)}")
    print(f"検証開始: 第{min_train + 1}回相当")
    print(f"既存完了: {len(done_dates)}")
    print(f"未処理: {len(pending)}")
    print(f"今回処理上限: {limit}")
    print(f"今回処理: {len(targets)}")
    print()

    for pos, i in enumerate(targets, start=1):
        actual = draws[i]
        preds = advanced_predict(
            draws[:i],
            num_tickets=num_tickets,
            pool_size=pool_size,
            hit_pattern_csv=detail_csv,
            before_date=actual.date,
            monte_carlo_iterations=monte_carlo,
            mcts_iterations=mcts,
            weights=weights,
            optimize=False,
        )
        results = [classify_loto7_prize(p.ticket, actual.main, actual.bonus, DEFAULT_PRIZE_TABLE) for p in preds]
        hits = [r.main_matches for r in results]
        grades = [r.grade for r in results if r.grade is not None]
        best_grade = min(grades) if grades else None

        row: Dict[str, object] = {
            "抽せん日": actual.date,
            "回別": actual.draw_no or "",
            "本数字": format_ticket(actual.main),
            "ボーナス数字": format_ticket(actual.bonus),
            "口数": len(preds),
            "購入金額": len(preds) * DEFAULT_UNIT_COST,
            "当せん金額": sum(r.prize for r in results),
            "収支": sum(r.prize for r in results) - len(preds) * DEFAULT_UNIT_COST,
            "最高等級": _grade_label(best_grade),
            "最高本数字一致数": max(hits) if hits else 0,
            "当せん口数": sum(1 for r in results if r.grade is not None),
        }
        for idx, (pred, result) in enumerate(zip(preds, results), start=1):
            row[f"予測{idx}"] = format_ticket(pred.ticket)
            row[f"予測{idx}_戦略"] = pred.strategy
            row[f"予測{idx}_本数字一致"] = result.main_matches
            row[f"予測{idx}_ボーナス一致"] = result.bonus_matches
            row[f"予測{idx}_等級"] = _grade_label(result.grade)
            row[f"予測{idx}_当せん金額"] = result.prize
            if result.main_matches >= 4:
                bank.add(pred.ticket, 1.0 + max(0, result.main_matches - 4) * 0.8)
            if result.main_matches >= 5:
                bank5.add(pred.ticket, result.main_matches, actual.date)

        rows.append(row)
        done_dates.add(actual.date)
        rows.sort(key=lambda r: str(r.get("抽せん日", "")))

        if pos % max(1, int(chunk_size)) == 0:
            summary = summarize_rows(rows, num_tickets, min_train, min_train, weights)
            _write_csv(detail_csv, rows)
            _write_csv(summary_csv, [summary])
            write_compat_report(summary, report_txt)
            bank.save(MEMORYBANK_CSV)
            bank5.save(MEMORYBANK_5PLUS_CSV)
            _save_resume(RESUME_JSON, len(done_dates), actual.date, False)
            print(f"[checkpoint] {pos}/{len(targets)} saved: {actual.date}")

    completed_all = len(_target_indices(len(draws), min_train, done_dates, draw_dates)) == 0
    summary = summarize_rows(rows, num_tickets, min_train, min_train, weights)
    summary["未処理回数"] = len(_target_indices(len(draws), min_train, done_dates, draw_dates))
    summary["完全完了"] = completed_all

    _write_csv(detail_csv, rows)
    _write_csv(summary_csv, [summary])
    write_compat_report(summary, report_txt)
    bank.save(MEMORYBANK_CSV)
    bank5.save(MEMORYBANK_5PLUS_CSV)
    _save_resume(RESUME_JSON, len(done_dates), str(rows[-1].get("抽せん日", "")) if rows else "", completed_all)

    if rows and (completed_all or os.getenv("LOTO7_TRAIN_MODELS_EACH_CHUNK", "0") == "1"):
        _train_models(detail_csv)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NEW_LOTO7 backtest in resumable chunks.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("LOTO7_BACKTEST_CHUNK_SIZE", "100")))
    parser.add_argument("--max-chunks", type=int, default=int(os.getenv("LOTO7_BACKTEST_MAX_CHUNKS", "1")))
    parser.add_argument("--detail-csv", default="loto7_backtest_detail.csv")
    parser.add_argument("--summary-csv", default="loto7_backtest_summary.csv")
    parser.add_argument("--report-txt", default="loto7_backtest_report.txt")
    args = parser.parse_args()

    summary = run_chunked_backtest(
        csv_path=args.csv,
        min_train=args.min_train,
        num_tickets=args.tickets,
        pool_size=args.pool_size,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        detail_csv=args.detail_csv,
        summary_csv=args.summary_csv,
        report_txt=args.report_txt,
    )
    print("=== Chunked Backtest Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
