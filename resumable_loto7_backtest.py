#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable NEW_LOTO7 walk-forward backtest with in-loop commit/push.

Why this version exists:
    GitHub Actions can cancel a job before a later "commit results" step runs.
    Therefore this script writes checkpoint files and optionally commits/pushes
    them from inside the processing loop.

Main behavior:
    - Reads existing loto7_backtest_detail.csv and skips completed draw dates.
    - Processes at most --chunk-size new target draws per run.
    - Writes detail/summary/resume/report after every --save-every draws.
    - When --commit-inside-loop is set, commits/pushes after every --commit-every draws.
    - Handles SIGTERM/SIGINT by flushing current results before exiting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
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

STOP_REQUESTED = False


def _handle_stop(signum, frame):  # type: ignore[no-untyped-def]
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[STOP] received signal={signum}; flushing checkpoint soon...", flush=True)


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


def run_cmd(cmd: Sequence[str], check: bool = False) -> int:
    print("[CMD] " + " ".join(cmd), flush=True)
    try:
        res = subprocess.run(cmd, text=True, check=check)
        return int(res.returncode)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] command failed rc={exc.returncode}: {' '.join(cmd)}", flush=True)
        return int(exc.returncode)
    except Exception as exc:
        print(f"[WARN] command error: {exc}", flush=True)
        return 1


def read_existing_rows(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc}", flush=True)
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
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)


def grade_label(grade: object) -> str:
    return "ハズレ" if grade is None else f"{grade}等"


def summarize(rows: Sequence[Dict[str, object]], tickets: int) -> Dict[str, object]:
    best: List[int] = []
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
        f"target_count: {resume.get('target_count')}",
        f"last_date: {resume.get('last_date')}",
        f"has_v2: {resume.get('has_v2')}",
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


def maybe_postprocess(args: argparse.Namespace) -> None:
    if not HAS_V2 or args.skip_postprocess:
        return
    try:
        cluster_hit_structures(args.detail_csv)
        train_meta_classifier(args.detail_csv)
        mem5 = build_memory5(args.detail_csv, None)
        mem5.save()
    except Exception as exc:
        print(f"[WARN] v2 postprocess skipped: {exc}", flush=True)


def checkpoint(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    done_dates: set[str],
    target_count: int,
    last_date: str,
    reason: str,
    do_commit: bool,
) -> Dict[str, object]:
    rows_sorted = sorted(rows, key=lambda r: str(r.get("抽せん日", "")))
    summary = summarize(rows_sorted, args.tickets)
    completed = len(done_dates) >= target_count
    resume = {
        "completed": completed,
        "completed_count": len(done_dates),
        "target_count": target_count,
        "last_date": last_date or (rows_sorted[-1].get("抽せん日") if rows_sorted else ""),
        "has_v2": HAS_V2,
        "reason": reason,
        "updated_at_epoch": int(time.time()),
    }

    write_rows(args.detail_csv, rows_sorted)
    write_rows(args.summary_csv, [summary])
    Path(args.resume_json).write_text(json.dumps(resume, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report_txt, summary, resume)

    if args.postprocess_every_checkpoint:
        maybe_postprocess(args)

    print(f"[CHECKPOINT] reason={reason} completed_count={len(done_dates)}/{target_count} last_date={resume.get('last_date')}", flush=True)

    if do_commit and args.commit_inside_loop:
        commit_checkpoint(args, reason=reason)

    return resume


def commit_checkpoint(args: argparse.Namespace, reason: str) -> None:
    files = [
        args.detail_csv,
        args.summary_csv,
        args.report_txt,
        args.resume_json,
        "loto7_memorybank.csv",
        "loto7_memorybank_5plus.csv",
        "loto7_advanced_weights.json",
        "loto7_meta_classifier.json",
        "loto7_hit_structure_clusters.csv",
    ]
    existing = [f for f in files if Path(f).exists()]
    if not existing:
        print("[COMMIT] no files exist to add", flush=True)
        return

    run_cmd(["git", "add", *existing])
    diff_rc = run_cmd(["git", "diff", "--cached", "--quiet"])
    if diff_rc == 0:
        print("[COMMIT] no staged changes", flush=True)
        return

    msg = f"resume loto7 backtest {reason}"
    run_cmd(["git", "commit", "-m", msg])
    if not args.no_pull_before_push:
        run_cmd(["git", "pull", "--rebase", "origin", os.getenv("GITHUB_REF_NAME", "main")])
    run_cmd(["git", "push"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--tickets", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5, help="Write checkpoint files every N processed draws")
    parser.add_argument("--commit-every", type=int, default=10, help="Commit/push every N processed draws when --commit-inside-loop is set")
    parser.add_argument("--commit-inside-loop", action="store_true", help="Commit/push from inside this Python process")
    parser.add_argument("--no-pull-before-push", action="store_true")
    parser.add_argument("--detail-csv", default="loto7_backtest_detail.csv")
    parser.add_argument("--summary-csv", default="loto7_backtest_summary.csv")
    parser.add_argument("--report-txt", default="loto7_backtest_report.txt")
    parser.add_argument("--resume-json", default="loto7_backtest_resume.json")
    parser.add_argument("--mcts-iterations", type=int, default=int(os.getenv("LOTO7_MCTS_ITERATIONS", "4000")))
    parser.add_argument("--monte-carlo-iterations", type=int, default=int(os.getenv("LOTO7_BACKTEST_MONTE_CARLO", "1200")))
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip clustering/meta/memory5 postprocess")
    parser.add_argument("--postprocess-every-checkpoint", action="store_true", help="Run v2 postprocess at every checkpoint; slower")
    args = parser.parse_args()

    draws = load_draws(args.csv)
    if len(draws) <= args.min_train:
        raise SystemExit("Not enough draw data")

    rows = read_existing_rows(args.detail_csv)
    done_dates = {str(row.get("抽せん日", "")) for row in rows if row.get("抽せん日")}
    target_count = max(0, len(draws) - args.min_train)
    processed = 0
    last_date = ""

    print(f"[START] existing_done={len(done_dates)} target={target_count} chunk_size={args.chunk_size} has_v2={HAS_V2}", flush=True)

    try:
        for i in range(args.min_train, len(draws)):
            if STOP_REQUESTED:
                break
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
            print(f"[PROCESSED] {actual.date} processed={processed}/{args.chunk_size} total_done={len(done_dates)}/{target_count}", flush=True)

            if processed % max(1, args.save_every) == 0:
                checkpoint(args, rows, done_dates, target_count, last_date, reason=f"save-{processed}", do_commit=False)

            if processed % max(1, args.commit_every) == 0:
                checkpoint(args, rows, done_dates, target_count, last_date, reason=f"commit-{processed}", do_commit=True)

    finally:
        # Always flush local files; if the runner gives us enough time on SIGTERM,
        # this also commits the last partial batch.
        reason = "final" if not STOP_REQUESTED else "signal-stop"
        if HAS_V2 and not args.postprocess_every_checkpoint:
            maybe_postprocess(args)
        resume = checkpoint(args, rows, done_dates, target_count, last_date, reason=reason, do_commit=True)
        print(json.dumps(resume, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
