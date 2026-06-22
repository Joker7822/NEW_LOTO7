#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holdout_evaluator.py

進化型探索で作成した loto7_best_model.json を固定し、holdout成績を
実当せん金額ベースで評価する。全期間バックテスト向けに途中保存・再開に対応。

重要:
    - roi / roi_percent は「収支 ÷ 購入額」で計算する
    - 従来の「払戻 ÷ 購入額」は payout_roi / payout_roi_percent として別出力する
    - 的中率は口数ベースと回別ベースの2種類を出力する
    - 各検証回の予測生成は train = draws[:idx] のみを使い、対象回以降は使わない
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from loto7_evolution_trainer import (
    Draw,
    evaluate_ticket,
    generate_tickets,
    load_best_model,
    load_draws,
)

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]
FIELDNAMES = [
    "draw_no", "date", "year", "combo_index", "ticket", "actual_main", "actual_bonus",
    "main_match", "bonus_match", "rank", "purchase_cost", "prize_amount", "profit", "prize_data_missing",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def format_yen(value: object) -> str:
    try:
        return f"{int(value):,}円"
    except Exception:
        return f"{value}円"


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def roi_from_profit(total_cost: int, total_payout: int) -> float:
    """収支ベースROI: (払戻額 - 購入額) / 購入額。"""
    if total_cost <= 0:
        return 0.0
    return (total_payout - total_cost) / total_cost


def roi_from_payout(total_cost: int, total_payout: int) -> float:
    """従来の回収率: 払戻額 / 購入額。互換確認用に別名で保持する。"""
    if total_cost <= 0:
        return 0.0
    return total_payout / total_cost


def empty_year_stats() -> Dict[str, object]:
    return {
        "target_draws": 0,
        "total_tickets": 0,
        "winning_ticket_count": 0,
        "ticket_hit_rate": 0.0,
        "ticket_hit_rate_percent": 0.0,
        "total_cost": 0,
        "total_payout": 0,
        "profit": 0,
        "roi": 0.0,
        "roi_percent": 0.0,
        "payout_roi": 0.0,
        "payout_roi_percent": 0.0,
        "max_main_match": 0,
        "rank_counts": {rank: 0 for rank in RANK_ORDER},
    }


def update_year_stats(stats: Dict[str, object], *, cost: int, payout: int, rank: str, main_match: int) -> None:
    stats["total_tickets"] = int(stats["total_tickets"]) + 1
    if rank != "外れ":
        stats["winning_ticket_count"] = int(stats.get("winning_ticket_count", 0)) + 1
    stats["total_cost"] = int(stats["total_cost"]) + cost
    stats["total_payout"] = int(stats["total_payout"]) + payout
    stats["profit"] = int(stats["total_payout"]) - int(stats["total_cost"])
    stats["max_main_match"] = max(int(stats["max_main_match"]), main_match)
    rank_counts = stats["rank_counts"]
    assert isinstance(rank_counts, dict)
    rank_counts[rank] = int(rank_counts.get(rank, 0)) + 1
    total_cost = int(stats["total_cost"])
    total_payout = int(stats["total_payout"])
    total_tickets = int(stats["total_tickets"])
    winning_ticket_count = int(stats.get("winning_ticket_count", 0))
    roi = roi_from_profit(total_cost, total_payout)
    payout_roi = roi_from_payout(total_cost, total_payout)
    ticket_hit_rate = pct(winning_ticket_count, total_tickets)
    stats["roi"] = round(roi, 6)
    stats["roi_percent"] = round(roi * 100.0, 3)
    stats["payout_roi"] = round(payout_roi, 6)
    stats["payout_roi_percent"] = round(payout_roi * 100.0, 3)
    stats["ticket_hit_rate"] = round(ticket_hit_rate, 6)
    stats["ticket_hit_rate_percent"] = round(ticket_hit_rate * 100.0, 3)


def select_target_indices(draws: Sequence[Draw], *, min_train_draws: int, holdout_start_draw: int, holdout_end_draw: Optional[int]) -> List[int]:
    out: List[int] = []
    for idx, draw in enumerate(draws):
        if idx < min_train_draws:
            continue
        if draw.draw_no < holdout_start_draw:
            continue
        if holdout_end_draw is not None and draw.draw_no > holdout_end_draw:
            continue
        out.append(idx)
    return out


def state_key(args: argparse.Namespace, *, model_id: str, model_score: float) -> Dict[str, object]:
    return {
        "csv": args.csv,
        "best_model": args.best_model,
        "model_id": model_id,
        "model_score": round(float(model_score), 8),
        "holdout_start_draw": args.holdout_start_draw,
        "holdout_end_draw": args.holdout_end_draw,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "min_train_draws": args.min_train_draws,
        "output": args.output,
    }


def load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def completed_draws_from_csv(path: Path, key: Dict[str, object], resume: bool, state_path: Path) -> Set[int]:
    if not resume or not path.exists() or not state_path.exists():
        return set()
    state = load_json(state_path)
    if state.get("state_key") != key:
        return set()
    done: Set[int] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                no = draw_no_int(row.get("draw_no"))
                if no is not None:
                    done.add(no)
    except Exception:
        return set()
    return done


def ensure_csv(path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def append_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)


def summarize_detail_csv(path: Path) -> Dict[str, object]:
    rank_counts = {rank: 0 for rank in RANK_ORDER}
    year_summary: Dict[str, Dict[str, object]] = {}
    max_main_match = 0
    total_cost = 0
    total_payout = 0
    total_tickets = 0
    missing_prize_draws: Set[int] = set()
    target_draws: Set[int] = set()
    winning_draws: Set[int] = set()
    winning_ticket_count = 0

    if not path.exists():
        return {
            "target_draws": 0,
            "total_tickets": 0,
            "total_cost": 0,
            "total_payout": 0,
            "profit": 0,
            "roi": 0.0,
            "roi_percent": 0.0,
            "payout_roi": 0.0,
            "payout_roi_percent": 0.0,
            "ticket_hit_rate": 0.0,
            "ticket_hit_rate_percent": 0.0,
            "draw_hit_rate": 0.0,
            "draw_hit_rate_percent": 0.0,
            "max_main_match": 0,
            "rank_counts": rank_counts,
            "missing_prize_draw_count": 0,
            "missing_prize_draws": [],
            "winning_draw_count": 0,
            "winning_ticket_count": 0,
            "year_summary": {},
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            draw_no = draw_no_int(row.get("draw_no"))
            if draw_no is not None:
                target_draws.add(draw_no)
            year = str(row.get("year") or "unknown")
            if year not in year_summary:
                year_summary[year] = empty_year_stats()
            rank = str(row.get("rank") or "外れ")
            cost = int(row.get("purchase_cost") or 0)
            payout = int(row.get("prize_amount") or 0)
            main_match = int(row.get("main_match") or 0)
            total_cost += cost
            total_payout += payout
            total_tickets += 1
            rank_counts[rank] = int(rank_counts.get(rank, 0)) + 1
            max_main_match = max(max_main_match, main_match)
            update_year_stats(year_summary[year], cost=cost, payout=payout, rank=rank, main_match=main_match)
            if rank != "外れ":
                winning_ticket_count += 1
                if draw_no is not None:
                    winning_draws.add(draw_no)
            if str(row.get("prize_data_missing") or "0") == "1" and draw_no is not None:
                missing_prize_draws.add(draw_no)

    profit = total_payout - total_cost
    roi = roi_from_profit(total_cost, total_payout)
    payout_roi = roi_from_payout(total_cost, total_payout)
    ticket_hit_rate = pct(winning_ticket_count, total_tickets)
    draw_hit_rate = pct(len(winning_draws), len(target_draws))
    return {
        "target_draws": len(target_draws),
        "total_tickets": total_tickets,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": profit,
        "roi": round(roi, 6),
        "roi_percent": round(roi * 100.0, 3),
        "payout_roi": round(payout_roi, 6),
        "payout_roi_percent": round(payout_roi * 100.0, 3),
        "ticket_hit_rate": round(ticket_hit_rate, 6),
        "ticket_hit_rate_percent": round(ticket_hit_rate * 100.0, 3),
        "draw_hit_rate": round(draw_hit_rate, 6),
        "draw_hit_rate_percent": round(draw_hit_rate * 100.0, 3),
        "max_main_match": max_main_match,
        "rank_counts": rank_counts,
        "missing_prize_draw_count": len(missing_prize_draws),
        "missing_prize_draws": sorted(missing_prize_draws),
        "winning_draw_count": len(winning_draws),
        "winning_ticket_count": winning_ticket_count,
        "year_summary": dict(sorted(year_summary.items())),
    }


def winning_detail_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = str(row.get("rank") or "外れ")
            if rank == "外れ":
                continue
            draw_no = draw_no_int(row.get("draw_no"))
            rows.append(
                {
                    "draw_no": draw_no if draw_no is not None else 0,
                    "date": row.get("date", ""),
                    "year": row.get("year", ""),
                    "combo_index": int(row.get("combo_index") or 0),
                    "ticket": row.get("ticket", ""),
                    "actual_main": row.get("actual_main", ""),
                    "actual_bonus": row.get("actual_bonus", ""),
                    "main_match": int(row.get("main_match") or 0),
                    "bonus_match": int(row.get("bonus_match") or 0),
                    "rank": rank,
                    "purchase_cost": int(row.get("purchase_cost") or 0),
                    "prize_amount": int(row.get("prize_amount") or 0),
                    "profit": int(row.get("profit") or 0),
                }
            )
    rows.sort(key=lambda x: (int(x["draw_no"]), int(x["combo_index"])))
    return rows


def should_safe_exit(start_monotonic: float, max_runtime_minutes: float, safe_exit_minutes: float) -> bool:
    if max_runtime_minutes <= 0:
        return False
    elapsed = (time.monotonic() - start_monotonic) / 60.0
    return elapsed >= max(0.0, max_runtime_minutes - safe_exit_minutes)


def write_state(path: Path, *, key: Dict[str, object], complete: bool, completed_draws: Sequence[int], total_targets: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": utc_now(),
        "complete": complete,
        "state_key": key,
        "completed_draw_count": len(set(completed_draws)),
        "total_targets": total_targets,
        "remaining_draw_count": max(0, total_targets - len(set(completed_draws))),
        "completed_draws": sorted(set(completed_draws)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(args: argparse.Namespace, *, genome, output_csv: Path, summary_json: Path, report_txt: Optional[Path], complete: bool, total_targets: int, state_path: Path) -> Dict[str, object]:
    stats = summarize_detail_csv(output_csv)
    return {
        "created_at": utc_now(),
        "complete": complete,
        "csv": args.csv,
        "best_model": args.best_model,
        "model_id": genome.id,
        "model_score": genome.score,
        "holdout_start_draw": args.holdout_start_draw,
        "holdout_end_draw": args.holdout_end_draw,
        "target_draws": stats["target_draws"],
        "total_target_draws": total_targets,
        "remaining_target_draws": max(0, total_targets - int(stats["target_draws"])),
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "total_tickets": stats["total_tickets"],
        "total_cost": stats["total_cost"],
        "total_payout": stats["total_payout"],
        "profit": stats["profit"],
        "roi": stats["roi"],
        "roi_percent": stats["roi_percent"],
        "payout_roi": stats["payout_roi"],
        "payout_roi_percent": stats["payout_roi_percent"],
        "ticket_hit_rate": stats["ticket_hit_rate"],
        "ticket_hit_rate_percent": stats["ticket_hit_rate_percent"],
        "draw_hit_rate": stats["draw_hit_rate"],
        "draw_hit_rate_percent": stats["draw_hit_rate_percent"],
        "max_main_match": stats["max_main_match"],
        "rank_counts": stats["rank_counts"],
        "missing_prize_draw_count": stats["missing_prize_draw_count"],
        "missing_prize_draws": stats["missing_prize_draws"],
        "winning_draw_count": stats["winning_draw_count"],
        "winning_ticket_count": stats["winning_ticket_count"],
        "year_summary": stats["year_summary"],
        "detail_csv": str(output_csv),
        "summary_json": str(summary_json),
        "report_txt": str(report_txt) if report_txt else None,
        "state_json": str(state_path),
    }


def write_text_report(summary: Dict[str, object], report_path: str) -> None:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rank_counts = summary.get("rank_counts", {})
    if not isinstance(rank_counts, dict):
        rank_counts = {}
    year_summary = summary.get("year_summary", {})
    if not isinstance(year_summary, dict):
        year_summary = {}
    wins = winning_detail_rows(Path(str(summary.get("detail_csv") or "")))

    lines: List[str] = []
    lines.append("LOTO7 Holdout Backtest Report")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"作成日時(UTC): {summary.get('created_at')}")
    lines.append(f"状態: {'完了' if summary.get('complete') else '途中保存'}")
    lines.append(f"CSV: {summary.get('csv')}")
    lines.append(f"モデル: {summary.get('best_model')}")
    lines.append(f"モデルID: {summary.get('model_id')}")
    lines.append(f"モデルスコア: {summary.get('model_score')}")
    lines.append("")
    lines.append("[検証条件]")
    lines.append(f"対象開始回: {summary.get('holdout_start_draw')}")
    lines.append(f"対象終了回: {summary.get('holdout_end_draw')}")
    lines.append(f"処理済み対象回数: {summary.get('target_draws')}")
    lines.append(f"全対象回数: {summary.get('total_target_draws')}")
    lines.append(f"残り対象回数: {summary.get('remaining_target_draws')}")
    lines.append(f"1回あたり購入口数: {summary.get('purchase_count')}")
    lines.append(f"1口単価: {format_yen(summary.get('unit_cost'))}")
    lines.append("")
    lines.append("[総合成績]")
    lines.append(f"総購入口数: {summary.get('total_tickets')}")
    lines.append(f"総購入額: {format_yen(summary.get('total_cost'))}")
    lines.append(f"総払戻額: {format_yen(summary.get('total_payout'))}")
    lines.append(f"総収支: {format_yen(summary.get('profit'))}")
    lines.append(f"収支率ROI: {summary.get('roi_percent')}%")
    lines.append(f"従来回収率(払戻÷購入): {summary.get('payout_roi_percent')}%")
    lines.append(f"的中率(口数): {summary.get('ticket_hit_rate_percent')}%")
    lines.append(f"的中回率(回別): {summary.get('draw_hit_rate_percent')}%")
    lines.append(f"最大本数字一致数: {summary.get('max_main_match')}")
    lines.append(f"当選回数: {summary.get('winning_draw_count')}")
    lines.append(f"当選口数: {summary.get('winning_ticket_count')}")
    lines.append("")
    lines.append("[等級別件数]")
    for rank in RANK_ORDER:
        lines.append(f"{rank}: {rank_counts.get(rank, 0)}")
    lines.append("")
    lines.append("[当選した回別の詳細]")
    lines.append("対象: 6等以上、つまり rank が 外れ 以外の予測口")
    if wins:
        current_draw = None
        for item in wins:
            if item["draw_no"] != current_draw:
                current_draw = item["draw_no"]
                lines.append("")
                lines.append(f"第{item['draw_no']}回 / {item['date']}")
                lines.append(f"実本数字: {item['actual_main']} / ボーナス: {item['actual_bonus']}")
            lines.append(
                f"  - {item['rank']} / {item['combo_index']}口目 / "
                f"予測: {item['ticket']} / "
                f"一致: 本数字{item['main_match']}個 + ボーナス{item['bonus_match']}個 / "
                f"払戻: {format_yen(item['prize_amount'])} / "
                f"収支: {format_yen(item['profit'])}"
            )
    else:
        lines.append("当選した回はまだありません。")
    lines.append("")
    lines.append("[年別成績]")
    if year_summary:
        for year, item in sorted(year_summary.items()):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"{year}: 対象回={item.get('target_draws')} / "
                f"購入={format_yen(item.get('total_cost'))} / "
                f"払戻={format_yen(item.get('total_payout'))} / "
                f"収支={format_yen(item.get('profit'))} / "
                f"収支率ROI={item.get('roi_percent')}% / "
                f"従来回収率={item.get('payout_roi_percent')}% / "
                f"的中率(口数)={item.get('ticket_hit_rate_percent')}% / "
                f"最大一致={item.get('max_main_match')}"
            )
    else:
        lines.append("年別成績なし")
    lines.append("")
    lines.append("[当せん金額データ欠損]")
    lines.append(f"欠損回数: {summary.get('missing_prize_draw_count')}")
    lines.append(f"欠損回: {summary.get('missing_prize_draws')}")
    lines.append("")
    lines.append("[出力ファイル]")
    lines.append(f"詳細CSV: {summary.get('detail_csv')}")
    lines.append(f"サマリーJSON: {summary.get('summary_json')}")
    lines.append(f"状態JSON: {summary.get('state_json')}")
    lines.append(f"テキストレポート: {report_path}")
    lines.append("")
    lines.append("注意: 宝くじはランダム性が高く、過去検証の成績は将来の当せんや利益を保証しません。")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_outputs(args: argparse.Namespace, *, genome, output_csv: Path, summary_json: Path, report_txt: Optional[Path], complete: bool, total_targets: int, state_path: Path) -> Dict[str, object]:
    summary = build_summary(args, genome=genome, output_csv=output_csv, summary_json=summary_json, report_txt=report_txt, complete=complete, total_targets=total_targets, state_path=state_path)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_txt is not None:
        write_text_report(summary, str(report_txt))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def evaluate_holdout(args: argparse.Namespace) -> int:
    start_monotonic = time.monotonic()
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    genome = load_best_model(args.best_model)
    if genome is None:
        raise SystemExit(f"best model not found or invalid: {args.best_model}")

    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    if not target_indices:
        raise SystemExit("no holdout targets selected")

    output_csv = Path(args.output)
    summary_json = Path(args.summary)
    report_txt = Path(args.report) if args.report else None
    state_path = Path(args.state)
    key = state_key(args, model_id=genome.id, model_score=genome.score)

    completed = completed_draws_from_csv(output_csv, key, args.resume, state_path)
    append = bool(completed)
    ensure_csv(output_csv, append=append)
    if completed:
        print(f"[RESUME] completed_draws={len(completed)} output={output_csv}")
    else:
        print(f"[START] total_targets={len(target_indices)} output={output_csv}")

    target_draw_nos = [draws[idx].draw_no for idx in target_indices]
    write_state(state_path, key=key, complete=False, completed_draws=sorted(completed), total_targets=len(target_indices))

    for idx in target_indices:
        target: Draw = draws[idx]
        if target.draw_no in completed:
            continue
        train = draws[:idx]
        tickets = generate_tickets(train, genome, args.purchase_count)
        prize_row = prize_rows.get(target.draw_no, {})
        y = draw_year(target)
        prize_missing = 0 if prize_row and has_any_prize_amount(prize_row) else 1
        rows: List[Dict[str, object]] = []
        for combo_index, ticket in enumerate(tickets, start=1):
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            payout = prize_amount_for_rank(prize_row, rank)
            cost = args.unit_cost
            rows.append(
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
                    "prize_data_missing": prize_missing,
                }
            )
        append_rows(output_csv, rows)
        completed.add(target.draw_no)
        write_state(state_path, key=key, complete=False, completed_draws=sorted(completed), total_targets=len(target_indices))

        if len(completed) % max(1, args.progress_every) == 0:
            print(f"[PROGRESS] completed={len(completed)}/{len(target_indices)} latest_draw={target.draw_no}")

        if should_safe_exit(start_monotonic, args.max_runtime_minutes, args.safe_exit_minutes):
            print(f"[SAFE EXIT] completed={len(completed)}/{len(target_indices)}. Resume next run with --resume.")
            write_summary_outputs(args, genome=genome, output_csv=output_csv, summary_json=summary_json, report_txt=report_txt, complete=False, total_targets=len(target_indices), state_path=state_path)
            return 0

    complete = set(target_draw_nos).issubset(completed)
    write_state(state_path, key=key, complete=complete, completed_draws=sorted(completed), total_targets=len(target_indices))
    summary = write_summary_outputs(args, genome=genome, output_csv=output_csv, summary_json=summary_json, report_txt=report_txt, complete=complete, total_targets=len(target_indices), state_path=state_path)

    if args.fail_on_missing_prize and complete and summary["missing_prize_draw_count"]:
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
    parser.add_argument("--report", default="outputs/holdout_report.txt")
    parser.add_argument("--state", default="outputs/holdout/holdout_state.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0)
    parser.add_argument("--safe-exit-minutes", type=float, default=10.0)
    parser.add_argument("--fail-on-missing-prize", action="store_true", help="当せん金額が未取得のholdout回があれば失敗扱いにする")
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.unit_cost <= 0:
        raise SystemExit("--unit-cost must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    if args.holdout_end_draw is not None and args.holdout_end_draw < args.holdout_start_draw:
        raise SystemExit("--holdout-end-draw must be >= --holdout-start-draw")
    return evaluate_holdout(args)


if __name__ == "__main__":
    raise SystemExit(main())
