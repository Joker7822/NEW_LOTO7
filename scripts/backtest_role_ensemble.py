#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/backtest_role_ensemble.py

role_ensemble 5口構造の専用バックテスト。

各対象回について、直前回までのデータのみを使って以下を比較する。
  1) role_ensemble: 本命 / 高一致 / 直近 / 中高補正 / 荒れ目 の5口
  2) best_model: 採用ベストモデル単体の上位5口

再開対応:
  - CSVへ1対象回ずつ逐次追記
  - state JSONへ完了済み対象回を保存
  - タイムアウト前にsafe_exitし、次回は未処理分だけ続行

出力:
  outputs/role_ensemble/role_ensemble_backtest.csv
  outputs/role_ensemble/role_ensemble_summary.json
  outputs/role_ensemble/role_ensemble_report.txt
  outputs/role_ensemble/role_ensemble_state.json

注意:
  宝くじはランダム性が高く、当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, evaluate_ticket, generate_tickets, load_draws  # noqa: E402
from merge_evolution_shards import (  # noqa: E402
    PRIZE_RANKS,
    RANK_ORDER,
    fmt_ticket,
    load_model,
    load_prize_rows,
    make_role_ensemble_prediction_rows,
    prize_amount_for_rank,
    select_target_indices,
)

DETAIL_FIELDS = [
    "system",
    "role_key",
    "role_label",
    "target_draw_no",
    "target_date",
    "base_latest_draw_no",
    "ticket_index",
    "numbers",
    "main_match",
    "bonus_match",
    "rank",
    "payout",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_numbers(text: object) -> Tuple[int, ...]:
    raw = str(text or "").replace(",", " ").strip()
    nums = tuple(sorted(int(part) for part in raw.split() if part.isdigit()))
    if len(nums) != 7:
        raise ValueError(f"invalid ticket numbers: {text!r}")
    return nums


def role_key_from_row(row: Dict[str, object], fallback: str) -> str:
    model_id = str(row.get("model_id", ""))
    if ":" in model_id:
        return model_id.rsplit(":", 1)[-1]
    return fallback


def empty_rank_counts() -> Dict[str, int]:
    return {rank: 0 for rank in RANK_ORDER}


def new_stats() -> Dict[str, object]:
    return {
        "draw_count": 0,
        "draw_hit_count": 0,
        "total_tickets": 0,
        "total_cost": 0,
        "total_payout": 0,
        "profit": 0,
        "roi": 0.0,
        "roi_percent": 0.0,
        "ticket_hit_rate_percent": 0.0,
        "draw_hit_rate_percent": 0.0,
        "max_main_match": 0,
        "max_bonus_match": 0,
        "grade_hit_count": 0,
        "high_grade_hit_count": 0,
        "rank_counts": empty_rank_counts(),
    }


def update_ticket_stats(stats: Dict[str, object], *, unit_cost: int, payout: int, main_match: int, bonus_match: int, rank: str) -> None:
    stats["total_tickets"] = int(stats["total_tickets"]) + 1
    stats["total_cost"] = int(stats["total_cost"]) + unit_cost
    stats["total_payout"] = int(stats["total_payout"]) + payout
    stats["max_main_match"] = max(int(stats["max_main_match"]), int(main_match))
    stats["max_bonus_match"] = max(int(stats["max_bonus_match"]), int(bonus_match))
    ranks = stats["rank_counts"]
    if isinstance(ranks, dict):
        ranks[rank] = int(ranks.get(rank, 0)) + 1


def finalize_stats(stats: Dict[str, object]) -> Dict[str, object]:
    total_cost = int(stats.get("total_cost", 0))
    total_payout = int(stats.get("total_payout", 0))
    total_tickets = int(stats.get("total_tickets", 0))
    draw_count = int(stats.get("draw_count", 0))
    ranks = stats.get("rank_counts", {})
    rank_counts = {rank: int(ranks.get(rank, 0)) for rank in RANK_ORDER} if isinstance(ranks, dict) else empty_rank_counts()
    grade_hit_count = sum(rank_counts.get(rank, 0) for rank in PRIZE_RANKS)
    high_grade_hit_count = sum(rank_counts.get(rank, 0) for rank in ["1等", "2等", "3等", "4等"])
    profit = total_payout - total_cost
    roi = (total_payout / total_cost) if total_cost else 0.0
    ticket_hit_rate = (grade_hit_count / total_tickets * 100.0) if total_tickets else 0.0
    draw_hit_rate = (int(stats.get("draw_hit_count", 0)) / draw_count * 100.0) if draw_count else 0.0
    out = dict(stats)
    out.update(
        {
            "total_cost": total_cost,
            "total_payout": total_payout,
            "profit": profit,
            "roi": round(roi, 6),
            "roi_percent": round(roi * 100.0, 3),
            "ticket_hit_rate_percent": round(ticket_hit_rate, 3),
            "draw_hit_rate_percent": round(draw_hit_rate, 3),
            "grade_hit_count": grade_hit_count,
            "high_grade_hit_count": high_grade_hit_count,
            "rank_counts": rank_counts,
        }
    )
    return out


def compare(role: Dict[str, object], best: Dict[str, object]) -> Dict[str, object]:
    role_roi = float(role.get("roi_percent", 0.0))
    best_roi = float(best.get("roi_percent", 0.0))
    role_profit = int(role.get("profit", 0))
    best_profit = int(best.get("profit", 0))
    return {
        "roi_percent_delta": round(role_roi - best_roi, 3),
        "profit_delta": role_profit - best_profit,
        "grade_hit_delta": int(role.get("grade_hit_count", 0)) - int(best.get("grade_hit_count", 0)),
        "high_grade_hit_delta": int(role.get("high_grade_hit_count", 0)) - int(best.get("high_grade_hit_count", 0)),
        "max_main_match_delta": int(role.get("max_main_match", 0)) - int(best.get("max_main_match", 0)),
        "winner": "role_ensemble" if (role_profit, role_roi) > (best_profit, best_roi) else "best_model",
    }


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_detail_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists() and out.stat().st_size > 0
    with out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_detail_csv(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_detail_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def completed_draws_from_details(rows: Sequence[Dict[str, object]], purchase_count: int) -> Set[int]:
    counts: Dict[int, Dict[str, int]] = {}
    for row in rows:
        try:
            draw_no = int(row.get("target_draw_no", 0))
        except Exception:
            continue
        system = str(row.get("system", ""))
        if system not in {"role_ensemble", "best_model"}:
            continue
        counts.setdefault(draw_no, {"role_ensemble": 0, "best_model": 0})
        counts[draw_no][system] = counts[draw_no].get(system, 0) + 1
    return {
        draw_no
        for draw_no, by_system in counts.items()
        if by_system.get("role_ensemble", 0) >= purchase_count and by_system.get("best_model", 0) >= purchase_count
    }


def filter_complete_detail_rows(rows: Sequence[Dict[str, object]], completed_draws: Set[int]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        try:
            draw_no = int(row.get("target_draw_no", 0))
        except Exception:
            continue
        if draw_no in completed_draws:
            out.append(dict(row))
    return out


def update_stats_from_detail_rows(rows: Sequence[Dict[str, object]], unit_cost: int) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Dict[str, object]]]:
    role_system_stats = new_stats()
    best_system_stats = new_stats()
    role_stats: Dict[str, Dict[str, object]] = {}
    draw_hit: Dict[Tuple[str, int], bool] = {}
    role_draw_hit: Dict[Tuple[str, int], bool] = {}
    draw_seen: Set[Tuple[str, int]] = set()
    role_draw_seen: Set[Tuple[str, int]] = set()

    for row in rows:
        system = str(row.get("system", ""))
        if system not in {"role_ensemble", "best_model"}:
            continue
        try:
            draw_no = int(row.get("target_draw_no", 0))
            payout = int(row.get("payout", 0))
            main_match = int(row.get("main_match", 0))
            bonus_match = int(row.get("bonus_match", 0))
        except Exception:
            continue
        rank = str(row.get("rank", "外れ")) or "外れ"
        stats = role_system_stats if system == "role_ensemble" else best_system_stats
        sys_key = (system, draw_no)
        if sys_key not in draw_seen:
            stats["draw_count"] = int(stats["draw_count"]) + 1
            draw_seen.add(sys_key)
            draw_hit[sys_key] = False
        update_ticket_stats(stats, unit_cost=unit_cost, payout=payout, main_match=main_match, bonus_match=bonus_match, rank=rank)
        if rank != "外れ":
            draw_hit[sys_key] = True

        if system == "role_ensemble":
            role_key = str(row.get("role_key", "unknown"))
            rstats = role_stats.setdefault(role_key, new_stats())
            rkey = (role_key, draw_no)
            if rkey not in role_draw_seen:
                rstats["draw_count"] = int(rstats["draw_count"]) + 1
                role_draw_seen.add(rkey)
                role_draw_hit[rkey] = False
            update_ticket_stats(rstats, unit_cost=unit_cost, payout=payout, main_match=main_match, bonus_match=bonus_match, rank=rank)
            if rank != "外れ":
                role_draw_hit[rkey] = True

    for (system, draw_no), hit in draw_hit.items():
        if hit:
            stats = role_system_stats if system == "role_ensemble" else best_system_stats
            stats["draw_hit_count"] = int(stats["draw_hit_count"]) + 1
    for (role_key, draw_no), hit in role_draw_hit.items():
        if hit:
            role_stats[role_key]["draw_hit_count"] = int(role_stats[role_key]["draw_hit_count"]) + 1

    return finalize_stats(role_system_stats), finalize_stats(best_system_stats), {k: finalize_stats(v) for k, v in sorted(role_stats.items())}


def write_report(path: str, summary: Dict[str, object]) -> None:
    role = summary.get("role_ensemble", {}) if isinstance(summary.get("role_ensemble"), dict) else {}
    best = summary.get("best_model", {}) if isinstance(summary.get("best_model"), dict) else {}
    comparison = summary.get("comparison", {}) if isinstance(summary.get("comparison"), dict) else {}
    roles = summary.get("roles", {}) if isinstance(summary.get("roles"), dict) else {}

    lines: List[str] = []
    lines.append("LOTO7 Role Ensemble Backtest Report")
    lines.append("===================================")
    lines.append("")
    lines.append(f"created_at: {summary.get('created_at')}")
    lines.append(f"status: {summary.get('status')}")
    lines.append(f"model: {summary.get('best_model_path')}")
    lines.append(f"genome_id: {summary.get('genome_id')}")
    lines.append(f"target_draws_total: {summary.get('target_draws_total')}")
    lines.append(f"completed_target_draws: {summary.get('completed_target_draws')}")
    lines.append(f"last_completed_draw_no: {summary.get('last_completed_draw_no')}")
    lines.append(f"purchase_count: {summary.get('purchase_count')}")
    lines.append(f"holdout_start_draw: {summary.get('holdout_start_draw')}")
    lines.append(f"min_train_draws: {summary.get('min_train_draws')}")
    lines.append("")
    lines.append("[Role Ensemble]")
    lines.append(f"roi_percent: {role.get('roi_percent')}")
    lines.append(f"profit: {role.get('profit')}")
    lines.append(f"grade_hit_count: {role.get('grade_hit_count')}")
    lines.append(f"high_grade_hit_count: {role.get('high_grade_hit_count')}")
    lines.append(f"max_main_match: {role.get('max_main_match')}")
    lines.append(f"draw_hit_rate_percent: {role.get('draw_hit_rate_percent')}")
    lines.append(f"rank_counts: {json.dumps(role.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}")
    lines.append("")
    lines.append("[Best Model Top 5]")
    lines.append(f"roi_percent: {best.get('roi_percent')}")
    lines.append(f"profit: {best.get('profit')}")
    lines.append(f"grade_hit_count: {best.get('grade_hit_count')}")
    lines.append(f"high_grade_hit_count: {best.get('high_grade_hit_count')}")
    lines.append(f"max_main_match: {best.get('max_main_match')}")
    lines.append(f"draw_hit_rate_percent: {best.get('draw_hit_rate_percent')}")
    lines.append(f"rank_counts: {json.dumps(best.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}")
    lines.append("")
    lines.append("[Comparison]")
    lines.append(f"winner: {comparison.get('winner')}")
    lines.append(f"roi_percent_delta: {comparison.get('roi_percent_delta')}")
    lines.append(f"profit_delta: {comparison.get('profit_delta')}")
    lines.append(f"grade_hit_delta: {comparison.get('grade_hit_delta')}")
    lines.append(f"high_grade_hit_delta: {comparison.get('high_grade_hit_delta')}")
    lines.append(f"max_main_match_delta: {comparison.get('max_main_match_delta')}")
    lines.append("")
    lines.append("[Role Breakdown]")
    for role_key, stats in roles.items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            f"{role_key}: ROI={stats.get('roi_percent')}% / profit={stats.get('profit')} / "
            f"grade={stats.get('grade_hit_count')} / high_grade={stats.get('high_grade_hit_count')} / "
            f"max_main={stats.get('max_main_match')} / ranks={json.dumps(stats.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}"
        )
    lines.append("")
    lines.append("注意: 過去検証上の比較であり、将来の当せんや利益を保証しません。")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_rows_for_draw(
    *,
    system: str,
    rows: Iterable[Dict[str, object]],
    target: Draw,
    prize_row: Dict[str, str],
    detail_rows: List[Dict[str, object]],
) -> None:
    for idx, row in enumerate(rows, start=1):
        nums = parse_numbers(row.get("numbers"))
        main_match, bonus_match, rank = evaluate_ticket(nums, target)
        payout = prize_amount_for_rank(prize_row, rank)
        role_key = str(row.get("role_key") or role_key_from_row(row, f"rank_{idx}"))
        role_label = str(row.get("support_models") or row.get("role_label") or role_key)
        detail_rows.append(
            {
                "system": system,
                "role_key": role_key,
                "role_label": role_label,
                "target_draw_no": target.draw_no,
                "target_date": target.date,
                "base_latest_draw_no": int(row.get("base_latest_draw_no", target.draw_no - 1)),
                "ticket_index": idx,
                "numbers": fmt_ticket(nums),
                "main_match": main_match,
                "bonus_match": bonus_match,
                "rank": rank,
                "payout": payout,
            }
        )


def state_matches(state: Dict[str, object], args: argparse.Namespace, genome_id: str, target_draw_nos: Sequence[int]) -> bool:
    return (
        state.get("csv") == args.csv
        and state.get("best_model") == args.best_model
        and state.get("genome_id") == genome_id
        and int(state.get("purchase_count", -1)) == int(args.purchase_count)
        and int(state.get("unit_cost", -1)) == int(args.unit_cost)
        and int(state.get("holdout_start_draw", -1)) == int(args.holdout_start_draw)
        and int(state.get("min_train_draws", -1)) == int(args.min_train_draws)
        and int(state.get("target_draws_total", -1)) == len(target_draw_nos)
    )


def load_resume_details(args: argparse.Namespace, genome_id: str, target_draw_nos: Sequence[int]) -> Tuple[List[Dict[str, object]], Set[int], List[int]]:
    if not args.resume:
        return [], set(), []
    state_path = Path(args.state)
    if not state_path.exists() or state_path.stat().st_size <= 0:
        return [], set(), []
    try:
        state = read_json(args.state)
    except Exception as exc:
        print(f"[WARN] cannot read state; starting fresh: {exc}")
        return [], set(), []
    if not state_matches(state, args, genome_id, target_draw_nos):
        print("[INFO] role ensemble state does not match current settings; starting fresh")
        return [], set(), []
    detail_rows = read_detail_csv(args.output)
    completed = completed_draws_from_details(detail_rows, args.purchase_count)
    cleaned = filter_complete_detail_rows(detail_rows, completed)
    if len(cleaned) != len(detail_rows):
        write_detail_csv(args.output, cleaned)
    missing = [int(x) for x in state.get("missing_prize_draws", []) if str(x).isdigit()]
    print(f"[RESUME] completed target draws: {len(completed)}/{len(target_draw_nos)}")
    return cleaned, completed, missing


def build_summary(
    *,
    args: argparse.Namespace,
    status: str,
    genome_id: str,
    target_indices: Sequence[int],
    draws: Sequence[Draw],
    detail_rows: Sequence[Dict[str, object]],
    missing_prize_draws: Sequence[int],
) -> Dict[str, object]:
    completed = completed_draws_from_details(detail_rows, args.purchase_count)
    role_summary, best_summary, roles_summary = update_stats_from_detail_rows(detail_rows, args.unit_cost)
    last_completed = max(completed) if completed else None
    return {
        "created_at": now_iso(),
        "status": status,
        "csv": args.csv,
        "best_model_path": args.best_model,
        "genome_id": genome_id,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "holdout_start_draw": args.holdout_start_draw,
        "holdout_end_draw": args.holdout_end_draw,
        "min_train_draws": args.min_train_draws,
        "max_targets": args.max_targets,
        "target_draws_total": len(target_indices),
        "completed_target_draws": len(completed),
        "first_target_draw_no": draws[target_indices[0]].draw_no if target_indices else None,
        "last_target_draw_no": draws[target_indices[-1]].draw_no if target_indices else None,
        "last_completed_draw_no": last_completed,
        "missing_prize_draw_count": len(set(missing_prize_draws)),
        "missing_prize_draws": sorted(set(missing_prize_draws)),
        "role_ensemble": role_summary,
        "best_model": best_summary,
        "comparison": compare(role_summary, best_summary),
        "roles": roles_summary,
        "output": args.output,
        "summary": args.summary,
        "report": args.report,
        "state": args.state,
    }


def save_state(
    *,
    args: argparse.Namespace,
    status: str,
    genome_id: str,
    target_draw_nos: Sequence[int],
    completed_draws: Set[int],
    missing_prize_draws: Sequence[int],
) -> None:
    payload = {
        "updated_at": now_iso(),
        "status": status,
        "csv": args.csv,
        "best_model": args.best_model,
        "genome_id": genome_id,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "holdout_start_draw": args.holdout_start_draw,
        "holdout_end_draw": args.holdout_end_draw,
        "min_train_draws": args.min_train_draws,
        "max_targets": args.max_targets,
        "target_draws_total": len(target_draw_nos),
        "completed_target_draws": len(completed_draws),
        "last_completed_draw_no": max(completed_draws) if completed_draws else None,
        "completed_draws": sorted(completed_draws),
        "missing_prize_draws": sorted(set(int(x) for x in missing_prize_draws)),
        "output": args.output,
        "summary": args.summary,
        "report": args.report,
    }
    write_json(args.state, payload)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest LOTO7 role_ensemble predictions against best_model top5.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=0, help="Use 0 for all selected targets; otherwise evaluate the most recent N targets.")
    parser.add_argument("--overlap-limit", type=int, default=4)
    parser.add_argument("--output", default="outputs/role_ensemble/role_ensemble_backtest.csv")
    parser.add_argument("--summary", default="outputs/role_ensemble/role_ensemble_summary.json")
    parser.add_argument("--report", default="outputs/role_ensemble/role_ensemble_report.txt")
    parser.add_argument("--state", default="outputs/role_ensemble/role_ensemble_state.json")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-runtime-minutes", type=float, default=320.0)
    parser.add_argument("--safe-exit-minutes", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.unit_cost <= 0:
        raise SystemExit("--unit-cost must be positive")
    if args.overlap_limit < 0 or args.overlap_limit > 7:
        raise SystemExit("--overlap-limit must be between 0 and 7")

    started = time.monotonic()
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    model_item = load_model(Path(args.best_model))
    if model_item is None:
        raise SystemExit(f"cannot load best model: {args.best_model}")
    genome = model_item["genome"]
    genome_id = str(getattr(genome, "id", ""))

    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    if args.max_targets and args.max_targets > 0:
        target_indices = target_indices[-args.max_targets :]
    target_indices = [idx for idx in target_indices if idx > 0]
    if not target_indices:
        raise SystemExit("no targets selected")
    target_draw_nos = [draws[idx].draw_no for idx in target_indices]

    detail_rows, completed_draws, missing_prize_draws = load_resume_details(args, genome_id, target_draw_nos)
    if not args.resume and Path(args.output).exists():
        Path(args.output).unlink()

    status = "completed"
    processed_this_run = 0
    for pos, idx in enumerate(target_indices, start=1):
        target = draws[idx]
        if target.draw_no in completed_draws:
            continue
        elapsed = (time.monotonic() - started) / 60.0
        if elapsed >= max(0.0, args.max_runtime_minutes - args.safe_exit_minutes):
            status = f"safe_exit_at_{elapsed:.2f}_minutes"
            break

        train = draws[:idx]
        prize_row = prize_rows.get(target.draw_no, {})
        if not prize_row:
            missing_prize_draws.append(target.draw_no)

        new_rows: List[Dict[str, object]] = []
        role_rows = make_role_ensemble_prediction_rows(
            genome, str(model_item.get("path", args.best_model)), train, args.purchase_count, args.overlap_limit
        )
        evaluate_rows_for_draw(system="role_ensemble", rows=role_rows, target=target, prize_row=prize_row, detail_rows=new_rows)

        best_rows = []
        for ticket_index, ticket in enumerate(generate_tickets(train, genome, args.purchase_count), start=1):
            best_rows.append(
                {
                    "numbers": fmt_ticket(ticket),
                    "model_id": f"{genome_id}:best_rank_{ticket_index}",
                    "role_key": f"best_rank_{ticket_index}",
                    "support_models": "best_model_top5",
                    "base_latest_draw_no": train[-1].draw_no,
                }
            )
        evaluate_rows_for_draw(system="best_model", rows=best_rows, target=target, prize_row=prize_row, detail_rows=new_rows)

        append_detail_csv(args.output, new_rows)
        detail_rows.extend(new_rows)
        completed_draws.add(target.draw_no)
        processed_this_run += 1

        if args.progress_every > 0 and processed_this_run % args.progress_every == 0:
            print(f"[PROGRESS] completed {len(completed_draws)}/{len(target_indices)} target draws; latest={target.draw_no}")
        save_state(
            args=args,
            status="running",
            genome_id=genome_id,
            target_draw_nos=target_draw_nos,
            completed_draws=completed_draws,
            missing_prize_draws=missing_prize_draws,
        )

    if len(completed_draws) >= len(target_indices):
        status = "completed"

    summary = build_summary(
        args=args,
        status=status,
        genome_id=genome_id,
        target_indices=target_indices,
        draws=draws,
        detail_rows=detail_rows,
        missing_prize_draws=missing_prize_draws,
    )
    write_json(args.summary, summary)
    write_report(args.report, summary)
    save_state(
        args=args,
        status=status,
        genome_id=genome_id,
        target_draw_nos=target_draw_nos,
        completed_draws=completed_draws,
        missing_prize_draws=missing_prize_draws,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
