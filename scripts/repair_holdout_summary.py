#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/repair_holdout_summary.py

holdout_result.csv を正として、holdout_summary.json の年別 target_draws=0 表示を補正し、
2020年以降など recent era の集計を追加する。

出力:
  outputs/holdout/holdout_summary.json        # 上書き補正
  outputs/holdout/recent_era_summary.json    # recent era専用JSON
  outputs/holdout/recent_era_report.txt      # recent era専用TXT

注意:
  過去検証上の集計であり、将来の当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, Set

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def draw_no_int(text: object) -> int | None:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def roi_from_profit(total_cost: int, total_payout: int) -> float:
    return (total_payout - total_cost) / total_cost if total_cost > 0 else 0.0


def roi_from_payout(total_cost: int, total_payout: int) -> float:
    return total_payout / total_cost if total_cost > 0 else 0.0


def format_yen(value: object) -> str:
    try:
        return f"{int(value):,}円"
    except Exception:
        return f"{value}円"


def empty_stats() -> Dict[str, object]:
    return {
        "target_draws": 0,
        "total_tickets": 0,
        "winning_ticket_count": 0,
        "winning_draw_count": 0,
        "ticket_hit_rate": 0.0,
        "ticket_hit_rate_percent": 0.0,
        "draw_hit_rate": 0.0,
        "draw_hit_rate_percent": 0.0,
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


def finalize(stats: Dict[str, object], draw_set: Set[int], winning_draws: Set[int]) -> Dict[str, object]:
    total_cost = int(stats.get("total_cost", 0))
    total_payout = int(stats.get("total_payout", 0))
    total_tickets = int(stats.get("total_tickets", 0))
    winning_tickets = int(stats.get("winning_ticket_count", 0))
    profit = total_payout - total_cost
    roi = roi_from_profit(total_cost, total_payout)
    payout_roi = roi_from_payout(total_cost, total_payout)
    ticket_hit_rate = pct(winning_tickets, total_tickets)
    draw_hit_rate = pct(len(winning_draws), len(draw_set))
    out = dict(stats)
    out.update(
        {
            "target_draws": len(draw_set),
            "winning_draw_count": len(winning_draws),
            "profit": profit,
            "roi": round(roi, 6),
            "roi_percent": round(roi * 100.0, 3),
            "payout_roi": round(payout_roi, 6),
            "payout_roi_percent": round(payout_roi * 100.0, 3),
            "ticket_hit_rate": round(ticket_hit_rate, 6),
            "ticket_hit_rate_percent": round(ticket_hit_rate * 100.0, 3),
            "draw_hit_rate": round(draw_hit_rate, 6),
            "draw_hit_rate_percent": round(draw_hit_rate * 100.0, 3),
        }
    )
    return out


def add_row(stats: Dict[str, object], row: Dict[str, str]) -> tuple[int | None, bool]:
    draw_no = draw_no_int(row.get("draw_no"))
    rank = str(row.get("rank") or "外れ")
    cost = int(row.get("purchase_cost") or 0)
    payout = int(row.get("prize_amount") or 0)
    main_match = int(row.get("main_match") or 0)
    stats["total_tickets"] = int(stats.get("total_tickets", 0)) + 1
    stats["total_cost"] = int(stats.get("total_cost", 0)) + cost
    stats["total_payout"] = int(stats.get("total_payout", 0)) + payout
    stats["max_main_match"] = max(int(stats.get("max_main_match", 0)), main_match)
    ranks = stats.get("rank_counts", {})
    if isinstance(ranks, dict):
        ranks[rank] = int(ranks.get(rank, 0)) + 1
    won = rank != "外れ"
    if won:
        stats["winning_ticket_count"] = int(stats.get("winning_ticket_count", 0)) + 1
    return draw_no, won


def build_from_detail(detail_csv: Path, recent_start_year: int) -> Dict[str, object]:
    year_stats: Dict[str, Dict[str, object]] = {}
    year_draws: Dict[str, Set[int]] = {}
    year_winning_draws: Dict[str, Set[int]] = {}
    recent_stats = empty_stats()
    recent_draws: Set[int] = set()
    recent_winning_draws: Set[int] = set()

    with detail_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = str(row.get("year") or "unknown")
            year_stats.setdefault(year, empty_stats())
            year_draws.setdefault(year, set())
            year_winning_draws.setdefault(year, set())
            draw_no, won = add_row(year_stats[year], row)
            if draw_no is not None:
                year_draws[year].add(draw_no)
                if won:
                    year_winning_draws[year].add(draw_no)
            try:
                y = int(year)
            except Exception:
                y = 0
            if y >= recent_start_year:
                recent_draw_no, recent_won = add_row(recent_stats, row)
                if recent_draw_no is not None:
                    recent_draws.add(recent_draw_no)
                    if recent_won:
                        recent_winning_draws.add(recent_draw_no)

    fixed_years = {
        year: finalize(stats, year_draws.get(year, set()), year_winning_draws.get(year, set()))
        for year, stats in sorted(year_stats.items())
    }
    recent = finalize(recent_stats, recent_draws, recent_winning_draws)
    recent["start_year"] = recent_start_year
    return {"year_summary": fixed_years, "recent_era_summary": recent}


def write_report(path: Path, payload: Dict[str, object]) -> None:
    recent = payload.get("recent_era_summary", {}) if isinstance(payload.get("recent_era_summary"), dict) else {}
    years = payload.get("year_summary", {}) if isinstance(payload.get("year_summary"), dict) else {}
    lines = [
        "LOTO7 Recent Era Holdout Report",
        "===============================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"detail_csv: {payload.get('detail_csv')}",
        f"recent_start_year: {recent.get('start_year')}",
        "",
        "[Recent Era Summary]",
        f"target_draws: {recent.get('target_draws')}",
        f"total_tickets: {recent.get('total_tickets')}",
        f"total_cost: {format_yen(recent.get('total_cost'))}",
        f"total_payout: {format_yen(recent.get('total_payout'))}",
        f"profit: {format_yen(recent.get('profit'))}",
        f"profit_roi_percent: {recent.get('roi_percent')}%",
        f"payout_roi_percent: {recent.get('payout_roi_percent')}%",
        f"ticket_hit_rate_percent: {recent.get('ticket_hit_rate_percent')}%",
        f"draw_hit_rate_percent: {recent.get('draw_hit_rate_percent')}%",
        f"max_main_match: {recent.get('max_main_match')}",
        f"rank_counts: {json.dumps(recent.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}",
        "",
        "[Year Summary Fixed]",
    ]
    for year, item in years.items():
        if not isinstance(item, dict):
            continue
        lines.append(
            f"{year}: target_draws={item.get('target_draws')} / payout_roi={item.get('payout_roi_percent')}% / "
            f"profit={format_yen(item.get('profit'))} / max_main={item.get('max_main_match')} / "
            f"hits={item.get('winning_ticket_count')}"
        )
    lines.append("")
    lines.append("注意: 過去検証上のrecent era評価であり、将来の当せんや利益を保証しません。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair holdout year summary and add recent era summary.")
    parser.add_argument("--detail", default="outputs/holdout/holdout_result.csv")
    parser.add_argument("--summary", default="outputs/holdout/holdout_summary.json")
    parser.add_argument("--recent-summary", default="outputs/holdout/recent_era_summary.json")
    parser.add_argument("--recent-report", default="outputs/holdout/recent_era_report.txt")
    parser.add_argument("--recent-start-year", type=int, default=2020)
    args = parser.parse_args()

    detail_path = Path(args.detail)
    summary_path = Path(args.summary)
    if not detail_path.exists():
        raise SystemExit(f"detail CSV not found: {detail_path}")
    if not summary_path.exists():
        raise SystemExit(f"summary JSON not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fixed = build_from_detail(detail_path, args.recent_start_year)
    summary["year_summary"] = fixed["year_summary"]
    summary["recent_era_summary"] = fixed["recent_era_summary"]
    summary["year_summary_repaired_at"] = now_iso()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    recent_payload = {
        "created_at": now_iso(),
        "kind": "loto7_recent_era_holdout_summary",
        "detail_csv": str(detail_path),
        "summary_json": str(summary_path),
        "recent_era_summary": fixed["recent_era_summary"],
        "year_summary": fixed["year_summary"],
    }
    recent_summary_path = Path(args.recent_summary)
    recent_summary_path.parent.mkdir(parents=True, exist_ok=True)
    recent_summary_path.write_text(json.dumps(recent_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(Path(args.recent_report), recent_payload)
    print(json.dumps(recent_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
