#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair holdout summaries from the detail CSV and add high-match metrics."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loto7.evaluation.core import RANK_ORDER, financial_metrics  # noqa: E402
from loto7.evaluation.hit_metrics import summarize_hit_metrics  # noqa: E402


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def draw_no_int(text: object) -> int | None:
    import re

    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else None


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
        "total_cost": 0,
        "total_payout": 0,
        "max_main_match": 0,
        "rank_counts": {rank: 0 for rank in RANK_ORDER},
    }


def finalize(
    stats: Dict[str, object],
    draw_set: Set[int],
    winning_draws: Set[int],
    draw_max_matches: Dict[int, int],
    ticket_matches: List[int],
) -> Dict[str, object]:
    out = dict(stats)
    out.update(
        financial_metrics(
            total_cost=int(stats.get("total_cost", 0)),
            total_payout=int(stats.get("total_payout", 0)),
            total_tickets=int(stats.get("total_tickets", 0)),
            winning_tickets=int(stats.get("winning_ticket_count", 0)),
            target_draws=len(draw_set),
            winning_draws=len(winning_draws),
        )
    )
    out["target_draws"] = len(draw_set)
    out["winning_draw_count"] = len(winning_draws)
    out.update(
        summarize_hit_metrics(
            [draw_max_matches[draw_no] for draw_no in sorted(draw_max_matches)],
            ticket_main_matches=ticket_matches,
        )
    )
    return out


def add_row(stats: Dict[str, object], row: Dict[str, str]) -> tuple[int | None, bool, int]:
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
    return draw_no, won, main_match


def build_from_detail(detail_csv: Path, recent_start_year: int) -> Dict[str, object]:
    year_stats: Dict[str, Dict[str, object]] = {}
    year_draws: Dict[str, Set[int]] = {}
    year_winning_draws: Dict[str, Set[int]] = {}
    year_draw_max: Dict[str, Dict[int, int]] = {}
    year_ticket_matches: Dict[str, List[int]] = {}
    recent_stats = empty_stats()
    recent_draws: Set[int] = set()
    recent_winning_draws: Set[int] = set()
    recent_draw_max: Dict[int, int] = {}
    recent_ticket_matches: List[int] = []

    with detail_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            year = str(row.get("year") or "unknown")
            year_stats.setdefault(year, empty_stats())
            year_draws.setdefault(year, set())
            year_winning_draws.setdefault(year, set())
            year_draw_max.setdefault(year, {})
            year_ticket_matches.setdefault(year, [])
            draw_no, won, main_match = add_row(year_stats[year], row)
            year_ticket_matches[year].append(main_match)
            if draw_no is not None:
                year_draws[year].add(draw_no)
                year_draw_max[year][draw_no] = max(year_draw_max[year].get(draw_no, 0), main_match)
                if won:
                    year_winning_draws[year].add(draw_no)
            try:
                numeric_year = int(year)
            except Exception:
                numeric_year = 0
            if numeric_year >= recent_start_year:
                recent_draw_no, recent_won, recent_main_match = add_row(recent_stats, row)
                recent_ticket_matches.append(recent_main_match)
                if recent_draw_no is not None:
                    recent_draws.add(recent_draw_no)
                    recent_draw_max[recent_draw_no] = max(
                        recent_draw_max.get(recent_draw_no, 0), recent_main_match
                    )
                    if recent_won:
                        recent_winning_draws.add(recent_draw_no)

    fixed_years = {
        year: finalize(
            stats,
            year_draws.get(year, set()),
            year_winning_draws.get(year, set()),
            year_draw_max.get(year, {}),
            year_ticket_matches.get(year, []),
        )
        for year, stats in sorted(year_stats.items())
    }
    recent = finalize(
        recent_stats,
        recent_draws,
        recent_winning_draws,
        recent_draw_max,
        recent_ticket_matches,
    )
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
        f"average_max_main_match: {recent.get('average_max_main_match')}",
        f"draw_main4_plus_rate_percent: {recent.get('draw_main4_plus_rate_percent')}%",
        f"draw_main5_plus_rate_percent: {recent.get('draw_main5_plus_rate_percent')}%",
        f"draw_main6_plus_rate_percent: {recent.get('draw_main6_plus_rate_percent')}%",
        f"hit_objective_score: {recent.get('hit_objective_score')}",
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
            f"main4+={item.get('draw_main4_plus_rate_percent')}% / "
            f"main5+={item.get('draw_main5_plus_rate_percent')}% / hits={item.get('winning_ticket_count')}"
        )
    lines.extend(
        [
            "",
            "注意: 過去検証上のrecent era評価であり、将来の当せんや利益を保証しません。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair holdout summaries and add accuracy-first metrics.")
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
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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
    recent_summary_path.write_text(
        json.dumps(recent_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(Path(args.recent_report), recent_payload)
    print(json.dumps(recent_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
