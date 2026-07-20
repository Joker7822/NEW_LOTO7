#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical LOTO7 prize loading and financial evaluation utilities.

All production, holdout and diagnostic evaluators should use these definitions.
Legacy ``roi`` aliases are intentionally defined as profit ROI:
    (total_payout - total_cost) / total_cost
The payout ratio is exposed separately as ``payout_roi``.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

EVALUATOR_VERSION = "loto7-evaluator-2026.07.18-v1"
RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]
HIGH_GRADE_RANKS = ["1等", "2等", "3等", "4等"]


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_no_int(text: object) -> Optional[int]:
    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else None


def parse_money_yen(text: object) -> int:
    raw = str(text or "").strip()
    if not raw or raw == "該当なし":
        return 0
    match = re.search(r"([0-9,]+)", raw)
    return int(match.group(1).replace(",", "")) if match else 0


def load_prize_rows(csv_path: str | Path) -> Dict[int, Dict[str, str]]:
    output: Dict[int, Dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            draw_no = draw_no_int(row.get("回別"))
            if draw_no is not None:
                output[draw_no] = {key: str(value or "").strip() for key, value in row.items()}
    return output


def has_any_prize_amount(row: Mapping[str, object]) -> bool:
    return any(str(row.get(f"{rank}当選金額", "")).strip() for rank in PRIZE_RANKS)


def prize_amount_for_rank(row: Mapping[str, object], rank: str) -> int:
    if rank == "外れ":
        return 0
    return parse_money_yen(row.get(f"{rank}当選金額", ""))


def profit_roi(total_cost: int, total_payout: int) -> float:
    return (total_payout - total_cost) / total_cost if total_cost > 0 else 0.0


def payout_roi(total_cost: int, total_payout: int) -> float:
    return total_payout / total_cost if total_cost > 0 else 0.0


def financial_metrics(
    *,
    total_cost: int,
    total_payout: int,
    total_tickets: int = 0,
    winning_tickets: int = 0,
    target_draws: int = 0,
    winning_draws: int = 0,
) -> Dict[str, object]:
    profit = int(total_payout) - int(total_cost)
    profit_ratio = profit_roi(int(total_cost), int(total_payout))
    payout_ratio = payout_roi(int(total_cost), int(total_payout))
    ticket_hit = winning_tickets / total_tickets if total_tickets > 0 else 0.0
    draw_hit = winning_draws / target_draws if target_draws > 0 else 0.0
    return {
        "total_cost": int(total_cost),
        "total_payout": int(total_payout),
        "profit": profit,
        "roi": round(profit_ratio, 6),
        "roi_percent": round(profit_ratio * 100.0, 3),
        "profit_roi": round(profit_ratio, 6),
        "profit_roi_percent": round(profit_ratio * 100.0, 3),
        "payout_roi": round(payout_ratio, 6),
        "payout_roi_percent": round(payout_ratio * 100.0, 3),
        "ticket_hit_rate": round(ticket_hit, 6),
        "ticket_hit_rate_percent": round(ticket_hit * 100.0, 3),
        "draw_hit_rate": round(draw_hit, 6),
        "draw_hit_rate_percent": round(draw_hit * 100.0, 3),
        "evaluator_version": EVALUATOR_VERSION,
    }


def finalize_stats(stats: Mapping[str, object]) -> Dict[str, object]:
    ranks_value = stats.get("rank_counts", {})
    ranks = ranks_value if isinstance(ranks_value, Mapping) else {}
    rank_counts = {rank: int(ranks.get(rank, 0) or 0) for rank in RANK_ORDER}
    winning_tickets = sum(rank_counts[rank] for rank in PRIZE_RANKS)
    high_grade = sum(rank_counts[rank] for rank in HIGH_GRADE_RANKS)
    output = dict(stats)
    output.update(
        financial_metrics(
            total_cost=int(stats.get("total_cost", 0) or 0),
            total_payout=int(stats.get("total_payout", 0) or 0),
            total_tickets=int(stats.get("total_tickets", 0) or 0),
            winning_tickets=winning_tickets,
            target_draws=int(stats.get("draw_count", 0) or 0),
            winning_draws=int(stats.get("draw_hit_count", 0) or 0),
        )
    )
    output["grade_hit_count"] = winning_tickets
    output["high_grade_hit_count"] = high_grade
    output["rank_counts"] = rank_counts
    return output
