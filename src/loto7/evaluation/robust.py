#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust historical metrics for LOTO7 models.

Financial stability remains a safety diagnostic. Payout-independent high-match
metrics are calculated alongside it and are intended to drive model comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import (  # noqa: E402
    Draw,
    Genome,
    evaluate_ticket,
    generate_tickets,
    genome_from_dict,
    load_draws,
)
from merge_evolution_shards import select_target_indices  # noqa: E402
from loto7.evaluation.core import (  # noqa: E402
    HIGH_GRADE_RANKS,
    PRIZE_RANKS,
    RANK_ORDER,
    has_any_prize_amount,
    load_prize_rows,
    prize_amount_for_rank,
)
from loto7.evaluation.hit_metrics import summarize_hit_metrics  # noqa: E402


def draw_year(draw: object) -> int:
    match = re.match(r"^(\d{4})", str(getattr(draw, "date", "") or ""))
    return int(match.group(1)) if match else 0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_genome(path: str) -> Genome:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("genome", payload)
    if not isinstance(raw, dict):
        raise SystemExit(f"invalid genome payload: {path}")
    return genome_from_dict(raw)


def indices_for_years(
    draws: Sequence[Draw],
    base_indices: Sequence[int],
    start_year: Optional[int],
    end_year: Optional[int],
) -> List[int]:
    selected: List[int] = []
    for index in base_indices:
        year = draw_year(draws[index])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(index)
    return selected


def _roi_percent(payout: int, cost: int) -> float:
    return round((payout / cost * 100.0) if cost else 0.0, 3)


def evaluate_model_robust(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[Draw],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 777,
    include_draw_records: bool = False,
) -> Dict[str, object]:
    rank_counts = {rank: 0 for rank in RANK_ORDER}
    draw_records: List[Dict[str, object]] = []
    missing_prize_draws: List[int] = []
    max_main_match = 0
    max_bonus_match = 0
    ticket_main_matches: List[int] = []
    portfolios: List[Sequence[Sequence[int]]] = []

    for index in target_indices:
        target = draws[index]
        tickets = generate_tickets(draws[:index], genome, purchase_count)
        portfolios.append(tickets)
        prize_row = prize_rows.get(target.draw_no, {})
        if not prize_row or not has_any_prize_amount(prize_row):
            missing_prize_draws.append(target.draw_no)

        draw_payout = 0
        draw_cost = 0
        draw_max_main = 0
        draw_max_bonus = 0
        for ticket in tickets:
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            ticket_main_matches.append(main_match)
            payout = prize_amount_for_rank(prize_row, rank)
            draw_cost += unit_cost
            draw_payout += payout
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            draw_max_main = max(draw_max_main, main_match)
            draw_max_bonus = max(draw_max_bonus, bonus_match)

        max_main_match = max(max_main_match, draw_max_main)
        max_bonus_match = max(max_bonus_match, draw_max_bonus)
        draw_records.append(
            {
                "draw_no": target.draw_no,
                "date": target.date,
                "year": draw_year(target),
                "cost": draw_cost,
                "payout": draw_payout,
                "profit": draw_payout - draw_cost,
                "max_main_match": draw_max_main,
                "max_bonus_match": draw_max_bonus,
            }
        )

    total_cost = sum(int(record["cost"]) for record in draw_records)
    total_payout = sum(int(record["payout"]) for record in draw_records)
    total_profit = total_payout - total_cost
    payouts = sorted((int(record["payout"]) for record in draw_records), reverse=True)
    top1 = payouts[0] if payouts else 0
    top2 = sum(payouts[:2])
    payout_share_top1 = (top1 / total_payout) if total_payout else 0.0
    payout_hhi = sum((payout / total_payout) ** 2 for payout in payouts) if total_payout else 0.0

    yearly: Dict[int, Dict[str, int]] = {}
    for record in draw_records:
        year = int(record["year"])
        bucket = yearly.setdefault(year, {"cost": 0, "payout": 0})
        bucket["cost"] += int(record["cost"])
        bucket["payout"] += int(record["payout"])
    yearly_roi = {
        str(year): _roi_percent(values["payout"], values["cost"])
        for year, values in sorted(yearly.items())
        if year > 0
    }
    yearly_values = list(yearly_roi.values())

    profits = sorted(int(record["profit"]) for record in draw_records)
    cvar_count = max(1, int(math.ceil(len(profits) * 0.20))) if profits else 0
    cvar20 = statistics.fmean(profits[:cvar_count]) if cvar_count else 0.0

    bootstrap_values: List[float] = []
    if draw_records and bootstrap_samples > 0:
        rng = random.Random(bootstrap_seed)
        sample_size = len(draw_records)
        for _ in range(bootstrap_samples):
            sampled = [draw_records[rng.randrange(sample_size)] for _ in range(sample_size)]
            sampled_cost = sum(int(record["cost"]) for record in sampled)
            sampled_payout = sum(int(record["payout"]) for record in sampled)
            bootstrap_values.append((sampled_payout / sampled_cost * 100.0) if sampled_cost else 0.0)

    hit_metrics = summarize_hit_metrics(
        [int(record["max_main_match"]) for record in draw_records],
        ticket_main_matches=ticket_main_matches,
        portfolios=portfolios,
    )
    result: Dict[str, object] = {
        "path": model_path,
        "genome_id": genome.id,
        "target_draws": len(draw_records),
        "purchase_count": purchase_count,
        "unit_cost": unit_cost,
        "total_tickets": len(draw_records) * purchase_count,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": total_profit,
        "roi": round((total_payout / total_cost) if total_cost else 0.0, 6),
        "roi_percent": _roi_percent(total_payout, total_cost),
        "profit_roi_percent": round((total_profit / total_cost * 100.0) if total_cost else 0.0, 3),
        "roi_excluding_top1_percent": _roi_percent(total_payout - top1, total_cost),
        "roi_excluding_top2_percent": _roi_percent(total_payout - top2, total_cost),
        "largest_draw_payout": top1,
        "largest_two_draw_payout": top2,
        "top1_payout_share": round(payout_share_top1, 6),
        "payout_hhi": round(payout_hhi, 6),
        "median_year_roi_percent": round(statistics.median(yearly_values), 3) if yearly_values else 0.0,
        "worst_year_roi_percent": round(min(yearly_values), 3) if yearly_values else 0.0,
        "positive_year_count": sum(1 for value in yearly_values if value >= 100.0),
        "year_count": len(yearly_values),
        "yearly_roi_percent": yearly_roi,
        "cvar20_profit_per_draw": round(cvar20, 3),
        "bootstrap_roi_percent_p05": round(percentile(bootstrap_values, 0.05), 3),
        "bootstrap_roi_percent_p50": round(percentile(bootstrap_values, 0.50), 3),
        "bootstrap_roi_percent_p95": round(percentile(bootstrap_values, 0.95), 3),
        "max_main_match": max_main_match,
        "max_bonus_match": max_bonus_match,
        "rank_counts": rank_counts,
        "grade_hit_count": sum(rank_counts.get(rank, 0) for rank in PRIZE_RANKS),
        "high_grade_hit_count": sum(rank_counts.get(rank, 0) for rank in HIGH_GRADE_RANKS),
        "missing_prize_draw_count": len(set(missing_prize_draws)),
        "missing_prize_draws": sorted(set(missing_prize_draws)),
        **hit_metrics,
    }
    if include_draw_records:
        result["draw_records"] = draw_records
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build robust payout and high-match metrics for a LOTO7 model.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base_indices = select_target_indices(
        draws,
        min_train_draws=1,
        holdout_start_draw=2,
        holdout_end_draw=None,
    )
    target_indices = indices_for_years(draws, base_indices, args.start_year, args.end_year)
    if not target_indices:
        raise SystemExit("no target draws selected")
    genome = load_genome(args.model)
    metrics = evaluate_model_robust(
        genome=genome,
        model_path=args.model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
