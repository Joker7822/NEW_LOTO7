"""Chronological, payout-aware model audit used by hardened promotion gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.metrics_schema import financial_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evolution.hit_first import hit_first_score
from loto7_evolution_trainer import evaluate_ticket, generate_tickets
from merge_evolution_shards import has_any_prize_amount, prize_amount_for_rank
from scripts.robust_model_metrics import load_genome


def year_of(draw: object) -> int:
    text = str(getattr(draw, "date", ""))
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 0


def fold_indices(indices: Sequence[int], count: int = 5) -> list[list[int]]:
    ordered = list(indices)
    quotient, remainder = divmod(len(ordered), count)
    result = []
    cursor = 0
    for position in range(count):
        size = quotient + (1 if position < remainder else 0)
        result.append(ordered[cursor : cursor + size])
        cursor += size
    return [part for part in result if part]


def evaluate_model(
    *,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Mapping[int, Mapping[str, str]],
    target_indices: Sequence[int],
    purchase_count: int = 5,
    unit_cost: int = 300,
) -> dict[str, object]:
    genome = load_genome(model_path)
    portfolios = []
    mains = []
    maximums: list[int] = []
    ticket_matches: list[int] = []
    draw_outcomes = []
    payouts = []
    missing_prizes = []
    for index in target_indices:
        target = draws[index]
        tickets = generate_tickets(draws[:index], genome, purchase_count)
        portfolios.append(tickets)
        mains.append(target.main)
        row = prize_rows.get(int(target.draw_no), {})
        if not row or not has_any_prize_amount(row):
            missing_prizes.append(int(target.draw_no))
        draw_payout = 0
        matches = []
        for ticket in tickets:
            main_match, _bonus_match, rank = evaluate_ticket(ticket, target)
            matches.append(main_match)
            ticket_matches.append(main_match)
            draw_payout += prize_amount_for_rank(row, rank)
        maximum = max(matches, default=0)
        maximums.append(maximum)
        payouts.append(draw_payout)
        draw_outcomes.append(
            {
                "draw_no": int(target.draw_no),
                "max_main_match": maximum,
                "draw4_plus": maximum >= 4,
            }
        )
    total_cost = len(target_indices) * purchase_count * unit_cost
    total_payout = sum(payouts)
    metrics = financial_metrics(total_cost, total_payout)
    metrics.update(
        summarize_hit_metrics(
            maximums,
            ticket_main_matches=ticket_matches,
            portfolios=portfolios,
        )
    )
    metrics.update(summarize_portfolio_ranking(portfolios, mains))
    top1 = max(payouts, default=0)
    metrics.update(
        {
            "path": model_path,
            "target_draws": len(target_indices),
            "purchase_count": purchase_count,
            "unit_cost": unit_cost,
            "top1_payout_share": round(top1 / total_payout, 6) if total_payout else 0.0,
            "missing_prize_draw_count": len(set(missing_prizes)),
            "missing_prize_draws": sorted(set(missing_prizes)),
            "draw_outcomes": draw_outcomes,
        }
    )
    metrics["hit_first_objective_score"] = hit_first_score(metrics)
    return metrics
