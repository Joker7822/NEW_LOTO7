#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LOTO7 null-strategy league and probability-of-backtest-overfitting diagnostic."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws  # noqa: E402
from merge_evolution_shards import load_prize_rows, prize_amount_for_rank, select_target_indices  # noqa: E402
from scripts.robust_model_metrics import evaluate_model_robust, indices_for_years, load_genome, percentile  # noqa: E402

Ticket = Tuple[int, ...]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def overlap(left: Ticket, right: Ticket) -> int:
    return len(set(left) & set(right))


def weighted_sample_without_replacement(rng: random.Random, weights: Mapping[int, float], count: int = 7) -> Ticket:
    available = list(range(1, 38))
    output: List[int] = []
    for _ in range(count):
        total = sum(max(1e-12, float(weights.get(number, 0.0))) for number in available)
        point = rng.random() * total
        cumulative = 0.0
        chosen = available[-1]
        for number in available:
            cumulative += max(1e-12, float(weights.get(number, 0.0)))
            if cumulative >= point:
                chosen = number
                break
        output.append(chosen)
        available.remove(chosen)
    return tuple(sorted(output))


def frequency_weights(draws: Sequence[Draw], lookback: int = 104) -> Dict[int, float]:
    counts = Counter(number for draw in draws[-lookback:] for number in draw.main)
    return {number: counts[number] + 1.0 for number in range(1, 38)}


def dormancy_weights(draws: Sequence[Draw]) -> Dict[int, float]:
    gap = {number: len(draws) + 1 for number in range(1, 38)}
    for age, draw in enumerate(reversed(draws)):
        for number in draw.main:
            gap[number] = min(gap[number], age + 1)
    return {number: float(gap[number] + 1) for number in range(1, 38)}


def recent_weights(draws: Sequence[Draw]) -> Dict[int, float]:
    counts = Counter(number for draw in draws[-52:] for number in draw.main)
    return {number: counts[number] * 2.0 + 1.0 for number in range(1, 38)}


def balanced_ticket(rng: random.Random) -> Ticket:
    for _ in range(500):
        ticket = tuple(sorted(rng.sample(range(1, 38), 7)))
        odd = sum(1 for number in ticket if number % 2)
        total = sum(ticket)
        span = ticket[-1] - ticket[0]
        if 3 <= odd <= 4 and 105 <= total <= 175 and span >= 23:
            return ticket
    return tuple(sorted(rng.sample(range(1, 38), 7)))


def strategy_ticket(train: Sequence[Draw], strategy: str, rng: random.Random) -> Ticket:
    if strategy == "random":
        return tuple(sorted(rng.sample(range(1, 38), 7)))
    if strategy == "balanced":
        return balanced_ticket(rng)
    if strategy == "frequency":
        return weighted_sample_without_replacement(rng, frequency_weights(train))
    if strategy == "dormancy":
        return weighted_sample_without_replacement(rng, dormancy_weights(train))
    if strategy == "recent":
        return weighted_sample_without_replacement(rng, recent_weights(train))
    if strategy == "hybrid":
        freq = frequency_weights(train)
        dormancy = dormancy_weights(train)
        recent = recent_weights(train)
        weights = {number: freq[number] * 0.45 + recent[number] * 0.35 + dormancy[number] * 0.20 for number in range(1, 38)}
        return weighted_sample_without_replacement(rng, weights)
    raise ValueError(f"unknown null strategy: {strategy}")


def strategy_portfolio(train: Sequence[Draw], strategy: str, seed: int, count: int = 5) -> List[Ticket]:
    rng = random.Random(seed)
    selected: List[Ticket] = []
    usage: Counter[int] = Counter()
    attempts = 0
    while len(selected) < count and attempts < 3000:
        attempts += 1
        ticket = strategy_ticket(train, strategy, rng)
        if ticket in selected or any(overlap(ticket, previous) > 4 for previous in selected):
            continue
        next_usage = usage.copy()
        next_usage.update(ticket)
        if any(value > 4 for value in next_usage.values()):
            continue
        selected.append(ticket)
        usage = next_usage
    while len(selected) < count:
        ticket = tuple(sorted(rng.sample(range(1, 38), 7)))
        if ticket not in selected:
            selected.append(ticket)
    return selected


def evaluate_strategy(
    draws: Sequence[Draw], target_indices: Sequence[int], prize_rows: Mapping[int, Dict[str, str]],
    *, strategy: str, seed: int, purchase_count: int, unit_cost: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for index in target_indices:
        target = draws[index]
        tickets = strategy_portfolio(draws[:index], strategy, seed + target.draw_no * 1009, purchase_count)
        payout = 0
        max_main = 0
        for ticket in tickets:
            main_match, _bonus_match, rank = evaluate_ticket(ticket, target)
            max_main = max(max_main, main_match)
            payout += prize_amount_for_rank(dict(prize_rows.get(target.draw_no, {})), rank)
        cost = purchase_count * unit_cost
        records.append({"draw_no": target.draw_no, "year": int(str(target.date)[:4]), "cost": cost,
                        "payout": payout, "profit": payout - cost, "max_main_match": max_main})
    return records


def summarize_records(records: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    cost = sum(int(record["cost"]) for record in records)
    payout = sum(int(record["payout"]) for record in records)
    payouts = sorted((int(record["payout"]) for record in records), reverse=True)
    top1 = payouts[0] if payouts else 0
    yearly: Dict[int, Dict[str, int]] = {}
    for record in records:
        bucket = yearly.setdefault(int(record["year"]), {"cost": 0, "payout": 0})
        bucket["cost"] += int(record["cost"])
        bucket["payout"] += int(record["payout"])
    yearly_roi = [values["payout"] / values["cost"] * 100.0 for values in yearly.values() if values["cost"]]
    roi = payout / cost * 100.0 if cost else 0.0
    top1_removed = (payout - top1) / cost * 100.0 if cost else 0.0
    median_year = statistics.median(yearly_roi) if yearly_roi else 0.0
    max_main = float(max((int(record["max_main_match"]) for record in records), default=0))
    score = 0.50 * top1_removed + 0.25 * median_year + 0.15 * roi + 0.10 * max_main * 10.0
    return {"roi_percent": roi, "roi_excluding_top1_percent": top1_removed,
            "median_year_roi_percent": median_year, "max_main_match": max_main,
            "robust_score": score}


def block_scores(records: Sequence[Mapping[str, object]], block_ids: Sequence[int], wanted: set[int]) -> float:
    selected = [record for record, block in zip(records, block_ids) if block in wanted]
    return summarize_records(selected)["robust_score"] if selected else 0.0


def probability_of_backtest_overfitting(
    records_by_strategy: Sequence[Sequence[Mapping[str, object]]], *, block_count: int = 6
) -> Dict[str, object]:
    if not records_by_strategy or not records_by_strategy[0]:
        return {"pbo": 1.0, "combinations": 0}
    size = len(records_by_strategy[0])
    block_ids = [min(block_count - 1, int(index * block_count / size)) for index in range(size)]
    half = block_count // 2
    failures = 0
    evaluated = 0
    logit_values: List[float] = []
    for in_blocks_tuple in combinations(range(block_count), half):
        in_blocks = set(in_blocks_tuple)
        out_blocks = set(range(block_count)) - in_blocks
        in_scores = [block_scores(records, block_ids, in_blocks) for records in records_by_strategy]
        winner = max(range(len(in_scores)), key=lambda index: in_scores[index])
        out_scores = [block_scores(records, block_ids, out_blocks) for records in records_by_strategy]
        ordered = sorted(out_scores, reverse=True)
        rank = ordered.index(out_scores[winner]) + 1
        percentile_rank = rank / len(out_scores)
        failures += int(percentile_rank > 0.50)
        evaluated += 1
        omega = min(1.0 - 1e-9, max(1e-9, percentile_rank))
        logit_values.append(math.log(omega / (1.0 - omega)))
    return {"pbo": round(failures / evaluated, 6) if evaluated else 1.0,
            "combinations": evaluated,
            "median_oos_rank_logit": round(statistics.median(logit_values), 6) if logit_values else 0.0,
            "block_count": block_count}


def write_report(path: Path, payload: Mapping[str, object]) -> None:
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    model = payload.get("model_metrics", {}) if isinstance(payload.get("model_metrics"), dict) else {}
    lines = ["LOTO7 Null Strategy League", "==========================", "",
             f"created_at: {payload.get('created_at')}", f"model: {payload.get('model')}",
             f"target_draws: {payload.get('target_draws')}",
             f"null_simulations: {payload.get('null_simulations')}",
             f"model_roi: {model.get('roi_percent')}",
             f"model_top1_removed_roi: {model.get('roi_excluding_top1_percent')}",
             f"null_exceedance_rate: {payload.get('model_percentile')}",
             f"pbo: {payload.get('pbo')}", f"passed: {decision.get('passed')}", "",
             "この検査は多数試行で偶然高く見える戦略を検出するための診断です。",
             "過去検証であり将来の当せんを保証しません。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a LOTO7 model against a null-strategy league.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="loto7_best_model.json")
    parser.add_argument("--summary", default="outputs/generation4/null_strategy_league_summary.json")
    parser.add_argument("--report", default="outputs/generation4/null_strategy_league_report.txt")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=240)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--seed", type=int, default=44007)
    parser.add_argument("--max-null-exceedance", type=float, default=0.10)
    parser.add_argument("--max-pbo", type=float, default=0.40)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base = select_target_indices(draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None)
    target_indices = indices_for_years(draws, base, args.start_year, args.end_year)
    if not target_indices:
        raise SystemExit("no null-league target draws selected")
    model_metrics = evaluate_model_robust(
        genome=load_genome(args.model), model_path=args.model, draws=draws, prize_rows=prize_rows,
        target_indices=target_indices, purchase_count=args.purchase_count, unit_cost=args.unit_cost,
        bootstrap_samples=200, bootstrap_seed=args.seed, include_draw_records=True)
    model_records = model_metrics.get("draw_records", [])
    if not isinstance(model_records, list):
        raise SystemExit("model draw records missing")
    model_summary = {"roi_percent": float(model_metrics.get("roi_percent", 0.0)),
                     "roi_excluding_top1_percent": float(model_metrics.get("roi_excluding_top1_percent", 0.0)),
                     "median_year_roi_percent": float(model_metrics.get("median_year_roi_percent", 0.0)),
                     "max_main_match": float(model_metrics.get("max_main_match", 0.0))}
    model_summary["robust_score"] = (0.50 * model_summary["roi_excluding_top1_percent"]
                                     + 0.25 * model_summary["median_year_roi_percent"]
                                     + 0.15 * model_summary["roi_percent"]
                                     + 0.10 * model_summary["max_main_match"] * 10.0)

    strategy_names = ("random", "balanced", "frequency", "dormancy", "recent", "hybrid")
    null_results: List[Dict[str, object]] = []
    all_records: List[Sequence[Mapping[str, object]]] = [model_records]
    for simulation in range(args.simulations):
        strategy = strategy_names[simulation % len(strategy_names)]
        records = evaluate_strategy(draws, target_indices, prize_rows, strategy=strategy,
                                    seed=args.seed + simulation * 7919,
                                    purchase_count=args.purchase_count, unit_cost=args.unit_cost)
        metrics = summarize_records(records)
        null_results.append({"simulation": simulation, "strategy": strategy,
                             **{key: round(value, 6) for key, value in metrics.items()}})
        all_records.append(records)

    model_score = float(model_summary["robust_score"])
    exceedance = sum(1 for result in null_results if float(result["robust_score"]) >= model_score) / len(null_results)
    pbo = probability_of_backtest_overfitting(all_records, block_count=6)
    null_scores = [float(result["robust_score"]) for result in null_results]
    null_top1 = [float(result["roi_excluding_top1_percent"]) for result in null_results]
    passed = bool(exceedance <= args.max_null_exceedance and float(pbo["pbo"]) <= args.max_pbo)
    payload: Dict[str, object] = {
        "created_at": now_iso(), "kind": "loto7_null_strategy_league", "csv": args.csv,
        "model": args.model, "target_draws": len(target_indices), "start_year": args.start_year,
        "end_year": args.end_year, "null_simulations": args.simulations,
        "strategy_types": list(strategy_names),
        "model_metrics": {key: round(value, 6) for key, value in model_summary.items()},
        "null_distribution": {
            "robust_score_p50": round(percentile(null_scores, 0.50), 6),
            "robust_score_p90": round(percentile(null_scores, 0.90), 6),
            "robust_score_p95": round(percentile(null_scores, 0.95), 6),
            "top1_removed_roi_p90": round(percentile(null_top1, 0.90), 6),
            "top1_removed_roi_p95": round(percentile(null_top1, 0.95), 6)},
        "model_percentile": round(exceedance, 6), "pbo": pbo.get("pbo"), "pbo_detail": pbo,
        "decision": {"passed": passed, "max_null_exceedance": args.max_null_exceedance,
                     "max_pbo": args.max_pbo,
                     "reasons": [f"null exceedance={exceedance:.6f}", f"PBO={float(pbo['pbo']):.6f}"]},
        "null_results": null_results,
        "notes": ["The model is compared with many deterministic null strategies under the same five-ticket cost.",
                  "PBO is a CSCV-style diagnostic and not a proof of predictability."]}
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(Path(args.report), payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "null_results"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
