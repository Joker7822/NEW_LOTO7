#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"replacement marker missing in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if content.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"append marker missing in {path}: {marker!r}")
    target.write_text(text.replace(marker, marker + content, 1), encoding="utf-8")


METRICS_SCHEMA = r'''
"""Canonical metric schema shared by every evaluator.

The legacy project used ``roi_percent`` for both payout recovery and profit ROI.
Version 2 removes that ambiguity. Compatibility aliases remain explicit and are
scheduled for removal after all downstream readers migrate.
"""
from __future__ import annotations

from typing import Dict, Mapping

METRIC_SCHEMA_VERSION = "loto7-metrics-2026.07.31-v2"


def financial_metrics(total_cost: int, total_payout: int) -> Dict[str, object]:
    cost = int(total_cost)
    payout = int(total_payout)
    profit = payout - cost
    payout_ratio = payout / cost if cost else 0.0
    profit_ratio = profit / cost if cost else 0.0
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "total_cost": cost,
        "total_payout": payout,
        "profit": profit,
        "payout_ratio": round(payout_ratio, 6),
        "payout_roi_percent": round(payout_ratio * 100.0, 3),
        "profit_ratio": round(profit_ratio, 6),
        "profit_roi_percent": round(profit_ratio * 100.0, 3),
        # Compatibility aliases. roi always means profit ROI in schema v2.
        "roi": round(profit_ratio, 6),
        "roi_percent": round(profit_ratio * 100.0, 3),
        "payout_roi": round(payout_ratio, 6),
        "deprecated_metric_aliases": ["roi", "roi_percent", "payout_roi"],
    }


def normalize_financial_metrics(payload: Mapping[str, object]) -> Dict[str, object]:
    result = dict(payload)
    cost = int(float(result.get("total_cost", 0) or 0))
    payout = int(float(result.get("total_payout", 0) or 0))
    result.update(financial_metrics(cost, payout))
    return result


def payout_roi_percent(payload: Mapping[str, object]) -> float:
    if payload.get("payout_roi_percent") is not None:
        return float(payload["payout_roi_percent"])
    if payload.get("payout_ratio") is not None:
        return float(payload["payout_ratio"]) * 100.0
    cost = float(payload.get("total_cost", 0) or 0)
    payout = float(payload.get("total_payout", 0) or 0)
    return payout / cost * 100.0 if cost else 0.0


def profit_roi_percent(payload: Mapping[str, object]) -> float:
    if payload.get("profit_roi_percent") is not None:
        return float(payload["profit_roi_percent"])
    if payload.get("profit_ratio") is not None:
        return float(payload["profit_ratio"]) * 100.0
    cost = float(payload.get("total_cost", 0) or 0)
    profit = float(payload.get("profit", 0) or 0)
    return profit / cost * 100.0 if cost else 0.0
'''

STATISTICS = r'''
"""Paired time-series inference utilities for model promotion gates."""
from __future__ import annotations

import math
import random
import statistics
from typing import Dict, Sequence


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


def paired_moving_block_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 2000,
    block_size: int = 0,
    seed: int = 20260731,
    confidence: float = 0.95,
) -> Dict[str, object]:
    if len(candidate) != len(baseline):
        raise ValueError("paired bootstrap sequences must have equal length")
    if not candidate:
        raise ValueError("paired bootstrap sequences must not be empty")
    differences = [float(left) - float(right) for left, right in zip(candidate, baseline)]
    size = len(differences)
    block = int(block_size) if block_size > 0 else max(2, round(size ** (1 / 3)))
    block = min(size, max(1, block))
    rng = random.Random(seed)
    estimates = []
    for _ in range(max(100, int(samples))):
        sampled = []
        while len(sampled) < size:
            start = rng.randrange(size)
            sampled.extend(differences[(start + offset) % size] for offset in range(block))
        estimates.append(statistics.fmean(sampled[:size]))
    alpha = (1.0 - confidence) / 2.0
    estimate = statistics.fmean(differences)
    return {
        "kind": "paired_moving_block_bootstrap",
        "sample_count": len(estimates),
        "pair_count": size,
        "block_size": block,
        "confidence": confidence,
        "estimate": round(estimate, 8),
        "ci_lower": round(percentile(estimates, alpha), 8),
        "ci_upper": round(percentile(estimates, 1.0 - alpha), 8),
        "probability_positive": round(sum(value > 0 for value in estimates) / len(estimates), 8),
    }


def wilson_interval(successes: int, total: int, confidence_z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + confidence_z**2 / total
    centre = (proportion + confidence_z**2 / (2.0 * total)) / denominator
    margin = confidence_z * math.sqrt(
        proportion * (1.0 - proportion) / total + confidence_z**2 / (4.0 * total**2)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
'''

RANKING = r'''
"""Payout-independent ranking diagnostics for five-ticket portfolios."""
from __future__ import annotations

from typing import Dict, Sequence

NUMBERS = tuple(range(1, 38))


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def summarize_portfolio_ranking(
    portfolios: Sequence[Sequence[Sequence[int]]],
    actual_main_numbers: Sequence[Sequence[int]],
) -> Dict[str, object]:
    if len(portfolios) != len(actual_main_numbers):
        raise ValueError("portfolio and result lengths differ")
    top_hits = {7: 0, 14: 0, 18: 0}
    rank_sum = 0.0
    winner_count = 0
    all_scores = []
    all_labels = []
    calibration_error = 0.0
    for portfolio, actual in zip(portfolios, actual_main_numbers):
        ticket_count = max(1, len(portfolio))
        counts = {number: 0 for number in NUMBERS}
        for ticket in portfolio:
            for number in set(int(value) for value in ticket):
                if number in counts:
                    counts[number] += 1
        scores = {number: counts[number] / ticket_count for number in NUMBERS}
        ordered = sorted(NUMBERS, key=lambda number: (scores[number], -number), reverse=True)
        actual_set = {int(value) for value in actual}
        for cutoff in top_hits:
            top_hits[cutoff] += len(actual_set.intersection(ordered[:cutoff]))
        positions = {number: index + 1 for index, number in enumerate(ordered)}
        rank_sum += sum(positions[number] for number in actual_set)
        winner_count += len(actual_set)
        labels = [1 if number in actual_set else 0 for number in NUMBERS]
        values = [scores[number] for number in NUMBERS]
        all_scores.extend(values)
        all_labels.extend(labels)
    brier = sum((score - label) ** 2 for score, label in zip(all_scores, all_labels)) / max(1, len(all_scores))
    # Five equal-width calibration bins.
    for lower_index in range(5):
        lower = lower_index / 5
        upper = (lower_index + 1) / 5
        members = [
            (score, label)
            for score, label in zip(all_scores, all_labels)
            if lower <= score <= upper if lower_index == 4 else lower <= score < upper
        ]
        if members:
            mean_score = sum(item[0] for item in members) / len(members)
            mean_label = sum(item[1] for item in members) / len(members)
            calibration_error += len(members) / len(all_scores) * abs(mean_score - mean_label)
    draw_count = max(1, len(portfolios))
    return {
        "ranking_metric_version": "loto7-portfolio-ranking-2026.07.31-v1",
        "top7_main_recall": round(top_hits[7] / (draw_count * 7), 6),
        "top14_main_recall": round(top_hits[14] / (draw_count * 7), 6),
        "top18_main_recall": round(top_hits[18] / (draw_count * 7), 6),
        "winning_number_mean_rank": round(rank_sum / max(1, winner_count), 6),
        "portfolio_inclusion_auc": round(_auc(all_scores, all_labels), 6),
        "portfolio_inclusion_brier": round(brier, 6),
        "portfolio_calibration_error": round(calibration_error, 6),
    }
'''

HIT_NULL = r'''
#!/usr/bin/env python3
"""Adaptive, search-adjusted, payout-independent Null League for Generation 5."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import wilson_interval
from loto7.evolution.hit_first import hit_first_score
from loto7_evolution_trainer import generate_tickets, load_draws
from merge_evolution_shards import select_target_indices
from scripts.generation5_evolver import load_genome


def evaluate_portfolios(portfolios: Sequence[Sequence[Sequence[int]]], mains: Sequence[Sequence[int]]) -> Dict[str, object]:
    max_matches = []
    ticket_matches = []
    for portfolio, main in zip(portfolios, mains):
        target = set(int(value) for value in main)
        matches = [len(target.intersection(int(value) for value in ticket)) for ticket in portfolio]
        ticket_matches.extend(matches)
        max_matches.append(max(matches, default=0))
    metrics = summarize_hit_metrics(max_matches, ticket_main_matches=ticket_matches, portfolios=portfolios)
    metrics.update(summarize_portfolio_ranking(portfolios, mains))
    metrics["hit_first_objective_score"] = hit_first_score(metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="outputs/generation5/generation5_candidate_model.json")
    parser.add_argument("--seed-bank", default="outputs/generation5/null_seed_bank.json")
    parser.add_argument("--seed-phase", default="final")
    parser.add_argument("--summary", default="outputs/generation5/null_strategy_league_summary.json")
    parser.add_argument("--report", default="outputs/generation5/null_strategy_league_report.txt")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--checkpoints", default="150,500,1000")
    parser.add_argument("--search-width", type=int, default=6)
    parser.add_argument("--max-null-exceedance", type=float, default=0.10)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    indices = select_target_indices(draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None)
    indices = [index for index in indices if int(str(draws[index].date)[:4] or 0) >= args.start_year]
    genome = load_genome(args.model)
    if genome is None:
        raise SystemExit("candidate genome missing")
    portfolios = [generate_tickets(draws[:index], genome, args.purchase_count) for index in indices]
    mains = [draws[index].main for index in indices]
    observed = evaluate_portfolios(portfolios, mains)
    observed_score = float(observed["hit_first_objective_score"])

    bank = json.loads(Path(args.seed_bank).read_text(encoding="utf-8"))
    seeds = [int(value) for value in bank["phases"][args.seed_phase]]
    checkpoints = sorted({int(value) for value in args.checkpoints.split(",") if int(value) > 0})
    if not checkpoints or checkpoints[-1] > len(seeds):
        raise SystemExit("seed bank does not contain requested adaptive checkpoint")

    raw_scores: List[float] = []
    search_adjusted_scores: List[float] = []
    decisions = []
    stopped_at = checkpoints[-1]
    for position, seed in enumerate(seeds[:checkpoints[-1]], start=1):
        rng = random.Random(seed)
        permuted = list(mains)
        rng.shuffle(permuted)
        score = float(evaluate_portfolios(portfolios, permuted)["hit_first_objective_score"])
        raw_scores.append(score)
        if len(raw_scores) % max(1, args.search_width) == 0:
            search_adjusted_scores.append(max(raw_scores[-args.search_width :]))
        if position in checkpoints:
            adjusted = search_adjusted_scores or raw_scores
            exceedances = sum(value >= observed_score for value in adjusted)
            rate = exceedances / len(adjusted)
            lower, upper = wilson_interval(exceedances, len(adjusted))
            verdict = "continue"
            if upper <= args.max_null_exceedance:
                verdict = "pass"
            elif lower > args.max_null_exceedance:
                verdict = "fail"
            decisions.append({
                "simulations": position,
                "search_adjusted_trials": len(adjusted),
                "exceedances": exceedances,
                "exceedance": round(rate, 6),
                "wilson_ci_lower": round(lower, 6),
                "wilson_ci_upper": round(upper, 6),
                "verdict": verdict,
            })
            if verdict != "continue":
                stopped_at = position
                break

    final = decisions[-1]
    passed = final["verdict"] == "pass"
    payload = {
        "kind": "loto7_hit_first_adaptive_null_league",
        "metric_schema_version": "loto7-metrics-2026.07.31-v2",
        "model": args.model,
        "target_draws": len(indices),
        "observed_metrics": observed,
        "observed_score": round(observed_score, 6),
        "adaptive_checkpoints": decisions,
        "stopped_at_simulations": stopped_at,
        "search_width": args.search_width,
        "null_distribution": {
            "raw_count": len(raw_scores),
            "search_adjusted_count": len(search_adjusted_scores),
            "raw_median": round(statistics.median(raw_scores), 6),
            "adjusted_median": round(statistics.median(search_adjusted_scores or raw_scores), 6),
            "adjusted_p90": round(sorted(search_adjusted_scores or raw_scores)[int(0.9 * (len(search_adjusted_scores or raw_scores) - 1))], 6),
        },
        "decision": {
            "passed": passed,
            "max_null_exceedance": args.max_null_exceedance,
            "exceedance": final["exceedance"],
            "wilson_ci_upper": final["wilson_ci_upper"],
            "reason": "upper confidence bound must be at or below the threshold",
        },
        "notes": [
            "Winning-number rows are permuted against sealed model portfolios.",
            "Maxima over search-width null trials adjust for candidate selection multiplicity.",
            "Payout values never contribute to this adoption score.",
        ],
    }
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(json.dumps({key: value for key, value in payload.items() if key != "null_distribution"}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

ABLATION = r'''
#!/usr/bin/env python3
"""Separate model ranking value from five-ticket portfolio diversification value."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Sequence

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evolution.hit_first import hit_first_score
from loto7_evolution_trainer import generate_tickets, load_draws
from merge_evolution_shards import select_target_indices
from scripts.generation5_evolver import load_genome


def random_portfolio(rng: random.Random) -> list[tuple[int, ...]]:
    result = []
    attempts = 0
    while len(result) < 5 and attempts < 5000:
        attempts += 1
        ticket = tuple(sorted(rng.sample(range(1, 38), 7)))
        if all(len(set(ticket).intersection(other)) <= 4 for other in result):
            result.append(ticket)
    return result


def metrics(portfolios: Sequence[Sequence[Sequence[int]]], mains: Sequence[Sequence[int]]) -> Dict[str, object]:
    maximums = []
    ticket_matches = []
    for portfolio, main in zip(portfolios, mains):
        target = set(main)
        values = [len(target.intersection(ticket)) for ticket in portfolio]
        maximums.append(max(values, default=0))
        ticket_matches.extend(values)
    result = summarize_hit_metrics(maximums, ticket_main_matches=ticket_matches, portfolios=portfolios)
    result.update(summarize_portfolio_ranking(portfolios, mains))
    result["hit_first_objective_score"] = hit_first_score(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="outputs/generation5/generation5_candidate_model.json")
    parser.add_argument("--output", default="outputs/generation5/prediction_ablation.json")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    draws = load_draws(args.csv)
    indices = select_target_indices(draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None)
    indices = [index for index in indices if int(str(draws[index].date)[:4] or 0) >= args.start_year]
    genome = load_genome(args.model)
    if genome is None:
        raise SystemExit("candidate genome missing")
    rng = random.Random(args.seed)
    learned = [generate_tickets(draws[:index], genome, 5) for index in indices]
    random_only = [random_portfolio(rng) for _ in indices]
    learned_candidates = [generate_tickets(draws[:index], genome, 20) for index in indices]
    learned_randomized = [rng.sample(list(portfolio), 5) for portfolio in learned_candidates]
    mains = [draws[index].main for index in indices]
    a = metrics(learned, mains)
    b = metrics(random_only, mains)
    c = metrics(learned_randomized, mains)
    payload = {
        "kind": "loto7_model_portfolio_ablation",
        "target_draws": len(indices),
        "A_learned_model_and_optimizer": a,
        "B_random_scores_and_diversified_portfolio": b,
        "C_learned_model_and_randomized_portfolio": c,
        "effects": {
            "model_ranking_value_A_minus_B": round(float(a["hit_first_objective_score"]) - float(b["hit_first_objective_score"]), 6),
            "portfolio_optimizer_value_A_minus_C": round(float(a["hit_first_objective_score"]) - float(c["hit_first_objective_score"]), 6),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["effects"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

TESTS = r'''
from __future__ import annotations

import unittest

from loto7.evaluation.metrics_schema import financial_metrics
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import paired_moving_block_bootstrap, wilson_interval


class StatisticalHardeningTests(unittest.TestCase):
    def test_financial_schema_separates_payout_and_profit(self) -> None:
        result = financial_metrics(1000, 250)
        self.assertEqual(result["payout_roi_percent"], 25.0)
        self.assertEqual(result["profit_roi_percent"], -75.0)
        self.assertEqual(result["roi_percent"], -75.0)

    def test_paired_bootstrap_detects_consistent_gain(self) -> None:
        result = paired_moving_block_bootstrap([2.0] * 40, [1.0] * 40, samples=500, seed=7)
        self.assertGreater(result["ci_lower"], 0.0)
        self.assertEqual(result["probability_positive"], 1.0)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(10, 100)
        self.assertLess(lower, 0.10)
        self.assertGreater(upper, 0.10)

    def test_portfolio_ranking_metrics_are_bounded(self) -> None:
        portfolio = [[(1, 2, 3, 4, 5, 6, 7)] * 5]
        result = summarize_portfolio_ranking(portfolio, [(1, 2, 3, 4, 5, 6, 7)])
        self.assertEqual(result["top7_main_recall"], 1.0)
        self.assertGreaterEqual(result["portfolio_inclusion_auc"], 0.5)
        self.assertLessEqual(result["portfolio_inclusion_brier"], 1.0)


if __name__ == "__main__":
    unittest.main()
'''

DOC = r'''
# Statistical Hardening v2

Applied on 2026-07-31.

## Canonical metrics

- `payout_roi_percent`: payout / cost * 100
- `profit_roi_percent`: (payout - cost) / cost * 100
- `roi_percent`: deprecated compatibility alias for profit ROI

Every new evaluator writes `metric_schema_version=loto7-metrics-2026.07.31-v2`.

## Promotion evidence

Generation 5 now records paired moving-block bootstrap intervals for average maximum
main-number match and draw-level 4+ reach. The fixed final gate uses an adaptive
payout-independent permutation Null League with 150, 500 and 1,000 checkpoints.
The upper Wilson confidence bound must be at or below 10%.

## Ablation

`prediction_ablation.json` separates model ranking value from portfolio optimizer
value through three controlled evaluations.

## Repository cleanup

Historical pre-hit-first backups and reproducible large detail CSV files are no
longer tracked. Current resume state, current histories, production predictions and
sealed evidence remain tracked.
'''

write("src/loto7/evaluation/metrics_schema.py", METRICS_SCHEMA)
write("src/loto7/evaluation/statistics.py", STATISTICS)
write("src/loto7/evaluation/ranking.py", RANKING)
write("scripts/run_hit_first_null_league.py", HIT_NULL)
write("scripts/run_prediction_ablation.py", ABLATION)
write("tests/test_statistical_hardening.py", TESTS)
write("docs/STATISTICAL_HARDENING_V2.md", DOC)
write("requirements.lock", "optuna==3.6.0\nnumpy==1.26.4\npandas==2.2.2\nscikit-learn==1.4.2\nxgboost==2.0.3\nlightgbm==4.3.0\ncatboost==1.2.5\nshap==0.45.1\ntorch==2.2.2\n")

# Canonical financial and ranking metrics in hit-first evaluator.
replace_once(
    "src/loto7/evolution/hit_first.py",
    "from loto7.evaluation.hit_metrics import summarize_hit_metrics\n",
    "from loto7.evaluation.hit_metrics import summarize_hit_metrics\nfrom loto7.evaluation.metrics_schema import financial_metrics\nfrom loto7.evaluation.ranking import summarize_portfolio_ranking\n",
)
replace_once(
    "src/loto7/evolution/hit_first.py",
    '                "portfolio": tickets,\n',
    '                "portfolio": tickets,\n                "main": list(target.main),\n',
)
replace_once(
    "src/loto7/evolution/hit_first.py",
    '    hit_metrics = summarize_hit_metrics(\n        [int(record["max_main_match"]) for record in draw_records],\n        ticket_main_matches=ticket_main_matches,\n        portfolios=portfolios,\n    )\n',
    '    hit_metrics = summarize_hit_metrics(\n        [int(record["max_main_match"]) for record in draw_records],\n        ticket_main_matches=ticket_main_matches,\n        portfolios=portfolios,\n    )\n    ranking_metrics = summarize_portfolio_ranking(\n        portfolios, [record["main"] for record in draw_records]\n    )\n',
)
replace_once(
    "src/loto7/evolution/hit_first.py",
    '        "total_cost": total_cost,\n        "total_payout": total_payout,\n        "profit": profit,\n        "roi": round((total_payout / total_cost) if total_cost else 0.0, 6),\n        "roi_percent": round(payout_roi_percent, 3),\n        "payout_roi_percent": round(payout_roi_percent, 3),\n',
    '        **financial_metrics(total_cost, total_payout),\n',
)
replace_once(
    "src/loto7/evolution/hit_first.py",
    '        "temporal_segment_match_score_min": round(float(temporal_min), 6),\n        **hit_metrics,\n',
    '        "temporal_segment_match_score_min": round(float(temporal_min), 6),\n        "draw_outcomes": [\n            {\n                "draw_no": int(record["draw_no"]),\n                "max_main_match": int(record["max_main_match"]),\n                "draw4_plus": int(record["max_main_match"]) >= 4,\n            }\n            for record in draw_records\n        ],\n        **hit_metrics,\n        **ranking_metrics,\n',
)

# Canonical finance fields in robust evaluator while retaining explicit aliases.
replace_once(
    "src/loto7/evaluation/robust.py",
    "from loto7.evaluation.hit_metrics import summarize_hit_metrics  # noqa: E402\n",
    "from loto7.evaluation.hit_metrics import summarize_hit_metrics  # noqa: E402\nfrom loto7.evaluation.metrics_schema import financial_metrics  # noqa: E402\n",
)
replace_once(
    "src/loto7/evaluation/robust.py",
    '        "total_cost": total_cost,\n        "total_payout": total_payout,\n        "profit": total_profit,\n        "roi": round((total_payout / total_cost) if total_cost else 0.0, 6),\n        "roi_percent": _roi_percent(total_payout, total_cost),\n        "profit_roi_percent": round((total_profit / total_cost * 100.0) if total_cost else 0.0, 3),\n',
    '        **financial_metrics(total_cost, total_payout),\n',
)
replace_once(
    "src/loto7/evaluation/robust.py",
    '        "roi_excluding_top1_percent": _roi_percent(total_payout - top1, total_cost),\n        "roi_excluding_top2_percent": _roi_percent(total_payout - top2, total_cost),\n',
    '        "payout_roi_excluding_top1_percent": _roi_percent(total_payout - top1, total_cost),\n        "payout_roi_excluding_top2_percent": _roi_percent(total_payout - top2, total_cost),\n        "roi_excluding_top1_percent": _roi_percent(total_payout - top1, total_cost),\n        "roi_excluding_top2_percent": _roi_percent(total_payout - top2, total_cost),\n',
)
replace_once(
    "src/loto7/evaluation/robust.py",
    '        "median_year_roi_percent": round(statistics.median(yearly_values), 3) if yearly_values else 0.0,\n        "worst_year_roi_percent": round(min(yearly_values), 3) if yearly_values else 0.0,\n',
    '        "median_year_payout_roi_percent": round(statistics.median(yearly_values), 3) if yearly_values else 0.0,\n        "worst_year_payout_roi_percent": round(min(yearly_values), 3) if yearly_values else 0.0,\n        "median_year_roi_percent": round(statistics.median(yearly_values), 3) if yearly_values else 0.0,\n        "worst_year_roi_percent": round(min(yearly_values), 3) if yearly_values else 0.0,\n',
)

# Paired bootstrap evidence in Generation 5 adoption.
replace_once(
    "src/loto7/evolution/generation5.py",
    "from loto7.evolution.hit_first import hit_first_score\n",
    "from loto7.evolution.hit_first import hit_first_score\nfrom loto7.evaluation.statistics import paired_moving_block_bootstrap\n",
)
replace_once(
    "src/loto7/evolution/generation5.py",
    "    max_top1_payout_share: float = 0.50,\n) -> Dict[str, object]:\n",
    "    max_top1_payout_share: float = 0.50,\n    bootstrap_samples: int = 2000,\n    bootstrap_block_size: int = 0,\n) -> Dict[str, object]:\n",
)
replace_once(
    "src/loto7/evolution/generation5.py",
    '    baseline_min = _float(baseline_wf, "fold_objective_min")\n',
    '    paired_evidence: Dict[str, object] = {"available": False}\n    candidate_outcomes = candidate_full.get("draw_outcomes")\n    baseline_outcomes = baseline_full.get("draw_outcomes")\n    if isinstance(candidate_outcomes, list) and isinstance(baseline_outcomes, list) and len(candidate_outcomes) == len(baseline_outcomes) and candidate_outcomes:\n        candidate_max = [_float(item, "max_main_match") for item in candidate_outcomes if isinstance(item, Mapping)]\n        baseline_max = [_float(item, "max_main_match") for item in baseline_outcomes if isinstance(item, Mapping)]\n        candidate_draw4 = [1.0 if bool(item.get("draw4_plus")) else 0.0 for item in candidate_outcomes if isinstance(item, Mapping)]\n        baseline_draw4 = [1.0 if bool(item.get("draw4_plus")) else 0.0 for item in baseline_outcomes if isinstance(item, Mapping)]\n        if len(candidate_max) == len(baseline_max) == len(candidate_draw4) == len(baseline_draw4):\n            paired_evidence = {\n                "available": True,\n                "average_max_main_match": paired_moving_block_bootstrap(candidate_max, baseline_max, samples=bootstrap_samples, block_size=bootstrap_block_size),\n                "draw4_plus_rate": paired_moving_block_bootstrap(candidate_draw4, baseline_draw4, samples=bootstrap_samples, block_size=bootstrap_block_size, seed=20260732),\n            }\n\n    baseline_min = _float(baseline_wf, "fold_objective_min")\n',
)
replace_once(
    "src/loto7/evolution/generation5.py",
    '        (_float(candidate_full, "top1_payout_share") <= max_top1_payout_share, f"top1 payout share={_float(candidate_full, \'top1_payout_share\'):.6f}", f"top1 payout share {_float(candidate_full, \'top1_payout_share\'):.6f} > {max_top1_payout_share:.6f}"),\n    ]\n',
    '        (_float(candidate_full, "top1_payout_share") <= max_top1_payout_share, f"top1 payout share={_float(candidate_full, \'top1_payout_share\'):.6f}", f"top1 payout share {_float(candidate_full, \'top1_payout_share\'):.6f} > {max_top1_payout_share:.6f}"),\n    ]\n    if bool(paired_evidence.get("available")):\n        average_ci = paired_evidence["average_max_main_match"]\n        draw4_ci = paired_evidence["draw4_plus_rate"]\n        checks.extend([\n            (float(average_ci["ci_lower"]) > 0.0, f"paired average-max CI lower={float(average_ci[\"ci_lower\"]):.8f}", f"paired average-max CI lower {float(average_ci[\"ci_lower\"]):.8f} <= 0"),\n            (float(draw4_ci["ci_lower"]) > 0.0, f"paired draw4+ CI lower={float(draw4_ci[\"ci_lower\"]):.8f}", f"paired draw4+ CI lower {float(draw4_ci[\"ci_lower\"]):.8f} <= 0"),\n        ])\n',
)
replace_once(
    "src/loto7/evolution/generation5.py",
    '        "deltas": {key: round(float(value), 6) for key, value in deltas.items()},\n        "reasons": reasons,\n',
    '        "deltas": {key: round(float(value), 6) for key, value in deltas.items()},\n        "paired_bootstrap": paired_evidence,\n        "reasons": reasons,\n',
)

# Generation 5 checkpoint/resume and bootstrap CLI.
replace_once(
    "scripts/generation5_evolver.py",
    "import random\nimport sys\n",
    "import random\nimport sys\nimport base64\nimport pickle\n",
)
insert_helpers = r'''


def save_checkpoint(path: str, *, dataset_sha: str, model_sha: str, generation: int, archives: Mapping[str, Sequence[Mapping[str, object]]], rng: random.Random, evaluated_count: int) -> None:
    payload = {
        "kind": "loto7_generation5_checkpoint",
        "objective_version": OBJECTIVE_VERSION,
        "dataset_sha256": dataset_sha,
        "baseline_model_sha256": model_sha,
        "generation_completed": generation,
        "evaluated_count": evaluated_count,
        "rng_state": base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii"),
        "archives": {
            island: [
                {**public_record(record), "genome": asdict(record["_genome"])}
                for record in records
                if isinstance(record.get("_genome"), Genome)
            ]
            for island, records in archives.items()
        },
    }
    write_json(path, payload)


def restore_checkpoint(path: str, *, dataset_sha: str, model_sha: str) -> tuple[int, int, Dict[str, List[Mapping[str, object]]], object]:
    payload = read_json(path)
    if payload.get("objective_version") != OBJECTIVE_VERSION or payload.get("dataset_sha256") != dataset_sha or payload.get("baseline_model_sha256") != model_sha:
        raise ValueError("checkpoint input fingerprint mismatch")
    restored: Dict[str, List[Mapping[str, object]]] = {island: [] for island in ISLANDS}
    raw_archives = payload.get("archives", {})
    if isinstance(raw_archives, Mapping):
        for island in ISLANDS:
            rows = raw_archives.get(island, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("genome"), Mapping):
                    continue
                record = {key: value for key, value in row.items() if key != "genome"}
                record["_genome"] = genome_from_dict(dict(row["genome"]))
                restored[island].append(record)
    state = pickle.loads(base64.b64decode(str(payload["rng_state"])))
    return int(payload.get("generation_completed", 0)), int(payload.get("evaluated_count", 0)), restored, state
'''
append_once("scripts/generation5_evolver.py", "def parser() -> argparse.ArgumentParser:\n", insert_helpers)
replace_once(
    "scripts/generation5_evolver.py",
    '    result.add_argument("--max-top1-payout-share", type=float, default=0.50)\n    return result\n',
    '    result.add_argument("--max-top1-payout-share", type=float, default=0.50)\n    result.add_argument("--bootstrap-samples", type=int, default=2000)\n    result.add_argument("--bootstrap-block-size", type=int, default=0)\n    result.add_argument("--checkpoint", default="outputs/state/generation5/checkpoint.json")\n    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)\n    return result\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '    archives: Dict[str, List[Mapping[str, object]]] = {\n        island: [] for island in ISLANDS\n    }\n    parent_pools: Dict[str, List[Genome]] = {\n        island: list(seed_models) for island in ISLANDS\n    }\n',
    '    archives: Dict[str, List[Mapping[str, object]]] = {island: [] for island in ISLANDS}\n    evaluated_count = 0\n    start_generation = 1\n    checkpoint_path = Path(args.checkpoint)\n    if args.resume and checkpoint_path.exists() and checkpoint_path.stat().st_size:\n        completed, evaluated_count, restored, rng_state = restore_checkpoint(args.checkpoint, dataset_sha=dataset_sha, model_sha=model_sha)\n        archives.update(restored)\n        rng.setstate(rng_state)\n        start_generation = completed + 1\n    parent_pools: Dict[str, List[Genome]] = {\n        island: [record["_genome"] for record in archives[island] if isinstance(record.get("_genome"), Genome)] or list(seed_models)\n        for island in ISLANDS\n    }\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '    evaluated_count = 0\n    generations_completed = 0\n',
    '    generations_completed = start_generation - 1\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '    for generation in range(1, args.generations + 1):\n',
    '    for generation in range(start_generation, args.generations + 1):\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '        generations_completed = generation\n        print(\n',
    '        generations_completed = generation\n        save_checkpoint(args.checkpoint, dataset_sha=dataset_sha, model_sha=model_sha, generation=generation, archives=archives, rng=rng, evaluated_count=evaluated_count)\n        print(\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '        max_top1_payout_share=args.max_top1_payout_share,\n    )\n',
    '        max_top1_payout_share=args.max_top1_payout_share,\n        bootstrap_samples=args.bootstrap_samples,\n        bootstrap_block_size=args.bootstrap_block_size,\n    )\n',
)
replace_once(
    "scripts/generation5_evolver.py",
    '        "candidate_model": args.candidate_model,\n',
    '        "candidate_model": args.candidate_model,\n        "checkpoint": args.checkpoint,\n        "resumed_from_generation": start_generation - 1,\n',
)

# Fixed payout Null League reads canonical payout metrics.
replace_once(
    "scripts/run_fixed_null_league.py",
    '        "roi_percent": float(model_metrics.get("roi_percent", 0.0)),\n',
    '        "roi_percent": float(model_metrics.get("payout_roi_percent", model_metrics.get("roi_percent", 0.0))),\n',
)
replace_once(
    "scripts/run_fixed_null_league.py",
    '            model_metrics.get("roi_excluding_top1_percent", 0.0)\n',
    '            model_metrics.get("payout_roi_excluding_top1_percent", model_metrics.get("roi_excluding_top1_percent", 0.0))\n',
)
replace_once(
    "scripts/run_fixed_null_league.py",
    '            model_metrics.get("median_year_roi_percent", 0.0)\n',
    '            model_metrics.get("median_year_payout_roi_percent", model_metrics.get("median_year_roi_percent", 0.0))\n',
)

# Promotion requires the hit-first Null gate and records schema versions.
replace_once(
    "scripts/promote_generation5_candidate.py",
    '        "candidate_objective_version_present": bool(\n            candidate.get("objective_version")\n        ),\n',
    '        "candidate_objective_version_present": bool(candidate.get("objective_version")),\n        "metric_schema_v2": str(summary.get("candidate", {}).get("full_metrics", {}).get("metric_schema_version", "")).endswith("v2"),\n',
)

# Workflow: larger deterministic final bank, adaptive hit-first gate, financial safety,
# ablation, always-written run status and quality checks.
workflow = ROOT / ".github/workflows/loto7_generation5.yml"
text = workflow.read_text(encoding="utf-8")
text = text.replace("assert len(combined) == 1000", "assert len(combined) == 1850")
text = text.replace("--final 150", "--final 1000")
text = text.replace(
    "      - name: Run fixed final Null League on Generation 5 candidate\n        run: |\n          set -euo pipefail\n          python scripts/run_fixed_null_league.py \\\n            --csv loto7.csv \\\n            --model outputs/generation5/generation5_candidate_model.json \\\n            --seed-bank outputs/generation5/null_seed_bank.json \\\n            --seed-phase final \\\n            --simulations 150 \\\n            --summary outputs/generation5/null_strategy_league_summary.json \\\n            --report outputs/generation5/null_strategy_league_report.txt \\\n            --start-year 2020 \\\n            --purchase-count 5 \\\n            --unit-cost 300 \\\n            --max-null-exceedance 0.10 \\\n            --max-pbo 0.40\n",
    "      - name: Run adaptive hit-first final Null League\n        run: |\n          set -euo pipefail\n          python scripts/run_hit_first_null_league.py \\\n            --csv loto7.csv \\\n            --model outputs/generation5/generation5_candidate_model.json \\\n            --seed-bank outputs/generation5/null_seed_bank.json \\\n            --seed-phase final \\\n            --checkpoints 150,500,1000 \\\n            --search-width 6 \\\n            --summary outputs/generation5/null_strategy_league_summary.json \\\n            --report outputs/generation5/null_strategy_league_report.txt \\\n            --start-year 2020 \\\n            --purchase-count 5 \\\n            --max-null-exceedance 0.10\n\n      - name: Run independent financial safety Null diagnostic\n        run: |\n          set -euo pipefail\n          python scripts/run_fixed_null_league.py \\\n            --csv loto7.csv \\\n            --model outputs/generation5/generation5_candidate_model.json \\\n            --seed-bank outputs/generation5/null_seed_bank.json \\\n            --seed-phase selection \\\n            --simulations 150 \\\n            --summary outputs/generation5/financial_null_summary.json \\\n            --report outputs/generation5/financial_null_report.txt \\\n            --start-year 2020 \\\n            --purchase-count 5 \\\n            --unit-cost 300 \\\n            --max-null-exceedance 1.0 \\\n            --max-pbo 0.40\n\n      - name: Separate model and portfolio contributions\n        run: |\n          python scripts/run_prediction_ablation.py \\\n            --csv loto7.csv \\\n            --model outputs/generation5/generation5_candidate_model.json \\\n            --output outputs/generation5/prediction_ablation.json \\\n            --start-year 2020\n"
)
text = text.replace(
    '            --max-top1-payout-share 0.50\n',
    '            --max-top1-payout-share 0.50 \\\n            --bootstrap-samples 2000 \\\n            --checkpoint outputs/state/generation5/checkpoint.json \\\n            --resume\n',
)
text = text.replace(
    "      - name: Upload Generation 5 evidence\n        if: always()\n",
    "      - name: Write Generation 5 run status\n        if: always()\n        run: |\n          python - <<'PY'\n          import json, os\n          from pathlib import Path\n          summary = Path('outputs/generation5/generation5_summary.json')\n          promotion = Path('outputs/generation5/promotion_decision.json')\n          payload = {\n              'kind': 'loto7_generation5_run_status',\n              'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),\n              'summary_exists': summary.exists() and summary.stat().st_size > 0,\n              'promotion_decision_exists': promotion.exists() and promotion.stat().st_size > 0,\n              'status': 'complete' if promotion.exists() and promotion.stat().st_size > 0 else 'incomplete',\n          }\n          path = Path('outputs/generation5/run_status.json')\n          path.parent.mkdir(parents=True, exist_ok=True)\n          path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n          PY\n\n      - name: Upload Generation 5 evidence\n        if: always()\n",
)
workflow.write_text(text, encoding="utf-8")

# Validation workflow runs hardening regression, lint, typing and coverage.
validation = ROOT / ".github/workflows/loto7_validation_tests.yml"
text = validation.read_text(encoding="utf-8")
text = text.replace("run: python -m pip install -e .", "run: python -m pip install -e '.[dev]'")
text = text.replace(
    "            tests/test_holdout_portfolio_repair.py\n",
    "            tests/test_holdout_portfolio_repair.py \\\n            src/loto7/evaluation/metrics_schema.py \\\n            src/loto7/evaluation/statistics.py \\\n            src/loto7/evaluation/ranking.py \\\n            scripts/run_hit_first_null_league.py \\\n            scripts/run_prediction_ablation.py \\\n            tests/test_statistical_hardening.py\n",
)
text = text.replace(
    "            tests.test_holdout_portfolio_repair -v\n",
    "            tests.test_holdout_portfolio_repair \\\n            tests.test_statistical_hardening -v\n\n      - name: Lint and type-check hardened core\n        run: |\n          ruff check src/loto7/evaluation/metrics_schema.py src/loto7/evaluation/statistics.py src/loto7/evaluation/ranking.py scripts/run_hit_first_null_league.py scripts/run_prediction_ablation.py tests/test_statistical_hardening.py\n          mypy src/loto7/evaluation/metrics_schema.py src/loto7/evaluation/statistics.py src/loto7/evaluation/ranking.py\n\n      - name: Measure hardened-core coverage\n        run: |\n          coverage run --source=src/loto7/evaluation -m unittest tests.test_statistical_hardening\n          coverage report --fail-under=65\n",
)
validation.write_text(text, encoding="utf-8")

# Package and static-analysis configuration.
pyproject = ROOT / "pyproject.toml"
text = pyproject.read_text(encoding="utf-8")
text = text.replace("no_implicit_optional = true\n", "no_implicit_optional = true\nignore_missing_imports = true\nwarn_return_any = true\n")
if "[tool.coverage.run]" not in text:
    text += "\n[tool.coverage.run]\nbranch = true\nsource = [\"src/loto7/evaluation\"]\n\n[tool.coverage.report]\nshow_missing = true\nfail_under = 65\n"
pyproject.write_text(text, encoding="utf-8")

# Remove clear obsolete/reproducible artifacts; preserve current resume state,
# current histories, production predictions and sealed evidence.
deleted = []
for pattern in ("outputs/**/*pre_hit_first*.csv", "outputs/**/*pre_hit_first*.json"):
    for path in ROOT.glob(pattern):
        if path.is_file():
            path.unlink()
            deleted.append(path.relative_to(ROOT).as_posix())
for relative in (
    "outputs/holdout/holdout_result.csv",
    "outputs/role_ensemble/role_ensemble_backtest.csv",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()
        deleted.append(relative)

gitignore = ROOT / ".gitignore"
ignore = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
for line in (
    "outputs/**/*pre_hit_first*",
    "outputs/holdout/holdout_result.csv",
    "outputs/role_ensemble/role_ensemble_backtest.csv",
    "outputs/generation5/*.tmp",
):
    if line not in ignore.splitlines():
        ignore += ("" if ignore.endswith("\n") or not ignore else "\n") + line + "\n"
gitignore.write_text(ignore, encoding="utf-8")
write("docs/architecture/cleanup_20260731.json", json.dumps({
    "kind": "loto7_repository_cleanup",
    "deleted_files": sorted(deleted),
    "preserved": [
        "current resume state",
        "current model histories",
        "production prediction and cumulative history",
        "sealed evidence",
    ],
}, ensure_ascii=False, indent=2, sort_keys=True))

# Remove the one-shot migration files from the final branch diff.
for relative in (
    "scripts/apply_full_statistical_hardening.py",
    ".github/workflows/apply_full_statistical_hardening.yml",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

print(json.dumps({"status": "applied", "deleted": sorted(deleted)}, ensure_ascii=False, indent=2))
