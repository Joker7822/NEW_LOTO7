#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from loto7.evolution.tiered_portfolio import (
    infer_anchor_profile,
    select_adaptive_tiered_generation5_portfolio,
    select_tiered_generation5_portfolio,
)
from loto7_evolution_trainer import (
    _GENERATION5_PREFIX,
    _score_generation5_candidates,
    _select_generation5_portfolio,
    _select_greedy_scored_portfolio,
    evaluate_ticket,
    genome_from_dict,
    load_draws,
)


def portfolio_shape(tickets):
    used = set()
    overlaps = []
    for index, ticket in enumerate(tickets):
        current = set(ticket)
        used.update(current)
        for previous in tickets[:index]:
            overlaps.append(len(current & set(previous)))
    return {
        "unique_numbers": len(used),
        "mean_pair_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "max_pair_overlap": max(overlaps, default=0),
    }


def evaluate_portfolio(tickets, target):
    matches = [evaluate_ticket(ticket, target)[0] for ticket in tickets]
    shape = portfolio_shape(tickets)
    return {
        "max_main_match": max(matches, default=0),
        "main4_plus": int(max(matches, default=0) >= 4),
        "main5_plus": int(max(matches, default=0) >= 5),
        **shape,
    }


def _empty_bucket():
    return {
        "max_main_sum": 0,
        "main4_plus": 0,
        "main5_plus": 0,
        "unique_sum": 0.0,
        "overlap_sum": 0.0,
        "max_overlap": 0,
        "targets": 0,
    }


def _update_bucket(bucket, metrics):
    bucket["max_main_sum"] += metrics["max_main_match"]
    bucket["main4_plus"] += metrics["main4_plus"]
    bucket["main5_plus"] += metrics["main5_plus"]
    bucket["unique_sum"] += metrics["unique_numbers"]
    bucket["overlap_sum"] += metrics["mean_pair_overlap"]
    bucket["max_overlap"] = max(bucket["max_overlap"], metrics["max_pair_overlap"])
    bucket["targets"] += 1


def _finalize_bucket(bucket):
    denominator = max(1, bucket["targets"])
    return {
        "targets": bucket["targets"],
        "average_max_main_match": bucket["max_main_sum"] / denominator,
        "draw_main4_plus_count": bucket["main4_plus"],
        "draw_main4_plus_rate": bucket["main4_plus"] / denominator,
        "draw_main5_plus_count": bucket["main5_plus"],
        "average_unique_numbers": bucket["unique_sum"] / denominator,
        "mean_pair_overlap": bucket["overlap_sum"] / denominator,
        "max_pair_overlap": bucket["max_overlap"],
    }


def recent_portfolio_benchmark(draws, genome, target_count: int = 104):
    start = max(52, len(draws) - target_count)
    midpoint = start + max(1, (len(draws) - start) // 2)
    mode_names = (
        "greedy_calibrated",
        "marginal_portfolio",
        "tiered_high_match_portfolio",
        "adaptive_tiered_portfolio",
    )
    modes = {mode: _empty_bucket() for mode in mode_names}
    halves = {
        "first_half": {mode: _empty_bucket() for mode in mode_names},
        "second_half": {mode: _empty_bucket() for mode in mode_names},
    }
    profile_counts: Counter[str] = Counter()
    overlap_targets: Counter[int] = Counter()
    profile_confidence_sum = 0.0
    targets = 0
    overlap_limit = min(4, int(genome.overlap_limit))

    for index in range(start, len(draws)):
        train = draws[:index]
        target = draws[index]
        scored = _score_generation5_candidates(train, genome)
        greedy = _select_greedy_scored_portfolio(scored, 5, overlap_limit)
        marginal = _select_generation5_portfolio(scored, 5, overlap_limit)
        tiered = select_tiered_generation5_portfolio(scored, 5, overlap_limit)
        adaptive, profile = select_adaptive_tiered_generation5_portfolio(
            scored, 5, overlap_limit
        )
        profile_counts[str(profile["name"])] += 1
        overlap_targets[int(profile["anchor_overlap_target"])] += 1
        profile_confidence_sum += float(profile["confidence"])

        half_name = "first_half" if index < midpoint else "second_half"
        for mode, tickets in (
            ("greedy_calibrated", greedy),
            ("marginal_portfolio", marginal),
            ("tiered_high_match_portfolio", tiered),
            ("adaptive_tiered_portfolio", adaptive),
        ):
            metrics = evaluate_portfolio(tickets, target)
            _update_bucket(modes[mode], metrics)
            _update_bucket(halves[half_name][mode], metrics)
        targets += 1

    result = {"targets": targets}
    for mode, bucket in modes.items():
        result[mode] = _finalize_bucket(bucket)
    result["halves"] = {
        half_name: {
            mode: _finalize_bucket(bucket)
            for mode, bucket in half_modes.items()
        }
        for half_name, half_modes in halves.items()
    }
    result["adaptive_profiles"] = {
        "counts": dict(sorted(profile_counts.items())),
        "overlap_targets": {
            str(key): value for key, value in sorted(overlap_targets.items())
        },
        "mean_confidence": profile_confidence_sum / max(1, targets),
    }
    return result


class Generation5PortfolioTests(unittest.TestCase):
    def test_marginal_selector_is_deterministic_and_respects_overlap(self):
        scored = [
            (100.0, (1, 2, 3, 4, 5, 6, 7)),
            (99.8, (1, 2, 3, 4, 8, 9, 10)),
            (99.6, (1, 2, 3, 4, 11, 12, 13)),
            (99.4, (5, 6, 7, 8, 11, 14, 15)),
            (99.2, (8, 9, 10, 11, 12, 16, 17)),
            (99.0, (5, 6, 13, 14, 15, 16, 18)),
            (98.8, (7, 9, 12, 14, 16, 17, 19)),
        ]
        left = _select_generation5_portfolio(scored, 5, 4)
        right = _select_generation5_portfolio(scored, 5, 4)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 5)
        for index, ticket in enumerate(left):
            for previous in left[:index]:
                self.assertLessEqual(len(set(ticket) & set(previous)), 4)

    def test_marginal_selector_coverage_is_not_worse_than_greedy_fixture(self):
        scored = [
            (100.0, (1, 2, 3, 4, 5, 6, 7)),
            (99.9, (1, 2, 3, 4, 8, 9, 10)),
            (99.8, (1, 2, 3, 4, 11, 12, 13)),
            (99.7, (1, 2, 5, 6, 14, 15, 16)),
            (99.6, (3, 4, 7, 8, 17, 18, 19)),
            (99.5, (9, 10, 11, 12, 20, 21, 22)),
            (99.4, (13, 14, 15, 16, 23, 24, 25)),
        ]
        greedy = _select_greedy_scored_portfolio(scored, 5, 4)
        marginal = _select_generation5_portfolio(scored, 5, 4)
        self.assertGreaterEqual(
            portfolio_shape(marginal)["unique_numbers"],
            portfolio_shape(greedy)["unique_numbers"],
        )

    def test_tiered_selector_is_deterministic_and_keeps_two_quality_anchors(self):
        scored = [
            (100.0, (1, 2, 3, 4, 5, 6, 7)),
            (99.9, (1, 2, 3, 4, 8, 9, 10)),
            (99.8, (1, 2, 3, 4, 11, 12, 13)),
            (99.7, (1, 2, 5, 6, 14, 15, 16)),
            (99.6, (3, 4, 7, 8, 17, 18, 19)),
            (99.5, (9, 10, 11, 12, 20, 21, 22)),
            (99.4, (13, 14, 15, 16, 23, 24, 25)),
            (99.3, (5, 6, 7, 17, 20, 24, 26)),
        ]
        left = select_tiered_generation5_portfolio(scored, 5, 4)
        right = select_tiered_generation5_portfolio(scored, 5, 4)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 5)
        self.assertEqual(left[0], scored[0][1])
        self.assertGreaterEqual(len(set(left[0]) & set(left[1])), 3)
        for index, ticket in enumerate(left):
            for previous in left[:index]:
                self.assertLessEqual(len(set(ticket) & set(previous)), 4)

    def test_adaptive_profile_is_deterministic_and_training_only(self):
        concentrated = [
            (100.0 - index * 0.1, (1, 2, 3, 4, 5 + (index % 3), 8 + (index % 4), 15 + (index % 5)))
            for index in range(30)
        ]
        left = infer_anchor_profile(concentrated)
        right = infer_anchor_profile(concentrated)
        self.assertEqual(left, right)
        self.assertIn(int(left["anchor_overlap_target"]), (2, 3, 4))
        tickets_a, profile_a = select_adaptive_tiered_generation5_portfolio(
            concentrated, 5, 4
        )
        tickets_b, profile_b = select_adaptive_tiered_generation5_portfolio(
            concentrated, 5, 4
        )
        self.assertEqual(tickets_a, tickets_b)
        self.assertEqual(profile_a, profile_b)

    def test_recent_real_data_portfolio_ab_diagnostic(self):
        csv_path = Path("loto7.csv")
        model_path = Path("loto7_best_model.json")
        self.assertTrue(csv_path.exists())
        self.assertTrue(model_path.exists())
        draws = load_draws(str(csv_path))
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        genome = genome_from_dict(payload.get("genome", payload))
        genome.id = f"{_GENERATION5_PREFIX}portfolio_ab"
        genome.overlap_limit = min(4, int(genome.overlap_limit))
        metrics = recent_portfolio_benchmark(draws, genome, target_count=104)
        print("GEN5_PORTFOLIO_AB=" + json.dumps(metrics, sort_keys=True))
        self.assertEqual(metrics["targets"], min(104, len(draws) - 52))
        for mode in (
            "greedy_calibrated",
            "marginal_portfolio",
            "tiered_high_match_portfolio",
            "adaptive_tiered_portfolio",
        ):
            self.assertEqual(metrics[mode]["max_pair_overlap"], genome.overlap_limit)
            self.assertGreaterEqual(metrics[mode]["average_unique_numbers"], 7.0)
            self.assertLessEqual(metrics[mode]["average_unique_numbers"], 35.0)
            self.assertGreaterEqual(metrics[mode]["average_max_main_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
