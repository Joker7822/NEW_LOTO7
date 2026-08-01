#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from loto7.evolution.generation5 import (
    OBJECTIVE_VERSION,
    aggregate_walk_forward_metrics,
    build_seed_bank,
    build_walk_forward_folds,
    generation5_adoption_gate,
    pareto_dominates,
    pareto_front,
    seeds_for_phase,
    successive_halving_counts,
)
from loto7_evolution_trainer import (
    Draw,
    Genome,
    _GENERATION5_PREFIX,
    _LEGACY_BLEND_NUMBER_SCORES,
    _generation5_blend_number_scores,
    _window_consensus_signal,
    blend_number_scores,
)
from scripts.optimize_role_strategy import (
    DEFAULT_COUNTS,
    build_strategy,
    robust_role_metrics,
)


def fold(
    score: float,
    *,
    average: float,
    draw4: float,
    draw5: int = 0,
    draw6: int = 0,
    unique: float = 14.0,
    overlap: float = 4.0,
    max_overlap: int = 4,
):
    return {
        "target_draws": 20,
        "match_quality_score": score,
        "temporal_segment_match_score_median": score,
        "temporal_segment_match_score_min": score,
        "diversity_quality_score": score,
        "average_max_main_match": average,
        "draw_main4_plus_rate_percent": draw4,
        "draw_main5_plus_rate_percent": draw5 / 20 * 100.0,
        "draw_main6_plus_rate_percent": draw6 / 20 * 100.0,
        "draw_main5_plus_count": draw5,
        "draw_main6_plus_count": draw6,
        "average_portfolio_unique_numbers": unique,
        "mean_ticket_pair_overlap": overlap,
        "max_ticket_pair_overlap": max_overlap,
    }


def record(folds, *, roi: float = 20.0, top1: float = 0.2):
    return {
        "fold_metrics": folds,
        "walk_forward": aggregate_walk_forward_metrics(folds),
        "full_metrics": {
            "payout_roi_percent": roi,
            "top1_payout_share": top1,
        },
    }


def role_row(
    draw_no: int,
    main_match: int,
    *,
    payout: int = 0,
    rank: str = "外れ",
):
    return {
        "target_draw_no": draw_no,
        "main_match": main_match,
        "rank": rank,
        "payout": payout,
    }


def scoring_genome(genome_id: str, *, dormancy_weight: float = 0.04) -> Genome:
    return Genome(
        id=genome_id,
        generation=1,
        full_weight=0.25,
        recent240_weight=0.25,
        recent120_weight=0.25,
        recent60_weight=0.25,
        pair_weight=0.08,
        pair_recency_weight=0.08,
        pair_stability_weight=0.08,
        triple_weight=0.03,
        dormancy_weight=dormancy_weight,
        odd_bonus=0.30,
        sum_bonus=0.30,
        low_high_bonus=0.20,
        consecutive_penalty=0.20,
        overlap_limit=4,
        pool_size=18,
        target_sum_min=85,
        target_sum_max=180,
        max_consecutive_pairs=2,
    )


def synthetic_draws(count: int = 96):
    draws = []
    for index in range(count):
        main = tuple(
            sorted((((index * 3) + (offset * 5)) % 37) + 1 for offset in range(7))
        )
        draws.append(
            Draw(
                draw_no=index + 1,
                date=f"2025-{(index % 12) + 1:02d}-01",
                main=main,
                bonus=tuple(),
            )
        )
    return draws


def mean_std(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


class Generation5PrecisionTests(unittest.TestCase):
    def test_objective_version_is_recent_stability_v2(self):
        self.assertIn("2026.08.01-v2", OBJECTIVE_VERSION)

    def test_seed_bank_is_deterministic_disjoint_and_partitioned(self):
        left = build_seed_bank("abc", evaluator_version="v1")
        right = build_seed_bank("abc", evaluator_version="v1")
        self.assertEqual(left, right)
        learning = seeds_for_phase(left, "learning")
        selection = seeds_for_phase(left, "selection")
        final = seeds_for_phase(left, "final")
        self.assertEqual(
            (len(learning), len(selection), len(final)),
            (700, 150, 150),
        )
        self.assertEqual(len(set(learning + selection + final)), 1000)

    def test_walk_forward_folds_are_chronological_and_complete(self):
        folds = build_walk_forward_folds(list(range(103)), fold_count=5)
        flat = [value for item in folds for value in item.target_indices]
        self.assertEqual(flat, list(range(103)))
        self.assertEqual(len(folds), 5)
        self.assertLess(
            folds[0].target_indices[-1],
            folds[-1].target_indices[0],
        )

    def test_successive_halving_never_reaches_zero(self):
        self.assertEqual(successive_halving_counts(4), [4, 2, 1])
        self.assertEqual(successive_halving_counts(1), [1, 1, 1])

    def test_recent_weighting_prefers_late_improvement(self):
        early = [fold(12.0, average=1.8, draw4=5.0)] + [
            fold(10.0, average=1.7, draw4=4.0) for _ in range(4)
        ]
        late = [
            fold(10.0, average=1.7, draw4=4.0) for _ in range(4)
        ] + [fold(12.0, average=1.8, draw4=5.0)]
        early_metrics = aggregate_walk_forward_metrics(early)
        late_metrics = aggregate_walk_forward_metrics(late)
        self.assertGreater(
            late_metrics["recent_weighted_objective"],
            early_metrics["recent_weighted_objective"],
        )
        self.assertGreater(
            late_metrics["generation5_score"],
            early_metrics["generation5_score"],
        )

    def test_pareto_front_keeps_tradeoffs(self):
        a = {
            "fold_objective_median": 12,
            "fold_objective_min": 10,
            "draw_main4_plus_rate_percent": 5,
            "average_max_main_match": 1.8,
            "draw_main5_plus_count": 2,
            "draw_main6_plus_count": 0,
            "average_portfolio_unique_numbers": 14,
            "mean_ticket_pair_overlap": 4,
            "max_ticket_pair_overlap": 4,
        }
        b = dict(a, fold_objective_median=11)
        c = dict(a, draw_main4_plus_rate_percent=6, fold_objective_min=9)
        self.assertTrue(pareto_dominates(a, b))
        self.assertFalse(pareto_dominates(a, c))
        front = pareto_front([a, b, c])
        self.assertIn(a, front)
        self.assertIn(c, front)
        self.assertNotIn(b, front)

    def test_gate_rejects_single_fold_improvement(self):
        baseline = [
            fold(10.0, average=1.75, draw4=4.0) for _ in range(5)
        ]
        candidate = [
            fold(10.0, average=1.78, draw4=4.5) for _ in range(4)
        ] + [fold(12.0, average=2.0, draw4=8.0)]
        decision = generation5_adoption_gate(
            record(candidate),
            record(baseline),
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["positive_folds"], 1)

    def test_gate_rejects_latest_fold_regression(self):
        baseline = [
            fold(10.0, average=1.70, draw4=4.0) for _ in range(5)
        ]
        candidate = [
            fold(10.5, average=1.80, draw4=5.2) for _ in range(4)
        ] + [fold(8.0, average=1.55, draw4=2.0)]
        decision = generation5_adoption_gate(
            record(candidate),
            record(baseline),
            min_recent_weighted_delta=-10.0,
        )
        self.assertFalse(decision["passed"])
        self.assertTrue(
            any(
                "latest fold delta" in item
                for item in decision["failures"]
            )
        )

    def test_gate_accepts_broad_improvement_with_diversity(self):
        baseline = [
            fold(
                10.0,
                average=1.70,
                draw4=4.0,
                unique=12.0,
                overlap=4.8,
                max_overlap=5,
            )
            for _ in range(5)
        ]
        candidate = [
            fold(
                10.2,
                average=1.78,
                draw4=5.0,
                unique=14.0,
                overlap=4.0,
                max_overlap=4,
            )
            for _ in range(5)
        ]
        decision = generation5_adoption_gate(
            record(candidate),
            record(baseline),
        )
        self.assertTrue(decision["passed"], decision["failures"])
        self.assertEqual(decision["positive_folds"], 5)

    def test_generation5_calibration_keeps_legacy_models_exact(self):
        draws = synthetic_draws()
        genome = scoring_genome("legacy_model")
        self.assertEqual(
            blend_number_scores(draws, genome),
            _LEGACY_BLEND_NUMBER_SCORES(draws, genome),
        )

    def test_generation5_calibration_preserves_score_scale(self):
        draws = synthetic_draws()
        genome = scoring_genome(f"{_GENERATION5_PREFIX}scale_test")
        legacy = _LEGACY_BLEND_NUMBER_SCORES(draws, genome)
        calibrated = _generation5_blend_number_scores(draws, genome)
        legacy_mean, legacy_std = mean_std(legacy.values())
        calibrated_mean, calibrated_std = mean_std(calibrated.values())
        self.assertAlmostEqual(calibrated_mean, legacy_mean, places=10)
        self.assertAlmostEqual(calibrated_std, legacy_std, places=10)

    def test_generation5_consensus_penalizes_unstable_window_spike(self):
        stable_number = 1
        unstable_number = 2
        signals = []
        unstable_values = [3.0, -1.0, -1.0, -1.0]
        for unstable in unstable_values:
            signal = {number: 0.0 for number in range(1, 38)}
            signal[stable_number] = 1.0
            signal[unstable_number] = unstable
            signals.append(signal)
        consensus = _window_consensus_signal(signals, [0.25] * 4)
        self.assertGreater(consensus[stable_number], consensus[unstable_number])

    def test_generation5_ranking_does_not_use_explicit_dormancy_bonus(self):
        draws = synthetic_draws()
        low = scoring_genome(
            f"{_GENERATION5_PREFIX}dormancy_low",
            dormancy_weight=0.0,
        )
        high = scoring_genome(
            f"{_GENERATION5_PREFIX}dormancy_high",
            dormancy_weight=0.08,
        )
        low_scores = _generation5_blend_number_scores(draws, low)
        high_scores = _generation5_blend_number_scores(draws, high)
        low_rank = sorted(low_scores, key=low_scores.get, reverse=True)
        high_rank = sorted(high_scores, key=high_scores.get, reverse=True)
        self.assertEqual(low_rank, high_rank)

    def test_role_payout_concentration_is_detected(self):
        rows = [role_row(index, 1) for index in range(1, 105)]
        rows[-1] = role_row(
            104,
            5,
            payout=1_000_000,
            rank="3等",
        )
        metrics = robust_role_metrics(
            rows,
            recent_draws=104,
            block_size=52,
            unit_cost=300,
        )
        self.assertGreater(metrics["top1_payout_share"], 0.99)

    def test_concentrated_role_is_capped(self):
        stable = [
            role_row(
                index,
                4 if index % 20 == 0 else 2,
                payout=10_000 if index % 20 == 0 else 0,
                rank="5等" if index % 20 == 0 else "外れ",
            )
            for index in range(1, 157)
        ]
        concentrated = [
            role_row(index, 1) for index in range(1, 157)
        ]
        concentrated[-1] = role_row(
            156,
            6,
            payout=2_000_000,
            rank="3等",
        )
        details = {role: stable for role in DEFAULT_COUNTS}
        details["recent120"] = concentrated
        summary = {
            "status": "completed",
            "completed_target_draws": 156,
            "target_draws_total": 156,
            "genome_id": "test",
        }
        strategy = build_strategy(
            summary,
            details,
            recent_draws=104,
            block_size=52,
        )
        self.assertEqual(strategy["role_caps"]["recent120"], 1)
        self.assertLessEqual(
            strategy["strategy_counts"]["recent120"],
            1,
        )
        self.assertGreater(
            strategy["scores"]["main_best"],
            strategy["scores"]["recent120"],
        )
        self.assertEqual(sum(strategy["strategy_counts"].values()), 5)


if __name__ == "__main__":
    unittest.main()
