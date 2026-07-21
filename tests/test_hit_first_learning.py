#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import loto7_model_self_evolver as self_evolver_cli
from loto7.evolution.hit_first import (
    OBJECTIVE_VERSION,
    adoption_decision,
    diversity_quality_score,
    evaluate_model_on_holdout,
    hit_first_key,
    hit_first_score,
    match_quality_score,
)
from loto7_evolution_trainer import Draw, Genome


def metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "objective_version": OBJECTIVE_VERSION,
        "average_max_main_match": 2.0,
        "draw_main4_plus_rate": 0.05,
        "draw_main4_plus_rate_percent": 5.0,
        "draw_main5_plus_count": 1,
        "draw_main5_plus_rate": 0.006,
        "draw_main6_plus_count": 0,
        "draw_main6_plus_rate": 0.0,
        "draw_main7_plus_rate": 0.0,
        "average_portfolio_unique_numbers": 14.0,
        "mean_ticket_pair_overlap": 3.8,
        "temporal_segment_match_score_median": 12.0,
        "temporal_segment_match_score_min": 10.0,
        "payout_roi_percent": 20.0,
        "roi_percent": 20.0,
        "profit": -1000,
        "top1_payout_share": 0.40,
    }
    base.update(overrides)
    base["match_quality_score"] = match_quality_score(base)
    base["diversity_quality_score"] = diversity_quality_score(base)
    base["hit_first_objective_score"] = hit_first_score(base)
    return base


def sample_genome(*, identifier: str = "test_hit_first", score: float = 0.0) -> Genome:
    return Genome(
        id=identifier,
        generation=0,
        full_weight=0.25,
        recent240_weight=0.25,
        recent120_weight=0.25,
        recent60_weight=0.25,
        pair_weight=0.08,
        pair_recency_weight=0.08,
        pair_stability_weight=0.08,
        triple_weight=0.02,
        dormancy_weight=0.01,
        odd_bonus=0.2,
        sum_bonus=0.2,
        low_high_bonus=0.2,
        consecutive_penalty=0.2,
        overlap_limit=4,
        pool_size=12,
        target_sum_min=40,
        target_sum_max=210,
        max_consecutive_pairs=3,
        score=score,
        max_main_match=6,
        best_rank_count=99,
    )


class HitFirstLearningTests(unittest.TestCase):
    def test_roi_and_profit_do_not_change_learning_score_or_ranking(self) -> None:
        low_finance = metrics(payout_roi_percent=8.0, roi_percent=8.0, profit=-500000)
        high_finance = metrics(payout_roi_percent=500.0, roi_percent=500.0, profit=9000000)
        self.assertEqual(hit_first_score(low_finance), hit_first_score(high_finance))
        self.assertEqual(hit_first_key(low_finance), hit_first_key(high_finance))

    def test_legacy_seed_score_is_reset_by_public_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_model.json"
            path.write_text(
                json.dumps({"genome": asdict(sample_genome(identifier="legacy", score=999999.0))}),
                encoding="utf-8",
            )
            seeds = self_evolver_cli._load_high_match_seed_genomes([str(path)])
        self.assertEqual(len(seeds), 1)
        genome = seeds[0][1]
        self.assertEqual(genome.score, 0.0)
        self.assertEqual(genome.max_main_match, 0)
        self.assertEqual(genome.best_rank_count, 0)

    def test_more_five_plus_reach_improves_learning_score(self) -> None:
        baseline = metrics(draw_main5_plus_count=0, draw_main5_plus_rate=0.0)
        improved = metrics(draw_main5_plus_count=3, draw_main5_plus_rate=0.02)
        self.assertGreater(hit_first_score(improved), hit_first_score(baseline))
        self.assertGreater(hit_first_key(improved), hit_first_key(baseline))

    def test_temporal_worst_segment_affects_learning_score(self) -> None:
        unstable = metrics(temporal_segment_match_score_min=3.0)
        stable = metrics(temporal_segment_match_score_min=11.0)
        self.assertGreater(hit_first_score(stable), hit_first_score(unstable))

    def test_portfolio_diversity_affects_learning_score(self) -> None:
        duplicated = metrics(average_portfolio_unique_numbers=9.0, mean_ticket_pair_overlap=5.8)
        diverse = metrics(average_portfolio_unique_numbers=18.0, mean_ticket_pair_overlap=3.0)
        self.assertGreater(hit_first_score(diverse), hit_first_score(duplicated))

    def test_walk_forward_evaluator_emits_complete_hit_first_metrics(self) -> None:
        draws = [
            Draw(
                draw_no=index + 1,
                date=f"2026-01-{index + 1:02d}",
                main=tuple(range(index + 1, index + 8)),
                bonus=(20 + index, 30 + index),
            )
            for index in range(8)
        ]
        result = evaluate_model_on_holdout(
            genome=sample_genome(),
            model_path="synthetic",
            draws=draws,
            prize_rows={},
            target_indices=[3, 4, 5, 6, 7],
            purchase_count=5,
            unit_cost=300,
        )
        self.assertEqual(result["target_draws"], 5)
        self.assertEqual(result["total_tickets"], 25)
        self.assertGreater(float(result["average_portfolio_unique_numbers"]), 0.0)
        self.assertEqual(len(result["temporal_segment_metrics"]), 4)
        self.assertEqual(result["hit_first_objective_score"], hit_first_score(result))

    def test_adoption_rejects_high_match_regression_even_with_high_roi(self) -> None:
        baseline = metrics()
        candidate = metrics(
            average_max_main_match=1.9,
            draw_main4_plus_rate=0.04,
            draw_main4_plus_rate_percent=4.0,
            draw_main5_plus_count=0,
            draw_main5_plus_rate=0.0,
            payout_roi_percent=900.0,
            roi_percent=900.0,
            profit=10000000,
        )
        passed, reasons = adoption_decision(candidate, baseline)
        self.assertFalse(passed)
        self.assertTrue(any("FAIL:" in reason for reason in reasons))

    def test_adoption_rejects_payout_concentration(self) -> None:
        baseline = metrics(top1_payout_share=0.45)
        candidate = metrics(
            average_max_main_match=2.1,
            draw_main4_plus_rate=0.06,
            draw_main4_plus_rate_percent=6.0,
            draw_main5_plus_count=2,
            draw_main5_plus_rate=0.012,
            temporal_segment_match_score_min=10.2,
            top1_payout_share=0.75,
        )
        passed, reasons = adoption_decision(candidate, baseline)
        self.assertFalse(passed)
        self.assertTrue(any("top1 payout share" in reason for reason in reasons))

    def test_adoption_accepts_robust_high_match_improvement(self) -> None:
        baseline = metrics()
        candidate = metrics(
            average_max_main_match=2.1,
            draw_main4_plus_rate=0.06,
            draw_main4_plus_rate_percent=6.0,
            draw_main5_plus_count=2,
            draw_main5_plus_rate=0.012,
            temporal_segment_match_score_median=12.5,
            temporal_segment_match_score_min=10.5,
            average_portfolio_unique_numbers=15.0,
            mean_ticket_pair_overlap=3.5,
            payout_roi_percent=18.0,
            roi_percent=18.0,
            top1_payout_share=0.45,
        )
        passed, reasons = adoption_decision(candidate, baseline)
        self.assertTrue(passed, reasons)


if __name__ == "__main__":
    unittest.main()
