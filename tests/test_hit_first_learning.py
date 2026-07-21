#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from loto7.evolution.hit_first import (
    OBJECTIVE_VERSION,
    adoption_decision,
    diversity_quality_score,
    hit_first_key,
    hit_first_score,
    match_quality_score,
)


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


class HitFirstLearningTests(unittest.TestCase):
    def test_roi_and_profit_do_not_change_learning_score(self) -> None:
        low_finance = metrics(payout_roi_percent=8.0, roi_percent=8.0, profit=-500000)
        high_finance = metrics(payout_roi_percent=500.0, roi_percent=500.0, profit=9000000)
        self.assertEqual(hit_first_score(low_finance), hit_first_score(high_finance))

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
