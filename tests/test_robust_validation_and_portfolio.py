#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from scripts.build_dual_model_prediction import Candidate, select_portfolio
from scripts.nested_walk_forward_validation import validate_fold_boundaries
from scripts.robust_model_metrics import percentile


class RobustValidationAndPortfolioTests(unittest.TestCase):
    def make_candidate(self, ticket, source, score):
        return Candidate(
            ticket=tuple(ticket),
            source=source,
            model_id=f"{source}_model",
            model_score=score,
            source_model=f"{source}.json",
            method=f"portfolio_{source}",
            support="test",
            individual_score=score,
            raw_rank=0,
            created_at="2026-07-15T00:00:00+00:00",
        )

    def test_nested_fold_boundaries_reject_future_leakage(self):
        validate_fold_boundaries(2021, 2022, 2023)
        with self.assertRaises(ValueError):
            validate_fold_boundaries(2023, 2023, 2024)
        with self.assertRaises(ValueError):
            validate_fold_boundaries(2021, 2024, 2023)

    def test_percentile_is_deterministic(self):
        values = [0.0, 10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 0.0), 0.0)
        self.assertEqual(percentile(values, 0.5), 20.0)
        self.assertEqual(percentile(values, 1.0), 40.0)

    def test_portfolio_is_selected_without_number_replacement(self):
        pools = {
            "full": [
                self.make_candidate((1, 2, 3, 4, 5, 6, 7), "full", 110.0),
                self.make_candidate((1, 8, 9, 10, 11, 12, 13), "full", 108.0),
                self.make_candidate((2, 14, 15, 16, 17, 18, 19), "full", 106.0),
            ],
            "recent": [
                self.make_candidate((3, 8, 14, 20, 21, 22, 23), "recent", 109.0),
                self.make_candidate((4, 9, 15, 24, 25, 26, 27), "recent", 107.0),
            ],
            "super": [],
            "regime": [
                self.make_candidate((5, 10, 16, 20, 28, 29, 30), "regime", 105.0),
                self.make_candidate((6, 11, 17, 21, 31, 32, 33), "regime", 104.0),
                self.make_candidate((7, 12, 18, 22, 34, 35, 36), "regime", 103.0),
            ],
        }
        selected, summary = select_portfolio(
            pools,
            {"full": 2, "recent": 1, "super": 0, "regime": 2},
            max_number_usage=3,
            max_pair_overlap=4,
            beam_width=80,
            candidates_per_step=20,
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(summary["post_selection_number_replacements"], 0)
        self.assertLessEqual(summary["max_number_usage"], 3)
        self.assertLessEqual(summary["max_pair_overlap"], 4)
        all_original = {candidate.ticket for candidates in pools.values() for candidate in candidates}
        self.assertTrue(all(candidate.ticket in all_original for candidate in selected))


if __name__ == "__main__":
    unittest.main()
