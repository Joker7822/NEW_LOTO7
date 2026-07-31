from __future__ import annotations

import unittest

from loto7.evaluation.metrics_schema import financial_metrics
from loto7.evaluation.null_permutation import adaptive_null_test
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import paired_moving_block_bootstrap, wilson_interval


class StatisticalHardeningTests(unittest.TestCase):
    def test_financial_schema_separates_payout_and_profit(self) -> None:
        result = financial_metrics(1000, 250)
        self.assertEqual(result["payout_roi_percent"], 25.0)
        self.assertEqual(result["profit_roi_percent"], -75.0)
        self.assertEqual(result["roi_percent"], -75.0)

    def test_paired_bootstrap_detects_consistent_gain(self) -> None:
        result = paired_moving_block_bootstrap(
            [2.0] * 40,
            [1.0] * 40,
            samples=500,
            seed=7,
        )
        self.assertGreater(result["ci_lower"], 0.0)
        self.assertEqual(result["probability_positive"], 1.0)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(10, 100)
        self.assertLess(lower, 0.10)
        self.assertGreater(upper, 0.10)

    def test_portfolio_ranking_metrics_are_bounded(self) -> None:
        portfolio = [[(1, 2, 3, 4, 5, 6, 7)] * 5]
        result = summarize_portfolio_ranking(
            portfolio,
            [(1, 2, 3, 4, 5, 6, 7)],
        )
        self.assertEqual(result["top7_main_recall"], 1.0)
        self.assertGreaterEqual(result["portfolio_inclusion_auc"], 0.5)
        self.assertLessEqual(result["portfolio_inclusion_brier"], 1.0)

    def test_adaptive_null_is_deterministic(self) -> None:
        portfolios = [[(1, 2, 3, 4, 5, 6, 7)] * 5] * 8
        mains = [(1, 2, 3, 4, 5, 6, 7)] * 8
        kwargs = {
            "portfolios": portfolios,
            "mains": mains,
            "seeds": list(range(1, 13)),
            "checkpoints": [12],
            "search_width": 3,
            "max_exceedance": 0.10,
        }
        self.assertEqual(adaptive_null_test(**kwargs), adaptive_null_test(**kwargs))


if __name__ == "__main__":
    unittest.main()
