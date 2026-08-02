from __future__ import annotations

import unittest

from loto7.evaluation.metrics_schema import financial_metrics
from loto7.evaluation.null_permutation import adaptive_null_test
from loto7.evaluation.ranking import summarize_portfolio_ranking
from loto7.evaluation.statistics import paired_moving_block_bootstrap, wilson_interval
from scripts.null_strategy_league import paired_model_pbo


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

    def test_adaptive_null_is_deterministic_and_preserves_seed_count(self) -> None:
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
        left = adaptive_null_test(**kwargs)
        right = adaptive_null_test(**kwargs)
        self.assertEqual(left, right)
        self.assertEqual(left["null_distribution"]["search_adjusted_count"], 12)
        self.assertEqual(left["null_distribution"]["raw_count"], 36)
        self.assertEqual(left["search_adjustment_method"], "within_seed_max")

    @staticmethod
    def _financial_records(payout: int, count: int = 12) -> list[dict[str, object]]:
        return [
            {
                "draw_no": index + 1,
                "year": 2020 + index // 4,
                "cost": 1500,
                "payout": payout,
                "profit": payout - 1500,
                "max_main_match": 4 if payout else 2,
            }
            for index in range(count)
        ]

    def test_paired_model_pbo_is_zero_for_stable_model_advantage(self) -> None:
        model = self._financial_records(3000)
        nulls = [self._financial_records(0) for _ in range(6)]
        result = paired_model_pbo(model, nulls, block_count=6)
        self.assertEqual(result["pbo"], 0.0)
        self.assertEqual(result["model_is_win_rate"], 1.0)
        self.assertEqual(result["oos_win_rate_when_selected"], 1.0)

    def test_paired_model_pbo_fails_closed_without_is_advantage(self) -> None:
        model = self._financial_records(0)
        nulls = [self._financial_records(3000) for _ in range(3)]
        result = paired_model_pbo(model, nulls, block_count=6)
        self.assertEqual(result["pbo"], 1.0)
        self.assertEqual(result["paired_comparisons"], 0)


if __name__ == "__main__":
    unittest.main()
