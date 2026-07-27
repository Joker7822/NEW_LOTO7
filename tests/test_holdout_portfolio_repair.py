from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loto7.evaluation.hit_metrics import summarize_hit_metrics
from scripts.repair_holdout_summary import build_from_detail


HEADER = [
    "draw_no",
    "date",
    "year",
    "combo_index",
    "ticket",
    "actual_main",
    "actual_bonus",
    "main_match",
    "bonus_match",
    "rank",
    "purchase_cost",
    "prize_amount",
    "profit",
    "prize_data_missing",
]


class HoldoutPortfolioRepairTests(unittest.TestCase):
    def _write_detail(self, path: Path, omit_last_ticket: bool = False) -> None:
        portfolios = {
            100: [
                "01 02 03 04 05 06 07",
                "01 02 03 08 09 10 11",
                "04 05 06 12 13 14 15",
                "07 08 09 16 17 18 19",
                "10 11 12 20 21 22 23",
            ],
            101: [
                "01 04 07 10 13 16 19",
                "02 05 08 11 14 17 20",
                "03 06 09 12 15 18 21",
                "01 05 09 13 17 21 25",
                "02 06 10 14 18 22 26",
            ],
        }
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER)
            writer.writeheader()
            for draw_no, tickets in portfolios.items():
                for combo_index, ticket in enumerate(tickets, start=1):
                    if omit_last_ticket and draw_no == 101 and combo_index == 5:
                        continue
                    writer.writerow(
                        {
                            "draw_no": draw_no,
                            "date": "2026-01-01",
                            "year": 2026,
                            "combo_index": combo_index,
                            "ticket": ticket,
                            "actual_main": "01 02 03 04 05 06 07",
                            "actual_bonus": "08 09",
                            "main_match": 4 if combo_index == 1 else 2,
                            "bonus_match": 0,
                            "rank": "6等" if combo_index == 1 else "外れ",
                            "purchase_cost": 300,
                            "prize_amount": 1000 if combo_index == 1 else 0,
                            "profit": 700 if combo_index == 1 else -300,
                            "prize_data_missing": 0,
                        }
                    )

    def test_reconstructs_portfolios_and_nonzero_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detail = Path(temp_dir) / "detail.csv"
            self._write_detail(detail)
            result = build_from_detail(detail, 2020)
        recent = result["recent_era_summary"]
        self.assertTrue(recent["portfolio_metrics_available"])
        self.assertEqual(recent["portfolio_metric_draw_count"], 2)
        self.assertEqual(recent["portfolio_expected_ticket_count"], 5)
        self.assertGreater(float(recent["average_portfolio_unique_numbers"]), 0.0)
        self.assertGreater(float(recent["mean_ticket_pair_overlap"]), 0.0)
        self.assertTrue(recent["hit_objective_score_complete"])

    def test_incomplete_portfolio_is_not_reported_as_zero_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detail = Path(temp_dir) / "detail.csv"
            self._write_detail(detail, omit_last_ticket=True)
            result = build_from_detail(detail, 2020)
        recent = result["recent_era_summary"]
        self.assertFalse(recent["portfolio_metrics_available"])
        self.assertIsNone(recent["average_portfolio_unique_numbers"])
        self.assertIsNone(recent["mean_ticket_pair_overlap"])
        self.assertIsNone(recent["max_ticket_pair_overlap"])
        self.assertFalse(recent["hit_objective_score_complete"])

    def test_missing_portfolio_argument_is_explicitly_unavailable(self) -> None:
        metrics = summarize_hit_metrics([1, 2, 3], ticket_main_matches=[1, 2, 3])
        self.assertFalse(metrics["portfolio_metrics_available"])
        self.assertIsNone(metrics["average_portfolio_unique_numbers"])
        self.assertFalse(metrics["hit_objective_score_complete"])


if __name__ == "__main__":
    unittest.main()
