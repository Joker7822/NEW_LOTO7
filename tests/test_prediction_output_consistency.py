#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.enforce_prediction_constraints import (
    normalize_rows,
    relative_confidence,
    write_final_report,
)
from scripts.update_prediction_history import confidence_for


class PredictionOutputConsistencyTests(unittest.TestCase):
    def test_relative_confidence_uses_shared_scale(self):
        self.assertEqual(
            [relative_confidence(index) for index in range(5)],
            [0.95, 0.90, 0.85, 0.80, 0.75],
        )

    def test_history_never_uses_raw_model_or_unscaled_strategy_score(self):
        row = {
            "ensemble_score": "128.1315",
            "model_score": "71271.89",
        }
        self.assertEqual(confidence_for(row, 4), "0.8")

    def test_normalize_rows_updates_numbers_and_scores(self):
        rows = [
            {
                "confidence_rank": str(index + 1),
                "combo_index": str(index + 1),
                "numbers": "01 02 03 04 05 06 07",
                "prediction_method": "dual_regime",
                "support_models": "test",
                "ensemble_score": "71271.89",
            }
            for index in range(5)
        ]
        tickets = [
            (1, 2, 3, 4, 5, 6, 7),
            (1, 8, 9, 10, 11, 12, 13),
            (2, 14, 15, 16, 17, 18, 19),
            (3, 20, 21, 22, 23, 24, 25),
            (4, 26, 27, 28, 29, 30, 31),
        ]
        normalize_rows(rows, tickets, [(4, 32, 31)])
        self.assertEqual([row["ensemble_score"] for row in rows], ["0.95", "0.90", "0.85", "0.80", "0.75"])
        self.assertEqual(rows[4]["numbers"], "04 26 27 28 29 30 31")
        self.assertIn("usage_guard", rows[4]["prediction_method"])

    def test_report_is_rebuilt_from_finalized_rows(self):
        tickets = [
            (1, 2, 3, 4, 5, 6, 7),
            (1, 8, 9, 10, 11, 12, 13),
            (2, 14, 15, 16, 17, 18, 19),
            (3, 20, 21, 22, 23, 24, 25),
            (4, 26, 27, 28, 29, 30, 31),
        ]
        rows = []
        for index, ticket in enumerate(tickets):
            rows.append(
                {
                    "confidence_rank": str(index + 1),
                    "base_latest_draw_no": "685",
                    "base_latest_date": "2026-07-10",
                    "prediction_draw_no": "686",
                    "combo_index": str(index + 1),
                    "numbers": " ".join(f"{number:02d}" for number in ticket),
                    "model_id": "model",
                    "model_score": "1",
                    "source_model": "model.json",
                    "prediction_method": "dual_full_period",
                    "ensemble_score": f"{relative_confidence(index):.2f}",
                    "support_models": "test",
                    "created_at": "2026-07-14T00:00:00+00:00",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "latest_prediction_report.txt"
            write_final_report(
                report,
                rows,
                tickets,
                [],
                Counter(number for ticket in tickets for number in ticket),
                4,
                4,
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("04 26 27 28 29 30 31", text)
            self.assertIn("相対スコア=0.75", text)
            self.assertNotIn("71271.89 / 方式", text)


if __name__ == "__main__":
    unittest.main()
