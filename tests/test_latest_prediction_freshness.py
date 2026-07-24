from __future__ import annotations

import unittest

from scripts.assert_latest_prediction_fresh import validate_freshness


class LatestPredictionFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = [
            {"抽せん日": "2026-07-10", "回別": "第685回"},
            {"抽せん日": "2026-07-17", "回別": "第686回"},
        ]
        self.prediction = [
            {
                "base_latest_draw_no": "686",
                "prediction_draw_no": "687",
                "numbers": "01 02 03 04 05 06 07",
            }
        ]
        self.history = [
            {"抽せん日": "2026-07-17", "回別": "第686回"},
            {"抽せん日": "2026-07-24", "回別": "第687回"},
        ]

    def test_next_draw_prediction_is_accepted(self) -> None:
        result = validate_freshness(self.dataset, self.prediction, self.history)
        self.assertEqual(result["latest_actual_draw_no"], 686)
        self.assertEqual(result["prediction_draw_no"], 687)
        self.assertEqual(result["history_target_count"], 1)

    def test_stale_prediction_is_rejected(self) -> None:
        stale = [dict(self.prediction[0], base_latest_draw_no="685", prediction_draw_no="686")]
        with self.assertRaisesRegex(ValueError, "prediction base is stale"):
            validate_freshness(self.dataset, stale, self.history)

    def test_skipped_draw_is_rejected(self) -> None:
        skipped = [dict(self.prediction[0], prediction_draw_no="688")]
        with self.assertRaisesRegex(ValueError, "prediction target is stale"):
            validate_freshness(self.dataset, skipped, self.history)

    def test_duplicate_history_row_is_rejected(self) -> None:
        duplicate = self.history + [{"抽せん日": "2026-07-24", "回別": "第687回"}]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_freshness(self.dataset, self.prediction, duplicate)

    def test_missing_history_row_is_rejected(self) -> None:
        missing = [row for row in self.history if row["回別"] != "第687回"]
        with self.assertRaisesRegex(ValueError, "count=0"):
            validate_freshness(self.dataset, self.prediction, missing)


if __name__ == "__main__":
    unittest.main()
