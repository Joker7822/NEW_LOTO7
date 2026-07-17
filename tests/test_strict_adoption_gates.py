#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from scripts.build_generation4_prediction_strict import main as strict_generation4_main
from scripts.strict_adoption_gates import (
    nested_total_roi_gate,
    null_league_adoption_gate,
    recalibrated_conformal_number_pool,
)


@dataclass
class FakeDraw:
    draw_no: int
    date: str
    main: tuple


class StrictAdoptionGateTests(unittest.TestCase):
    def make_draws(self, count=180):
        draws = []
        for index in range(count):
            numbers = tuple(((index + offset) % 37) + 1 for offset in range(7))
            draws.append(FakeDraw(index + 1, f"2024-01-{(index % 28) + 1:02d}", numbers))
        return draws

    def test_null_league_false_completely_rejects(self):
        result = null_league_adoption_gate(
            {"decision": {"passed": False}, "model_percentile": 0.4, "pbo": 0.3},
            require_available=True,
        )
        self.assertFalse(result["adoption_allowed"])

    def test_missing_null_league_is_fail_closed_when_required(self):
        result = null_league_adoption_gate(None, require_available=True)
        self.assertFalse(result["adoption_allowed"])

    def test_generation4_entrypoint_stops_before_prediction_on_null_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            null_summary = root / "null.json"
            gate_summary = root / "gate.json"
            null_summary.write_text(
                json.dumps({"decision": {"passed": False}, "model_percentile": 0.4, "pbo": 0.3}),
                encoding="utf-8",
            )
            result = strict_generation4_main(
                [
                    "--require-null-league-summary",
                    "--null-league-summary",
                    str(null_summary),
                    "--strict-gate-summary",
                    str(gate_summary),
                ]
            )
            self.assertEqual(result, 2)
            payload = json.loads(gate_summary.read_text(encoding="utf-8"))
            self.assertFalse(payload["adoption_allowed"])

    def test_nested_total_roi_below_baseline_is_rejected(self):
        summary = {
            "reference_model_id": "candidate",
            "future_leakage_detected": False,
            "folds": [
                {
                    "baseline_metrics": {"total_cost": 1000, "total_payout": 200},
                    "candidate_metrics": {"total_cost": 1000, "total_payout": 150},
                },
                {
                    "baseline_metrics": {"total_cost": 1000, "total_payout": 100},
                    "candidate_metrics": {"total_cost": 1000, "total_payout": 100},
                },
            ],
        }
        result = nested_total_roi_gate(
            summary,
            min_candidate_roi_percent=8.0,
            min_roi_delta_percent=0.0,
            expected_model_id="candidate",
        )
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["candidate"]["roi_percent"], 12.5)
        self.assertAlmostEqual(result["roi_delta_percent"], -2.5)

    def test_nested_total_roi_above_both_thresholds_passes(self):
        summary = {
            "reference_model_id": "candidate",
            "future_leakage_detected": False,
            "folds": [
                {
                    "baseline_metrics": {"total_cost": 1000, "total_payout": 100},
                    "candidate_metrics": {"total_cost": 1000, "total_payout": 180},
                }
            ],
        }
        result = nested_total_roi_gate(
            summary,
            min_candidate_roi_percent=8.0,
            min_roi_delta_percent=0.0,
            expected_model_id="candidate",
        )
        self.assertTrue(result["passed"])

    def test_conformal_recalibration_uses_prior_only_and_bounds_pool(self):
        fake_scores = {number: float(38 - number) for number in range(1, 38)}
        with patch("scripts.generation4_core.exp_weighted_number_scores", return_value=fake_scores):
            result = recalibrated_conformal_number_pool(
                self.make_draws(),
                alpha=0.20,
                calibration_draws=60,
                min_pool_size=14,
                max_pool_size=24,
                required_hits=4,
            )
        self.assertFalse(result["future_data_used"])
        self.assertGreaterEqual(result["pool_size"], 14)
        self.assertLessEqual(result["pool_size"], 24)
        self.assertIn("empirical_draw_coverage", result)
        self.assertEqual(result["required_main_hits"], 4)
        self.assertEqual(
            result["recalibration_method"],
            "rolling_prior_top_k_minimum_hit_coverage_v1",
        )


if __name__ == "__main__":
    unittest.main()
