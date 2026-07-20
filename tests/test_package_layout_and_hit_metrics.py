#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loto7.evaluation.core import EVALUATOR_VERSION
from loto7.evaluation.hit_metrics import summarize_hit_metrics
from loto7.validation.hit_rate_gate import evaluate_nested_hit_gate
from scripts.evaluation_core import EVALUATOR_VERSION as LEGACY_EVALUATOR_VERSION


class PackageAndHitMetricTests(unittest.TestCase):
    def test_legacy_wrapper_uses_packaged_evaluator(self) -> None:
        self.assertEqual(EVALUATOR_VERSION, LEGACY_EVALUATOR_VERSION)

    def test_draw_level_high_match_metrics(self) -> None:
        metrics = summarize_hit_metrics(
            [3, 4, 5, 6],
            ticket_main_matches=[3, 4, 4, 5, 6],
            portfolios=[[
                [1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 8, 9, 10, 11],
            ]],
        )
        self.assertEqual(metrics["draw_main4_plus_count"], 3)
        self.assertEqual(metrics["draw_main5_plus_count"], 2)
        self.assertEqual(metrics["draw_main6_plus_count"], 1)
        self.assertEqual(metrics["draw_main4_plus_rate_percent"], 75.0)
        self.assertEqual(metrics["average_portfolio_unique_numbers"], 11.0)
        self.assertGreater(float(metrics["hit_objective_score"]), 0.0)

    def test_gate_rejects_single_improved_fold(self) -> None:
        def metrics(score: float, main4: int, main5: int, average: float) -> dict[str, object]:
            return {
                "target_draws": 10,
                "hit_objective_score": score,
                "draw_main4_plus_count": main4,
                "draw_main5_plus_count": main5,
                "draw_main6_plus_count": 0,
                "draw_main7_count": 0,
                "average_max_main_match": average,
            }

        baseline = metrics(10.0, 2, 0, 3.0)
        nested = {"folds": [
            {"label": "a", "baseline_metrics": baseline, "candidate_metrics": metrics(10.0, 2, 0, 3.0)},
            {"label": "b", "baseline_metrics": baseline, "candidate_metrics": metrics(10.0, 2, 0, 3.0)},
            {"label": "c", "baseline_metrics": baseline, "candidate_metrics": metrics(11.0, 3, 1, 3.2)},
        ]}
        decision = evaluate_nested_hit_gate(nested)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["positive_folds"], 1)


if __name__ == "__main__":
    unittest.main()
