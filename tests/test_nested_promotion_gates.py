#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.promote_nested_candidate as promotion


class NestedPromotionGateTests(unittest.TestCase):
    def make_nested(self, candidate_id, roi_deltas, top1_deltas=None):
        top1_deltas = roi_deltas if top1_deltas is None else top1_deltas
        folds = [
            {
                "roi_delta_percent": roi_delta,
                "top1_removed_roi_delta_percent": top1_delta,
            }
            for roi_delta, top1_delta in zip(roi_deltas, top1_deltas)
        ]
        ordered_roi = sorted(float(value) for value in roi_deltas)
        ordered_top1 = sorted(float(value) for value in top1_deltas)
        middle = len(ordered_roi) // 2
        return {
            "reference_model_id": candidate_id,
            "future_leakage_detected": False,
            "folds": folds,
            # These legacy counters are intentionally wrong in some tests. The
            # promotion script must recalculate from fold rows.
            "positive_roi_delta_folds": len(folds),
            "median_roi_delta_percent": ordered_roi[middle],
            "worst_roi_delta_percent": min(ordered_roi),
            "median_top1_removed_roi_delta_percent": ordered_top1[middle],
        }

    def metrics(self, *, roi, top1_removed, share, top2_removed=5.0, bootstrap=4.0):
        return {
            "roi_percent": roi,
            "roi_excluding_top1_percent": top1_removed,
            "roi_excluding_top2_percent": top2_removed,
            "bootstrap_roi_percent_p05": bootstrap,
            "top1_payout_share": share,
        }

    def run_gate(
        self,
        *,
        baseline_content="baseline",
        candidate_content="candidate",
        baseline_id="baseline-id",
        candidate_id="candidate-id",
        nested=None,
        baseline_metrics=None,
        candidate_metrics=None,
    ):
        nested = nested or self.make_nested(candidate_id, [0.5, 0.5, 0.5])
        baseline_metrics = baseline_metrics or self.metrics(
            roi=10.0,
            top1_removed=8.0,
            share=0.30,
        )
        candidate_metrics = candidate_metrics or self.metrics(
            roi=10.5,
            top1_removed=8.5,
            share=0.50,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            best = root / "best.json"
            summary = root / "nested.json"
            decision = root / "decision.json"
            report = root / "report.txt"
            baseline.write_text(baseline_content, encoding="utf-8")
            candidate.write_text(candidate_content, encoding="utf-8")
            summary.write_text(json.dumps(nested), encoding="utf-8")

            def fake_load_genome(path):
                return SimpleNamespace(
                    id=baseline_id if Path(path) == baseline else candidate_id
                )

            with patch.object(promotion, "load_genome", side_effect=fake_load_genome), patch.object(
                promotion,
                "evaluate",
                side_effect=[baseline_metrics, candidate_metrics],
            ):
                result = promotion.main(
                    [
                        "--baseline-model",
                        str(baseline),
                        "--candidate-model",
                        str(candidate),
                        "--best-model",
                        str(best),
                        "--nested-summary",
                        str(summary),
                        "--decision",
                        str(decision),
                        "--report",
                        str(report),
                    ]
                )
            payload = json.loads(decision.read_text(encoding="utf-8"))
            best_content = best.read_text(encoding="utf-8") if best.exists() else None
            return result, payload, best_content

    def test_equal_and_subthreshold_folds_are_not_positive(self):
        nested = self.make_nested("candidate", [0.0, 0.49, 0.5])
        self.assertEqual(promotion.count_improved_folds(nested, 0.5), 1)
        self.assertEqual(promotion.count_improved_folds(nested, 0.0), 3)

    def test_same_model_id_is_rejected(self):
        _, payload, best_content = self.run_gate(
            baseline_content="one",
            candidate_content="two",
            baseline_id="same-id",
            candidate_id="same-id",
            nested=self.make_nested("same-id", [1.0, 1.0, 1.0]),
            candidate_metrics=self.metrics(roi=11.0, top1_removed=9.0, share=0.20),
        )
        self.assertFalse(payload["decision"]["promoted"])
        self.assertIsNone(best_content)
        self.assertTrue(
            any("model ID is identical" in item for item in payload["decision"]["warnings"])
        )

    def test_same_sha_is_rejected_even_with_different_ids(self):
        _, payload, best_content = self.run_gate(
            baseline_content="identical",
            candidate_content="identical",
            nested=self.make_nested("candidate-id", [1.0, 1.0, 1.0]),
            candidate_metrics=self.metrics(roi=11.0, top1_removed=9.0, share=0.20),
        )
        self.assertFalse(payload["decision"]["promoted"])
        self.assertIsNone(best_content)
        self.assertTrue(
            any("SHA-256 is identical" in item for item in payload["decision"]["warnings"])
        )

    def test_zero_improvement_and_high_payout_dependency_are_rejected(self):
        _, payload, best_content = self.run_gate(
            nested=self.make_nested("candidate-id", [0.0, 0.0, 0.0]),
            candidate_metrics=self.metrics(roi=10.0, top1_removed=8.0, share=0.817378),
        )
        warnings = payload["decision"]["warnings"]
        self.assertFalse(payload["decision"]["promoted"])
        self.assertIsNone(best_content)
        self.assertTrue(any("materially improved folds failed" in item for item in warnings))
        self.assertTrue(any("focus ROI delta failed" in item for item in warnings))
        self.assertTrue(any("top1 payout share failed" in item for item in warnings))

    def test_exact_minimum_improvement_and_share_boundary_pass(self):
        _, payload, best_content = self.run_gate(
            nested=self.make_nested("candidate-id", [0.5, 0.5, 0.5]),
            candidate_metrics=self.metrics(roi=10.5, top1_removed=8.5, share=0.50),
        )
        self.assertTrue(payload["decision"]["promoted"])
        self.assertTrue(payload["decision"]["copy_performed"])
        self.assertEqual(best_content, "candidate")
        self.assertEqual(payload["thresholds"]["max_top1_payout_share"], 0.50)
        self.assertEqual(payload["thresholds"]["min_focus_roi_delta_percent"], 0.5)


if __name__ == "__main__":
    unittest.main()
