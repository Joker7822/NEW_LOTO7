#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from scripts.build_dual_model_prediction import Candidate
from scripts.build_generation4_prediction import select_generation4_portfolio
from scripts.generation4_core import (
    allocate_quotas,
    conformal_number_pool,
    detect_change_point,
    dpp_logdet,
    eprocess_from_history,
    hypergraph_weights,
)
from scripts.seal_generation4_prediction import canonical_digest


@dataclass
class FakeDraw:
    draw_no: int
    date: str
    main: tuple
    bonus: tuple = ()


class Generation4PipelineTests(unittest.TestCase):
    def make_draws(self, count=180):
        draws = []
        for index in range(count):
            start = (index * 3) % 37
            numbers = tuple(sorted({((start + offset * 5) % 37) + 1 for offset in range(7)}))
            while len(numbers) < 7:
                numbers = tuple(sorted(set(numbers) | {((start + len(numbers) * 7) % 37) + 1}))
            draws.append(FakeDraw(index + 1, f"2024-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}", numbers[:7]))
        return draws

    def make_candidate(self, ticket, source, score):
        return Candidate(
            ticket=tuple(ticket), source=source, model_id=f"{source}_model",
            model_score=score, source_model=f"{source}.json", method=f"generation4_{source}",
            support="test", individual_score=score, raw_rank=0,
            created_at="2026-07-16T00:00:00+00:00")

    def test_conformal_pool_uses_prior_only_and_is_bounded(self):
        result = conformal_number_pool(
            self.make_draws(), alpha=0.20, calibration_draws=60,
            min_pool_size=14, max_pool_size=24)
        self.assertFalse(result["future_data_used"])
        self.assertGreaterEqual(result["pool_size"], 14)
        self.assertLessEqual(result["pool_size"], 24)
        self.assertEqual(len(result["numbers"]), len(set(result["numbers"])))

    def test_change_point_score_is_bounded(self):
        result = detect_change_point(self.make_draws(180), recent_window=52, reference_window=104)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_quota_allocation_sums_to_five(self):
        quotas = allocate_quotas(
            {"full": 0.30, "recent": 0.30, "super": 0.15, "regime": 0.25},
            purchase_count=5, super_independent=True)
        self.assertEqual(sum(quotas.values()), 5)
        self.assertGreaterEqual(quotas["full"], 1)
        self.assertGreaterEqual(quotas["recent"], 1)
        self.assertGreaterEqual(quotas["regime"], 1)

    def test_dpp_prefers_non_identical_features(self):
        same = [(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 8)]
        diverse = [(1, 2, 3, 4, 5, 6, 7), (20, 21, 22, 23, 24, 25, 26)]
        same_score = dpp_logdet(same, ["full", "full"], [100.0, 100.0])
        diverse_score = dpp_logdet(diverse, ["full", "recent"], [100.0, 100.0])
        self.assertGreater(diverse_score, same_score)

    def test_generation4_selector_enforces_hard_constraints(self):
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
        graph = hypergraph_weights(self.make_draws(80))
        selected, summary = select_generation4_portfolio(
            pools, {"full": 2, "recent": 1, "super": 0, "regime": 2},
            graph_weights=graph, conformal_numbers=set(range(1, 25)),
            max_number_usage=3, max_pair_overlap=4, beam_width=100,
            candidates_per_step=20, dpp_weight=3.0, hypergraph_weight=0.075,
            conformal_weight=0.55)
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(summary["max_number_usage"], 3)
        self.assertLessEqual(summary["max_pair_overlap"], 4)
        self.assertEqual(summary["post_selection_number_replacements"], 0)

    def test_eprocess_requires_minimum_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["prediction_draw_no", "strategy", "status", "utility"])
                writer.writeheader()
                for draw_no in range(1, 11):
                    writer.writerow({"prediction_draw_no": draw_no, "strategy": "generation4", "status": "evaluated", "utility": 0.7})
                    writer.writerow({"prediction_draw_no": draw_no, "strategy": "beam_baseline", "status": "evaluated", "utility": 0.4})
            result = eprocess_from_history(str(path), min_evaluated_draws=30)
            self.assertEqual(result["evaluated_draws"], 10)
            self.assertEqual(result["decision"], "continue")

    def test_manifest_digest_is_canonical(self):
        left = canonical_digest({"b": 2, "a": 1})
        right = canonical_digest({"a": 1, "b": 2})
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)


if __name__ == "__main__":
    unittest.main()
