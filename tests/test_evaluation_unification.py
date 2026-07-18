from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation_core import EVALUATOR_VERSION, financial_metrics
from scripts.verify_evaluator_consistency import verify


class EvaluationUnificationTests(unittest.TestCase):
    def test_financial_metrics_use_profit_roi_and_separate_payout_roi(self) -> None:
        metrics = financial_metrics(
            total_cost=1000,
            total_payout=1250,
            total_tickets=10,
            winning_tickets=2,
            target_draws=2,
            winning_draws=1,
        )
        self.assertEqual(metrics["profit"], 250)
        self.assertEqual(metrics["roi_percent"], 25.0)
        self.assertEqual(metrics["profit_roi_percent"], 25.0)
        self.assertEqual(metrics["payout_roi_percent"], 125.0)
        self.assertEqual(metrics["ticket_hit_rate_percent"], 20.0)
        self.assertEqual(metrics["draw_hit_rate_percent"], 50.0)
        self.assertEqual(metrics["evaluator_version"], EVALUATOR_VERSION)

    def test_holdout_and_role_best_rows_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.csv"
            role = root / "role.csv"
            with holdout.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["draw_no", "combo_index", "ticket", "main_match", "bonus_match", "rank", "prize_amount"])
                writer.writeheader()
                writer.writerow({"draw_no": 2, "combo_index": 1, "ticket": "01 02 03 04 05 06 07", "main_match": 4, "bonus_match": 0, "rank": "5等", "prize_amount": 1500})
            with role.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["system", "target_draw_no", "ticket_index", "numbers", "main_match", "bonus_match", "rank", "payout"])
                writer.writeheader()
                writer.writerow({"system": "best_model", "target_draw_no": 2, "ticket_index": 1, "numbers": "01 02 03 04 05 06 07", "main_match": 4, "bonus_match": 0, "rank": "5等", "payout": 1500})
            payload = verify(holdout, role, 300)
            self.assertTrue(payload["passed"])
            self.assertTrue(payload["metrics_equal"])

    def test_stale_role_resume_output_is_deleted(self) -> None:
        from scripts.backtest_role_ensemble import load_resume_details

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.json"
            model.write_text("{}", encoding="utf-8")
            output = root / "detail.csv"
            output.write_text("stale\n", encoding="utf-8")
            state = root / "state.json"
            state.write_text(json.dumps({"csv": "loto7.csv", "best_model": str(model), "genome_id": "g1", "purchase_count": 5, "unit_cost": 300, "holdout_start_draw": 2, "min_train_draws": 1, "target_draws_total": 1}), encoding="utf-8")
            args = SimpleNamespace(
                resume=True,
                state=str(state),
                output=str(output),
                csv="loto7.csv",
                best_model=str(model),
                purchase_count=5,
                unit_cost=300,
                holdout_start_draw=2,
                min_train_draws=1,
                max_targets=0,
            )
            rows, completed, missing, removed = load_resume_details(args, "g1", [2])
            self.assertEqual(rows, [])
            self.assertEqual(completed, set())
            self.assertFalse(output.exists())

    def test_shadow_evaluate_only_does_not_append_latest_prediction(self) -> None:
        from scripts.update_generation4_shadow_history import main as update_shadow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draws = root / "loto7.csv"
            headers = ["抽せん日", "本数字", "ボーナス数字", "回別"] + [f"{rank}当選金額" for rank in ["1等", "2等", "3等", "4等", "5等", "6等"]]
            with draws.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerow({"抽せん日": "2026-01-01", "本数字": "01 02 03 04 05 06 07", "ボーナス数字": "08 09", "回別": "第1回", "1等当選金額": "1000円"})
                writer.writerow({"抽せん日": "2026-01-08", "本数字": "01 02 03 04 05 06 07", "ボーナス数字": "08 09", "回別": "第2回", "1等当選金額": "1000円"})
                writer.writerow({"抽せん日": "2026-01-15", "本数字": "10 11 12 13 14 15 16", "ボーナス数字": "17 18", "回別": "第3回", "1等当選金額": "1000円"})
            history = root / "shadow.csv"
            fields = ["prediction_draw_no", "prediction_date", "strategy", "tickets_json", "status", "actual_main", "actual_bonus", "max_main_match", "total_main_matches", "winning_tickets", "total_payout", "utility", "created_at"]
            with history.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"prediction_draw_no": "2", "strategy": "generation4", "tickets_json": json.dumps(["01 02 03 04 05 06 07"]), "status": "pending"})
            latest = root / "latest.json"
            latest.write_text(json.dumps({"prediction_draw_no": 3, "strategies": {"generation4": ["10 11 12 13 14 15 16"]}}), encoding="utf-8")
            summary = root / "summary.json"
            report = root / "report.txt"
            result = update_shadow([
                "--csv", str(draws), "--latest-shadow", str(latest), "--history", str(history),
                "--summary", str(summary), "--report", str(report), "--evaluate-only",
            ])
            self.assertEqual(result, 0)
            with history.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prediction_draw_no"], "2")
            self.assertEqual(rows[0]["status"], "evaluated")
            self.assertEqual(rows[0]["total_payout"], "1000")


if __name__ == "__main__":
    unittest.main()
