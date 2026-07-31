from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/loto7_prediction_freshness_watchdog.yml"


class PredictionFreshnessWatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_watchdog_checks_canonical_freshness_inputs(self) -> None:
        self.assertIn("scripts/assert_latest_prediction_fresh.py", self.text)
        self.assertIn("--csv loto7.csv", self.text)
        self.assertIn("--prediction outputs/evolution_best_prediction.csv", self.text)
        self.assertIn("--history outputs/evolution_prediction_history.csv", self.text)

    def test_watchdog_dispatches_only_the_canonical_publisher(self) -> None:
        self.assertIn("createWorkflowDispatch", self.text)
        self.assertIn("workflow_id: 'loto7_refresh_latest_prediction.yml'", self.text)
        self.assertIn("steps.freshness.outcome == 'failure'", self.text)
        self.assertNotIn("git push", self.text)
        self.assertNotIn("git add", self.text)
        self.assertNotIn("scripts/build_generation4_prediction.py", self.text)

    def test_watchdog_runs_after_evolution_and_during_result_window(self) -> None:
        self.assertIn("- LOTO7 Evolution Trainer", self.text)
        self.assertIn('cron: "30 11,12,13,14,15 * * 5"', self.text)
        self.assertIn("actions: write", self.text)
        self.assertIn("contents: read", self.text)


if __name__ == "__main__":
    unittest.main()
