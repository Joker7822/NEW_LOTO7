from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_repository_architecture import detect_production_writers, workflow_name


class ProductionOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "config/repository_layout.json").read_text(encoding="utf-8"))
        workflow_dir = ROOT / ".github/workflows"
        self.workflows = {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")])
        }

    def test_publisher_is_the_only_production_writer(self) -> None:
        canonical = self.config["production_workflow"]
        writers = detect_production_writers(self.workflows, self.config["production_outputs"])
        for output in self.config["production_outputs"]:
            self.assertEqual(writers.get(output), [canonical], output)

    def test_generation4_is_diagnostic_only(self) -> None:
        path = self.config["generation4_evaluation_workflow"]
        text = self.workflows[path]
        self.assertEqual(
            workflow_name(text, Path(path).stem),
            self.config["generation4_evaluation_workflow_name"],
        )
        self.assertNotIn("--prediction outputs/evolution_best_prediction.csv", text)
        self.assertNotIn("--prediction-report outputs/holdout/latest_prediction_report.txt", text)
        self.assertNotIn("--output outputs/evolution_prediction_history_result.txt", text)
        self.assertIn("--prediction outputs/generation4/candidate_prediction.csv", text)

    def test_publisher_uses_latest_state_wins_and_generation4_trigger(self) -> None:
        path = self.config["production_workflow"]
        text = self.workflows[path]
        self.assertEqual(workflow_name(text, Path(path).stem), self.config["production_workflow_name"])
        self.assertIn("cancel-in-progress: true", text)
        for source in self.config["workflow_run_sources"]:
            self.assertIn(f"- {source}", text)


if __name__ == "__main__":
    unittest.main()
