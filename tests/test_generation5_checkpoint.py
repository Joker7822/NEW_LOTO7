#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import generation5_checkpoint_runner as runner


class Generation5CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_cwd = Path.cwd()
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(self.temp_dir.name)
        Path("outputs/generation5").mkdir(parents=True, exist_ok=True)
        Path("outputs/state/generation5").mkdir(parents=True, exist_ok=True)
        Path("loto7.csv").write_text("draw\n1\n", encoding="utf-8")
        Path("loto7_best_model.json").write_text(
            json.dumps({"genome": {"id": "baseline"}}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        os.chdir(self.previous_cwd)
        self.temp_dir.cleanup()

    @staticmethod
    def fake_evolver(calls: list[list[str]]):
        def invoke(arguments: list[str]) -> int:
            calls.append(list(arguments))

            def value(option: str) -> str:
                return arguments[arguments.index(option) + 1]

            candidate = Path(value("--candidate-model"))
            summary = Path(value("--summary"))
            report = Path(value("--report"))
            candidate.parent.mkdir(parents=True, exist_ok=True)
            summary.parent.mkdir(parents=True, exist_ok=True)
            report.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(
                json.dumps({"genome": {"id": f"candidate-{len(calls)}"}}),
                encoding="utf-8",
            )
            summary.write_text(
                json.dumps({"kind": "fake_generation5_summary"}),
                encoding="utf-8",
            )
            report.write_text("fake report\n", encoding="utf-8")
            return 0

        return invoke

    def test_resume_uses_canonical_candidate_when_stage_file_is_missing(self) -> None:
        canonical = Path("outputs/generation5/generation5_candidate_model.json")
        canonical.write_text(
            json.dumps({"genome": {"id": "candidate-1"}}),
            encoding="utf-8",
        )
        checkpoint = {
            "objective_version": runner.OBJECTIVE_VERSION,
            "dataset_sha256": runner.sha256("loto7.csv"),
            "baseline_model_sha256": runner.sha256("loto7_best_model.json"),
            "completed_generation": 1,
            "candidate": "outputs/generation5/stages/generation_001_candidate.json",
            "canonical_candidate": str(canonical),
        }
        Path("outputs/state/generation5/checkpoint.json").write_text(
            json.dumps(checkpoint),
            encoding="utf-8",
        )

        calls: list[list[str]] = []
        with patch.object(runner, "run_generation5", self.fake_evolver(calls)):
            result = runner.main(
                [
                    "--generations",
                    "2",
                    "--archive-size",
                    "12",
                    "--max-runtime-minutes-per-stage",
                    "20",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        seed_index = calls[0].index("--seed-patterns")
        self.assertIn(str(canonical), calls[0][seed_index + 1 :])
        self.assertIn("--archive-size", calls[0])
        self.assertEqual(calls[0][calls[0].index("--archive-size") + 1], "12")

        updated = json.loads(
            Path("outputs/state/generation5/checkpoint.json").read_text(encoding="utf-8")
        )
        self.assertEqual(updated["completed_generation"], 2)
        self.assertEqual(updated["candidate"], str(canonical))
        self.assertTrue(updated["stage_candidate"].endswith("generation_002_candidate.json"))

        summary = json.loads(
            Path("outputs/generation5/generation5_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["checkpoint"]["completed_generation"], 2)
        self.assertEqual(summary["checkpoint"]["resumed_from"], str(canonical))

    def test_stage_seed_is_deterministic_and_changes_by_generation(self) -> None:
        dataset = "a" * 64
        baseline = "b" * 64
        first = runner.stage_seed(dataset, baseline, 1, 0)
        repeated = runner.stage_seed(dataset, baseline, 1, 0)
        second = runner.stage_seed(dataset, baseline, 2, 0)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)

    def test_managed_evolver_options_are_rejected_from_passthrough(self) -> None:
        with self.assertRaises(SystemExit):
            runner.main(
                [
                    "--max-runtime-minutes-per-stage",
                    "20",
                    "--seed",
                    "123",
                ]
            )


if __name__ == "__main__":
    unittest.main()
