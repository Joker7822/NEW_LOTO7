from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generation5_selection_evolver import load_selection_seeds, selection_candidate_key
from scripts.split_null_seed_bank import split_seed_bank


class Generation5SelectionNullTests(unittest.TestCase):
    def test_split_seed_bank_physically_isolates_selection_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.json"
            selection_path = root / "selection.json"
            final_path = root / "final.json"
            master.write_text(
                json.dumps(
                    {
                        "kind": "master",
                        "version": "v1",
                        "dataset_sha256": "dataset",
                        "evaluator_version": "evaluator",
                        "phases": {
                            "learning": list(range(1, 701)),
                            "selection": list(range(1001, 1151)),
                            "final": list(range(2001, 2151)),
                        },
                    }
                ),
                encoding="utf-8",
            )
            selection, final = split_seed_bank(
                master,
                selection_output=selection_path,
                final_output=final_path,
            )
            self.assertEqual(set(selection["phases"]), {"selection"})
            self.assertEqual(set(final["phases"]), {"final"})
            selection_values = set(selection["phases"]["selection"])
            final_values = set(final["phases"]["final"])
            self.assertEqual(len(selection_values), 150)
            self.assertEqual(len(final_values), 150)
            self.assertFalse(selection_values & final_values)
            self.assertNotIn("final", json.loads(selection_path.read_text(encoding="utf-8"))["phases"])
            self.assertNotIn("selection", json.loads(final_path.read_text(encoding="utf-8"))["phases"])

    def test_selection_loader_rejects_any_non_selection_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "phase": "selection",
                        "phases": {
                            "selection": list(range(1, 151)),
                            "final": list(range(1001, 1151)),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_selection_seeds(path, expected_count=150)

    def test_selection_key_prefers_internal_gate_then_null_resistance(self) -> None:
        internally_valid = {
            "adoption_passed": True,
            "decision": {"passed": False, "wilson_ci_upper": 0.20, "exceedance": 0.12},
            "null_margin_vs_adjusted_p90": 0.1,
            "internal_key": [1.0],
        }
        invalid_but_null_pass = {
            "adoption_passed": False,
            "decision": {"passed": True, "wilson_ci_upper": 0.08, "exceedance": 0.03},
            "null_margin_vs_adjusted_p90": 0.5,
            "internal_key": [5.0],
        }
        self.assertGreater(
            selection_candidate_key(internally_valid),
            selection_candidate_key(invalid_but_null_pass),
        )

        stronger_null = {
            "adoption_passed": True,
            "decision": {"passed": True, "wilson_ci_upper": 0.07, "exceedance": 0.02},
            "null_margin_vs_adjusted_p90": 0.4,
            "internal_key": [0.5],
        }
        weaker_null = {
            "adoption_passed": True,
            "decision": {"passed": True, "wilson_ci_upper": 0.09, "exceedance": 0.04},
            "null_margin_vs_adjusted_p90": 0.6,
            "internal_key": [5.0],
        }
        self.assertGreater(selection_candidate_key(stronger_null), selection_candidate_key(weaker_null))

    def test_workflow_runs_production_evolution_before_fixed_final(self) -> None:
        workflow = Path(".github/workflows/loto7_generation5.yml").read_text(encoding="utf-8")
        self.assertIn("github.event_name == 'push'", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("scripts/generation5_selection_evolver.py", workflow)
        self.assertIn("--selection-seed-bank outputs/generation5/selection_null_seed_bank.json", workflow)
        self.assertIn("--seed-bank outputs/generation5/final_null_seed_bank.json", workflow)
        self.assertLess(
            workflow.index("scripts/generation5_selection_evolver.py"),
            workflow.index("Run fixed final financial Null League"),
        )


if __name__ == "__main__":
    unittest.main()
