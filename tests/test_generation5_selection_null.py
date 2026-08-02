from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loto7.evaluation.null_permutation import adaptive_null_test
from scripts.generation5_selection_evolver import (
    checkpoint_stability_summary,
    load_selection_seeds,
    parse_selection_checkpoints,
    selection_candidate_key,
)
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
            self.assertNotIn(
                "final",
                json.loads(selection_path.read_text(encoding="utf-8"))["phases"],
            )
            self.assertNotIn(
                "selection",
                json.loads(final_path.read_text(encoding="utf-8"))["phases"],
            )

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

    def test_predeclared_selection_checkpoints_end_at_full_phase(self) -> None:
        self.assertEqual(
            parse_selection_checkpoints("50,100,150", seed_count=150),
            [50, 100, 150],
        )
        with self.assertRaises(ValueError):
            parse_selection_checkpoints("50,100", seed_count=150)
        with self.assertRaises(ValueError):
            parse_selection_checkpoints("50,100,151", seed_count=150)

    def test_selection_key_prefers_internal_gate_then_null_resistance(self) -> None:
        internally_valid = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": False,
                "explicit_failures": 0,
                "worst_exceedance": 0.12,
                "minimum_margin_vs_adjusted_p90": 0.1,
                "exceedance_stddev": 0.01,
                "exceedance_deterioration": 0.01,
            },
            "decision": {
                "passed": False,
                "wilson_ci_upper": 0.20,
                "exceedance": 0.12,
            },
            "null_margin_vs_adjusted_p90": 0.1,
            "internal_key": [1.0],
        }
        invalid_but_null_pass = {
            "adoption_passed": False,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.03,
                "minimum_margin_vs_adjusted_p90": 0.5,
                "exceedance_stddev": 0.0,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.08,
                "exceedance": 0.03,
            },
            "null_margin_vs_adjusted_p90": 0.5,
            "internal_key": [5.0],
        }
        self.assertGreater(
            selection_candidate_key(internally_valid),
            selection_candidate_key(invalid_but_null_pass),
        )

        stronger_null = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.03,
                "minimum_margin_vs_adjusted_p90": 0.4,
                "exceedance_stddev": 0.002,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.07,
                "exceedance": 0.02,
            },
            "null_margin_vs_adjusted_p90": 0.4,
            "internal_key": [0.5],
        }
        weaker_null = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.05,
                "minimum_margin_vs_adjusted_p90": 0.2,
                "exceedance_stddev": 0.01,
                "exceedance_deterioration": 0.01,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.09,
                "exceedance": 0.04,
            },
            "null_margin_vs_adjusted_p90": 0.6,
            "internal_key": [5.0],
        }
        self.assertGreater(
            selection_candidate_key(stronger_null),
            selection_candidate_key(weaker_null),
        )

    def test_checkpoint_stability_rejects_interim_failure(self) -> None:
        stable = checkpoint_stability_summary(
            {
                "decision": {"passed": True},
                "adaptive_checkpoints": [
                    {
                        "simulations": 50,
                        "exceedance": 0.04,
                        "observed_margin_vs_adjusted_p90": 0.3,
                        "verdict": "continue",
                    },
                    {
                        "simulations": 100,
                        "exceedance": 0.03,
                        "observed_margin_vs_adjusted_p90": 0.4,
                        "verdict": "pass",
                    },
                    {
                        "simulations": 150,
                        "exceedance": 0.02,
                        "observed_margin_vs_adjusted_p90": 0.5,
                        "verdict": "pass",
                    },
                ],
            },
            max_exceedance=0.10,
        )
        self.assertTrue(stable["passed"])

        unstable = checkpoint_stability_summary(
            {
                "decision": {"passed": True},
                "adaptive_checkpoints": [
                    {
                        "simulations": 50,
                        "exceedance": 0.14,
                        "observed_margin_vs_adjusted_p90": -0.2,
                        "verdict": "fail",
                    },
                    {
                        "simulations": 100,
                        "exceedance": 0.08,
                        "observed_margin_vs_adjusted_p90": 0.1,
                        "verdict": "continue",
                    },
                    {
                        "simulations": 150,
                        "exceedance": 0.04,
                        "observed_margin_vs_adjusted_p90": 0.3,
                        "verdict": "pass",
                    },
                ],
            },
            max_exceedance=0.10,
        )
        self.assertFalse(unstable["passed"])
        self.assertEqual(unstable["explicit_failures"], 1)

    def test_null_test_can_observe_all_selection_checkpoints_without_early_stop(self) -> None:
        portfolios = [[(1, 2, 3, 4, 5, 6, 7)] * 5] * 12
        mains = [(1, 2, 3, 4, 5, 6, 7)] * 12
        result = adaptive_null_test(
            portfolios=portfolios,
            mains=mains,
            seeds=list(range(1, 13)),
            checkpoints=[4, 8, 12],
            search_width=2,
            max_exceedance=0.10,
            stop_early=False,
        )
        checkpoints = result["adaptive_checkpoints"]
        self.assertEqual([item["simulations"] for item in checkpoints], [4, 8, 12])
        self.assertEqual(result["null_distribution"]["search_adjusted_count"], 12)
        for item in checkpoints:
            self.assertIn("adjusted_p90", item)
            self.assertIn("observed_margin_vs_adjusted_p90", item)

    def test_selection_key_prefers_checkpoint_stability_over_final_only_strength(self) -> None:
        stable = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.05,
                "minimum_margin_vs_adjusted_p90": 0.2,
                "exceedance_stddev": 0.01,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.08,
                "exceedance": 0.04,
            },
            "null_margin_vs_adjusted_p90": 0.3,
            "internal_key": [1.0],
        }
        final_only = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": False,
                "explicit_failures": 1,
                "worst_exceedance": 0.14,
                "minimum_margin_vs_adjusted_p90": -0.2,
                "exceedance_stddev": 0.04,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.04,
                "exceedance": 0.01,
            },
            "null_margin_vs_adjusted_p90": 1.0,
            "internal_key": [9.0],
        }
        self.assertGreater(selection_candidate_key(stable), selection_candidate_key(final_only))

    def test_zero_exceedance_and_zero_wilson_are_preserved_as_best_values(self) -> None:
        perfect_null_resistance = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.0,
                "minimum_margin_vs_adjusted_p90": 0.0,
                "exceedance_stddev": 0.0,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.0,
                "exceedance": 0.0,
            },
            "null_margin_vs_adjusted_p90": 0.0,
            "internal_key": [0.1],
        }
        merely_strong = {
            "adoption_passed": True,
            "checkpoint_stability": {
                "passed": True,
                "explicit_failures": 0,
                "worst_exceedance": 0.01,
                "minimum_margin_vs_adjusted_p90": 0.1,
                "exceedance_stddev": 0.0,
                "exceedance_deterioration": 0.0,
            },
            "decision": {
                "passed": True,
                "wilson_ci_upper": 0.01,
                "exceedance": 0.01,
            },
            "null_margin_vs_adjusted_p90": 1.0,
            "internal_key": [9.0],
        }
        self.assertGreater(
            selection_candidate_key(perfect_null_resistance),
            selection_candidate_key(merely_strong),
        )

    def test_workflow_runs_production_evolution_before_fixed_final(self) -> None:
        workflow = Path(".github/workflows/loto7_generation5.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("github.event_name == 'push'", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("scripts/generation5_selection_evolver.py", workflow)
        self.assertIn(
            "--selection-seed-bank outputs/generation5/selection_null_seed_bank.json",
            workflow,
        )
        self.assertIn(
            "--seed-bank outputs/generation5/final_null_seed_bank.json", workflow
        )
        self.assertLess(
            workflow.index("scripts/generation5_selection_evolver.py"),
            workflow.index("Run fixed final financial Null League"),
        )


if __name__ == "__main__":
    unittest.main()
