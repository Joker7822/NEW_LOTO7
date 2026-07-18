#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"marker not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace(
    ".github/workflows/loto7_evolution.yml",
    '''            --progress-every 10 \\
            --max-runtime-minutes 320 \\
            --safe-exit-minutes 30
      - name: Commit role ensemble backtest outputs
''',
    '''            --progress-every 10 \\
            --max-runtime-minutes 320 \\
            --safe-exit-minutes 30
      - name: Verify unified evaluator consistency
        run: |
          set -euo pipefail
          python scripts/verify_evaluator_consistency.py \\
            --holdout outputs/holdout/holdout_result.csv \\
            --role outputs/role_ensemble/role_ensemble_backtest.csv \\
            --output outputs/role_ensemble/evaluator_consistency.json \\
            --unit-cost 300
      - name: Commit role ensemble backtest outputs
''',
)

replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''            --unit-cost 300 \\
            --seed "${GITHUB_RUN_ID}"

      - name: Build Generation 4 prediction through strict adoption gates
''',
    '''            --unit-cost 300 \\
            --seed "${GITHUB_RUN_ID}"

      - name: Refresh existing production results before adoption decision
        run: |
          set -euo pipefail
          python scripts/check_prediction_history_results.py \\
            --history outputs/evolution_prediction_history.csv \\
            --csv loto7.csv \\
            --output outputs/evolution_prediction_history_result.txt

          python scripts/update_generation4_shadow_history.py \\
            --csv loto7.csv \\
            --latest-shadow outputs/generation4/latest_shadow_predictions.json \\
            --history outputs/generation4/shadow_history.csv \\
            --summary outputs/generation4/champion_challenger_summary.json \\
            --report outputs/generation4/champion_challenger_report.txt \\
            --challenger generation4 \\
            --champion beam_baseline \\
            --promotion-threshold 20 \\
            --min-evaluated-draws 30 \\
            --evaluate-only

      - name: Build Generation 4 prediction through strict adoption gates
''',
)
replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''          test -s outputs/generation4/strict_adoption_gate.json

          if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
    '''          test -s outputs/generation4/strict_adoption_gate.json
          test -s outputs/evolution_prediction_history_result.txt
          test -s outputs/generation4/shadow_history.csv
          test -s outputs/generation4/champion_challenger_summary.json
          test -s outputs/generation4/champion_challenger_report.txt

          if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
)
replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''          cp -f outputs/generation4/null_strategy_league_report.txt "$SNAPSHOT/outputs/generation4/"

          if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
    '''          cp -f outputs/generation4/null_strategy_league_report.txt "$SNAPSHOT/outputs/generation4/"
          cp -f outputs/evolution_prediction_history_result.txt "$SNAPSHOT/outputs/"
          cp -f outputs/generation4/shadow_history.csv "$SNAPSHOT/outputs/generation4/"
          cp -f outputs/generation4/champion_challenger_summary.json "$SNAPSHOT/outputs/generation4/"
          cp -f outputs/generation4/champion_challenger_report.txt "$SNAPSHOT/outputs/generation4/"

          if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
)
replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''            cp -f "$SNAPSHOT/outputs/generation4/null_strategy_league_report.txt" outputs/generation4/

            git add -f \\
''',
    '''            cp -f "$SNAPSHOT/outputs/generation4/null_strategy_league_report.txt" outputs/generation4/
            cp -f "$SNAPSHOT/outputs/evolution_prediction_history_result.txt" outputs/evolution_prediction_history_result.txt
            cp -f "$SNAPSHOT/outputs/generation4/shadow_history.csv" outputs/generation4/shadow_history.csv
            cp -f "$SNAPSHOT/outputs/generation4/champion_challenger_summary.json" outputs/generation4/champion_challenger_summary.json
            cp -f "$SNAPSHOT/outputs/generation4/champion_challenger_report.txt" outputs/generation4/champion_challenger_report.txt

            git add -f \\
''',
)
replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''              outputs/generation4/null_strategy_league_summary.json \\
              outputs/generation4/null_strategy_league_report.txt

            if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
    '''              outputs/generation4/null_strategy_league_summary.json \\
              outputs/generation4/null_strategy_league_report.txt \\
              outputs/evolution_prediction_history_result.txt \\
              outputs/generation4/shadow_history.csv \\
              outputs/generation4/champion_challenger_summary.json \\
              outputs/generation4/champion_challenger_report.txt

            if [ "$ADOPTION_ALLOWED" = "true" ]; then
''',
)
replace(
    ".github/workflows/loto7_generation4_run.yml",
    '''            else
              COMMIT_MESSAGE="Record rejected LOTO7 Generation 4 adoption [skip ci]"
            fi
''',
    '''            else
              COMMIT_MESSAGE="Record rejected LOTO7 Generation 4 adoption and refresh live results [skip ci]"
            fi
''',
)

replace(
    ".github/workflows/loto7_validation_tests.yml",
    '''      - "scripts/update_generation4_shadow_history.py"
      - "scripts/seal_generation4_prediction.py"
''',
    '''      - "scripts/evaluation_core.py"
      - "scripts/verify_evaluator_consistency.py"
      - "scripts/update_generation4_shadow_history.py"
      - "scripts/check_prediction_history_results.py"
      - "holdout_evaluator.py"
      - "scripts/backtest_role_ensemble.py"
      - "scripts/seal_generation4_prediction.py"
''',
)
replace(
    ".github/workflows/loto7_validation_tests.yml",
    '''      - "tests/test_strict_adoption_gates.py"
''',
    '''      - "tests/test_strict_adoption_gates.py"
      - "tests/test_evaluation_unification.py"
''',
)
replace(
    ".github/workflows/loto7_validation_tests.yml",
    '''            scripts/null_strategy_league.py \\
            scripts/update_generation4_shadow_history.py \\
''',
    '''            scripts/null_strategy_league.py \\
            scripts/evaluation_core.py \\
            scripts/verify_evaluator_consistency.py \\
            scripts/update_generation4_shadow_history.py \\
            scripts/check_prediction_history_results.py \\
            holdout_evaluator.py \\
            scripts/backtest_role_ensemble.py \\
''',
)
replace(
    ".github/workflows/loto7_validation_tests.yml",
    '''            tests/test_generation4_pipeline.py \\
            tests/test_strict_adoption_gates.py
''',
    '''            tests/test_generation4_pipeline.py \\
            tests/test_strict_adoption_gates.py \\
            tests/test_evaluation_unification.py
''',
)
replace(
    ".github/workflows/loto7_validation_tests.yml",
    '''            tests.test_generation4_pipeline \\
            tests.test_strict_adoption_gates -v
''',
    '''            tests.test_generation4_pipeline \\
            tests.test_strict_adoption_gates \\
            tests.test_evaluation_unification -v
''',
)

print("Applied workflow changes for evaluator unification and rejection-safe live updates.")
