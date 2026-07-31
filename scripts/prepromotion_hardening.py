"""Run all hardened evidence builders immediately before Generation 5 promotion."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Sequence

from scripts.harden_generation5_summary import main as harden_summary_main
from scripts.run_hit_first_null_league import main as hit_null_main
from scripts.run_prediction_ablation import main as ablation_main


def _run_cli(function: Callable[[], int], arguments: Sequence[str]) -> None:
    original = list(sys.argv)
    try:
        sys.argv = [function.__module__, *arguments]
        if function() != 0:
            raise RuntimeError(f"command failed: {function.__module__}")
    finally:
        sys.argv = original


def run_prepromotion_hardening(
    *,
    candidate: str,
    baseline: str,
    summary: str,
    null_summary: str,
    financial_null_summary: str,
    seed_bank: str,
    ablation: str,
) -> None:
    null_file = Path(null_summary)
    financial_file = Path(financial_null_summary)
    if null_file.exists() and null_file.stat().st_size:
        financial_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(null_file, financial_file)
    try:
        _run_cli(
            harden_summary_main,
            [
                "--candidate", candidate,
                "--baseline", baseline,
                "--summary", summary,
                "--start-year", "2020",
                "--bootstrap-samples", "2000",
            ],
        )
        _run_cli(
            hit_null_main,
            [
                "--model", candidate,
                "--seed-bank", seed_bank,
                "--seed-phase", "final",
                "--summary", null_summary,
                "--report", "outputs/generation5/null_strategy_league_report.txt",
                "--start-year", "2020",
                "--checkpoints", "150,500,1000",
                "--search-width", "6",
                "--max-null-exceedance", "0.10",
            ],
        )
        _run_cli(
            ablation_main,
            [
                "--model", candidate,
                "--output", ablation,
                "--start-year", "2020",
            ],
        )
    except Exception as error:
        output = Path("outputs/generation5/hardening_error.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "kind": "loto7_generation5_hardening_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fail_closed": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
