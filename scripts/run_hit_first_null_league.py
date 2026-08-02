#!/usr/bin/env python3
"""Run fixed-final search-adjusted hit-first permutation Null League."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from loto7.evaluation.model_audit import year_of
from loto7.evaluation.null_permutation import adaptive_null_test
from loto7_evolution_trainer import generate_tickets, load_draws
from merge_evolution_shards import select_target_indices
from scripts.robust_model_metrics import load_genome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="outputs/generation5/generation5_candidate_model.json")
    parser.add_argument("--seed-bank", default="outputs/generation5/null_seed_bank.json")
    parser.add_argument("--seed-phase", default="final")
    parser.add_argument(
        "--summary", default="outputs/generation5/null_strategy_league_summary.json"
    )
    parser.add_argument("--report", default="outputs/generation5/null_strategy_league_report.txt")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--checkpoints", default="150")
    parser.add_argument("--search-width", type=int, default=6)
    parser.add_argument("--max-null-exceedance", type=float, default=0.10)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    indices = select_target_indices(
        draws, min_train_draws=52, holdout_start_draw=2, holdout_end_draw=None
    )
    indices = [index for index in indices if year_of(draws[index]) >= args.start_year]
    genome = load_genome(args.model)
    portfolios = [generate_tickets(draws[:index], genome, args.purchase_count) for index in indices]
    mains = [draws[index].main for index in indices]

    bank = json.loads(Path(args.seed_bank).read_text(encoding="utf-8"))
    phases = bank.get("phases", {})
    if not isinstance(phases, Mapping):
        raise SystemExit("invalid seed bank")
    raw_phase = phases.get(args.seed_phase, [])
    if not isinstance(raw_phase, list) or not raw_phase:
        raise SystemExit(f"fixed seed phase is missing or empty: {args.seed_phase}")
    seeds = [int(value) for value in raw_phase]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"fixed seed phase contains duplicate seeds: {args.seed_phase}")

    requested = sorted(
        {int(value) for value in args.checkpoints.split(",") if int(value) > 0}
    )
    checkpoints = [value for value in requested if value <= len(seeds)]
    if not checkpoints:
        raise SystemExit(
            f"seed phase {args.seed_phase} has {len(seeds)} seeds; no requested checkpoint fits"
        )

    result = adaptive_null_test(
        portfolios=portfolios,
        mains=mains,
        seeds=seeds,
        checkpoints=checkpoints,
        search_width=args.search_width,
        max_exceedance=args.max_null_exceedance,
    )
    payload = {
        "kind": "loto7_hit_first_fixed_final_null_league",
        "metric_schema_version": "loto7-metrics-2026.07.31-v2",
        "model": args.model,
        "target_draws": len(indices),
        "seed_bank": {
            "path": args.seed_bank,
            "phase": args.seed_phase,
            "fixed_phase_only": True,
            "fixed_seed_count": len(seeds),
            "version": bank.get("version"),
            "dataset_sha256": bank.get("dataset_sha256"),
            "evaluator_version": bank.get("evaluator_version"),
        },
        **result,
        "notes": [
            "Only the requested fixed seed phase is used; no selection/learning seed fallback is allowed.",
            "Each fixed-final seed owns search-width internal permutations and contributes one max-adjusted trial.",
            "Winning-number rows are permuted against fixed chronological portfolios.",
            "Payout values never contribute to this adoption decision.",
        ],
    }
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.report).write_text(
        json.dumps(
            {key: value for key, value in payload.items() if key != "null_distribution"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
