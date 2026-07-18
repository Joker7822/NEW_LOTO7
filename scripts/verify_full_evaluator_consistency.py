#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate holdout best-model tickets and compare them exactly.

With ``--sample-count 0`` this audits every holdout draw. A positive sample count
selects evenly spaced draws across the entire holdout period, which is suitable
for routine CI while preserving the full audit command for manual verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import evaluate_ticket, generate_tickets, load_best_model, load_draws
from scripts.evaluation_core import EVALUATOR_VERSION, financial_metrics, load_prize_rows, prize_amount_for_rank

Key = Tuple[int, int]
_DRAWS = []
_GENOME = None
_PRIZES: Dict[int, Dict[str, str]] = {}
_PURCHASE_COUNT = 5


def _initialize_worker(csv_path: str, model_path: str, purchase_count: int) -> None:
    global _DRAWS, _GENOME, _PRIZES, _PURCHASE_COUNT
    _DRAWS = load_draws(csv_path)
    _GENOME = load_best_model(model_path)
    _PRIZES = load_prize_rows(csv_path)
    _PURCHASE_COUNT = int(purchase_count)
    if _GENOME is None:
        raise RuntimeError(f"cannot load model: {model_path}")


def _evaluate_index(index: int) -> List[Dict[str, object]]:
    target = _DRAWS[index]
    tickets = generate_tickets(_DRAWS[:index], _GENOME, _PURCHASE_COUNT)
    prize_row = _PRIZES.get(target.draw_no, {})
    rows: List[Dict[str, object]] = []
    for ticket_index, ticket in enumerate(tickets, start=1):
        main_match, bonus_match, rank = evaluate_ticket(ticket, target)
        rows.append(
            {
                "draw_no": target.draw_no,
                "ticket_index": ticket_index,
                "ticket": " ".join(f"{number:02d}" for number in ticket),
                "main_match": main_match,
                "bonus_match": bonus_match,
                "rank": rank,
                "payout": prize_amount_for_rank(prize_row, rank),
            }
        )
    return rows


def _normalized_ticket(value: object) -> str:
    return " ".join(
        f"{int(token):02d}"
        for token in str(value or "").replace(",", " ").split()
        if token.isdigit()
    )


def _stratified_indices(indices: Sequence[int], sample_count: int) -> List[int]:
    ordered = list(indices)
    if sample_count <= 0 or sample_count >= len(ordered):
        return ordered
    if sample_count == 1:
        return [ordered[-1]]
    positions = {
        round(offset * (len(ordered) - 1) / (sample_count - 1))
        for offset in range(sample_count)
    }
    return [ordered[position] for position in sorted(positions)]


def verify(
    *,
    csv_path: str,
    model_path: str,
    holdout_path: str,
    purchase_count: int,
    unit_cost: int,
    workers: int,
    sample_count: int = 0,
) -> Dict[str, object]:
    with Path(holdout_path).open("r", encoding="utf-8-sig", newline="") as stream:
        all_holdout_rows = list(csv.DictReader(stream))
    draws = load_draws(csv_path)
    available_draw_nos = {int(row["draw_no"]) for row in all_holdout_rows}
    all_target_indices = [
        index for index, draw in enumerate(draws) if draw.draw_no in available_draw_nos
    ]
    target_indices = _stratified_indices(all_target_indices, sample_count)
    selected_draw_nos = {draws[index].draw_no for index in target_indices}
    holdout_rows = [
        row for row in all_holdout_rows if int(row["draw_no"]) in selected_draw_nos
    ]
    expected: Dict[Key, Dict[str, object]] = {
        (int(row["draw_no"]), int(row["combo_index"])): {
            "ticket": _normalized_ticket(row.get("ticket")),
            "main_match": int(row.get("main_match", 0) or 0),
            "bonus_match": int(row.get("bonus_match", 0) or 0),
            "rank": str(row.get("rank", "外れ") or "外れ"),
            "payout": int(row.get("prize_amount", 0) or 0),
        }
        for row in holdout_rows
    }
    actual: Dict[Key, Dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(csv_path, model_path, purchase_count),
    ) as executor:
        for rows in executor.map(_evaluate_index, target_indices, chunksize=1):
            for row in rows:
                actual[(int(row["draw_no"]), int(row["ticket_index"]))] = row

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches: List[Dict[str, object]] = []
    for key in sorted(set(expected) & set(actual)):
        left = expected[key]
        right = actual[key]
        differences = {
            field: [left[field], right[field]]
            for field in ("ticket", "main_match", "bonus_match", "rank", "payout")
            if left[field] != right[field]
        }
        if differences:
            mismatches.append(
                {"draw_no": key[0], "ticket_index": key[1], "differences": differences}
            )

    total_payout = sum(int(row["payout"]) for row in actual.values())
    winning_tickets = sum(1 for row in actual.values() if row["rank"] != "外れ")
    winning_draws = len(
        {key[0] for key, row in actual.items() if row["rank"] != "外れ"}
    )
    metrics = financial_metrics(
        total_cost=len(actual) * unit_cost,
        total_payout=total_payout,
        total_tickets=len(actual),
        winning_tickets=winning_tickets,
        target_draws=len(target_indices),
        winning_draws=winning_draws,
    )
    expected_ticket_count = len(target_indices) * purchase_count
    return {
        "kind": "loto7_evaluator_consistency_audit",
        "evaluator_version": EVALUATOR_VERSION,
        "workers": workers,
        "available_target_draws": len(all_target_indices),
        "sample_count_requested": sample_count,
        "target_draws": len(target_indices),
        "sampled_draw_nos": sorted(selected_draw_nos),
        "expected_tickets": len(expected),
        "regenerated_tickets": len(actual),
        "calculated_expected_ticket_count": expected_ticket_count,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatch_count": len(mismatches),
        "missing_samples": missing[:20],
        "extra_samples": extra[:20],
        "mismatch_samples": mismatches[:20],
        "metrics": metrics,
        "full_history_audit": len(target_indices) == len(all_target_indices),
        "passed": (
            not missing
            and not extra
            and not mismatches
            and len(expected) == expected_ticket_count
            and len(actual) == expected_ticket_count
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--model", default="loto7_best_model.json")
    parser.add_argument("--holdout", default="outputs/holdout/holdout_result.csv")
    parser.add_argument("--output", default="outputs/role_ensemble/evaluator_consistency.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--sample-count",
        type=int,
        default=36 if os.environ.get("GITHUB_ACTIONS") == "true" else 0,
        help="0 audits all holdout draws; a positive value selects evenly spaced draws",
    )
    args = parser.parse_args(argv)
    if (
        args.purchase_count <= 0
        or args.unit_cost <= 0
        or args.workers <= 0
        or args.sample_count < 0
    ):
        raise SystemExit("purchase-count, unit-cost and workers must be positive; sample-count must be nonnegative")
    payload = verify(
        csv_path=args.csv,
        model_path=args.model,
        holdout_path=args.holdout,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
        workers=args.workers,
        sample_count=args.sample_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
