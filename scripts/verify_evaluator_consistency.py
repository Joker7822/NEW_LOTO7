#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that holdout and role-backtest best-model evaluations are identical."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.evaluation_core import EVALUATOR_VERSION, financial_metrics

Key = Tuple[int, int]


def normalize_ticket(value: object) -> str:
    return " ".join(f"{int(token):02d}" for token in str(value or "").replace(",", " ").split() if token.isdigit())


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def verify(holdout_path: Path, role_path: Path, unit_cost: int = 300) -> Dict[str, object]:
    holdout_rows = read_rows(holdout_path)
    role_rows = [row for row in read_rows(role_path) if row.get("system") == "best_model"]
    holdout: Dict[Key, Dict[str, str]] = {
        (int(row["draw_no"]), int(row["combo_index"])): row for row in holdout_rows
    }
    role: Dict[Key, Dict[str, str]] = {
        (int(row["target_draw_no"]), int(row["ticket_index"])): row for row in role_rows
    }
    missing_in_role = sorted(set(holdout) - set(role))
    missing_in_holdout = sorted(set(role) - set(holdout))
    mismatches: List[Dict[str, object]] = []
    for key in sorted(set(holdout) & set(role)):
        left = holdout[key]
        right = role[key]
        checks = {
            "ticket": (normalize_ticket(left.get("ticket")), normalize_ticket(right.get("numbers"))),
            "main_match": (str(left.get("main_match", "")), str(right.get("main_match", ""))),
            "bonus_match": (str(left.get("bonus_match", "")), str(right.get("bonus_match", ""))),
            "rank": (str(left.get("rank", "")), str(right.get("rank", ""))),
            "payout": (str(left.get("prize_amount", "0")), str(right.get("payout", "0"))),
        }
        differences = {name: values for name, values in checks.items() if values[0] != values[1]}
        if differences:
            mismatches.append({"draw_no": key[0], "ticket_index": key[1], "differences": differences})

    def aggregate(rows: List[Dict[str, str]], *, holdout_format: bool) -> Dict[str, object]:
        total_payout = sum(int(row.get("prize_amount" if holdout_format else "payout", 0) or 0) for row in rows)
        rank_key = "rank"
        winners = sum(1 for row in rows if row.get(rank_key) not in {"", "外れ"})
        draw_numbers = {int(row.get("draw_no" if holdout_format else "target_draw_no", 0) or 0) for row in rows}
        winning_draw_numbers = {
            int(row.get("draw_no" if holdout_format else "target_draw_no", 0) or 0)
            for row in rows if row.get(rank_key) not in {"", "外れ"}
        }
        return financial_metrics(
            total_cost=len(rows) * unit_cost,
            total_payout=total_payout,
            total_tickets=len(rows),
            winning_tickets=winners,
            target_draws=len(draw_numbers),
            winning_draws=len(winning_draw_numbers),
        )

    holdout_metrics = aggregate(holdout_rows, holdout_format=True)
    role_metrics = aggregate(role_rows, holdout_format=False)
    metrics_equal = all(
        holdout_metrics.get(key) == role_metrics.get(key)
        for key in ("total_cost", "total_payout", "profit", "profit_roi_percent", "payout_roi_percent", "ticket_hit_rate_percent", "draw_hit_rate_percent")
    )
    payload: Dict[str, object] = {
        "kind": "loto7_evaluator_consistency",
        "evaluator_version": EVALUATOR_VERSION,
        "holdout_rows": len(holdout_rows),
        "role_best_model_rows": len(role_rows),
        "missing_in_role_count": len(missing_in_role),
        "missing_in_holdout_count": len(missing_in_holdout),
        "mismatch_count": len(mismatches),
        "mismatch_samples": mismatches[:20],
        "holdout_metrics": holdout_metrics,
        "role_best_model_metrics": role_metrics,
        "metrics_equal": metrics_equal,
        "passed": not missing_in_role and not missing_in_holdout and not mismatches and metrics_equal,
    }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="outputs/holdout/holdout_result.csv")
    parser.add_argument("--role", default="outputs/role_ensemble/role_ensemble_backtest.csv")
    parser.add_argument("--output", default="outputs/role_ensemble/evaluator_consistency.json")
    parser.add_argument("--unit-cost", type=int, default=300)
    args = parser.parse_args(argv)
    payload = verify(Path(args.holdout), Path(args.role), args.unit_cost)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
