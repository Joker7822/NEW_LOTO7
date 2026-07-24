#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that the published LOTO7 prediction targets the next undrawn draw.

The production model-adoption decision and prediction publication are separate
concerns. Even when a challenger is rejected, the repository must publish a
fresh prediction generated from the currently approved models.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def draw_no_int(value: object) -> Optional[int]:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"CSV is empty or missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def latest_actual_draw_no(rows: Iterable[Dict[str, str]]) -> int:
    values = [draw_no_int(row.get("回別") or row.get("draw_no")) for row in rows]
    draw_numbers = [value for value in values if value is not None]
    if not draw_numbers:
        raise ValueError("dataset contains no usable draw number")
    return max(draw_numbers)


def single_prediction_draw_no(rows: Iterable[Dict[str, str]]) -> int:
    values = {
        value
        for row in rows
        if (value := draw_no_int(row.get("prediction_draw_no"))) is not None
    }
    if len(values) != 1:
        raise ValueError(f"prediction rows must target exactly one draw: {sorted(values)}")
    return next(iter(values))


def single_base_draw_no(rows: Iterable[Dict[str, str]]) -> int:
    values = {
        value
        for row in rows
        if (value := draw_no_int(row.get("base_latest_draw_no"))) is not None
    }
    if len(values) != 1:
        raise ValueError(f"prediction rows must use exactly one base draw: {sorted(values)}")
    return next(iter(values))


def history_draw_numbers(rows: Iterable[Dict[str, str]]) -> List[int]:
    return [
        value
        for row in rows
        if (value := draw_no_int(row.get("回別") or row.get("prediction_draw_no"))) is not None
    ]


def validate_freshness(
    dataset_rows: List[Dict[str, str]],
    prediction_rows: List[Dict[str, str]],
    history_rows: List[Dict[str, str]],
) -> Dict[str, int]:
    actual = latest_actual_draw_no(dataset_rows)
    expected = actual + 1
    base = single_base_draw_no(prediction_rows)
    target = single_prediction_draw_no(prediction_rows)

    if base != actual:
        raise ValueError(f"prediction base is stale: base={base} latest_actual={actual}")
    if target != expected:
        raise ValueError(f"prediction target is stale: target={target} expected={expected}")

    history_numbers = history_draw_numbers(history_rows)
    target_count = history_numbers.count(expected)
    if target_count != 1:
        raise ValueError(
            f"history must contain the next draw exactly once: draw={expected} count={target_count}"
        )

    return {
        "latest_actual_draw_no": actual,
        "expected_prediction_draw_no": expected,
        "prediction_base_draw_no": base,
        "prediction_draw_no": target,
        "history_target_count": target_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the latest LOTO7 prediction targets latest actual draw + 1."
    )
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--history", default="outputs/evolution_prediction_history.csv")
    args = parser.parse_args()

    result = validate_freshness(
        read_csv(Path(args.csv)),
        read_csv(Path(args.prediction)),
        read_csv(Path(args.history)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
