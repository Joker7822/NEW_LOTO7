#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append or replace latest LOTO7 prediction rows in a cumulative CSV.

The latest prediction CSV currently has no explicit future draw date column, so
`prediction_draw_no` is used as the stable event key. If a future version adds
`prediction_draw_date`, that date is used as the primary key instead.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fieldnames_for(rows: Iterable[Dict[str, str]]) -> List[str]:
    preferred = [
        "history_saved_at",
        "prediction_key",
        "confidence_rank",
        "base_latest_draw_no",
        "base_latest_date",
        "prediction_draw_no",
        "prediction_draw_date",
        "combo_index",
        "numbers",
        "model_id",
        "model_score",
        "source_model",
        "prediction_method",
        "ensemble_score",
        "support_models",
        "created_at",
    ]
    seen = set()
    fields: List[str] = []
    for name in preferred:
        fields.append(name)
        seen.add(name)
    for row in rows:
        for name in row.keys():
            if name and name not in seen:
                fields.append(name)
                seen.add(name)
    return fields


def row_key(row: Dict[str, str]) -> str:
    # Future-proof: if prediction_draw_date exists, use it as the event date key.
    for field in ("prediction_draw_date", "prediction_date", "draw_date", "prediction_draw_no", "base_latest_date"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    raise ValueError("prediction row has no usable event key")


def sort_key(row: Dict[str, str]) -> tuple:
    key = str(row.get("prediction_key", ""))
    rank_raw = str(row.get("confidence_rank", row.get("combo_index", "9999")))
    try:
        rank = int(rank_raw)
    except ValueError:
        rank = 9999
    try:
        numeric_key = int(key)
        return (0, numeric_key, rank)
    except ValueError:
        return (1, key, rank)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cumulative LOTO7 prediction history CSV.")
    parser.add_argument("--latest", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--history", default="outputs/evolution_prediction_history.csv")
    args = parser.parse_args()

    latest_path = Path(args.latest)
    history_path = Path(args.history)
    latest_rows = read_rows(latest_path)
    if not latest_rows:
        raise SystemExit(f"latest prediction CSV is empty or missing: {latest_path}")

    saved_at = dt.datetime.now(dt.timezone.utc).isoformat()
    new_rows: List[Dict[str, str]] = []
    keys = set()
    for row in latest_rows:
        merged = {k: ("" if v is None else str(v)) for k, v in row.items()}
        key = row_key(merged)
        merged["prediction_key"] = key
        merged["history_saved_at"] = saved_at
        keys.add(key)
        new_rows.append(merged)

    old_rows = read_rows(history_path)
    kept_rows = [row for row in old_rows if row_key(row) not in keys]
    combined = kept_rows + new_rows
    combined.sort(key=sort_key)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_for(combined)
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in combined:
            writer.writerow({name: row.get(name, "") for name in fields})

    print(f"updated {history_path}: replaced_keys={sorted(keys)} rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
