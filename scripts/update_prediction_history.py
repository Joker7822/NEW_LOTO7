#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create/update a wide cumulative LOTO7 prediction history CSV.

Output format:
抽せん日,回別,予測1,信頼度1,...,予測5,信頼度5

One row represents one target draw. When the same draw date or draw number already
exists, it is replaced by the latest prediction set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_no_int(text: object) -> Optional[int]:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def parse_nums(text: object) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def fmt_prediction(text: object) -> str:
    nums = parse_nums(text)
    if len(nums) != 7:
        return str(text or "").strip()
    return ", ".join(str(n) for n in nums)


def parse_date(text: object) -> Optional[dt.date]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except Exception:
        return None


def load_draw_dates(csv_path: Path) -> Dict[int, str]:
    if not csv_path.exists():
        return {}
    out: Dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = draw_no_int(row.get("回別"))
            date = str(row.get("抽せん日") or "").strip()
            if no is not None and date:
                out[no] = date
    return out


def target_draw(row: Dict[str, str], draw_dates: Dict[int, str]) -> Tuple[str, str]:
    prediction_no = draw_no_int(row.get("prediction_draw_no"))

    for field in ("抽せん日", "prediction_draw_date", "prediction_date", "draw_date"):
        value = str(row.get(field, "")).strip()
        if value:
            draw_date = value[:10]
            draw_label = f"第{prediction_no}回" if prediction_no is not None else ""
            return draw_date, draw_label

    if prediction_no is not None and prediction_no in draw_dates:
        return draw_dates[prediction_no], f"第{prediction_no}回"

    base_date = parse_date(row.get("base_latest_date"))
    base_no = draw_no_int(row.get("base_latest_draw_no"))
    if base_date:
        draw_no = prediction_no or ((base_no + 1) if base_no is not None else None)
        return (base_date + dt.timedelta(days=7)).isoformat(), (f"第{draw_no}回" if draw_no is not None else "")

    raise ValueError("latest prediction row has no usable target draw date")


def confidence_for(row: Dict[str, str], rank: int) -> str:
    for field in ("confidence", "confidence_score", "ensemble_score", "model_score", "score"):
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            return f"{value:.3f}".rstrip("0").rstrip(".")
    value = max(0.0, 0.97 - (rank - 1) * 0.01)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def output_fields(max_predictions: int) -> List[str]:
    fields = ["抽せん日", "回別"]
    for i in range(1, max_predictions + 1):
        fields.extend([f"予測{i}", f"信頼度{i}"])
    return fields


def sort_latest_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    def key(row: Dict[str, str]) -> tuple:
        rank = draw_no_int(row.get("confidence_rank")) or draw_no_int(row.get("combo_index")) or 9999
        return (rank, str(row.get("numbers", "")))

    return sorted(rows, key=key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update wide cumulative LOTO7 prediction history CSV.")
    parser.add_argument("--latest", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--max-predictions", type=int, default=5)
    args = parser.parse_args()

    latest_path = Path(args.latest)
    history_path = Path(args.history)
    latest_rows = read_rows(latest_path)
    if not latest_rows:
        raise SystemExit(f"latest prediction CSV is empty or missing: {latest_path}")

    max_predictions = max(1, int(args.max_predictions))
    draw_dates = load_draw_dates(Path(args.csv))
    draw_date, draw_label = target_draw(latest_rows[0], draw_dates)

    new_row = {field: "" for field in output_fields(max_predictions)}
    new_row["抽せん日"] = draw_date
    new_row["回別"] = draw_label
    for idx, row in enumerate(sort_latest_rows(latest_rows)[:max_predictions], start=1):
        new_row[f"予測{idx}"] = fmt_prediction(row.get("numbers"))
        new_row[f"信頼度{idx}"] = confidence_for(row, idx)

    existing_rows = read_rows(history_path)
    kept_rows = [
        row for row in existing_rows
        if row.get("抽せん日")
        and row.get("抽せん日") != draw_date
        and (not draw_label or row.get("回別") != draw_label)
    ]
    combined = kept_rows + [new_row]
    combined.sort(key=lambda r: str(r.get("抽せん日", "")))

    fields = output_fields(max_predictions)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in combined:
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"updated {history_path}: draw_date={draw_date} draw={draw_label} rows={len(combined)} predictions={min(len(latest_rows), max_predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
