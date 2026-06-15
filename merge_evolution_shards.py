#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_evolution_shards.py

複数shardで出力された loto7_best_model_shardXX_of_YY.json を統合し、
score最大のGenomeを loto7_best_model.json と最新予測CSVへ反映する。

例:
    python merge_evolution_shards.py --csv loto7.csv --purchase-count 5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

from loto7_evolution_trainer import Genome, generate_tickets, genome_from_dict, load_draws


def load_model(path: Path) -> Optional[Dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        genome_data = data.get("genome", data)
        genome = genome_from_dict(genome_data)
        return {"path": str(path), "payload": data, "genome": genome}
    except Exception as exc:
        print(f"[WARN] skip invalid model: {path} ({exc})")
        return None


def find_models(patterns: List[str]) -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    for pattern in patterns:
        for name in sorted(glob.glob(pattern)):
            item = load_model(Path(name))
            if item is not None:
                found.append(item)
    # path重複を排除
    unique: Dict[str, Dict[str, object]] = {}
    for item in found:
        unique[str(item["path"])] = item
    return list(unique.values())


def fmt_ticket(ticket) -> str:
    return " ".join(f"{n:02d}" for n in ticket)


def write_prediction(csv_path: str, best: Genome, source_model: str, draws, purchase_count: int) -> None:
    latest = draws[-1]
    tickets = generate_tickets(draws, best, purchase_count)
    rows = []
    for idx, ticket in enumerate(tickets, start=1):
        rows.append(
            {
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": fmt_ticket(ticket),
                "model_id": best.id,
                "model_score": round(best.score, 6),
                "source_model": source_model,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index",
                "numbers", "model_id", "model_score", "source_model", "created_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LOTO7 evolution shard best models.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--patterns", nargs="*", default=["loto7_best_model_shard*_of_*.json", "outputs/loto7_best_model_shard*_of_*.json"])
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--summary", default="outputs/evolution_merged_summary.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    args = parser.parse_args(argv)

    models = find_models(args.patterns)
    if not models:
        raise SystemExit(f"no shard best models found: {args.patterns}")

    models.sort(key=lambda item: item["genome"].score, reverse=True)  # type: ignore[index, union-attr]
    best_item = models[0]
    best: Genome = best_item["genome"]  # type: ignore[assignment]
    source_model = str(best_item["path"])

    payload = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_model": source_model,
        "merged_from": [str(item["path"]) for item in models],
        "purchase_count": args.purchase_count,
        "genome": best.__dict__,
    }
    Path(args.best_model).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    draws = load_draws(args.csv)
    write_prediction(args.prediction, best, source_model, draws, args.purchase_count)

    summary = {
        "updated_at": payload["updated_at"],
        "selected_model": source_model,
        "selected_genome_id": best.id,
        "selected_score": best.score,
        "model_count": len(models),
        "best_model": args.best_model,
        "prediction": args.prediction,
        "candidates": [
            {"path": str(item["path"]), "genome_id": item["genome"].id, "score": item["genome"].score}  # type: ignore[index, union-attr]
            for item in models
        ],
    }
    out_summary = Path(args.summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
