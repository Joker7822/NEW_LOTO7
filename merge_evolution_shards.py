#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_evolution_shards.py

複数shardで出力された loto7_best_model_shardXX_of_YY.json を統合し、
score最大のGenomeを loto7_best_model.json と最新予測CSV/TXTへ反映する。

目的:
    - shard別に独立探索した最良モデルを1つの採用モデルへ統合する
    - 採用モデル、最新予測、統合サマリー、run manifestを出力する
    - 最新予測を信頼度の高い順に5口、テキスト形式でも出力する
    - モデル数不足や不正JSONを検出し、誤った採用を防ぐ

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

    unique: Dict[str, Dict[str, object]] = {}
    for item in found:
        unique[str(item["path"])] = item
    return list(unique.values())


def fmt_ticket(ticket) -> str:
    return " ".join(f"{n:02d}" for n in ticket)


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_prediction_rows(best: Genome, source_model: str, draws, purchase_count: int) -> List[Dict[str, object]]:
    latest = draws[-1]
    tickets = generate_tickets(draws, best, purchase_count)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: List[Dict[str, object]] = []
    for idx, ticket in enumerate(tickets, start=1):
        rows.append(
            {
                "confidence_rank": idx,
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": fmt_ticket(ticket),
                "model_id": best.id,
                "model_score": round(best.score, 6),
                "source_model": source_model,
                "created_at": created_at,
            }
        )
    return rows


def write_prediction(csv_path: str, rows: List[Dict[str, object]]) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "confidence_rank", "base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index",
                "numbers", "model_id", "model_score", "source_model", "created_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_report(report_path: str, rows: List[Dict[str, object]], best: Genome, source_model: str, model_count: int, min_models: int) -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0] if rows else {}

    lines: List[str] = []
    lines.append("LOTO7 Latest Prediction Report")
    lines.append("=" * 31)
    lines.append("")
    lines.append(f"作成日時(UTC): {first.get('created_at')}")
    lines.append(f"基準最新回: 第{first.get('base_latest_draw_no')}回")
    lines.append(f"基準最新抽せん日: {first.get('base_latest_date')}")
    lines.append(f"予測対象回: 第{first.get('prediction_draw_no')}回")
    lines.append("")
    lines.append("[採用モデル]")
    lines.append(f"モデルID: {best.id}")
    lines.append(f"モデルスコア: {round(best.score, 6)}")
    lines.append(f"採用元shardモデル: {source_model}")
    lines.append(f"統合対象モデル数: {model_count}")
    lines.append(f"必要最小モデル数: {min_models}")
    lines.append("")
    lines.append("[最新予測 5口: 信頼度の高い順]")
    for row in rows:
        lines.append(f"{int(row['confidence_rank'])}位 / {int(row['combo_index'])}口目: {row['numbers']}")
    lines.append("")
    lines.append("[読み方]")
    lines.append("1位がこのモデル内で最も信頼度が高い組み合わせです。")
    lines.append("信頼度順位は、採用Genomeが生成した候補のスコア順を保持しています。")
    lines.append("")
    lines.append("[出力ファイル]")
    lines.append("CSV: outputs/evolution_best_prediction.csv")
    lines.append(f"TXT: {report_path}")
    lines.append("")
    lines.append("注意: 宝くじはランダム性が高く、この予測は当せんや利益を保証するものではありません。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge LOTO7 evolution shard best models.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--patterns", nargs="*", default=["loto7_best_model_shard*_of_*.json", "outputs/loto7_best_model_shard*_of_*.json"])
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--summary", default="outputs/evolution_merged_summary.json")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-models", type=int, default=1, help="統合に必要な最小モデル数。shard数と同じ値にすると欠損を検出できる。")
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.min_models <= 0:
        raise SystemExit("--min-models must be positive")

    models = find_models(args.patterns)
    if not models:
        raise SystemExit(f"no shard best models found: {args.patterns}")
    if len(models) < args.min_models:
        raise SystemExit(f"not enough shard models: found={len(models)} required={args.min_models}")

    models.sort(key=lambda item: item["genome"].score, reverse=True)  # type: ignore[index, union-attr]
    best_item = models[0]
    best: Genome = best_item["genome"]  # type: ignore[assignment]
    source_model = str(best_item["path"])

    updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "updated_at": updated_at,
        "source_model": source_model,
        "merged_from": [str(item["path"]) for item in models],
        "purchase_count": args.purchase_count,
        "genome": best.__dict__,
    }
    write_json(args.best_model, payload)

    draws = load_draws(args.csv)
    prediction_rows = make_prediction_rows(best, source_model, draws, args.purchase_count)
    write_prediction(args.prediction, prediction_rows)
    write_prediction_report(args.prediction_report, prediction_rows, best, source_model, len(models), args.min_models)

    candidates = [
        {"rank": i + 1, "path": str(item["path"]), "genome_id": item["genome"].id, "score": item["genome"].score}  # type: ignore[index, union-attr]
        for i, item in enumerate(models)
    ]
    summary = {
        "updated_at": updated_at,
        "selected_model": source_model,
        "selected_genome_id": best.id,
        "selected_score": best.score,
        "model_count": len(models),
        "min_models": args.min_models,
        "csv": args.csv,
        "latest_draw_no": draws[-1].draw_no if draws else None,
        "latest_draw_date": draws[-1].date if draws else None,
        "best_model": args.best_model,
        "prediction": args.prediction,
        "prediction_report": args.prediction_report,
        "candidates": candidates,
    }
    write_json(args.summary, summary)

    manifest = {
        "created_at": updated_at,
        "kind": "loto7_evolution_merge",
        "csv": args.csv,
        "latest_draw_no": draws[-1].draw_no if draws else None,
        "latest_draw_date": draws[-1].date if draws else None,
        "best_model": args.best_model,
        "prediction": args.prediction,
        "prediction_report": args.prediction_report,
        "summary": args.summary,
        "selected_model": source_model,
        "selected_genome_id": best.id,
        "selected_score": best.score,
        "purchase_count": args.purchase_count,
        "model_count": len(models),
    }
    write_json(args.manifest, manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
