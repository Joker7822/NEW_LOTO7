#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_recent_era_model_comparison.py

Compare Recent Era performance between:
  - full-period model: loto7_best_model.json
  - recent-era dedicated model: outputs/recent_era/recent_era_best_model.json

This intentionally does NOT overwrite outputs/holdout/holdout_summary.json.
That file remains the holdout summary for the full-period adopted model.

Outputs:
  - outputs/recent_era/recent_era_model_comparison_summary.json
  - outputs/recent_era/recent_era_model_comparison_report.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Genome, genome_from_dict, load_draws  # noqa: E402
from merge_evolution_shards import evaluate_model_on_holdout, load_prize_rows, select_target_indices  # noqa: E402


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_genome_payload(path: str) -> Tuple[Genome, Dict[str, object]]:
    payload = read_json(path)
    raw = payload.get("genome", payload)
    if not isinstance(raw, dict):
        raise SystemExit(f"invalid genome payload: {path}")
    return genome_from_dict(raw), payload


def model_id(payload: Dict[str, object], genome: Genome) -> str:
    selected = payload.get("selected_holdout", {})
    if isinstance(selected, dict) and selected.get("genome_id"):
        return str(selected.get("genome_id"))
    return str(getattr(genome, "id", ""))


def draw_year(draw: object) -> int:
    raw = str(getattr(draw, "date", "") or "")
    m = re.match(r"^(\d{4})", raw)
    return int(m.group(1)) if m else 0


def recent_indices(draws: Sequence[object], indices: Sequence[int], start_year: int) -> list[int]:
    return [idx for idx in indices if 0 <= idx < len(draws) and draw_year(draws[idx]) >= start_year]


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "target_draws",
        "total_tickets",
        "total_cost",
        "total_payout",
        "profit",
        "roi",
        "roi_percent",
        "draw_hit_rate_percent",
        "ticket_hit_rate_percent",
        "grade_hit_count",
        "high_grade_hit_count",
        "max_main_match",
        "max_bonus_match",
        "rank_counts",
        "missing_prize_draw_count",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def evaluate(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, object]],
    indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    return dict(
        evaluate_model_on_holdout(
            genome=genome,
            model_path=model_path,
            draws=draws,
            prize_rows=prize_rows,
            target_indices=indices,
            purchase_count=purchase_count,
            unit_cost=unit_cost,
        )
    )


def delta(candidate: Dict[str, object], baseline: Dict[str, object]) -> Dict[str, object]:
    def f(key: str) -> float:
        try:
            return float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0))
        except Exception:
            return 0.0

    def i(key: str) -> int:
        try:
            return int(candidate.get(key, 0)) - int(baseline.get(key, 0))
        except Exception:
            return 0

    return {
        "roi_percent_delta": round(f("roi_percent"), 3),
        "profit_delta": i("profit"),
        "total_payout_delta": i("total_payout"),
        "grade_hit_count_delta": i("grade_hit_count"),
        "high_grade_hit_count_delta": i("high_grade_hit_count"),
        "max_main_match_delta": i("max_main_match"),
    }


def write_report(path: str, payload: Dict[str, object]) -> None:
    full = payload.get("full_period_model_recent_era", {}) if isinstance(payload.get("full_period_model_recent_era"), dict) else {}
    recent = payload.get("recent_era_dedicated_model", {}) if isinstance(payload.get("recent_era_dedicated_model"), dict) else {}
    d = payload.get("delta_recent_minus_full", {}) if isinstance(payload.get("delta_recent_minus_full"), dict) else {}

    fm = full.get("metrics", {}) if isinstance(full.get("metrics"), dict) else {}
    rm = recent.get("metrics", {}) if isinstance(recent.get("metrics"), dict) else {}

    lines = [
        "LOTO7 Recent Era Model Comparison",
        "===================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"recent_start_year: {payload.get('recent_start_year')}",
        f"target_draws: {payload.get('recent_target_draws')}",
        "",
        "[Full-period model evaluated on Recent Era]",
        f"model_id: {full.get('model_id')}",
        f"roi_percent: {fm.get('roi_percent')}",
        f"profit: {fm.get('profit')}",
        f"total_payout: {fm.get('total_payout')}",
        f"grade_hit_count: {fm.get('grade_hit_count')}",
        f"high_grade_hit_count: {fm.get('high_grade_hit_count')}",
        f"max_main_match: {fm.get('max_main_match')}",
        f"rank_counts: {json.dumps(fm.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}",
        "",
        "[Recent Era dedicated model]",
        f"model_id: {recent.get('model_id')}",
        f"roi_percent: {rm.get('roi_percent')}",
        f"profit: {rm.get('profit')}",
        f"total_payout: {rm.get('total_payout')}",
        f"grade_hit_count: {rm.get('grade_hit_count')}",
        f"high_grade_hit_count: {rm.get('high_grade_hit_count')}",
        f"max_main_match: {rm.get('max_main_match')}",
        f"rank_counts: {json.dumps(rm.get('rank_counts', {}), ensure_ascii=False, sort_keys=True)}",
        "",
        "[Delta: recent dedicated - full-period model on Recent Era]",
        f"roi_percent_delta: {d.get('roi_percent_delta')}",
        f"profit_delta: {d.get('profit_delta')}",
        f"total_payout_delta: {d.get('total_payout_delta')}",
        f"grade_hit_count_delta: {d.get('grade_hit_count_delta')}",
        f"high_grade_hit_count_delta: {d.get('high_grade_hit_count_delta')}",
        f"max_main_match_delta: {d.get('max_main_match_delta')}",
        "",
        "Note: outputs/holdout/holdout_summary.json remains the full-period model summary.",
        "注意: 過去検証上の比較であり、将来の当せんや利益を保証しません。",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Recent Era performance between full-period and dedicated recent-era LOTO7 models.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--summary", default="outputs/recent_era/recent_era_model_comparison_summary.json")
    parser.add_argument("--report", default="outputs/recent_era/recent_era_model_comparison_report.txt")
    parser.add_argument("--recent-start-year", type=int, default=2020)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    args = parser.parse_args()

    full_path = Path(args.full_model)
    recent_path = Path(args.recent_model)
    if not full_path.exists():
        raise SystemExit(f"full model not found: {full_path}")
    if not recent_path.exists():
        raise SystemExit(f"recent model not found: {recent_path}")

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    r_indices = recent_indices(draws, target_indices, args.recent_start_year)
    if not r_indices:
        raise SystemExit(f"no recent-era target indices for start_year={args.recent_start_year}")

    full_genome, full_payload = load_genome_payload(args.full_model)
    recent_genome, recent_payload = load_genome_payload(args.recent_model)

    full_metrics = evaluate(
        genome=full_genome,
        model_path=args.full_model,
        draws=draws,
        prize_rows=prize_rows,
        indices=r_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )
    recent_metrics = evaluate(
        genome=recent_genome,
        model_path=args.recent_model,
        draws=draws,
        prize_rows=prize_rows,
        indices=r_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )

    payload = {
        "created_at": now_iso(),
        "kind": "loto7_recent_era_model_comparison",
        "csv": args.csv,
        "recent_start_year": args.recent_start_year,
        "recent_target_draws": len(r_indices),
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "full_period_model_recent_era": {
            "path": args.full_model,
            "model_id": model_id(full_payload, full_genome),
            "metrics": compact_metrics(full_metrics),
        },
        "recent_era_dedicated_model": {
            "path": args.recent_model,
            "model_id": model_id(recent_payload, recent_genome),
            "metrics": compact_metrics(recent_metrics),
        },
        "delta_recent_minus_full": delta(recent_metrics, full_metrics),
        "notes": [
            "outputs/holdout/holdout_summary.json is intentionally kept as the full-period adopted model summary.",
            "This file compares Recent Era performance separately to avoid mixing full-period and recent-era dedicated model scores.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.summary, payload)
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
