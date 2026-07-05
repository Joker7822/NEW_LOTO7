#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/recent_era_dedicated_guard.py

Recent Era Dedicated Guard

2020年以降などの直近時代専用モデルを守るためのガード。
このモデルは Dual Prediction の Recent Era 枠だけで使うため、
全期間ROIの悪化では復元しない。

採否条件:
  - Recent Era ROI が baseline 以上
  - Recent Era profit が baseline 以上
  - Recent Era ROI が最低ライン以上
  - 高等級件数が落ちない、または allow-high-grade-drop 指定

出力:
  outputs/recent_era/recent_era_model_guard.json
  outputs/recent_era/recent_era_model_guard_report.txt

注意:
  過去検証上のガードであり、将来の当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def draw_year(draw: object) -> int:
    raw = str(getattr(draw, "date", "") or "")
    m = re.match(r"^(\d{4})", raw)
    return int(m.group(1)) if m else 0


def filter_recent_indices(draws: Sequence[object], indices: Sequence[int], start_year: int) -> List[int]:
    return [idx for idx in indices if 0 <= idx < len(draws) and draw_year(draws[idx]) >= start_year]


def pct_value(metrics: Dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except Exception:
        return 0.0


def int_value(metrics: Dict[str, object], key: str) -> int:
    try:
        return int(metrics.get(key, 0))
    except Exception:
        return 0


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "target_draws",
        "total_tickets",
        "total_cost",
        "total_payout",
        "profit",
        "roi",
        "roi_percent",
        "grade_hit_count",
        "high_grade_hit_count",
        "max_main_match",
        "max_bonus_match",
        "rank_counts",
        "missing_prize_draw_count",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def evaluate_model(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, object]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    return dict(
        evaluate_model_on_holdout(
            genome=genome,
            model_path=model_path,
            draws=draws,
            prize_rows=prize_rows,
            target_indices=target_indices,
            purchase_count=purchase_count,
            unit_cost=unit_cost,
        )
    )


def decide_recent_only(
    *,
    baseline_recent: Dict[str, object],
    candidate_recent: Dict[str, object],
    same_model: bool,
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []

    if same_model:
        reasons.append("same model id; recent-era dedicated guard keeps current model")
        return True, reasons, warnings

    recent_roi_delta = pct_value(candidate_recent, "roi_percent") - pct_value(baseline_recent, "roi_percent")
    recent_profit_delta = int_value(candidate_recent, "profit") - int_value(baseline_recent, "profit")
    recent_roi = pct_value(candidate_recent, "roi_percent")
    recent_high_delta = int_value(candidate_recent, "high_grade_hit_count") - int_value(baseline_recent, "high_grade_hit_count")

    recent_roi_ok = recent_roi_delta >= args.min_recent_roi_delta_percent
    recent_profit_ok = recent_profit_delta >= args.min_recent_profit_delta
    recent_floor_ok = recent_roi >= args.min_recent_payout_roi_percent
    recent_high_ok = args.allow_recent_high_grade_drop or recent_high_delta >= 0

    if recent_roi_ok:
        reasons.append(f"recent roi ok: delta={recent_roi_delta:.3f}pt")
    else:
        warnings.append(f"recent roi dropped: delta={recent_roi_delta:.3f}pt < {args.min_recent_roi_delta_percent:.3f}pt")

    if recent_profit_ok:
        reasons.append(f"recent profit ok: delta={recent_profit_delta}")
    else:
        warnings.append(f"recent profit dropped: delta={recent_profit_delta} < {args.min_recent_profit_delta}")

    if recent_floor_ok:
        reasons.append(f"recent roi floor ok: {recent_roi:.3f}% >= {args.min_recent_payout_roi_percent:.3f}%")
    else:
        warnings.append(f"recent roi below floor: {recent_roi:.3f}% < {args.min_recent_payout_roi_percent:.3f}%")

    if recent_high_ok:
        reasons.append(f"recent high-grade ok: delta={recent_high_delta}")
    else:
        warnings.append(f"recent high-grade dropped: delta={recent_high_delta}")

    accepted = bool(recent_roi_ok and recent_profit_ok and recent_floor_ok and recent_high_ok)
    return accepted, reasons, warnings


def write_report(path: str, payload: Dict[str, object]) -> None:
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    baseline = payload.get("baseline", {}) if isinstance(payload.get("baseline"), dict) else {}
    candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate"), dict) else {}
    b_recent = baseline.get("recent", {}) if isinstance(baseline.get("recent"), dict) else {}
    c_recent = candidate.get("recent", {}) if isinstance(candidate.get("recent"), dict) else {}

    lines = [
        "LOTO7 Recent Era Dedicated Guard Report",
        "========================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"recent_start_year: {payload.get('recent_start_year')}",
        f"accepted: {decision.get('accepted')}",
        f"restored_baseline: {decision.get('restored_baseline')}",
        f"baseline_model_id: {baseline.get('model_id')}",
        f"candidate_model_id: {candidate.get('model_id')}",
        "",
        "[Decision Reasons]",
    ]
    for item in decision.get("reasons", []) if isinstance(decision.get("reasons"), list) else []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("[Warnings]")
    warnings = decision.get("warnings", []) if isinstance(decision.get("warnings"), list) else []
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "[Recent Era]",
            f"baseline: roi={b_recent.get('roi_percent')}% profit={b_recent.get('profit')} high_grade={b_recent.get('high_grade_hit_count')} max_main={b_recent.get('max_main_match')}",
            f"candidate: roi={c_recent.get('roi_percent')}% profit={c_recent.get('profit')} high_grade={c_recent.get('high_grade_hit_count')} max_main={c_recent.get('max_main_match')}",
            "",
            "Full-period ROI is intentionally ignored here because this model is used only for Recent Era prediction slots.",
            "注意: 過去検証上のrecent era専用ガードであり、将来の当せんや利益を保証しません。",
        ]
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a Recent Era dedicated LOTO7 model using recent-era metrics only.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline-model", default="outputs/recent_era/recent_era_baseline_before_run.json")
    parser.add_argument("--candidate-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--best-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--summary", default="outputs/recent_era/recent_era_model_guard.json")
    parser.add_argument("--report", default="outputs/recent_era/recent_era_model_guard_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--recent-start-year", type=int, default=2020)
    parser.add_argument("--min-recent-roi-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-recent-profit-delta", type=int, default=0)
    parser.add_argument("--min-recent-payout-roi-percent", type=float, default=8.0)
    parser.add_argument("--allow-recent-high-grade-drop", action="store_true")
    parser.add_argument("--restore-on-fail", action="store_true", default=True)
    parser.add_argument("--no-restore-on-fail", dest="restore_on_fail", action="store_false")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_model)
    candidate_path = Path(args.candidate_model)
    if not baseline_path.exists():
        raise SystemExit(f"baseline model snapshot not found: {baseline_path}")
    if not candidate_path.exists():
        raise SystemExit(f"candidate model not found: {candidate_path}")

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    if args.max_targets and args.max_targets > 0:
        target_indices = target_indices[-args.max_targets :]
    recent_indices = filter_recent_indices(draws, target_indices, args.recent_start_year)
    if not target_indices:
        raise SystemExit("no target indices selected")
    if not recent_indices:
        raise SystemExit("no recent-era target indices selected")

    baseline_genome, _baseline_payload = load_genome_payload(args.baseline_model)
    candidate_genome, _candidate_payload = load_genome_payload(args.candidate_model)
    same_model = baseline_genome.id == candidate_genome.id

    baseline_recent = evaluate_model(
        genome=baseline_genome,
        model_path=args.baseline_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=recent_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )
    candidate_recent = evaluate_model(
        genome=candidate_genome,
        model_path=args.candidate_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=recent_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )

    accepted, reasons, warnings = decide_recent_only(
        baseline_recent=baseline_recent,
        candidate_recent=candidate_recent,
        same_model=same_model,
        args=args,
    )

    restored = False
    if accepted:
        Path(args.best_model).parent.mkdir(parents=True, exist_ok=True)
        if Path(args.candidate_model).resolve() != Path(args.best_model).resolve():
            shutil.copyfile(args.candidate_model, args.best_model)
    elif args.restore_on_fail:
        Path(args.best_model).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.baseline_model, args.best_model)
        restored = True

    payload = {
        "created_at": now_iso(),
        "kind": "loto7_recent_era_dedicated_guard",
        "csv": args.csv,
        "recent_start_year": args.recent_start_year,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "recent_target_draws_total": len(recent_indices),
        "baseline": {
            "path": args.baseline_model,
            "model_id": baseline_genome.id,
            "recent": compact_metrics(baseline_recent),
        },
        "candidate": {
            "path": args.candidate_model,
            "model_id": candidate_genome.id,
            "recent": compact_metrics(candidate_recent),
        },
        "decision": {
            "accepted": accepted,
            "same_model": same_model,
            "restore_on_fail": bool(args.restore_on_fail),
            "restored_baseline": restored,
            "reasons": reasons,
            "warnings": warnings,
            "thresholds": {
                "min_recent_roi_delta_percent": args.min_recent_roi_delta_percent,
                "min_recent_profit_delta": args.min_recent_profit_delta,
                "min_recent_payout_roi_percent": args.min_recent_payout_roi_percent,
                "allow_recent_high_grade_drop": bool(args.allow_recent_high_grade_drop),
            },
        },
        "best_model_after_guard": args.best_model,
        "notes": [
            "This guard intentionally ignores full-period ROI.",
            "Recent Era dedicated models are used only for Recent Era prediction slots.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.summary, payload)
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
