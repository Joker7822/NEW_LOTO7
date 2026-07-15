#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promote a LOTO7 candidate only after true nested validation.

This is the second stage of model adoption. The normal evolution workflow may
produce a candidate, but this script promotes it only when sealed nested folds
and robust payout diagnostics pass. A candidate already stored at the target
path is handled without copying the file onto itself.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import load_draws  # noqa: E402
from merge_evolution_shards import load_prize_rows, select_target_indices  # noqa: E402
from scripts.robust_model_metrics import evaluate_model_robust, indices_for_years, load_genome  # noqa: E402


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def evaluate(path: str, args: argparse.Namespace) -> Dict[str, object]:
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base = select_target_indices(draws, min_train_draws=1, holdout_start_draw=2, holdout_end_draw=None)
    indices = indices_for_years(draws, base, args.focus_start_year, args.focus_end_year)
    if not indices:
        raise SystemExit("no focus draws selected")
    return evaluate_model_robust(
        genome=load_genome(path),
        model_path=path,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.focus_start_year,
    )


def write_report(path: str, payload: Dict[str, object]) -> None:
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    lines = [
        "LOTO7 Nested Candidate Promotion",
        "================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"candidate_model_id: {payload.get('candidate_model_id')}",
        f"baseline_model_id: {payload.get('baseline_model_id')}",
        f"promoted: {decision.get('promoted')}",
        f"copy_performed: {decision.get('copy_performed')}",
        "",
        "[Reasons]",
    ]
    for reason in decision.get("reasons", []) if isinstance(decision.get("reasons"), list) else []:
        lines.append(f"- {reason}")
    lines.extend(["", "[Warnings]"])
    warnings = decision.get("warnings", []) if isinstance(decision.get("warnings"), list) else []
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "[Baseline Robust Metrics]",
            json.dumps(payload.get("baseline_metrics", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "[Candidate Robust Metrics]",
            json.dumps(payload.get("candidate_metrics", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "[Nested Summary]",
            json.dumps(payload.get("nested_summary", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Historical validation does not guarantee future lottery results.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a candidate after nested and robust validation.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--nested-summary", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--focus-start-year", type=int, default=2020)
    parser.add_argument("--focus-end-year", type=int, default=None)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--min-median-roi-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-worst-roi-delta-percent", type=float, default=-30.0)
    parser.add_argument("--min-median-top1-removed-delta-percent", type=float, default=-5.0)
    parser.add_argument("--min-focus-roi-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-top1-removed-delta-percent", type=float, default=-5.0)
    parser.add_argument("--min-top2-removed-delta-percent", type=float, default=-10.0)
    parser.add_argument("--min-bootstrap-p05-delta-percent", type=float, default=-15.0)
    parser.add_argument("--max-top1-payout-share", type=float, default=0.90)
    args = parser.parse_args()

    for path in (args.baseline_model, args.candidate_model, args.nested_summary):
        if not Path(path).exists():
            raise SystemExit(f"required file missing: {path}")

    baseline_genome = load_genome(args.baseline_model)
    candidate_genome = load_genome(args.candidate_model)
    nested = read_json(args.nested_summary)
    reasons: List[str] = []
    warnings: List[str] = []
    passed = True

    nested_reference = str(nested.get("reference_model_id") or "")
    if nested_reference != candidate_genome.id:
        passed = False
        warnings.append(f"nested model mismatch: nested={nested_reference} candidate={candidate_genome.id}")
    if bool(nested.get("future_leakage_detected")):
        passed = False
        warnings.append("future leakage detected by nested validation")

    positive_folds = as_int(nested.get("positive_roi_delta_folds"))
    median_delta = as_float(nested.get("median_roi_delta_percent"))
    worst_delta = as_float(nested.get("worst_roi_delta_percent"))
    median_top1_delta = as_float(nested.get("median_top1_removed_roi_delta_percent"))
    nested_checks = [
        (positive_folds >= args.min_positive_folds, f"positive folds={positive_folds}", f"positive folds failed: {positive_folds}"),
        (median_delta >= args.min_median_roi_delta_percent, f"median ROI delta={median_delta:.3f}pt", f"median ROI delta failed: {median_delta:.3f}pt"),
        (worst_delta >= args.min_worst_roi_delta_percent, f"worst ROI delta={worst_delta:.3f}pt", f"worst ROI delta failed: {worst_delta:.3f}pt"),
        (
            median_top1_delta >= args.min_median_top1_removed_delta_percent,
            f"median top1-removed delta={median_top1_delta:.3f}pt",
            f"median top1-removed delta failed: {median_top1_delta:.3f}pt",
        ),
    ]
    for ok, reason, warning in nested_checks:
        if ok:
            reasons.append(reason)
        else:
            passed = False
            warnings.append(warning)

    baseline_metrics = evaluate(args.baseline_model, args)
    candidate_metrics = evaluate(args.candidate_model, args)
    robust_checks = [
        (
            "focus ROI",
            as_float(candidate_metrics.get("roi_percent")) - as_float(baseline_metrics.get("roi_percent")),
            args.min_focus_roi_delta_percent,
        ),
        (
            "top1-removed ROI",
            as_float(candidate_metrics.get("roi_excluding_top1_percent")) - as_float(baseline_metrics.get("roi_excluding_top1_percent")),
            args.min_top1_removed_delta_percent,
        ),
        (
            "top2-removed ROI",
            as_float(candidate_metrics.get("roi_excluding_top2_percent")) - as_float(baseline_metrics.get("roi_excluding_top2_percent")),
            args.min_top2_removed_delta_percent,
        ),
        (
            "bootstrap p05 ROI",
            as_float(candidate_metrics.get("bootstrap_roi_percent_p05")) - as_float(baseline_metrics.get("bootstrap_roi_percent_p05")),
            args.min_bootstrap_p05_delta_percent,
        ),
    ]
    for label, delta, threshold in robust_checks:
        if delta >= threshold:
            reasons.append(f"{label} delta ok: {delta:.3f}pt")
        else:
            passed = False
            warnings.append(f"{label} delta failed: {delta:.3f}pt < {threshold:.3f}pt")

    candidate_share = as_float(candidate_metrics.get("top1_payout_share"))
    if candidate_share > args.max_top1_payout_share:
        passed = False
        warnings.append(f"top1 payout share failed: {candidate_share:.3f} > {args.max_top1_payout_share:.3f}")
    else:
        reasons.append(f"top1 payout share ok: {candidate_share:.3f}")

    promoted = False
    copy_performed = False
    if passed:
        target = Path(args.best_model)
        source = Path(args.candidate_model)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
            copy_performed = True
            reasons.append("candidate copied to production best-model path")
        else:
            reasons.append("candidate already stored at production best-model path; copy skipped")
        promoted = True

    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_nested_candidate_promotion",
        "baseline_model_path": args.baseline_model,
        "baseline_model_id": baseline_genome.id,
        "candidate_model_path": args.candidate_model,
        "candidate_model_id": candidate_genome.id,
        "best_model_path": args.best_model,
        "focus_start_year": args.focus_start_year,
        "focus_end_year": args.focus_end_year,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "nested_summary": nested,
        "decision": {
            "promoted": promoted,
            "copy_performed": copy_performed,
            "same_source_and_target": Path(args.candidate_model).resolve() == Path(args.best_model).resolve(),
            "reasons": reasons,
            "warnings": warnings,
        },
        "notes": [
            "Production promotion requires sealed nested folds and robust payout diagnostics.",
            "No promotion occurs when the nested summary belongs to a different model ID.",
            "Same source and target paths are handled without a SameFileError.",
        ],
    }
    output = Path(args.decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
