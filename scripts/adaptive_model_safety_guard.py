#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive post-evolution safety guard for specialized LOTO7 models.

Checks focus-window performance, chronological stability, operational-history
overlap, payout concentration, bootstrap downside, and optional true nested
walk-forward evidence.  Same-model candidates never bypass checks.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Genome, generate_tickets, genome_from_dict, load_draws  # noqa: E402
from merge_evolution_shards import load_prize_rows, select_target_indices, ticket_key  # noqa: E402
from scripts.robust_model_metrics import evaluate_model_robust  # noqa: E402

Ticket = Tuple[int, ...]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_genome_payload(path: str) -> Tuple[Genome, Dict[str, object]]:
    payload = read_json(path)
    raw = payload.get("genome", payload)
    if not isinstance(raw, dict):
        raise SystemExit(f"invalid genome payload: {path}")
    return genome_from_dict(raw), payload


def draw_year(draw: object) -> int:
    match = re.match(r"^(\d{4})", str(getattr(draw, "date", "") or ""))
    return int(match.group(1)) if match else 0


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


def parse_periods(raw: str) -> List[Tuple[Optional[int], Optional[int], str]]:
    periods: List[Tuple[Optional[int], Optional[int], str]] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
            start = int(left) if left.strip() else None
            end = int(right) if right.strip() else None
        else:
            start = int(item)
            end = int(item)
        periods.append((start, end, f"{start if start is not None else ''}:{end if end is not None else ''}"))
    return periods


def indices_for_period(draws: Sequence[object], base_indices: Sequence[int], start_year: Optional[int], end_year: Optional[int]) -> List[int]:
    output: List[int] = []
    for index in base_indices:
        year = draw_year(draws[index])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        output.append(index)
    return output


def parse_ticket(text: object) -> Optional[Ticket]:
    numbers = [int(value) for value in re.findall(r"\d+", str(text or ""))]
    if len(numbers) < 7:
        return None
    numbers = numbers[:7]
    if len(set(numbers)) != 7 or any(number < 1 or number > 37 for number in numbers):
        return None
    return ticket_key(numbers)


def load_history_tickets(path: str) -> List[Ticket]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        return []
    output: List[Ticket] = []
    seen = set()
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if not str(key or "").startswith("予測"):
                    continue
                ticket = parse_ticket(value)
                if ticket is None or ticket in seen:
                    continue
                seen.add(ticket)
                output.append(ticket)
    return output


def latest_model_tickets(genome: Genome, draws: Sequence[object], count: int) -> List[Ticket]:
    return [ticket_key(ticket) for ticket in generate_tickets(draws, genome, max(1, count))]


def operational_profile(tickets: Sequence[Ticket], history: Sequence[Ticket]) -> Dict[str, object]:
    if not tickets or not history:
        return {
            "ticket_count": len(tickets), "history_ticket_count": len(history),
            "exact_duplicates": 0, "max_overlap": 0, "avg_max_overlap": 0.0,
            "high_overlap_count": 0, "penalty": 0.0,
        }
    maximums: List[int] = []
    exact = 0
    for ticket in tickets:
        overlaps = [len(set(ticket) & set(previous)) for previous in history]
        maximums.append(max(overlaps) if overlaps else 0)
        if ticket in history:
            exact += 1
    maximum = max(maximums) if maximums else 0
    average = sum(maximums) / len(maximums) if maximums else 0.0
    high_count = sum(1 for value in maximums if value >= 6)
    penalty = exact * 1000.0 + high_count * 220.0 + max(0.0, average - 4.0) * 75.0
    return {
        "ticket_count": len(tickets),
        "history_ticket_count": len(history),
        "exact_duplicates": exact,
        "max_overlap": maximum,
        "avg_max_overlap": round(average, 3),
        "high_overlap_count": high_count,
        "penalty": round(penalty, 3),
        "tickets": [" ".join(f"{number:02d}" for number in ticket) for ticket in tickets],
    }


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "target_draws", "total_tickets", "total_cost", "total_payout", "profit",
        "roi", "roi_percent", "profit_roi_percent", "roi_excluding_top1_percent",
        "roi_excluding_top2_percent", "largest_draw_payout", "top1_payout_share",
        "payout_hhi", "median_year_roi_percent", "worst_year_roi_percent",
        "positive_year_count", "year_count", "cvar20_profit_per_draw",
        "bootstrap_roi_percent_p05", "bootstrap_roi_percent_p50", "bootstrap_roi_percent_p95",
        "grade_hit_count", "high_grade_hit_count", "max_main_match", "max_bonus_match",
        "rank_counts", "missing_prize_draw_count",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def robust_checks(
    baseline: Dict[str, object], candidate: Dict[str, object], args: argparse.Namespace
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []
    if as_int(candidate.get("target_draws")) < args.min_robust_draws:
        reasons.append(f"robust checks skipped: target_draws={candidate.get('target_draws')}")
        return True, reasons, warnings

    checks = [
        (
            "top1-removed ROI",
            as_float(candidate.get("roi_excluding_top1_percent")),
            as_float(baseline.get("roi_excluding_top1_percent")),
            args.min_top1_removed_roi_delta_percent,
        ),
        (
            "top2-removed ROI",
            as_float(candidate.get("roi_excluding_top2_percent")),
            as_float(baseline.get("roi_excluding_top2_percent")),
            args.min_top2_removed_roi_delta_percent,
        ),
        (
            "bootstrap p05 ROI",
            as_float(candidate.get("bootstrap_roi_percent_p05")),
            as_float(baseline.get("bootstrap_roi_percent_p05")),
            args.min_bootstrap_p05_delta_percent,
        ),
        (
            "worst-year ROI",
            as_float(candidate.get("worst_year_roi_percent")),
            as_float(baseline.get("worst_year_roi_percent")),
            args.min_worst_year_roi_delta_percent,
        ),
    ]
    passed = True
    for label, candidate_value, baseline_value, minimum_delta in checks:
        delta = candidate_value - baseline_value
        if delta < minimum_delta:
            passed = False
            warnings.append(
                f"{label} failed: candidate={candidate_value:.3f}% baseline={baseline_value:.3f}% delta={delta:.3f}pt"
            )
        else:
            reasons.append(
                f"{label} ok: candidate={candidate_value:.3f}% baseline={baseline_value:.3f}% delta={delta:.3f}pt"
            )

    candidate_share = as_float(candidate.get("top1_payout_share"))
    baseline_share = as_float(baseline.get("top1_payout_share"))
    allowed_share = max(args.max_top1_payout_share, baseline_share + args.max_top1_share_increase)
    if candidate_share > allowed_share:
        passed = False
        warnings.append(
            f"top1 payout concentration failed: candidate={candidate_share:.3f} allowed={allowed_share:.3f}"
        )
    else:
        reasons.append(
            f"top1 payout concentration ok: candidate={candidate_share:.3f} allowed={allowed_share:.3f}"
        )
    return passed, reasons, warnings


def nested_checks(path: str, candidate_model_id: str, args: argparse.Namespace) -> Tuple[bool, Dict[str, object], List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []
    file_path = Path(path) if path else None
    if file_path is None or not file_path.exists():
        message = "nested summary unavailable"
        if args.require_nested_summary:
            warnings.append(message)
            return False, {}, reasons, warnings
        reasons.append(message + "; soft check skipped")
        return True, {}, reasons, warnings
    try:
        summary = read_json(str(file_path))
    except Exception as exc:
        warnings.append(f"cannot read nested summary: {exc}")
        return (not args.require_nested_summary), {}, reasons, warnings

    reference_id = str(summary.get("reference_model_id") or "")
    if reference_id != candidate_model_id:
        message = f"nested summary model mismatch: summary={reference_id} candidate={candidate_model_id}"
        if args.require_nested_summary:
            warnings.append(message)
            return False, summary, reasons, warnings
        reasons.append(message + "; soft check skipped")
        return True, summary, reasons, warnings

    passed = True
    if bool(summary.get("future_leakage_detected")):
        passed = False
        warnings.append("nested validation reports future leakage")
    positive = as_int(summary.get("positive_roi_delta_folds"))
    median_delta = as_float(summary.get("median_roi_delta_percent"))
    worst_delta = as_float(summary.get("worst_roi_delta_percent"))
    top1_median_delta = as_float(summary.get("median_top1_removed_roi_delta_percent"))
    if positive < args.min_nested_positive_folds:
        passed = False
        warnings.append(f"nested positive folds failed: {positive} < {args.min_nested_positive_folds}")
    if median_delta < args.min_nested_median_roi_delta_percent:
        passed = False
        warnings.append(f"nested median ROI delta failed: {median_delta:.3f}pt")
    if worst_delta < args.min_nested_worst_roi_delta_percent:
        passed = False
        warnings.append(f"nested worst ROI delta failed: {worst_delta:.3f}pt")
    if top1_median_delta < args.min_nested_top1_removed_delta_percent:
        passed = False
        warnings.append(f"nested top1-removed median delta failed: {top1_median_delta:.3f}pt")
    if passed:
        reasons.append(
            f"nested validation ok: positive={positive} median={median_delta:.3f}pt worst={worst_delta:.3f}pt"
        )
    return passed, summary, reasons, warnings


def decide(
    *,
    focus_baseline: Dict[str, object],
    focus_candidate: Dict[str, object],
    walk_forward: Dict[str, Dict[str, object]],
    baseline_operational: Dict[str, object],
    candidate_operational: Dict[str, object],
    same_model: bool,
    independent_from_reference: Optional[bool],
    nested_ok: bool,
    nested_reasons: Sequence[str],
    nested_warnings: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = list(nested_reasons)
    warnings: List[str] = list(nested_warnings)
    if same_model:
        reasons.append("same model id detected; all safety checks still executed")

    independence_ok = True
    if args.require_independent_from_reference:
        independence_ok = independent_from_reference is True
        if independence_ok:
            reasons.append("model is independent from the reference model")
        else:
            warnings.append("model is not independent from the reference model")

    roi_delta = as_float(focus_candidate.get("roi_percent")) - as_float(focus_baseline.get("roi_percent"))
    profit_delta = as_int(focus_candidate.get("profit")) - as_int(focus_baseline.get("profit"))
    candidate_roi = as_float(focus_candidate.get("roi_percent"))
    focus_ok = (
        candidate_roi >= args.min_focus_roi_percent
        and roi_delta >= args.min_focus_roi_delta_percent
        and profit_delta >= args.min_focus_profit_delta
    )
    if focus_ok:
        reasons.append(f"focus window ok: roi={candidate_roi:.3f}% delta={roi_delta:.3f}pt profit_delta={profit_delta}")
    else:
        warnings.append(f"focus window failed: roi={candidate_roi:.3f}% delta={roi_delta:.3f}pt profit_delta={profit_delta}")

    high_delta = as_int(focus_candidate.get("high_grade_hit_count")) - as_int(focus_baseline.get("high_grade_hit_count"))
    max_main_delta = as_int(focus_candidate.get("max_main_match")) - as_int(focus_baseline.get("max_main_match"))
    high_ok = args.allow_high_grade_drop or high_delta >= 0 or max_main_delta >= args.allow_high_grade_drop_if_max_main_delta
    if high_ok:
        reasons.append(f"high-grade/max-main ok: high_delta={high_delta} max_main_delta={max_main_delta}")
    else:
        warnings.append(f"high-grade dropped: high_delta={high_delta} max_main_delta={max_main_delta}")

    walk_ok = True
    for label, pair in walk_forward.items():
        baseline = pair.get("baseline", {}) if isinstance(pair, dict) else {}
        candidate = pair.get("candidate", {}) if isinstance(pair, dict) else {}
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            continue
        draw_count = as_int(candidate.get("target_draws"))
        if draw_count < args.min_walk_forward_draws:
            reasons.append(f"walk-forward {label} skipped: target_draws={draw_count}")
            continue
        candidate_period_roi = as_float(candidate.get("roi_percent"))
        baseline_period_roi = as_float(baseline.get("roi_percent"))
        delta = candidate_period_roi - baseline_period_roi
        if candidate_period_roi < args.min_walk_forward_roi_percent or delta < args.max_walk_forward_roi_drop_percent:
            walk_ok = False
            warnings.append(
                f"walk-forward {label} failed: candidate={candidate_period_roi:.3f}% baseline={baseline_period_roi:.3f}% delta={delta:.3f}pt"
            )
        else:
            reasons.append(
                f"walk-forward {label} ok: candidate={candidate_period_roi:.3f}% baseline={baseline_period_roi:.3f}% delta={delta:.3f}pt"
            )

    operational_ok = True
    if as_int(candidate_operational.get("history_ticket_count")) > 0:
        baseline_penalty = as_float(baseline_operational.get("penalty"))
        candidate_penalty = as_float(candidate_operational.get("penalty"))
        candidate_exact = as_int(candidate_operational.get("exact_duplicates"))
        candidate_max_overlap = as_int(candidate_operational.get("max_overlap"))
        if candidate_exact > args.max_history_exact_duplicates:
            operational_ok = False
            warnings.append(f"too many exact history duplicates: {candidate_exact}")
        if candidate_max_overlap > args.max_history_max_overlap:
            operational_ok = False
            warnings.append(f"history max overlap exceeded: {candidate_max_overlap} > {args.max_history_max_overlap}")
        if candidate_penalty > baseline_penalty + args.max_history_penalty_delta:
            operational_ok = False
            warnings.append(
                f"history penalty worsened: baseline={baseline_penalty:.3f} candidate={candidate_penalty:.3f}"
            )
        if operational_ok:
            reasons.append(
                f"operational history ok: exact={candidate_exact} max_overlap={candidate_max_overlap} penalty={candidate_penalty:.3f}"
            )
    else:
        reasons.append("operational history skipped: no history tickets")

    robust_ok, robust_reasons, robust_warnings = robust_checks(focus_baseline, focus_candidate, args)
    reasons.extend(robust_reasons)
    warnings.extend(robust_warnings)
    accepted = bool(independence_ok and focus_ok and high_ok and walk_ok and operational_ok and robust_ok and nested_ok)
    return accepted, reasons, warnings


def write_report(path: str, payload: Dict[str, object]) -> None:
    decision = payload.get("decision", {}) if isinstance(payload.get("decision"), dict) else {}
    lines = [
        "LOTO7 Adaptive Model Safety Guard",
        "=================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"focus_start_year: {payload.get('focus_start_year')}",
        f"accepted: {decision.get('accepted')}",
        f"restored_baseline: {decision.get('restored_baseline')}",
        f"independent_from_reference: {payload.get('independent_from_reference')}",
        "",
        "[Reasons]",
    ]
    for item in decision.get("reasons", []) if isinstance(decision.get("reasons"), list) else []:
        lines.append(f"- {item}")
    lines.extend(["", "[Warnings]"])
    warnings = decision.get("warnings", []) if isinstance(decision.get("warnings"), list) else []
    lines.extend([f"- {item}" for item in warnings] if warnings else ["- none"])
    for title, key in (
        ("Focus Baseline Robust Metrics", "focus_baseline"),
        ("Focus Candidate Robust Metrics", "focus_candidate"),
        ("Operational History", "operational_history"),
        ("Walk Forward", "walk_forward"),
        ("Nested Walk Forward", "nested_walk_forward"),
    ):
        lines.extend(["", f"[{title}]", json.dumps(payload.get(key, {}), ensure_ascii=False, indent=2, sort_keys=True)])
    lines.extend(["", "注意: 過去検証上の安全ガードであり、将来の当せんや利益を保証しません。"])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a model with robust ROI, history, walk-forward and nested evidence.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--reference-model", default="")
    parser.add_argument("--require-independent-from-reference", action="store_true")
    parser.add_argument("--nested-summary", default="outputs/validation/nested_walk_forward_summary.json")
    parser.add_argument("--require-nested-summary", action="store_true")
    parser.add_argument("--focus-start-year", type=int, default=2020)
    parser.add_argument("--focus-end-year", type=int, default=None)
    parser.add_argument("--walk-forward-periods", default="2020:2022,2023:")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=350)
    parser.add_argument("--min-focus-roi-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-focus-profit-delta", type=int, default=0)
    parser.add_argument("--min-focus-roi-percent", type=float, default=8.0)
    parser.add_argument("--allow-high-grade-drop", action="store_true")
    parser.add_argument("--allow-high-grade-drop-if-max-main-delta", type=int, default=1)
    parser.add_argument("--min-walk-forward-draws", type=int, default=40)
    parser.add_argument("--min-walk-forward-roi-percent", type=float, default=1.0)
    parser.add_argument("--max-walk-forward-roi-drop-percent", type=float, default=-80.0)
    parser.add_argument("--max-history-exact-duplicates", type=int, default=2)
    parser.add_argument("--max-history-max-overlap", type=int, default=6)
    parser.add_argument("--max-history-penalty-delta", type=float, default=250.0)
    parser.add_argument("--min-robust-draws", type=int, default=60)
    parser.add_argument("--min-top1-removed-roi-delta-percent", type=float, default=-10.0)
    parser.add_argument("--min-top2-removed-roi-delta-percent", type=float, default=-15.0)
    parser.add_argument("--min-bootstrap-p05-delta-percent", type=float, default=-20.0)
    parser.add_argument("--min-worst-year-roi-delta-percent", type=float, default=-25.0)
    parser.add_argument("--max-top1-payout-share", type=float, default=0.90)
    parser.add_argument("--max-top1-share-increase", type=float, default=0.05)
    parser.add_argument("--min-nested-positive-folds", type=int, default=2)
    parser.add_argument("--min-nested-median-roi-delta-percent", type=float, default=0.0)
    parser.add_argument("--min-nested-worst-roi-delta-percent", type=float, default=-30.0)
    parser.add_argument("--min-nested-top1-removed-delta-percent", type=float, default=-5.0)
    parser.add_argument("--restore-on-fail", action="store_true", default=True)
    parser.add_argument("--no-restore-on-fail", dest="restore_on_fail", action="store_false")
    args = parser.parse_args()

    for required in (args.baseline_model, args.candidate_model):
        if not Path(required).exists():
            raise SystemExit(f"model not found: {required}")
    if args.require_independent_from_reference and not args.reference_model:
        raise SystemExit("--require-independent-from-reference requires --reference-model")

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    focus_indices = indices_for_period(draws, base_indices, args.focus_start_year, args.focus_end_year)
    if not focus_indices:
        raise SystemExit("no focus target indices selected")

    baseline_genome, _ = load_genome_payload(args.baseline_model)
    candidate_genome, _ = load_genome_payload(args.candidate_model)
    same_model = baseline_genome.id == candidate_genome.id
    reference_model_id: Optional[str] = None
    independent_from_reference: Optional[bool] = None
    if args.reference_model:
        reference_genome, _ = load_genome_payload(args.reference_model)
        reference_model_id = reference_genome.id
        independent_from_reference = candidate_genome.id != reference_genome.id

    def evaluate(genome: Genome, model_path: str, indices: Sequence[int], seed: int) -> Dict[str, object]:
        return evaluate_model_robust(
            genome=genome,
            model_path=model_path,
            draws=draws,
            prize_rows=prize_rows,
            target_indices=indices,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=seed,
        )

    focus_baseline = evaluate(baseline_genome, args.baseline_model, focus_indices, 101)
    focus_candidate = evaluate(candidate_genome, args.candidate_model, focus_indices, 101)
    walk_forward: Dict[str, Dict[str, object]] = {}
    for period_index, (start_year, end_year, label) in enumerate(parse_periods(args.walk_forward_periods), start=1):
        indices = indices_for_period(draws, base_indices, start_year, end_year)
        if not indices:
            walk_forward[label] = {"target_draws": 0, "baseline": {}, "candidate": {}}
            continue
        walk_forward[label] = {
            "target_draws": len(indices),
            "baseline": compact_metrics(evaluate(baseline_genome, args.baseline_model, indices, 200 + period_index)),
            "candidate": compact_metrics(evaluate(candidate_genome, args.candidate_model, indices, 200 + period_index)),
        }

    history = load_history_tickets(args.prediction_history)
    baseline_operational = operational_profile(latest_model_tickets(baseline_genome, draws, args.purchase_count), history)
    candidate_operational = operational_profile(latest_model_tickets(candidate_genome, draws, args.purchase_count), history)
    nested_ok, nested_summary, nested_reasons, nested_warnings = nested_checks(
        args.nested_summary, candidate_genome.id, args
    )
    accepted, reasons, warnings = decide(
        focus_baseline=focus_baseline,
        focus_candidate=focus_candidate,
        walk_forward=walk_forward,
        baseline_operational=baseline_operational,
        candidate_operational=candidate_operational,
        same_model=same_model,
        independent_from_reference=independent_from_reference,
        nested_ok=nested_ok,
        nested_reasons=nested_reasons,
        nested_warnings=nested_warnings,
        args=args,
    )

    restored = False
    Path(args.best_model).parent.mkdir(parents=True, exist_ok=True)
    if accepted:
        if Path(args.candidate_model).resolve() != Path(args.best_model).resolve():
            shutil.copyfile(args.candidate_model, args.best_model)
    elif args.restore_on_fail:
        shutil.copyfile(args.baseline_model, args.best_model)
        restored = True

    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_adaptive_model_safety_guard_v2",
        "csv": args.csv,
        "focus_start_year": args.focus_start_year,
        "focus_end_year": args.focus_end_year,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "baseline": {"path": args.baseline_model, "model_id": baseline_genome.id},
        "candidate": {"path": args.candidate_model, "model_id": candidate_genome.id},
        "reference_model": {"path": args.reference_model, "model_id": reference_model_id} if args.reference_model else None,
        "independent_from_reference": independent_from_reference,
        "focus_baseline": compact_metrics(focus_baseline),
        "focus_candidate": compact_metrics(focus_candidate),
        "walk_forward": walk_forward,
        "nested_walk_forward": nested_summary,
        "operational_history": {
            "history_path": args.prediction_history,
            "baseline": baseline_operational,
            "candidate": candidate_operational,
        },
        "decision": {
            "accepted": accepted,
            "same_model": same_model,
            "restore_on_fail": bool(args.restore_on_fail),
            "restored_baseline": restored,
            "reasons": reasons,
            "warnings": warnings,
            "robust_metrics_enabled": True,
            "nested_summary_required": bool(args.require_nested_summary),
        },
        "best_model_after_guard": args.best_model,
        "notes": [
            "Normal ROI, payout concentration, top-prize-excluded ROI, bootstrap downside and year stability are evaluated.",
            "True nested evidence is used when its reference model ID matches the candidate.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.summary, payload)
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
