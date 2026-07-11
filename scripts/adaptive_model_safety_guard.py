#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/adaptive_model_safety_guard.py

Post-adoption safety guard for specialized LOTO7 models.

Adds two protections that are intentionally outside the raw self-evolver:
  1. Walk-forward overfit guard: evaluate baseline/candidate across chronological windows.
  2. Operational history guard: avoid adopting a candidate whose latest tickets overlap too much
     with previous real-operation prediction tickets.

This script can restore the baseline model if the candidate fails the safety checks.

Notes:
  - This is a historical backtest/operational-safety guard only.
  - Lottery outcomes are random; this does not guarantee winnings or profit.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Genome, generate_tickets, genome_from_dict, load_draws  # noqa: E402
from merge_evolution_shards import evaluate_model_on_holdout, load_prize_rows, select_target_indices, ticket_key  # noqa: E402


Ticket = Tuple[int, ...]


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
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if ":" in item:
            left, right = item.split(":", 1)
            start = int(left) if left.strip() else None
            end = int(right) if right.strip() else None
        else:
            start = int(item)
            end = int(item)
        label = f"{start if start is not None else ''}:{end if end is not None else ''}"
        periods.append((start, end, label))
    return periods


def indices_for_year_period(draws: Sequence[object], base_indices: Sequence[int], start: Optional[int], end: Optional[int]) -> List[int]:
    out: List[int] = []
    for idx in base_indices:
        if idx < 0 or idx >= len(draws):
            continue
        year = draw_year(draws[idx])
        if start is not None and year < start:
            continue
        if end is not None and year > end:
            continue
        out.append(idx)
    return out


def evaluate(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, str]],
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


def parse_ticket(text: object) -> Optional[Ticket]:
    nums = [int(x) for x in re.findall(r"\d+", str(text or ""))]
    if len(nums) < 7:
        return None
    nums = nums[:7]
    if any(n < 1 or n > 37 for n in nums):
        return None
    return ticket_key(nums)


def load_history_tickets(path: str) -> List[Ticket]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return []
    tickets: List[Ticket] = []
    seen = set()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if not str(key or "").startswith("予測"):
                    continue
                ticket = parse_ticket(value)
                if ticket is None or ticket in seen:
                    continue
                seen.add(ticket)
                tickets.append(ticket)
    return tickets


def latest_model_tickets(genome: Genome, draws: Sequence[object], count: int) -> List[Ticket]:
    return [ticket_key(t) for t in generate_tickets(draws, genome, max(1, count))]


def operational_profile(tickets: Sequence[Ticket], history: Sequence[Ticket]) -> Dict[str, object]:
    if not tickets or not history:
        return {
            "ticket_count": len(tickets),
            "history_ticket_count": len(history),
            "exact_duplicates": 0,
            "max_overlap": 0,
            "avg_max_overlap": 0.0,
            "high_overlap_count": 0,
            "penalty": 0.0,
        }
    exact = 0
    max_overlaps: List[int] = []
    for ticket in tickets:
        overlaps = [len(set(ticket) & set(prev)) for prev in history]
        best = max(overlaps) if overlaps else 0
        max_overlaps.append(best)
        if ticket in history:
            exact += 1
    max_overlap = max(max_overlaps) if max_overlaps else 0
    high_overlap = sum(1 for v in max_overlaps if v >= 6)
    avg = sum(max_overlaps) / float(len(max_overlaps)) if max_overlaps else 0.0
    penalty = exact * 1000.0 + high_overlap * 220.0 + max(0.0, avg - 4.0) * 75.0
    return {
        "ticket_count": len(tickets),
        "history_ticket_count": len(history),
        "exact_duplicates": exact,
        "max_overlap": max_overlap,
        "avg_max_overlap": round(avg, 3),
        "high_overlap_count": high_overlap,
        "penalty": round(penalty, 3),
        "tickets": [" ".join(f"{n:02d}" for n in ticket) for ticket in tickets],
    }


def decide(
    *,
    focus_baseline: Dict[str, object],
    focus_candidate: Dict[str, object],
    walk_forward: Dict[str, Dict[str, object]],
    baseline_operational: Dict[str, object],
    candidate_operational: Dict[str, object],
    same_model: bool,
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []

    if same_model:
        reasons.append("same model id; safety guard keeps current model")
        return True, reasons, warnings

    focus_roi_delta = as_float(focus_candidate.get("roi_percent")) - as_float(focus_baseline.get("roi_percent"))
    focus_profit_delta = as_int(focus_candidate.get("profit")) - as_int(focus_baseline.get("profit"))
    focus_roi = as_float(focus_candidate.get("roi_percent"))
    high_delta = as_int(focus_candidate.get("high_grade_hit_count")) - as_int(focus_baseline.get("high_grade_hit_count"))
    max_main_delta = as_int(focus_candidate.get("max_main_match")) - as_int(focus_baseline.get("max_main_match"))

    focus_ok = focus_roi_delta >= args.min_focus_roi_delta_percent and focus_profit_delta >= args.min_focus_profit_delta and focus_roi >= args.min_focus_roi_percent
    if focus_ok:
        reasons.append(f"focus window ok: roi_delta={focus_roi_delta:.3f}pt profit_delta={focus_profit_delta}")
    else:
        warnings.append(
            f"focus window weak: roi={focus_roi:.3f}% delta={focus_roi_delta:.3f}pt profit_delta={focus_profit_delta}"
        )

    high_ok = args.allow_high_grade_drop or high_delta >= 0 or max_main_delta >= args.allow_high_grade_drop_if_max_main_delta
    if high_ok:
        reasons.append(f"high-grade/max-main ok: high_delta={high_delta} max_main_delta={max_main_delta}")
    else:
        warnings.append(f"high-grade dropped without max-main compensation: high_delta={high_delta} max_main_delta={max_main_delta}")

    wf_ok = True
    for label, pair in walk_forward.items():
        b = pair.get("baseline", {}) if isinstance(pair, dict) else {}
        c = pair.get("candidate", {}) if isinstance(pair, dict) else {}
        if not isinstance(b, dict) or not isinstance(c, dict):
            continue
        target_draws = as_int(c.get("target_draws"))
        if target_draws < args.min_walk_forward_draws:
            reasons.append(f"walk-forward {label} skipped: target_draws={target_draws}")
            continue
        c_roi = as_float(c.get("roi_percent"))
        b_roi = as_float(b.get("roi_percent"))
        c_profit = as_int(c.get("profit"))
        b_profit = as_int(b.get("profit"))
        roi_delta = c_roi - b_roi
        profit_delta = c_profit - b_profit
        # Allow weak older windows for specialized models, but block severe collapse.
        if c_roi < args.min_walk_forward_roi_percent or roi_delta < args.max_walk_forward_roi_drop_percent:
            wf_ok = False
            warnings.append(
                f"walk-forward {label} weak: candidate_roi={c_roi:.3f}% baseline_roi={b_roi:.3f}% delta={roi_delta:.3f}pt"
            )
        else:
            reasons.append(
                f"walk-forward {label} ok: candidate_roi={c_roi:.3f}% baseline_roi={b_roi:.3f}% delta={roi_delta:.3f}pt profit_delta={profit_delta}"
            )

    op_ok = True
    if as_int(candidate_operational.get("history_ticket_count")) > 0:
        b_penalty = as_float(baseline_operational.get("penalty"))
        c_penalty = as_float(candidate_operational.get("penalty"))
        b_exact = as_int(baseline_operational.get("exact_duplicates"))
        c_exact = as_int(candidate_operational.get("exact_duplicates"))
        c_max_overlap = as_int(candidate_operational.get("max_overlap"))
        if c_exact > max(args.max_history_exact_duplicates, b_exact) or c_max_overlap > args.max_history_max_overlap:
            op_ok = False
            warnings.append(
                f"operational history overlap too high: exact={c_exact} max_overlap={c_max_overlap} penalty={c_penalty}"
            )
        elif c_penalty > b_penalty + args.max_history_penalty_delta:
            op_ok = False
            warnings.append(
                f"operational history penalty worsened: baseline={b_penalty} candidate={c_penalty}"
            )
        else:
            reasons.append(
                f"operational history ok: exact={c_exact} max_overlap={c_max_overlap} penalty_delta={c_penalty - b_penalty:.3f}"
            )
    else:
        reasons.append("operational history skipped: no history tickets")

    accepted = bool(focus_ok and high_ok and wf_ok and op_ok)
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
        "",
        "[Reasons]",
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
    lines.extend([
        "",
        "[Focus Baseline]",
        json.dumps(payload.get("focus_baseline", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Focus Candidate]",
        json.dumps(payload.get("focus_candidate", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Operational History]",
        json.dumps(payload.get("operational_history", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Walk Forward]",
        json.dumps(payload.get("walk_forward", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "注意: 過去検証と運用履歴上の安全ガードであり、将来の当せんや利益を保証しません。",
    ])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a LOTO7 model with focus-window, walk-forward, and operational-history checks.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--focus-start-year", type=int, default=2020)
    parser.add_argument("--focus-end-year", type=int, default=None)
    parser.add_argument("--walk-forward-periods", default="2020:2022,2023:")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
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
    parser.add_argument("--restore-on-fail", action="store_true", default=True)
    parser.add_argument("--no-restore-on-fail", dest="restore_on_fail", action="store_false")
    args = parser.parse_args()

    if not Path(args.baseline_model).exists():
        raise SystemExit(f"baseline model not found: {args.baseline_model}")
    if not Path(args.candidate_model).exists():
        raise SystemExit(f"candidate model not found: {args.candidate_model}")

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    base_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    if not base_indices:
        raise SystemExit("no target indices selected")

    focus_indices = indices_for_year_period(draws, base_indices, args.focus_start_year, args.focus_end_year)
    if not focus_indices:
        raise SystemExit("no focus target indices selected")

    baseline_genome, _baseline_payload = load_genome_payload(args.baseline_model)
    candidate_genome, _candidate_payload = load_genome_payload(args.candidate_model)
    same_model = baseline_genome.id == candidate_genome.id

    focus_baseline = evaluate(
        genome=baseline_genome,
        model_path=args.baseline_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=focus_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )
    focus_candidate = evaluate(
        genome=candidate_genome,
        model_path=args.candidate_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=focus_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )

    walk_forward: Dict[str, Dict[str, object]] = {}
    for start, end, label in parse_periods(args.walk_forward_periods):
        indices = indices_for_year_period(draws, base_indices, start, end)
        if not indices:
            walk_forward[label] = {"target_draws": 0, "baseline": {}, "candidate": {}}
            continue
        walk_forward[label] = {
            "target_draws": len(indices),
            "baseline": compact_metrics(
                evaluate(
                    genome=baseline_genome,
                    model_path=args.baseline_model,
                    draws=draws,
                    prize_rows=prize_rows,
                    target_indices=indices,
                    purchase_count=args.purchase_count,
                    unit_cost=args.unit_cost,
                )
            ),
            "candidate": compact_metrics(
                evaluate(
                    genome=candidate_genome,
                    model_path=args.candidate_model,
                    draws=draws,
                    prize_rows=prize_rows,
                    target_indices=indices,
                    purchase_count=args.purchase_count,
                    unit_cost=args.unit_cost,
                )
            ),
        }

    history = load_history_tickets(args.prediction_history)
    baseline_operational = operational_profile(latest_model_tickets(baseline_genome, draws, args.purchase_count), history)
    candidate_operational = operational_profile(latest_model_tickets(candidate_genome, draws, args.purchase_count), history)

    accepted, reasons, warnings = decide(
        focus_baseline=focus_baseline,
        focus_candidate=focus_candidate,
        walk_forward=walk_forward,
        baseline_operational=baseline_operational,
        candidate_operational=candidate_operational,
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
        "kind": "loto7_adaptive_model_safety_guard",
        "csv": args.csv,
        "focus_start_year": args.focus_start_year,
        "focus_end_year": args.focus_end_year,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "baseline": {"path": args.baseline_model, "model_id": baseline_genome.id},
        "candidate": {"path": args.candidate_model, "model_id": candidate_genome.id},
        "focus_baseline": compact_metrics(focus_baseline),
        "focus_candidate": compact_metrics(focus_candidate),
        "walk_forward": walk_forward,
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
            "thresholds": {
                "min_focus_roi_delta_percent": args.min_focus_roi_delta_percent,
                "min_focus_profit_delta": args.min_focus_profit_delta,
                "min_focus_roi_percent": args.min_focus_roi_percent,
                "allow_high_grade_drop": bool(args.allow_high_grade_drop),
                "allow_high_grade_drop_if_max_main_delta": args.allow_high_grade_drop_if_max_main_delta,
                "min_walk_forward_roi_percent": args.min_walk_forward_roi_percent,
                "max_walk_forward_roi_drop_percent": args.max_walk_forward_roi_drop_percent,
                "max_history_exact_duplicates": args.max_history_exact_duplicates,
                "max_history_max_overlap": args.max_history_max_overlap,
                "max_history_penalty_delta": args.max_history_penalty_delta,
            },
        },
        "best_model_after_guard": args.best_model,
        "notes": [
            "Focus-window improvement, walk-forward stability, and operational-history overlap are all considered before keeping the candidate.",
            "This guard is conservative against overfitting and repeated real-operation losing patterns.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.summary, payload)
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
