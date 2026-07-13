#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/adaptive_model_safety_guard.py

Post-adoption safety guard for specialized LOTO7 models.

Checks:
  1. Focus-window ROI/profit.
  2. Chronological walk-forward stability.
  3. Operational prediction-history overlap.
  4. Optional independence from a reference model (used by Super Recent).

The checks are always executed, even when baseline and candidate model IDs are equal.
This prevents same-model candidates from bypassing history/walk-forward protection.
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
    return {key: metrics.get(key) for key in keys if key in metrics}


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


def indices_for_period(
    draws: Sequence[object],
    base_indices: Sequence[int],
    start_year: Optional[int],
    end_year: Optional[int],
) -> List[int]:
    selected: List[int] = []
    for idx in base_indices:
        if idx < 0 or idx >= len(draws):
            continue
        year = draw_year(draws[idx])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(idx)
    return selected


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
    numbers = [int(value) for value in re.findall(r"\d+", str(text or ""))]
    if len(numbers) < 7:
        return None
    numbers = numbers[:7]
    if any(number < 1 or number > 37 for number in numbers):
        return None
    return ticket_key(numbers)


def load_history_tickets(path: str) -> List[Ticket]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return []
    output: List[Ticket] = []
    seen = set()
    with p.open("r", encoding="utf-8-sig", newline="") as stream:
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
            "ticket_count": len(tickets),
            "history_ticket_count": len(history),
            "exact_duplicates": 0,
            "max_overlap": 0,
            "avg_max_overlap": 0.0,
            "high_overlap_count": 0,
            "penalty": 0.0,
        }

    maximums: List[int] = []
    exact_duplicates = 0
    for ticket in tickets:
        overlaps = [len(set(ticket) & set(previous)) for previous in history]
        maximums.append(max(overlaps) if overlaps else 0)
        if ticket in history:
            exact_duplicates += 1

    maximum = max(maximums) if maximums else 0
    average = sum(maximums) / float(len(maximums)) if maximums else 0.0
    high_overlap_count = sum(1 for value in maximums if value >= 6)
    penalty = exact_duplicates * 1000.0 + high_overlap_count * 220.0 + max(0.0, average - 4.0) * 75.0
    return {
        "ticket_count": len(tickets),
        "history_ticket_count": len(history),
        "exact_duplicates": exact_duplicates,
        "max_overlap": maximum,
        "avg_max_overlap": round(average, 3),
        "high_overlap_count": high_overlap_count,
        "penalty": round(penalty, 3),
        "tickets": [" ".join(f"{number:02d}" for number in ticket) for ticket in tickets],
    }


def decide(
    *,
    focus_baseline: Dict[str, object],
    focus_candidate: Dict[str, object],
    walk_forward: Dict[str, Dict[str, object]],
    baseline_operational: Dict[str, object],
    candidate_operational: Dict[str, object],
    same_model: bool,
    independent_from_reference: Optional[bool],
    args: argparse.Namespace,
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    warnings: List[str] = []

    # Do not return early for same-model candidates. All safety checks below must run.
    if same_model:
        reasons.append("same model id detected; full safety checks were still executed")

    independence_ok = True
    if args.require_independent_from_reference:
        independence_ok = independent_from_reference is True
        if independence_ok:
            reasons.append("model id is independent from the reference model")
        else:
            warnings.append("model id is not independent from the reference model")

    roi_delta = as_float(focus_candidate.get("roi_percent")) - as_float(focus_baseline.get("roi_percent"))
    profit_delta = as_int(focus_candidate.get("profit")) - as_int(focus_baseline.get("profit"))
    candidate_roi = as_float(focus_candidate.get("roi_percent"))
    high_grade_delta = as_int(focus_candidate.get("high_grade_hit_count")) - as_int(focus_baseline.get("high_grade_hit_count"))
    max_main_delta = as_int(focus_candidate.get("max_main_match")) - as_int(focus_baseline.get("max_main_match"))

    focus_ok = (
        candidate_roi >= args.min_focus_roi_percent
        and roi_delta >= args.min_focus_roi_delta_percent
        and profit_delta >= args.min_focus_profit_delta
    )
    if focus_ok:
        reasons.append(f"focus window ok: roi={candidate_roi:.3f}% roi_delta={roi_delta:.3f}pt profit_delta={profit_delta}")
    else:
        warnings.append(f"focus window failed: roi={candidate_roi:.3f}% roi_delta={roi_delta:.3f}pt profit_delta={profit_delta}")

    high_grade_ok = (
        args.allow_high_grade_drop
        or high_grade_delta >= 0
        or max_main_delta >= args.allow_high_grade_drop_if_max_main_delta
    )
    if high_grade_ok:
        reasons.append(f"high-grade/max-main ok: high_delta={high_grade_delta} max_main_delta={max_main_delta}")
    else:
        warnings.append(f"high-grade dropped without compensation: high_delta={high_grade_delta} max_main_delta={max_main_delta}")

    walk_forward_ok = True
    for label, pair in walk_forward.items():
        baseline = pair.get("baseline", {}) if isinstance(pair, dict) else {}
        candidate = pair.get("candidate", {}) if isinstance(pair, dict) else {}
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            continue
        draws_count = as_int(candidate.get("target_draws"))
        if draws_count < args.min_walk_forward_draws:
            reasons.append(f"walk-forward {label} skipped: target_draws={draws_count}")
            continue
        baseline_roi = as_float(baseline.get("roi_percent"))
        period_roi = as_float(candidate.get("roi_percent"))
        period_delta = period_roi - baseline_roi
        if period_roi < args.min_walk_forward_roi_percent or period_delta < args.max_walk_forward_roi_drop_percent:
            walk_forward_ok = False
            warnings.append(
                f"walk-forward {label} failed: candidate_roi={period_roi:.3f}% baseline_roi={baseline_roi:.3f}% delta={period_delta:.3f}pt"
            )
        else:
            reasons.append(
                f"walk-forward {label} ok: candidate_roi={period_roi:.3f}% baseline_roi={baseline_roi:.3f}% delta={period_delta:.3f}pt"
            )

    operational_ok = True
    if as_int(candidate_operational.get("history_ticket_count")) > 0:
        baseline_penalty = as_float(baseline_operational.get("penalty"))
        candidate_penalty = as_float(candidate_operational.get("penalty"))
        baseline_exact = as_int(baseline_operational.get("exact_duplicates"))
        candidate_exact = as_int(candidate_operational.get("exact_duplicates"))
        candidate_max_overlap = as_int(candidate_operational.get("max_overlap"))
        if candidate_exact > max(args.max_history_exact_duplicates, baseline_exact):
            operational_ok = False
            warnings.append(f"too many exact history duplicates: {candidate_exact}")
        if candidate_max_overlap > args.max_history_max_overlap:
            operational_ok = False
            warnings.append(
                f"history max overlap exceeded: {candidate_max_overlap} > {args.max_history_max_overlap}"
            )
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

    accepted = bool(independence_ok and focus_ok and high_grade_ok and walk_forward_ok and operational_ok)
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
    lines.extend(
        [
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
        ]
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Guard a LOTO7 model with focus, walk-forward, history, and independence checks.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--reference-model", default="")
    parser.add_argument("--require-independent-from-reference", action="store_true")
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
    if not base_indices:
        raise SystemExit("no target indices selected")

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
    for start_year, end_year, label in parse_periods(args.walk_forward_periods):
        indices = indices_for_period(draws, base_indices, start_year, end_year)
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
        independent_from_reference=independent_from_reference,
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
        "reference_model": {"path": args.reference_model, "model_id": reference_model_id} if args.reference_model else None,
        "independent_from_reference": independent_from_reference,
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
                "require_independent_from_reference": bool(args.require_independent_from_reference),
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
            "Same-model candidates do not bypass focus, walk-forward, or operational-history checks.",
            "Super Recent can require a model ID different from the Recent Era reference model.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.summary, payload)
    write_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
