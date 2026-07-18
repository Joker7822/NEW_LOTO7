#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update fourth-generation shadow history and anytime-valid e-process state."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, evaluate_ticket, load_draws  # noqa: E402
from scripts.evaluation_core import load_prize_rows, prize_amount_for_rank  # noqa: E402
from scripts.generation4_core import bounded_strategy_utility, eprocess_from_history  # noqa: E402

Ticket = Tuple[int, ...]
FIELDS = [
    "prediction_draw_no", "prediction_date", "strategy", "tickets_json", "status",
    "actual_main", "actual_bonus", "max_main_match", "total_main_matches",
    "winning_tickets", "total_payout", "utility", "created_at",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_ticket(text: object) -> Optional[Ticket]:
    values = [int(token) for token in str(text or "").replace(",", " ").split() if token.strip().isdigit()]
    key = tuple(sorted(values))
    if len(key) != 7 or len(set(key)) != 7 or any(number < 1 or number > 37 for number in key):
        return None
    return key


def parse_strategy_tickets(value: object) -> List[Ticket]:
    output: List[Ticket] = []
    if not isinstance(value, list):
        return output
    for item in value:
        ticket = parse_ticket(item)
        if ticket is not None:
            output.append(ticket)
    return output


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def evaluate_row(row: Dict[str, str], target: Draw, prize_row: Dict[str, str]) -> None:
    try:
        raw = json.loads(row.get("tickets_json") or "[]")
    except json.JSONDecodeError:
        raw = []
    tickets = parse_strategy_tickets(raw)
    max_main = 0
    total_main = 0
    winners = 0
    payout = 0
    for ticket in tickets:
        main_match, _bonus_match, rank = evaluate_ticket(ticket, target)
        max_main = max(max_main, main_match)
        total_main += main_match
        if rank != "外れ":
            winners += 1
        payout += prize_amount_for_rank(prize_row, rank)
    row.update({
        "prediction_date": target.date,
        "status": "evaluated",
        "actual_main": " ".join(f"{number:02d}" for number in target.main),
        "actual_bonus": " ".join(f"{number:02d}" for number in target.bonus),
        "max_main_match": str(max_main),
        "total_main_matches": str(total_main),
        "winning_tickets": str(winners),
        "total_payout": str(payout),
        "utility": f"{bounded_strategy_utility(max_main, total_main, winners):.9f}",
    })


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item.get("prediction_draw_no") or 0), item.get("strategy") or "")):
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    temp.replace(path)


def write_report(path: Path, payload: Dict[str, object]) -> None:
    state = payload.get("eprocess", {}) if isinstance(payload.get("eprocess"), dict) else {}
    lines = [
        "LOTO7 Generation 4 Champion / Challenger",
        "========================================",
        "",
        f"created_at: {payload.get('created_at')}",
        f"history_rows: {payload.get('history_rows')}",
        f"evaluated_rows: {payload.get('evaluated_rows')}",
        f"pending_rows: {payload.get('pending_rows')}",
        f"e_value: {state.get('e_value')}",
        f"reverse_e_value: {state.get('reverse_e_value')}",
        f"evaluated_draws: {state.get('evaluated_draws')}",
        f"decision: {state.get('decision')}",
        "",
        "e-processは抽せんごとに確認しても停止時点の誤判定を抑えるための逐次証拠です。",
        "当せんや利益を保証するものではありません。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Update generation 4 shadow history and e-process.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--latest-shadow", default="outputs/generation4/latest_shadow_predictions.json")
    parser.add_argument("--history", default="outputs/generation4/shadow_history.csv")
    parser.add_argument("--summary", default="outputs/generation4/champion_challenger_summary.json")
    parser.add_argument("--report", default="outputs/generation4/champion_challenger_report.txt")
    parser.add_argument("--challenger", default="generation4")
    parser.add_argument("--champion", default="beam_baseline")
    parser.add_argument("--betting-fraction", type=float, default=0.25)
    parser.add_argument("--promotion-threshold", type=float, default=20.0)
    parser.add_argument("--min-evaluated-draws", type=int, default=30)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing rows without appending latest-shadow predictions")
    args = parser.parse_args(argv)

    draws = load_draws(args.csv)
    draws_by_no = {draw.draw_no: draw for draw in draws}
    prize_rows = load_prize_rows(args.csv)
    history_path = Path(args.history)
    rows = load_rows(history_path)

    for row in rows:
        if row.get("status") == "evaluated":
            continue
        try:
            draw_no = int(row.get("prediction_draw_no") or 0)
        except ValueError:
            continue
        target = draws_by_no.get(draw_no)
        if target is not None:
            evaluate_row(row, target, prize_rows.get(draw_no, {}))

    latest_path = Path(args.latest_shadow)
    if not args.evaluate_only and latest_path.exists() and latest_path.stat().st_size > 0:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        draw_no = int(latest.get("prediction_draw_no") or 0)
        strategies = latest.get("strategies", {})
        existing = {(int(row.get("prediction_draw_no") or 0), row.get("strategy") or "") for row in rows}
        if isinstance(strategies, dict) and draw_no > 0:
            for strategy, tickets in strategies.items():
                key = (draw_no, str(strategy))
                if key in existing or not isinstance(tickets, list):
                    continue
                row = {
                    "prediction_draw_no": str(draw_no),
                    "prediction_date": "",
                    "strategy": str(strategy),
                    "tickets_json": json.dumps(tickets, ensure_ascii=False, separators=(",", ":")),
                    "status": "pending",
                    "actual_main": "", "actual_bonus": "", "max_main_match": "",
                    "total_main_matches": "", "winning_tickets": "", "total_payout": "",
                    "utility": "", "created_at": now_iso(),
                }
                target = draws_by_no.get(draw_no)
                if target is not None:
                    evaluate_row(row, target, prize_rows.get(draw_no, {}))
                rows.append(row)

    write_rows(history_path, rows)
    eprocess = eprocess_from_history(
        args.history,
        challenger=args.challenger,
        champion=args.champion,
        betting_fraction=args.betting_fraction,
        promotion_threshold=args.promotion_threshold,
        min_evaluated_draws=args.min_evaluated_draws,
    )
    payload: Dict[str, object] = {
        "created_at": now_iso(),
        "kind": "loto7_generation4_champion_challenger",
        "history_path": args.history,
        "history_rows": len(rows),
        "evaluated_rows": sum(1 for row in rows if row.get("status") == "evaluated"),
        "pending_rows": sum(1 for row in rows if row.get("status") != "evaluated"),
        "update_mode": "evaluate_only" if args.evaluate_only else "evaluate_and_append_latest",
        "eprocess": eprocess,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(Path(args.report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
