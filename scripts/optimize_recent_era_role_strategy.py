#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/optimize_recent_era_role_strategy.py

Recent Era Role Strategy

role_ensemble_backtest.csv を2020年以降などの直近時代に絞り、
役割別成績からRecent Era専用の役割配分を生成する。

出力:
  outputs/recent_era/recent_role_strategy.json
  outputs/recent_era/recent_role_strategy_report.txt

注意:
  過去検証上の役割配分最適化であり、将来の当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROLE_LABELS = {
    "main_best": "本命: 採用ベストモデル",
    "high_match": "高一致狙い: ペア/3連/最大一致重視",
    "recent120": "直近寄り: 直近120回/60回の流れ重視",
    "mid_high": "中高数字補正: 20番台後半〜30番台も押さえる",
    "contrarian": "荒れ目/逆張り: 休眠・広めレンジ・低重複",
}
ROLES = list(ROLE_LABELS.keys())
RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_year(row: Dict[str, str]) -> int:
    raw = str(row.get("target_date") or "")
    m = re.match(r"^(\d{4})", raw)
    return int(m.group(1)) if m else 0


def empty_stats() -> Dict[str, object]:
    return {
        "draw_count": 0,
        "draw_hit_count": 0,
        "total_tickets": 0,
        "total_cost": 0,
        "total_payout": 0,
        "profit": 0,
        "roi": 0.0,
        "roi_percent": 0.0,
        "ticket_hit_rate_percent": 0.0,
        "draw_hit_rate_percent": 0.0,
        "grade_hit_count": 0,
        "high_grade_hit_count": 0,
        "max_main_match": 0,
        "max_bonus_match": 0,
        "rank_counts": {rank: 0 for rank in RANK_ORDER},
    }


def finalize(stats: Dict[str, object], draw_set: Set[int], hit_draw_set: Set[int], unit_cost: int) -> Dict[str, object]:
    total_tickets = int(stats.get("total_tickets", 0))
    total_cost = total_tickets * unit_cost
    total_payout = int(stats.get("total_payout", 0))
    ranks = stats.get("rank_counts", {}) if isinstance(stats.get("rank_counts"), dict) else {}
    grade_hit = sum(int(ranks.get(rank, 0)) for rank in PRIZE_RANKS)
    high_grade = sum(int(ranks.get(rank, 0)) for rank in ["1等", "2等", "3等", "4等"])
    profit = total_payout - total_cost
    roi = total_payout / total_cost if total_cost else 0.0
    out = dict(stats)
    out.update(
        {
            "draw_count": len(draw_set),
            "draw_hit_count": len(hit_draw_set),
            "total_cost": total_cost,
            "profit": profit,
            "roi": round(roi, 6),
            "roi_percent": round(roi * 100.0, 3),
            "grade_hit_count": grade_hit,
            "high_grade_hit_count": high_grade,
            "ticket_hit_rate_percent": round((grade_hit / total_tickets * 100.0) if total_tickets else 0.0, 3),
            "draw_hit_rate_percent": round((len(hit_draw_set) / len(draw_set) * 100.0) if draw_set else 0.0, 3),
        }
    )
    return out


def score_role(role: str, stats: Dict[str, object]) -> float:
    roi = float(stats.get("roi_percent", 0.0))
    profit = int(stats.get("profit", 0))
    draw_count = max(1, int(stats.get("draw_count", 0)))
    grade = int(stats.get("grade_hit_count", 0))
    high = int(stats.get("high_grade_hit_count", 0))
    max_main = int(stats.get("max_main_match", 0))
    hit_rate = float(stats.get("ticket_hit_rate_percent", 0.0))
    ranks = stats.get("rank_counts", {}) if isinstance(stats.get("rank_counts"), dict) else {}
    fifth = int(ranks.get("5等", 0))
    sixth = int(ranks.get("6等", 0))
    score = 0.0
    score += roi * 2.2
    score += (profit / draw_count) * 0.055
    score += high * 120.0
    score += grade * 3.0
    score += max_main * 18.0
    score += hit_rate * 3.0
    score += fifth * 1.4
    score -= max(0, sixth - fifth * 2) * 0.8
    if role == "recent120":
        score += 35.0
    if role == "high_match":
        score += 18.0
    if role == "main_best":
        score -= 20.0
    if role == "contrarian" and roi < 30.0:
        score -= 30.0
    return round(score, 6)


def normalize_counts(scores: Dict[str, float], purchase_count: int, max_main_best: int) -> Dict[str, int]:
    counts = {role: 0 for role in ROLES}
    shifted_min = min(scores.values()) if scores else 0.0
    shifted = {role: max(0.01, scores.get(role, 0.0) - shifted_min + 1.0) for role in ROLES}
    for _ in range(purchase_count):
        candidates = [role for role in ROLES if not (role == "main_best" and counts[role] >= max_main_best)]
        candidates = [role for role in candidates if counts[role] < 2] or candidates
        chosen = max(candidates, key=lambda role: shifted.get(role, 0.0) / float(counts[role] + 1))
        counts[chosen] += 1
    while sum(counts.values()) > purchase_count:
        role = min([r for r in ROLES if counts[r] > 0], key=lambda r: shifted.get(r, 0.0))
        counts[role] -= 1
    while sum(counts.values()) < purchase_count:
        candidates = [role for role in ROLES if not (role == "main_best" and counts[role] >= max_main_best)]
        role = max(candidates, key=lambda r: shifted.get(r, 0.0) / float(counts[r] + 1))
        counts[role] += 1
    return counts


def build_strategy(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Dict[str, object]:
    stats_by_role = {role: empty_stats() for role in ROLES}
    draws_by_role: Dict[str, Set[int]] = {role: set() for role in ROLES}
    hit_draws_by_role: Dict[str, Set[int]] = {role: set() for role in ROLES}

    for row in rows:
        if row.get("system") != "role_ensemble":
            continue
        if row_year(row) < args.start_year:
            continue
        role = str(row.get("role_key") or "")
        if role not in ROLE_LABELS:
            continue
        try:
            draw_no = int(row.get("target_draw_no") or 0)
            payout = int(row.get("payout") or 0)
            main_match = int(row.get("main_match") or 0)
            bonus_match = int(row.get("bonus_match") or 0)
        except Exception:
            continue
        rank = str(row.get("rank") or "外れ")
        stats = stats_by_role[role]
        stats["total_tickets"] = int(stats["total_tickets"]) + 1
        stats["total_payout"] = int(stats["total_payout"]) + payout
        stats["max_main_match"] = max(int(stats["max_main_match"]), main_match)
        stats["max_bonus_match"] = max(int(stats["max_bonus_match"]), bonus_match)
        ranks = stats["rank_counts"]
        if isinstance(ranks, dict):
            ranks[rank] = int(ranks.get(rank, 0)) + 1
        draws_by_role[role].add(draw_no)
        if rank != "外れ":
            hit_draws_by_role[role].add(draw_no)

    finalized = {
        role: finalize(stats_by_role[role], draws_by_role[role], hit_draws_by_role[role], args.unit_cost)
        for role in ROLES
    }
    completed = max((int(v.get("draw_count", 0)) for v in finalized.values()), default=0)
    scores = {role: score_role(role, finalized[role]) for role in ROLES}
    if completed < args.min_completed_draws:
        counts = {"main_best": 1, "high_match": 1, "recent120": 1, "mid_high": 1, "contrarian": 1}
        reason = f"fallback_default: completed={completed} < min_completed_draws={args.min_completed_draws}"
    else:
        counts = normalize_counts(scores, args.purchase_count, args.max_main_best)
        reason = "optimized_from_recent_era_role_backtest"

    role_sequence: List[Dict[str, object]] = []
    for role, count in counts.items():
        for _ in range(int(count)):
            role_sequence.append({"role": role, "label": ROLE_LABELS[role]})
    role_sequence.sort(key=lambda item: (item["role"] == "main_best", -scores.get(str(item["role"]), 0.0)))
    role_sequence = role_sequence[: args.purchase_count]

    return {
        "created_at": now_iso(),
        "kind": "loto7_recent_era_role_strategy",
        "source": args.input,
        "start_year": args.start_year,
        "purchase_count": args.purchase_count,
        "max_main_best": args.max_main_best,
        "completed_target_draws": completed,
        "strategy_counts": counts,
        "role_sequence": role_sequence,
        "scores": scores,
        "roles": finalized,
        "reason": reason,
        "notes": [
            "Role counts are optimized only from recent-era role_ensemble backtest rows.",
            "main_best is capped to reduce dependence on old high-payout hits.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }


def write_report(path: str, strategy: Dict[str, object]) -> None:
    lines = [
        "LOTO7 Recent Era Role Strategy Report",
        "====================================",
        "",
        f"created_at: {strategy.get('created_at')}",
        f"start_year: {strategy.get('start_year')}",
        f"reason: {strategy.get('reason')}",
        f"completed_target_draws: {strategy.get('completed_target_draws')}",
        f"max_main_best: {strategy.get('max_main_best')}",
        "",
        "[Strategy Counts]",
        json.dumps(strategy.get("strategy_counts", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Role Sequence]",
    ]
    for i, item in enumerate(strategy.get("role_sequence", []), start=1):
        if isinstance(item, dict):
            lines.append(f"{i}: {item.get('role')} / {item.get('label')}")
    lines.extend(["", "[Scores]", json.dumps(strategy.get("scores", {}), ensure_ascii=False, indent=2, sort_keys=True), "", "[Recent Role Stats]"])
    roles = strategy.get("roles", {}) if isinstance(strategy.get("roles"), dict) else {}
    for role, stats in roles.items():
        if isinstance(stats, dict):
            lines.append(
                f"{role}: ROI={stats.get('roi_percent')}% / profit={stats.get('profit')} / "
                f"grade={stats.get('grade_hit_count')} / high_grade={stats.get('high_grade_hit_count')} / "
                f"max_main={stats.get('max_main_match')}"
            )
    lines.append("")
    lines.append("注意: 過去検証上のRecent Era役割配分であり、将来の当せんや利益を保証しません。")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize recent-era role strategy from role ensemble backtest CSV.")
    parser.add_argument("--input", default="outputs/role_ensemble/role_ensemble_backtest.csv")
    parser.add_argument("--output", default="outputs/recent_era/recent_role_strategy.json")
    parser.add_argument("--report", default="outputs/recent_era/recent_role_strategy_report.txt")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--max-main-best", type=int, default=1)
    parser.add_argument("--min-completed-draws", type=int, default=80)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    strategy = build_strategy(rows, args)
    write_json(args.output, strategy)
    write_report(args.report, strategy)
    print(json.dumps(strategy, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
