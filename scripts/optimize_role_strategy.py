#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/optimize_role_strategy.py

Role Ensemble Backtest の役割別成績から、次回予測5口の役割配分を自動決定する。

入力:
  outputs/role_ensemble/role_ensemble_summary.json

出力:
  outputs/role_ensemble/role_strategy.json
  outputs/role_ensemble/role_strategy_report.txt

方針:
  - 完走前の途中集計でも暫定strategyを作る
  - ROI/収支/4等以上/最大一致/6等偏重抑制を複合評価
  - 5口合計になるように配分
  - 極端な偏りを避けるため、最低1口は本命または高一致系に残す

注意:
  過去検証上の役割配分最適化であり、将来の当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROLE_LABELS = {
    "main_best": "本命: 採用ベストモデル",
    "high_match": "高一致狙い: ペア/3連/最大一致重視",
    "recent120": "直近寄り: 直近120回/60回の流れ重視",
    "mid_high": "中高数字補正: 20番台後半〜30番台も押さえる",
    "contrarian": "荒れ目/逆張り: 休眠・広めレンジ・低重複",
}
DEFAULT_COUNTS = {
    "main_best": 1,
    "high_match": 1,
    "recent120": 1,
    "mid_high": 1,
    "contrarian": 1,
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rank_counts(stats: Dict[str, object]) -> Dict[str, int]:
    raw = stats.get("rank_counts", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


def score_role(role_key: str, stats: Dict[str, object]) -> float:
    ranks = rank_counts(stats)
    draw_count = max(1, int(stats.get("draw_count", 0)))
    total_tickets = max(1, int(stats.get("total_tickets", 0)))
    roi_percent = float(stats.get("roi_percent", 0.0))
    profit = int(stats.get("profit", 0))
    high_grade = int(stats.get("high_grade_hit_count", 0))
    grade = int(stats.get("grade_hit_count", 0))
    max_main = int(stats.get("max_main_match", 0))
    hit_rate = float(stats.get("ticket_hit_rate_percent", 0.0))

    # 役割ごとの比較なので、母数差が出ても公平になるよう1口/1回あたりへ正規化。
    profit_per_draw = profit / draw_count
    high_per_draw = high_grade / draw_count
    grade_per_ticket = grade / total_tickets
    fifth_per_draw = ranks.get("5等", 0) / draw_count
    sixth_per_draw = ranks.get("6等", 0) / draw_count

    score = 0.0
    score += roi_percent * 1.8
    score += profit_per_draw * 0.04
    score += high_per_draw * 900.0
    score += grade_per_ticket * 140.0
    score += max_main * 12.0
    score += hit_rate * 2.5
    score += fifth_per_draw * 45.0
    score -= max(0.0, sixth_per_draw - fifth_per_draw * 1.6) * 20.0

    # 戦略上の安定枠を軽く優遇。
    if role_key == "main_best":
        score += 12.0
    if role_key == "high_match":
        score += 18.0
    if role_key == "mid_high":
        score += 10.0
    if role_key == "contrarian" and roi_percent < 50.0:
        score -= 25.0
    return round(score, 6)


def normalize_counts(scores: Dict[str, float], purchase_count: int) -> Dict[str, int]:
    roles = list(DEFAULT_COUNTS.keys())
    if purchase_count <= 0:
        raise ValueError("purchase_count must be positive")
    if not scores:
        return dict(DEFAULT_COUNTS)

    # 低スコアでも完全排除しすぎないよう、正値へシフト。
    min_score = min(scores.values())
    shifted = {role: max(0.01, scores.get(role, 0.0) - min_score + 1.0) for role in roles}

    # まず最強役割に1口。高一致か本命のどちらかを最低1口残す。
    counts = {role: 0 for role in roles}
    top_role = max(roles, key=lambda r: shifted.get(r, 0.0))
    counts[top_role] += 1
    remaining = purchase_count - 1
    if remaining > 0 and counts.get("main_best", 0) == 0 and counts.get("high_match", 0) == 0:
        anchor = "high_match" if shifted.get("high_match", 0.0) >= shifted.get("main_best", 0.0) else "main_best"
        counts[anchor] += 1
        remaining -= 1

    for _ in range(max(0, remaining)):
        # 3口以上の集中は避ける。最終手段では許容。
        candidates = [r for r in roles if counts[r] < 2]
        if not candidates:
            candidates = roles
        chosen = max(candidates, key=lambda r: shifted.get(r, 0.0) / float(counts[r] + 1))
        counts[chosen] += 1

    # 合計補正。
    while sum(counts.values()) > purchase_count:
        removable = [r for r in roles if counts[r] > 0 and not (r in {"main_best", "high_match"} and counts["main_best"] + counts["high_match"] <= 1)]
        role = min(removable or roles, key=lambda r: shifted.get(r, 0.0))
        counts[role] -= 1
    while sum(counts.values()) < purchase_count:
        role = max(roles, key=lambda r: shifted.get(r, 0.0) / float(counts[r] + 1))
        counts[role] += 1
    return counts


def build_strategy(summary: Dict[str, object], purchase_count: int, min_completed_draws: int) -> Dict[str, object]:
    roles_raw = summary.get("roles", {})
    roles = roles_raw if isinstance(roles_raw, dict) else {}
    completed = int(summary.get("completed_target_draws", 0) or summary.get("target_draws", 0) or 0)
    status = str(summary.get("status", "unknown"))

    if completed < min_completed_draws or not roles:
        scores = {role: 0.0 for role in DEFAULT_COUNTS}
        counts = dict(DEFAULT_COUNTS)
        reason = f"fallback_default: completed_draws={completed} < min_completed_draws={min_completed_draws} or no role stats"
    else:
        scores = {}
        for role, stats in roles.items():
            if isinstance(stats, dict) and role in DEFAULT_COUNTS:
                scores[role] = score_role(role, stats)
        for role in DEFAULT_COUNTS:
            scores.setdefault(role, 0.0)
        counts = normalize_counts(scores, purchase_count)
        reason = "optimized_from_role_ensemble_backtest"

    role_sequence: List[Dict[str, object]] = []
    for role, count in counts.items():
        for _ in range(int(count)):
            role_sequence.append({"role": role, "label": ROLE_LABELS.get(role, role)})

    # 強い順に並べる。ただし本命がある場合は先頭に寄せる。
    role_sequence.sort(key=lambda item: (item["role"] != "main_best", -scores.get(str(item["role"]), 0.0)))
    role_sequence = role_sequence[:purchase_count]

    return {
        "created_at": now_iso(),
        "kind": "loto7_role_strategy",
        "source_summary_status": status,
        "source_completed_target_draws": completed,
        "source_target_draws_total": summary.get("target_draws_total"),
        "source_genome_id": summary.get("genome_id"),
        "purchase_count": purchase_count,
        "strategy_counts": counts,
        "role_sequence": role_sequence,
        "scores": scores,
        "reason": reason,
        "notes": [
            "Role counts are optimized from role_ensemble backtest role breakdown.",
            "Fallback default is used until enough completed target draws are available.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }


def write_report(path: str, strategy: Dict[str, object], summary: Dict[str, object]) -> None:
    roles = summary.get("roles", {}) if isinstance(summary.get("roles"), dict) else {}
    lines = [
        "LOTO7 Role Strategy Optimizer Report",
        "====================================",
        "",
        f"created_at: {strategy.get('created_at')}",
        f"reason: {strategy.get('reason')}",
        f"source_status: {strategy.get('source_summary_status')}",
        f"completed_target_draws: {strategy.get('source_completed_target_draws')} / {strategy.get('source_target_draws_total')}",
        f"genome_id: {strategy.get('source_genome_id')}",
        "",
        "[Strategy Counts]",
        json.dumps(strategy.get("strategy_counts", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Role Sequence]",
    ]
    for i, item in enumerate(strategy.get("role_sequence", []), start=1):
        if isinstance(item, dict):
            lines.append(f"{i}: {item.get('role')} / {item.get('label')}")
    lines.extend(["", "[Scores]", json.dumps(strategy.get("scores", {}), ensure_ascii=False, indent=2, sort_keys=True), "", "[Role Source Stats]"])
    for role, stats in roles.items():
        if isinstance(stats, dict):
            lines.append(
                f"{role}: ROI={stats.get('roi_percent')}% / profit={stats.get('profit')} / "
                f"grade={stats.get('grade_hit_count')} / high_grade={stats.get('high_grade_hit_count')} / "
                f"max_main={stats.get('max_main_match')}"
            )
    lines.append("")
    lines.append("注意: 過去検証上の役割配分最適化であり、将来の当せんや利益を保証しません。")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize LOTO7 role strategy from role ensemble backtest summary.")
    parser.add_argument("--summary", default="outputs/role_ensemble/role_ensemble_summary.json")
    parser.add_argument("--output", default="outputs/role_ensemble/role_strategy.json")
    parser.add_argument("--report", default="outputs/role_ensemble/role_strategy_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-completed-draws", type=int, default=80)
    args = parser.parse_args()

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if not Path(args.summary).exists():
        fallback = build_strategy({}, args.purchase_count, args.min_completed_draws)
        fallback["reason"] = f"fallback_default: summary not found: {args.summary}"
        write_json(args.output, fallback)
        write_report(args.report, fallback, {})
        print(json.dumps(fallback, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    summary = read_json(args.summary)
    strategy = build_strategy(summary, args.purchase_count, args.min_completed_draws)
    write_json(args.output, strategy)
    write_report(args.report, strategy, summary)
    print(json.dumps(strategy, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
