#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Choose the five role tickets from stable hit evidence, not jackpot outliers."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

ROLE_LABELS = {
    "main_best": "本命: 採用ベストモデル",
    "high_match": "高一致狙い: ペア/3連/最大一致重視",
    "recent120": "直近寄り: 直近120回/60回の流れ重視",
    "mid_high": "中高数字補正: 20番台後半〜30番台も押さえる",
    "contrarian": "荒れ目/逆張り: 休眠・広めレンジ・低重複",
}
DEFAULT_COUNTS = {role: 1 for role in ROLE_LABELS}
SCORING_VERSION = "loto7-role-strategy-2026.08.01-v2"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _int(row: Mapping[str, object], key: str) -> int:
    try:
        return int(float(str(row.get(key, 0) or 0).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def load_detail(path: str) -> Dict[str, List[Dict[str, object]]]:
    result = {role: [] for role in DEFAULT_COUNTS}
    source = Path(path)
    if not source.exists():
        return result
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw.get("system") != "role_ensemble":
                continue
            role = str(raw.get("role_key", ""))
            if role in result:
                result[role].append(dict(raw))
    for rows in result.values():
        rows.sort(key=lambda row: _int(row, "target_draw_no"))
    return result


def _score_rows(rows: Sequence[Mapping[str, object]], unit_cost: int) -> Dict[str, float]:
    draws = len(rows)
    if draws <= 0:
        return {
            "draws": 0.0,
            "hit_score": 0.0,
            "average_main_match": 0.0,
            "draw4_rate_percent": 0.0,
            "draw5_rate_percent": 0.0,
            "draw6_rate_percent": 0.0,
            "ticket_hit_rate_percent": 0.0,
            "payout_roi_percent": 0.0,
        }
    matches = [_int(row, "main_match") for row in rows]
    payouts = [_int(row, "payout") for row in rows]
    draw4 = sum(value >= 4 for value in matches) / draws * 100.0
    draw5 = sum(value >= 5 for value in matches) / draws * 100.0
    draw6 = sum(value >= 6 for value in matches) / draws * 100.0
    hit_rate = sum(str(row.get("rank", "外れ")) != "外れ" for row in rows) / draws * 100.0
    average = statistics.mean(matches)
    roi = sum(payouts) / max(1, draws * unit_cost) * 100.0
    hit_score = average * 20.0 + draw4 * 3.0 + draw5 * 8.0 + draw6 * 15.0 + hit_rate * 0.5
    return {
        "draws": float(draws),
        "hit_score": round(hit_score, 6),
        "average_main_match": round(average, 6),
        "draw4_rate_percent": round(draw4, 6),
        "draw5_rate_percent": round(draw5, 6),
        "draw6_rate_percent": round(draw6, 6),
        "ticket_hit_rate_percent": round(hit_rate, 6),
        "payout_roi_percent": round(roi, 6),
    }


def robust_role_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    recent_draws: int = 156,
    block_size: int = 52,
    unit_cost: int = 300,
) -> Dict[str, object]:
    ordered = sorted(rows, key=lambda row: _int(row, "target_draw_no"))
    full = _score_rows(ordered, unit_cost)
    recent = _score_rows(ordered[-recent_draws:], unit_cost)
    short = _score_rows(ordered[-max(block_size, recent_draws // 2):], unit_cost)
    blocks = [
        _score_rows(ordered[index:index + block_size], unit_cost)
        for index in range(0, len(ordered), block_size)
        if len(ordered[index:index + block_size]) >= max(12, block_size // 2)
    ]
    block_scores = [float(item["hit_score"]) for item in blocks] or [float(full["hit_score"])]
    payouts = sorted((_int(row, "payout") for row in ordered), reverse=True)
    total_payout = sum(payouts)
    top1 = payouts[0] / total_payout if total_payout > 0 else 0.0
    top3 = sum(payouts[:3]) / total_payout if total_payout > 0 else 0.0
    shrunk_roi = (
        float(full["payout_roi_percent"]) * len(ordered) + 15.0 * max(52, block_size)
    ) / max(1, len(ordered) + max(52, block_size))
    shrunk_roi = max(-100.0, min(40.0, shrunk_roi))
    score = (
        0.35 * float(recent["hit_score"])
        + 0.20 * float(short["hit_score"])
        + 0.20 * statistics.median(block_scores)
        + 0.15 * min(block_scores)
        + 0.10 * float(full["hit_score"])
        - 0.15 * (statistics.pstdev(block_scores) if len(block_scores) > 1 else 0.0)
        - 22.0 * top1
        - 8.0 * top3
        + 0.05 * shrunk_roi
    )
    return {
        "scoring_version": SCORING_VERSION,
        "score": round(score, 6),
        "full": full,
        "recent": recent,
        "short_recent": short,
        "block_count": len(blocks),
        "block_score_median": round(statistics.median(block_scores), 6),
        "block_score_min": round(min(block_scores), 6),
        "block_score_stddev": round(statistics.pstdev(block_scores) if len(block_scores) > 1 else 0.0, 6),
        "top1_payout_share": round(top1, 6),
        "top3_payout_share": round(top3, 6),
        "shrunk_payout_roi_percent": round(shrunk_roi, 6),
    }


def _fallback_counts(purchase_count: int, max_main_best: int) -> Dict[str, int]:
    counts = {role: 0 for role in DEFAULT_COUNTS}
    order = ["main_best", "high_match", "recent120", "mid_high", "contrarian"]
    for role in order:
        if sum(counts.values()) >= purchase_count:
            break
        if role != "main_best" or max_main_best > 0:
            counts[role] += 1
    while sum(counts.values()) < purchase_count:
        counts["high_match"] += 1
    return counts


def _allocate(
    scores: Mapping[str, float], caps: Mapping[str, int], purchase_count: int
) -> Dict[str, int]:
    counts = {role: 0 for role in DEFAULT_COUNTS}
    for _ in range(purchase_count):
        candidates = [
            role for role in DEFAULT_COUNTS
            if counts[role] < max(0, int(caps.get(role, 0)))
        ]
        if not candidates:
            candidates = list(DEFAULT_COUNTS)
        chosen = max(
            candidates,
            key=lambda role: (
                scores.get(role, 0.0) / (counts[role] + 1),
                scores.get(role, 0.0),
                role,
            ),
        )
        counts[chosen] += 1
    if counts["main_best"] + counts["high_match"] == 0:
        donor = max(
            (role for role in counts if counts[role] > 0),
            key=lambda role: counts[role],
        )
        counts[donor] -= 1
        counts["high_match"] += 1
    return counts


def build_strategy(
    summary: Mapping[str, object],
    detail_rows: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    *,
    purchase_count: int = 5,
    min_completed_draws: int = 80,
    max_main_best: int = 1,
    max_role_count: int = 2,
    max_concentrated_role_count: int = 1,
    max_top1_share: float = 0.35,
    recent_draws: int = 156,
    block_size: int = 52,
    unit_cost: int = 300,
) -> Dict[str, object]:
    completed = int(summary.get("completed_target_draws", 0) or summary.get("target_draws", 0) or 0)
    details = detail_rows or {}
    metrics = {
        role: robust_role_metrics(
            details.get(role, []),
            recent_draws=recent_draws,
            block_size=block_size,
            unit_cost=unit_cost,
        )
        for role in DEFAULT_COUNTS
    }
    scores = {role: float(value["score"]) for role, value in metrics.items()}
    scores["main_best"] += 3.0
    scores["high_match"] += 2.0
    caps = {role: max_role_count for role in DEFAULT_COUNTS}
    caps["main_best"] = max_main_best
    concentrated: List[str] = []
    for role, value in metrics.items():
        if (
            float(value["top1_payout_share"]) > max_top1_share
            or float(value["top3_payout_share"]) > 0.65
        ):
            caps[role] = min(caps[role], max_concentrated_role_count)
            concentrated.append(role)
    enough_detail = completed >= min_completed_draws and all(
        details.get(role) for role in DEFAULT_COUNTS
    )
    if enough_detail:
        counts = _allocate(scores, caps, purchase_count)
        reason = "robust_recent_block_hit_optimization"
    else:
        counts = _fallback_counts(purchase_count, max_main_best)
        reason = "fallback_default_insufficient_detail"
    sequence: List[Dict[str, object]] = []
    for role in sorted(
        DEFAULT_COUNTS,
        key=lambda item: (item != "main_best", -scores[item]),
    ):
        sequence.extend(
            {"role": role, "label": ROLE_LABELS[role]}
            for _ in range(counts[role])
        )
    return {
        "created_at": now_iso(),
        "kind": "loto7_role_strategy",
        "scoring_version": SCORING_VERSION,
        "source_summary_status": summary.get("status", "unknown"),
        "source_completed_target_draws": completed,
        "source_target_draws_total": summary.get("target_draws_total"),
        "source_genome_id": summary.get("genome_id"),
        "purchase_count": purchase_count,
        "max_main_best": max_main_best,
        "strategy_counts": counts,
        "role_sequence": sequence[:purchase_count],
        "scores": {role: round(score, 6) for role, score in scores.items()},
        "role_caps": caps,
        "concentrated_roles": concentrated,
        "robust_metrics": metrics,
        "reason": reason,
        "notes": [
            "Main-number hit stability and recent chronological blocks determine role counts.",
            "Raw profit is not a learning reward; payout ROI is strongly shrunk and only lightly weighted.",
            "Roles dominated by one payout are capped at one ticket.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }


def write_report(path: str, strategy: Mapping[str, object]) -> None:
    lines = [
        "LOTO7 Robust Role Strategy Report",
        "=================================",
        f"created_at: {strategy.get('created_at')}",
        f"scoring_version: {strategy.get('scoring_version')}",
        f"reason: {strategy.get('reason')}",
        f"counts: {json.dumps(strategy.get('strategy_counts', {}), ensure_ascii=False, sort_keys=True)}",
        f"concentrated_roles: {strategy.get('concentrated_roles')}",
        "",
        "[Scores]",
        json.dumps(strategy.get("scores", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Robust Metrics]",
        json.dumps(strategy.get("robust_metrics", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "注意: 過去検証上の配分最適化であり、将来の当せんや利益を保証しません。",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize LOTO7 role strategy from stable hit evidence."
    )
    parser.add_argument("--summary", default="outputs/role_ensemble/role_ensemble_summary.json")
    parser.add_argument("--detail", default="outputs/role_ensemble/role_ensemble_backtest.csv")
    parser.add_argument("--output", default="outputs/role_ensemble/role_strategy.json")
    parser.add_argument("--report", default="outputs/role_ensemble/role_strategy_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-completed-draws", type=int, default=80)
    parser.add_argument("--max-main-best", type=int, default=1)
    parser.add_argument("--max-role-count", type=int, default=2)
    parser.add_argument("--max-concentrated-role-count", type=int, default=1)
    parser.add_argument("--max-top1-share", type=float, default=0.35)
    parser.add_argument("--recent-draws", type=int, default=156)
    parser.add_argument("--block-size", type=int, default=52)
    parser.add_argument("--unit-cost", type=int, default=300)
    args = parser.parse_args()
    if args.purchase_count <= 0 or args.max_main_best < 0:
        raise SystemExit("invalid purchase count or main-best cap")
    summary = read_json(args.summary) if Path(args.summary).exists() else {}
    strategy = build_strategy(
        summary,
        load_detail(args.detail),
        purchase_count=args.purchase_count,
        min_completed_draws=args.min_completed_draws,
        max_main_best=args.max_main_best,
        max_role_count=args.max_role_count,
        max_concentrated_role_count=args.max_concentrated_role_count,
        max_top1_share=args.max_top1_share,
        recent_draws=args.recent_draws,
        block_size=args.block_size,
        unit_cost=args.unit_cost,
    )
    write_json(args.output, strategy)
    write_report(args.report, strategy)
    print(json.dumps(strategy, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
