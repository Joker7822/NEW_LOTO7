#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/detect_loto7_regime.py

最新の抽せん履歴から現在の「レジーム(状態)」を判定し、
その状態に合わせた role_ensemble 用の5口配分を生成する。

入力:
  loto7.csv
  outputs/role_ensemble/role_strategy.json  (任意: Role Strategy Optimizerの基本配分)

出力:
  outputs/role_ensemble/regime_state.json
  outputs/role_ensemble/regime_strategy.json
  outputs/role_ensemble/regime_strategy_report.txt

注意:
  過去傾向の状態分類であり、将来の当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Draw, load_draws  # noqa: E402

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
REGIME_ADJUSTMENTS = {
    "mid_high_trend": {"mid_high": 2, "high_match": 1, "main_best": 1, "contrarian": 1, "recent120": 0},
    "low_balanced_trend": {"main_best": 1, "recent120": 2, "high_match": 1, "mid_high": 1, "contrarian": 0},
    "recent_repeat_trend": {"recent120": 2, "high_match": 2, "main_best": 1, "mid_high": 0, "contrarian": 0},
    "consecutive_trend": {"high_match": 2, "recent120": 1, "main_best": 1, "mid_high": 1, "contrarian": 0},
    "high_variance_trend": {"contrarian": 2, "high_match": 1, "mid_high": 1, "main_best": 1, "recent120": 0},
    "contrarian_gap_trend": {"contrarian": 2, "mid_high": 1, "main_best": 1, "high_match": 1, "recent120": 0},
    "balanced": DEFAULT_COUNTS,
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_if_exists(path: str) -> Dict[str, object]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] cannot read json {path}: {exc}")
        return {}


def avg(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def consecutive_pairs(nums: Sequence[int]) -> int:
    ordered = sorted(nums)
    return sum(1 for a, b in zip(ordered, ordered[1:]) if b == a + 1)


def draw_overlap(a: Draw, b: Draw) -> int:
    return len(set(a.main) & set(b.main))


def window_metrics(draws: Sequence[Draw]) -> Dict[str, float]:
    if not draws:
        return {
            "avg_sum": 0.0,
            "avg_high_count": 0.0,
            "avg_low_count": 0.0,
            "avg_mid_high_count": 0.0,
            "avg_odd_count": 0.0,
            "avg_consecutive_pairs": 0.0,
            "sum_stddev": 0.0,
            "avg_repeat_from_previous": 0.0,
        }
    sums = [sum(d.main) for d in draws]
    high_counts = [sum(1 for n in d.main if n >= 25) for d in draws]
    low_counts = [sum(1 for n in d.main if n <= 18) for d in draws]
    mid_high_counts = [sum(1 for n in d.main if 21 <= n <= 36) for d in draws]
    odd_counts = [sum(1 for n in d.main if n % 2) for d in draws]
    consecutive = [consecutive_pairs(d.main) for d in draws]
    repeats = [draw_overlap(prev, cur) for prev, cur in zip(draws, draws[1:])]
    return {
        "avg_sum": round(avg(sums), 4),
        "avg_high_count": round(avg(high_counts), 4),
        "avg_low_count": round(avg(low_counts), 4),
        "avg_mid_high_count": round(avg(mid_high_counts), 4),
        "avg_odd_count": round(avg(odd_counts), 4),
        "avg_consecutive_pairs": round(avg(consecutive), 4),
        "sum_stddev": round(statistics.pstdev(sums), 4) if len(sums) >= 2 else 0.0,
        "avg_repeat_from_previous": round(avg(repeats), 4),
    }


def score_regimes(recent20: Dict[str, float], recent60: Dict[str, float], prior120: Dict[str, float]) -> Dict[str, float]:
    sum_delta = recent20["avg_sum"] - prior120["avg_sum"]
    high_delta = recent20["avg_high_count"] - prior120["avg_high_count"]
    low_delta = recent20["avg_low_count"] - prior120["avg_low_count"]
    repeat_delta = recent20["avg_repeat_from_previous"] - prior120["avg_repeat_from_previous"]
    consecutive_delta = recent20["avg_consecutive_pairs"] - prior120["avg_consecutive_pairs"]
    variance_delta = recent20["sum_stddev"] - prior120["sum_stddev"]
    mid_high_delta = recent20["avg_mid_high_count"] - prior120["avg_mid_high_count"]

    scores = {
        "mid_high_trend": sum_delta * 0.8 + high_delta * 9.0 + mid_high_delta * 8.0,
        "low_balanced_trend": (-sum_delta) * 0.55 + low_delta * 8.0 - max(0.0, high_delta) * 5.0,
        "recent_repeat_trend": repeat_delta * 18.0 + recent20["avg_repeat_from_previous"] * 5.0,
        "consecutive_trend": consecutive_delta * 16.0 + recent20["avg_consecutive_pairs"] * 3.0,
        "high_variance_trend": variance_delta * 1.15 + abs(sum_delta) * 0.25,
        "contrarian_gap_trend": max(0.0, 1.0 - recent20["avg_repeat_from_previous"]) * 8.0 + max(0.0, variance_delta) * 0.45,
        "balanced": 8.0 - abs(sum_delta) * 0.08 - abs(high_delta) * 4.0 - abs(repeat_delta) * 6.0,
    }
    # 直近60も少し混ぜて、20回だけの過敏反応を抑える。
    if recent60["avg_sum"] > prior120["avg_sum"] + 4 and recent60["avg_mid_high_count"] > prior120["avg_mid_high_count"]:
        scores["mid_high_trend"] += 4.0
    if recent60["avg_repeat_from_previous"] > prior120["avg_repeat_from_previous"] + 0.15:
        scores["recent_repeat_trend"] += 3.5
    if recent60["sum_stddev"] > prior120["sum_stddev"] + 3:
        scores["high_variance_trend"] += 3.0
    return {k: round(v, 6) for k, v in scores.items()}


def normalize_counts(counts: Dict[str, int], purchase_count: int) -> Dict[str, int]:
    out = {role: max(0, int(counts.get(role, 0))) for role in DEFAULT_COUNTS}
    if sum(out.values()) == 0:
        out = dict(DEFAULT_COUNTS)
    while sum(out.values()) > purchase_count:
        removable = [r for r in out if out[r] > 0 and not (r in {"main_best", "high_match"} and out["main_best"] + out["high_match"] <= 1)]
        role = removable[-1] if removable else max(out, key=lambda r: out[r])
        out[role] -= 1
    priority = ["main_best", "high_match", "mid_high", "recent120", "contrarian"]
    while sum(out.values()) < purchase_count:
        role = priority[sum(out.values()) % len(priority)]
        out[role] += 1
    return out


def counts_from_base_strategy(path: str) -> Dict[str, int]:
    payload = read_json_if_exists(path)
    raw = payload.get("strategy_counts", {})
    if isinstance(raw, dict):
        counts = {role: int(raw.get(role, 0) or 0) for role in DEFAULT_COUNTS}
        if sum(counts.values()) > 0:
            return counts
    seq = payload.get("role_sequence", [])
    counts = {role: 0 for role in DEFAULT_COUNTS}
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, dict):
                role = str(item.get("role", ""))
                if role in counts:
                    counts[role] += 1
    return counts if sum(counts.values()) > 0 else dict(DEFAULT_COUNTS)


def blend_counts(base_counts: Dict[str, int], regime_counts: Dict[str, int], purchase_count: int, regime_strength: float) -> Dict[str, int]:
    # 強いレジームほどregime_counts寄り。弱いとbase strategyを尊重。
    blended: Dict[str, int] = {}
    for role in DEFAULT_COUNTS:
        raw = base_counts.get(role, 0) * (1.0 - regime_strength) + regime_counts.get(role, 0) * regime_strength
        blended[role] = int(round(raw))
    return normalize_counts(blended, purchase_count)


def role_sequence_from_counts(counts: Dict[str, int], regime_scores: Dict[str, float], regime: str, purchase_count: int) -> List[Dict[str, str]]:
    # レジームごとに口順を少し変える。
    if regime in {"mid_high_trend", "high_variance_trend"}:
        order = ["main_best", "mid_high", "high_match", "contrarian", "recent120"]
    elif regime in {"recent_repeat_trend", "consecutive_trend"}:
        order = ["main_best", "high_match", "recent120", "mid_high", "contrarian"]
    elif regime == "contrarian_gap_trend":
        order = ["main_best", "contrarian", "mid_high", "high_match", "recent120"]
    else:
        order = ["main_best", "high_match", "recent120", "mid_high", "contrarian"]

    sequence: List[Dict[str, str]] = []
    for role in order:
        for _ in range(max(0, counts.get(role, 0))):
            sequence.append({"role": role, "label": ROLE_LABELS[role]})
            if len(sequence) >= purchase_count:
                return sequence
    for role in order:
        sequence.append({"role": role, "label": ROLE_LABELS[role]})
        if len(sequence) >= purchase_count:
            break
    return sequence[:purchase_count]


def write_report(path: str, state: Dict[str, object], strategy: Dict[str, object]) -> None:
    lines = [
        "LOTO7 Regime-Based Role Strategy Report",
        "========================================",
        "",
        f"created_at: {state.get('created_at')}",
        f"latest_draw_no: {state.get('latest_draw_no')}",
        f"latest_draw_date: {state.get('latest_draw_date')}",
        f"regime: {state.get('regime')}",
        f"regime_strength: {state.get('regime_strength')}",
        "",
        "[Regime Scores]",
        json.dumps(state.get("regime_scores", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Recent 20 Metrics]",
        json.dumps(state.get("recent20_metrics", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Prior 120 Metrics]",
        json.dumps(state.get("prior120_metrics", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Strategy Counts]",
        json.dumps(strategy.get("strategy_counts", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "[Role Sequence]",
    ]
    for i, item in enumerate(strategy.get("role_sequence", []), start=1):
        if isinstance(item, dict):
            lines.append(f"{i}: {item.get('role')} / {item.get('label')}")
    lines.append("")
    lines.append("注意: レジーム判定は過去傾向の分類であり、将来の当せんや利益を保証しません。")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect current LOTO7 regime and generate regime-based role strategy.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--base-strategy", default="outputs/role_ensemble/role_strategy.json")
    parser.add_argument("--output", default="outputs/role_ensemble/regime_state.json")
    parser.add_argument("--strategy-output", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--report", default="outputs/role_ensemble/regime_strategy_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--recent-window", type=int, default=20)
    parser.add_argument("--medium-window", type=int, default=60)
    parser.add_argument("--baseline-window", type=int, default=120)
    args = parser.parse_args()

    draws = load_draws(args.csv)
    if len(draws) < max(args.recent_window + 2, 10):
        raise SystemExit("not enough draws for regime detection")

    recent20_draws = draws[-args.recent_window :]
    recent60_draws = draws[-args.medium_window :]
    prior_start = max(0, len(draws) - args.medium_window - args.baseline_window)
    prior_end = max(0, len(draws) - args.medium_window)
    prior_draws = draws[prior_start:prior_end] or draws[: -args.recent_window]

    recent20 = window_metrics(recent20_draws)
    recent60 = window_metrics(recent60_draws)
    prior120 = window_metrics(prior_draws)
    scores = score_regimes(recent20, recent60, prior120)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    regime = ranked[0][0]
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) >= 2 else 0.0
    strength = max(0.35, min(0.85, 0.45 + (top - second) / 25.0))
    if regime == "balanced":
        strength = 0.35

    base_counts = normalize_counts(counts_from_base_strategy(args.base_strategy), args.purchase_count)
    regime_counts = normalize_counts(REGIME_ADJUSTMENTS.get(regime, DEFAULT_COUNTS), args.purchase_count)
    final_counts = blend_counts(base_counts, regime_counts, args.purchase_count, strength)
    sequence = role_sequence_from_counts(final_counts, scores, regime, args.purchase_count)

    state = {
        "created_at": now_iso(),
        "kind": "loto7_regime_state",
        "csv": args.csv,
        "latest_draw_no": draws[-1].draw_no,
        "latest_draw_date": draws[-1].date,
        "latest_main": list(draws[-1].main),
        "latest_bonus": list(draws[-1].bonus),
        "regime": regime,
        "regime_strength": round(strength, 6),
        "regime_scores": scores,
        "recent20_metrics": recent20,
        "recent60_metrics": recent60,
        "prior120_metrics": prior120,
        "base_strategy": args.base_strategy,
        "base_counts": base_counts,
        "regime_counts": regime_counts,
        "final_counts": final_counts,
    }
    strategy = {
        "created_at": now_iso(),
        "kind": "loto7_regime_role_strategy",
        "source": "detect_loto7_regime.py",
        "source_regime_state": args.output,
        "source_base_strategy": args.base_strategy,
        "regime": regime,
        "regime_strength": round(strength, 6),
        "purchase_count": args.purchase_count,
        "strategy_counts": final_counts,
        "role_sequence": sequence,
        "scores": scores,
        "notes": [
            "Regime-based strategy adjusts Role Strategy Optimizer output using recent draw state.",
            "Fallback/default role strategy is used when the base strategy is missing.",
            "This does not guarantee lottery winnings or profit.",
        ],
    }
    write_json(args.output, state)
    write_json(args.strategy_output, strategy)
    write_report(args.report, state, strategy)
    print(json.dumps({"state": state, "strategy": strategy}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
