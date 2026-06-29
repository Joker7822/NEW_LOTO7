#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_evolution_shards.py

複数shardで出力された候補モデル、または採用済み best model を統合し、
holdout成績で再ランキングして最終採用モデルと最新予測を出力する。

最新予測は以下をサポートする。
  - best_model: 採用モデル単体の上位候補
  - ensemble: 複数モデル合議制
  - role_ensemble: 5口を役割分担する構造型予測

role_ensemble は outputs/role_ensemble/role_strategy.json があれば、
Role Strategy Optimizer が決めた役割配分を使用する。

注意:
  宝くじ抽せんはランダム性が高く、当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loto7_evolution_trainer import Draw, Genome, evaluate_ticket, generate_tickets, genome_from_dict, load_draws

RANK_ORDER = ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]
PRIZE_RANKS = ["1等", "2等", "3等", "4等", "5等", "6等"]
CURRENT_8_SHARD_PATTERNS = ["loto7_best_model_shard*_of_08.json", "outputs/loto7_best_model_shard*_of_08.json"]
ROLE_ORDER = [
    ("main_best", "本命: 採用ベストモデル"),
    ("high_match", "高一致狙い: ペア/3連/最大一致重視"),
    ("recent120", "直近寄り: 直近120回/60回の流れ重視"),
    ("mid_high", "中高数字補正: 20番台後半〜30番台も押さえる"),
    ("contrarian", "荒れ目/逆張り: 休眠・広めレンジ・低重複"),
]
ROLE_LABELS = {role: label for role, label in ROLE_ORDER}


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def normalize_weights(values: Sequence[float]) -> Tuple[float, float, float, float]:
    clipped = [max(0.01, float(v)) for v in values]
    total = sum(clipped) or 1.0
    return tuple(v / total for v in clipped)  # type: ignore[return-value]


def load_model(path: Path) -> Optional[Dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        genome_data = data.get("genome", data)
        genome = genome_from_dict(genome_data)
        return {"path": str(path), "payload": data, "genome": genome}
    except Exception as exc:
        print(f"[WARN] skip invalid model: {path} ({exc})")
        return None


def find_models(patterns: List[str]) -> List[Dict[str, object]]:
    found: List[Dict[str, object]] = []
    for pattern in patterns:
        for name in sorted(glob.glob(pattern)):
            item = load_model(Path(name))
            if item is not None:
                found.append(item)

    unique: Dict[str, Dict[str, object]] = {}
    for item in found:
        unique[str(item["path"])] = item
    return list(unique.values())


def fmt_ticket(ticket: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in ticket)


def ticket_key(ticket: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(n) for n in ticket))


def parse_money_yen(text: object) -> int:
    raw = str(text or "").strip()
    if not raw or raw == "該当なし":
        return 0
    m = re.search(r"([0-9,]+)", raw)
    if not m:
        return 0
    return int(m.group(1).replace(",", ""))


def draw_no_int(text: object) -> Optional[int]:
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_prize_rows(csv_path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = draw_no_int(row.get("回別"))
            if no is not None:
                out[no] = {k: str(v or "").strip() for k, v in row.items()}
    return out


def has_any_prize_amount(row: Dict[str, str]) -> bool:
    return any(str(row.get(f"{rank}当選金額", "")).strip() for rank in PRIZE_RANKS)


def prize_amount_for_rank(row: Dict[str, str], rank: str) -> int:
    if rank == "外れ":
        return 0
    return parse_money_yen(row.get(f"{rank}当選金額", ""))


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_yen(value: object) -> str:
    try:
        return f"{int(value):,}円"
    except Exception:
        return f"{value}円"


def select_target_indices(draws: Sequence[Draw], *, min_train_draws: int, holdout_start_draw: int, holdout_end_draw: Optional[int]) -> List[int]:
    out: List[int] = []
    for idx, draw in enumerate(draws):
        if idx < min_train_draws:
            continue
        if draw.draw_no < holdout_start_draw:
            continue
        if holdout_end_draw is not None and draw.draw_no > holdout_end_draw:
            continue
        out.append(idx)
    return out


def evaluate_model_on_holdout(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[Draw],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Dict[str, object]:
    rank_counts = {rank: 0 for rank in RANK_ORDER}
    max_main_match = 0
    max_bonus_match = 0
    total_cost = 0
    total_payout = 0
    total_tickets = 0
    missing_prize_draws: List[int] = []

    for idx in target_indices:
        target = draws[idx]
        train = draws[:idx]
        tickets = generate_tickets(train, genome, purchase_count)
        prize_row = prize_rows.get(target.draw_no, {})
        if not prize_row or not has_any_prize_amount(prize_row):
            missing_prize_draws.append(target.draw_no)

        for ticket in tickets:
            main_match, bonus_match, rank = evaluate_ticket(ticket, target)
            payout = prize_amount_for_rank(prize_row, rank)
            total_cost += unit_cost
            total_payout += payout
            total_tickets += 1
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            max_main_match = max(max_main_match, main_match)
            max_bonus_match = max(max_bonus_match, bonus_match)

    profit = total_payout - total_cost
    roi = (total_payout / total_cost) if total_cost else 0.0
    grade_hit_count = sum(rank_counts.get(rank, 0) for rank in PRIZE_RANKS)
    high_grade_hit_count = sum(rank_counts.get(rank, 0) for rank in ["1等", "2等", "3等", "4等"])

    return {
        "path": model_path,
        "genome_id": genome.id,
        "evolution_score": genome.score,
        "target_draws": len(target_indices),
        "purchase_count": purchase_count,
        "unit_cost": unit_cost,
        "total_tickets": total_tickets,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "profit": profit,
        "roi": round(roi, 6),
        "roi_percent": round(roi * 100.0, 3),
        "max_main_match": max_main_match,
        "max_bonus_match": max_bonus_match,
        "rank_counts": rank_counts,
        "grade_hit_count": grade_hit_count,
        "high_grade_hit_count": high_grade_hit_count,
        "missing_prize_draw_count": len(set(missing_prize_draws)),
        "missing_prize_draws": sorted(set(missing_prize_draws)),
    }


def _metrics(item: Dict[str, object]) -> Dict[str, object]:
    metrics = item.get("holdout", {})
    return metrics if isinstance(metrics, dict) else {}


def _rank_counts(metrics: Dict[str, object]) -> Dict[str, object]:
    rank_counts = metrics.get("rank_counts", {})
    return rank_counts if isinstance(rank_counts, dict) else {}


def holdout_roi_sort_key(item: Dict[str, object]) -> tuple:
    metrics = _metrics(item)
    rank_counts = _rank_counts(metrics)
    genome = item.get("genome")
    evo_score = getattr(genome, "score", 0.0)
    return (
        float(metrics.get("roi", 0.0)),
        int(metrics.get("max_main_match", 0)),
        int(rank_counts.get("1等", 0)),
        int(rank_counts.get("2等", 0)),
        int(rank_counts.get("3等", 0)),
        int(rank_counts.get("4等", 0)),
        int(rank_counts.get("5等", 0)),
        int(rank_counts.get("6等", 0)),
        float(evo_score),
    )


def holdout_balanced_sort_key(item: Dict[str, object]) -> tuple:
    metrics = _metrics(item)
    rank_counts = _rank_counts(metrics)
    genome = item.get("genome")
    evo_score = getattr(genome, "score", 0.0)
    return (
        int(metrics.get("max_main_match", 0)),
        int(metrics.get("high_grade_hit_count", 0)),
        int(rank_counts.get("1等", 0)),
        int(rank_counts.get("2等", 0)),
        int(rank_counts.get("3等", 0)),
        int(rank_counts.get("4等", 0)),
        float(metrics.get("roi", 0.0)),
        int(rank_counts.get("5等", 0)),
        int(rank_counts.get("6等", 0)),
        float(evo_score),
    )


def selection_sort_description(selection_mode: str) -> str:
    if selection_mode in {"holdout", "holdout_roi"}:
        return "ROI → 最大本数字一致数 → 1等→2等→3等→4等→5等→6等 → Evolutionスコア"
    if selection_mode == "holdout_balanced":
        return "最大本数字一致数 → 4等以上件数 → 1等→2等→3等→4等 → ROI → 5等→6等 → Evolutionスコア"
    return "Evolutionスコア"


def rerank_models_by_holdout(
    models: List[Dict[str, object]],
    *,
    draws: Sequence[Draw],
    prize_rows: Dict[int, Dict[str, str]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
    selection_mode: str,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for item in models:
        genome: Genome = item["genome"]  # type: ignore[assignment]
        model_path = str(item["path"])
        metrics = evaluate_model_on_holdout(
            genome=genome,
            model_path=model_path,
            draws=draws,
            prize_rows=prize_rows,
            target_indices=target_indices,
            purchase_count=purchase_count,
            unit_cost=unit_cost,
        )
        copied = dict(item)
        copied["holdout"] = metrics
        enriched.append(copied)

    if selection_mode == "holdout_balanced":
        enriched.sort(key=holdout_balanced_sort_key, reverse=True)
    else:
        enriched.sort(key=holdout_roi_sort_key, reverse=True)
    return enriched


def write_model_selection_report(report_path: str, ranked_models: Sequence[Dict[str, object]], summary: Dict[str, object]) -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("LOTO7 Model Selection Report")
    lines.append("=" * 30)
    lines.append("")
    lines.append(f"作成日時(UTC): {summary.get('updated_at')}")
    lines.append(f"選定方式: {summary.get('selection_mode')}")
    lines.append(f"holdout開始回: {summary.get('selection_holdout_start_draw')}")
    lines.append(f"holdout終了回: {summary.get('selection_holdout_end_draw')}")
    lines.append(f"検証対象回数: {summary.get('selection_target_draws')}")
    lines.append(f"候補モデル数: {summary.get('model_count')}")
    lines.append("")
    lines.append("[並び替え基準]")
    lines.append(str(summary.get("selection_sort_description")))
    lines.append("")
    lines.append("[最終採用モデル]")
    lines.append(f"採用モデル: {summary.get('selected_model')}")
    lines.append(f"モデルID: {summary.get('selected_genome_id')}")
    lines.append(f"Evolutionスコア: {summary.get('selected_score')}")
    lines.append(f"採用理由: {summary.get('selection_reason')}")
    lines.append("")
    lines.append("[候補モデルランキング]")
    for rank, item in enumerate(ranked_models, start=1):
        metrics = _metrics(item)
        rank_counts = _rank_counts(metrics)
        lines.append(
            f"{rank}位: {item.get('path')} / "
            f"ROI={metrics.get('roi_percent')}% / "
            f"収支={format_yen(metrics.get('profit'))} / "
            f"最大一致={metrics.get('max_main_match')} / "
            f"4等以上={metrics.get('high_grade_hit_count')} / "
            f"1等={rank_counts.get('1等', 0)}, 2等={rank_counts.get('2等', 0)}, 3等={rank_counts.get('3等', 0)}, "
            f"4等={rank_counts.get('4等', 0)}, 5等={rank_counts.get('5等', 0)}, 6等={rank_counts.get('6等', 0)} / "
            f"EvolutionScore={metrics.get('evolution_score')}"
        )
    lines.append("")
    lines.append("注意: holdout再ランキングは過去検証上の優劣であり、将来の当せんや利益を保証するものではありません。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_weight_for_ensemble(item: Dict[str, object], rank_index: int) -> float:
    metrics = _metrics(item)
    rank_counts = _rank_counts(metrics)
    base = 1.0 / float(rank_index + 1)
    roi = max(0.0, float(metrics.get("roi", 0.0)))
    max_match = float(metrics.get("max_main_match", 0.0))
    high_grade = float(metrics.get("high_grade_hit_count", 0.0))
    fifth = float(rank_counts.get("5等", 0))
    sixth = float(rank_counts.get("6等", 0))
    return base + roi * 2.0 + max_match * 0.15 + high_grade * 0.08 + fifth * 0.004 + sixth * 0.002


def load_role_strategy(path: str, purchase_count: int) -> List[Tuple[str, str]]:
    fallback = ROLE_ORDER[: max(1, purchase_count)]
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return fallback
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] cannot load role strategy {path}: {exc}")
        return fallback
    raw_sequence = payload.get("role_sequence", [])
    out: List[Tuple[str, str]] = []
    if isinstance(raw_sequence, list):
        for item in raw_sequence:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            if role not in ROLE_LABELS:
                continue
            label = str(item.get("label") or ROLE_LABELS.get(role, role))
            out.append((role, label))
            if len(out) >= purchase_count:
                break
    if not out:
        counts = payload.get("strategy_counts", {})
        if isinstance(counts, dict):
            for role, _label in ROLE_ORDER:
                count = int(counts.get(role, 0) or 0)
                for _ in range(max(0, count)):
                    out.append((role, ROLE_LABELS[role]))
                    if len(out) >= purchase_count:
                        break
                if len(out) >= purchase_count:
                    break
    while len(out) < purchase_count:
        role, label = fallback[len(out) % len(fallback)]
        out.append((role, label))
    return out[:purchase_count]


def clone_genome_for_role(base: Genome, role: str) -> Genome:
    data = dict(base.__dict__)
    data["id"] = f"{base.id}_{role}"
    data["score"] = 0.0
    data["max_main_match"] = 0
    data["best_rank_count"] = 0

    if role == "main_best":
        return Genome(**data)

    if role == "high_match":
        w = normalize_weights([
            data["full_weight"] * 1.15,
            data["recent240_weight"] * 1.05,
            data["recent120_weight"] * 0.95,
            data["recent60_weight"] * 0.90,
        ])
        data.update({"full_weight": w[0], "recent240_weight": w[1], "recent120_weight": w[2], "recent60_weight": w[3]})
        data["pair_weight"] = clamp(data["pair_weight"] * 1.45 + 0.03, 0.02, 0.38)
        data["pair_recency_weight"] = clamp(data["pair_recency_weight"] * 1.25, 0.02, 0.42)
        data["pair_stability_weight"] = clamp(data["pair_stability_weight"] * 1.40 + 0.03, 0.02, 0.38)
        data["triple_weight"] = clamp(data["triple_weight"] * 2.20 + 0.02, 0.00, 0.22)
        data["pool_size"] = max(15, min(20, int(data["pool_size"])))
        data["overlap_limit"] = min(4, int(data["overlap_limit"]))

    elif role == "recent120":
        w = normalize_weights([
            data["full_weight"] * 0.45,
            data["recent240_weight"] * 0.80,
            data["recent120_weight"] * 1.65,
            data["recent60_weight"] * 1.80,
        ])
        data.update({"full_weight": w[0], "recent240_weight": w[1], "recent120_weight": w[2], "recent60_weight": w[3]})
        data["pair_recency_weight"] = clamp(data["pair_recency_weight"] * 1.60 + 0.04, 0.02, 0.45)
        data["dormancy_weight"] = clamp(data["dormancy_weight"] * 0.70, 0.00, 0.08)
        data["pool_size"] = max(15, min(21, int(data["pool_size"]) + 1))

    elif role == "mid_high":
        w = normalize_weights([
            data["full_weight"] * 0.80,
            data["recent240_weight"] * 1.05,
            data["recent120_weight"] * 1.10,
            data["recent60_weight"] * 0.95,
        ])
        data.update({"full_weight": w[0], "recent240_weight": w[1], "recent120_weight": w[2], "recent60_weight": w[3]})
        data["target_sum_min"] = max(118, int(data["target_sum_min"]))
        data["target_sum_max"] = max(172, int(data["target_sum_max"]))
        data["sum_bonus"] = clamp(data["sum_bonus"] * 1.25 + 0.05, 0.10, 1.10)
        data["low_high_bonus"] = clamp(data["low_high_bonus"] * 0.80, 0.00, 0.70)
        data["pool_size"] = max(17, min(24, int(data["pool_size"]) + 3))

    elif role == "contrarian":
        w = normalize_weights([
            data["full_weight"] * 1.10,
            data["recent240_weight"] * 0.85,
            data["recent120_weight"] * 0.70,
            data["recent60_weight"] * 0.55,
        ])
        data.update({"full_weight": w[0], "recent240_weight": w[1], "recent120_weight": w[2], "recent60_weight": w[3]})
        data["dormancy_weight"] = clamp(data["dormancy_weight"] * 1.80 + 0.02, 0.01, 0.12)
        data["pair_weight"] = clamp(data["pair_weight"] * 0.65, 0.00, 0.25)
        data["pair_recency_weight"] = clamp(data["pair_recency_weight"] * 0.60, 0.00, 0.30)
        data["triple_weight"] = clamp(data["triple_weight"] * 0.55, 0.00, 0.12)
        data["target_sum_min"] = min(95, int(data["target_sum_min"]))
        data["target_sum_max"] = max(190, int(data["target_sum_max"]))
        data["pool_size"] = max(20, min(27, int(data["pool_size"]) + 5))
        data["overlap_limit"] = min(4, int(data["overlap_limit"]))
        data["max_consecutive_pairs"] = max(0, min(2, int(data["max_consecutive_pairs"])))

    if int(data["target_sum_min"]) >= int(data["target_sum_max"]):
        data["target_sum_min"] = 100
        data["target_sum_max"] = 180
    return Genome(**data)


def role_ticket_score(ticket: Sequence[int], role: str, latest_draw: Draw) -> float:
    nums = tuple(sorted(ticket))
    total = sum(nums)
    odd = sum(1 for n in nums if n % 2)
    low = sum(1 for n in nums if n <= 18)
    mid_high = sum(1 for n in nums if 21 <= n <= 36)
    high = sum(1 for n in nums if n >= 25)
    latest_overlap = len(set(nums) & set(latest_draw.main))
    consecutive_pairs = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)

    score = 0.0
    if 3 <= odd <= 4:
        score += 4.0
    if 3 <= low <= 4:
        score += 3.0
    score -= abs(total - 140) * 0.03

    if role == "main_best":
        score += 20.0
    elif role == "high_match":
        score += latest_overlap * 1.6
        score += 2.0 if 110 <= total <= 170 else -2.0
        score -= max(0, consecutive_pairs - 2) * 1.5
    elif role == "recent120":
        score += latest_overlap * 2.0
        score += 2.5 if 105 <= total <= 165 else -1.5
        score += 1.0 if 2 <= consecutive_pairs <= 3 else 0.0
    elif role == "mid_high":
        score += mid_high * 1.2
        score += high * 0.8
        score += 4.0 if 125 <= total <= 185 else -3.0
        score -= max(0, low - 4) * 1.5
    elif role == "contrarian":
        score += max(0, 3 - latest_overlap) * 1.8
        score += 2.0 if high >= 2 else 0.0
        score += 2.0 if 95 <= total <= 195 else -1.0
        score -= latest_overlap * 0.8
    return round(score, 6)


def make_prediction_rows(best: Genome, source_model: str, draws, purchase_count: int) -> List[Dict[str, object]]:
    latest = draws[-1]
    tickets = generate_tickets(draws, best, purchase_count)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: List[Dict[str, object]] = []
    for idx, ticket in enumerate(tickets, start=1):
        rows.append(
            {
                "confidence_rank": idx,
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": fmt_ticket(ticket),
                "model_id": best.id,
                "model_score": round(best.score, 6),
                "source_model": source_model,
                "prediction_method": "best_model",
                "ensemble_score": "",
                "support_models": "",
                "created_at": created_at,
            }
        )
    return rows


def make_role_ensemble_prediction_rows(
    best: Genome,
    source_model: str,
    draws: Sequence[Draw],
    purchase_count: int,
    overlap_limit: int,
    role_sequence: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Dict[str, object]]:
    latest = draws[-1]
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    selected: List[Tuple[str, str, Tuple[int, ...], float]] = []
    used = set()
    roles = list(role_sequence) if role_sequence else ROLE_ORDER[: max(1, purchase_count)]
    while len(roles) < purchase_count:
        roles.append(ROLE_ORDER[len(roles) % len(ROLE_ORDER)])
    roles = roles[:purchase_count]
    candidate_count = max(40, purchase_count * 12)

    for role, label in roles:
        genome = clone_genome_for_role(best, role)
        try:
            raw_tickets = generate_tickets(draws, genome, candidate_count)
        except Exception as exc:
            print(f"[WARN] role_ensemble fallback role={role}: {exc}")
            raw_tickets = generate_tickets(draws, best, candidate_count)
        ranked = sorted(
            [(ticket_key(ticket), role_ticket_score(ticket, role, latest)) for ticket in raw_tickets],
            key=lambda kv: kv[1],
            reverse=True,
        )
        chosen: Optional[Tuple[Tuple[int, ...], float]] = None
        for key, score in ranked:
            if key in used:
                continue
            if all(len(set(key) & set(prev)) <= overlap_limit for _role, _label, prev, _score in selected):
                chosen = (key, score)
                break
        if chosen is None:
            for key, score in ranked:
                if key not in used:
                    chosen = (key, score)
                    break
        if chosen is not None:
            used.add(chosen[0])
            selected.append((role, label, chosen[0], chosen[1]))

    if len(selected) < purchase_count:
        fallback_tickets = generate_tickets(draws, best, max(purchase_count * 4, 20))
        for ticket in fallback_tickets:
            key = ticket_key(ticket)
            if key in used:
                continue
            selected.append(("fallback", "補完: 採用ベストモデル", key, role_ticket_score(key, "main_best", latest)))
            used.add(key)
            if len(selected) >= purchase_count:
                break

    rows: List[Dict[str, object]] = []
    for idx, (role, label, ticket, score) in enumerate(selected[:purchase_count], start=1):
        rows.append(
            {
                "confidence_rank": idx,
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": fmt_ticket(ticket),
                "model_id": f"{best.id}:{role}",
                "model_score": round(best.score, 6),
                "source_model": source_model,
                "prediction_method": "role_ensemble",
                "ensemble_score": round(score, 6),
                "support_models": label,
                "created_at": created_at,
            }
        )
    return rows


def make_ensemble_prediction_rows(
    ranked_models: Sequence[Dict[str, object]],
    *,
    draws: Sequence[Draw],
    purchase_count: int,
    candidates_per_model: int,
    overlap_limit: int,
    fallback_best: Genome,
    fallback_source: str,
) -> List[Dict[str, object]]:
    latest = draws[-1]
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    candidate_scores: Dict[Tuple[int, ...], float] = {}
    support: Dict[Tuple[int, ...], List[str]] = {}
    number_votes = {n: 0.0 for n in range(1, 38)}

    candidate_count = max(purchase_count, candidates_per_model)
    for model_rank, item in enumerate(ranked_models):
        genome: Genome = item["genome"]  # type: ignore[assignment]
        source = str(item.get("path"))
        weight = model_weight_for_ensemble(item, model_rank)
        try:
            tickets = generate_tickets(draws, genome, candidate_count)
        except Exception as exc:
            print(f"[WARN] ensemble skip model={source}: {exc}")
            continue
        for ticket_rank, ticket in enumerate(tickets, start=1):
            key = ticket_key(ticket)
            rank_weight = 1.0 / float(ticket_rank)
            score = weight * rank_weight
            candidate_scores[key] = candidate_scores.get(key, 0.0) + score
            support.setdefault(key, []).append(f"{source}#{ticket_rank}")
            for n in key:
                number_votes[n] += weight * rank_weight

    for key in list(candidate_scores.keys()):
        consensus_bonus = sum(number_votes.get(n, 0.0) for n in key) * 0.012
        diversity_bonus = len(set(support.get(key, []))) * 0.03
        candidate_scores[key] += consensus_bonus + diversity_bonus

    ranked_tickets = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[Tuple[Tuple[int, ...], float]] = []
    for key, score in ranked_tickets:
        if all(len(set(key) & set(prev_key)) <= overlap_limit for prev_key, _ in selected):
            selected.append((key, score))
        if len(selected) >= purchase_count:
            break
    if len(selected) < purchase_count:
        for key, score in ranked_tickets:
            if key not in [prev_key for prev_key, _ in selected]:
                selected.append((key, score))
            if len(selected) >= purchase_count:
                break

    if not selected:
        return make_prediction_rows(fallback_best, fallback_source, draws, purchase_count)

    rows: List[Dict[str, object]] = []
    for idx, (ticket, score) in enumerate(selected[:purchase_count], start=1):
        rows.append(
            {
                "confidence_rank": idx,
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": fmt_ticket(ticket),
                "model_id": "ensemble_v1",
                "model_score": round(score, 6),
                "source_model": "ensemble_of_ranked_shards",
                "prediction_method": "ensemble",
                "ensemble_score": round(score, 6),
                "support_models": ";".join(support.get(ticket, [])[:8]),
                "created_at": created_at,
            }
        )
    return rows


def write_prediction(csv_path: str, rows: List[Dict[str, object]]) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "confidence_rank", "base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index",
                "numbers", "model_id", "model_score", "source_model", "prediction_method", "ensemble_score", "support_models", "created_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_report(report_path: str, rows: List[Dict[str, object]], best: Genome, source_model: str, model_count: int, min_models: int, selection_reason: str, prediction_mode: str, role_strategy_path: str = "") -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0] if rows else {}

    lines: List[str] = []
    lines.append("LOTO7 Latest Prediction Report")
    lines.append("=" * 31)
    lines.append("")
    lines.append(f"作成日時(UTC): {first.get('created_at')}")
    lines.append(f"基準最新回: 第{first.get('base_latest_draw_no')}回")
    lines.append(f"基準最新抽せん日: {first.get('base_latest_date')}")
    lines.append(f"予測対象回: 第{first.get('prediction_draw_no')}回")
    lines.append("")
    lines.append("[採用モデル]")
    lines.append(f"モデルID: {best.id}")
    lines.append(f"モデルスコア: {round(best.score, 6)}")
    lines.append(f"採用元shardモデル: {source_model}")
    lines.append(f"統合対象モデル数: {model_count}")
    lines.append(f"必要最小モデル数: {min_models}")
    lines.append(f"採用基準: {selection_reason}")
    lines.append(f"予測方式: {prediction_mode}")
    if prediction_mode == "role_ensemble":
        lines.append(f"役割戦略: {role_strategy_path or 'default_role_order'}")
    lines.append("")
    lines.append("[最新予測 5口: 信頼度の高い順]")
    for row in rows:
        extra = ""
        if row.get("prediction_method") in {"ensemble", "role_ensemble"}:
            extra = f" / score={row.get('ensemble_score')}"
        role = f" / {row.get('support_models')}" if row.get("prediction_method") == "role_ensemble" else ""
        lines.append(f"{int(row['confidence_rank'])}位 / {int(row['combo_index'])}口目: {row['numbers']}{extra}{role}")
    lines.append("")
    lines.append("[読み方]")
    if prediction_mode == "ensemble":
        lines.append("8 shard候補モデルの合議制で、複数モデルが推す組み合わせとholdout実績を再スコアリングしています。")
        lines.append("1位が合議制スコアで最も高い組み合わせです。")
    elif prediction_mode == "role_ensemble":
        lines.append("5口を同一モデルの単純上位ではなく、役割別に分けています。")
        lines.append("role_strategy.json が存在する場合は、Role Strategy Optimizer の配分を使用します。")
    else:
        lines.append("1位がこのモデル内で最も信頼度が高い組み合わせです。")
        lines.append("信頼度順位は、採用Genomeが生成した候補のスコア順を保持しています。")
    lines.append("")
    lines.append("[出力ファイル]")
    lines.append("CSV: outputs/evolution_best_prediction.csv")
    lines.append(f"TXT: {report_path}")
    lines.append("")
    lines.append("注意: 宝くじはランダム性が高く、この予測は当せんや利益を保証するものではありません。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge and rerank LOTO7 evolution shard best models.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--patterns", nargs="*", default=CURRENT_8_SHARD_PATTERNS)
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--summary", default="outputs/evolution_merged_summary.json")
    parser.add_argument("--manifest", default="outputs/run_manifest.json")
    parser.add_argument("--model-selection-summary", default="outputs/holdout/model_selection_summary.json")
    parser.add_argument("--model-selection-report", default="outputs/holdout/model_selection_report.txt")
    parser.add_argument("--role-strategy", default="outputs/role_ensemble/role_strategy.json")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-models", type=int, default=1, help="統合に必要な最小モデル数。8を指定すると現行8 shardの欠損を検出できる。")
    parser.add_argument("--selection-mode", choices=["holdout", "holdout_roi", "holdout_balanced", "evolution_score"], default="holdout_balanced")
    parser.add_argument("--prediction-mode", choices=["ensemble", "best_model", "role_ensemble"], default="role_ensemble")
    parser.add_argument("--ensemble-candidates-per-model", type=int, default=10)
    parser.add_argument("--ensemble-overlap-limit", type=int, default=4)
    parser.add_argument("--selection-holdout-start-draw", type=int, default=641)
    parser.add_argument("--selection-holdout-end-draw", type=int, default=None)
    parser.add_argument("--selection-min-train-draws", type=int, default=60)
    parser.add_argument("--selection-unit-cost", type=int, default=300)
    args = parser.parse_args(argv)

    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.min_models <= 0:
        raise SystemExit("--min-models must be positive")
    if args.selection_unit_cost <= 0:
        raise SystemExit("--selection-unit-cost must be positive")
    if args.ensemble_candidates_per_model < args.purchase_count:
        raise SystemExit("--ensemble-candidates-per-model must be >= --purchase-count")
    if args.ensemble_overlap_limit < 0 or args.ensemble_overlap_limit > 7:
        raise SystemExit("--ensemble-overlap-limit must be between 0 and 7")
    if args.selection_holdout_end_draw is not None and args.selection_holdout_end_draw < args.selection_holdout_start_draw:
        raise SystemExit("--selection-holdout-end-draw must be >= --selection-holdout-start-draw")

    models = find_models(args.patterns)
    if not models:
        raise SystemExit(f"no shard best models found: {args.patterns}")
    if len(models) < args.min_models:
        raise SystemExit(f"not enough shard models: found={len(models)} required={args.min_models}")

    normalized_selection_mode = "holdout_roi" if args.selection_mode == "holdout" else args.selection_mode
    sort_description = selection_sort_description(normalized_selection_mode)

    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.selection_min_train_draws,
        holdout_start_draw=args.selection_holdout_start_draw,
        holdout_end_draw=args.selection_holdout_end_draw,
    )
    if normalized_selection_mode in {"holdout_roi", "holdout_balanced"} and not target_indices:
        raise SystemExit("no holdout targets selected for model selection")

    if normalized_selection_mode in {"holdout_roi", "holdout_balanced"}:
        ranked_models = rerank_models_by_holdout(
            models,
            draws=draws,
            prize_rows=prize_rows,
            target_indices=target_indices,
            purchase_count=args.purchase_count,
            unit_cost=args.selection_unit_cost,
            selection_mode=normalized_selection_mode,
        )
        selection_reason = f"holdout再ランキング: {sort_description}"
    else:
        ranked_models = sorted(models, key=lambda item: item["genome"].score, reverse=True)  # type: ignore[index, union-attr]
        for item in ranked_models:
            genome: Genome = item["genome"]  # type: ignore[assignment]
            item["holdout"] = {
                "path": str(item["path"]),
                "genome_id": genome.id,
                "evolution_score": genome.score,
                "target_draws": len(target_indices),
                "rank_counts": {rank: 0 for rank in RANK_ORDER},
                "roi": 0.0,
                "roi_percent": 0.0,
                "max_main_match": 0,
                "high_grade_hit_count": 0,
            }
        selection_reason = "Evolutionスコア最大"

    best_item = ranked_models[0]
    best: Genome = best_item["genome"]  # type: ignore[assignment]
    source_model = str(best_item["path"])
    selected_holdout = best_item.get("holdout", {}) if isinstance(best_item.get("holdout", {}), dict) else {}
    role_sequence = load_role_strategy(args.role_strategy, args.purchase_count) if args.prediction_mode == "role_ensemble" else []

    updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "updated_at": updated_at,
        "selection_mode": normalized_selection_mode,
        "selection_reason": selection_reason,
        "selection_sort_description": sort_description,
        "prediction_mode": args.prediction_mode,
        "role_strategy": args.role_strategy if args.prediction_mode == "role_ensemble" else "",
        "role_sequence": [{"role": role, "label": label} for role, label in role_sequence],
        "source_model": source_model,
        "selected_holdout": selected_holdout,
        "merged_from": [str(item["path"]) for item in ranked_models],
        "purchase_count": args.purchase_count,
        "genome": best.__dict__,
    }
    write_json(args.best_model, payload)

    if args.prediction_mode == "ensemble" and len(ranked_models) >= 2:
        prediction_rows = make_ensemble_prediction_rows(
            ranked_models,
            draws=draws,
            purchase_count=args.purchase_count,
            candidates_per_model=args.ensemble_candidates_per_model,
            overlap_limit=args.ensemble_overlap_limit,
            fallback_best=best,
            fallback_source=source_model,
        )
    elif args.prediction_mode == "role_ensemble":
        prediction_rows = make_role_ensemble_prediction_rows(
            best,
            source_model,
            draws,
            args.purchase_count,
            args.ensemble_overlap_limit,
            role_sequence=role_sequence,
        )
    else:
        prediction_rows = make_prediction_rows(best, source_model, draws, args.purchase_count)
    write_prediction(args.prediction, prediction_rows)
    write_prediction_report(
        args.prediction_report,
        prediction_rows,
        best,
        source_model,
        len(ranked_models),
        args.min_models,
        selection_reason,
        args.prediction_mode,
        args.role_strategy if args.prediction_mode == "role_ensemble" else "",
    )

    candidates = []
    for i, item in enumerate(ranked_models):
        genome: Genome = item["genome"]  # type: ignore[assignment]
        candidates.append(
            {
                "rank": i + 1,
                "path": str(item["path"]),
                "genome_id": genome.id,
                "score": genome.score,
                "ensemble_weight": round(model_weight_for_ensemble(item, i), 6),
                "holdout": item.get("holdout"),
            }
        )

    summary = {
        "updated_at": updated_at,
        "selection_mode": normalized_selection_mode,
        "selection_reason": selection_reason,
        "selection_sort_description": sort_description,
        "prediction_mode": args.prediction_mode,
        "role_strategy": args.role_strategy if args.prediction_mode == "role_ensemble" else "",
        "role_sequence": [{"role": role, "label": label} for role, label in role_sequence],
        "ensemble_candidates_per_model": args.ensemble_candidates_per_model,
        "ensemble_overlap_limit": args.ensemble_overlap_limit,
        "role_strategy_fallback": [label for _role, label in ROLE_ORDER[: args.purchase_count]] if args.prediction_mode == "role_ensemble" else [],
        "selection_holdout_start_draw": args.selection_holdout_start_draw,
        "selection_holdout_end_draw": args.selection_holdout_end_draw,
        "selection_min_train_draws": args.selection_min_train_draws,
        "selection_unit_cost": args.selection_unit_cost,
        "selection_target_draws": len(target_indices),
        "selected_model": source_model,
        "selected_genome_id": best.id,
        "selected_score": best.score,
        "selected_holdout": selected_holdout,
        "model_count": len(ranked_models),
        "min_models": args.min_models,
        "model_patterns": args.patterns,
        "csv": args.csv,
        "latest_draw_no": draws[-1].draw_no if draws else None,
        "latest_draw_date": draws[-1].date if draws else None,
        "best_model": args.best_model,
        "prediction": args.prediction,
        "prediction_report": args.prediction_report,
        "model_selection_summary": args.model_selection_summary,
        "model_selection_report": args.model_selection_report,
        "candidates": candidates,
    }
    write_json(args.summary, summary)
    write_json(args.model_selection_summary, summary)
    write_model_selection_report(args.model_selection_report, ranked_models, summary)

    manifest = {
        "created_at": updated_at,
        "kind": "loto7_evolution_merge_holdout_rerank_ensemble",
        "csv": args.csv,
        "latest_draw_no": draws[-1].draw_no if draws else None,
        "latest_draw_date": draws[-1].date if draws else None,
        "best_model": args.best_model,
        "prediction": args.prediction,
        "prediction_report": args.prediction_report,
        "summary": args.summary,
        "model_selection_summary": args.model_selection_summary,
        "model_selection_report": args.model_selection_report,
        "selection_mode": normalized_selection_mode,
        "selection_reason": selection_reason,
        "selection_sort_description": sort_description,
        "prediction_mode": args.prediction_mode,
        "role_strategy": args.role_strategy if args.prediction_mode == "role_ensemble" else "",
        "role_sequence": [{"role": role, "label": label} for role, label in role_sequence],
        "selected_model": source_model,
        "selected_genome_id": best.id,
        "selected_score": best.score,
        "selected_holdout": selected_holdout,
        "purchase_count": args.purchase_count,
        "model_count": len(ranked_models),
        "model_patterns": args.patterns,
    }
    write_json(args.manifest, manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
