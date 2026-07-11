#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_dual_model_prediction.py

Adaptive Multi-Model Prediction

5口を以下の構成で生成する。
  - 全期間型 best model
  - Recent Era専用モデル
  - Super Recent 2023+ 専用モデル
  - Regime Strategy補正枠

追加改善:
  - 5口間の重複削減
  - 実運用履歴との高重複/完全一致を減点
  - 数字プール多様性を加点
  - 1等狙いのため、合計値/奇偶/レンジ/分散のバランスを加点

注意:
  宝くじはランダム性が高く、当せんや利益を保証しない。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loto7_evolution_trainer import Genome, Draw, generate_tickets, load_draws  # noqa: E402
from merge_evolution_shards import (  # noqa: E402
    fmt_ticket,
    load_model,
    load_role_strategy,
    make_role_ensemble_prediction_rows,
    ticket_key,
    write_prediction,
    write_prediction_report,
)

Ticket = Tuple[int, ...]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_required_model(path: str) -> Dict[str, object]:
    item = load_model(Path(path))
    if item is None:
        raise SystemExit(f"cannot load model: {path}")
    return item


def load_optional_model(path: str, fallback: Dict[str, object]) -> Tuple[Dict[str, object], bool]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return fallback, True
    item = load_model(p)
    if item is None:
        return fallback, True
    return item, False


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
    seen: Set[Ticket] = set()
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


def max_overlap(ticket: Ticket, others: Sequence[Ticket]) -> int:
    if not others:
        return 0
    s = set(ticket)
    return max(len(s & set(o)) for o in others)


def avg_overlap(ticket: Ticket, others: Sequence[Ticket]) -> float:
    if not others:
        return 0.0
    s = set(ticket)
    return sum(len(s & set(o)) for o in others) / float(len(others))


def diversity_score(ticket: Ticket, latest: Draw) -> float:
    nums = tuple(sorted(ticket))
    total = sum(nums)
    odd = sum(1 for n in nums if n % 2)
    low = sum(1 for n in nums if n <= 18)
    high = sum(1 for n in nums if n >= 25)
    bins = [0, 0, 0, 0]
    for n in nums:
        if n <= 9:
            bins[0] += 1
        elif n <= 19:
            bins[1] += 1
        elif n <= 29:
            bins[2] += 1
        else:
            bins[3] += 1
    span = nums[-1] - nums[0]
    consecutive_pairs = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    latest_overlap = len(set(nums) & set(latest.main))

    score = 0.0
    # 数字プール多様性: 4つの帯域に分散しているほど良い。
    score += sum(1.2 for b in bins if b > 0)
    score -= max(0, max(bins) - 3) * 1.1
    score += 1.4 if 2 <= low <= 5 else -1.0
    score += 1.2 if 2 <= high <= 4 else -0.8
    score += 1.0 if 24 <= span <= 36 else -0.8

    # 1等狙い強化: 過去に多いバランス型レンジを重視しつつ、固まりすぎを避ける。
    score += 2.4 if 105 <= total <= 175 else -abs(total - 140) * 0.035
    score += 2.0 if 3 <= odd <= 4 else -1.2
    score -= max(0, consecutive_pairs - 2) * 0.9
    score += 1.0 if 1 <= latest_overlap <= 3 else -1.0
    return score


def ticket_score(
    ticket: Ticket,
    *,
    raw_rank: int,
    used: Sequence[Ticket],
    history: Sequence[Ticket],
    latest: Draw,
    strict_overlap_limit: int,
    history_overlap_limit: int,
    diversity_weight: float,
    first_prize_weight: float,
) -> float:
    score = 100.0 - raw_rank * 0.10
    used_max = max_overlap(ticket, used)
    hist_max = max_overlap(ticket, history)
    hist_avg = avg_overlap(ticket, history)

    score += diversity_score(ticket, latest) * diversity_weight
    score += diversity_score(ticket, latest) * first_prize_weight * 0.55

    if used_max > strict_overlap_limit:
        score -= (used_max - strict_overlap_limit) * 70.0
    else:
        score += (strict_overlap_limit - used_max) * 8.0

    if ticket in history:
        score -= 250.0
    if hist_max > history_overlap_limit:
        score -= (hist_max - history_overlap_limit) * 80.0
    score -= max(0.0, hist_avg - 4.0) * 12.0
    return score


def pick_adaptive_tickets(
    genome: Genome,
    draws: Sequence[Draw],
    count: int,
    used: Set[Ticket],
    *,
    history: Sequence[Ticket],
    strict_overlap_limit: int,
    history_overlap_limit: int,
    diversity_weight: float,
    first_prize_weight: float,
) -> List[Ticket]:
    if count <= 0:
        return []
    latest = draws[-1]
    raw = [ticket_key(t) for t in generate_tickets(draws, genome, max(160, count * 80))]
    candidates: List[Tuple[float, Ticket]] = []
    seen: Set[Ticket] = set()
    used_list = list(used)
    for idx, key in enumerate(raw):
        if key in seen or key in used:
            continue
        seen.add(key)
        score = ticket_score(
            key,
            raw_rank=idx,
            used=used_list,
            history=history,
            latest=latest,
            strict_overlap_limit=strict_overlap_limit,
            history_overlap_limit=history_overlap_limit,
            diversity_weight=diversity_weight,
            first_prize_weight=first_prize_weight,
        )
        candidates.append((score, key))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: List[Ticket] = []
    for _score, key in candidates:
        all_prev = list(used) + selected
        if all(len(set(key) & set(prev)) <= strict_overlap_limit for prev in all_prev):
            selected.append(key)
            used.add(key)
        if len(selected) >= count:
            break

    # 必要数に届かない場合だけ段階的に緩和する。
    if len(selected) < count:
        relaxed_limit = min(6, strict_overlap_limit + 1)
        for _score, key in candidates:
            if key in used:
                continue
            all_prev = list(used) + selected
            if all(len(set(key) & set(prev)) <= relaxed_limit for prev in all_prev):
                selected.append(key)
                used.add(key)
            if len(selected) >= count:
                break

    if len(selected) < count:
        for _score, key in candidates:
            if key in used:
                continue
            selected.append(key)
            used.add(key)
            if len(selected) >= count:
                break

    return selected[:count]


def make_row(
    *,
    rank: int,
    ticket: Ticket,
    genome: Genome,
    source_model: str,
    method: str,
    support: str,
    base_latest_draw_no: int,
    base_latest_date: str,
    created_at: str,
    score: float,
) -> Dict[str, object]:
    return {
        "confidence_rank": rank,
        "base_latest_draw_no": base_latest_draw_no,
        "base_latest_date": base_latest_date,
        "prediction_draw_no": base_latest_draw_no + 1,
        "combo_index": rank,
        "numbers": fmt_ticket(ticket),
        "model_id": genome.id,
        "model_score": round(float(getattr(genome, "score", 0.0)), 6),
        "source_model": source_model,
        "prediction_method": method,
        "ensemble_score": round(score, 6),
        "support_models": support,
        "created_at": created_at,
    }


def append_model_rows(
    *,
    rows: List[Dict[str, object]],
    tickets: Sequence[Ticket],
    genome: Genome,
    source_model: str,
    method: str,
    support: str,
    latest: Draw,
    created_at: str,
    base_score: float,
) -> None:
    for ticket in tickets:
        rows.append(
            make_row(
                rank=len(rows) + 1,
                ticket=ticket,
                genome=genome,
                source_model=source_model,
                method=method,
                support=support,
                base_latest_draw_no=latest.draw_no,
                base_latest_date=latest.date,
                created_at=created_at,
                score=base_score - len(rows),
            )
        )


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Genome, str]:
    draws = load_draws(args.csv)
    if not draws:
        raise SystemExit(f"no draws loaded: {args.csv}")
    latest = draws[-1]
    created_at = now_iso()
    history = load_history_tickets(args.prediction_history)

    full_item = load_required_model(args.full_model)
    full_genome: Genome = full_item["genome"]  # type: ignore[assignment]
    recent_item, recent_fallback = load_optional_model(args.recent_model, full_item)
    recent_genome: Genome = recent_item["genome"]  # type: ignore[assignment]
    super_item, super_fallback = load_optional_model(args.super_recent_model, recent_item)
    super_genome: Genome = super_item["genome"]  # type: ignore[assignment]

    rows: List[Dict[str, object]] = []
    used: Set[Ticket] = set()

    full_tickets = pick_adaptive_tickets(
        full_genome,
        draws,
        args.full_count,
        used,
        history=history,
        strict_overlap_limit=args.strict_overlap_limit,
        history_overlap_limit=args.history_overlap_limit,
        diversity_weight=args.diversity_weight,
        first_prize_weight=args.first_prize_weight,
    )
    append_model_rows(
        rows=rows,
        tickets=full_tickets,
        genome=full_genome,
        source_model=str(full_item.get("path") or args.full_model),
        method="dual_full_period",
        support="全期間型: best_model / diversity+1等狙い補正",
        latest=latest,
        created_at=created_at,
        base_score=30.0,
    )

    recent_tickets = pick_adaptive_tickets(
        recent_genome,
        draws,
        args.recent_count,
        used,
        history=history,
        strict_overlap_limit=args.strict_overlap_limit,
        history_overlap_limit=args.history_overlap_limit,
        diversity_weight=args.diversity_weight,
        first_prize_weight=args.first_prize_weight,
    )
    recent_support = "Recent Era型: 2020年以降専用モデル / 履歴重複抑制" if not recent_fallback else "Recent Era型: fallback to full-period model"
    append_model_rows(
        rows=rows,
        tickets=recent_tickets,
        genome=recent_genome,
        source_model=str(recent_item.get("path") or args.recent_model),
        method="dual_recent_era",
        support=recent_support,
        latest=latest,
        created_at=created_at,
        base_score=22.0,
    )

    super_tickets = pick_adaptive_tickets(
        super_genome,
        draws,
        args.super_recent_count,
        used,
        history=history,
        strict_overlap_limit=args.strict_overlap_limit,
        history_overlap_limit=args.history_overlap_limit,
        diversity_weight=args.diversity_weight * 1.15,
        first_prize_weight=args.first_prize_weight * 1.20,
    )
    if super_tickets:
        super_support = "Super Recent型: 2023年以降専用モデル / 直近追従+1等狙い" if not super_fallback else "Super Recent型: fallback to Recent Era model"
        append_model_rows(
            rows=rows,
            tickets=super_tickets,
            genome=super_genome,
            source_model=str(super_item.get("path") or args.super_recent_model),
            method="dual_super_recent",
            support=super_support,
            latest=latest,
            created_at=created_at,
            base_score=19.0,
        )

    regime_needed = max(0, args.purchase_count - len(rows))
    if regime_needed > 0:
        role_sequence = load_role_strategy(args.regime_strategy, regime_needed)
        regime_rows = make_role_ensemble_prediction_rows(
            full_genome,
            str(full_item.get("path") or args.full_model),
            draws,
            max(1, regime_needed * 3),
            args.strict_overlap_limit,
            role_sequence=role_sequence,
        )
        candidates: List[Tuple[float, Dict[str, object], Ticket]] = []
        for idx, r in enumerate(regime_rows):
            nums = parse_ticket(r.get("numbers", ""))
            if nums is None or nums in used:
                continue
            score = ticket_score(
                nums,
                raw_rank=idx,
                used=list(used),
                history=history,
                latest=latest,
                strict_overlap_limit=args.strict_overlap_limit,
                history_overlap_limit=args.history_overlap_limit,
                diversity_weight=args.diversity_weight * 1.2,
                first_prize_weight=args.first_prize_weight,
            )
            candidates.append((score, dict(r), nums))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, r, nums in candidates:
            if any(len(set(nums) & set(prev)) > args.strict_overlap_limit for prev in used):
                continue
            used.add(nums)
            row = dict(r)
            row["confidence_rank"] = len(rows) + 1
            row["combo_index"] = len(rows) + 1
            row["prediction_method"] = "dual_regime"
            row["ensemble_score"] = round(score, 6)
            row["support_models"] = f"Regime型: {row.get('support_models', '')} / 重複削減補正"
            rows.append(row)
            if len(rows) >= args.purchase_count:
                break

    if len(rows) < args.purchase_count:
        fallback = pick_adaptive_tickets(
            full_genome,
            draws,
            args.purchase_count - len(rows),
            used,
            history=history,
            strict_overlap_limit=args.strict_overlap_limit,
            history_overlap_limit=args.history_overlap_limit,
            diversity_weight=args.diversity_weight,
            first_prize_weight=args.first_prize_weight,
        )
        append_model_rows(
            rows=rows,
            tickets=fallback,
            genome=full_genome,
            source_model=str(full_item.get("path") or args.full_model),
            method="dual_fallback",
            support="補完: 全期間型best_model / adaptive diversity",
            latest=latest,
            created_at=created_at,
            base_score=10.0,
        )

    rows = rows[: args.purchase_count]
    for i, row in enumerate(rows, start=1):
        row["confidence_rank"] = i
        row["combo_index"] = i
    return rows, full_genome, str(full_item.get("path") or args.full_model)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build LOTO7 adaptive dual/super-recent model prediction.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--full-model", default="loto7_best_model.json")
    parser.add_argument("--recent-model", default="outputs/recent_era/recent_era_best_model.json")
    parser.add_argument("--super-recent-model", default="outputs/super_recent/super_recent_best_model.json")
    parser.add_argument("--regime-strategy", default="outputs/role_ensemble/regime_strategy.json")
    parser.add_argument("--prediction-history", default="outputs/evolution_prediction_history.csv")
    parser.add_argument("--prediction", default="outputs/evolution_best_prediction.csv")
    parser.add_argument("--prediction-report", default="outputs/holdout/latest_prediction_report.txt")
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--full-count", type=int, default=2)
    parser.add_argument("--recent-count", type=int, default=1)
    parser.add_argument("--super-recent-count", type=int, default=1)
    parser.add_argument("--overlap-limit", type=int, default=4, help="Backward-compatible alias for --strict-overlap-limit.")
    parser.add_argument("--strict-overlap-limit", type=int, default=None)
    parser.add_argument("--history-overlap-limit", type=int, default=6)
    parser.add_argument("--diversity-weight", type=float, default=1.0)
    parser.add_argument("--first-prize-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.strict_overlap_limit is None:
        args.strict_overlap_limit = args.overlap_limit
    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.full_count < 0 or args.recent_count < 0 or args.super_recent_count < 0:
        raise SystemExit("full/recent/super_recent counts must be non-negative")
    if args.full_count + args.recent_count + args.super_recent_count > args.purchase_count:
        raise SystemExit("full_count + recent_count + super_recent_count must be <= purchase_count")

    rows, full_genome, source_model = build_rows(args)
    write_prediction(args.prediction, rows)
    write_prediction_report(
        args.prediction_report,
        rows,
        full_genome,
        source_model,
        model_count=3,
        min_models=1,
        selection_reason="Adaptive Prediction: 全期間型 + Recent Era + Super Recent 2023+ + Regime / 重複削減・履歴抑制・多様性・1等狙い補正",
        prediction_mode="adaptive_dual_super_recent",
        role_strategy_path=args.regime_strategy,
    )
    print(f"[OK] prediction={args.prediction}")
    print(f"[OK] report={args.prediction_report}")
    print(f"[OK] full_model={args.full_model}")
    print(f"[OK] recent_model={args.recent_model}")
    print(f"[OK] super_recent_model={args.super_recent_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
