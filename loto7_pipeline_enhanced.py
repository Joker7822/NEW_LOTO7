#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced NEW_LOTO7 pipeline.

Implemented without third-party dependencies:
- all-period + recent 240/120/60 ensemble number scoring
- multi-window affinity pair scoring
- dormant number and dormant pair boost
- constraint score for sum, odd/even, low/high, consecutive numbers, repeat count
- deterministic Monte Carlo candidate expansion
- 5+ hit oriented structure score
- same output files as loto7_pipeline.py for compatibility
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loto7_pipeline import (
    Draw,
    append_csv,
    evaluate_combo,
    git_commit_push,
    infer_existing_purchase_count,
    load_draws,
    load_json,
    run_scraping,
    save_json,
    write_csv,
)

NUMBERS = tuple(range(1, 38))
PICK_SIZE = 7
DEFAULT_UNIT_COST = 300


def _window(draws: Sequence[Draw], n: int) -> Sequence[Draw]:
    if n <= 0 or len(draws) <= n:
        return draws
    return draws[-n:]


def _norm(counter: Dict[object, float], key: object) -> float:
    if not counter:
        return 0.0
    mx = max(counter.values()) if counter else 0.0
    return float(counter.get(key, 0.0)) / mx if mx else 0.0


def number_scores(train: Sequence[Draw]) -> Dict[int, float]:
    """Ensemble score: all history + recent 240/120/60 + dormant gap."""
    out = {n: 0.0 for n in NUMBERS}
    windows = [(0, 0.28), (240, 0.34), (120, 0.23), (60, 0.15)]

    for win, weight in windows:
        subset = list(_window(train, win))
        if not subset:
            continue
        freq = Counter()
        last_seen = {n: -1 for n in NUMBERS}
        total = len(subset)
        for idx, draw in enumerate(subset):
            age = total - idx - 1
            decay = 0.985 ** age if win != 60 else 0.975 ** age
            for n in draw.main:
                freq[n] += decay
                last_seen[n] = idx
        mx = max(freq.values()) if freq else 1.0
        for n in NUMBERS:
            gap = total - last_seen[n] - 1 if last_seen[n] >= 0 else total
            dormant = min(gap, 34) / 34.0
            out[n] += weight * ((freq[n] / mx if mx else 0.0) * 0.82 + dormant * 0.18)
    return out


def pair_scores(train: Sequence[Draw]) -> Dict[Tuple[int, int], float]:
    """Multi-window affinity and pair-recency score."""
    out: Dict[Tuple[int, int], float] = {}
    windows = [(0, 0.28), (240, 0.32), (120, 0.25), (60, 0.15)]
    all_pairs = [tuple(sorted(p)) for p in itertools.combinations(NUMBERS, 2)]

    for win, weight in windows:
        subset = list(_window(train, win))
        if not subset:
            continue
        c: Counter[Tuple[int, int]] = Counter()
        last_seen = {p: -1 for p in all_pairs}
        total = len(subset)
        for idx, draw in enumerate(subset):
            age = total - idx - 1
            decay = 0.99 ** age
            for p in itertools.combinations(draw.main, 2):
                key = tuple(sorted(p))
                c[key] += decay
                last_seen[key] = idx
        mx = max(c.values()) if c else 1.0
        for p in all_pairs:
            gap = total - last_seen[p] - 1 if last_seen[p] >= 0 else total
            dormant_pair = min(gap, 80) / 80.0
            out[p] = out.get(p, 0.0) + weight * ((c[p] / mx if mx else 0.0) * 0.88 + dormant_pair * 0.12)
    return out


def triple_scores(train: Sequence[Draw]) -> Dict[Tuple[int, int, int], float]:
    subset = list(_window(train, 180))
    c: Counter[Tuple[int, int, int]] = Counter()
    total = len(subset)
    for idx, draw in enumerate(subset):
        age = total - idx - 1
        decay = 0.992 ** age
        for tri in itertools.combinations(draw.main, 3):
            c[tuple(sorted(tri))] += decay
    mx = max(c.values()) if c else 1.0
    return {k: v / mx for k, v in c.items()} if mx else {}


def constraint_score(combo: Sequence[int], train: Sequence[Draw]) -> float:
    nums = tuple(sorted(combo))
    total = sum(nums)
    odd = sum(n % 2 for n in nums)
    low = sum(1 for n in nums if n <= 18)
    consecutive = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
    repeat_last = len(set(nums) & set(train[-1].main)) if train else 0
    last_digit_max = max(Counter(n % 10 for n in nums).values())
    decade_count = len(set(n // 10 for n in nums))

    score = 0.0
    score += max(0.0, 1.0 - abs(total - 135) / 55.0) * 0.26
    score += (1.0 if odd in (3, 4) else 0.55 if odd in (2, 5) else 0.05) * 0.20
    score += (1.0 if low in (3, 4) else 0.55 if low in (2, 5) else 0.05) * 0.18
    score += (1.0 if consecutive <= 1 else 0.55 if consecutive == 2 else 0.05) * 0.11
    score += (1.0 if repeat_last in (1, 2) else 0.58 if repeat_last in (0, 3) else 0.10) * 0.10
    score += (1.0 if last_digit_max <= 2 else 0.45 if last_digit_max == 3 else 0.05) * 0.07
    score += (1.0 if decade_count >= 3 else 0.35) * 0.08
    return score


def structure5_score(combo: Sequence[int], train: Sequence[Draw]) -> float:
    """Structure score designed to improve 5+ overlap chances without future data."""
    nums = set(combo)
    recent = list(_window(train, 240))
    if not recent:
        return 0.0
    overlaps = Counter(len(nums & set(d.main)) for d in recent)
    # Avoid exact overfitting to recent draws, but reward structures that repeatedly create 3/4 overlaps.
    return min(1.0, (overlaps.get(4, 0) * 1.25 + overlaps.get(3, 0) * 0.45 + overlaps.get(5, 0) * 0.25) / max(len(recent) * 0.18, 1.0))


def combo_score(
    combo: Sequence[int],
    train: Sequence[Draw],
    ns: Dict[int, float],
    ps: Dict[Tuple[int, int], float],
    ts: Dict[Tuple[int, int, int], float],
) -> float:
    nums = tuple(sorted(combo))
    single = sum(ns.get(n, 0.0) for n in nums) / PICK_SIZE
    pair = sum(ps.get(tuple(sorted((a, b))), 0.0) for a, b in itertools.combinations(nums, 2)) / 21.0
    triple = sum(ts.get(tuple(sorted(t)), 0.0) for t in itertools.combinations(nums, 3)) / 35.0
    constraint = constraint_score(nums, train)
    structure5 = structure5_score(nums, train)
    return 0.36 * single + 0.25 * pair + 0.12 * triple + 0.17 * constraint + 0.10 * structure5


def build_candidate_pool(ns: Dict[int, float], train: Sequence[Draw], pool_size: int) -> List[int]:
    ranked = [n for n, _ in sorted(ns.items(), key=lambda kv: kv[1], reverse=True)]
    base = ranked[: max(pool_size, 18)]
    # Add dormant numbers so the search is not only hot-number biased.
    last_seen = {n: -1 for n in NUMBERS}
    for idx, draw in enumerate(train):
        for n in draw.main:
            last_seen[n] = idx
    dormant = sorted(NUMBERS, key=lambda n: last_seen[n])[:6]
    pool = sorted(set(base + dormant))[: max(pool_size, 24)]
    return pool


def deterministic_monte_carlo(pool: Sequence[int], ns: Dict[int, float], trials: int, seed: int) -> List[Tuple[int, ...]]:
    rng = random.Random(seed)
    weights = [max(ns.get(n, 0.0), 0.001) for n in pool]
    candidates = set()
    pool = list(pool)
    for _ in range(max(0, trials)):
        available = pool[:]
        available_weights = weights[:]
        pick: List[int] = []
        for _j in range(PICK_SIZE):
            total_w = sum(available_weights)
            r = rng.random() * total_w
            acc = 0.0
            chosen_idx = 0
            for i, w in enumerate(available_weights):
                acc += w
                if acc >= r:
                    chosen_idx = i
                    break
            pick.append(available.pop(chosen_idx))
            available_weights.pop(chosen_idx)
        candidates.add(tuple(sorted(pick)))
    return list(candidates)


def generate_candidates(train: Sequence[Draw], purchase_count: int = 5, pool_size: int = 24) -> List[Tuple[int, ...]]:
    if not train:
        raise ValueError("train is empty")
    ns = number_scores(train)
    ps = pair_scores(train)
    ts = triple_scores(train)
    pool = build_candidate_pool(ns, train, pool_size=pool_size)

    candidates = set(itertools.combinations(pool, PICK_SIZE))
    seed = int(train[-1].draw_no) if train else 42
    candidates.update(deterministic_monte_carlo(pool, ns, trials=3500, seed=seed))

    scored = [(combo_score(c, train, ns, ps, ts), tuple(sorted(c))) for c in candidates]
    scored.sort(reverse=True, key=lambda x: (x[0], -sum(x[1])))

    selected: List[Tuple[int, ...]] = []
    for score, combo in scored:
        # Five tickets: keep diversity, but allow enough overlap for high-hit clusters.
        if all(len(set(combo) & set(prev)) <= 5 for prev in selected):
            selected.append(combo)
        if len(selected) >= purchase_count:
            break
    if len(selected) < purchase_count:
        for _, combo in scored:
            if combo not in selected:
                selected.append(combo)
            if len(selected) >= purchase_count:
                break
    return selected[:purchase_count]


def reset_backtest_outputs(result_csv: str, summary_csv: str, resume_state_path: str, reason: str) -> None:
    print(f"[RESET] rebuilding backtest outputs: {reason}")
    for path in [result_csv, summary_csv, resume_state_path]:
        p = Path(path)
        if p.exists():
            p.unlink()
            print(f"[RESET] removed {path}")


def run_backtest(
    draws: Sequence[Draw],
    output_dir: str,
    resume_state_path: str,
    purchase_count: int,
    min_train_draws: int,
    max_targets: Optional[int],
    push_every: int,
    force_rebuild: bool = False,
    pool_size: int = 24,
) -> Dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_csv = str(out / "loto7_backtest_result.csv")
    summary_csv = str(out / "loto7_backtest_summary.csv")

    state = load_json(resume_state_path)
    existing_purchase_count = infer_existing_purchase_count(result_csv)
    reset_reason = ""
    if force_rebuild:
        reset_reason = "--force-rebuild"
    elif state:
        if int(state.get("purchase_count", purchase_count)) != int(purchase_count):
            reset_reason = f"purchase_count changed: {state.get('purchase_count')} -> {purchase_count}"
        elif int(state.get("min_train_draws", min_train_draws)) != int(min_train_draws):
            reset_reason = f"min_train_draws changed: {state.get('min_train_draws')} -> {min_train_draws}"
        elif str(state.get("model", "")) != "enhanced_v1":
            reset_reason = "model changed -> enhanced_v1"
    elif existing_purchase_count is not None and int(existing_purchase_count) != int(purchase_count):
        reset_reason = f"existing ticket count changed: {existing_purchase_count} -> {purchase_count}"

    if reset_reason:
        reset_backtest_outputs(result_csv, summary_csv, resume_state_path, reset_reason)
        state = {}

    last_completed = int(state.get("last_completed_draw_no", 0) or 0)
    processed_now = 0
    targets = [d for i, d in enumerate(draws) if i >= min_train_draws and d.draw_no > last_completed]
    if max_targets is not None:
        targets = targets[:max_targets]
    print(f"[BACKTEST] model=enhanced_v1 targets={len(targets)} last_completed={last_completed} purchase_count={purchase_count}")

    for target in targets:
        target_index = next(i for i, d in enumerate(draws) if d.draw_no == target.draw_no)
        train = list(draws[:target_index])
        combos = generate_candidates(train, purchase_count=purchase_count, pool_size=pool_size)
        rows = []
        for idx, combo in enumerate(combos, start=1):
            main_match, bonus_match, rank = evaluate_combo(combo, target)
            rows.append(
                {
                    "target_draw_no": target.draw_no,
                    "target_date": target.date,
                    "combo_index": idx,
                    "numbers": " ".join(f"{n:02d}" for n in combo),
                    "actual_main": " ".join(f"{n:02d}" for n in target.main),
                    "actual_bonus": " ".join(f"{n:02d}" for n in target.bonus),
                    "main_match": main_match,
                    "bonus_match": bonus_match,
                    "prize_rank": rank,
                    "model": "enhanced_v1",
                }
            )
        append_csv(
            result_csv,
            ["target_draw_no", "target_date", "combo_index", "numbers", "actual_main", "actual_bonus", "main_match", "bonus_match", "prize_rank", "model"],
            rows,
        )
        processed_now += 1
        save_json(
            resume_state_path,
            {
                "last_completed_draw_no": target.draw_no,
                "last_completed_date": target.date,
                "processed_now": processed_now,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purchase_count": purchase_count,
                "min_train_draws": min_train_draws,
                "model": "enhanced_v1",
                "pool_size": pool_size,
            },
        )
        if push_every > 0 and processed_now % push_every == 0:
            git_commit_push(
                f"Update enhanced LOTO7 backtest progress up to draw {target.draw_no}",
                [result_csv, summary_csv, resume_state_path],
            )

    total_rows = 0
    total_targets = set()
    rank_counts: Counter[str] = Counter()
    max_main_match = 0
    if Path(result_csv).exists():
        with Path(result_csv).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                total_rows += 1
                total_targets.add(row.get("target_draw_no", ""))
                rank_counts[row.get("prize_rank", "外れ") or "外れ"] += 1
                try:
                    max_main_match = max(max_main_match, int(row.get("main_match", 0) or 0))
                except ValueError:
                    pass

    summary_rows = []
    for rank in ["1等", "2等", "3等", "4等", "5等", "6等", "外れ"]:
        count = rank_counts.get(rank, 0)
        summary_rows.append({"metric": rank, "value": count, "rate": f"{(count / total_rows * 100):.6f}%" if total_rows else "0.000000%"})
    summary_rows.extend(
        [
            {"metric": "targets", "value": len(total_targets), "rate": ""},
            {"metric": "tickets", "value": total_rows, "rate": ""},
            {"metric": "purchase_count", "value": purchase_count, "rate": ""},
            {"metric": "min_train_draws", "value": min_train_draws, "rate": ""},
            {"metric": "processed_now", "value": processed_now, "rate": ""},
            {"metric": "max_main_match", "value": max_main_match, "rate": ""},
            {"metric": "model", "value": "enhanced_v1", "rate": ""},
            {"metric": "pool_size", "value": pool_size, "rate": ""},
            {"metric": "updated_at", "value": dt.datetime.now(dt.timezone.utc).isoformat(), "rate": ""},
        ]
    )
    write_csv(summary_csv, ["metric", "value", "rate"], summary_rows)
    return {
        "processed_now": processed_now,
        "targets_total": len(total_targets),
        "tickets_total": total_rows,
        "max_main_match": max_main_match,
        "result_csv": result_csv,
        "summary_csv": summary_csv,
        "model": "enhanced_v1",
    }


def predict_latest(draws: Sequence[Draw], output_dir: str, purchase_count: int, pool_size: int) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combos = generate_candidates(draws, purchase_count=purchase_count, pool_size=pool_size)
    latest = draws[-1]
    path = str(out / "loto7_latest_prediction.csv")
    rows = []
    for idx, combo in enumerate(combos, start=1):
        rows.append(
            {
                "base_latest_draw_no": latest.draw_no,
                "base_latest_date": latest.date,
                "prediction_draw_no": latest.draw_no + 1,
                "combo_index": idx,
                "numbers": " ".join(f"{n:02d}" for n in combo),
                "model": "enhanced_v1",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
    write_csv(path, ["base_latest_draw_no", "base_latest_date", "prediction_draw_no", "combo_index", "numbers", "model", "created_at"], rows)
    print(f"[PREDICT] wrote {path}")
    for row in rows:
        print(f"  {row['combo_index']}: {row['numbers']}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NEW_LOTO7 enhanced resumable pipeline")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume-state", default="outputs/resume_state.json")
    parser.add_argument("--run-scraping", action="store_true")
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--max-targets", default=None)
    parser.add_argument("--push-every", type=int, default=100)
    parser.add_argument("--push-final", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--pool-size", type=int, default=int(os.getenv("LOTO7_ENHANCED_POOL_SIZE", "24")))
    args = parser.parse_args(argv)

    max_targets: Optional[int]
    if args.max_targets is None or str(args.max_targets).lower() in {"all", "none", ""}:
        max_targets = None
    else:
        max_targets = int(args.max_targets)

    if args.run_scraping:
        run_scraping(args.csv, months=max(1, args.months))
    draws = load_draws(args.csv)
    if len(draws) <= args.min_train_draws:
        raise SystemExit(f"draws is too small: {len(draws)} <= min_train_draws={args.min_train_draws}")

    result: Dict[str, object] = {}
    if not args.skip_backtest:
        result = run_backtest(
            draws=draws,
            output_dir=args.output_dir,
            resume_state_path=args.resume_state,
            purchase_count=args.purchase_count,
            min_train_draws=args.min_train_draws,
            max_targets=max_targets,
            push_every=args.push_every,
            force_rebuild=args.force_rebuild,
            pool_size=args.pool_size,
        )
        print(f"[SUMMARY] {json.dumps(result, ensure_ascii=False)}")

    prediction_path = predict_latest(draws, output_dir=args.output_dir, purchase_count=args.purchase_count, pool_size=args.pool_size)
    if args.push_final:
        git_commit_push(
            "Update enhanced LOTO7 pipeline outputs",
            [
                args.csv,
                args.resume_state,
                str(Path(args.output_dir) / "loto7_backtest_result.csv"),
                str(Path(args.output_dir) / "loto7_backtest_summary.csv"),
                prediction_path,
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
