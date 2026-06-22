#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_model_self_evolver.py

モデルJSONだけを自己進化する専用スクリプト。
コード本体やworkflowは変更せず、Genomeパラメータだけを小さく変異させ、
holdoutの回収率・収支で採用判定する。

標準では100回反復する。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loto7_evolution_trainer import Genome, crossover, genome_from_dict, load_draws, mutate, random_genome
from merge_evolution_shards import evaluate_model_on_holdout, load_prize_rows, select_target_indices


DEFAULT_SEED_PATTERNS = [
    "loto7_best_model.json",
    "loto7_best_model_shard*_of_08.json",
    "outputs/loto7_best_model_shard*_of_08.json",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_genome_file(path: str) -> Optional[Genome]:
    try:
        payload = read_json(path)
        raw = payload.get("genome", payload)
        if not isinstance(raw, dict):
            return None
        return genome_from_dict(raw)
    except Exception as exc:
        print(f"[WARN] skip model {path}: {exc}")
        return None


def load_seed_genomes(patterns: Sequence[str]) -> List[Tuple[str, Genome]]:
    out: List[Tuple[str, Genome]] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            genome = load_genome_file(path)
            if genome is None or genome.id in seen:
                continue
            seen.add(genome.id)
            out.append((path, genome))
    return out


def append_history(path: str, row: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    fields = list(row.keys())
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rank_counts(metrics: Dict[str, object]) -> Dict[str, int]:
    raw = metrics.get("rank_counts", {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


def score_key(metrics: Dict[str, object]) -> Tuple[float, int, int, int, int, int, float]:
    ranks = rank_counts(metrics)
    return (
        float(metrics.get("roi", 0.0)),
        int(metrics.get("profit", 0)),
        int(ranks.get("1等", 0)),
        int(ranks.get("2等", 0)),
        int(ranks.get("3等", 0)),
        int(ranks.get("4等", 0)),
        float(metrics.get("max_main_match", 0.0)),
    )


def is_adoptable(
    candidate: Dict[str, object],
    baseline: Dict[str, object],
    *,
    min_roi_delta: float,
    min_profit_delta: int,
    allow_high_grade_drop: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    c_roi = float(candidate.get("roi", 0.0))
    b_roi = float(baseline.get("roi", 0.0))
    c_profit = int(candidate.get("profit", 0))
    b_profit = int(baseline.get("profit", 0))
    c_high = int(candidate.get("high_grade_hit_count", 0))
    b_high = int(baseline.get("high_grade_hit_count", 0))

    roi_ok = c_roi >= b_roi + min_roi_delta
    profit_ok = c_profit >= b_profit + min_profit_delta
    high_ok = allow_high_grade_drop or c_high >= b_high

    if roi_ok:
        reasons.append(f"roi {b_roi:.6f} -> {c_roi:.6f}")
    if profit_ok:
        reasons.append(f"profit {b_profit} -> {c_profit}")
    if not high_ok:
        reasons.append(f"high_grade dropped {b_high} -> {c_high}")

    return bool(roi_ok and profit_ok and high_ok), reasons


def make_candidate(parent_pool: Sequence[Genome], iteration: int, rng: random.Random) -> Genome:
    if len(parent_pool) >= 2 and rng.random() < 0.45:
        a, b = rng.sample(list(parent_pool), 2)
        return crossover(a, b, iteration, iteration, rng)
    parent = rng.choice(list(parent_pool))
    return mutate(parent, iteration, iteration, rng)


def payload_for_model(
    *,
    genome: Genome,
    source: str,
    baseline_metrics: Dict[str, object],
    selected_metrics: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "updated_at": now_iso(),
        "kind": "loto7_model_only_self_evolution",
        "source": source,
        "csv": args.csv,
        "purchase_count": args.purchase_count,
        "selection_mode": "model_only_holdout_roi",
        "baseline_metrics": baseline_metrics,
        "selected_holdout": selected_metrics,
        "genome": asdict(genome),
        "notes": [
            "Only Genome/model JSON is self-evolved.",
            "This does not guarantee future lottery winnings or profit.",
        ],
    }


def write_report(path: str, summary: Dict[str, object]) -> None:
    baseline = summary.get("baseline_metrics", {})
    best = summary.get("best_metrics", {})
    adoption = summary.get("adoption", {})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LOTO7 Model-Only Self Evolution Report",
        "======================================",
        "",
        f"created_at: {summary.get('created_at')}",
        f"status: {summary.get('status')}",
        f"iterations_requested: {summary.get('iterations_requested')}",
        f"iterations_evaluated: {summary.get('iterations_evaluated')}",
        "",
        "[Baseline]",
        f"roi_percent: {baseline.get('roi_percent') if isinstance(baseline, dict) else ''}",
        f"profit: {baseline.get('profit') if isinstance(baseline, dict) else ''}",
        f"high_grade_hit_count: {baseline.get('high_grade_hit_count') if isinstance(baseline, dict) else ''}",
        f"max_main_match: {baseline.get('max_main_match') if isinstance(baseline, dict) else ''}",
        "",
        "[Best Candidate]",
        f"roi_percent: {best.get('roi_percent') if isinstance(best, dict) else ''}",
        f"profit: {best.get('profit') if isinstance(best, dict) else ''}",
        f"high_grade_hit_count: {best.get('high_grade_hit_count') if isinstance(best, dict) else ''}",
        f"max_main_match: {best.get('max_main_match') if isinstance(best, dict) else ''}",
        "",
        "[Adoption]",
        json.dumps(adoption, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "注意: 過去検証上の改善候補であり、将来の当せんや利益を保証しません。",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Self-evolve only LOTO7 model Genome JSON.")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--seed-patterns", nargs="*", default=DEFAULT_SEED_PATTERNS)
    parser.add_argument("--output-dir", default="outputs/model_self_evolution")
    parser.add_argument("--candidate-model", default="outputs/model_self_evolution/best_candidate_model.json")
    parser.add_argument("--summary", default="outputs/model_self_evolution/summary.json")
    parser.add_argument("--report", default="outputs/model_self_evolution/report.txt")
    parser.add_argument("--history", default="outputs/model_self_evolution/history.csv")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=180)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-roi-delta-percent", type=float, default=0.10)
    parser.add_argument("--min-profit-delta", type=int, default=0)
    parser.add_argument("--allow-high-grade-drop", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Overwrite --best-model only if the candidate passes adoption rules.")
    parser.add_argument("--max-runtime-minutes", type=float, default=65.0)
    parser.add_argument("--safe-exit-minutes", type=float, default=5.0)
    args = parser.parse_args(argv)

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")

    started = time.monotonic()
    rng = random.Random(args.seed)
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )
    target_indices = target_indices[:: max(1, args.target_stride)]
    if args.max_targets and args.max_targets > 0:
        target_indices = target_indices[-args.max_targets :]
    if not target_indices:
        raise SystemExit("no holdout target draws selected")

    seeds = load_seed_genomes(args.seed_patterns)
    baseline = load_genome_file(args.best_model)
    if baseline is None:
        baseline = seeds[0][1] if seeds else random_genome(0, 0, rng)
    parent_pool = [genome for _, genome in seeds] or [baseline]

    baseline_metrics = evaluate_model_on_holdout(
        genome=baseline,
        model_path=args.best_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )
    best_genome = baseline
    best_metrics = baseline_metrics
    status = "completed"
    evaluated = 0

    for iteration in range(1, args.iterations + 1):
        elapsed_minutes = (time.monotonic() - started) / 60.0
        if elapsed_minutes >= max(0.0, args.max_runtime_minutes - args.safe_exit_minutes):
            status = f"safe_exit_at_{elapsed_minutes:.2f}_minutes"
            break

        candidate = make_candidate(parent_pool + [best_genome], iteration, rng)
        metrics = evaluate_model_on_holdout(
            genome=candidate,
            model_path=f"candidate_{iteration}",
            draws=draws,
            prize_rows=prize_rows,
            target_indices=target_indices,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
        )
        evaluated += 1
        if score_key(metrics) > score_key(best_metrics):
            best_genome = candidate
            best_metrics = metrics
            parent_pool.append(candidate)
            parent_pool = sorted(parent_pool, key=lambda g: g.score, reverse=True)[-24:]

        append_history(
            args.history,
            {
                "created_at": now_iso(),
                "iteration": iteration,
                "genome_id": candidate.id,
                "roi_percent": metrics.get("roi_percent"),
                "profit": metrics.get("profit"),
                "high_grade_hit_count": metrics.get("high_grade_hit_count"),
                "max_main_match": metrics.get("max_main_match"),
                "best_roi_percent": best_metrics.get("roi_percent"),
                "best_profit": best_metrics.get("profit"),
            },
        )

    adopted, reasons = is_adoptable(
        best_metrics,
        baseline_metrics,
        min_roi_delta=args.min_roi_delta_percent / 100.0,
        min_profit_delta=args.min_profit_delta,
        allow_high_grade_drop=args.allow_high_grade_drop,
    )

    candidate_payload = payload_for_model(
        genome=best_genome,
        source="model_self_evolver",
        baseline_metrics=baseline_metrics,
        selected_metrics=best_metrics,
        args=args,
    )
    write_json(args.candidate_model, candidate_payload)

    if adopted and args.apply:
        write_json(args.best_model, candidate_payload)

    summary = {
        "created_at": now_iso(),
        "status": status,
        "iterations_requested": args.iterations,
        "iterations_evaluated": evaluated,
        "target_draws": len(target_indices),
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "candidate_model": args.candidate_model,
        "best_model": args.best_model,
        "adoption": {
            "adopted": adopted,
            "apply_requested": bool(args.apply),
            "applied": bool(adopted and args.apply),
            "reasons": reasons,
            "min_roi_delta_percent": args.min_roi_delta_percent,
            "min_profit_delta": args.min_profit_delta,
            "allow_high_grade_drop": bool(args.allow_high_grade_drop),
        },
    }
    write_json(args.summary, summary)
    write_report(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
