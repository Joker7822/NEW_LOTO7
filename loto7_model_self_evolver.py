#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_model_self_evolver.py

モデルJSONだけを自己進化する専用スクリプト。
採用済みベストモデルを土台にしつつ、探索幅・親プール・評価スコアを強化して
局所最適から抜けやすくする。

注意:
  宝くじ抽せんはランダム性が高く、的中・利益は保証しない。
  このスクリプトは「過去検証で相対的に良い戦略」を探索するためのもの。
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import glob
import json
import os
import pickle
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loto7_evolution_trainer import Genome, crossover, genome_from_dict, load_draws, mutate, random_genome
from merge_evolution_shards import evaluate_model_on_holdout, load_prize_rows, select_target_indices


DEFAULT_SEED_PATTERNS = [
    "loto7_best_model.json",
]

RANK_SCORE_WEIGHTS = {
    "1等": 50000.0,
    "2等": 22000.0,
    "3等": 9000.0,
    "4等": 1400.0,
    "5等": 150.0,
    "6等": 25.0,
}


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


def composite_score(metrics: Dict[str, object]) -> float:
    """Exploration score used to keep useful candidates in the parent pool.

    Adoption still requires ROI/profit/high-grade rules. This score is broader:
    it rewards ROI, profit, 4等以上, 5等, max match, and reduces dead-end 6等-only drift.
    """
    ranks = rank_counts(metrics)
    roi = float(metrics.get("roi", 0.0))
    profit = int(metrics.get("profit", 0))
    grade_hit = int(metrics.get("grade_hit_count", 0))
    high_grade = int(metrics.get("high_grade_hit_count", 0))
    max_main = int(metrics.get("max_main_match", 0))

    score = roi * 10000.0
    score += profit / 1000.0
    score += high_grade * 850.0
    score += grade_hit * 18.0
    score += max_main * 450.0
    for rank, weight in RANK_SCORE_WEIGHTS.items():
        score += ranks.get(rank, 0) * weight
    # 6等が多いだけの候補へ寄りすぎないよう軽く抑制。
    score -= max(0, ranks.get("6等", 0) - ranks.get("5等", 0) * 2) * 3.0
    return round(score, 6)


def score_key(metrics: Dict[str, object]) -> Tuple[float, float, int, int, int, int, int, int]:
    ranks = rank_counts(metrics)
    return (
        composite_score(metrics),
        float(metrics.get("roi", 0.0)),
        int(metrics.get("profit", 0)),
        int(ranks.get("1等", 0)),
        int(ranks.get("2等", 0)),
        int(ranks.get("3等", 0)),
        int(ranks.get("4等", 0)),
        int(metrics.get("max_main_match", 0)),
    )


def assign_score(genome: Genome, metrics: Dict[str, object]) -> Genome:
    genome.score = composite_score(metrics)
    genome.max_main_match = int(metrics.get("max_main_match", 0))
    genome.best_rank_count = int(metrics.get("high_grade_hit_count", 0))
    return genome


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
    if not roi_ok:
        reasons.append(f"roi not improved enough {b_roi:.6f} -> {c_roi:.6f}")
    if not profit_ok:
        reasons.append(f"profit not improved enough {b_profit} -> {c_profit}")

    return bool(roi_ok and profit_ok and high_ok), reasons


def exploratory_mutate(genome: Genome, iteration: int, rng: random.Random, intensity: int = 3) -> Genome:
    out = genome
    for _ in range(max(1, intensity)):
        out = mutate(out, iteration, iteration, rng)
    return out


def make_candidate(
    parent_pool: Sequence[Genome],
    iteration: int,
    rng: random.Random,
    *,
    exploration_rate: float,
) -> Genome:
    parents = list(parent_pool)
    if not parents or rng.random() < 0.10:
        return random_genome(iteration, iteration, rng)

    if rng.random() < exploration_rate:
        parent = rng.choice(parents)
        return exploratory_mutate(parent, iteration, rng, intensity=rng.choice([2, 3, 4, 5]))

    if len(parents) >= 2 and rng.random() < 0.55:
        a, b = rng.sample(parents, 2)
        return crossover(a, b, iteration, iteration, rng)

    parent = rng.choice(parents)
    return mutate(parent, iteration, iteration, rng)


def encode_rng_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def restore_rng_state(rng: random.Random, encoded: object) -> bool:
    if not isinstance(encoded, str) or not encoded:
        return False
    try:
        rng.setstate(pickle.loads(base64.b64decode(encoded.encode("ascii"))))
        return True
    except Exception as exc:
        print(f"[WARN] failed to restore RNG state: {exc}")
        return False


def state_matches(state: Dict[str, object], args: argparse.Namespace, target_draws: int) -> bool:
    return (
        state.get("csv") == args.csv
        and int(state.get("purchase_count", -1)) == int(args.purchase_count)
        and int(state.get("unit_cost", -1)) == int(args.unit_cost)
        and int(state.get("target_draws", -1)) == int(target_draws)
    )


def load_resume_state(path: str, args: argparse.Namespace, target_draws: int) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return None
    try:
        state = read_json(path)
    except Exception as exc:
        print(f"[WARN] cannot read state {path}: {exc}")
        return None
    if not state_matches(state, args, target_draws):
        print("[INFO] state exists but does not match current run settings; starting fresh")
        return None
    last_iteration = int(state.get("last_iteration", 0))
    status = str(state.get("status", ""))
    if last_iteration >= args.iterations:
        print("[INFO] previous run already reached requested iterations; starting fresh")
        return None
    if status == "completed":
        print("[INFO] previous matching run already completed; starting fresh")
        return None
    return state


def genomes_from_state(raw_items: object) -> List[Genome]:
    if not isinstance(raw_items, list):
        return []
    out: List[Genome] = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            genome = genome_from_dict(item)
        except Exception:
            continue
        if genome.id in seen:
            continue
        seen.add(genome.id)
        out.append(genome)
    return out


def trim_parent_pool(parent_pool: Sequence[Genome], limit: int) -> List[Genome]:
    seen = set()
    out: List[Genome] = []
    for genome in sorted(parent_pool, key=lambda g: float(getattr(g, "score", 0.0)), reverse=True):
        if genome.id in seen:
            continue
        seen.add(genome.id)
        out.append(genome)
        if len(out) >= limit:
            break
    return out


def save_state(
    path: str,
    *,
    args: argparse.Namespace,
    status: str,
    target_draws: int,
    last_iteration: int,
    evaluated_total: int,
    best_genome: Genome,
    best_metrics: Dict[str, object],
    baseline_genome: Genome,
    baseline_metrics: Dict[str, object],
    parent_pool: Sequence[Genome],
    rng: random.Random,
) -> None:
    payload = {
        "updated_at": now_iso(),
        "status": status,
        "csv": args.csv,
        "purchase_count": args.purchase_count,
        "unit_cost": args.unit_cost,
        "target_draws": target_draws,
        "iterations_requested": args.iterations,
        "last_iteration": last_iteration,
        "iterations_evaluated": evaluated_total,
        "objective": "composite_roi_profit_highgrade_exploration",
        "best_genome": asdict(best_genome),
        "best_metrics": best_metrics,
        "baseline_genome": asdict(baseline_genome),
        "baseline_metrics": baseline_metrics,
        "parent_pool": [asdict(g) for g in trim_parent_pool(parent_pool, args.parent_pool_size)],
        "rng_state": encode_rng_state(rng),
    }
    write_json(path, payload)


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
        "selection_mode": "model_only_holdout_roi_with_exploration_score",
        "objective_score": composite_score(selected_metrics),
        "baseline_metrics": baseline_metrics,
        "selected_holdout": selected_metrics,
        "genome": asdict(genome),
        "notes": [
            "Only Genome/model JSON is self-evolved.",
            "Parent-pool diversity and composite exploration score are used before strict adoption.",
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
        f"resume_used: {summary.get('resume_used')}",
        f"start_iteration: {summary.get('start_iteration')}",
        f"last_iteration: {summary.get('last_iteration')}",
        f"iterations_requested: {summary.get('iterations_requested')}",
        f"iterations_evaluated: {summary.get('iterations_evaluated')}",
        f"iterations_evaluated_this_run: {summary.get('iterations_evaluated_this_run')}",
        f"target_draws: {summary.get('target_draws')}",
        f"objective: {summary.get('objective')}",
        "",
        "[Baseline]",
        f"objective_score: {composite_score(baseline) if isinstance(baseline, dict) else ''}",
        f"roi_percent: {baseline.get('roi_percent') if isinstance(baseline, dict) else ''}",
        f"profit: {baseline.get('profit') if isinstance(baseline, dict) else ''}",
        f"grade_hit_count: {baseline.get('grade_hit_count') if isinstance(baseline, dict) else ''}",
        f"high_grade_hit_count: {baseline.get('high_grade_hit_count') if isinstance(baseline, dict) else ''}",
        f"max_main_match: {baseline.get('max_main_match') if isinstance(baseline, dict) else ''}",
        "",
        "[Best Candidate]",
        f"objective_score: {composite_score(best) if isinstance(best, dict) else ''}",
        f"roi_percent: {best.get('roi_percent') if isinstance(best, dict) else ''}",
        f"profit: {best.get('profit') if isinstance(best, dict) else ''}",
        f"grade_hit_count: {best.get('grade_hit_count') if isinstance(best, dict) else ''}",
        f"high_grade_hit_count: {best.get('high_grade_hit_count') if isinstance(best, dict) else ''}",
        f"max_main_match: {best.get('max_main_match') if isinstance(best, dict) else ''}",
        "",
        "[Adoption]",
        json.dumps(adoption, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "注意: 過去検証上の改善候補であり、将来の当せんや利益を保証しません。",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_git(args: Sequence[str]) -> int:
    proc = subprocess.run(["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout.rstrip())
    return int(proc.returncode)


def checkpoint_commit(args: argparse.Namespace, *, iteration: int, adopted: bool) -> None:
    if not args.checkpoint_commit:
        return
    try:
        paths = [args.best_model, args.candidate_model, args.summary, args.report, args.history, args.state]
        existing = [p for p in paths if Path(p).exists()]
        if not existing:
            return
        add_rc = run_git(["add", "-f", *existing])
        if add_rc != 0:
            print(f"[WARN] checkpoint git add failed at iteration {iteration}")
            return
        if run_git(["diff", "--cached", "--quiet"]) == 0:
            return
        label = "adopted" if adopted else "candidate"
        if run_git(["commit", "-m", f"Checkpoint LOTO7 model self-evolution {label} at iteration {iteration} [skip ci]"]) != 0:
            print(f"[WARN] checkpoint git commit failed at iteration {iteration}")
            return
        branch = args.checkpoint_branch or os.environ.get("GITHUB_REF_NAME", "")
        push_rc = run_git(["push", "origin", f"HEAD:{branch}"]) if branch else run_git(["push"])
        if push_rc != 0:
            print(f"[WARN] checkpoint git push failed at iteration {iteration}; final workflow commit can still save outputs")
            return
        print(f"[CHECKPOINT] committed best model update at iteration {iteration}")
    except Exception as exc:
        print(f"[WARN] checkpoint commit failed at iteration {iteration}: {exc}")


def build_summary(
    *,
    status: str,
    resume_used: bool,
    start_iteration: int,
    last_iteration: int,
    evaluated_total: int,
    evaluated_this_run: int,
    target_draws: int,
    baseline_metrics: Dict[str, object],
    best_metrics: Dict[str, object],
    adopted: bool,
    reasons: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_at": now_iso(),
        "status": status,
        "resume_used": resume_used,
        "state": args.state,
        "start_iteration": start_iteration,
        "last_iteration": last_iteration,
        "iterations_requested": args.iterations,
        "iterations_evaluated": evaluated_total,
        "iterations_evaluated_this_run": evaluated_this_run,
        "target_draws": target_draws,
        "objective": "composite_roi_profit_highgrade_exploration",
        "parent_pool_size": args.parent_pool_size,
        "exploration_rate": args.exploration_rate,
        "random_parents": args.random_parents,
        "baseline_objective_score": composite_score(baseline_metrics),
        "best_objective_score": composite_score(best_metrics),
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


def write_candidate_outputs(
    *,
    args: argparse.Namespace,
    best_genome: Genome,
    baseline_metrics: Dict[str, object],
    best_metrics: Dict[str, object],
    status: str,
    resume_used: bool,
    start_iteration: int,
    last_iteration: int,
    evaluated_total: int,
    evaluated_this_run: int,
    target_draws: int,
) -> Tuple[bool, List[str], Dict[str, object]]:
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
    summary = build_summary(
        status=status,
        resume_used=resume_used,
        start_iteration=start_iteration,
        last_iteration=last_iteration,
        evaluated_total=evaluated_total,
        evaluated_this_run=evaluated_this_run,
        target_draws=target_draws,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        adopted=adopted,
        reasons=reasons,
        args=args,
    )
    write_json(args.summary, summary)
    write_report(args.report, summary)
    return adopted, reasons, summary


def evaluate_candidate(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, object]],
    target_indices: Sequence[int],
    purchase_count: int,
    unit_cost: int,
) -> Tuple[Genome, Dict[str, object]]:
    metrics = evaluate_model_on_holdout(
        genome=genome,
        model_path=model_path,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=purchase_count,
        unit_cost=unit_cost,
    )
    assign_score(genome, metrics)
    return genome, metrics


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
    parser.add_argument("--state", default="outputs/model_self_evolution/state.json")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--unit-cost", type=int, default=300)
    parser.add_argument("--min-train-draws", type=int, default=1)
    parser.add_argument("--holdout-start-draw", type=int, default=2)
    parser.add_argument("--holdout-end-draw", type=int, default=None)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--min-roi-delta-percent", type=float, default=0.10)
    parser.add_argument("--min-profit-delta", type=int, default=0)
    parser.add_argument("--allow-high-grade-drop", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Overwrite --best-model only if the candidate passes adoption rules.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from --state when compatible. Enabled by default.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore any existing --state and start fresh.")
    parser.add_argument("--max-runtime-minutes", type=float, default=65.0)
    parser.add_argument("--safe-exit-minutes", type=float, default=5.0)
    parser.add_argument("--checkpoint-commit", action="store_true", help="Commit and push whenever a new in-run best candidate is found.")
    parser.add_argument("--checkpoint-branch", default="", help="Branch to push checkpoint commits to. Defaults to GITHUB_REF_NAME.")
    parser.add_argument("--parent-pool-size", type=int, default=36)
    parser.add_argument("--random-parents", type=int, default=12)
    parser.add_argument("--exploration-rate", type=float, default=0.30)
    parser.add_argument("--stagnation-patience", type=int, default=50)
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

    baseline_genome, baseline_metrics = evaluate_candidate(
        genome=baseline,
        model_path=args.best_model,
        draws=draws,
        prize_rows=prize_rows,
        target_indices=target_indices,
        purchase_count=args.purchase_count,
        unit_cost=args.unit_cost,
    )
    best_genome = baseline_genome
    best_metrics = baseline_metrics

    parent_pool: List[Genome] = [baseline_genome]
    for _, genome in seeds:
        parent_pool.append(genome)
    for i in range(max(0, args.random_parents)):
        if i % 2 == 0:
            parent_pool.append(exploratory_mutate(baseline_genome, i + 1, rng, intensity=rng.choice([2, 3, 4])))
        else:
            parent_pool.append(random_genome(0, i, rng))
    parent_pool = trim_parent_pool(parent_pool, args.parent_pool_size)

    status = "completed"
    last_iteration = 0
    evaluated_total = 0
    evaluated_this_run = 0
    resume_used = False
    start_iteration = 1
    stagnation = 0

    if args.resume:
        state = load_resume_state(args.state, args, len(target_indices))
        if state:
            try:
                baseline_genome = genome_from_dict(state.get("baseline_genome", {}))
                baseline_metrics = state.get("baseline_metrics", baseline_metrics)  # type: ignore[assignment]
                assign_score(baseline_genome, baseline_metrics)
                best_genome = genome_from_dict(state.get("best_genome", {}))
                best_metrics = state.get("best_metrics", best_metrics)  # type: ignore[assignment]
                assign_score(best_genome, best_metrics)
                restored_pool = genomes_from_state(state.get("parent_pool"))
                if restored_pool:
                    parent_pool = trim_parent_pool(restored_pool + [best_genome, baseline_genome], args.parent_pool_size)
                restore_rng_state(rng, state.get("rng_state"))
                last_iteration = int(state.get("last_iteration", 0))
                evaluated_total = int(state.get("iterations_evaluated", last_iteration))
                start_iteration = last_iteration + 1
                resume_used = True
                print(f"[RESUME] continuing from iteration {start_iteration}/{args.iterations}")
            except Exception as exc:
                print(f"[WARN] failed to resume state; starting fresh: {exc}")

    for iteration in range(start_iteration, args.iterations + 1):
        elapsed_minutes = (time.monotonic() - started) / 60.0
        if elapsed_minutes >= max(0.0, args.max_runtime_minutes - args.safe_exit_minutes):
            status = f"safe_exit_at_{elapsed_minutes:.2f}_minutes"
            break

        if stagnation >= args.stagnation_patience:
            parent_pool.append(random_genome(iteration, iteration, rng))
            parent_pool.append(exploratory_mutate(best_genome, iteration, rng, intensity=5))
            parent_pool = trim_parent_pool(parent_pool, args.parent_pool_size)
            stagnation = 0
            print(f"[EXPLORE] injected diversity at iteration {iteration}")

        candidate = make_candidate(parent_pool + [best_genome], iteration, rng, exploration_rate=args.exploration_rate)
        candidate, metrics = evaluate_candidate(
            genome=candidate,
            model_path=f"candidate_{iteration}",
            draws=draws,
            prize_rows=prize_rows,
            target_indices=target_indices,
            purchase_count=args.purchase_count,
            unit_cost=args.unit_cost,
        )
        evaluated_total += 1
        evaluated_this_run += 1
        last_iteration = iteration

        parent_pool.append(candidate)
        parent_pool = trim_parent_pool(parent_pool + [best_genome, baseline_genome], args.parent_pool_size)
        improved = score_key(metrics) > score_key(best_metrics)
        if improved:
            best_genome = candidate
            best_metrics = metrics
            stagnation = 0
        else:
            stagnation += 1

        ranks = rank_counts(metrics)
        append_history(
            args.history,
            {
                "created_at": now_iso(),
                "iteration": iteration,
                "genome_id": candidate.id,
                "objective_score": candidate.score,
                "roi_percent": metrics.get("roi_percent"),
                "profit": metrics.get("profit"),
                "grade_hit_count": metrics.get("grade_hit_count"),
                "high_grade_hit_count": metrics.get("high_grade_hit_count"),
                "max_main_match": metrics.get("max_main_match"),
                "rank_3": ranks.get("3等", 0),
                "rank_4": ranks.get("4等", 0),
                "rank_5": ranks.get("5等", 0),
                "rank_6": ranks.get("6等", 0),
                "best_objective_score": composite_score(best_metrics),
                "best_roi_percent": best_metrics.get("roi_percent"),
                "best_profit": best_metrics.get("profit"),
            },
        )

        save_state(
            args.state,
            args=args,
            status="running",
            target_draws=len(target_indices),
            last_iteration=last_iteration,
            evaluated_total=evaluated_total,
            best_genome=best_genome,
            best_metrics=best_metrics,
            baseline_genome=baseline_genome,
            baseline_metrics=baseline_metrics,
            parent_pool=parent_pool,
            rng=rng,
        )

        if improved:
            adopted_now, _reasons, _summary = write_candidate_outputs(
                args=args,
                best_genome=best_genome,
                baseline_metrics=baseline_metrics,
                best_metrics=best_metrics,
                status="running",
                resume_used=resume_used,
                start_iteration=start_iteration,
                last_iteration=last_iteration,
                evaluated_total=evaluated_total,
                evaluated_this_run=evaluated_this_run,
                target_draws=len(target_indices),
            )
            checkpoint_commit(args, iteration=iteration, adopted=adopted_now)

    if last_iteration >= args.iterations:
        status = "completed"

    adopted, reasons, summary = write_candidate_outputs(
        args=args,
        best_genome=best_genome,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        status=status,
        resume_used=resume_used,
        start_iteration=start_iteration,
        last_iteration=last_iteration,
        evaluated_total=evaluated_total,
        evaluated_this_run=evaluated_this_run,
        target_draws=len(target_indices),
    )

    save_state(
        args.state,
        args=args,
        status=status,
        target_draws=len(target_indices),
        last_iteration=last_iteration,
        evaluated_total=evaluated_total,
        best_genome=best_genome,
        best_metrics=best_metrics,
        baseline_genome=baseline_genome,
        baseline_metrics=baseline_metrics,
        parent_pool=parent_pool,
        rng=rng,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
