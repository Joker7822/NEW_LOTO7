#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-match-first LOTO7 model self evolution.

The learning objective rewards main-number match quality, temporal robustness,
and five-ticket diversity. ROI, profit and payout concentration are not learning
rewards; they are used only as independent adoption safety gates.
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
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.evolution.hit_first import (  # noqa: E402
    OBJECTIVE_NAME,
    OBJECTIVE_VERSION,
    adoption_decision,
    evaluate_model_on_holdout,
    hit_first_key,
    hit_first_score,
)
from loto7_evolution_trainer import (  # noqa: E402
    Genome,
    crossover,
    genome_from_dict,
    load_draws,
    mutate,
    random_genome,
)
from merge_evolution_shards import load_prize_rows, select_target_indices  # noqa: E402

DEFAULT_SEED_PATTERNS = ["loto7_best_model.json"]

HISTORY_FIELDS = [
    "created_at",
    "iteration",
    "genome_id",
    "objective_name",
    "objective_version",
    "hit_first_objective_score",
    "match_quality_score",
    "temporal_segment_match_score_min",
    "temporal_segment_match_score_median",
    "diversity_quality_score",
    "average_max_main_match",
    "draw_main4_plus_rate_percent",
    "draw_main5_plus_count",
    "draw_main6_plus_count",
    "average_portfolio_unique_numbers",
    "mean_ticket_pair_overlap",
    "payout_roi_percent",
    "profit",
    "top1_payout_share",
    "best_hit_first_objective_score",
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    found: List[Tuple[str, Genome]] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            genome = load_genome_file(path)
            if genome is None or genome.id in seen:
                continue
            seen.add(genome.id)
            found.append((path, genome))
    return found


def assign_score(genome: Genome, metrics: Dict[str, object]) -> Genome:
    genome.score = hit_first_score(metrics)
    genome.max_main_match = int(metrics.get("max_main_match", 0) or 0)
    genome.best_rank_count = int(metrics.get("draw_main5_plus_count", 0) or 0)
    return genome


def trim_parent_pool(parent_pool: Sequence[Genome], limit: int) -> List[Genome]:
    unique: Dict[str, Genome] = {}
    for genome in parent_pool:
        current = unique.get(genome.id)
        if current is None or float(genome.score) > float(current.score):
            unique[genome.id] = genome
    return sorted(unique.values(), key=lambda item: float(item.score), reverse=True)[: max(1, limit)]


def exploratory_mutate(genome: Genome, iteration: int, rng: random.Random, intensity: int = 3) -> Genome:
    candidate = genome
    for index in range(max(1, intensity)):
        candidate = mutate(candidate, iteration, iteration * 10 + index, rng)
    return candidate


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
        return exploratory_mutate(
            rng.choice(parents),
            iteration,
            rng,
            intensity=rng.choice([2, 3, 4, 5]),
        )
    if len(parents) >= 2 and rng.random() < 0.55:
        left, right = rng.sample(parents, 2)
        return crossover(left, right, iteration, iteration, rng)
    return mutate(rng.choice(parents), iteration, iteration, rng)


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
        state.get("objective_version") == OBJECTIVE_VERSION
        and state.get("csv") == args.csv
        and int(state.get("purchase_count", -1)) == int(args.purchase_count)
        and int(state.get("unit_cost", -1)) == int(args.unit_cost)
        and int(state.get("target_draws", -1)) == int(target_draws)
    )


def load_resume_state(path: str, args: argparse.Namespace, target_draws: int) -> Optional[Dict[str, object]]:
    state_path = Path(path)
    if not state_path.exists() or state_path.stat().st_size <= 0:
        return None
    try:
        state = read_json(path)
    except Exception as exc:
        print(f"[WARN] cannot read state {path}: {exc}")
        return None
    if not state_matches(state, args, target_draws):
        print("[INFO] previous state uses different settings/objective; starting high-match learning fresh")
        return None
    if int(state.get("last_iteration", 0)) >= args.iterations:
        return None
    if str(state.get("status", "")) == "completed":
        return None
    return state


def genomes_from_state(raw_items: object) -> List[Genome]:
    if not isinstance(raw_items, list):
        return []
    result: List[Genome] = []
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
        result.append(genome)
    return result


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
    write_json(
        path,
        {
            "updated_at": now_iso(),
            "status": status,
            "objective": OBJECTIVE_NAME,
            "objective_version": OBJECTIVE_VERSION,
            "csv": args.csv,
            "purchase_count": args.purchase_count,
            "unit_cost": args.unit_cost,
            "target_draws": target_draws,
            "iterations_requested": args.iterations,
            "last_iteration": last_iteration,
            "iterations_evaluated": evaluated_total,
            "best_genome": asdict(best_genome),
            "best_metrics": best_metrics,
            "baseline_genome": asdict(baseline_genome),
            "baseline_metrics": baseline_metrics,
            "parent_pool": [asdict(item) for item in trim_parent_pool(parent_pool, args.parent_pool_size)],
            "rng_state": encode_rng_state(rng),
        },
    )


def prepare_history(path: str) -> None:
    target = Path(path)
    if not target.exists() or target.stat().st_size <= 0:
        return
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            existing = next(csv.reader(handle), [])
    except Exception:
        existing = []
    if existing == HISTORY_FIELDS:
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    legacy = target.with_name(f"{target.stem}.pre_hit_first_{stamp}{target.suffix}")
    target.replace(legacy)
    print(f"[MIGRATE] preserved old history as {legacy}")


def append_history(path: str, row: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists() and target.stat().st_size > 0
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def evaluate_candidate(
    *,
    genome: Genome,
    model_path: str,
    draws: Sequence[object],
    prize_rows: Dict[int, Dict[str, str]],
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
    return assign_score(genome, metrics), metrics


def candidate_payload(
    *,
    genome: Genome,
    baseline_metrics: Dict[str, object],
    selected_metrics: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "updated_at": now_iso(),
        "kind": "loto7_model_only_self_evolution",
        "source": "model_self_evolver",
        "csv": args.csv,
        "purchase_count": args.purchase_count,
        "selection_mode": "hit_first_temporal_robustness_then_financial_safety",
        "objective": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "objective_score": hit_first_score(selected_metrics),
        "baseline_metrics": baseline_metrics,
        "selected_holdout": selected_metrics,
        "genome": asdict(genome),
        "notes": [
            "ROI and profit do not contribute to the learning score.",
            "Main-number 4+/5+/6+ reach, temporal robustness and portfolio diversity drive selection.",
            "Financial values are independent adoption safety gates only.",
            "This does not guarantee future lottery winnings or profit.",
        ],
    }


def adoption_for(args: argparse.Namespace, best: Dict[str, object], baseline: Dict[str, object]) -> Tuple[bool, List[str]]:
    return adoption_decision(
        best,
        baseline,
        min_objective_delta=args.min_hit_objective_delta,
        min_draw4_rate_delta_percent=args.min_draw4_rate_delta_percent,
        min_draw5_count_delta=args.min_draw5_count_delta,
        min_average_max_delta=args.min_average_max_delta,
        min_temporal_min_delta=args.min_temporal_min_delta,
        min_payout_roi_percent=args.min_payout_roi_percent,
        max_roi_drop_percent=args.max_roi_drop_percent,
        max_top1_payout_share=args.max_top1_payout_share,
    )


def build_summary(
    *,
    args: argparse.Namespace,
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
        "objective": OBJECTIVE_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "parent_pool_size": args.parent_pool_size,
        "exploration_rate": args.exploration_rate,
        "random_parents": args.random_parents,
        "baseline_objective_score": hit_first_score(baseline_metrics),
        "best_objective_score": hit_first_score(best_metrics),
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "candidate_model": args.candidate_model,
        "best_model": args.best_model,
        "adoption": {
            "adopted": adopted,
            "apply_requested": bool(args.apply),
            "applied": bool(adopted and args.apply),
            "reasons": reasons,
            "min_hit_objective_delta": args.min_hit_objective_delta,
            "min_draw4_rate_delta_percent": args.min_draw4_rate_delta_percent,
            "min_draw5_count_delta": args.min_draw5_count_delta,
            "min_average_max_delta": args.min_average_max_delta,
            "min_temporal_min_delta": args.min_temporal_min_delta,
            "min_payout_roi_percent": args.min_payout_roi_percent,
            "max_roi_drop_percent": args.max_roi_drop_percent,
            "max_top1_payout_share": args.max_top1_payout_share,
            "legacy_min_roi_delta_percent_ignored_as_learning_reward": args.min_roi_delta_percent,
            "legacy_min_profit_delta_ignored_as_learning_reward": args.min_profit_delta,
        },
    }


def write_report(path: str, summary: Dict[str, object]) -> None:
    baseline = summary.get("baseline_metrics", {})
    best = summary.get("best_metrics", {})
    lines = [
        "LOTO7 High-Match-First Self Evolution Report",
        "==============================================",
        "",
        f"created_at: {summary.get('created_at')}",
        f"status: {summary.get('status')}",
        f"objective: {summary.get('objective')}",
        f"objective_version: {summary.get('objective_version')}",
        f"resume_used: {summary.get('resume_used')}",
        f"iterations_evaluated: {summary.get('iterations_evaluated')}",
        f"target_draws: {summary.get('target_draws')}",
        "",
        "[Baseline]",
        f"hit_first_objective_score: {hit_first_score(baseline) if isinstance(baseline, dict) else ''}",
        f"average_max_main_match: {baseline.get('average_max_main_match') if isinstance(baseline, dict) else ''}",
        f"draw_main4_plus_rate_percent: {baseline.get('draw_main4_plus_rate_percent') if isinstance(baseline, dict) else ''}",
        f"draw_main5_plus_count: {baseline.get('draw_main5_plus_count') if isinstance(baseline, dict) else ''}",
        f"worst_segment_score: {baseline.get('temporal_segment_match_score_min') if isinstance(baseline, dict) else ''}",
        f"payout_roi_percent(safety only): {baseline.get('payout_roi_percent') if isinstance(baseline, dict) else ''}",
        "",
        "[Best Candidate]",
        f"hit_first_objective_score: {hit_first_score(best) if isinstance(best, dict) else ''}",
        f"average_max_main_match: {best.get('average_max_main_match') if isinstance(best, dict) else ''}",
        f"draw_main4_plus_rate_percent: {best.get('draw_main4_plus_rate_percent') if isinstance(best, dict) else ''}",
        f"draw_main5_plus_count: {best.get('draw_main5_plus_count') if isinstance(best, dict) else ''}",
        f"worst_segment_score: {best.get('temporal_segment_match_score_min') if isinstance(best, dict) else ''}",
        f"payout_roi_percent(safety only): {best.get('payout_roi_percent') if isinstance(best, dict) else ''}",
        "",
        "[Adoption]",
        json.dumps(summary.get("adoption", {}), ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "ROI/profit are not learning rewards. They are adoption safety checks only.",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
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
    adopted, reasons = adoption_for(args, best_metrics, baseline_metrics)
    payload = candidate_payload(
        genome=best_genome,
        baseline_metrics=baseline_metrics,
        selected_metrics=best_metrics,
        args=args,
    )
    write_json(args.candidate_model, payload)
    if adopted and args.apply:
        write_json(args.best_model, payload)
    summary = build_summary(
        args=args,
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
    )
    write_json(args.summary, summary)
    write_report(args.report, summary)
    return adopted, reasons, summary


def run_git(arguments: Sequence[str]) -> int:
    process = subprocess.run(["git", *arguments], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.stdout:
        print(process.stdout.rstrip())
    return int(process.returncode)


def checkpoint_commit(args: argparse.Namespace, *, iteration: int, adopted: bool) -> None:
    if not args.checkpoint_commit:
        return
    paths = [args.best_model, args.candidate_model, args.summary, args.report, args.history, args.state]
    existing = [path for path in paths if Path(path).exists()]
    if not existing or run_git(["add", "-f", *existing]) != 0:
        return
    if run_git(["diff", "--cached", "--quiet"]) == 0:
        return
    label = "adopted" if adopted else "candidate"
    if run_git(["commit", "-m", f"Checkpoint LOTO7 hit-first {label} at iteration {iteration} [skip ci]"]) != 0:
        return
    branch = args.checkpoint_branch or os.environ.get("GITHUB_REF_NAME", "")
    push_args = ["push", "origin", f"HEAD:{branch}"] if branch else ["push"]
    if run_git(push_args) != 0:
        print("[WARN] checkpoint push failed; final workflow commit can still save outputs")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Self-evolve LOTO7 Genome using high-match-first learning.")
    result.add_argument("--csv", default="loto7.csv")
    result.add_argument("--best-model", default="loto7_best_model.json")
    result.add_argument("--seed-patterns", nargs="*", default=DEFAULT_SEED_PATTERNS)
    result.add_argument("--output-dir", default="outputs/model_self_evolution")
    result.add_argument("--candidate-model", default="outputs/model_self_evolution/best_candidate_model.json")
    result.add_argument("--summary", default="outputs/model_self_evolution/summary.json")
    result.add_argument("--report", default="outputs/model_self_evolution/report.txt")
    result.add_argument("--history", default="outputs/model_self_evolution/history.csv")
    result.add_argument("--state", default="outputs/model_self_evolution/state.json")
    result.add_argument("--iterations", type=int, default=1000)
    result.add_argument("--purchase-count", type=int, default=5)
    result.add_argument("--unit-cost", type=int, default=300)
    result.add_argument("--min-train-draws", type=int, default=1)
    result.add_argument("--holdout-start-draw", type=int, default=2)
    result.add_argument("--holdout-end-draw", type=int, default=None)
    result.add_argument("--target-stride", type=int, default=1)
    result.add_argument("--max-targets", type=int, default=0)
    result.add_argument("--seed", type=int, default=777)
    result.add_argument("--apply", action="store_true")
    result.add_argument("--resume", dest="resume", action="store_true", default=True)
    result.add_argument("--no-resume", dest="resume", action="store_false")
    result.add_argument("--max-runtime-minutes", type=float, default=65.0)
    result.add_argument("--safe-exit-minutes", type=float, default=5.0)
    result.add_argument("--checkpoint-commit", action="store_true")
    result.add_argument("--checkpoint-branch", default="")
    result.add_argument("--parent-pool-size", type=int, default=36)
    result.add_argument("--random-parents", type=int, default=12)
    result.add_argument("--exploration-rate", type=float, default=0.30)
    result.add_argument("--stagnation-patience", type=int, default=50)

    # High-match adoption thresholds.
    result.add_argument("--min-hit-objective-delta", type=float, default=0.05)
    result.add_argument("--min-draw4-rate-delta-percent", type=float, default=0.0)
    result.add_argument("--min-draw5-count-delta", type=int, default=0)
    result.add_argument("--min-average-max-delta", type=float, default=0.0)
    result.add_argument("--min-temporal-min-delta", type=float, default=0.0)

    # Financial safety only. Kept separate from the learning objective.
    result.add_argument("--min-payout-roi-percent", type=float, default=8.0)
    result.add_argument("--max-roi-drop-percent", type=float, default=5.0)
    result.add_argument("--max-top1-payout-share", type=float, default=0.50)

    # Legacy CLI compatibility; these values never add learning points.
    result.add_argument("--min-roi-delta-percent", type=float, default=0.0)
    result.add_argument("--min-profit-delta", type=int, default=0)
    result.add_argument("--allow-high-grade-drop", action="store_true")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")

    prepare_history(args.history)
    started = time.monotonic()
    rng = random.Random(args.seed)
    draws = load_draws(args.csv)
    prize_rows = load_prize_rows(args.csv)
    target_indices = select_target_indices(
        draws,
        min_train_draws=args.min_train_draws,
        holdout_start_draw=args.holdout_start_draw,
        holdout_end_draw=args.holdout_end_draw,
    )[:: max(1, args.target_stride)]
    if args.max_targets > 0:
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
    parent_pool.extend(genome for _, genome in seeds if genome.id != baseline_genome.id)
    for index in range(max(0, args.random_parents)):
        if index % 2 == 0:
            parent_pool.append(exploratory_mutate(baseline_genome, index + 1, rng, rng.choice([2, 3, 4])))
        else:
            parent_pool.append(random_genome(0, index, rng))
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
                restored = genomes_from_state(state.get("parent_pool"))
                if restored:
                    parent_pool = trim_parent_pool(restored + [best_genome, baseline_genome], args.parent_pool_size)
                restore_rng_state(rng, state.get("rng_state"))
                last_iteration = int(state.get("last_iteration", 0))
                evaluated_total = int(state.get("iterations_evaluated", last_iteration))
                start_iteration = last_iteration + 1
                resume_used = True
                print(f"[RESUME] high-match learning from iteration {start_iteration}/{args.iterations}")
            except Exception as exc:
                print(f"[WARN] failed to resume state; starting fresh: {exc}")

    for iteration in range(start_iteration, args.iterations + 1):
        elapsed_minutes = (time.monotonic() - started) / 60.0
        if elapsed_minutes >= max(0.0, args.max_runtime_minutes - args.safe_exit_minutes):
            status = f"safe_exit_at_{elapsed_minutes:.2f}_minutes"
            break

        if stagnation >= args.stagnation_patience:
            parent_pool.extend(
                [
                    random_genome(iteration, iteration, rng),
                    exploratory_mutate(best_genome, iteration, rng, intensity=5),
                ]
            )
            parent_pool = trim_parent_pool(parent_pool, args.parent_pool_size)
            stagnation = 0
            print(f"[EXPLORE] injected high-match diversity at iteration {iteration}")

        candidate = make_candidate(
            parent_pool + [best_genome],
            iteration,
            rng,
            exploration_rate=args.exploration_rate,
        )
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

        parent_pool = trim_parent_pool(parent_pool + [candidate, best_genome, baseline_genome], args.parent_pool_size)
        improved = hit_first_key(metrics) > hit_first_key(best_metrics)
        if improved:
            best_genome = candidate
            best_metrics = metrics
            stagnation = 0
        else:
            stagnation += 1

        append_history(
            args.history,
            {
                "created_at": now_iso(),
                "iteration": iteration,
                "genome_id": candidate.id,
                "objective_name": OBJECTIVE_NAME,
                "objective_version": OBJECTIVE_VERSION,
                "hit_first_objective_score": hit_first_score(metrics),
                "match_quality_score": metrics.get("match_quality_score"),
                "temporal_segment_match_score_min": metrics.get("temporal_segment_match_score_min"),
                "temporal_segment_match_score_median": metrics.get("temporal_segment_match_score_median"),
                "diversity_quality_score": metrics.get("diversity_quality_score"),
                "average_max_main_match": metrics.get("average_max_main_match"),
                "draw_main4_plus_rate_percent": metrics.get("draw_main4_plus_rate_percent"),
                "draw_main5_plus_count": metrics.get("draw_main5_plus_count"),
                "draw_main6_plus_count": metrics.get("draw_main6_plus_count"),
                "average_portfolio_unique_numbers": metrics.get("average_portfolio_unique_numbers"),
                "mean_ticket_pair_overlap": metrics.get("mean_ticket_pair_overlap"),
                "payout_roi_percent": metrics.get("payout_roi_percent"),
                "profit": metrics.get("profit"),
                "top1_payout_share": metrics.get("top1_payout_share"),
                "best_hit_first_objective_score": hit_first_score(best_metrics),
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
            adopted_now, _, _ = write_outputs(
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

    _, _, summary = write_outputs(
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
