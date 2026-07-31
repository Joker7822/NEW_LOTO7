#!/usr/bin/env python3
"""Run Generation 5 in resumable generation-sized stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

from scripts.generation5_evolver import main as run_generation5

OBJECTIVE_VERSION = "loto7-generation5-checkpoint-2026.07.31-v1"


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline", default="loto7_best_model.json")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--island-population", type=int, default=4)
    parser.add_argument("--checkpoint", default="outputs/state/generation5/checkpoint.json")
    parser.add_argument("--candidate", default="outputs/generation5/generation5_candidate_model.json")
    parser.add_argument("--summary", default="outputs/generation5/generation5_summary.json")
    parser.add_argument("--report", default="outputs/generation5/generation5_report.txt")
    parser.add_argument("--history", default="outputs/generation5/generation5_history.csv")
    parser.add_argument("--max-runtime-minutes-per-stage", type=float, default=75.0)
    args = parser.parse_args()
    if args.generations <= 0 or args.island_population <= 0:
        raise SystemExit("generations and island population must be positive")

    dataset_sha = sha256(args.csv)
    baseline_sha = sha256(args.baseline)
    checkpoint_path = Path(args.checkpoint)
    checkpoint: Dict[str, object] = {}
    if checkpoint_path.exists() and checkpoint_path.stat().st_size:
        checkpoint = read_json(checkpoint_path)
        valid = (
            checkpoint.get("objective_version") == OBJECTIVE_VERSION
            and checkpoint.get("dataset_sha256") == dataset_sha
            and checkpoint.get("baseline_model_sha256") == baseline_sha
        )
        if not valid:
            checkpoint = {}
    completed = int(checkpoint.get("completed_generation", 0) or 0)
    previous_candidate = str(checkpoint.get("candidate", "") or "")
    if completed >= args.generations and Path(args.candidate).exists():
        print(json.dumps({"status": "already_complete", "completed_generation": completed}))
        return 0

    stages = Path("outputs/generation5/stages")
    stages.mkdir(parents=True, exist_ok=True)
    for generation in range(completed + 1, args.generations + 1):
        stage_candidate = stages / f"generation_{generation:03d}_candidate.json"
        stage_summary = stages / f"generation_{generation:03d}_summary.json"
        stage_report = stages / f"generation_{generation:03d}_report.txt"
        seeds = [args.baseline]
        if previous_candidate and Path(previous_candidate).exists():
            seeds.append(previous_candidate)
        for optional in (
            "outputs/recent_era/recent_era_best_model.json",
            "outputs/super_recent/super_recent_best_model.json",
        ):
            if Path(optional).exists():
                seeds.append(optional)
        argv = [
            "--csv", args.csv,
            "--best-model", args.baseline,
            "--seed-patterns", *seeds,
            "--candidate-model", str(stage_candidate),
            "--summary", str(stage_summary),
            "--report", str(stage_report),
            "--history", args.history,
            "--generations", "1",
            "--island-population", str(args.island_population),
            "--max-runtime-minutes", str(args.max_runtime_minutes_per_stage),
            "--safe-exit-minutes", "10",
        ]
        if run_generation5(argv) != 0:
            raise RuntimeError(f"Generation 5 stage failed: {generation}")
        shutil.copyfile(stage_candidate, args.candidate)
        shutil.copyfile(stage_summary, args.summary)
        shutil.copyfile(stage_report, args.report)
        previous_candidate = str(stage_candidate)
        checkpoint = {
            "kind": "loto7_generation5_checkpoint",
            "objective_version": OBJECTIVE_VERSION,
            "dataset_sha256": dataset_sha,
            "baseline_model_sha256": baseline_sha,
            "completed_generation": generation,
            "requested_generations": args.generations,
            "candidate": previous_candidate,
            "canonical_candidate": args.candidate,
            "canonical_summary": args.summary,
        }
        write_json(checkpoint_path, checkpoint)
        print(json.dumps(checkpoint, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
