#!/usr/bin/env python3
"""Run Generation 5 in resumable generation-sized stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from scripts.generation5_evolver import main as run_generation5

OBJECTIVE_VERSION = "loto7-generation5-checkpoint-2026.07.31-v2"

_MANAGED_EVOLVER_OPTIONS = {
    "--csv",
    "--best-model",
    "--seed-patterns",
    "--candidate-model",
    "--summary",
    "--report",
    "--history",
    "--generations",
    "--island-population",
    "--max-runtime-minutes",
    "--safe-exit-minutes",
    "--seed",
}


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stage_seed(
    dataset_sha: str,
    baseline_sha: str,
    generation: int,
    base_seed: int,
) -> int:
    material = f"{dataset_sha}:{baseline_sha}:{generation}:{base_seed}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:15], 16)


def validate_passthrough(arguments: Sequence[str]) -> None:
    conflicts = sorted(
        {
            token.split("=", 1)[0]
            for token in arguments
            if token.startswith("--")
            and token.split("=", 1)[0] in _MANAGED_EVOLVER_OPTIONS
        }
    )
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(f"checkpoint runner manages these options directly: {joined}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--baseline", default="loto7_best_model.json")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--island-population", type=int, default=4)
    parser.add_argument("--checkpoint", default="outputs/state/generation5/checkpoint.json")
    parser.add_argument(
        "--candidate", default="outputs/generation5/generation5_candidate_model.json"
    )
    parser.add_argument("--summary", default="outputs/generation5/generation5_summary.json")
    parser.add_argument("--report", default="outputs/generation5/generation5_report.txt")
    parser.add_argument("--history", default="outputs/generation5/generation5_history.csv")
    parser.add_argument("--stages-dir", default="outputs/generation5/stages")
    parser.add_argument("--max-runtime-minutes-per-stage", type=float, default=75.0)
    parser.add_argument("--base-seed", type=int, default=0)
    args, passthrough = parser.parse_known_args(argv)
    validate_passthrough(passthrough)

    if args.generations <= 0 or args.island_population <= 0:
        raise SystemExit("generations and island population must be positive")
    if args.max_runtime_minutes_per_stage <= 10.0:
        raise SystemExit("max runtime per stage must be greater than the 10 minute safe-exit window")

    dataset_sha = sha256(args.csv)
    baseline_sha = sha256(args.baseline)
    checkpoint_path = Path(args.checkpoint)
    checkpoint: dict[str, object] = {}
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
    canonical_candidate = Path(args.candidate)
    previous_candidate = str(checkpoint.get("candidate", "") or "")
    if previous_candidate and not Path(previous_candidate).exists():
        checkpoint_canonical = Path(
            str(checkpoint.get("canonical_candidate", args.candidate) or args.candidate)
        )
        previous_candidate = str(checkpoint_canonical) if checkpoint_canonical.exists() else ""

    if completed >= args.generations and canonical_candidate.exists():
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "completed_generation": completed,
                    "candidate": str(canonical_candidate),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    stages = Path(args.stages_dir)
    stages.mkdir(parents=True, exist_ok=True)
    for generation in range(completed + 1, args.generations + 1):
        stage_candidate = stages / f"generation_{generation:03d}_candidate.json"
        stage_summary = stages / f"generation_{generation:03d}_summary.json"
        stage_report = stages / f"generation_{generation:03d}_report.txt"
        resumed_from = previous_candidate
        seeds = [args.baseline]
        if previous_candidate and Path(previous_candidate).exists():
            seeds.append(previous_candidate)
        for optional in (
            "outputs/recent_era/recent_era_best_model.json",
            "outputs/super_recent/super_recent_best_model.json",
        ):
            if Path(optional).exists():
                seeds.append(optional)

        current_seed = stage_seed(
            dataset_sha,
            baseline_sha,
            generation,
            args.base_seed,
        )
        evolver_argv = [
            "--csv",
            args.csv,
            "--best-model",
            args.baseline,
            "--seed-patterns",
            *seeds,
            "--candidate-model",
            str(stage_candidate),
            "--summary",
            str(stage_summary),
            "--report",
            str(stage_report),
            "--history",
            args.history,
            "--generations",
            "1",
            "--island-population",
            str(args.island_population),
            "--seed",
            str(current_seed),
            "--max-runtime-minutes",
            str(args.max_runtime_minutes_per_stage),
            "--safe-exit-minutes",
            "10",
            *passthrough,
        ]
        if run_generation5(evolver_argv) != 0:
            raise RuntimeError(f"Generation 5 stage failed: {generation}")

        canonical_candidate.parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage_candidate, canonical_candidate)
        shutil.copyfile(stage_summary, args.summary)
        shutil.copyfile(stage_report, args.report)

        summary = read_json(Path(args.summary))
        summary["checkpoint"] = {
            "objective_version": OBJECTIVE_VERSION,
            "completed_generation": generation,
            "requested_generations": args.generations,
            "stage_seed": current_seed,
            "resumed_from": resumed_from or None,
            "canonical_candidate": str(canonical_candidate),
        }
        write_json(Path(args.summary), summary)

        previous_candidate = str(canonical_candidate)
        checkpoint = {
            "kind": "loto7_generation5_checkpoint",
            "objective_version": OBJECTIVE_VERSION,
            "dataset_sha256": dataset_sha,
            "baseline_model_sha256": baseline_sha,
            "completed_generation": generation,
            "requested_generations": args.generations,
            "candidate": str(canonical_candidate),
            "stage_candidate": str(stage_candidate),
            "canonical_candidate": str(canonical_candidate),
            "canonical_summary": args.summary,
            "canonical_report": args.report,
            "history": args.history,
            "stage_seed": current_seed,
            "passthrough_arguments": list(passthrough),
        }
        write_json(checkpoint_path, checkpoint)
        print(json.dumps(checkpoint, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
