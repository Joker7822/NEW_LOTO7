#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_safe_evolution_runner.py

loto7_evolution_trainer.py を安全に起動するためのラッパー。

目的:
    - --resume 時に state 内の前回引数と現在引数を照合する
    - 条件不一致のまま過去結果と新規結果が混ざる事故を防ぐ
    - push系デフォルトを安全側に倒す
    - 実体の学習ロジックは既存の loto7_evolution_trainer.py をそのまま使う

例:
    python loto7_safe_evolution_runner.py \
      --resume \
      --generations 100 \
      --population 100 \
      --elite-count 10 \
      --purchase-count 5 \
      --num-shards 4 \
      --shard-id 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# generations は「何世代まで進めるか」の上限なので、resume時に増やす運用を許可する。
# 以下の検証条件が変わると過去評価と新規評価が混ざるため停止する。
CHECK_KEYS = [
    "population",
    "elite_count",
    "purchase_count",
    "min_train_draws",
    "max_targets",
    "target_stride",
    "seed",
    "shard_id",
    "num_shards",
]


def default_state_path(output_dir: str, shard_id: int, num_shards: int) -> str:
    return str(Path(output_dir) / f"evolution_state_shard{shard_id:02d}_of_{num_shards:02d}.json")


def load_state(path: str) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        raise SystemExit(f"failed to read state file: {path} ({exc})")


def validate_resume_args(state: Dict[str, object], args: argparse.Namespace) -> None:
    saved = state.get("args")
    if not isinstance(saved, dict):
        raise SystemExit("state file does not contain args; refuse unsafe resume")

    mismatches = []
    for key in CHECK_KEYS:
        previous = saved.get(key)
        current = getattr(args, key)
        if str(previous) != str(current):
            mismatches.append((key, previous, current))

    saved_generation = int(state.get("generation", 0))
    if args.generations <= saved_generation:
        raise SystemExit(
            f"--generations must be greater than saved generation when resuming: "
            f"saved_generation={saved_generation}, generations={args.generations}"
        )

    if mismatches:
        lines = ["Resume条件が前回と一致しません。安全のため停止します。"]
        for key, previous, current in mismatches:
            lines.append(f"- {key}: previous={previous}, current={current}")
        lines.append("条件を変更したい場合は --resume を外すか、別の --state-path を指定してください。")
        raise SystemExit("\n".join(lines))


def build_trainer_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "loto7_evolution_trainer.py",
        "--csv", args.csv,
        "--output-dir", args.output_dir,
        "--best-model", args.best_model,
        "--shard-id", str(args.shard_id),
        "--num-shards", str(args.num_shards),
        "--workers", str(args.workers),
        "--push-every-genome", str(args.push_every_genome),
        "--max-runtime-minutes", str(args.max_runtime_minutes),
        "--safe-exit-minutes", str(args.safe_exit_minutes),
        "--generations", str(args.generations),
        "--population", str(args.population),
        "--elite-count", str(args.elite_count),
        "--purchase-count", str(args.purchase_count),
        "--min-train-draws", str(args.min_train_draws),
        "--max-targets", str(args.max_targets),
        "--target-stride", str(args.target_stride),
        "--seed", str(args.seed),
        "--push-every-generation", str(args.push_every_generation),
    ]
    if args.state_path:
        cmd.extend(["--state-path", args.state_path])
    if args.resume:
        cmd.append("--resume")
    if args.push_final:
        cmd.append("--push-final")
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Safe wrapper for loto7_evolution_trainer.py")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--best-model", default="loto7_best_model.json")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-every-genome", type=int, default=0)
    parser.add_argument("--max-runtime-minutes", type=int, default=330)
    parser.add_argument("--safe-exit-minutes", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--purchase-count", type=int, default=5)
    parser.add_argument("--min-train-draws", type=int, default=60)
    parser.add_argument("--max-targets", default="all")
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--push-every-generation", type=int, default=0, help="安全側デフォルト。必要な時だけ1以上にする。")
    parser.add_argument("--push-final", action="store_true")
    args = parser.parse_args(argv)

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.population < 4:
        raise SystemExit("--population must be >= 4")
    if args.elite_count < 1 or args.elite_count >= args.population:
        raise SystemExit("--elite-count must be >=1 and < population")
    if args.purchase_count <= 0:
        raise SystemExit("--purchase-count must be positive")
    if args.safe_exit_minutes >= args.max_runtime_minutes:
        raise SystemExit("--safe-exit-minutes must be smaller than --max-runtime-minutes")

    state_path = args.state_path or default_state_path(args.output_dir, args.shard_id, args.num_shards)
    if args.resume:
        state = load_state(state_path)
        if state is None:
            raise SystemExit(f"--resume was specified but state file was not found: {state_path}")
        validate_resume_args(state, args)

    cmd = build_trainer_command(args)
    print("[SAFE RUN]", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
