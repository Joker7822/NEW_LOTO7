#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def norm(value: object) -> str:
    return str(value if value is not None else "").strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate saved evolution state arguments before resume.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--elite-count", type=int, required=True)
    parser.add_argument("--purchase-count", type=int, required=True)
    parser.add_argument("--min-train-draws", type=int, required=True)
    parser.add_argument("--max-targets", required=True)
    parser.add_argument("--target-stride", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    path = Path(args.state)
    if not path.exists():
        raise SystemExit(f"state file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    saved = data.get("args")
    if not isinstance(saved, dict):
        raise SystemExit("state args missing")

    mismatches = []
    for key in CHECK_KEYS:
        old = saved.get(key)
        new = getattr(args, key)
        if norm(old) != norm(new):
            mismatches.append((key, old, new))

    if mismatches:
        print("resume arguments do not match saved state")
        for key, old, new in mismatches:
            print(f"{key}: saved={old} current={new}")
        raise SystemExit(1)

    print(f"resume arguments match: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
