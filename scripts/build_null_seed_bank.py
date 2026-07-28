#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a deterministic, disjoint Null Strategy League seed bank."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loto7.evolution.generation5 import build_seed_bank, file_sha256  # noqa: E402
from loto7.evaluation.core import EVALUATOR_VERSION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument(
        "--output", default="outputs/generation5/null_seed_bank.json"
    )
    parser.add_argument("--learning", type=int, default=700)
    parser.add_argument("--selection", type=int, default=150)
    parser.add_argument("--final", type=int, default=150)
    parser.add_argument("--namespace", default="NEW_LOTO7")
    args = parser.parse_args()

    payload = build_seed_bank(
        file_sha256(args.csv),
        evaluator_version=EVALUATOR_VERSION,
        learning_count=args.learning,
        selection_count=args.selection,
        final_count=args.final,
        namespace=args.namespace,
    )
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload["csv"] = args.csv
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "phases"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
