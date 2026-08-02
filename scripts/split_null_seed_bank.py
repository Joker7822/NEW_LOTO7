#!/usr/bin/env python3
"""Split the Generation 5 master Null seed bank into physically isolated phase files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_payload(master: Mapping[str, object], phase: str, *, source_sha256: str) -> dict[str, object]:
    phases = master.get("phases")
    if not isinstance(phases, Mapping):
        raise ValueError("master seed bank phases are missing")
    raw = phases.get(phase)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"master seed bank phase is missing or empty: {phase}")
    seeds = [int(value) for value in raw]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"master seed bank phase contains duplicate seeds: {phase}")
    return {
        "kind": "loto7_fixed_null_seed_phase",
        "version": master.get("version"),
        "namespace": master.get("namespace"),
        "dataset_sha256": master.get("dataset_sha256"),
        "evaluator_version": master.get("evaluator_version"),
        "source_bank_sha256": source_sha256,
        "phase": phase,
        "seed_count": len(seeds),
        "phases": {phase: seeds},
    }


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def split_seed_bank(
    master_path: str | Path,
    *,
    selection_output: str | Path,
    final_output: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    master = read_json(master_path)
    source_hash = sha256(master_path)
    selection = phase_payload(master, "selection", source_sha256=source_hash)
    final = phase_payload(master, "final", source_sha256=source_hash)
    selection_seeds = set(selection["phases"]["selection"])  # type: ignore[index]
    final_seeds = set(final["phases"]["final"])  # type: ignore[index]
    if selection_seeds & final_seeds:
        raise ValueError("selection and final Null seeds overlap")
    write_json(selection_output, selection)
    write_json(final_output, final)
    return selection, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/generation5/null_seed_bank.json")
    parser.add_argument(
        "--selection-output",
        default="outputs/generation5/selection_null_seed_bank.json",
    )
    parser.add_argument(
        "--final-output",
        default="outputs/generation5/final_null_seed_bank.json",
    )
    args = parser.parse_args()
    selection, final = split_seed_bank(
        args.input,
        selection_output=args.selection_output,
        final_output=args.final_output,
    )
    print(
        json.dumps(
            {
                "selection_seed_count": selection["seed_count"],
                "final_seed_count": final["seed_count"],
                "selection_output": args.selection_output,
                "final_output": args.final_output,
                "physically_isolated": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
