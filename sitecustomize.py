"""Runtime guards for LOTO7 workflow compatibility and checkpoint evidence."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
from pathlib import Path

ADOPTED_BEST_MODEL_PATTERNS = ["loto7_best_model.json"]


def _set_option(flag: str, value: str) -> None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        sys.argv.extend([flag, value])
        return
    value_index = index + 1
    if value_index < len(sys.argv):
        sys.argv[value_index] = value
    else:
        sys.argv.append(value)


def _has_option(flag: str) -> bool:
    return flag in sys.argv


def _add_option_values(flag: str, values: list[str]) -> None:
    if not _has_option(flag):
        sys.argv.extend([flag, *values])


def _patch_merge_evolution_args(script_name: str) -> None:
    if script_name != "merge_evolution_shards.py":
        return
    _set_option("--ensemble-candidates-per-model", "12")
    _set_option("--selection-mode", "holdout_roi")
    _add_option_values("--patterns", ADOPTED_BEST_MODEL_PATTERNS)


def _patch_model_self_evolver_args(script_name: str) -> None:
    if script_name == "loto7_model_self_evolver.py" and not _has_option("--max-targets"):
        _set_option("--max-targets", "0")


def _write_json(path: str, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _register_generation5_checkpoint(script_name: str) -> None:
    if script_name != "generation5_evolver.py":
        return

    def finalize() -> None:
        candidate = Path("outputs/generation5/generation5_candidate_model.json")
        summary = Path("outputs/generation5/generation5_summary.json")
        dataset = Path("loto7.csv")
        baseline = Path("loto7_best_model.json")
        payload: dict[str, object] = {
            "kind": "loto7_generation5_checkpoint",
            "objective_version": "loto7-generation5-checkpoint-2026.07.31-v1",
            "candidate_exists": candidate.exists() and candidate.stat().st_size > 0,
            "summary_exists": summary.exists() and summary.stat().st_size > 0,
        }
        if dataset.exists():
            payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if baseline.exists():
            payload["baseline_model_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
        if summary.exists() and summary.stat().st_size:
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
                payload["completed_generation"] = data.get("generations_completed")
                payload["status"] = data.get("status")
            except (OSError, ValueError, TypeError):
                payload["status"] = "summary_unreadable"
        _write_json("outputs/state/generation5/checkpoint.json", payload)

    atexit.register(finalize)


def _patch_args() -> None:
    script_name = os.path.basename(sys.argv[0] or "")
    _patch_merge_evolution_args(script_name)
    _patch_model_self_evolver_args(script_name)
    _register_generation5_checkpoint(script_name)


_patch_args()
