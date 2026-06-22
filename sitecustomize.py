"""Runtime guard for LOTO7 GitHub Actions CLI arguments.

Python imports ``sitecustomize`` automatically when it is available on
``sys.path``. This guard is tightly scoped to LOTO7 workflow entry points
and leaves unrelated commands unchanged.
"""

from __future__ import annotations

import os
import sys


SELF_EVOLVED_MODEL_PATTERNS = [
    "loto7_best_model.json",
    "outputs/model_self_evolution/best_candidate_model.json",
]


def _set_option(flag: str, value: str) -> None:
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        sys.argv.extend([flag, value])
        return

    value_index = idx + 1
    if value_index < len(sys.argv):
        sys.argv[value_index] = value
    else:
        sys.argv.append(value)


def _has_option(flag: str) -> bool:
    return flag in sys.argv


def _add_option_values(flag: str, values: list[str]) -> None:
    if _has_option(flag):
        return
    sys.argv.extend([flag, *values])


def _patch_merge_evolution_args(script_name: str) -> None:
    if script_name != "merge_evolution_shards.py":
        return

    # Keep the Actions merge path aligned with the latest self-evolution config.
    _set_option("--ensemble-candidates-per-model", "12")

    # Prefer the built-in holdout ROI ranking path for model selection.
    _set_option("--selection-mode", "holdout_roi")

    # Keep model-only self-evolution outputs in the candidate pool.
    _add_option_values("--patterns", SELF_EVOLVED_MODEL_PATTERNS)


def _patch_model_self_evolver_args(script_name: str) -> None:
    if script_name != "loto7_model_self_evolver.py":
        return

    # Default to full-history evaluation when max-targets is not specified.
    if not _has_option("--max-targets"):
        _set_option("--max-targets", "0")


def _patch_args() -> None:
    script_name = os.path.basename(sys.argv[0] or "")
    _patch_merge_evolution_args(script_name)
    _patch_model_self_evolver_args(script_name)


_patch_args()
