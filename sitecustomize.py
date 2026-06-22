"""Runtime guard for LOTO7 GitHub Actions CLI arguments.

Python imports ``sitecustomize`` automatically when it is available on
``sys.path``. This guard is tightly scoped to ``merge_evolution_shards.py``
and leaves other commands unchanged.
"""

from __future__ import annotations

import os
import sys


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


def _patch_merge_evolution_args() -> None:
    script_name = os.path.basename(sys.argv[0] or "")
    if script_name != "merge_evolution_shards.py":
        return

    # Keep the Actions merge path aligned with the latest self-evolution config.
    _set_option("--ensemble-candidates-per-model", "12")

    # Prefer the built-in holdout ROI ranking path for model selection.
    _set_option("--selection-mode", "holdout_roi")


_patch_merge_evolution_args()
