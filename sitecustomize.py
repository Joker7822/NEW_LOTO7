"""Runtime guard for LOTO7 GitHub Actions CLI arguments.

This file is intentionally tiny and tightly scoped. Python imports
``sitecustomize`` automatically when it is available on ``sys.path``.
The guard below only touches the scheduled/Actions merge step for
``merge_evolution_shards.py`` and leaves all other commands unchanged.
"""

from __future__ import annotations

import os
import sys


def _patch_merge_evolution_candidates_per_model() -> None:
    script_name = os.path.basename(sys.argv[0] or "")
    if script_name != "merge_evolution_shards.py":
        return

    flag = "--ensemble-candidates-per-model"
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return

    value_index = idx + 1
    if value_index >= len(sys.argv):
        return

    # PR #1 self-evolution config raised ensemble.candidates_per_model to 12.
    # Keep the normal Actions merge path aligned even if an older workflow
    # invocation still passes the previous hard-coded value.
    if sys.argv[value_index] == "10":
        sys.argv[value_index] = "12"


_patch_merge_evolution_candidates_per_model()
