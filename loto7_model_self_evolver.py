#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for the high-match-first self evolver.

The implementation lives in ``_loto7_model_self_evolver_impl.py``. Seed models
may contain scores created by older ROI-centered objectives; those persisted
scores are cleared before the new learning run starts.

Direct replacement of ``loto7_best_model.json`` is denied by default. Automated
promotion is owned by Generation 5. Legacy manual application requires ``--apply``
and either an explicit authorization environment variable or a workflow-dispatch
run of the standalone ``LOTO7 Model Self Evolution`` workflow.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import _loto7_model_self_evolver_impl as _impl
from _loto7_model_self_evolver_impl import *  # noqa: F401,F403

_original_load_seed_genomes = _impl.load_seed_genomes


def _load_high_match_seed_genomes(patterns):
    seeds = _original_load_seed_genomes(patterns)
    for _path, genome in seeds:
        genome.score = 0.0
        genome.max_main_match = 0
        genome.best_rank_count = 0
    return seeds


_impl.load_seed_genomes = _load_high_match_seed_genomes


def _legacy_apply_authorized() -> bool:
    explicit = os.environ.get(
        "LOTO7_ALLOW_LEGACY_DIRECT_APPLY", ""
    ).strip().lower() in {"1", "true", "yes"}
    standalone_dispatch = (
        os.environ.get("GITHUB_EVENT_NAME", "") == "workflow_dispatch"
        and os.environ.get("GITHUB_WORKFLOW", "")
        == "LOTO7 Model Self Evolution"
    )
    return explicit or standalone_dispatch


def _authorized_argv(argv: Optional[List[str]] = None) -> List[str]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--apply" not in arguments:
        return arguments
    if _legacy_apply_authorized():
        print("[AUTH] explicit legacy direct-apply authorization accepted")
        return arguments
    arguments = [argument for argument in arguments if argument != "--apply"]
    print(
        "[AUTH] --apply ignored: automatic model promotion is owned by Generation 5; "
        "use the standalone Model Self Evolution workflow dispatch or set "
        "LOTO7_ALLOW_LEGACY_DIRECT_APPLY=1 for an explicit manual override",
        file=sys.stderr,
    )
    return arguments


def main(argv: Optional[List[str]] = None) -> int:
    return _impl.main(_authorized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
