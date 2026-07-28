#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for the high-match-first self evolver.

The implementation lives in ``_loto7_model_self_evolver_impl.py``. Seed models
may contain scores created by older ROI-centered objectives; those persisted
scores are cleared before the new learning run starts.

Direct replacement of ``loto7_best_model.json`` is denied by default. Automated
promotion is owned by Generation 5. Legacy manual application requires both the
``--apply`` argument and ``LOTO7_ALLOW_LEGACY_DIRECT_APPLY=1``.
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


def _authorized_argv(argv: Optional[List[str]] = None) -> List[str]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--apply" not in arguments:
        return arguments
    allowed = os.environ.get("LOTO7_ALLOW_LEGACY_DIRECT_APPLY", "").strip().lower()
    if allowed in {"1", "true", "yes"}:
        print("[AUTH] explicit legacy direct-apply authorization accepted")
        return arguments
    arguments = [argument for argument in arguments if argument != "--apply"]
    print(
        "[AUTH] --apply ignored: automatic model promotion is owned by Generation 5; "
        "set LOTO7_ALLOW_LEGACY_DIRECT_APPLY=1 only for an explicit manual override",
        file=sys.stderr,
    )
    return arguments


def main(argv: Optional[List[str]] = None) -> int:
    return _impl.main(_authorized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
