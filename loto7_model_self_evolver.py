#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility CLI for the high-match-first self evolver.

The implementation lives in ``_loto7_model_self_evolver_impl.py``. Seed models
may contain scores created by older ROI-centered objectives; those persisted
scores are cleared before the new learning run starts.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    raise SystemExit(_impl.main())
