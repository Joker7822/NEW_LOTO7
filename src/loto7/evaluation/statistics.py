"""Paired time-series inference utilities for promotion decisions."""
from __future__ import annotations

import math
import random
import statistics
from typing import Dict, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_moving_block_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 2000,
    block_size: int = 0,
    seed: int = 20260731,
    confidence: float = 0.95,
) -> Dict[str, object]:
    if len(candidate) != len(baseline):
        raise ValueError("paired sequences must have equal length")
    if not candidate:
        raise ValueError("paired sequences must not be empty")
    differences = [float(left) - float(right) for left, right in zip(candidate, baseline)]
    size = len(differences)
    block = int(block_size) if block_size > 0 else max(2, round(size ** (1 / 3)))
    block = min(size, max(1, block))
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(100, int(samples))):
        sampled: list[float] = []
        while len(sampled) < size:
            start = rng.randrange(size)
            sampled.extend(
                differences[(start + offset) % size] for offset in range(block)
            )
        estimates.append(statistics.fmean(sampled[:size]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "kind": "paired_moving_block_bootstrap",
        "sample_count": len(estimates),
        "pair_count": size,
        "block_size": block,
        "confidence": confidence,
        "estimate": round(statistics.fmean(differences), 8),
        "ci_lower": round(percentile(estimates, alpha), 8),
        "ci_upper": round(percentile(estimates, 1.0 - alpha), 8),
        "probability_positive": round(
            sum(value > 0 for value in estimates) / len(estimates), 8
        ),
    }


def wilson_interval(
    successes: int,
    total: int,
    confidence_z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + confidence_z**2 / total
    centre = (proportion + confidence_z**2 / (2.0 * total)) / denominator
    margin = confidence_z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + confidence_z**2 / (4.0 * total**2)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
