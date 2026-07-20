#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical and legacy-compatible output paths for NEW_LOTO7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

OUTPUT_LAYOUT_VERSION = 2


@dataclass(frozen=True)
class OutputBinding:
    key: str
    legacy: str
    canonical: str
    category: str
    resumable: bool = False


BINDINGS = (
    OutputBinding("latest_prediction", "outputs/evolution_best_prediction.csv", "outputs/production/latest_prediction.csv", "production"),
    OutputBinding("prediction_history", "outputs/evolution_prediction_history.csv", "outputs/production/prediction_history.csv", "production"),
    OutputBinding("prediction_history_result", "outputs/evolution_prediction_history_result.txt", "outputs/production/prediction_history_result.txt", "production"),
    OutputBinding("latest_prediction_report", "outputs/holdout/latest_prediction_report.txt", "outputs/production/latest_prediction_report.txt", "production"),
    OutputBinding("sealed_manifest", "outputs/generation4/latest_sealed_manifest.json", "outputs/evidence/generation4/latest_sealed_manifest.json", "evidence"),
    OutputBinding("sealed_index", "outputs/generation4/sealed_index.json", "outputs/evidence/generation4/sealed_index.json", "evidence"),
    OutputBinding("sealed_directory", "outputs/generation4/sealed", "outputs/evidence/generation4/sealed", "evidence"),
    OutputBinding("full_state", "outputs/model_self_evolution", "outputs/state/full", "state", True),
    OutputBinding("recent_state", "outputs/recent_era", "outputs/state/recent", "state", True),
    OutputBinding("super_recent_state", "outputs/super_recent", "outputs/state/super_recent", "state", True),
    OutputBinding("validation_evidence", "outputs/validation", "outputs/evidence/validation", "evidence", True),
    OutputBinding("holdout_diagnostics", "outputs/holdout", "outputs/diagnostics/holdout", "diagnostics"),
    OutputBinding("role_diagnostics", "outputs/role_ensemble", "outputs/diagnostics/role_ensemble", "diagnostics"),
    OutputBinding("generation4_diagnostics", "outputs/generation4", "outputs/diagnostics/generation4", "diagnostics"),
)

BY_KEY: Dict[str, OutputBinding] = {binding.key: binding for binding in BINDINGS}


def binding(key: str) -> OutputBinding:
    try:
        return BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown output binding: {key}") from exc


def canonical_path(key: str, root: str | Path = ".") -> Path:
    return Path(root) / binding(key).canonical


def legacy_path(key: str, root: str | Path = ".") -> Path:
    return Path(root) / binding(key).legacy


def resolve_existing(key: str, root: str | Path = ".") -> Path:
    """Prefer canonical data while retaining old resume paths during migration."""
    canonical = canonical_path(key, root)
    if canonical.exists():
        return canonical
    return legacy_path(key, root)


def resumable_bindings() -> Iterable[OutputBinding]:
    return (item for item in BINDINGS if item.resumable)
