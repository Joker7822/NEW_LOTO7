#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promote a Generation 5 candidate only after internal and fixed-null gates pass."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Mapping


def read_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default="outputs/generation5/generation5_candidate_model.json",
    )
    parser.add_argument("--baseline", default="loto7_best_model.json")
    parser.add_argument(
        "--summary", default="outputs/generation5/generation5_summary.json"
    )
    parser.add_argument(
        "--null-summary",
        default="outputs/generation5/null_strategy_league_summary.json",
    )
    parser.add_argument(
        "--decision", default="outputs/generation5/promotion_decision.json"
    )
    parser.add_argument(
        "--report", default="outputs/generation5/promotion_report.txt"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    candidate = read_json(args.candidate)
    baseline = read_json(args.baseline)
    summary = read_json(args.summary)
    null_summary = read_json(args.null_summary)

    adoption = (
        summary.get("adoption")
        if isinstance(summary.get("adoption"), Mapping)
        else {}
    )
    null_decision = (
        null_summary.get("decision")
        if isinstance(null_summary.get("decision"), Mapping)
        else {}
    )
    candidate_genome = (
        candidate.get("genome")
        if isinstance(candidate.get("genome"), Mapping)
        else {}
    )
    baseline_genome = (
        baseline.get("genome", baseline) if isinstance(baseline, Mapping) else {}
    )
    candidate_id = str(candidate_genome.get("id", ""))
    baseline_id = (
        str(baseline_genome.get("id", ""))
        if isinstance(baseline_genome, Mapping)
        else ""
    )
    candidate_sha = sha256(args.candidate)
    baseline_sha = sha256(args.baseline)

    checks = {
        "generation5_internal_gate": bool(adoption.get("passed")),
        "fixed_null_league_gate": bool(null_decision.get("passed")),
        "different_genome_id": bool(candidate_id and candidate_id != baseline_id),
        "different_model_sha256": candidate_sha != baseline_sha,
        "candidate_objective_version_present": bool(
            candidate.get("objective_version")
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    promoted = not failures
    applied = bool(promoted and args.apply)
    if applied:
        shutil.copyfile(args.candidate, args.baseline)

    payload: Dict[str, object] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "loto7_generation5_promotion_decision",
        "promoted": promoted,
        "apply_requested": bool(args.apply),
        "applied": applied,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "candidate_genome_id": candidate_id,
        "baseline_genome_id": baseline_id,
        "candidate_sha256": candidate_sha,
        "baseline_sha256_before": baseline_sha,
        "checks": checks,
        "failures": failures,
        "internal_gate": adoption,
        "null_gate": null_decision,
    }
    target = Path(args.decision)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "LOTO7 Generation 5 Promotion Decision",
        "=====================================",
        "",
        f"promoted: {promoted}",
        f"apply_requested: {args.apply}",
        f"applied: {applied}",
        f"candidate_genome_id: {candidate_id}",
        f"baseline_genome_id: {baseline_id}",
        "failures:",
        *([f"- {item}" for item in failures] or ["- none"]),
    ]
    Path(args.report).write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
